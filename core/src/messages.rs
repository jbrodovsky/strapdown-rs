// gnss_degrader.rs
use chrono::{DateTime, Utc};
use nalgebra::Vector3;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::IMUData;
use crate::earth::meters_ned_to_dlat_dlon;
use crate::measurements::{
    GPSPositionAndVelocityMeasurement, MeasurementModel, RelativeAltitudeMeasurement,
};
use crate::sim::TestDataRecord;
/// Scheduler for controlling when GNSS measurements are emitted into the simulation.
///
/// This models denial- or jamming-like effects that reduce the *rate* of
/// available GNSS updates, independent of their content (which is handled by
/// [`GnssFaultModel`]).
///
/// By separating scheduling from corruption, you can experiment with outages,
/// degraded update rates, or duty-cycled availability while keeping the
/// measurement noise model orthogonal.
///
/// ## Usage
/// - `PassThrough` → GNSS data is delivered at its native logging rate.
/// - `FixedInterval` → Down-sample the GNSS stream to a constant interval,
///   simulating jamming that allows only low-rate fixes.
/// - `DutyCycle` → Alternate between ON and OFF windows of fixed length,
///   simulating periodic outages.
///
/// See also [`GnssDegradationConfig`] for how this is combined with a
/// [`GnssFaultModel`] and a random seed.
///
/// ## Examples
///
/// ```
/// use strapdown::messages::GnssScheduler;
///
/// // Keep all GNSS fixes (no scheduling)
/// let sched = GnssScheduler::PassThrough;
///
/// // Deliver a GNSS fix every 10 seconds, starting at t=0
/// let sched = GnssScheduler::FixedInterval { interval_s: 10.0, phase_s: 0.0 };
///
/// // Alternate 5 s ON, 15 s OFF, starting in ON state at t=0
/// let sched = GnssScheduler::DutyCycle { on_s: 5.0, off_s: 15.0, start_phase_s: 0.0 };
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GnssScheduler {
    /// Pass every GNSS fix through to the filter with no rate reduction.
    ///
    /// Useful as a baseline when you want to test only fault injection without
    /// simulating outages or reduced update rates.
    #[default]
    PassThrough,
    /// Emit GNSS measurements at a fixed interval, discarding those in between.
    ///
    /// This simulates reduced-rate operation under jamming or low-power conditions.
    ///
    /// * `interval_s` — Desired interval between emitted GNSS fixes, in seconds.
    /// * `phase_s` — Initial time offset before the first emission, in seconds.
    FixedInterval {
        /// Desired interval between GNSS fixes (seconds).
        interval_s: f64,
        /// Initial phase offset before the first emitted fix (seconds).
        phase_s: f64,
    },
    /// Alternate between ON and OFF windows to create duty-cycled outages.
    ///
    /// This simulates conditions like periodic GNSS denial or environments
    /// where reception is available only intermittently (e.g., urban canyon).
    ///
    /// * `on_s` — Duration of each ON window (seconds).
    /// * `off_s` — Duration of each OFF window (seconds).
    /// * `start_phase_s` — Initial time offset before the first toggle (seconds).
    DutyCycle {
        /// Duration of each ON window (seconds).
        on_s: f64,
        /// Duration of each OFF window (seconds).
        off_s: f64,
        /// Initial phase offset before the first ON/OFF toggle (seconds).
        start_phase_s: f64,
    },
}

/// Models how GNSS measurement *content* is corrupted before it reaches the filter.
///
/// This is complementary to [`GnssScheduler`], which decides *when* GNSS
/// updates are delivered. `GnssFaultModel` decides *what* corruption to apply
/// to each delivered measurement. Together, they allow you to simulate a wide
/// range of denial, jamming, or spoofing conditions.
///
/// Typical usage is to wrap a "truth-like" GNSS fix (from your dataset) with
/// one of these variants before passing it to the UKF update step.
///
/// ## Variants
///
/// - `None`: deliver the fix unchanged.
/// - `Degraded`: add AR(1)-correlated noise to position and velocity, and
///   inflate the advertised covariance. Simulates low-SNR or multi-path conditions.
/// - `SlowBias`: apply a slowly drifting offset in N/E position and velocity.
///   Simulates soft spoofing where the trajectory is nudged gradually away
///   from truth.
/// - `Hijack`: apply a hard constant offset in N/E position during a fixed time
///   window. Simulates hard spoofing where the solution is forced onto a
///   parallel displaced track.
/// - `Combo`: apply several fault models in sequence (output of one feeds into
///   the next), allowing composition of multiple effects.
///
/// ## Examples
///
/// ```
/// use strapdown::messages::GnssFaultModel;
///
/// // No corruption (baseline)
/// let fault = GnssFaultModel::None;
///
/// // Degraded accuracy: ~3 m wander, ~0.3 m/s vel wander, 5x inflated R
/// let fault = GnssFaultModel::Degraded {
///     rho_pos: 0.99,
///     sigma_pos_m: 3.0,
///     rho_vel: 0.95,
///     sigma_vel_mps: 0.3,
///     r_scale: 5.0,
/// };
///
/// // Slow bias drifting north at 2 cm/s
/// let fault = GnssFaultModel::SlowBias {
///     drift_n_mps: 0.02,
///     drift_e_mps: 0.0,
///     q_bias: 1e-6,
///     rotate_omega_rps: 0.0,
/// };
///
/// // Hijack: apply 50 m north offset between 120–180 s
/// let fault = GnssFaultModel::Hijack {
///     offset_n_m: 50.0,
///     offset_e_m: 0.0,
///     start_s: 120.0,
///     duration_s: 60.0,
/// };
///
/// // Combo: first drift slowly, then add hijack window
/// let fault = GnssFaultModel::Combo(vec![
///     GnssFaultModel::SlowBias { drift_n_mps: 0.02, drift_e_mps: 0.0, q_bias: 1e-6, rotate_omega_rps: 0.0 },
///     GnssFaultModel::Hijack { offset_n_m: 50.0, offset_e_m: 0.0, start_s: 120.0, duration_s: 60.0 },
/// ]);
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GnssFaultModel {
    /// No corruption; GNSS fixes are passed through unchanged.
    #[default]
    None,

    /// (2) Degraded accuracy: AR(1)-correlated noise on position and velocity,
    /// plus inflated advertised covariance. Models low-SNR or multi-path cases.
    Degraded {
        /// AR(1) correlation coefficient for position error (close to 1.0).
        rho_pos: f64,
        /// AR(1) innovation standard deviation for position error (meters).
        sigma_pos_m: f64,
        /// AR(1) correlation coefficient for velocity error.
        rho_vel: f64,
        /// AR(1) innovation standard deviation for velocity error (m/s).
        sigma_vel_mps: f64,
        /// Scale factor for inflating the advertised measurement noise covariance.
        r_scale: f64,
    },

    /// (5) Slow drifting bias (soft spoof), applied in N/E meters and velocity.
    ///
    /// Models gradual displacement of the navigation solution that appears
    /// plausible to the filter.
    SlowBias {
        /// Northward drift rate (m/s).
        drift_n_mps: f64,
        /// Eastward drift rate (m/s).
        drift_e_mps: f64,
        /// Random walk PSD (m²/s³) for adding a small stochastic component to the bias.
        q_bias: f64,
        /// Optional slow rotation of drift direction (rad/s).
        rotate_omega_rps: f64,
    },

    /// (6) Hard spoof window: apply a constant N/E offset for a fixed time window.
    ///
    /// Simulates abrupt hijacking of the trajectory.
    Hijack {
        /// North offset in meters.
        offset_n_m: f64,
        /// East offset in meters.
        offset_e_m: f64,
        /// Start time of spoofing window (s).
        start_s: f64,
        /// Duration of spoofing window (s).
        duration_s: f64,
    },

    /// Compose multiple effects by chaining models together.
    ///
    /// The output of one model is fed as the input to the next. This allows
    /// combining e.g. `SlowBias` with a `Hijack` to simulate multi-stage spoofing.
    Combo(Vec<GnssFaultModel>),
}

/// Configuration container for GNSS degradation in simulation.
///
/// This ties together a [`GnssScheduler`] (which controls *when* GNSS fixes
/// are delivered), a [`GnssFaultModel`] (which controls *what* corruption is
/// applied to each fix), and a random seed for reproducibility.
///
/// By keeping scheduling and fault injection separate but bundled here, you can
/// easily swap in different scenarios or repeat experiments deterministically.
///
/// ## Fields
///
/// - `scheduler`: Controls emission rate / outage pattern (e.g. pass-through,
///   fixed-interval, duty-cycled).
/// - `fault`: Corrupts measurement content (e.g. degraded AR(1) wander, slow
///   bias, hijack).
/// - `seed`: Seed for the internal random number generator, ensuring runs are
///   reproducible for debugging and A/B comparisons.
///
/// ## Example
///
/// ```
/// use strapdown::messages::{GnssDegradationConfig, GnssScheduler, GnssFaultModel};
///
/// // Deliver GNSS every 10 seconds, with AR(1)-degraded accuracy.
/// let cfg = GnssDegradationConfig {
///     scheduler: GnssScheduler::FixedInterval { interval_s: 10.0, phase_s: 0.0 },
///     fault: GnssFaultModel::Degraded {
///         rho_pos: 0.99,
///         sigma_pos_m: 3.0,
///         rho_vel: 0.95,
///         sigma_vel_mps: 0.3,
///         r_scale: 5.0,
///     },
///     seed: 42,
/// };
/// ```
/// Default seed value for reproducible simulations
fn default_seed() -> u64 {
    42
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GnssDegradationConfig {
    /// Scheduler that determines when GNSS measurements are emitted
    /// (e.g., pass-through, fixed interval, or duty-cycled).
    #[serde(default)]
    pub scheduler: GnssScheduler,

    /// Fault model that corrupts the contents of each emitted GNSS measurement
    /// (e.g., degraded wander, slow bias drift, hijack).
    #[serde(default)]
    pub fault: GnssFaultModel,

    /// Random number generator seed for deterministic tests and reproducibility.
    ///
    /// Use the same seed to repeat scenarios exactly; change it to get a new
    /// realization of stochastic processes such as AR(1) degradation.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for GnssDegradationConfig {
    fn default() -> Self {
        GnssDegradationConfig {
            scheduler: GnssScheduler::default(),
            fault: GnssFaultModel::default(),
            seed: default_seed(),
        }
    }
}

impl GnssDegradationConfig {
    /// Write the configuration to a JSON file (pretty-printed).
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self).map_err(io::Error::other)
    }

    /// Read the configuration from a JSON file.
    pub fn from_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        serde_json::from_reader(file).map_err(io::Error::other)
    }
    /// Write the configuration as YAML.
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;
        let s = serde_yaml::to_string(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from YAML.
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        serde_yaml::from_reader(file).map_err(io::Error::other)
    }
    /// Write the configuration as TOML.
    pub fn to_toml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;
        let s = toml::to_string(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }
    /// Read the configuration from TOML.
    pub fn from_toml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut s = String::new();
        let mut file = File::open(path)?;
        file.read_to_string(&mut s)?;
        toml::from_str(&s).map_err(io::Error::other)
    }
    /// Generic write: choose format by file extension (.json/.yaml/.yml/.toml)
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("json") => self.to_json(p),
            Some("yaml") | Some("yml") => self.to_yaml(p),
            Some("toml") => self.to_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension",
            )),
        }
    }
    /// Generic read: choose format by file extension (.json/.yaml/.yml/.toml)
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("json") => Self::from_json(p),
            Some("yaml") | Some("yml") => Self::from_yaml(p),
            Some("toml") => Self::from_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension",
            )),
        }
    }
}
/// A simulation event delivered to the filter in time order.
///
/// Events represent sensor updates or other observations that occur during
/// playback of recorded data. The event stream is built by combining raw
/// logged records with a [`GnssDegradationConfig`] (for GNSS scheduling and
/// fault injection), and then fed to the UKF loop.
///
/// Each variant bundles both the measurement itself and the elapsed simulation
/// time when it occurred. This allows the filter to advance its state correctly
/// and to process updates at realistic intervals.
///
/// ## Variants
///
/// - `Imu`: An inertial measurement unit (IMU) step, including the time delta
///   since the previous step. Drives the prediction step.
/// - `Gnss`: A GNSS position/velocity fix (possibly degraded or spoofed).
/// - `Altitude`: A relative altitude or barometric measurement, constraining
///   vertical drift.
/// - `GravityAnomaly`: A gravity anomaly measurement, derived from accelerometer
///   or dedicated gravimeter data and matched against a gravity anomaly map.
/// - `MagneticAnomaly`: A magnetic anomaly measurement, derived from
///   magnetometer data (magnitude or components) and matched against a magnetic
///   anomaly map.
///
/// ## Extensibility
///
/// You can add further variants (e.g., `Pressure`, `SonarDepth`, `StarTracker`)
/// in the same style if more sensors are to be fused.
///
/// ## Example
///
/// ```
/// use strapdown::messages::Event;
/// use strapdown::IMUData;
/// use strapdown::measurements::GPSPositionAndVelocityMeasurement;
/// use nalgebra::Vector3;
/// // An IMU event with 0.01 s timestep
/// let imu_event = Event::Imu {
///     dt_s: 0.01,
///     imu: IMUData { accel: Vector3::new(0.0, 0.1, -9.8),
///                    gyro: Vector3::new(0.001, 0.0, 0.0) },
///     elapsed_s: 1.23,
/// };
///
/// // A GNSS event
/// let gnss_meas = GPSPositionAndVelocityMeasurement {
///     latitude: 39.95,
///     longitude: -75.16,
///     altitude: 30.0,
///     northward_velocity: 0.1,
///     eastward_velocity: -0.2,
///     horizontal_noise_std: 5.0,
///     vertical_noise_std: 10.0,
///     velocity_noise_std: 0.2,
/// };
/// let gnss_event = Event::Measurement { meas: Box::new(gnss_meas), elapsed_s: 2.0 };
/// ```
pub enum Event {
    /// IMU prediction step.
    ///
    /// - `dt_s`: Time delta since the previous event (seconds).
    /// - `imu`: Inertial data record (accelerometer, gyroscope, etc.).
    /// - `elapsed_s`: Elapsed simulation time at this event (seconds).
    Imu {
        dt_s: f64,
        imu: IMUData,
        elapsed_s: f64,
    },
    /// Any measurement that implements the MeasurementModel trait.
    Measurement {
        meas: Box<dyn MeasurementModel>, // trait object
        elapsed_s: f64,
    },
}
pub struct EventStream {
    pub start_time: DateTime<Utc>,
    pub events: Vec<Event>,
}
// -------- internal state for AR(1) and bias integration --------
/// Internal state used to realize stochastic GNSS fault models.
///
/// `FaultState` holds the evolving error terms for [`GnssFaultModel`] variants
/// that require memory across timesteps, such as:
///
/// - **Degraded (AR(1))**: maintains correlated error states for position and
///   velocity, updated each epoch with an autoregressive process.
/// - **SlowBias**: integrates a slow, possibly rotating bias in N/E position,
///   with optional random walk.
/// - **Hijack**: does not need state, but still shares the RNG.
///
/// This struct is not exposed outside the degradation machinery. It is created
/// once at the beginning of a run (using a deterministic RNG seed) and then
/// updated as each measurement is processed.
///
/// ## Fields
///
/// - `e_n_m`, `e_e_m`, `e_u_m`: AR(1) position error states in the N/E/U
///   directions (meters).
/// - `ev_n_mps`, `ev_e_mps`, `ev_u_mps`: AR(1) velocity error states in the
///   N/E/U directions (m/s).
/// - `b_n_m`, `b_e_m`: integrated bias terms for slow-bias models (meters).
/// - `rng`: deterministic random number generator used for injecting noise
///   (seeded from [`FaultState::new`]).
///
/// ## Example
///
/// ```ignore
/// use strapdown::messages::FaultState;
///
/// // Create a new state with a fixed seed for reproducibility
/// let mut st = FaultState::new(42);
///
/// // At each timestep, the AR(1) and bias states are updated by the
/// // degradation logic (not shown here).
/// ```
#[derive(Clone, Debug)]
pub struct FaultState {
    /// AR(1) position error state (north, meters).
    e_n_m: f64,
    /// AR(1) position error state (east, meters).
    e_e_m: f64,
    /// AR(1) position error state (up, meters).
    e_u_m: f64,
    /// AR(1) velocity error state (north, m/s).
    ev_n_mps: f64,
    /// AR(1) velocity error state (east, m/s).
    ev_e_mps: f64,
    /// AR(1) velocity error state (up, m/s).
    // ev_u_mps: f64,
    /// Integrated slow bias (north, meters).
    b_n_m: f64,
    /// Integrated slow bias (east, meters).
    b_e_m: f64,
    /// Deterministic RNG for generating noise realizations.
    rng: rand::rngs::StdRng,
}
impl FaultState {
    /// Construct a new `FaultState` with all error terms initialized to zero.
    ///
    /// The random number generator is seeded from the provided `seed`, so
    /// repeated runs with the same seed yield identical noise realizations.
    pub fn new(seed: u64) -> Self {
        Self {
            e_n_m: 0.0,
            e_e_m: 0.0,
            e_u_m: 0.0,
            ev_n_mps: 0.0,
            ev_e_mps: 0.0,
            //ev_u_mps: 0.0,
            b_n_m: 0.0,
            b_e_m: 0.0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}
// -------- helpers --------
/// Advance an AR(1) (autoregressive) process by one timestep.
///
/// Updates the error state `x` according to
///
/// ```text
/// x_t = ρ · x_{t-1} + σ · w_t
/// ```
///
/// where:
/// - `rho` (`ρ`) is the correlation coefficient, typically close to 1.0
///   (e.g., 0.95–0.995 for GNSS error wander).
/// - `sigma` (`σ`) is the innovation standard deviation.
/// - `w_t` is white Gaussian noise, sampled here from `N(0, σ²)`.
///
/// This is used to model time-correlated measurement errors, such as degraded
/// GNSS position/velocity noise, in contrast to purely white (independent) noise.
///
/// ## Arguments
/// - `x`: Mutable reference to the AR(1) state to be updated.
/// - `rho`: Correlation coefficient.
/// - `sigma`: Innovation standard deviation.
/// - `rng`: Deterministic random number generator used for noise sampling.
///
/// ## Example
///
/// ```ignore
/// use rand::SeedableRng;
///
/// let mut x = 0.0;
/// let mut rng = SeedableRng::seed_from_u64(42);
/// for _ in 0..5 {
///     ar1_step(&mut x, 0.99, 3.0, &mut rng);
///     println!("New error state: {x}");
/// }
/// ```
fn ar1_step(x: &mut f64, rho: f64, sigma: f64, rng: &mut rand::rngs::StdRng) {
    let n = Normal::new(0.0, sigma.max(0.0)).unwrap();
    *x = rho * *x + n.sample(rng);
}
/// Apply a GNSS fault model to a truth-like GNSS fix, producing a corrupted measurement.
///
/// This function implements the *content* transformation for GNSS faults
/// (complementing the *rate/outage* control handled by the scheduler).
/// Given the current time `t`, time step `dt`, and an input GNSS
/// position/velocity plus advertised standard deviations, it returns a new
/// (possibly corrupted) measurement in the same units.
///
/// The stochastic components use and update the internal [`FaultState`]
/// (AR(1) states, slow-bias integrators, and RNG), so repeated calls with the
/// same seed are reproducible.
///
/// # Arguments
/// - `fault`: The [`GnssFaultModel`] variant describing which corruption to apply.
/// - `st`: Mutable internal state for AR(1) and slow-bias models (updated in place).
/// - `t`: Elapsed time of this measurement (seconds).
/// - `dt`: Time since the previous step (seconds).
/// - `lat_deg`, `lon_deg`: Input geodetic latitude/longitude **in degrees** (truth-like).
/// - `alt_m`: Altitude above the ellipsoid (meters).
/// - `vn_mps`, `ve_mps`: N/E components of velocity (m/s).
/// - `horiz_std_m`, `vert_std_m`, `vel_std_mps`: Advertised 1σ standard deviations
///   for horizontal position, vertical position, and velocity, respectively.
///   These may be scaled by some fault modes to reflect degraded confidence.
///
/// # Returns
/// A 7-tuple:
/// `(lat_deg, lon_deg, alt_m, vn_mps, ve_mps, horiz_std_m, vel_std_mps)`
/// representing the *corrupted* latitude, longitude (degrees), altitude (m),
/// N/E velocities (m/s), horizontal position std (m), and velocity std (m/s).
///
/// > **Note:** The current implementation passes `vert_std_m` through unchanged.
/// > If you also want to degrade vertical accuracy, extend the relevant branches
/// > to scale it and return it (and update the function signature/uses accordingly).
///
/// # Behavior by variant
/// - **`GnssFaultModel::None`**  
///   Returns inputs unchanged (baseline).
///
/// - **`GnssFaultModel::Degraded`**  
///   Adds AR(1)-correlated errors to position (N/E/U, meters) and velocity
///   (N/E, m/s). The position error is mapped to Δlat/Δlon via an ellipsoidal
///   small-offset conversion. The advertised horizontal/velocity standard
///   deviations are multiplied by `r_scale`.
///
/// - **`GnssFaultModel::SlowBias`**  
///   Integrates a slowly drifting N/E bias (m) with optional slow rotation of
///   drift direction and small random-walk perturbation. A small consistent
///   velocity bias (N/E, m/s) is also applied to keep the corruption plausible.
///
/// - **`GnssFaultModel::Hijack`**  
///   Applies a constant N/E offset (meters) within a time window
///   `[start_s, start_s + duration_s]`, mapping it to Δlat/Δlon. Outside the
///   window, measurements pass through unchanged.
///
/// - **`GnssFaultModel::Combo`**  
///   Intended to compose multiple effects by feeding the output of one model as
///   the input to the next. (Wire up the call loop to `apply_fault` for each
///   sub-model if composition is desired.)
///
/// # Units & conventions
/// - Inputs/outputs for latitude and longitude are **degrees**; internal small-angle
///   calculations convert to **radians** and back.
/// - Altitude is in meters; velocities are in m/s (N/E components).
/// - Standard deviations are 1σ values (not variances).
///
/// # Numerical notes
/// - Conversion from N/E meter offsets to Δlat/Δlon uses WGS-84 principal radii
///   with a small `cos(lat)` clamp near the poles to avoid singularities.
/// - AR(1) updates use a Normal(0, σ) innovation each step; see [`ar1_step`].
///
/// # Examples
/// ```ignore
/// // Degraded GNSS with AR(1) wander and inflated R
/// let mut st = FaultState::new(42);
/// let (lat, lon, alt, vn, ve, hstd, vstd) = apply_fault(
///     &GnssFaultModel::Degraded {
///         rho_pos: 0.99, sigma_pos_m: 3.0,
///         rho_vel: 0.95, sigma_vel_mps: 0.3,
///         r_scale: 5.0,
///     },
///     &mut st,
///     t, dt,
///     lat_deg, lon_deg, alt_m,
///     vn_mps, ve_mps,
///     horiz_std_m, vert_std_m, vel_std_mps,
/// );
/// ```
pub fn apply_fault(
    fault: &GnssFaultModel,
    st: &mut FaultState,
    t: f64,
    dt: f64,
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
    vn_mps: f64,
    ve_mps: f64,
    horiz_std_m: f64,
    vert_std_m: f64,
    vel_std_mps: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) /* lat, lon, alt, vn, ve, horiz_std, vel_std */ {
    match fault {
        GnssFaultModel::None => (
            lat_deg,
            lon_deg,
            alt_m,
            vn_mps,
            ve_mps,
            horiz_std_m,
            vel_std_mps,
        ),

        GnssFaultModel::Degraded {
            rho_pos,
            sigma_pos_m,
            rho_vel,
            sigma_vel_mps,
            r_scale,
        } => {
            ar1_step(&mut st.e_n_m, *rho_pos, *sigma_pos_m, &mut st.rng);
            ar1_step(&mut st.e_e_m, *rho_pos, *sigma_pos_m, &mut st.rng);
            ar1_step(&mut st.e_u_m, *rho_pos, *sigma_pos_m, &mut st.rng);
            ar1_step(&mut st.ev_n_mps, *rho_vel, *sigma_vel_mps, &mut st.rng);
            ar1_step(&mut st.ev_e_mps, *rho_vel, *sigma_vel_mps, &mut st.rng);

            let (dlat, dlon) =
                meters_ned_to_dlat_dlon(lat_deg.to_radians(), alt_m, st.e_n_m, st.e_e_m);
            let lat_c = lat_deg + dlat.to_degrees();
            let lon_c = lon_deg + dlon.to_degrees();
            let alt_c = alt_m + st.e_u_m;

            let vn_c = vn_mps + st.ev_n_mps;
            let ve_c = ve_mps + st.ev_e_mps;

            (
                lat_c,
                lon_c,
                alt_c,
                vn_c,
                ve_c,
                horiz_std_m * r_scale,
                vel_std_mps * r_scale,
            )
        }

        GnssFaultModel::SlowBias {
            drift_n_mps,
            drift_e_mps,
            q_bias,
            rotate_omega_rps,
        } => {
            // integrate bias with optional slow rotation
            let (mut bn_dot, mut be_dot) = (*drift_n_mps, *drift_e_mps);
            if *rotate_omega_rps != 0.0 {
                let th = rotate_omega_rps * t;
                let c = th.cos();
                let s = th.sin();
                let (n0, e0) = (*drift_n_mps, *drift_e_mps);
                bn_dot = c * n0 - s * e0;
                be_dot = s * n0 + c * e0;
            }
            st.b_n_m += bn_dot * dt;
            st.b_e_m += be_dot * dt;
            if *q_bias > 0.0 {
                ar1_step(&mut st.b_n_m, 1.0, (q_bias * dt).sqrt(), &mut st.rng);
                ar1_step(&mut st.b_e_m, 1.0, (q_bias * dt).sqrt(), &mut st.rng);
            }
            let (dlat, dlon) =
                meters_ned_to_dlat_dlon(lat_deg.to_radians(), alt_m, st.b_n_m, st.b_e_m);
            let lat_c = lat_deg + dlat.to_degrees();
            let lon_c = lon_deg + dlon.to_degrees();
            let vn_c = vn_mps + bn_dot;
            let ve_c = ve_mps + be_dot;

            (lat_c, lon_c, alt_m, vn_c, ve_c, horiz_std_m, vel_std_mps)
        }

        GnssFaultModel::Hijack {
            offset_n_m,
            offset_e_m,
            start_s,
            duration_s,
        } => {
            if t >= *start_s && t <= (start_s + duration_s) {
                let (dlat, dlon) =
                    meters_ned_to_dlat_dlon(lat_deg.to_radians(), alt_m, *offset_n_m, *offset_e_m);
                let lat_c = lat_deg + dlat.to_degrees();
                let lon_c = lon_deg + dlon.to_degrees();
                (
                    lat_c,
                    lon_c,
                    alt_m,
                    vn_mps,
                    ve_mps,
                    horiz_std_m,
                    vel_std_mps,
                )
            } else {
                (
                    lat_deg,
                    lon_deg,
                    alt_m,
                    vn_mps,
                    ve_mps,
                    horiz_std_m,
                    vel_std_mps,
                )
            }
        }

        GnssFaultModel::Combo(models) => {
            let mut out = (
                lat_deg,
                lon_deg,
                alt_m,
                vn_mps,
                ve_mps,
                horiz_std_m,
                vel_std_mps,
            );
            for m in models {
                out = apply_fault(
                    m, st, t, dt, out.0, out.1, out.2, out.3, out.4, out.5, vert_std_m, out.6,
                );
            }
            out
        }
    }
}
// --------------------------- public API ---------------------------
/// Build a time-ordered event stream from recorded data and a GNSS degradation
/// configuration.
///
/// This function converts raw `records` into a vector of [`Event`]s suitable
/// for an event-driven filter loop. It:
///
/// 1. Normalizes the record timestamps to **elapsed seconds** from the first sample.
/// 2. Emits an [`Event::Imu`] at each step with `dt_s = t[i] - t[i-1]`.
/// 3. Uses the provided [`GnssDegradationConfig`] to decide *when* to emit GNSS
///    (via the [`GnssScheduler`]) and *how* to corrupt that GNSS fix
///    (via the [`GnssFaultModel`], applied by [`apply_fault`]).
/// 4. Appends each emitted GNSS fix as an [`Event::Gnss`] with the same `elapsed_s`
///    as the IMU step.
///
/// The resulting event stream cleanly separates simulation policy (scheduling
/// and corruption) from the filter loop, enabling reproducible scenario testing.
///
/// # Arguments
/// - `records`: Source telemetry, ordered by time, providing IMU and GNSS-like
///   fields (lat/lon/alt/speed/bearing/accuracies).
/// - `cfg`: GNSS degradation configuration combining a scheduler (*when*) and a
///   fault model (*what*), plus a seed for deterministic noise.
///
/// # Returns
/// A `EventStream` containing an interleaved sequence of IMU and (optionally
/// down-sampled/corrupted) GNSS events, ordered by `elapsed_s`.
///
/// # Scheduling semantics
/// - [`GnssScheduler::PassThrough`]: emit a GNSS event at every record step.
/// - [`GnssScheduler::FixedInterval`]: emit when `elapsed_s >= next_emit_time`,
///   then advance `next_emit_time += interval_s` (with initial `phase_s`).
/// - [`GnssScheduler::DutyCycle`]: toggle ON/OFF windows of lengths `on_s`/`off_s`
///   starting at `start_phase_s`; emit only at the boundary into the ON window.
///
/// # Corruption semantics
/// The truth-like GNSS (lat/lon/alt + velocity derived from `speed`/`bearing`)
/// is transformed by [`apply_fault`] according to `cfg.fault`:
/// - `None`: unchanged.
/// - `Degraded`: AR(1) wander on position/velocity; advertised horizontal and
///   velocity sigmas scaled by `r_scale`.
/// - `SlowBias`: integrates a drifting N/E bias (with optional rotation/random walk).
/// - `Hijack`: applies a constant N/E offset within a time window.
/// - `Combo`: intended for sequential composition (hook up as needed).
///
/// > **Note:** The current implementation passes `vertical_noise_std` through
/// > unchanged. If you also want to degrade vertical accuracy, extend the
/// > `apply_fault` branch and adjust the GNSS measurement construction.
///
/// # Units & conventions
/// - Elapsed time is in **seconds** from the first record.
/// - `lat_deg`, `lon_deg` are **degrees**; small-offset conversions use radians internally.
/// - Altitude (m), velocities (m/s), standard deviations are **1σ** (not variances).
///
/// # Preconditions & caveats
/// - `records.len() >= 2` and timestamps are monotonically increasing.
/// - If your `horizontal_accuracy`/`vertical_accuracy` fields are *variances*,
///   adjust the `.sqrt()` usage accordingly.
/// - The event vector capacity is sized roughly to `2 * records.len()` (IMU + GNSS).
///
/// # Example
/// ```ignore
/// let cfg = GnssDegradationConfig {
///     scheduler: GnssScheduler::FixedInterval { interval_s: 10.0, phase_s: 0.0 },
///     fault: GnssFaultModel::Degraded {
///         rho_pos: 0.99, sigma_pos_m: 3.0,
///         rho_vel: 0.95, sigma_vel_mps: 0.3,
///         r_scale: 5.0,
///     },
///     seed: 42,
/// };
/// let events = build_event_stream(&records, &cfg);
/// // feed into your event-driven filter loop
/// ```
pub fn build_event_stream(records: &[TestDataRecord], cfg: &GnssDegradationConfig) -> EventStream {
    let start_time = records[0].time;
    let records_with_elapsed: Vec<(f64, &TestDataRecord)> = records
        .iter()
        .map(|r| ((r.time - start_time).num_milliseconds() as f64 / 1000.0, r))
        .collect();
    let mut events = Vec::with_capacity(records_with_elapsed.len() * 2);
    let mut st = FaultState::new(cfg.seed);

    // Scheduler state
    let mut next_emit_time = match cfg.scheduler {
        GnssScheduler::PassThrough => 0.0,
        GnssScheduler::FixedInterval { phase_s, .. } => phase_s,
        GnssScheduler::DutyCycle { start_phase_s, .. } => start_phase_s,
    };
    let mut duty_on = true;
    // Through preprocessing we assert that the first record must have a NED position
    // but it may or may not have IMU or other such measurements.
    let reference_altitude = records[0].altitude;
    for w in records_with_elapsed.windows(2) {
        let (t0, _) = (&w[0].0, &w[0].1);
        let (t1, r1) = (&w[1].0, &w[1].1);
        let dt = t1 - t0;

        // Build IMU event at t1 only if accel and gyro components are present
        let imu_components = [
            r1.acc_x, r1.acc_y, r1.acc_z, r1.gyro_x, r1.gyro_y, r1.gyro_z,
        ];
        let imu_present = imu_components.iter().all(|v| !v.is_nan());
        if imu_present {
            let imu = IMUData {
                accel: Vector3::new(r1.acc_x, r1.acc_y, r1.acc_z),
                gyro: Vector3::new(r1.gyro_x, r1.gyro_y, r1.gyro_z),
                // add other fields as your UKF expects
            };
            events.push(Event::Imu {
                dt_s: dt,
                imu,
                elapsed_s: *t1,
            });
        }

        // Decide if GNSS should be emitted at t1
        let should_emit = match cfg.scheduler {
            GnssScheduler::PassThrough => true,
            GnssScheduler::FixedInterval { interval_s, .. } => {
                if *t1 + 1e-9 >= next_emit_time {
                    next_emit_time += interval_s;
                    true
                } else {
                    false
                }
            }
            GnssScheduler::DutyCycle { on_s, off_s, .. } => {
                let window = if duty_on { on_s } else { off_s };
                if *t1 + 1e-9 >= next_emit_time {
                    duty_on = !duty_on;
                    next_emit_time += window;
                    duty_on // only emit when toggling into ON
                } else {
                    false
                }
            }
        };

        if should_emit {
            // Only create GNSS event when the core GNSS values are present
            let gnss_required = [r1.latitude, r1.longitude, r1.altitude, r1.speed, r1.bearing];
            let gnss_present = gnss_required.iter().all(|v| !v.is_nan());
            if gnss_present {
                // Truth-like GNSS from r1
                let lat = r1.latitude;
                let lon = r1.longitude;
                let alt = r1.altitude;
                let bearing_rad = r1.bearing.to_radians();
                let vn = r1.speed * bearing_rad.cos();
                let ve = r1.speed * bearing_rad.sin();

                // Use your provided accuracies (adjust if these are variances vs std).
                // If an accuracy is missing (NaN), substitute a conservative default
                // to avoid propagating NaN into the measurement noise.
                let horiz_std = if r1.horizontal_accuracy.is_nan() {
                    15.0
                } else {
                    r1.horizontal_accuracy.max(1e-3)
                };
                let vert_std = if r1.vertical_accuracy.is_nan() {
                    1000.0
                } else {
                    r1.vertical_accuracy.max(1e-3)
                };
                let vel_std = if r1.speed_accuracy.is_nan() {
                    100.0
                } else {
                    r1.speed_accuracy.max(0.1)
                };

                let (lat_c, lon_c, alt_c, vn_c, ve_c, horiz_c, vel_c) = apply_fault(
                    &cfg.fault, &mut st, *t1, dt, lat, lon, alt, vn, ve, horiz_std, vert_std,
                    vel_std,
                );

                let meas = GPSPositionAndVelocityMeasurement {
                    latitude: lat_c,
                    longitude: lon_c,
                    altitude: alt_c,
                    northward_velocity: vn_c,
                    eastward_velocity: ve_c,
                    horizontal_noise_std: horiz_c,
                    vertical_noise_std: vert_std, // pass-through here; you can also degrade it if desired
                    velocity_noise_std: vel_c,
                };
                events.push(Event::Measurement {
                    meas: Box::new(meas),
                    elapsed_s: *t1,
                });
            }
        }
        if !r1.relative_altitude.is_nan() {
            let baro: RelativeAltitudeMeasurement = RelativeAltitudeMeasurement {
                relative_altitude: r1.relative_altitude,
                reference_altitude,
            };
            events.push(Event::Measurement {
                meas: Box::new(baro),
                elapsed_s: *t1,
            });
        }
    }
    EventStream { start_time, events }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::{TimeZone, Utc};

    fn create_test_records(count: usize, interval_secs: f64) -> Vec<TestDataRecord> {
        let base_time = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let mut records = Vec::with_capacity(count);

        for i in 0..count {
            let time_ms = (i as f64 * interval_secs * 1000.0) as i64;
            let time = base_time + chrono::Duration::milliseconds(time_ms);
            let record = TestDataRecord {
                time,
                latitude: 37.0,
                longitude: -122.0,
                altitude: 100.0,
                bearing: 45.0,
                speed: 5.0,
                acc_x: 0.0,
                acc_y: 0.0,
                acc_z: 9.81,
                gyro_x: 0.0,
                gyro_y: 0.0,
                gyro_z: 0.01,
                qx: 0.0,
                qy: 0.0,
                qz: 0.0,
                qw: 1.0,
                roll: 0.0,
                pitch: 0.0,
                yaw: 0.0,
                mag_x: 0.0,
                mag_y: 0.0,
                mag_z: 0.0,
                relative_altitude: 0.0,
                pressure: 1013.25,
                grav_x: 0.0,
                grav_y: 0.0,
                grav_z: 9.81,
                horizontal_accuracy: 2.0,
                vertical_accuracy: 4.0,
                speed_accuracy: 0.5,
                bearing_accuracy: 1.0,
            };
            records.push(record);
        }
        records
    }
    #[test]
    fn test_passthrough_scheduler() {
        let records = create_test_records(10, 0.1); // 10 records, 0.1s apart
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            seed: 42,
        };

        let events = build_event_stream(&records, &config);

        // We expect IMU events for each record except the first,
        // and GNSS events for each record except the first
        assert_eq!(events.events.len(), 27); // 9 IMU + 9 GNSS + 9 Baro

        // Count IMU and GNSS events
        let imu_count = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Imu { .. }))
            .count();
        let gnss_count = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .into_iter()
            .filter(|e| {
                if let Event::Measurement { meas, .. } = e {
                    meas.as_any().is::<GPSPositionAndVelocityMeasurement>()
                } else {
                    false
                }
            })
            .count();
        assert_eq!(imu_count, 9);
        assert_eq!(gnss_count, 9);
    }
    #[test]
    fn test_fixed_interval_scheduler() {
        let records = create_test_records(20, 0.1); // 20 records, 0.1s apart
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::FixedInterval {
                interval_s: 0.5,
                phase_s: 0.0,
            },
            fault: GnssFaultModel::None,
            seed: 42,
        };

        let events = build_event_stream(&records, &config);

        // We expect IMU events for each record except the first,
        // and GNSS events every 0.5s (so at records 5, 10, 15...)
        let imu_count = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Imu { .. }))
            .count();
        let measurements = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .count();
        assert_eq!(imu_count, 19);
        assert_eq!(measurements, 23); // At 0.5s, 1.0s, 1.5s, and 1.9s + 19 baro measurements
    }
    #[test]
    fn test_duty_cycle_scheduler() {
        let records = create_test_records(60, 1.0); // 60 records, 1s apart, 60 seconds total
        assert!(
            records.len() == 60,
            "{}",
            format!("Expected 60 records, found: {}", records.len())
        );

        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::DutyCycle {
                on_s: 1.0,
                off_s: 1.0,
                start_phase_s: 0.0,
            },
            fault: GnssFaultModel::None,
            seed: 42,
        };
        //
        let events = build_event_stream(&records, &config);
        //assert!(events.len() == 60, "{}", format!("Expected 60 events, found: {}", events.len()));
        // We should only have GNSS events when turning ON
        // (so at 0.0s, 2.0s, 4.0s, ...)
        let measurements: Vec<&Event> = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .collect();
        // We initialize off of the first event and only get GNSS updates every two seconds starting from 1.0
        // (60 - 1) // 2 = 29 + 59 baro
        assert!(measurements.len() >= 2);
        assert!(
            measurements.len() == 88,
            "{}",
            format!("Expected 88 GNSS events, found: {}", measurements.len())
        );
    }
    #[test]
    fn test_degraded_fault_model() {
        let records = create_test_records(10, 0.1);
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::Degraded {
                rho_pos: 0.99,
                sigma_pos_m: 3.0,
                rho_vel: 0.95,
                sigma_vel_mps: 0.3,
                r_scale: 5.0,
            },
            seed: 500,
        };

        let events = build_event_stream(&records, &config);

        // Find GNSS events
        let measurements: Vec<&Event> = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .collect();
        let gnss_events: Vec<&Event> = measurements
            .into_iter()
            .filter(|e| {
                if let Event::Measurement { meas, .. } = e {
                    meas.as_any().is::<GPSPositionAndVelocityMeasurement>()
                } else {
                    false
                }
            })
            .collect();

        // Check R-scaling is applied
        for event in &gnss_events {
            let meas = if let Event::Measurement { meas, .. } = event {
                meas.as_any()
                    .downcast_ref::<GPSPositionAndVelocityMeasurement>()
            } else {
                None
            };
            let Some(meas) = meas else {
                continue;
            };
            // Original horizontal accuracy is 2.0
            assert_approx_eq!(meas.horizontal_noise_std, 9.0, 2.0);
            // Original velocity accuracy is 0.5
            assert_approx_eq!(meas.velocity_noise_std, 2.5, 0.1);
        }

        // Check that positions are perturbed
        let original_lat = 37.0;
        let original_lon = -122.0;

        let mut all_same = true;
        let mut prev_lat = None;
        let mut prev_lon = None;

        for event in &gnss_events {
            if let Event::Measurement { meas, .. } = event {
                let meas = meas
                    .as_any()
                    .downcast_ref::<GPSPositionAndVelocityMeasurement>()
                    .unwrap();
                // Positions should be perturbed from original
                assert_approx_eq!(meas.latitude, original_lat, 1e-3);
                assert_approx_eq!(meas.longitude, original_lon, 1e-3);

                // Check if positions vary between measurements
                if let Some(prev_lat) = prev_lat {
                    if (meas.latitude - prev_lat as f64).abs() > 1e-10 {
                        all_same = false;
                    }
                }
                prev_lat = Some(meas.latitude);

                if let Some(prev_lon) = prev_lon {
                    if (meas.longitude - prev_lon as f64).abs() > 1e-10 {
                        all_same = false;
                    }
                }
                prev_lon = Some(meas.longitude);
            }
        }
        // Positions should vary between measurements due to AR(1) process
        assert!(!all_same);
    }

    #[test]
    fn slow_bias_adds_velocity_bias() {
        let mut st = FaultState::new(123);
        let fault = GnssFaultModel::SlowBias {
            drift_n_mps: 0.02,
            drift_e_mps: -0.01,
            q_bias: 0.0,
            rotate_omega_rps: 0.0,
        };

        // zero “truth” velocity
        let (_lat, _lon, _alt, vn_c, ve_c, _hstd, _vstd) = apply_fault(
            &fault, &mut st, /*t*/ 10.0, /*dt*/ 1.0, /*lat_deg*/ 40.0,
            /*lon_deg*/ -75.0, /*alt_m*/ 0.0, /*vn_mps*/ 0.0, /*ve_mps*/ 0.0,
            /*horiz_std_m*/ 3.0, /*vert_std_m*/ 5.0, /*vel_std_mps*/ 0.2,
        );

        assert_approx_eq!(vn_c, 0.02, 0.001);
        assert_approx_eq!(ve_c, -0.01, 0.001);
    }

    #[test]
    fn test_hijack_fault_model() {
        let records = create_test_records(30, 0.1); // 30 records, 0.1s apart, total 3.0 seconds
        let offset_n = 50.0;
        let offset_e = 30.0;

        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::Hijack {
                offset_n_m: offset_n,
                offset_e_m: offset_e,
                start_s: 1.0,
                duration_s: 1.0, // Hijack from 1.0s to 2.0s
            },
            seed: 42,
        };

        let events = build_event_stream(&records, &config);

        // Find GNSS events and group by time
        let mut gnss_by_time: Vec<(f64, &GPSPositionAndVelocityMeasurement)> = Vec::new();
        for event in &events.events {
            if let Event::Measurement { meas, elapsed_s } = event {
                if let Some(gps) = meas
                    .as_any()
                    .downcast_ref::<GPSPositionAndVelocityMeasurement>()
                {
                    gnss_by_time.push((*elapsed_s, gps));
                }
            }
        }
        gnss_by_time.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Original position
        let original_lat = 37.0;
        let original_lon = -122.0;

        // Check measurements before, during, and after hijack
        for (time, meas) in gnss_by_time {
            if time < 1.0 - 1e-6 || time > 2.0 + 1e-6 {
                // Before hijack or after hijack: positions should be near original
                assert!((meas.latitude - original_lat).abs() < 1e-6);
                assert!((meas.longitude - original_lon).abs() < 1e-6);
            } else {
                // During hijack: positions should be offset
                assert!((meas.latitude - original_lat).abs() > 1e-6);
                assert!((meas.longitude - original_lon).abs() > 1e-6);
            }
        }
    }

    #[test]
    fn test_combo_fault_model() {
        // Test that the combo fault model functionality exists
        // Note: Due to commented code in apply_fault for Combo, this is a minimal test
        let records = create_test_records(10, 0.1);

        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::Combo(vec![GnssFaultModel::None, GnssFaultModel::None]),
            seed: 42,
        };

        // This should at least not crash
        let events = build_event_stream(&records, &config);
        assert!(events.events.len() > 0);
    }

    #[test]
    fn test_ar1_step() {
        // Test the AR(1) process step function
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // For rho=0, value should be replaced by Gaussian noise
        let mut x = 10.0;
        ar1_step(&mut x, 0.0, 1.0, &mut rng);
        assert!(x != 10.0); // Should have changed
        assert!(x.abs() < 5.0); // Should be reasonably close to 0 (5-sigma event very unlikely)

        // For rho=1, sigma=0, value should remain unchanged
        let mut x = 10.0;
        ar1_step(&mut x, 1.0, 0.0, &mut rng);
        assert_eq!(x, 10.0);

        // For negative sigma, should be treated as 0
        let mut x = 10.0;
        ar1_step(&mut x, 0.5, -1.0, &mut rng);
        assert_eq!(x, 5.0); // 0.5 * 10.0 + 0.0
    }

    #[test]
    fn test_slow_bias_fault_with_rotation() {
        // Test slow bias fault with rotation (rotate_omega_rps != 0.0)
        let records = create_test_records(10, 0.1);
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::SlowBias {
                drift_n_mps: 1.0,
                drift_e_mps: 0.5,
                rotate_omega_rps: 0.1,
                q_bias: 0.0,
            },
            seed: 42,
        };

        let events = build_event_stream(&records, &config);
        // Should have events
        assert!(events.events.len() > 0);

        // Check that some GNSS measurements exist
        let gnss_count = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .count();
        assert!(gnss_count > 0);
    }

    #[test]
    fn test_slow_bias_fault_with_q_bias() {
        // Test slow bias fault with q_bias > 0.0
        let records = create_test_records(10, 0.1);
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::SlowBias {
                drift_n_mps: 0.0,
                drift_e_mps: 0.0,
                rotate_omega_rps: 0.0,
                q_bias: 1.0,
            },
            seed: 42,
        };

        let events = build_event_stream(&records, &config);
        // Should have events
        assert!(events.events.len() > 0);

        // Check that some GNSS measurements exist
        let gnss_count = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .count();
        assert!(gnss_count > 0);
    }

    #[test]
    fn test_nan_accuracy_handling() {
        // Test handling of NaN values in accuracy fields
        let base_time = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let mut records = Vec::new();

        // First record (reference)
        records.push(TestDataRecord {
            time: base_time,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            bearing: 45.0,
            speed: 5.0,
            acc_x: 0.0,
            acc_y: 0.0,
            acc_z: 9.81,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.01,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            qw: 1.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            relative_altitude: 0.0,
            pressure: 1013.25,
            grav_x: 0.0,
            grav_y: 0.0,
            grav_z: 9.81,
            horizontal_accuracy: 2.0,
            vertical_accuracy: 4.0,
            speed_accuracy: 0.5,
            bearing_accuracy: 1.0,
        });

        // Second record with NaN accuracies
        records.push(TestDataRecord {
            time: base_time + chrono::Duration::milliseconds(100),
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            bearing: 45.0,
            speed: 5.0,
            acc_x: 0.0,
            acc_y: 0.0,
            acc_z: 9.81,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.01,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            qw: 1.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            relative_altitude: 0.0,
            pressure: 1013.25,
            grav_x: 0.0,
            grav_y: 0.0,
            grav_z: 9.81,
            horizontal_accuracy: f64::NAN,
            vertical_accuracy: f64::NAN,
            speed_accuracy: f64::NAN,
            bearing_accuracy: 1.0,
        });

        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            seed: 42,
        };

        let events = build_event_stream(&records, &config);

        // Should have events even with NaN accuracies
        assert!(events.events.len() > 0);

        // Check that GNSS measurements were created
        let gnss_count = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .count();
        assert!(gnss_count > 0);
    }

    #[test]
    fn test_duty_cycle_scheduler_toggles() {
        // Test DutyCycle scheduler to ensure it toggles states
        let records = create_test_records(20, 0.1);
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::DutyCycle {
                on_s: 0.5,
                off_s: 0.5,
                start_phase_s: 0.0,
            },
            fault: GnssFaultModel::None,
            seed: 42,
        };

        let events = build_event_stream(&records, &config);
        // Should have events
        assert!(events.events.len() > 0);
    }
}
#[cfg(test)]
mod serialization_tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_cfg() -> GnssDegradationConfig {
        GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::Degraded {
                rho_pos: 0.99,
                sigma_pos_m: 3.0,
                rho_vel: 0.95,
                sigma_vel_mps: 0.3,
                r_scale: 5.0,
            },
            seed: 42,
        }
    }

    #[test]
    fn json_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("json");
        cfg.to_json(&path).unwrap();
        let loaded = GnssDegradationConfig::from_json(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }

    #[test]
    fn yaml_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("yaml");
        cfg.to_yaml(&path).unwrap();
        let loaded = GnssDegradationConfig::from_yaml(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }

    #[test]
    fn toml_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("toml");
        cfg.to_toml(&path).unwrap();
        let loaded = GnssDegradationConfig::from_toml(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }

    #[test]
    fn generic_dispatch_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("json");
        cfg.to_file(&path).unwrap();
        let loaded = GnssDegradationConfig::from_file(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }

    #[test]
    fn default_config_roundtrip() {
        // Test that default config can be serialized and read back
        let cfg = GnssDegradationConfig::default();

        // Verify default values
        assert_eq!(cfg.seed, 42);
        assert!(matches!(cfg.scheduler, GnssScheduler::PassThrough));
        assert!(matches!(cfg.fault, GnssFaultModel::None));

        // Test JSON roundtrip
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("json");
        cfg.to_file(&path).unwrap();
        let loaded = GnssDegradationConfig::from_file(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
        assert!(matches!(loaded.scheduler, GnssScheduler::PassThrough));
        assert!(matches!(loaded.fault, GnssFaultModel::None));

        // Test YAML roundtrip
        let f2 = NamedTempFile::new().unwrap();
        let path2 = f2.path().with_extension("yaml");
        cfg.to_file(&path2).unwrap();
        let loaded2 = GnssDegradationConfig::from_file(&path2).unwrap();
        assert_eq!(cfg.seed, loaded2.seed);

        // Test TOML roundtrip
        let f3 = NamedTempFile::new().unwrap();
        let path3 = f3.path().with_extension("toml");
        cfg.to_file(&path3).unwrap();
        let loaded3 = GnssDegradationConfig::from_file(&path3).unwrap();
        assert_eq!(cfg.seed, loaded3.seed);
    }

    #[test]
    fn unsupported_extension_error() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("txt");

        // Test to_file with unsupported extension
        let result = cfg.to_file(&path);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);

        // Test from_file with unsupported extension
        let result = GnssDegradationConfig::from_file(&path);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn yml_extension_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("yml");
        cfg.to_file(&path).unwrap();
        let loaded = GnssDegradationConfig::from_file(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }
}
