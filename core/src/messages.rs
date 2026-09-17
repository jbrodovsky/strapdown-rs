// gnss_degrader.rs
use chrono::{DateTime, Datelike, Utc};
use nalgebra::Vector3;
use rand::SeedableRng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::earth::meters_ned_to_dlat_dlon;
use crate::measurements::{
    BAROMETRIC_ALTITUDE_NOISE_M, GPSPositionAndVelocityMeasurement, MAG_YAW_NOISE,
    MagnetometerYawMeasurement, MeasurementModel, RelativeAltitudeMeasurement,
};
use crate::sim::TestDataRecord;
use crate::{IMUData, StrapdownError};
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
/// See also [`AidingConfig`] for how this is combined with a
/// [`GnssFaultModel`] and a random seed.
///
/// ## Examples
///
/// ```
/// use strapdown::messages::MeasurementScheduler;
///
/// // Keep all GNSS fixes (no scheduling)
/// let sched = MeasurementScheduler::PassThrough;
///
/// // Deliver a GNSS fix every 10 seconds, starting at t=0
/// let sched = MeasurementScheduler::FixedInterval { interval_s: 10.0, phase_s: 0.0 };
///
/// // Alternate 5 s ON, 15 s OFF, starting in ON state at t=0
/// let sched = MeasurementScheduler::DutyCycle { on_s: 5.0, off_s: 15.0, start_phase_s: 0.0 };
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MeasurementScheduler {
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
/// This is complementary to [`MeasurementScheduler`], which decides *when* GNSS
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
/// - `Degraded`: add AR(1)-correlated noise to position and velocity, and inflate the
///   advertised 1-sigma accuracies (so `R` moves by the square of `r_scale`). Simulates
///   low-SNR or multi-path conditions.
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
/// // Degraded accuracy: ~3 m wander, ~0.3 m/s vel wander, sigmas x5 (so R x25)
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
    /// plus inflated advertised accuracies. Models low-SNR or multi-path cases.
    Degraded {
        /// AR(1) correlation coefficient for position error (close to 1.0).
        rho_pos: f64,
        /// AR(1) innovation standard deviation for position error (meters).
        sigma_pos_m: f64,
        /// AR(1) correlation coefficient for velocity error.
        rho_vel: f64,
        /// AR(1) innovation standard deviation for velocity error (m/s).
        sigma_vel_mps: f64,
        /// Multiplies the advertised 1-sigma accuracies -- *not* the covariance.
        ///
        /// `apply_fault` returns `horizontal_accuracy * r_scale` and
        /// `speed_accuracy * r_scale`, and the measurement models square those to build
        /// `R` ([`crate::measurements::GPSPositionAndVelocityMeasurement`]), so the noise
        /// covariance is inflated by `r_scale` **squared**: the common value 5.0 gives a
        /// 25x `R`, not a 5x one.
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
        /// Random-walk rate (m²/s, equivalently (m/√s)²) at which the metre-valued bias
        /// accumulates variance: each step perturbs the bias by a zero-mean draw with
        /// standard deviation `sqrt(q_bias * dt)`, so its variance grows by `q_bias * dt`.
        /// Set to zero to disable the stochastic component and leave a purely
        /// deterministic drift.
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
    Combo(Vec<Self>),
}

/// Default seed value for reproducible simulations
const fn default_seed() -> u64 {
    42
}

/// Default emission schedule for the barometer and the magnetometer: one measurement per second.
///
/// Not [`MeasurementScheduler::PassThrough`], which is what these two channels effectively had before
/// they were scheduled at all. A barometer and a magnetometer are aiding sources like any
/// other, and emitting one per record ties their update rate to the *log's* rate rather than
/// to the sensor's. On a 1 Hz recording that happens to be right; on the 50 Hz synthetic
/// trajectories it delivered fifty pressure readings and fifty headings a second, each entering
/// the filter as an independent fix. A heading re-derived from the same field vector fifty
/// times is one measurement counted fifty times, and the UKF diverges on it (#375).
///
/// One per second matches the rate the reference recording logs these channels at, so it is
/// also the schedule under which every baseline number in `core/tests/perf_baseline.json` that
/// was measured on real data was measured.
const fn default_aiding_scheduler() -> MeasurementScheduler {
    MeasurementScheduler::FixedInterval {
        interval_s: 1.0,
        phase_s: 0.0,
    }
}

/// Configuration container for GNSS degradation in simulation.
///
/// This ties together a [`MeasurementScheduler`] (which controls *when* GNSS fixes
/// are delivered), a [`GnssFaultModel`] (which controls *what* corruption is
/// applied to each fix), and a random seed for reproducibility.
///
/// By keeping scheduling and fault injection separate but bundled here, you can
/// easily swap in different scenarios or repeat experiments deterministically.
///
/// ## Fields
///
/// - `scheduler`: Controls GNSS emission rate / outage pattern (e.g. pass-through,
///   fixed-interval, duty-cycled).
/// - `fault`: Corrupts measurement content (e.g. degraded AR(1) wander, slow
///   bias, hijack).
/// - `baro_scheduler`, `magnetometer_scheduler`: the same thing for the other two aiding
///   channels, each with its own independent state, both defaulting to 1 Hz rather than to
///   pass-through. Only GNSS has a fault model; the other two are scheduled but not corrupted.
/// - `seed`: Seed for the internal random number generator, ensuring runs are
///   reproducible for debugging and A/B comparisons.
///
/// The name is now narrower than the contents -- it schedules three sensors and degrades one.
/// Renaming it, and [`MeasurementScheduler`] with it, is on the 1.0 API-freeze list rather than done
/// here, so that a mechanical 112-site rename does not ride along with a behaviour change.
///
/// ## Example
///
/// ```
/// use strapdown::messages::{AidingConfig, MeasurementScheduler, GnssFaultModel};
///
/// // Deliver GNSS every 10 seconds, with AR(1)-degraded accuracy.
/// let cfg = AidingConfig {
///     scheduler: MeasurementScheduler::FixedInterval { interval_s: 10.0, phase_s: 0.0 },
///     fault: GnssFaultModel::Degraded {
///         rho_pos: 0.99,
///         sigma_pos_m: 3.0,
///         rho_vel: 0.95,
///         sigma_vel_mps: 0.3,
///         r_scale: 5.0,
///     },
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AidingConfig {
    /// Scheduler that determines when GNSS measurements are emitted
    /// (e.g., pass-through, fixed interval, or duty-cycled).
    #[serde(default)]
    pub scheduler: MeasurementScheduler,

    /// Fault model that corrupts the contents of each emitted GNSS measurement
    /// (e.g., degraded wander, slow bias drift, hijack).
    #[serde(default)]
    pub fault: GnssFaultModel,

    /// Scheduler that determines when barometric altitude measurements are emitted.
    ///
    /// Defaults to one per second (see [`default_aiding_scheduler`]), *not* to
    /// [`MeasurementScheduler::PassThrough`]. [`MeasurementScheduler::DutyCycle`] gives a barometer outage
    /// the same way it gives a GNSS one.
    #[serde(default = "default_aiding_scheduler")]
    pub baro_scheduler: MeasurementScheduler,

    /// Scheduler that determines when magnetometer heading measurements are emitted.
    ///
    /// Defaults to one per second (see [`default_aiding_scheduler`]), *not* to
    /// [`MeasurementScheduler::PassThrough`]. The heading a magnetometer yields is derived from a
    /// field vector, so re-reading it faster than the field changes adds no information while
    /// adding weight.
    #[serde(default = "default_aiding_scheduler")]
    pub magnetometer_scheduler: MeasurementScheduler,

    /// One-sigma barometric altitude noise, metres, applied to every
    /// [`RelativeAltitudeMeasurement`] this module builds.
    ///
    /// Defaults to [`BAROMETRIC_ALTITUDE_NOISE_M`]. A **standard deviation**: the value it
    /// replaces lived in a trait impl as `diag([5.0])` and was a variance, so this default is
    /// its square root and $R$ is unchanged (#375).
    ///
    /// The Sensor Logger format carries no pressure-accuracy column, so this cannot come from
    /// the record the way `horizontal_accuracy` feeds the GNSS models; a scenario that wants a
    /// good barometer or a bad one sets it here.
    #[serde(default = "default_baro_noise_std_m")]
    pub baro_noise_std_m: f64,

    /// Which filter state holds the barometric bias, if the run's filter estimates one.
    ///
    /// `None` -- the default -- is the 15-state case and leaves the barometer modelled as
    /// unbiased, exactly as before #372.
    ///
    /// An index rather than a flag, because a state vector carries no labels: "the last state"
    /// is a gravity map bias on one run and a barometric bias on another. Whoever builds the
    /// filter knows its layout and declares it here; a value the filter's state cannot reach is
    /// rejected by `RelativeAltitudeMeasurement` rather than silently ignored.
    #[serde(default)]
    pub baro_bias_index: Option<usize>,

    /// Random number generator seed for deterministic tests and reproducibility.
    ///
    /// Use the same seed to repeat scenarios exactly; change it to get a new
    /// realization of stochastic processes such as AR(1) degradation.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

/// Serde default for [`AidingConfig::baro_noise_std_m`].
const fn default_baro_noise_std_m() -> f64 {
    BAROMETRIC_ALTITUDE_NOISE_M
}

impl Default for AidingConfig {
    fn default() -> Self {
        Self {
            scheduler: MeasurementScheduler::default(),
            fault: GnssFaultModel::default(),
            baro_scheduler: default_aiding_scheduler(),
            magnetometer_scheduler: default_aiding_scheduler(),
            baro_noise_std_m: default_baro_noise_std_m(),
            baro_bias_index: None,
            seed: default_seed(),
        }
    }
}

impl AidingConfig {
    /// Write the configuration to a JSON file (pretty-printed).
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as JSON.
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self).map_err(io::Error::other)
    }

    /// Read the configuration from a JSON file.
    /// # Errors
    /// If the file cannot be read, or its contents are not valid JSON.
    pub fn from_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        serde_json::from_reader(file).map_err(io::Error::other)
    }
    /// Write the configuration as YAML.
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as YAML.
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;
        let s = serde_yaml::to_string(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from YAML.
    /// # Errors
    /// If the file cannot be read, or its contents are not valid YAML.
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        serde_yaml::from_reader(file).map_err(io::Error::other)
    }
    /// Write the configuration as TOML.
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as TOML.
    pub fn to_toml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;
        let s = toml::to_string(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }
    /// Read the configuration from TOML.
    /// # Errors
    /// If the file cannot be read, or its contents are not valid TOML.
    pub fn from_toml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut s = String::new();
        let mut file = File::open(path)?;
        file.read_to_string(&mut s)?;
        toml::from_str(&s).map_err(io::Error::other)
    }
    /// Generic write: choose format by file extension (.json/.yaml/.yml/.toml)
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as the inferred format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase);
        match ext.as_deref() {
            Some("json") => self.to_json(p),
            Some("yaml" | "yml") => self.to_yaml(p),
            Some("toml") => self.to_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension",
            )),
        }
    }
    /// Generic read: choose format by file extension (.json/.yaml/.yml/.toml)
    /// # Errors
    /// If the file cannot be read, or its contents are not valid the inferred format.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase);
        match ext.as_deref() {
            Some("json") => Self::from_json(p),
            Some("yaml" | "yml") => Self::from_yaml(p),
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
/// logged records with a [`AidingConfig`] (for GNSS scheduling and
/// fault injection), and then fed to the UKF loop.
///
/// Each variant bundles both the measurement itself and the elapsed simulation
/// time when it occurred. This allows the filter to advance its state correctly
/// and to process updates at realistic intervals.
///
/// ## Variants
///
/// - `Imu`: An inertial measurement unit (IMU) step, carrying the interval between
///   this source record and the one before it -- not the time since the previously
///   emitted event. Drives the prediction step.
/// - `Measurement`: Any boxed [`MeasurementModel`], driving the update step. In a
///   stream from [`build_event_stream`] this is a GNSS position/velocity fix
///   (possibly degraded or spoofed), a relative altitude measurement, or a
///   magnetometer yaw measurement; `strapdown-geonav` additionally puts gravity and
///   magnetic anomaly measurements into the same variant.
///
/// ## Extensibility
///
/// A new sensor is added by implementing [`MeasurementModel`] for it, not by adding a
/// variant here -- `Measurement` already carries any implementor.
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
    /// - `dt_s`: Integration interval for this step (seconds): the gap between this
    ///   record's timestamp and the previous record's, *not* the time since the previously
    ///   emitted event.
    /// - `imu`: Inertial data record (accelerometer, gyroscope, etc.).
    /// - `elapsed_s`: Elapsed simulation time at this event (seconds).
    Imu {
        /// Integration interval for this step (seconds): the gap between this record's
        /// timestamp and the previous record's.
        dt_s: f64,
        /// Body-frame specific force (m/s^2) and angular rate (rad/s) for this step,
        /// taken from the record unmodified -- gravity is still present in `accel` and is
        /// removed by the strapdown mechanization during propagation.
        imu: IMUData,
        /// Elapsed simulation time at this event (seconds since
        /// [`EventStream::start_time`]).
        elapsed_s: f64,
    },
    /// Any measurement that implements the `MeasurementModel` trait.
    Measurement {
        /// The measurement to run the filter's update step against, held as a trait
        /// object so one stream can carry GNSS, barometric, magnetometer and
        /// geophysical updates without the filter loop knowing their concrete types.
        meas: Box<dyn MeasurementModel>,
        /// Elapsed simulation time at this event (seconds since
        /// [`EventStream::start_time`]).
        elapsed_s: f64,
    },
}

impl std::fmt::Debug for Event {
    /// `Event::Measurement` carries a `Box<dyn MeasurementModel>`, and that trait
    /// deliberately has no `Debug` supertrait -- requiring one would force it on
    /// every downstream measurement model. The payload is therefore elided and
    /// only the discriminant and timing are shown.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Imu {
                dt_s,
                imu,
                elapsed_s,
            } => f
                .debug_struct("Event::Imu")
                .field("dt_s", dt_s)
                .field("imu", imu)
                .field("elapsed_s", elapsed_s)
                .finish(),
            Self::Measurement { elapsed_s, .. } => f
                .debug_struct("Event::Measurement")
                .field("meas", &"<dyn MeasurementModel>")
                .field("elapsed_s", elapsed_s)
                .finish(),
        }
    }
}
/// A time-ordered sequence of [`Event`]s together with the absolute time its clock runs from.
///
/// Produced by [`build_event_stream`] from a slice of [`TestDataRecord`], and consumed by the
/// event-driven filter loops in `strapdown-sim`. Every event carries an `elapsed_s` measured
/// from `start_time`, so a consumer recovers the absolute timestamp of an event as
/// `start_time + elapsed_s` -- which is how each navigation solution gets stamped.
///
/// Events are ordered by non-decreasing `elapsed_s`, and a single source record may contribute
/// several events at the same instant: an [`Event::Imu`] step followed by whichever
/// measurements that epoch carries.
#[derive(Debug)]
pub struct EventStream {
    /// UTC timestamp of the first source record; the origin every event's `elapsed_s` is
    /// measured from.
    pub start_time: DateTime<Utc>,
    /// The events themselves, ordered by elapsed time.
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
/// - **`SlowBias`**: integrates a slow, possibly rotating bias in N/E position,
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
/// ```
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
fn ar1_step(x: &mut f64, rho: f64, sigma: f64, rng: &mut rand::rngs::StdRng) {
    // `max(0.0)` clamps negatives and NaN alike, so the argument is always valid.
    let n = crate::normal_with_std(sigma.max(0.0));
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
/// - **`GnssFaultModel::None`**\
///   Returns inputs unchanged (baseline).
///
/// - **`GnssFaultModel::Degraded`**\
///   Adds AR(1)-correlated errors to position (N/E/U, meters) and velocity
///   (N/E, m/s). The position error is mapped to Δlat/Δlon via an ellipsoidal
///   small-offset conversion. The advertised horizontal/velocity standard
///   deviations are multiplied by `r_scale`.
///
/// - **`GnssFaultModel::SlowBias`**\
///   Integrates a slowly drifting N/E bias (m) with optional slow rotation of
///   drift direction and small random-walk perturbation. A small consistent
///   velocity bias (N/E, m/s) is also applied to keep the corruption plausible.
///
/// - **`GnssFaultModel::Hijack`**\
///   Applies a constant N/E offset (meters) within a time window
///   `[start_s, start_s + duration_s]`, mapping it to Δlat/Δlon. Outside the
///   window, measurements pass through unchanged.
///
/// - **`GnssFaultModel::Combo`**\
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
/// - AR(1) updates use a Normal(0, σ) innovation each step.
///
/// # Examples
/// ```
/// // Degraded GNSS with AR(1) wander and inflated R
/// use strapdown::messages::{GnssFaultModel, FaultState, apply_fault};
/// let mut st = FaultState::new(42);
/// let t = 100.0; // current time (s)
/// let dt = 1.0;  // time since last GNSS fix (s
/// let lat_deg = 39.95;
/// let lon_deg = -75.16;
/// let alt_m = 30.0;
/// let vn_mps = 0.1;
/// let ve_mps = -0.2;
/// let horiz_std_m = 5.0;
/// let vert_std_m = 10.0;
/// let vel_std_mps = 0.2;
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
///     horiz_std_m, vert_std_m,
/// );
/// ```
#[allow(clippy::too_many_arguments)]
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
                    m, st, t, dt, out.0, out.1, out.2, out.3, out.4, out.5, out.6,
                );
            }
            out
        }
    }
}
// --------------------------- public API ---------------------------
/// Tolerance for comparing a sample's elapsed time against a scheduler boundary.
///
/// Elapsed times are reconstructed from millisecond timestamps, so a sample that should land
/// exactly on a boundary can arrive a few ULP short of it. Without the slack a 1 Hz stream
/// against a whole-second window drops or gains a fix depending on rounding.
const DUTY_CYCLE_EPSILON_S: f64 = 1e-9;

/// Slack on a [`MeasurementScheduler::FixedInterval`] comparison, for the same reason
/// [`DUTY_CYCLE_EPSILON_S`] exists: an elapsed time built from integer milliseconds is not
/// exactly the multiple of `interval_s` it is meant to be, and a bare `>=` would drop a fix
/// on a record that is a rounding error early.
const SCHEDULE_EPSILON_S: f64 = 1e-9;

/// The emission clock a [`MeasurementScheduler`] starts from, before any record is seen.
///
/// Only [`MeasurementScheduler::FixedInterval`] carries state between records; the other two variants
/// decide from `elapsed_s` alone and their initial value is never read.
const fn initial_emit_time(scheduler: &MeasurementScheduler) -> f64 {
    match *scheduler {
        MeasurementScheduler::FixedInterval { phase_s, .. } => phase_s,
        MeasurementScheduler::PassThrough | MeasurementScheduler::DutyCycle { .. } => 0.0,
    }
}

/// Whether `scheduler` emits at `elapsed_s`, advancing `next_emit_time` if it does.
///
/// Each aided channel owns its own `next_emit_time`: a barometer on a 1 s schedule and a GNSS
/// receiver on a 10 s one must not share an emission clock.
///
/// A `FixedInterval` advances by whole intervals from `phase_s`, so its tick times stay on
/// exact multiples and cannot creep as a run gets long. It is a *rate limit*, not a resampler:
/// it emits at the first record at or after each tick, so a log sampled slower than the
/// interval emits on every record and one sampled faster emits on roughly every `interval_s`
/// of them.
///
/// The clock advances **past** `elapsed_s`, not by a single interval. Advancing once let it
/// fall behind across a gap in the log and then burst: on a 1 s schedule with records at 0.5,
/// 2.5 and 2.6 s, the 2.5 s record emitted and left the tick at 2.0 s, so the 2.6 s record
/// emitted too -- two fixes 0.1 s apart from something advertised as a 1 Hz rate limit. The
/// baseline never saw it, because `test_data.csv` is spaced at exactly 1.000 s and the
/// synthetic trajectories at exactly 0.02 s; a Sensor Logger export with a dropped sample is
/// not.
fn should_emit(scheduler: &MeasurementScheduler, elapsed_s: f64, next_emit_time: &mut f64) -> bool {
    match *scheduler {
        MeasurementScheduler::PassThrough => true,
        MeasurementScheduler::FixedInterval { interval_s, .. } => {
            if elapsed_s + SCHEDULE_EPSILON_S >= *next_emit_time {
                // Step to the first tick strictly after this record, so a gap in the log
                // cannot leave the clock behind and let the next record through early. A
                // non-positive or non-finite interval would never advance and would emit on
                // every record; that degrades to `PassThrough`, which is the same choice
                // `duty_cycle_is_on` makes for a degenerate cycle and for the same reason --
                // silently withholding an aiding channel is far harder to notice than
                // delivering it too often.
                if interval_s > 0.0 && interval_s.is_finite() {
                    let behind = elapsed_s + SCHEDULE_EPSILON_S - *next_emit_time;
                    let whole_intervals = (behind / interval_s).floor() + 1.0;
                    *next_emit_time += whole_intervals * interval_s;
                }
                true
            } else {
                false
            }
        }
        MeasurementScheduler::DutyCycle {
            on_s,
            off_s,
            start_phase_s,
        } => duty_cycle_is_on(elapsed_s, on_s, off_s, start_phase_s),
    }
}

/// Whether a [`MeasurementScheduler::DutyCycle`] is inside an ON window at `elapsed_s`.
///
/// The timeline is `start_phase_s` of initial ON, then `off_s` OFF and `on_s` ON repeating,
/// which is what "initial phase offset before the first ON/OFF toggle" describes: the first
/// toggle takes the scheduler out of its initial ON state.
///
/// Computed from `elapsed_s` directly rather than by stepping a toggle once per sample. The
/// stepping version emitted a fix only on the sample where the state flipped *into* ON and
/// returned `false` for every other sample in the window, so `on_s: 1800.0, off_s: 600.0`
/// delivered two fixes across an 89-minute recording instead of roughly four thousand
/// (#312). Deriving the state from the clock also means a window shorter than the sample
/// interval cannot leave the scheduler a boundary behind for the rest of the run.
fn duty_cycle_is_on(elapsed_s: f64, on_s: f64, off_s: f64, start_phase_s: f64) -> bool {
    if elapsed_s + DUTY_CYCLE_EPSILON_S < start_phase_s {
        return true;
    }
    let cycle_s = on_s + off_s;
    if cycle_s <= 0.0 || !cycle_s.is_finite() {
        // A non-positive or non-finite cycle cannot describe an outage. Deliver every fix
        // rather than withholding all of them: silently suppressing GNSS is the failure mode
        // #312 was, and it is much harder to notice than an outage that does not happen.
        return true;
    }
    let into_cycle = (elapsed_s - start_phase_s).rem_euclid(cycle_s);
    into_cycle + DUTY_CYCLE_EPSILON_S >= off_s
}

/// Build a time-ordered event stream from recorded data and a GNSS degradation
/// configuration.
///
/// This function converts raw `records` into a vector of [`Event`]s suitable
/// for an event-driven filter loop. It:
///
/// 1. Normalizes the record timestamps to **elapsed seconds** from the first sample.
/// 2. Emits an [`Event::Imu`] at each step with `dt_s = t[i] - t[i-1]`, provided the
///    record carries all six accelerometer and gyroscope components. If any of them is
///    `NaN`, only the [`Event::Imu`] is skipped: the relative-altitude and magnetometer
///    events for that same record are still emitted, and the GNSS event is too if it
///    would otherwise have been (it is independently gated by the scheduler and by its
///    own `NaN` check over the fix columns). The next IMU event's `dt_s` still spans only
///    one record interval rather than absorbing the skipped one.
/// 3. Uses the provided [`AidingConfig`] to decide *when* to emit GNSS
///    (via the [`MeasurementScheduler`]) and *how* to corrupt that GNSS fix
///    (via the [`GnssFaultModel`], applied by [`apply_fault`]).
/// 4. Appends each emitted GNSS fix as an [`Event::Measurement`] carrying a
///    [`GPSPositionAndVelocityMeasurement`], with the same `elapsed_s` as the IMU step.
/// 5. Appends a [`RelativeAltitudeMeasurement`] and a [`MagnetometerYawMeasurement`]
///    whenever the record carries them *and* their own scheduler says so. They are
///    scheduled independently of GNSS and of each other -- so baro and magnetometer
///    updates continue through a GNSS outage -- but they are never faulted: the fault
///    model governs GNSS only. Until #375 they were not scheduled either, which tied
///    their update rate to the log's sample rate: 1 Hz on a Sensor Logger recording,
///    50 Hz on a synthetic trajectory, where a heading re-derived from the same field
///    vector fifty times entered the filter as fifty independent fixes.
///
/// The resulting event stream cleanly separates simulation policy (scheduling
/// and corruption) from the filter loop, enabling reproducible scenario testing.
///
/// # Arguments
/// - `records`: Source telemetry, ordered by time, providing IMU and GNSS-like
///   fields (lat/lon/alt/speed/bearing/accuracies).
/// - `cfg`: GNSS degradation configuration combining a scheduler (*when*) and a
///   fault model (*what*), plus a seed for deterministic noise and a scheduler each
///   for the barometer and the magnetometer.
/// - `is_enu`: the local-level frame the filter consuming this stream works in -- `true` for
///   ENU, `false` for NED. Only the [`MagnetometerYawMeasurement`] reads it, and it must
///   match the state being updated: the heading a magnetometer implies is a different number
///   in the two conventions, not merely a different sign, so a mismatch drives yaw to a
///   reflection of the truth rather than weakening the aid (#305). It is a parameter rather
///   than a field on `cfg` deliberately: `strapdown-sim` already carries the frame at the top
///   level of its own configuration, and a second copy inside `AidingConfig` would
///   be a second source of truth for one physical fact -- which is how the reflection went
///   unnoticed in the first place. This mirrors [`crate::sim::dead_reckoning`], which took
///   the same argument for the same reason in #296.
///
/// # Returns
/// `Ok(EventStream)` -- an interleaved sequence of IMU and (optionally
/// down-sampled/corrupted) GNSS events, ordered by `elapsed_s` -- or the error described
/// under `# Errors` below.
///
/// # Scheduling semantics
/// - [`MeasurementScheduler::PassThrough`]: emit a GNSS event at every record step.
/// - [`MeasurementScheduler::FixedInterval`]: emit when `elapsed_s >= next_emit_time`,
///   then advance `next_emit_time += interval_s` (with initial `phase_s`).
/// - [`MeasurementScheduler::DutyCycle`]: emit at every record step that falls inside an ON
///   window. The timeline is `start_phase_s` of initial ON, then `off_s` OFF and `on_s` ON
///   repeating. See `duty_cycle_is_on`.
///
/// The same three rules drive `cfg.baro_scheduler` and `cfg.magnetometer_scheduler`, each
/// with its own emission clock. A channel's clock advances on schedule whether or not the
/// record carries that channel's data, so a sensor that starts logging mid-recording joins
/// its schedule rather than firing a burst.
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
/// - `records` is non-empty and its timestamps are monotonically increasing. A single
///   record is legal and produces an empty event list, since events are built from
///   adjacent pairs; an empty slice is an error, see below.
/// - The accuracy columns (`horizontal_accuracy`, `vertical_accuracy`, `speed_accuracy`)
///   are read as 1σ standard deviations, not variances; no square root is taken. A `NaN`
///   falls back to a conservative default -- 15.0 m horizontal, 1000.0 m vertical,
///   100.0 m/s velocity -- and a finite value is floored before use, at 1e-3 m for both
///   position accuracies and at 0.1 m/s for speed, so a logged 0.0 cannot produce a
///   zero-variance `R`.
/// - The event vector capacity is sized roughly to `2 * records.len()` (IMU + GNSS).
///
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] if `records` is empty. The first record supplies
/// both the stream's `start_time` and the reference altitude for relative-altitude
/// measurements, and neither has a defensible default: an `EventStream` has no representable
/// "no epoch". A slice of length one is *accepted* and yields an empty event list, so the
/// boundary is emptiness, not "fewer than two".
///
/// # Example
/// ```
/// use strapdown::messages::{build_event_stream, AidingConfig, MeasurementScheduler, GnssFaultModel};
/// use strapdown::sim::TestDataRecord;
///
/// # fn main() -> Result<(), strapdown::StrapdownError> {
/// let records = vec![TestDataRecord::default(); 10]; // load or generate your test data
/// let cfg = AidingConfig {
///     scheduler: MeasurementScheduler::FixedInterval { interval_s: 10.0, phase_s: 0.0 },
///     fault: GnssFaultModel::Degraded {
///         rho_pos: 0.99, sigma_pos_m: 3.0,
///         rho_vel: 0.95, sigma_vel_mps: 0.3,
///         r_scale: 5.0,
///     },
///     ..Default::default()
/// };
/// let events = build_event_stream(&records, &cfg, false)?; // false = NED
/// // feed into your event-driven filter loop
/// # Ok(())
/// # }
/// ```
pub fn build_event_stream(
    records: &[TestDataRecord],
    cfg: &AidingConfig,
    is_enu: bool,
) -> Result<EventStream, StrapdownError> {
    // The first record is load-bearing twice over -- it fixes the epoch the elapsed clock is
    // measured from and the datum the relative-altitude measurements are referenced to -- so
    // an empty slice cannot produce a meaningful stream and is rejected up front rather than
    // indexed into.
    let first = records
        .first()
        .ok_or_else(|| StrapdownError::InvalidConfiguration {
            field: "event stream records",
            reason: "cannot build an event stream from zero records: the first record supplies \
                     the stream's start time and the relative-altitude reference"
                .to_owned(),
        })?;
    let start_time = first.time;
    let records_with_elapsed: Vec<(f64, &TestDataRecord)> = records
        .iter()
        .map(|r| ((r.time - start_time).num_milliseconds() as f64 / 1000.0, r))
        .collect();
    let mut events = Vec::with_capacity(records_with_elapsed.len() * 2);
    let mut st = FaultState::new(cfg.seed);

    // Scheduler state, one clock per aided channel. Only `FixedInterval` needs any:
    // `PassThrough` emits unconditionally and `DutyCycle` derives its window from the elapsed
    // clock, see `duty_cycle_is_on`. The three must not share a clock -- that would couple a
    // barometer's rate to the GNSS receiver's.
    let mut next_gnss_emit_time = initial_emit_time(&cfg.scheduler);
    let mut next_baro_emit_time = initial_emit_time(&cfg.baro_scheduler);
    let mut next_magnetometer_emit_time = initial_emit_time(&cfg.magnetometer_scheduler);
    // Through preprocessing we assert that the first record must have a NED position
    // but it may or may not have IMU or other such measurements.
    let reference_altitude = first.altitude;
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
        let emit_gnss = should_emit(&cfg.scheduler, *t1, &mut next_gnss_emit_time);

        if emit_gnss {
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

                // The record's accuracy columns are 1-sigma standard deviations, floored
                // so a logged zero cannot produce a singular R. See the caveats on
                // `build_event_stream`.
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
                    &cfg.fault, &mut st, *t1, dt, lat, lon, alt, vn, ve, horiz_std, vel_std,
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
        // The barometer and the magnetometer are scheduled on the same footing as GNSS. Until
        // #375 they were emitted once per record window, outside the scheduler entirely, which
        // tied their rate to the log's: 1 Hz on the reference recording and 50 Hz on the
        // synthetic trajectories.
        if should_emit(&cfg.baro_scheduler, *t1, &mut next_baro_emit_time)
            && !r1.relative_altitude.is_nan()
        {
            let baro: RelativeAltitudeMeasurement = RelativeAltitudeMeasurement {
                relative_altitude: r1.relative_altitude,
                reference_altitude,
                noise_std: cfg.baro_noise_std_m,
                bias_index: cfg.baro_bias_index,
            };
            events.push(Event::Measurement {
                meas: Box::new(baro),
                elapsed_s: *t1,
            });
        }
        if should_emit(
            &cfg.magnetometer_scheduler,
            *t1,
            &mut next_magnetometer_emit_time,
        ) && [r1.mag_x, r1.mag_y, r1.mag_z].iter().all(|v| !v.is_nan())
        {
            let mag_meas = MagnetometerYawMeasurement {
                mag_x: r1.mag_x,
                mag_y: r1.mag_y,
                mag_z: r1.mag_z,
                noise_std: MAG_YAW_NOISE, // set a default noise std; adjust as needed
                apply_declination: true,
                year: r1.time.year(),
                // `ordinal()`, not `day()`: the WMM wants the day of the *year*, and `day()`
                // is the day of the month, so every record before this fix claimed to be in
                // the first 31 days of January. The cost is small -- declination moves 0.004
                // deg over that span at this dataset's position, well inside the model's own
                // uncertainty -- but a date field that is wrong by construction is not
                // something to leave for the next reader to rediscover (#305).
                day_of_year: r1.time.ordinal() as u16,
                is_enu,
            };
            events.push(Event::Measurement {
                meas: Box::new(mag_meas),
                elapsed_s: *t1,
            });
        }
    }
    Ok(EventStream { start_time, events })
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

    /// How many events in `stream` carry a measurement of type `M`.
    ///
    /// The three aided channels are told apart by their concrete type rather than by
    /// dimension: GNSS is the only one with more than one row, but the barometer and the
    /// magnetometer are both scalar, and once they are scheduled independently a test has to
    /// be able to say which of the two it is looking at.
    fn count_of<M: MeasurementModel + 'static>(stream: &EventStream) -> usize {
        times_of::<M>(stream).len()
    }

    /// The elapsed times at which `stream` carries a measurement of type `M`.
    fn times_of<M: MeasurementModel + 'static>(stream: &EventStream) -> Vec<f64> {
        stream
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Measurement { meas, elapsed_s } if meas.as_any().is::<M>() => {
                    Some(*elapsed_s)
                }
                _ => None,
            })
            .collect()
    }

    /// The first measurement of type `M` in `stream`, for reading a field off it.
    fn first_of<M: MeasurementModel + 'static>(stream: &EventStream) -> Option<&M> {
        stream.events.iter().find_map(|event| match event {
            Event::Measurement { meas, .. } => meas.as_any().downcast_ref::<M>(),
            Event::Imu { .. } => None,
        })
    }

    /// An empty slice must return the error, not index out of bounds (#311). `build_event_stream`
    /// is `pub` library code, so an empty read from `TestDataRecord::from_csv` -- which skips
    /// unparseable rows rather than failing -- must not abort the process.
    #[test]
    fn empty_records_are_an_error_not_a_panic() {
        let err = build_event_stream(&[], &AidingConfig::default(), false).unwrap_err();
        assert!(
            matches!(
                err,
                StrapdownError::InvalidConfiguration { field, .. } if field == "event stream records"
            ),
            "an empty record slice must report an invalid configuration, got: {err}"
        );
    }

    /// A single record is the boundary the guard must not move: it supplies `start_time` and
    /// the altitude reference, and the event list is empty because events are built from
    /// adjacent pairs. A guard written as `len() < 2` would wrongly reject this.
    #[test]
    fn single_record_yields_a_stream_with_no_events() {
        let records = create_test_records(1, 0.1);
        let stream = build_event_stream(&records, &AidingConfig::default(), false).unwrap();
        assert_eq!(stream.start_time, records[0].time);
        assert!(
            stream.events.is_empty(),
            "one record spans no interval, so it can produce no events"
        );
    }

    #[test]
    fn test_passthrough_scheduler() {
        let records = create_test_records(10, 0.1); // 10 records, 0.1s apart
        let config = AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::None,
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();

        // 9 IMU + 9 GNSS, plus one baro and one magnetometer. `PassThrough` applies to GNSS
        // alone; the other two channels take their 1 Hz default, and 10 records 0.1 s apart
        // span only 0.9 s, so each fires once (#375). Before they were scheduled this read 36
        // -- nine of each, i.e. the log's rate rather than the sensors'.
        assert_eq!(events.events.len(), 20);
        assert_eq!(count_of::<RelativeAltitudeMeasurement>(&events), 1);
        assert_eq!(count_of::<MagnetometerYawMeasurement>(&events), 1);

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
        let config = AidingConfig {
            scheduler: MeasurementScheduler::FixedInterval {
                interval_s: 0.5,
                phase_s: 0.0,
            },
            fault: GnssFaultModel::None,
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();

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
        // GNSS at 0.5, 1.0, 1.5 and 1.9 s; baro and magnetometer at 0.1 and 1.0 s, their 1 Hz
        // default over a 1.9 s span. This read 42 before #375 scheduled them -- 19 each.
        assert_eq!(measurements, 4 + 2 + 2);
        assert_eq!(count_of::<RelativeAltitudeMeasurement>(&events), 2);
        assert_eq!(count_of::<MagnetometerYawMeasurement>(&events), 2);
    }
    #[test]
    fn test_duty_cycle_scheduler() {
        let records = create_test_records(60, 1.0); // 60 records, 1s apart, 60 seconds total
        assert!(
            records.len() == 60,
            "{}",
            format!("Expected 60 records, found: {}", records.len())
        );

        let config = AidingConfig {
            scheduler: MeasurementScheduler::DutyCycle {
                on_s: 1.0,
                off_s: 1.0,
                start_phase_s: 0.0,
            },
            fault: GnssFaultModel::None,
            ..Default::default()
        };
        //
        let events = build_event_stream(&records, &config, false).unwrap();
        let measurements = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .count();

        // `windows(2)` emits one step per record after the first, so t1 runs 1..=59: 59
        // steps, each carrying an unscheduled baro and mag measurement. With `on_s: 1.0,
        // off_s: 1.0, start_phase_s: 0.0` the cycle is 2 s of OFF-then-ON, so odd seconds
        // are ON: 30 of the 59 steps carry a GNSS fix.
        //
        // This previously expected 147, i.e. 29 fixes, which was the #312 bug written down
        // as a requirement: the scheduler emitted only on the sample where the window
        // toggled into ON rather than throughout it. The old comment said so in as many
        // words -- "We should only have GNSS events when turning ON".
        assert_eq!(
            measurements,
            30 + 59 + 59,
            "expected 30 GNSS fixes plus 59 baro and 59 mag"
        );
    }

    /// The barometer and the magnetometer fire at their own rate, not at the log's.
    ///
    /// This is the defect #375 names. Before it, both channels were emitted once per record
    /// window, outside the scheduler entirely, so their update rate was whatever the log
    /// happened to be sampled at. A 1 Hz Sensor Logger recording made that look correct; the
    /// 50 Hz synthetic trajectories in `core/tests/perf_baseline.rs` got fifty pressure
    /// readings and fifty derived headings a second, each entering the filter as an
    /// independent fix with full weight, and the UKF diverged on it.
    ///
    /// Same 10 s of wall clock, three log rates, one assertion: the count must follow the
    /// schedule and not the sampling. The IMU count, which legitimately does follow the log,
    /// is asserted alongside so the test cannot pass by producing nothing.
    #[test]
    fn aiding_channels_fire_at_their_own_rate_not_the_logs() {
        for (count, interval_s, imu_events) in [(11, 1.0, 10), (101, 0.1, 100), (501, 0.02, 500)] {
            let records = create_test_records(count, interval_s);
            let stream = build_event_stream(&records, &AidingConfig::default(), false).unwrap();

            let imu = stream
                .events
                .iter()
                .filter(|e| matches!(e, Event::Imu { .. }))
                .count();
            assert_eq!(
                imu, imu_events,
                "the IMU rate is the log's rate, and must still be, at {interval_s}s spacing"
            );

            for (channel, times) in [
                (
                    "barometer",
                    times_of::<RelativeAltitudeMeasurement>(&stream),
                ),
                (
                    "magnetometer",
                    times_of::<MagnetometerYawMeasurement>(&stream),
                ),
            ] {
                // Ten seconds on a 1 Hz schedule is ten or eleven fixes: a log sampled faster
                // than the schedule has a record available before the first scheduled tick at
                // t = 0, and takes it. That leading partial interval is the *only* freedom the
                // count has -- 50 Hz gives 11, not 500.
                assert!(
                    (10..=11).contains(&times.len()),
                    "{channel} fired {} times over 10 s at {interval_s}s spacing; the schedule \
                     is 1 Hz and the log rate must not enter into it",
                    times.len()
                );
                // The rate itself, which a count alone cannot show: every gap but the leading
                // partial one is exactly the scheduled interval.
                for pair in times.windows(2).skip(usize::from(times.len() == 11)) {
                    assert_approx_eq!(pair[1] - pair[0], 1.0, 1e-9);
                }
            }
        }
    }

    /// A gap in the log must not let the next records through early.
    ///
    /// The emission clock used to advance by a single interval per emission, so a record
    /// arriving after a gap emitted and left the tick behind it -- and the very next record,
    /// milliseconds later, passed too. A 1 Hz rate limit that delivers two fixes 0.1 s apart
    /// is not a rate limit, and on an aiding channel it is the same over-counting #375 exists
    /// to stop, just triggered by the data instead of the configuration.
    ///
    /// Invisible to every baseline scenario: `test_data.csv` is spaced at exactly 1.000 s and
    /// the synthetic trajectories at exactly 0.02 s, so no gated row has a gap in it. A
    /// Sensor Logger export with a dropped sample does.
    #[test]
    fn a_gap_in_the_log_does_not_let_the_next_records_through_early() {
        let base_time = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        // 0.0, then a 2 s hole, then three records 100 ms apart.
        let offsets_s = [0.0, 0.5, 2.5, 2.6, 2.7, 3.6];
        let mut records = create_test_records(offsets_s.len(), 1.0);
        for (record, offset) in records.iter_mut().zip(offsets_s) {
            record.time = base_time + chrono::Duration::milliseconds((offset * 1000.0) as i64);
        }

        let stream = build_event_stream(&records, &AidingConfig::default(), false).unwrap();
        let times = times_of::<RelativeAltitudeMeasurement>(&stream);

        for pair in times.windows(2) {
            assert!(
                pair[1] - pair[0] > 0.5,
                "a 1 Hz schedule emitted at {:?}, which contains a {:.3} s gap between \
                 consecutive fixes -- the clock fell behind across the hole in the log and \
                 then burst",
                times,
                pair[1] - pair[0]
            );
        }
        // 0.5 takes the first tick, 2.5 the one after the hole, 3.6 the next.
        assert_eq!(times.len(), 3, "emitted at {times:?}");
    }

    /// A 1 Hz log is left exactly as it was, which is what keeps the real-data baselines fixed.
    ///
    /// `core/tests/test_data.csv` is 5,366 records spaced at exactly 1.000 s, and every
    /// `real_*` row in `core/tests/perf_baseline.json` was measured with one baro and one
    /// magnetometer update per record. The 1 Hz default must therefore be a no-op on a 1 Hz
    /// log -- if it skipped even one record the baselines would move, and a scheduling change
    /// would have silently become an accuracy change.
    #[test]
    fn the_default_schedule_is_a_no_op_on_a_one_hertz_log() {
        let records = create_test_records(600, 1.0);
        let stream = build_event_stream(&records, &AidingConfig::default(), false).unwrap();
        assert_eq!(count_of::<RelativeAltitudeMeasurement>(&stream), 599);
        assert_eq!(count_of::<MagnetometerYawMeasurement>(&stream), 599);
    }

    /// Each channel keeps its own emission clock, so one sensor's rate cannot set another's.
    #[test]
    fn each_aiding_channel_keeps_its_own_emission_clock() {
        let records = create_test_records(1001, 0.02); // 20 s at 50 Hz
        let config = AidingConfig {
            scheduler: MeasurementScheduler::FixedInterval {
                interval_s: 10.0,
                phase_s: 0.0,
            },
            baro_scheduler: MeasurementScheduler::FixedInterval {
                interval_s: 2.0,
                phase_s: 0.0,
            },
            magnetometer_scheduler: MeasurementScheduler::PassThrough,
            ..Default::default()
        };
        let stream = build_event_stream(&records, &config, false).unwrap();

        let gnss = count_of::<GPSPositionAndVelocityMeasurement>(&stream);
        // Fixes at 0.02, 10.0 and 20.0 s.
        assert_eq!(gnss, 3, "GNSS on its own 10 s schedule");
        // 0.02 s, then every 2 s through 20.0 s.
        assert_eq!(
            count_of::<RelativeAltitudeMeasurement>(&stream),
            11,
            "the barometer on its own 2 s schedule"
        );
        // Pass-through means one per record window, the old unconditional behaviour, which is
        // still available -- it is just no longer what you get without asking.
        assert_eq!(
            count_of::<MagnetometerYawMeasurement>(&stream),
            1000,
            "a pass-through magnetometer still fires on every record"
        );
    }

    /// A duty-cycled magnetometer produces a heading outage, the way a duty-cycled GNSS
    /// produces a position one. #372 wants this for the barometer and #371 for the
    /// magnetometer, and reusing the GNSS scheduler is what makes it free.
    #[test]
    fn the_barometers_noise_is_configurable_and_its_default_holds_r_where_it_was() {
        let records = create_test_records(10, 1.0);

        // 1. The default reproduces the variance the hardcoded `diag([5.0])` produced.
        //    `5.0` was an `R` entry -- a variance -- so the default `noise_std` is its square
        //    root, and squaring it must land back on 5.0. No `f64` squares to exactly 5.0, so
        //    this is the ulp the round trip costs, not a tolerance for a retune.
        let stream =
            build_event_stream(&records, &AidingConfig::default(), false).expect("default stream");
        let baro = first_of::<RelativeAltitudeMeasurement>(&stream)
            .expect("a default config emits barometric altitude");
        assert_approx_eq!(baro.noise_std, BAROMETRIC_ALTITUDE_NOISE_M, 1e-15);
        assert_approx_eq!(baro.get_noise()[(0, 0)], 5.0, 1e-14);

        // 2. A scenario can ask for a good barometer or a bad one, which is the whole of
        //    #375: `R` follows the *square* of what it sets.
        for noise_std in [0.1_f64, 25.0] {
            let stream = build_event_stream(
                &records,
                &AidingConfig {
                    baro_noise_std_m: noise_std,
                    ..Default::default()
                },
                false,
            )
            .expect("configured stream");
            let baro = first_of::<RelativeAltitudeMeasurement>(&stream)
                .expect("a configured barometer is still emitted");
            assert_approx_eq!(baro.noise_std, noise_std, 1e-15);
            assert_approx_eq!(baro.get_noise()[(0, 0)], noise_std * noise_std, 1e-12);
        }
    }

    #[test]
    fn an_aiding_channel_can_be_duty_cycled_into_an_outage() {
        let records = create_test_records(101, 1.0); // 100 s at 1 Hz
        let config = AidingConfig {
            magnetometer_scheduler: MeasurementScheduler::DutyCycle {
                on_s: 20.0,
                off_s: 10.0,
                start_phase_s: 20.0,
            },
            ..Default::default()
        };
        let stream = build_event_stream(&records, &config, false).unwrap();

        let heading_times: Vec<f64> = stream
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Measurement { meas, elapsed_s }
                    if meas.as_any().is::<MagnetometerYawMeasurement>() =>
                {
                    Some(*elapsed_s)
                }
                _ => None,
            })
            .collect();

        // 20 s of initial ON, then the cycle: 10 s OFF then 20 s ON, repeating from t = 20.
        // So 20-30, 50-60 and 80-90 are OFF and everything else is ON.
        for off in [25.0, 55.0, 85.0] {
            assert!(
                !heading_times.iter().any(|t| (t - off).abs() < 0.5),
                "no heading fix at {off}s, which is inside an OFF window"
            );
        }
        for on in [10.0, 35.0, 65.0, 95.0] {
            assert!(
                heading_times.iter().any(|t| (t - on).abs() < 0.5),
                "a heading fix at {on}s, which is inside an ON window"
            );
        }
    }

    /// A duty cycle must withhold GNSS for the whole OFF window and deliver it for the whole
    /// ON window -- the property #312 broke, and the one a toggle-counting test cannot see.
    #[test]
    fn test_duty_cycle_emits_throughout_on_window() {
        // 1 Hz for 100 s, 20 s ON then 10 s OFF, no initial phase.
        let records = create_test_records(100, 1.0);
        let config = AidingConfig {
            scheduler: MeasurementScheduler::DutyCycle {
                on_s: 20.0,
                off_s: 10.0,
                start_phase_s: 20.0,
            },
            fault: GnssFaultModel::None,
            ..Default::default()
        };

        let stream = build_event_stream(&records, &config, false).unwrap();
        // The GNSS fix is the only multi-dimensional measurement in the stream; baro and mag
        // are scalar and are not scheduled.
        let fix_times: Vec<f64> = stream
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Measurement { meas, elapsed_s } if meas.get_dimension() > 1 => {
                    Some(*elapsed_s)
                }
                _ => None,
            })
            .collect();

        // Timeline: ON [0, 20), OFF [20, 30), ON [30, 50), OFF [50, 60), ON [60, 80),
        // OFF [80, 90), ON [90, 100).
        let expect_on = |t: f64| {
            let into = (t - 20.0).rem_euclid(30.0);
            t < 20.0 || into >= 10.0
        };
        for t in 1..100 {
            let t = f64::from(t);
            let delivered = fix_times.iter().any(|fix| (fix - t).abs() < 1e-6);
            assert_eq!(
                delivered,
                expect_on(t),
                "at t={t} s the fix should{} have been delivered",
                if expect_on(t) { "" } else { " not" }
            );
        }

        // And the counts, so a regression that shifts every window by one still fails.
        assert_eq!(
            fix_times.len(),
            (1..100).filter(|t| expect_on(f64::from(*t))).count(),
            "delivered fix count should match the ON windows"
        );
    }

    /// A configuration that cannot describe an outage must not silently suppress all GNSS.
    ///
    /// Withholding every fix is far harder to notice than an outage that fails to happen --
    /// it looks like a filter problem, not a config problem -- so a degenerate cycle passes
    /// fixes through instead.
    #[test]
    fn test_duty_cycle_with_degenerate_cycle_passes_fixes_through() {
        let records = create_test_records(20, 1.0);
        for (on_s, off_s) in [(0.0, 0.0), (-5.0, 0.0)] {
            let config = AidingConfig {
                scheduler: MeasurementScheduler::DutyCycle {
                    on_s,
                    off_s,
                    start_phase_s: 0.0,
                },
                fault: GnssFaultModel::None,
                ..Default::default()
            };
            let stream = build_event_stream(&records, &config, false).unwrap();
            let fixes = stream
                .events
                .iter()
                .filter(|event| {
                    matches!(event, Event::Measurement { meas, .. } if meas.get_dimension() > 1)
                })
                .count();
            assert_eq!(
                fixes, 19,
                "on_s={on_s}, off_s={off_s} should deliver every fix, not withhold them"
            );
        }
    }
    #[test]
    fn test_degraded_fault_model() {
        let records = create_test_records(10, 0.1);
        let config = AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::Degraded {
                rho_pos: 0.99,
                sigma_pos_m: 3.0,
                rho_vel: 0.95,
                sigma_vel_mps: 0.3,
                r_scale: 5.0,
            },
            seed: 500,
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();

        // Find GNSS events
        let gnss_events: Vec<&Event> = events
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
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
        let mut prev_lat: Option<f64> = None;
        let mut prev_lon: Option<f64> = None;

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
                if let Some(prev_lat) = prev_lat
                    && (meas.latitude - prev_lat).abs() > 1e-10
                {
                    all_same = false;
                }
                prev_lat = Some(meas.latitude);

                if let Some(prev_lon) = prev_lon
                    && (meas.longitude - prev_lon).abs() > 1e-10
                {
                    all_same = false;
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
            /*horiz_std_m*/ 3.0, /*vert_std_m*/ /*vel_std_mps*/ 0.2,
        );

        assert_approx_eq!(vn_c, 0.02, 0.001);
        assert_approx_eq!(ve_c, -0.01, 0.001);
    }

    #[test]
    fn test_hijack_fault_model() {
        let records = create_test_records(30, 0.1); // 30 records, 0.1s apart, total 3.0 seconds
        let offset_n = 50.0;
        let offset_e = 30.0;

        let config = AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::Hijack {
                offset_n_m: offset_n,
                offset_e_m: offset_e,
                start_s: 1.0,
                duration_s: 1.0, // Hijack from 1.0s to 2.0s
            },
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();

        // Find GNSS events and group by time
        let mut gnss_by_time: Vec<(f64, &GPSPositionAndVelocityMeasurement)> = Vec::new();
        for event in &events.events {
            if let Event::Measurement { meas, elapsed_s } = event
                && let Some(gps) = meas
                    .as_any()
                    .downcast_ref::<GPSPositionAndVelocityMeasurement>()
            {
                gnss_by_time.push((*elapsed_s, gps));
            }
        }
        gnss_by_time.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Original position
        let original_lat = 37.0;
        let original_lon = -122.0;

        // Check measurements before, during, and after hijack
        for (time, meas) in gnss_by_time {
            if (1.0 - 1e-6..=2.0 + 1e-6).contains(&time) {
                // During hijack: positions should be offset
                assert!((meas.latitude - original_lat).abs() > 1e-6);
                assert!((meas.longitude - original_lon).abs() > 1e-6);
            } else {
                // Before hijack or after hijack: positions should be near original
                assert!((meas.latitude - original_lat).abs() < 1e-6);
                assert!((meas.longitude - original_lon).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_combo_fault_model() {
        // Test that the combo fault model functionality exists
        // Note: Due to commented code in apply_fault for Combo, this is a minimal test
        let records = create_test_records(10, 0.1);

        let config = AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::Combo(vec![GnssFaultModel::None, GnssFaultModel::None]),
            ..Default::default()
        };

        // This should at least not crash
        let events = build_event_stream(&records, &config, false).unwrap();
        assert!(!events.events.is_empty());
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
        let config = AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::SlowBias {
                drift_n_mps: 1.0,
                drift_e_mps: 0.5,
                rotate_omega_rps: 0.1,
                q_bias: 0.0,
            },
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();
        // Should have events
        assert!(!events.events.is_empty());

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
        let config = AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::SlowBias {
                drift_n_mps: 0.0,
                drift_e_mps: 0.0,
                rotate_omega_rps: 0.0,
                q_bias: 1.0,
            },
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();
        // Should have events
        assert!(!events.events.is_empty());

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
        // First record (reference), second record with NaN accuracies
        let records = vec![
            TestDataRecord {
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
            },
            TestDataRecord {
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
            },
        ];

        let config = AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::None,
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();

        // Should have events even with NaN accuracies
        assert!(!events.events.is_empty());

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
        let config = AidingConfig {
            scheduler: MeasurementScheduler::DutyCycle {
                on_s: 0.5,
                off_s: 0.5,
                start_phase_s: 0.0,
            },
            fault: GnssFaultModel::None,
            ..Default::default()
        };

        let events = build_event_stream(&records, &config, false).unwrap();
        // Should have events
        assert!(!events.events.is_empty());
    }
}
#[cfg(test)]
mod serialization_tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_cfg() -> AidingConfig {
        AidingConfig {
            scheduler: MeasurementScheduler::PassThrough,
            fault: GnssFaultModel::Degraded {
                rho_pos: 0.99,
                sigma_pos_m: 3.0,
                rho_vel: 0.95,
                sigma_vel_mps: 0.3,
                r_scale: 5.0,
            },
            ..Default::default()
        }
    }

    #[test]
    fn json_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("json");
        cfg.to_json(&path).unwrap();
        let loaded = AidingConfig::from_json(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }

    #[test]
    fn yaml_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("yaml");
        cfg.to_yaml(&path).unwrap();
        let loaded = AidingConfig::from_yaml(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }

    #[test]
    fn toml_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("toml");
        cfg.to_toml(&path).unwrap();
        let loaded = AidingConfig::from_toml(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }

    #[test]
    fn generic_dispatch_roundtrip() {
        let cfg = sample_cfg();
        let f = NamedTempFile::new().unwrap();
        let path = f.path().with_extension("json");
        cfg.to_file(&path).unwrap();
        let loaded = AidingConfig::from_file(&path).unwrap();
        assert_eq!(cfg.seed, loaded.seed);
    }
}
