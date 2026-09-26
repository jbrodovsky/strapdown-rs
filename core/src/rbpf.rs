//! Rao-Blackwellized (marginalized) particle filter for map-aided inertial navigation, as
//! Canciani & Raquet describe it.
//!
//! # The filter
//!
//! This is the marginalized particle filter of Schön, Gustafsson & Nordlund (2005) in the form
//! Canciani & Raquet apply it to magnetic-anomaly navigation ("Airborne Magnetic Anomaly
//! Navigation", IEEE TAES 53(1):67-80, 2017, §III and Algorithm 1; Canciani's AFIT
//! dissertation, 2016, Ch. 3). Equation numbers below are the paper's.
//!
//! * A **nominal trajectory** is mechanized with the IMU samples, with **barometer aiding in
//!   the mechanization**: a third-order loop that feeds the altitude difference from the
//!   barometer back into altitude, vertical velocity and a vertical-acceleration correction.
//!   The filter estimates the errors in that aided solution (eq. 4).
//! * **Particles sample horizontal position error only**, `(δlat, δlon)` (eq. 5).
//! * **Every other state is Rao-Blackwellised** into one Kalman filter whose covariance all
//!   particles share (Algorithm 1, step 1b): the paper's eleven linear states, `[δh, δv(3),
//!   ε(3), δh_a, δâ, V, c]` -- altitude, velocity and nav-frame tilt error (the Pinson model),
//!   the barometer-aiding error and the loop's vertical-acceleration error, and for each map
//!   channel a first-order Gauss-Markov temporal variation `V` and a constant offset `c`.
//! * **The time update runs once per measurement epoch** (Algorithm 1, steps 2-3 and 8-9):
//!   between measurements only the nominal is mechanized, while the error-state transition and
//!   the process noise accumulate. The nonlinear process noise is zero by default (eq. 19;
//!   see departure 6); the linear noise is the paper's `diag(0, VRW, ARW, B, 0, T, 0)`
//!   (eqs. 20-22).
//! * **Measurement update** (steps 5-7): each particle is weighted by its residual under
//!   `C P Cᵀ + R`, the likelihood with the linear states integrated out (eq. 24), and its
//!   linear states then take a Kalman step with a gain shared by the whole cloud (eqs. 26-29).
//!   For a map, `C` selects `V + c` (eq. 17).
//!
//! # State
//!
//! | partition | states | units |
//! |---|---|---|
//! | sampled | `δlat, δlon` | rad |
//! | linear | `δh` | m, positive up |
//! | | `δv_N, δv_E, δv_vert` | m/s, vertical in the nominal's frame |
//! | | `ε` (3) | rad, nav-frame tilt: `C_true = exp([ε×]) C_nom` |
//! | | `δh_a` | m: error of the barometer altitude the loop is aided with |
//! | | `δâ` | m/s²: error of the loop's vertical-acceleration correction |
//! | | `V_k, c_k` per map channel | the channel's unit |
//!
//! Errors are truth minus nominal throughout. The ordering is that of the nine-state
//! transition Jacobian followed by the loop and map states, so the partition is the first two
//! rows of `F` against the rest. [`RaoBlackwellizedParticleFilter::estimate`] reports the
//! Kalman filters' layout, `[lat, lon, alt, v(3), roll, pitch, yaw, b_a(3), b_g(3)]`, followed
//! by one map bias `V_k + c_k` per channel. The filter estimates no IMU biases, so those six
//! rows are zero with zero variance.
//!
//! # The barometer loop's error model
//!
//! The paper prints the loop's coupling into the Pinson model -- its `D` and `B` blocks
//! (eqs. 12-13) -- but neither the loop's feedback on the altitude error itself nor its gains.
//! Both are completed here from the standard third-order loop (Titterton & Weston), with the
//! gains placing all three closed-loop poles at `-1 / tau` for
//! [`RbpfConfig::baro_loop_time_constant_s`]. With `e = h - (h_baro - b̂)`, the nominal runs
//! `ḣ = v_up - k₁e`, `v̇_up = a_up - k₂e - â`, `â̇ = k₃e`, and the errors follow
//! `δḣ = δv_up - k₁(δh + δh_a)`, `δv̇_up = … - k₂(δh + δh_a) - δâ`, `δâ̇ = k₃(δh + δh_a)`.
//!
//! # Where this departs from the paper
//!
//! 1. **Closed loop.** The weighted-mean error is fed back into the nominal after every
//!    update. Canciani & Raquet ran a navigation-grade INS open loop and note that "feedback
//!    may be required with a less accurate INS" (p. 76).
//! 2. **The WGS84 discrete-time error Jacobian** ([`state_transition_jacobian`]) stands in for
//!    the spherical-Earth continuous matrices of eqs. 8-11: the same Pinson structure.
//! 3. **GNSS and magnetometer-heading fixes** are supported alongside the map, through the same
//!    update. The paper's filter takes the magnetometer map only.
//! 4. **Roughening** after resampling, which the paper does not use
//!    ([`RbpfConfig::roughening_factor`]; `0.0` turns it off).
//! 5. **The accumulated process noise carries each state's own decay** across an epoch, where
//!    eqs. 20-22 are first order in the epoch length. The two agree for epochs short against
//!    the correlation times; for the long gaps between sparse fixes the first-order form would
//!    misstate a Gauss-Markov state's variance, and this does not.
//! 6. **Horizontal position process noise is configurable**
//!    ([`RbpfConfig::horizontal_process_noise_std_m`]). Its default is the paper's zero
//!    (eq. 19), which diverges with GNSS-rate fixes on MEMS data; the configurations under
//!    `conf/` use 1 m per root-second.

use crate::StrapdownError;
use crate::gating::{
    GatePolicy, GateRecovery, InnovationGate, UpdateOutcome, normalized_innovation_squared,
};
use crate::horizontal_meters_to_radians;
use crate::kalman::{expand_measurement_jacobian, imu_sample_from_input};
use crate::linalg::{robust_spd_solve, symmetrize};
use crate::linearize::{
    attitude_reset_jacobian, euler_rate_matrix_inverse, state_transition_jacobian,
};
use crate::measurements::{MeasurementModel, RelativeAltitudeMeasurement};
use crate::particle::{
    ParticleResamplingStrategy, multinomial_resample, residual_resample, stratified_resample,
    systematic_resample,
};
use crate::{IMUQuality, ImuSample, InputModel, NavigationFilter, StrapdownState, mechanize};

use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, Rotation3, SymmetricEigen, Vector2, Vector3};
use rand::prelude::*;
use rand_distr::Normal;

/// Dimension of the sampled partition: latitude and longitude error.
const SAMPLED_STATE_DIM: usize = 2;

/// Width of the reported state before the map biases: the Kalman filters' fifteen navigation
/// and IMU-bias states, the last six of which this filter reports as zero.
const REPORTED_BASE_DIM: usize = 15;

/// Reported layouts must agree with the one [`crate::sim::NavigationResult`] reads, where map
/// biases start after the fifteen navigation and IMU-bias states.
const _: () = assert!(REPORTED_BASE_DIM == crate::sim::NAVIGATION_STATES);

/// Indices of the three attitude angles within the reported state.
///
/// These are the channels that live on the circle rather than the line, so they are the ones
/// [`RaoBlackwellizedParticleFilter::estimate`] averages with [`circular_mean`] and differences
/// with [`crate::wrap_to_pi`].
const ATTITUDE_STATE_INDICES: [usize; 3] = [6, 7, 8];

/// First reported index of the (unestimated, zero) IMU bias block.
const REPORTED_IMU_BIAS: usize = 9;

/// Default time constant of the barometer loop, seconds. The paper gives no gains; ten seconds
/// is a loop fast enough to hold a MEMS vertical channel, and is configurable.
pub const DEFAULT_BARO_LOOP_TIME_CONSTANT_S: f64 = 10.0;

/// Default steady-state standard deviation of the barometer-aiding error `δh_a`, metres.
///
/// The reference-pressure drift over an hour that the Kalman filters' barometric bias is sized
/// from, [`crate::sim::BARO_BIAS_DRIFT_M_PER_HOUR`].
pub const DEFAULT_BARO_ERROR_STD_M: f64 = crate::sim::BARO_BIAS_DRIFT_M_PER_HOUR;

/// Default correlation time of the barometer-aiding error, seconds.
pub const DEFAULT_BARO_ERROR_TIME_CONSTANT_S: f64 = 3600.0;

/// Default correlation time of a map channel's temporal variation `V`, seconds: Canciani &
/// Raquet's five minutes (§III).
pub const DEFAULT_MAP_VARIATION_TIME_CONSTANT_S: f64 = 300.0;

/// Gordon, Salmond & Smith's roughening coefficient, as a fraction of the cloud extent.
///
/// The 1993 paper tunes `K` per problem and uses 0.2 for its examples. That value is kept
/// here: on the gated `real_rbpf_slice__rbpf` scenario it is what stops the cloud collapsing
/// without measurably widening the epochs that were never degenerate.
const DEFAULT_ROUGHENING_FACTOR: f64 = 0.2;

/// Relative eigenvalue floor below which a direction of the position innovation `N` is
/// treated as carrying no information. See [`position_innovation_factors`].
const POSITION_INNOVATION_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Iteration cap on the 2x2 eigendecomposition of `N`, which converges in a handful.
const POSITION_INNOVATION_MAX_ITERATIONS: usize = 100;

/// Weighted mean of angles, computed on the circle.
///
/// The mean direction of the unit vectors at `angles`, weighted by `weights`:
/// `atan2(sum w sin(x), sum w cos(x))`. Unlike a linear mean this is invariant to where
/// each angle is wrapped, so a cloud straddling the +/-pi branch cut averages to the
/// direction between its members rather than to the far side of the circle, and the
/// result is always on `[-pi, pi]`.
///
/// It agrees with the linear mean to second order in the spread, so a tight cloud is
/// unaffected. A cloud with no mean direction -- one spread evenly around the circle, so
/// that the resultant vector is zero -- returns 0 rather than failing, which is
/// `atan2(0, 0)`.
fn circular_mean<'a>(angles: impl Iterator<Item = (&'a f64, f64)>) -> f64 {
    let (sin_sum, cos_sum) = angles.fold((0.0, 0.0), |(s, c), (angle, weight)| {
        (s + weight * angle.sin(), c + weight * angle.cos())
    });
    sin_sum.atan2(cos_sum)
}

/// Convert a (north, east) standard deviation in metres to radians of latitude and longitude.
///
/// The two factors differ by `cos(latitude)` and are supplied by
/// [`crate::horizontal_meters_to_radians`], which reads them off the WGS84 radii of
/// curvature at the point given. Applying the latitude factor to both -- what this did
/// until #331 -- leaves the east extent short by that cosine.
fn horizontal_std_to_radians(
    north_m: f64,
    east_m: f64,
    latitude_rad: f64,
    altitude_m: f64,
) -> Vector2<f64> {
    let (latitude_radians_per_meter, longitude_radians_per_meter) =
        horizontal_meters_to_radians(latitude_rad.to_degrees(), altitude_m);
    Vector2::new(
        north_m * latitude_radians_per_meter,
        east_m * longitude_radians_per_meter,
    )
}

/// One-step decay `exp(-dt / tau)` of a first-order Gauss-Markov process.
///
/// An infinite correlation time gives exactly 1: a random constant.
fn markov_decay(dt: f64, time_constant_s: f64) -> f64 {
    (-dt / time_constant_s).exp()
}

/// One-step driving variance of a first-order Gauss-Markov process of steady-state standard
/// deviation `std` and correlation time `time_constant_s`.
///
/// The exact discretisation, `std^2 (1 - exp(-2 dt / tau))`, rather than the continuous
/// `2 std^2 / tau * dt` the paper writes (eqs. 21-22): with the decay of [`markov_decay`] it
/// makes `std^2` the exact fixed point of the variance recursion at any step size. The two
/// agree to first order in `dt / tau`. An infinite correlation time gives zero.
fn markov_process_variance(dt: f64, time_constant_s: f64, std: f64) -> f64 {
    -(std * std) * (-2.0 * dt / time_constant_s).exp_m1()
}

/// Where each state sits in a particle's linear partition.
///
/// `[δh, δv(3), ε(3), δh_a, δâ]`, then `(V_k, c_k)` for each map channel: the paper's `xˡ`
/// (eq. 4). Linear index `i` is row `i + 2` of the full error-state transition.
#[derive(Clone, Copy, Debug)]
struct LinearLayout {
    /// Number of map channels, each carrying a `V` and a `c`.
    map_channels: usize,
}

impl LinearLayout {
    /// Altitude error.
    const ALTITUDE: usize = 0;
    /// First of the three velocity errors.
    const VELOCITY: usize = 1;
    /// Vertical velocity error, in the nominal's frame.
    const VERTICAL_VELOCITY: usize = 3;
    /// First of the three tilt errors.
    const TILT: usize = 4;
    /// Error of the barometer altitude the loop is aided with, `δh_a`.
    const BARO_ERROR: usize = 7;
    /// Error of the loop's vertical-acceleration correction, `δâ`.
    const ACCEL_CORRECTION: usize = 8;
    /// First map state.
    const MAP_BASE: usize = 9;

    /// Temporal variation `V` of map channel `channel`.
    const fn variation(channel: usize) -> usize {
        Self::MAP_BASE + 2 * channel
    }

    /// Constant offset `c` of map channel `channel`.
    const fn offset(channel: usize) -> usize {
        Self::MAP_BASE + 2 * channel + 1
    }

    /// Width of the linear partition.
    const fn dim(self) -> usize {
        Self::MAP_BASE + 2 * self.map_channels
    }

    /// Width of the reported state.
    const fn reported_dim(self) -> usize {
        REPORTED_BASE_DIM + self.map_channels
    }

    /// Width of the full error vector `[xⁿ; xˡ]`.
    const fn full_dim(self) -> usize {
        SAMPLED_STATE_DIM + self.dim()
    }
}

/// Refuse a configuration the filter cannot carry.
///
/// The per-channel map vectors must each hold exactly [`RbpfConfig::map_bias_channels`]
/// entries: a short one would leave a channel with no prior, and there is no default that is
/// right in every unit. Standard deviations must be finite and non-negative, initial values
/// finite, and correlation and loop time constants positive -- a correlation time may be
/// infinite, a random constant.
///
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] naming the offending field.
fn validate_config(config: &RbpfConfig) -> Result<(), StrapdownError> {
    let invalid = |field: &'static str, reason: String| {
        Err(StrapdownError::InvalidConfiguration { field, reason })
    };
    for (field, values) in [
        ("map_bias_initial", &config.map_bias_initial),
        ("map_bias_init_std", &config.map_bias_init_std),
        ("map_variation_std", &config.map_variation_std),
        (
            "map_variation_time_constant_s",
            &config.map_variation_time_constant_s,
        ),
    ] {
        if values.len() != config.map_bias_channels {
            return invalid(
                field,
                format!(
                    "holds {} entries for {} map channels; give exactly one per channel",
                    values.len(),
                    config.map_bias_channels
                ),
            );
        }
    }
    if let Some(value) = config.map_bias_initial.iter().find(|v| !v.is_finite()) {
        return invalid("map_bias_initial", format!("{value} is not a usable value"));
    }
    let unusable_std = |value: f64| !value.is_finite() || value < 0.0;
    for (field, values) in [
        ("map_bias_init_std", &config.map_bias_init_std),
        ("map_variation_std", &config.map_variation_std),
    ] {
        if let Some(value) = values.iter().find(|v| unusable_std(**v)) {
            return invalid(field, format!("{value} is not a usable standard deviation"));
        }
    }
    let unusable_time_constant = |value: f64| value.is_nan() || value <= 0.0;
    if let Some(value) = config
        .map_variation_time_constant_s
        .iter()
        .find(|tau| unusable_time_constant(**tau))
    {
        return invalid(
            "map_variation_time_constant_s",
            format!("{value} is not a usable correlation time; it must be positive"),
        );
    }
    for (field, std) in [
        ("baro_error_std_m", config.baro_error_std_m),
        (
            "vertical_accel_error_init_std_mps2",
            config.vertical_accel_error_init_std_mps2,
        ),
    ] {
        if unusable_std(std) {
            return invalid(field, format!("{std} is not a usable standard deviation"));
        }
    }
    if unusable_time_constant(config.baro_error_time_constant_s) {
        return invalid(
            "baro_error_time_constant_s",
            format!(
                "{} is not a usable correlation time; it must be positive",
                config.baro_error_time_constant_s
            ),
        );
    }
    let loop_tau = config.baro_loop_time_constant_s;
    if !loop_tau.is_finite() || loop_tau <= 0.0 {
        return invalid(
            "baro_loop_time_constant_s",
            format!(
                "{loop_tau} is not a usable loop time constant; it must be positive and finite"
            ),
        );
    }
    Ok(())
}

/// RBPF configuration parameters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RbpfConfig {
    /// Number of particles in the cloud; fixed for the life of the filter.
    pub num_particles: usize,
    /// Strategy used to resample the cloud once the effective sample size drops below
    /// the trigger described by [`RbpfConfig::effective_sample_threshold`].
    pub resampling_strategy: ParticleResamplingStrategy,
    /// Resampling trigger as a fraction of `num_particles`: the cloud is resampled when the
    /// effective sample size drops below this fraction of the particle count. The default,
    /// `1.0`, resamples after every weighting update, as Canciani & Raquet do (step 6).
    pub effective_sample_threshold: f64,
    /// Initial position-error standard deviation in metres, as (north, east, up).
    ///
    /// The horizontal entries spread the particle cloud, converted to radians through the
    /// WGS84 radii of curvature at the nominal position (#331). The up entry is the prior of
    /// the altitude error, which is a Kalman state and so lives in the conditional covariance
    /// rather than in the cloud.
    pub position_init_std_m: Vector3<f64>,
    /// Initial standard deviation of each of the three velocity error states, in m/s.
    pub velocity_init_std_mps: f64,
    /// Initial standard deviation of each of the three tilt error states, in radians.
    pub attitude_init_std_rad: f64,
    /// Velocity random walk, **m/s per root-second**, on the three velocity error states: the
    /// paper's VRW (eq. 20). Its variance grows linearly in elapsed time, whatever the log's
    /// sample rate (#374).
    pub velocity_process_noise_std_mps: f64,
    /// Angular random walk, **rad per root-second**, on the three tilt error states: the
    /// paper's ARW (eq. 20).
    pub attitude_process_noise_std_rad: f64,
    /// Random walk on the sampled horizontal position error, (north, east) in **m per
    /// root-second**, accumulated over each epoch as `(std * sqrt(epoch))^2`.
    ///
    /// Zero -- the default -- is the paper's eq. 19. It does not survive GNSS-rate fixes on
    /// MEMS data: each epoch's time update is then a noiseless observation of the velocity and
    /// tilt errors, which moves their uncertainty out of the conditional covariance and into
    /// the particles' spread, and resampling on a metre-level fix destroys that spread. On the
    /// reference recording the conditional velocity sigma fell to millimetres per second,
    /// GNSS velocity fixes stopped correcting anything, and the solution ran 17 km off; 1 m per
    /// root-second held it to 3.8 m. The configurations under `conf/` set that.
    pub horizontal_process_noise_std_m: Vector2<f64>,
    /// Time constant of the barometer loop, seconds. The gains place all three closed-loop
    /// poles at `-1 / tau`: `k₁ = 3/tau`, `k₂ = 3/tau² + 2g/R`, `k₃ = 1/tau³`.
    pub baro_loop_time_constant_s: f64,
    /// Steady-state (and initial) standard deviation of the barometer-aiding error `δh_a`,
    /// metres: `σ_b` in the paper's eq. 21.
    pub baro_error_std_m: f64,
    /// Correlation time of the barometer-aiding error, seconds: `τ_b` in the paper's eqs.
    /// 12 and 21.
    pub baro_error_time_constant_s: f64,
    /// Initial standard deviation of the loop's vertical-acceleration error `δâ`, m/s²: the
    /// accelerometer error the loop has not yet learned. The paper gives it no driving noise.
    pub vertical_accel_error_init_std_mps2: f64,
    /// Number of map channels, each estimated as a temporal variation `V` plus a constant
    /// offset `c` (Canciani & Raquet eqs. 14, 16-17).
    ///
    /// A map measurement is predicted as the map value at the particle plus `V + c`, and the
    /// reported state carries that sum, one entry per channel after the fifteen navigation and
    /// bias states. [`Self::map_bias_initial`], [`Self::map_bias_init_std`],
    /// [`Self::map_variation_std`] and [`Self::map_variation_time_constant_s`] describe the
    /// channels one entry each, and all four must be exactly this long.
    pub map_bias_channels: usize,
    /// Initial map bias of each channel, in its own units; seeds the constant offset `c`.
    pub map_bias_initial: Vec<f64>,
    /// Prior standard deviation of each channel's total bias `V + c`, in its own units.
    ///
    /// `V` starts from its stationary distribution, [`Self::map_variation_std`], and `c`
    /// takes the rest, `sqrt(max(0, init_std^2 - variation_std^2))`, so the total prior is the
    /// one a single-bias filter would be given.
    pub map_bias_init_std: Vec<f64>,
    /// Steady-state standard deviation of each channel's temporal variation `V`: `σ_tv` in the
    /// paper's eq. 22.
    pub map_variation_std: Vec<f64>,
    /// Correlation time of each channel's temporal variation `V`, seconds: `τ_TV` in eqs. 14
    /// and 22. Infinite makes it a second random constant.
    pub map_variation_time_constant_s: Vec<f64>,
    /// Seed for the filter's random number generator, which draws the initial particle
    /// spread, the per-epoch position draws and the resampling indices. Runs with the same
    /// seed and the same inputs are reproducible.
    pub seed: u64,
    /// Roughening coefficient applied to the position cloud after a resample, as a
    /// fraction of the cloud's own pre-resample extent. `0.0` disables it, which is what
    /// Canciani & Raquet do.
    ///
    /// Resampling clones particles *exactly*: the survivors are bit-identical copies of
    /// their ancestors. When the weights degenerate onto one particle the whole cloud becomes
    /// copies of a single point and the reported position covariance drops to the float noise
    /// around zero; #385 measured horizontal sigmas of four *nanometres* against a GNSS fix
    /// specified at 3.81 m.
    ///
    /// Roughening is the standard repair (Gordon, Salmond & Smith 1993, §II-D): after
    /// resampling, jitter each particle by a Gaussian whose width is `K * E * N^(-1/d)`, where
    /// `E` is the cloud's extent along that axis, `N` the particle count and `d = 2` the
    /// dimension of the sampled partition. `E` is measured **before** the resample; afterwards
    /// the extent is the zero this is meant to repair.
    pub roughening_factor: f64,
}

impl Default for RbpfConfig {
    fn default() -> Self {
        Self {
            num_particles: 500,
            resampling_strategy: ParticleResamplingStrategy::Systematic,
            effective_sample_threshold: 1.0,
            position_init_std_m: Vector3::new(10.0, 10.0, 5.0),
            velocity_init_std_mps: 1.0,
            attitude_init_std_rad: 0.1,
            velocity_process_noise_std_mps: 1e-3,
            attitude_process_noise_std_rad: 0.01,
            horizontal_process_noise_std_m: Vector2::zeros(),
            baro_loop_time_constant_s: DEFAULT_BARO_LOOP_TIME_CONSTANT_S,
            baro_error_std_m: DEFAULT_BARO_ERROR_STD_M,
            baro_error_time_constant_s: DEFAULT_BARO_ERROR_TIME_CONSTANT_S,
            vertical_accel_error_init_std_mps2: IMUQuality::Consumer.accel_bias_instability_mps2(),
            map_bias_channels: 0,
            map_bias_initial: Vec::new(),
            map_bias_init_std: Vec::new(),
            map_variation_std: Vec::new(),
            map_variation_time_constant_s: Vec::new(),
            seed: 42,
            roughening_factor: DEFAULT_ROUGHENING_FACTOR,
        }
    }
}

/// One particle: a horizontal position hypothesis and the conditional mean of every other
/// error state given it.
#[derive(Clone, Debug)]
pub struct RbpfParticle {
    /// Horizontal position error relative to the nominal, added to it to form this particle's
    /// position: latitude and longitude, radians.
    pub position_error: Vector2<f64>,
    /// Conditional mean of the linear error states, added to the nominal: altitude, velocity,
    /// tilt, barometer-aiding error, vertical-acceleration error, then `(V, c)` per map
    /// channel. See the module docs for the layout.
    pub linear_state: DVector<f64>,
    /// Normalized importance weight; the weights of the cloud sum to one.
    pub weight: f64,
}

/// The nominal side of the barometer loop.
#[derive(Clone, Copy, Debug, Default)]
struct BaroLoop {
    /// The latest barometric altitude, metres; the loop is inactive until the first arrives.
    altitude: Option<f64>,
    /// Estimate of the barometer-aiding error, `b̂`, subtracted from the barometric altitude.
    baro_error: f64,
    /// The loop's vertical-acceleration correction, `â`, m/s².
    accel_correction: f64,
}

/// The error-state transition and process noise accumulated since the last time update.
#[derive(Clone, Debug)]
struct PendingTimeUpdate {
    /// Product of the per-sample transitions, full error vector wide.
    transition: DMatrix<f64>,
    /// Diagonal of the accumulated process noise on the linear partition.
    linear_noise: DVector<f64>,
    /// Seconds accumulated, for the sampled pair's random walk.
    elapsed_s: f64,
}

/// Rao-Blackwellized particle filter implementation.
#[derive(Debug)]
pub struct RaoBlackwellizedParticleFilter {
    config: RbpfConfig,
    layout: LinearLayout,
    particles: Vec<RbpfParticle>,
    /// Conditional covariance of the linear states, shared by every particle.
    linear_covariance: DMatrix<f64>,
    nominal: StrapdownState,
    baro: BaroLoop,
    /// Nominal temporal variation `V` of each map channel.
    nominal_variation: DVector<f64>,
    /// Nominal constant offset `c` of each map channel.
    nominal_offset: DVector<f64>,
    /// Time update awaiting the next measurement, if any sample has arrived since the last.
    pending: Option<PendingTimeUpdate>,
    rng: StdRng,
    /// Innovation gate applied by `update` together with the recovery policy that keeps
    /// a rejection from being permanent; an empty gate accepts every measurement.
    gate_policy: GatePolicy,
}

impl RaoBlackwellizedParticleFilter {
    /// Create a new RBPF with particles spread horizontally around the nominal state.
    ///
    /// Only horizontal position is drawn. Every other state starts every particle at the same
    /// conditional estimate -- zero error against the nominal -- with its prior in the shared
    /// conditional covariance.
    ///
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] if a `position_init_std_m` component is not a
    /// usable standard deviation, or, via `validate_config`, if the map channels or the
    /// barometer loop are not described usably.
    pub fn new(nominal: StrapdownState, config: RbpfConfig) -> Result<Self, StrapdownError> {
        validate_config(&config)?;
        let layout = LinearLayout {
            map_channels: config.map_bias_channels,
        };
        let mut rng = StdRng::seed_from_u64(config.seed);

        let pos_std = horizontal_std_to_radians(
            config.position_init_std_m[0],
            config.position_init_std_m[1],
            nominal.latitude,
            nominal.altitude,
        );
        let position_normal = |axis: usize, name: &'static str| {
            Normal::new(0.0, pos_std[axis]).map_err(|e| StrapdownError::InvalidConfiguration {
                field: name,
                reason: format!("{} is not a usable standard deviation: {e}", pos_std[axis]),
            })
        };
        let normal_lat = position_normal(0, "position_init_std_m[0]")?;
        let normal_lon = position_normal(1, "position_init_std_m[1]")?;
        if !config.position_init_std_m[2].is_finite() || config.position_init_std_m[2] < 0.0 {
            return Err(StrapdownError::InvalidConfiguration {
                field: "position_init_std_m[2]",
                reason: format!(
                    "{} is not a usable standard deviation",
                    config.position_init_std_m[2]
                ),
            });
        }

        let weight = 1.0 / config.num_particles as f64;
        let particles = (0..config.num_particles)
            .map(|_| RbpfParticle {
                position_error: Vector2::new(
                    normal_lat.sample(&mut rng),
                    normal_lon.sample(&mut rng),
                ),
                linear_state: DVector::zeros(layout.dim()),
                weight,
            })
            .collect();

        Ok(Self {
            linear_covariance: initial_linear_covariance(&config, layout),
            layout,
            particles,
            nominal,
            baro: BaroLoop::default(),
            nominal_variation: DVector::zeros(config.map_bias_channels),
            nominal_offset: DVector::from_column_slice(&config.map_bias_initial),
            pending: None,
            rng,
            gate_policy: GatePolicy::default(),
            config,
        })
    }

    /// Access the nominal INS state.
    pub const fn nominal_state(&self) -> &StrapdownState {
        &self.nominal
    }

    /// The barometer loop's vertical-acceleration correction, `â`, m/s².
    pub const fn baro_accel_correction(&self) -> f64 {
        self.baro.accel_correction
    }

    /// The barometer loop's estimate of the barometer-aiding error, `b̂`, metres.
    pub const fn baro_error_estimate(&self) -> f64 {
        self.baro.baro_error
    }

    /// The particle cloud.
    pub fn particles(&self) -> &[RbpfParticle] {
        &self.particles
    }

    /// The conditional covariance of the linear states, shared by every particle, as of the
    /// last time update.
    pub const fn linear_covariance(&self) -> &DMatrix<f64> {
        &self.linear_covariance
    }

    /// Number of linear states each particle carries: the paper's nine navigation and
    /// barometer states, plus two per map channel.
    pub const fn linear_state_dim(&self) -> usize {
        self.layout.dim()
    }

    /// Weighted-mean estimate of each map channel's temporal variation `V` and constant offset
    /// `c`, in that order: the nominals plus the cloud's mean errors.
    pub fn map_bias_components(&self) -> (DVector<f64>, DVector<f64>) {
        let mut variation = self.nominal_variation.clone();
        let mut offset = self.nominal_offset.clone();
        for particle in &self.particles {
            for channel in 0..self.layout.map_channels {
                variation[channel] +=
                    particle.weight * particle.linear_state[LinearLayout::variation(channel)];
                offset[channel] +=
                    particle.weight * particle.linear_state[LinearLayout::offset(channel)];
            }
        }
        (variation, offset)
    }

    /// Weighted-mean estimate of each map channel's total bias `V + c`: what a map
    /// measurement adds to the map value, and what [`Self::estimate`] reports.
    pub fn map_bias_estimate(&self) -> DVector<f64> {
        let (variation, offset) = self.map_bias_components();
        variation + offset
    }

    /// The barometer loop's gains `(k₁, k₂, k₃)`: all three closed-loop poles at `-1 / tau`.
    ///
    /// The unaided vertical channel's error grows as `δḧ = (2g/R) δh`, so the characteristic
    /// polynomial of the aided channel is `s³ + k₁s² + (k₂ - 2g/R)s + k₃`, and matching it to
    /// `(s + 1/tau)³` gives `k₂` its `2g/R` term.
    fn baro_loop_gains(&self) -> (f64, f64, f64) {
        let tau = self.config.baro_loop_time_constant_s;
        let latitude_deg = self.nominal.latitude.to_degrees();
        let gravity = crate::earth::gravity(&latitude_deg, &self.nominal.altitude);
        let (meridian, transverse, _) =
            crate::earth::principal_radii(&latitude_deg, &self.nominal.altitude);
        let radius = (meridian * transverse).sqrt() + self.nominal.altitude;
        (
            3.0 / tau,
            (2.0 * gravity).mul_add(1.0 / radius, 3.0 / (tau * tau)),
            1.0 / (tau * tau * tau),
        )
    }

    /// Mechanize the nominal through one inertial sample, and accumulate the error-state
    /// transition and process noise the next time update will apply.
    ///
    /// Only the nominal moves here: the particles are propagated once per measurement epoch,
    /// in [`Self::flush_time_update`] (Algorithm 1, steps 2-3).
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `sample.dt` is not strictly positive, and errors
    /// propagated from [`mechanize`].
    fn predict_sample(&mut self, sample: &ImuSample) -> Result<(), StrapdownError> {
        let dt = sample.dt;
        // The Jacobian is derived in the rate domain, so it needs the average rates.
        let rates = sample.to_rates()?;
        let transition = self.error_transition(&rates.accel, &rates.gyro, dt);
        let noise = self.process_noise(dt);

        mechanize(&mut self.nominal, sample)?;
        self.apply_baro_loop(dt);
        self.decay_nominal_markov_states(dt);

        let (full_dim, linear_dim) = (self.layout.full_dim(), self.layout.dim());
        let pending = self.pending.get_or_insert_with(|| PendingTimeUpdate {
            transition: DMatrix::identity(full_dim, full_dim),
            linear_noise: DVector::zeros(linear_dim),
            elapsed_s: 0.0,
        });
        pending.elapsed_s += dt;
        // Each diagonal noise term is carried through its own state's decay over the epoch, so
        // a Gauss-Markov state stays exactly stationary however long the epoch; see departure
        // 5 in the module docs.
        for (index, variance) in noise.iter().enumerate() {
            let decay = transition[(SAMPLED_STATE_DIM + index, SAMPLED_STATE_DIM + index)];
            pending.linear_noise[index] =
                (decay * decay).mul_add(pending.linear_noise[index], *variance);
        }
        pending.transition = &transition * &pending.transition;
        Ok(())
    }

    /// The full error-state transition matrix over one sample, evaluated on the nominal.
    ///
    /// The navigation block is [`state_transition_jacobian`] in its nav-frame rotation-vector
    /// form, the discrete counterpart of the Pinson matrices of eqs. 8-11 whose tilt is the
    /// `ε` this filter carries. While the barometer loop is running its feedback terms are
    /// added (see the module docs). The barometer-aiding error and each map channel's `V`
    /// decay as Gauss-Markov processes; `δâ` and `c` are constant.
    fn error_transition(&self, accel: &Vector3<f64>, gyro: &Vector3<f64>, dt: f64) -> DMatrix<f64> {
        let layout = self.layout;
        let navigation = state_transition_jacobian(&self.nominal, accel, gyro, dt);
        let mut transition = DMatrix::<f64>::identity(layout.full_dim(), layout.full_dim());
        transition.view_mut((0, 0), (9, 9)).copy_from(&navigation);

        let row = |linear: usize| SAMPLED_STATE_DIM + linear;
        let altitude = row(LinearLayout::ALTITUDE);
        let vertical_velocity = row(LinearLayout::VERTICAL_VELOCITY);
        let baro_error = row(LinearLayout::BARO_ERROR);
        let accel_correction = row(LinearLayout::ACCEL_CORRECTION);
        transition[(baro_error, baro_error)] =
            markov_decay(dt, self.config.baro_error_time_constant_s);
        if self.baro.altitude.is_some() {
            let (k1, k2, k3) = self.baro_loop_gains();
            // Vertical velocity is positive up in ENU and positive down in NED.
            let up = if self.nominal.is_enu { 1.0 } else { -1.0 };
            for source in [altitude, baro_error] {
                transition[(altitude, source)] -= k1 * dt;
                transition[(vertical_velocity, source)] -= up * k2 * dt;
                transition[(accel_correction, source)] += k3 * dt;
            }
            transition[(vertical_velocity, accel_correction)] -= up * dt;
        }
        for channel in 0..layout.map_channels {
            let index = row(LinearLayout::variation(channel));
            transition[(index, index)] =
                markov_decay(dt, self.config.map_variation_time_constant_s[channel]);
        }
        transition
    }

    /// Diagonal process noise on the linear partition over one sample of `dt`: the paper's
    /// `diag(0, VRW, ARW, B, 0, T, 0)` (eqs. 20-22). The sampled pair's, if any, is added once
    /// per epoch in [`Self::flush_time_update`].
    ///
    /// Random walks enter as `(rate * sqrt(dt))^2`, so their variance grows linearly in
    /// elapsed time whatever the log's sample rate (#374); Gauss-Markov states through
    /// [`markov_process_variance`].
    fn process_noise(&self, dt: f64) -> DVector<f64> {
        let config = &self.config;
        let mut noise = DVector::<f64>::zeros(self.layout.dim());
        let velocity = config.velocity_process_noise_std_mps.powi(2) * dt;
        let tilt = config.attitude_process_noise_std_rad.powi(2) * dt;
        for axis in 0..3 {
            noise[LinearLayout::VELOCITY + axis] = velocity;
            noise[LinearLayout::TILT + axis] = tilt;
        }
        noise[LinearLayout::BARO_ERROR] = markov_process_variance(
            dt,
            config.baro_error_time_constant_s,
            config.baro_error_std_m,
        );
        for channel in 0..self.layout.map_channels {
            noise[LinearLayout::variation(channel)] = markov_process_variance(
                dt,
                config.map_variation_time_constant_s[channel],
                config.map_variation_std[channel],
            );
        }
        noise
    }

    /// Close the barometer loop on the nominal over one sample: the third-order aiding the
    /// paper applies "directly to the mechanization equations".
    fn apply_baro_loop(&mut self, dt: f64) {
        let Some(baro_altitude) = self.baro.altitude else {
            return;
        };
        let (k1, k2, k3) = self.baro_loop_gains();
        let difference = self.nominal.altitude - (baro_altitude - self.baro.baro_error);
        let up = if self.nominal.is_enu { 1.0 } else { -1.0 };
        self.nominal.altitude -= k1 * difference * dt;
        self.nominal.velocity_vertical -=
            up * k2.mul_add(difference, self.baro.accel_correction) * dt;
        self.baro.accel_correction += k3 * difference * dt;
    }

    /// Carry the nominal Gauss-Markov states forward: their expected value decays by the same
    /// factor their error does, so the nominal and the error model stay one process.
    fn decay_nominal_markov_states(&mut self, dt: f64) {
        self.baro.baro_error *= markov_decay(dt, self.config.baro_error_time_constant_s);
        for channel in 0..self.layout.map_channels {
            self.nominal_variation[channel] *=
                markov_decay(dt, self.config.map_variation_time_constant_s[channel]);
        }
    }

    /// Apply the time update accumulated since the last measurement epoch, if any.
    ///
    /// The marginalized particle filter's eqs. 30-35 over the whole epoch, with `A` the
    /// product of the per-sample transitions and the sampled pair's configured random walk --
    /// zero by default, as eq. 19 has it -- over the epoch's duration. The
    /// conditional covariance recursion depends only on the shared matrices, so it runs once
    /// for the cloud; each particle then draws its position from `N` and moves its linear
    /// states by what that draw says about them through `L`.
    ///
    /// # Errors
    /// [`StrapdownError::NonFinite`] if `N` is not finite.
    fn flush_time_update(&mut self) -> Result<(), StrapdownError> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        let linear_dim = self.layout.dim();
        let transition = &pending.transition;
        let f_nn = transition
            .view((0, 0), (SAMPLED_STATE_DIM, SAMPLED_STATE_DIM))
            .into_owned();
        let f_nl = transition
            .view((0, SAMPLED_STATE_DIM), (SAMPLED_STATE_DIM, linear_dim))
            .into_owned();
        let f_ln = transition
            .view((SAMPLED_STATE_DIM, 0), (linear_dim, SAMPLED_STATE_DIM))
            .into_owned();
        let f_ll = transition
            .view(
                (SAMPLED_STATE_DIM, SAMPLED_STATE_DIM),
                (linear_dim, linear_dim),
            )
            .into_owned();

        let (latitude_per_meter, longitude_per_meter) =
            horizontal_meters_to_radians(self.nominal.latitude.to_degrees(), self.nominal.altitude);
        let covariance = &self.linear_covariance;
        let sampled_noise = horizontal_std_to_radians(
            self.config.horizontal_process_noise_std_m[0],
            self.config.horizontal_process_noise_std_m[1],
            self.nominal.latitude,
            self.nominal.altitude,
        ) * pending.elapsed_s.sqrt();
        let n = symmetrize(
            &(&f_nl * covariance * f_nl.transpose()
                + DMatrix::from_diagonal(&DVector::from_vec(vec![
                    sampled_noise[0].powi(2),
                    sampled_noise[1].powi(2),
                ]))),
        );
        let (n_pseudo_inverse, n_root) =
            position_innovation_factors(&n, Vector2::new(latitude_per_meter, longitude_per_meter))?;
        let l = &f_ll * covariance * f_nl.transpose() * &n_pseudo_inverse;
        self.linear_covariance = symmetrize(
            &(&f_ll * covariance * f_ll.transpose()
                + DMatrix::from_diagonal(&pending.linear_noise)
                - &l * &n * l.transpose()),
        );

        // The cloud as two matrices, one column per particle, so the epoch is a handful of
        // matrix products rather than a matrix-vector product per particle. The draws are
        // taken particle by particle, so a seeded run draws the same numbers however the
        // products are arranged.
        let (sampled, linear) = self.stacked_errors();
        let normal = crate::normal_with_std(1.0);
        let mut draws = DMatrix::<f64>::zeros(SAMPLED_STATE_DIM, self.particles.len());
        for column in 0..self.particles.len() {
            for row in 0..SAMPLED_STATE_DIM {
                draws[(row, column)] = normal.sample(&mut self.rng);
            }
        }
        // Eq. 30. `z - A^n_l x^l` in eq. 35 is exactly the draw, `N^(1/2) e`: the particle's
        // new position less its deterministic part.
        let position_noise = &n_root * draws;
        let sampled_next = &f_nn * &sampled + &f_nl * &linear + &position_noise;
        let linear_next = &f_ll * &linear + &f_ln * &sampled + &l * &position_noise;
        self.write_back_errors(&sampled_next, &linear_next);
        Ok(())
    }

    /// The cloud's errors as two matrices, one column per particle: horizontal position and
    /// the linear partition.
    fn stacked_errors(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let count = self.particles.len();
        let mut sampled = DMatrix::<f64>::zeros(SAMPLED_STATE_DIM, count);
        let mut linear = DMatrix::<f64>::zeros(self.layout.dim(), count);
        for (column, particle) in self.particles.iter().enumerate() {
            sampled.set_column(column, &particle.position_error);
            linear.set_column(column, &particle.linear_state);
        }
        (sampled, linear)
    }

    /// Write stacked errors back into the particles, column by column.
    fn write_back_errors(&mut self, sampled: &DMatrix<f64>, linear: &DMatrix<f64>) {
        for (column, particle) in self.particles.iter_mut().enumerate() {
            particle.position_error = Vector2::new(sampled[(0, column)], sampled[(1, column)]);
            particle.linear_state.copy_from(&linear.column(column));
        }
    }

    /// The cloud's errors and their shared conditional covariance as they stand now, with any
    /// pending time update applied deterministically: the means through the accumulated
    /// transition, the covariance as `A C Aᵀ + Q` over the full error vector.
    ///
    /// This is what lets the estimate between measurement epochs describe the nominal's
    /// present rather than the last epoch's, without drawing: the draw a flush would take is
    /// the `A^n_l P (A^n_l)ᵀ` block of the returned covariance, not a spread of the means.
    /// Reporting as often as a caller likes therefore never moves the RNG, and a run's result
    /// does not depend on how often it was summarised.
    fn current_errors(&self) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let (sampled, linear) = self.stacked_errors();
        let full_dim = self.layout.full_dim();
        let mut conditional = DMatrix::<f64>::zeros(full_dim, full_dim);
        conditional
            .view_mut(
                (SAMPLED_STATE_DIM, SAMPLED_STATE_DIM),
                (self.layout.dim(), self.layout.dim()),
            )
            .copy_from(&self.linear_covariance);
        let Some(pending) = &self.pending else {
            return (sampled, linear, conditional);
        };
        let mut stacked = DMatrix::<f64>::zeros(full_dim, self.particles.len());
        stacked
            .view_mut((0, 0), (SAMPLED_STATE_DIM, self.particles.len()))
            .copy_from(&sampled);
        stacked
            .view_mut(
                (SAMPLED_STATE_DIM, 0),
                (self.layout.dim(), self.particles.len()),
            )
            .copy_from(&linear);
        let propagated = &pending.transition * stacked;
        let mut noise = DVector::<f64>::zeros(full_dim);
        noise
            .rows_mut(SAMPLED_STATE_DIM, self.layout.dim())
            .copy_from(&pending.linear_noise);
        let conditional = symmetrize(
            &(&pending.transition * conditional * pending.transition.transpose()
                + DMatrix::from_diagonal(&noise)),
        );
        (
            propagated.rows(0, SAMPLED_STATE_DIM).into_owned(),
            propagated
                .rows(SAMPLED_STATE_DIM, self.layout.dim())
                .into_owned(),
            conditional,
        )
    }

    /// One particle's reported state: the nominal plus its errors, in the Kalman filters'
    /// layout.
    ///
    /// Attitude is composed rather than added -- `exp([ε×]) C_nom`, the chart the tilt is
    /// propagated and injected in -- and read back as Euler angles on the principal branch.
    /// Adding the tilt to the Euler angles was the assembly half of #349. The IMU bias rows
    /// stay zero: this filter estimates none.
    fn reported_state(
        &self,
        position_error: &Vector2<f64>,
        linear_state: &DVector<f64>,
    ) -> DVector<f64> {
        let mut state = DVector::<f64>::zeros(self.layout.reported_dim());
        state[0] = self.nominal.latitude + position_error[0];
        state[1] = self.nominal.longitude + position_error[1];
        state[2] = self.nominal.altitude + linear_state[LinearLayout::ALTITUDE];
        state[3] = self.nominal.velocity_north + linear_state[LinearLayout::VELOCITY];
        state[4] = self.nominal.velocity_east + linear_state[LinearLayout::VELOCITY + 1];
        state[5] = self.nominal.velocity_vertical + linear_state[LinearLayout::VERTICAL_VELOCITY];
        let tilt = linear_state
            .fixed_rows::<3>(LinearLayout::TILT)
            .into_owned();
        let attitude = Rotation3::from_scaled_axis(tilt) * self.nominal.attitude;
        let (roll, pitch, yaw) = attitude.euler_angles();
        state[6] = roll;
        state[7] = pitch;
        state[8] = yaw;
        for channel in 0..self.layout.map_channels {
            state[REPORTED_BASE_DIM + channel] = self.nominal_variation[channel]
                + self.nominal_offset[channel]
                + linear_state[LinearLayout::variation(channel)]
                + linear_state[LinearLayout::offset(channel)];
        }
        state
    }

    /// Reported states of every particle, from stacked errors.
    fn reported_states(&self, sampled: &DMatrix<f64>, linear: &DMatrix<f64>) -> Vec<DVector<f64>> {
        (0..self.particles.len())
            .map(|column| {
                self.reported_state(
                    &Vector2::new(sampled[(0, column)], sampled[(1, column)]),
                    &linear.column(column).into_owned(),
                )
            })
            .collect()
    }

    /// The linear map `T` from the full error vector `[xⁿ; xˡ]` to the reported state, at the
    /// nominal.
    ///
    /// Identity from the horizontal pair, altitude and velocity onto their reported rows;
    /// `[1 1]` from each channel's `V` and `c` onto its total bias; and for attitude the Euler
    /// angles' response to a nav-frame tilt, `∂Φ/∂ε = E(Φ)⁻¹`. Near gimbal lock, where that
    /// inverse stops meaning anything, the identity stands in, the policy
    /// [`crate::linearize::bias_coupling_blocks`] follows. The barometer states report nowhere.
    fn error_to_reported(&self) -> DMatrix<f64> {
        let mut map = DMatrix::<f64>::zeros(self.layout.reported_dim(), self.layout.full_dim());
        let column = |linear: usize| SAMPLED_STATE_DIM + linear;
        map[(0, 0)] = 1.0;
        map[(1, 1)] = 1.0;
        map[(2, column(LinearLayout::ALTITUDE))] = 1.0;
        for axis in 0..3 {
            map[(3 + axis, column(LinearLayout::VELOCITY + axis))] = 1.0;
        }
        let (roll, pitch, yaw) = self.nominal.attitude.euler_angles();
        let euler_from_tilt =
            euler_rate_matrix_inverse(roll, pitch, yaw).unwrap_or_else(Matrix3::identity);
        map.view_mut(
            (ATTITUDE_STATE_INDICES[0], column(LinearLayout::TILT)),
            (3, 3),
        )
        .copy_from(&euler_from_tilt);
        for channel in 0..self.layout.map_channels {
            map[(
                REPORTED_BASE_DIM + channel,
                column(LinearLayout::variation(channel)),
            )] = 1.0;
            map[(
                REPORTED_BASE_DIM + channel,
                column(LinearLayout::offset(channel)),
            )] = 1.0;
        }
        map
    }

    /// Weighted mean and covariance of the reported state, now.
    ///
    /// # Attitude is averaged on the circle
    ///
    /// The three attitude angles live on the circle, where a linear mean is wrong in kind: two
    /// particles at +179 deg and -179 deg average to 0 deg, the opposite heading, and the
    /// unwrapped sum can leave `[-pi, pi]` entirely, which the reported solution must not do
    /// (#314). So the attitude channels take the mean *direction* of the cloud, and the
    /// covariance differences their residuals with [`crate::wrap_to_pi`].
    ///
    /// # The covariance is the law of total covariance
    ///
    /// Everything but horizontal position is Rao-Blackwellised: each particle holds a
    /// conditional mean *and* the shared conditional covariance. The marginal covariance is
    /// the weighted spread of the particles' states plus `T C Tᵀ`, where `C` is the
    /// conditional covariance of the full error vector and `T` the linear map from it onto the
    /// reported state: the identity for position and velocity, `∂Φ/∂ε = E(Φ)⁻¹` for attitude,
    /// and `[1 1]` from each channel's `V` and `c` onto its total.
    ///
    /// # Between measurement epochs
    ///
    /// The particles are propagated once per epoch, but the estimate describes the present:
    /// any accumulated time update is applied deterministically first (see
    /// `current_errors`), so the covariance grows between fixes as the filter's own model says
    /// it should, without consuming a draw.
    ///
    /// The six IMU bias rows are zero, mean and covariance: the filter carries no bias states.
    pub fn estimate(&self) -> (DVector<f64>, DMatrix<f64>) {
        let (sampled, linear, conditional) = self.current_errors();
        let states = self.reported_states(&sampled, &linear);
        let (mut mean, spread) = self.weighted_moments(&states, self.layout.reported_dim());
        let map = self.error_to_reported();
        let mut covariance = symmetrize(&(spread + &map * conditional * map.transpose()));
        // Assigned rather than computed: a product that happened to come out `-0.0` would be
        // refused by the health monitor.
        for row in REPORTED_IMU_BIAS..REPORTED_BASE_DIM {
            mean[row] = 0.0;
            for column in 0..covariance.ncols() {
                covariance[(row, column)] = 0.0;
                covariance[(column, row)] = 0.0;
            }
        }
        (mean, covariance)
    }

    /// Weighted mean of the reported state alone, attitude on the circle: the point a
    /// measurement is linearised at.
    fn reported_mean(&self) -> DVector<f64> {
        let (sampled, linear, _) = self.current_errors();
        let states = self.reported_states(&sampled, &linear);
        self.weighted_moments(&states, self.layout.reported_dim()).0
    }

    /// Weighted mean and spread of an assembled cloud, attitude handled on the circle.
    ///
    /// `dim` is the caller's own width rather than `states[0].len()` so that a cloud with no
    /// particles still reports the shape its caller promised.
    fn weighted_moments(
        &self,
        states: &[DVector<f64>],
        dim: usize,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let mut mean = DVector::<f64>::zeros(dim);
        for (state, particle) in states.iter().zip(&self.particles) {
            mean += state * particle.weight;
        }
        for index in ATTITUDE_STATE_INDICES {
            mean[index] = circular_mean(
                states
                    .iter()
                    .zip(&self.particles)
                    .map(|(state, particle)| (&state[index], particle.weight)),
            );
        }

        let mut spread = DMatrix::<f64>::zeros(dim, dim);
        for (state, particle) in states.iter().zip(&self.particles) {
            let mut diff = state - &mean;
            for index in ATTITUDE_STATE_INDICES {
                diff[index] = crate::wrap_to_pi(diff[index]);
            }
            spread += particle.weight * (&diff * diff.transpose());
        }
        (mean, symmetrize(&spread))
    }

    /// Score a measurement against the particle cloud summarised as a Gaussian.
    ///
    /// Takes `&mut self` because a rejection is not inert: the cloud is spread out by
    /// [`GateRecovery::rejection_inflation`] on the way out, which is this filter's form of
    /// the covariance inflation the Kalman filters apply, and without it one rejection would
    /// make every subsequent fix disagree by more (#340).
    ///
    /// Always computes the NIS, gate or no gate, because
    /// [`sim::health::HealthMonitor`](crate::sim::health::HealthMonitor) consumes it to catch a
    /// filter that has diverged. The summary is [`Self::estimate`], the same layout the update
    /// hands the measurement, and its covariance carries the conditional half, so `S` includes
    /// the map-bias and altitude uncertainty the weights' marginal likelihood does.
    ///
    /// # Errors
    /// Whatever the measurement model returns when evaluated at the ensemble mean, or a
    /// singular innovation covariance from
    /// [`normalized_innovation_squared`](crate::gating::normalized_innovation_squared).
    fn evaluate_ensemble_gate<M: MeasurementModel + ?Sized>(
        &mut self,
        measurement: &M,
    ) -> Result<UpdateOutcome, StrapdownError> {
        let (mean, covariance) = self.estimate();
        // Jacobian first: a geophysical model off the edge of its map reports that here,
        // whereas `get_expected_measurement` would quietly return NaN.
        let h = measurement.get_jacobian(&mean)?;
        let z_hat = measurement.get_expected_measurement(&mean);
        let mut innovation = measurement.get_measurement(&mean)? - z_hat;
        measurement.wrap_residual(&mut innovation);

        // The navigation models return a fixed nine columns; pad to the summary's width.
        let h = expand_measurement_jacobian(h, covariance.ncols())?;
        let s = &h * &covariance * h.transpose() + measurement.get_noise();
        let dof = innovation.len();
        let nis = normalized_innovation_squared(&innovation, &s)?;
        let decision = self.gate_policy.decide(nis, dof, "RBPF");
        if !decision.outcome.accepted {
            self.inflate_particle_spread(decision.covariance_inflation);
        }
        Ok(decision.outcome)
    }

    /// Multiply the ensemble covariance by `factor`: the cloud's spread and the shared
    /// conditional covariance alike.
    ///
    /// The Kalman filters recover from a gated-out measurement by scaling `P`. Here the
    /// reported covariance is the particles' spread plus `T C Tᵀ`, so replacing every
    /// particle's errors by `x̄ + sqrt(f)(x_i - x̄)` scales the first half by exactly `f` and
    /// multiplying the shared `P` by `f` scales the second, leaving the mean, the weights and
    /// the particle identities alone and drawing nothing from the RNG (multiplicative
    /// inflation, Anderson & Anderson 1999). It is applied right after a time update, when
    /// nothing is pending. Unlike the Kalman filters it stretches every direction rather than
    /// the observed subspace, so a rejection on one sensor widens every other sensor's gain.
    ///
    /// A `factor` at or below 1, or one that is not finite, is a no-op.
    fn inflate_particle_spread(&mut self, factor: f64) {
        if !factor.is_finite() || factor <= 1.0 || self.particles.is_empty() {
            return;
        }
        let scale = factor.sqrt();
        let (mean_position, mean_linear) = self.mean_errors();
        for particle in &mut self.particles {
            particle.position_error =
                mean_position + scale * (particle.position_error - mean_position);
            particle.linear_state = &mean_linear + scale * (&particle.linear_state - &mean_linear);
        }
        self.linear_covariance *= factor;
    }

    /// Weighted means of the particles' horizontal and linear errors.
    fn mean_errors(&self) -> (Vector2<f64>, DVector<f64>) {
        let mut position = Vector2::zeros();
        let mut linear = DVector::<f64>::zeros(self.layout.dim());
        for particle in &self.particles {
            position += particle.weight * particle.position_error;
            linear += particle.weight * &particle.linear_state;
        }
        (position, linear)
    }

    /// Compute the effective sample size.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.particles.iter().map(|p| p.weight.powi(2)).sum();
        if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 }
    }

    /// The marginalized particle filter's measurement update (Algorithm 1, steps 5-7), for
    /// any measurement. Runs on a cloud whose time update has been applied.
    ///
    /// A measurement is `y = h(xⁿ) + C xˡ + w` (eq. 15): nonlinear in horizontal position,
    /// linear -- or linearised -- in everything else. `C` is the measurement's Jacobian at the
    /// ensemble mean carried onto the linear partition through `T`; its latitude and longitude
    /// columns are dropped, since the particles carry those. Then, with `S = C P Cᵀ + R`:
    ///
    /// 1. each particle's residual is `r_i = z - h(state_i)`, where `state_i` holds its own
    ///    position *and* its own conditional means, so `r_i` is `y - h(xⁿ_i) - C xˡ_i`;
    /// 2. its weight is multiplied by `N(r_i; 0, S)`, the likelihood with the linear states
    ///    integrated out (eq. 24);
    /// 3. its linear states take the Kalman step `K r_i` with the shared gain `K = P Cᵀ S⁻¹`,
    ///    and the shared `P` takes the Joseph-form update (eqs. 26-29).
    ///
    /// For a map this is exactly the paper's update: `C` selects `V + c`. A GNSS fix weights on
    /// latitude and longitude and corrects altitude and, with velocity, the velocity states;
    /// a heading corrects tilt through `∂yaw/∂ε`.
    ///
    /// `z` is evaluated **once, at the ensemble mean**, as the Kalman filters evaluate it. Some
    /// models fold state-dependent terms into the measurement itself -- the Eötvös correction
    /// into gravity, levelling into a magnetometer heading -- and a per-particle `z` would put
    /// a residual term in `r_i` that neither `C` nor `S` models.
    ///
    /// A particle whose residual is not finite -- one that has wandered off the map -- gets zero
    /// weight and no Kalman step: `K * NaN` is NaN, and recentring would carry it into the
    /// nominal whatever the weight. If no particle scores a usable residual the fix carried no
    /// information, and neither the estimates nor `P` move.
    ///
    /// # Errors
    /// Propagated from the measurement model -- chiefly a geophysical model whose ensemble
    /// mean has left its map -- and [`StrapdownError::DimensionMismatch`] if the measurement
    /// observes IMU bias states, which this filter does not carry (ZARU). Correcting nothing
    /// there would be indistinguishable, in a log, from correcting something.
    fn mpf_update<M: MeasurementModel + ?Sized>(
        &mut self,
        measurement: &M,
    ) -> Result<(), StrapdownError> {
        if self.particles.is_empty() {
            return Ok(());
        }
        let mean = self.reported_mean();
        let jacobian = expand_measurement_jacobian(
            measurement.get_jacobian(&mean)?,
            self.layout.reported_dim(),
        )?;
        refuse_unmodelled_states(&jacobian)?;
        let z = measurement.get_measurement(&mean)?;
        let map = self.error_to_reported();
        let observation = &jacobian
            * map
                .view(
                    (0, SAMPLED_STATE_DIM),
                    (self.layout.reported_dim(), self.layout.dim()),
                )
                .into_owned();
        let noise = measurement.get_noise();
        let innovation_covariance = symmetrize(
            &(&observation * &self.linear_covariance * observation.transpose() + &noise),
        );
        let identity =
            DMatrix::identity(innovation_covariance.nrows(), innovation_covariance.ncols());
        let innovation_inverse = robust_spd_solve(&innovation_covariance, &identity)?;
        let gain = &self.linear_covariance * observation.transpose() * &innovation_inverse;

        let residuals: Vec<DVector<f64>> = self
            .particles
            .iter()
            .map(|particle| {
                let state = self.reported_state(&particle.position_error, &particle.linear_state);
                let mut residual = &z - measurement.get_expected_measurement(&state);
                measurement.wrap_residual(&mut residual);
                residual
            })
            .collect();
        self.reweight(&residuals, &innovation_inverse);

        let is_usable = |residual: &DVector<f64>| residual.iter().all(|value| value.is_finite());
        if residuals.iter().any(is_usable) {
            for (particle, residual) in self.particles.iter_mut().zip(&residuals) {
                if is_usable(residual) {
                    particle.linear_state += &gain * residual;
                }
            }
            self.linear_covariance =
                joseph_update(&self.linear_covariance, &gain, &observation, &noise);
        }

        self.recenter();
        self.maybe_resample();
        Ok(())
    }

    /// Multiply each particle's weight by `N(r_i; 0, S)` and renormalise.
    ///
    /// In the log domain with the largest term subtracted before exponentiating, so a
    /// confident fix does not underflow every weight to zero. If every weight still comes out
    /// zero -- every residual non-finite -- the cloud is left uniform rather than undefined.
    fn reweight(&mut self, residuals: &[DVector<f64>], innovation_inverse: &DMatrix<f64>) {
        let log_weights: Vec<f64> = self
            .particles
            .iter()
            .zip(residuals)
            .map(|(particle, residual)| {
                particle.weight.ln() + gaussian_log_likelihood(residual, innovation_inverse)
            })
            .collect();
        let max_log = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for (particle, log_weight) in self.particles.iter_mut().zip(&log_weights) {
            particle.weight = (log_weight - max_log).exp();
            if !particle.weight.is_finite() {
                particle.weight = 0.0;
            }
            sum += particle.weight;
        }
        if sum > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= sum;
            }
        } else {
            let uniform = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.weight = uniform;
            }
        }
    }

    /// Fold the cloud's weighted-mean error into the nominal and leave the cloud zero-mean:
    /// closed-loop feedback, departure 1 in the module docs.
    ///
    /// Position, altitude, velocity, the barometer loop's two states and the map states are
    /// additive -- `b̂` and `â` take `δh_a` and `δâ`, so the loop runs on the corrected
    /// values from here on. Tilt is composed on the left, `C_nom <- exp([ε̄×]) C_nom`, the chart
    /// it is propagated in; because the tilt reference has moved, the conditional covariance's
    /// tilt block is transported by the left Jacobian `J_l(ε̄)`, the second-order correction
    /// the ESKF applies for its own right-hand error (#398).
    ///
    /// A non-finite mean -- which only a non-finite particle can produce -- is left in the
    /// cloud rather than written into the nominal, so the health monitor sees the cloud fail
    /// rather than a nominal silently turned to NaN.
    fn recenter(&mut self) {
        let (mean_position, mean_linear) = self.mean_errors();
        if !(mean_position.iter().all(|v| v.is_finite())
            && mean_linear.iter().all(|v| v.is_finite()))
        {
            return;
        }
        self.nominal.latitude += mean_position[0];
        self.nominal.longitude += mean_position[1];
        self.nominal.altitude += mean_linear[LinearLayout::ALTITUDE];
        self.nominal.velocity_north += mean_linear[LinearLayout::VELOCITY];
        self.nominal.velocity_east += mean_linear[LinearLayout::VELOCITY + 1];
        self.nominal.velocity_vertical += mean_linear[LinearLayout::VERTICAL_VELOCITY];
        let tilt = mean_linear.fixed_rows::<3>(LinearLayout::TILT).into_owned();
        self.nominal.attitude = Rotation3::from_scaled_axis(tilt) * self.nominal.attitude;
        self.baro.baro_error += mean_linear[LinearLayout::BARO_ERROR];
        self.baro.accel_correction += mean_linear[LinearLayout::ACCEL_CORRECTION];
        for channel in 0..self.layout.map_channels {
            self.nominal_variation[channel] += mean_linear[LinearLayout::variation(channel)];
            self.nominal_offset[channel] += mean_linear[LinearLayout::offset(channel)];
        }

        let reset = attitude_reset_jacobian(&(-tilt));
        if reset != Matrix3::identity() {
            let dim = self.layout.dim();
            let mut transport = DMatrix::<f64>::identity(dim, dim);
            transport
                .view_mut((LinearLayout::TILT, LinearLayout::TILT), (3, 3))
                .copy_from(&reset);
            self.linear_covariance =
                symmetrize(&(&transport * &self.linear_covariance * transport.transpose()));
        }

        for particle in &mut self.particles {
            particle.position_error -= mean_position;
            particle.linear_state -= &mean_linear;
        }
    }

    fn maybe_resample(&mut self) {
        let n_eff = self.effective_sample_size();
        let threshold = self.config.effective_sample_threshold * self.particles.len() as f64;
        if n_eff >= threshold {
            return;
        }

        // Measured before the resample: afterwards the extent along a collapsed axis is the
        // zero this is here to repair. See `RbpfConfig::roughening_factor`.
        let extent = self.position_extent();

        let weights: Vec<f64> = self.particles.iter().map(|p| p.weight).collect();
        let indices = match self.config.resampling_strategy {
            ParticleResamplingStrategy::Multinomial => {
                multinomial_resample(&weights, self.particles.len(), &mut self.rng)
            }
            ParticleResamplingStrategy::Systematic => {
                systematic_resample(&weights, self.particles.len(), &mut self.rng)
            }
            ParticleResamplingStrategy::Stratified => {
                stratified_resample(&weights, self.particles.len(), &mut self.rng)
            }
            ParticleResamplingStrategy::Residual => {
                residual_resample(&weights, self.particles.len(), &mut self.rng)
            }
        };

        let uniform = 1.0 / self.particles.len() as f64;
        self.particles = indices
            .into_iter()
            .map(|index| RbpfParticle {
                weight: uniform,
                ..self.particles[index].clone()
            })
            .collect();

        self.roughen(&extent);
    }

    /// Per-axis extent of the horizontal cloud: the spread the roughening jitter is scaled by.
    ///
    /// Max-minus-min rather than a standard deviation, following Gordon, Salmond & Smith, and
    /// because it is the quantity that goes to exactly zero when every particle is a copy of
    /// one ancestor -- which is the state being detected.
    fn position_extent(&self) -> Vector2<f64> {
        let mut lo = Vector2::repeat(f64::INFINITY);
        let mut hi = Vector2::repeat(f64::NEG_INFINITY);
        for particle in &self.particles {
            for axis in 0..SAMPLED_STATE_DIM {
                lo[axis] = lo[axis].min(particle.position_error[axis]);
                hi[axis] = hi[axis].max(particle.position_error[axis]);
            }
        }
        let mut extent = Vector2::zeros();
        for axis in 0..SAMPLED_STATE_DIM {
            let span = hi[axis] - lo[axis];
            // A non-finite particle would poison every axis; leave the extent at zero and
            // let the health monitor report the real problem rather than jittering by NaN.
            extent[axis] = if span.is_finite() { span } else { 0.0 };
        }
        extent
    }

    /// Jitter the resampled cloud so that duplicated particles stop being identical.
    ///
    /// `sigma = K * extent * N^(-1/d)` per axis, drawn independently, with `d = 2`, the
    /// sampled partition. The linear partition is carried by the conditional covariance rather
    /// than by cloud spread, so it is neither counted nor jittered. A zero `extent` on an axis
    /// leaves that axis alone: inventing a width there would be fabricating uncertainty.
    fn roughen(&mut self, extent: &Vector2<f64>) {
        let factor = self.config.roughening_factor;
        if factor <= 0.0 || self.particles.is_empty() {
            return;
        }
        let scale = (self.particles.len() as f64).powf(-1.0 / SAMPLED_STATE_DIM as f64);
        let normal = crate::normal_with_std(1.0);
        for particle in &mut self.particles {
            for axis in 0..SAMPLED_STATE_DIM {
                let sigma = factor * extent[axis] * scale;
                if sigma > 0.0 {
                    particle.position_error[axis] += sigma * normal.sample(&mut self.rng);
                }
            }
        }
    }
}

/// Refuse a measurement whose Jacobian observes the IMU biases, which this filter does not
/// carry.
///
/// # Errors
/// [`StrapdownError::DimensionMismatch`] naming the bias states.
fn refuse_unmodelled_states(jacobian: &DMatrix<f64>) -> Result<(), StrapdownError> {
    let observes_biases = (REPORTED_IMU_BIAS..REPORTED_BASE_DIM)
        .any(|column| jacobian.column(column).iter().any(|entry| *entry != 0.0));
    if observes_biases {
        return Err(StrapdownError::DimensionMismatch {
            what: "IMU bias states observed by a measurement, which the RBPF does not carry",
            expected: REPORTED_BASE_DIM,
            got: REPORTED_IMU_BIAS,
        });
    }
    Ok(())
}

/// The initial conditional covariance: diagonal, from the configured priors.
///
/// A map channel's total prior is split between its temporal variation, which starts from its
/// stationary distribution, and its constant offset, which takes the remainder. The barometer
/// error starts from its own stationary distribution.
fn initial_linear_covariance(config: &RbpfConfig, layout: LinearLayout) -> DMatrix<f64> {
    let mut covariance = DMatrix::<f64>::zeros(layout.dim(), layout.dim());
    covariance[(LinearLayout::ALTITUDE, LinearLayout::ALTITUDE)] =
        config.position_init_std_m[2].powi(2);
    for axis in 0..3 {
        covariance[(LinearLayout::VELOCITY + axis, LinearLayout::VELOCITY + axis)] =
            config.velocity_init_std_mps.powi(2);
        covariance[(LinearLayout::TILT + axis, LinearLayout::TILT + axis)] =
            config.attitude_init_std_rad.powi(2);
    }
    covariance[(LinearLayout::BARO_ERROR, LinearLayout::BARO_ERROR)] =
        config.baro_error_std_m.powi(2);
    covariance[(
        LinearLayout::ACCEL_CORRECTION,
        LinearLayout::ACCEL_CORRECTION,
    )] = config.vertical_accel_error_init_std_mps2.powi(2);
    for channel in 0..layout.map_channels {
        let variation_variance = config.map_variation_std[channel].powi(2);
        let total_variance = config.map_bias_init_std[channel].powi(2);
        let (variation, offset) = (
            LinearLayout::variation(channel),
            LinearLayout::offset(channel),
        );
        covariance[(variation, variation)] = variation_variance;
        covariance[(offset, offset)] = (total_variance - variation_variance).max(0.0);
    }
    covariance
}

/// The two factors of the position innovation covariance `N` the time update needs: its
/// pseudo-inverse, for `L`, and a square root, for the particles' draw.
///
/// `N` is in rad² and routinely tiny -- with the paper's zero `Qⁿ` it is
/// `A^n_l P (A^n_l)ᵀ` -- and can be singular outright, so it is scaled to metres with the radii
/// of curvature, decomposed once, and its eigenvalues clamped at zero. Directions below a
/// relative tolerance of the largest eigenvalue carry no information and are left out of the
/// inverse, so a degenerate `N` gives `L = 0` rather than the identity an inversion fallback
/// would substitute; the root is formed from the clamped eigenvalues, so it reproduces `N`
/// exactly where `N` is positive semi-definite.
///
/// # Errors
/// [`StrapdownError::NonFinite`] if `N` is not finite, or if its eigendecomposition does not
/// converge.
fn position_innovation_factors(
    n: &DMatrix<f64>,
    radians_per_meter: Vector2<f64>,
) -> Result<(DMatrix<f64>, DMatrix<f64>), StrapdownError> {
    let not_finite = || StrapdownError::NonFinite {
        what: "RBPF position innovation covariance",
    };
    if !n.iter().all(|value| value.is_finite()) {
        return Err(not_finite());
    }
    let meters_per_radian = Matrix2::from_diagonal(&radians_per_meter.map(f64::recip));
    let radians = Matrix2::from_diagonal(&radians_per_meter);
    let n_radians = Matrix2::new(n[(0, 0)], n[(0, 1)], n[(1, 0)], n[(1, 1)]);
    let n_meters = meters_per_radian * n_radians * meters_per_radian;
    let eigen = SymmetricEigen::try_new(n_meters, f64::EPSILON, POSITION_INNOVATION_MAX_ITERATIONS)
        .ok_or_else(not_finite)?;
    let largest = eigen.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = largest * POSITION_INNOVATION_RELATIVE_TOLERANCE;
    let mut inverse_meters = Matrix2::zeros();
    let mut root_meters = Matrix2::zeros();
    for (eigenvalue, eigenvector) in eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
    {
        let eigenvalue = eigenvalue.max(0.0);
        let projector = eigenvector * eigenvector.transpose();
        if eigenvalue > tolerance {
            inverse_meters += projector / eigenvalue;
        }
        root_meters += projector * eigenvalue.sqrt();
    }
    // `N = D⁻¹ N_m D⁻¹` with `D` metres per radian, so `N⁺ = D N_m⁺ D` and `D⁻¹ root_m` is a
    // root of `N`.
    let inverse = meters_per_radian * inverse_meters * meters_per_radian;
    let root = radians * root_meters;
    Ok((
        DMatrix::from_column_slice(2, 2, inverse.as_slice()),
        DMatrix::from_column_slice(2, 2, root.as_slice()),
    ))
}

/// Joseph-form covariance update `(I - K C) P (I - K C)ᵀ + K R Kᵀ`, symmetrised.
///
/// Equal to the paper's `P - K S Kᵀ` (eq. 27) for the optimal gain, and positive semi-definite
/// by construction for any gain, which the subtraction is not in floating point.
fn joseph_update(
    covariance: &DMatrix<f64>,
    gain: &DMatrix<f64>,
    observation: &DMatrix<f64>,
    noise: &DMatrix<f64>,
) -> DMatrix<f64> {
    let complement =
        DMatrix::<f64>::identity(covariance.nrows(), covariance.ncols()) - gain * observation;
    symmetrize(
        &(&complement * covariance * complement.transpose() + gain * noise * gain.transpose()),
    )
}

impl NavigationFilter for RaoBlackwellizedParticleFilter {
    /// Predict step: mechanize the nominal, with barometer aiding, and accumulate the time
    /// update the particles take at the next measurement epoch.
    ///
    /// # Arguments
    ///
    /// * `control_input` - an [`ImuSample`] (integrated $\Delta v$ / $\Delta\theta$) or,
    ///   for callers still holding instantaneous rates, an
    ///   [`IMUData`](crate::IMUData).
    /// * `dt` - Time step in seconds. When `control_input` is an [`ImuSample`] this must
    ///   agree with the sample's own `dt`; see the Errors section.
    ///
    /// # Errors
    /// * [`StrapdownError::UnsupportedInput`] if `control_input` is neither inertial form.
    /// * [`StrapdownError::InconsistentTimestep`] if an [`ImuSample`]'s `dt` disagrees with
    ///   the `dt` argument.
    /// * [`StrapdownError::OutOfRange`] or [`StrapdownError::NonFinite`] propagated from
    ///   [`mechanize`].
    fn predict(&mut self, control_input: &dyn InputModel, dt: f64) -> Result<(), StrapdownError> {
        let sample = imu_sample_from_input(control_input, "RaoBlackwellizedParticleFilter", dt)?;
        self.predict_sample(&sample)
    }

    /// Update step: aid the barometer loop, or take a measurement epoch.
    ///
    /// A barometric altitude is not a filter measurement here: it is the input of the
    /// mechanization's aiding loop, as in the paper, and is stored for the loop to use from
    /// the next sample on. It is reported accepted with a zero NIS.
    ///
    /// Anything else is a measurement epoch: the accumulated time update is applied, then the
    /// ensemble gate, then the marginalized update -- reweighting and the shared Kalman step.
    ///
    /// # Innovation gating
    ///
    /// The NIS is evaluated against the *ensemble* mean and covariance, not against any
    /// single particle. This is an approximation the Kalman filters do not need to make -- a
    /// multi-modal cloud has no meaningful single innovation -- so treat a gated RBPF as a
    /// coarse outlier screen rather than the consistency test it is for the EKF/UKF/ESKF.
    ///
    /// # Errors
    /// Propagates measurement failures — chiefly a geophysical model whose particle has
    /// drifted off the loaded map. Callers should consult
    /// [`StrapdownError::is_recoverable`] and skip the measurement rather than abort.
    fn update(
        &mut self,
        measurement: &dyn MeasurementModel,
    ) -> Result<UpdateOutcome, StrapdownError> {
        if measurement
            .as_any()
            .downcast_ref::<RelativeAltitudeMeasurement>()
            .is_some()
        {
            // The reading is state-independent; the vector only has to be wide enough for the
            // model's own bias check, and the event stream declares no bias for this filter.
            let altitude =
                measurement.get_measurement(&DVector::zeros(self.layout.reported_dim()))?[0];
            if altitude.is_finite() {
                self.baro.altitude = Some(altitude);
            }
            return Ok(UpdateOutcome::accepted(0.0, 1));
        }
        self.flush_time_update()?;
        let outcome = self.evaluate_ensemble_gate(measurement)?;
        if !outcome.accepted {
            return Ok(outcome);
        }
        self.mpf_update(measurement)?;
        Ok(outcome)
    }

    fn set_innovation_gate(&mut self, gate: Option<InnovationGate>) -> bool {
        self.gate_policy.set_gate(gate);
        true
    }

    fn set_gate_recovery(&mut self, recovery: GateRecovery) -> bool {
        self.gate_policy.set_recovery(recovery);
        true
    }

    /// The weighted mean of the reported state: fifteen navigation and (zero) IMU-bias states,
    /// then one total bias per map channel.
    ///
    /// Computed from the particle cloud on each call; [`Self::estimate`] returns the mean and
    /// covariance together and is cheaper when both are wanted.
    fn get_estimate(&self) -> DVector<f64> {
        self.estimate().0
    }

    /// The covariance of the reported state. See [`Self::get_estimate`].
    fn get_certainty(&self) -> DMatrix<f64> {
        self.estimate().1
    }
}

/// `-r^T S^-1 r / 2`, given `S^-1`; minus infinity for a non-finite residual.
///
/// The normalising constant is omitted: every particle is scored against the same `S`, so it
/// cancels in the weights.
fn gaussian_log_likelihood(residual: &DVector<f64>, innovation_inverse: &DMatrix<f64>) -> f64 {
    if residual.iter().any(|v| !v.is_finite()) {
        return f64::NEG_INFINITY;
    }
    -0.5 * (residual.transpose() * innovation_inverse * residual)[(0, 0)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurements::{
        GPSPositionAndVelocityMeasurement, GPSPositionMeasurement, MagnetometerYawMeasurement,
        ZaruMeasurement,
    };
    use crate::{IMUData, earth, generate_scenario_data};
    use assert_approx_eq::assert_approx_eq;

    /// Latitude used by the #331 spread tests. Well away from the equator, where the
    /// `cos(latitude)` the longitude conversion needs is 0.5 -- so the defect the tests
    /// guard against halves the east spread rather than nudging it.
    const SPREAD_TEST_LATITUDE_DEG: f64 = 60.0;
    const SPREAD_TEST_LONGITUDE_DEG: f64 = -105.0;
    const SPREAD_TEST_ALTITUDE_M: f64 = 1000.0;

    /// Root-mean-square of a sample, which is its standard deviation when its mean is zero.
    fn root_mean_square(values: impl Iterator<Item = f64>) -> f64 {
        let (sum_of_squares, count) = values.fold((0.0, 0_usize), |(sum, n), value| {
            (value.mul_add(value, sum), n + 1)
        });
        (sum_of_squares / count as f64).sqrt()
    }

    /// A particle's horizontal error as a ground displacement in metres, measured
    /// independently of the conversion under test: great-circle distance on a sphere rather
    /// than the WGS84 radii the filter divides by, which is why the assertions carry 3%.
    fn position_error_ground_meters(
        nominal: &StrapdownState,
        position_error: &Vector2<f64>,
    ) -> Vector2<f64> {
        let north_m = earth::haversine_distance(
            nominal.latitude,
            nominal.longitude,
            nominal.latitude + position_error[0],
            nominal.longitude,
        );
        let east_m = earth::haversine_distance(
            nominal.latitude,
            nominal.longitude,
            nominal.latitude,
            nominal.longitude + position_error[1],
        );
        Vector2::new(north_m, east_m)
    }

    fn spread_test_nominal_state() -> StrapdownState {
        StrapdownState {
            latitude: SPREAD_TEST_LATITUDE_DEG.to_radians(),
            longitude: SPREAD_TEST_LONGITUDE_DEG.to_radians(),
            altitude: SPREAD_TEST_ALTITUDE_M,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        }
    }

    /// A level, stationary ENU accelerometer reading at the spread-test position.
    fn stationary_imu() -> IMUData {
        IMUData {
            accel: Vector3::new(
                0.0,
                0.0,
                earth::gravity(&SPREAD_TEST_LATITUDE_DEG, &SPREAD_TEST_ALTITUDE_M),
            ),
            gyro: Vector3::zeros(),
        }
    }

    /// A configuration carrying `channels` map channels with small, distinct priors.
    fn map_config(channels: usize) -> RbpfConfig {
        RbpfConfig {
            num_particles: 256,
            map_bias_channels: channels,
            map_bias_initial: vec![0.0; channels],
            map_bias_init_std: vec![0.5; channels],
            map_variation_std: vec![0.2; channels],
            map_variation_time_constant_s: vec![DEFAULT_MAP_VARIATION_TIME_CONSTANT_S; channels],
            ..RbpfConfig::default()
        }
    }

    fn filter_with_map_channels(channels: usize, seed: u64) -> RaoBlackwellizedParticleFilter {
        RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                seed,
                ..map_config(channels)
            },
        )
        .unwrap()
    }

    /// The paper's partition: two sampled states, the nine linear navigation and barometer
    /// states, then `(V, c)` per map channel -- eleven linear states for the paper's one
    /// channel -- and a report in the Kalman filters' fifteen-state layout.
    #[test]
    fn rbpf_partition_is_the_papers() {
        for (channels, linear, reported) in [(0, 9, 15), (1, 11, 16), (2, 13, 17)] {
            let rbpf = RaoBlackwellizedParticleFilter::new(
                spread_test_nominal_state(),
                RbpfConfig {
                    num_particles: 8,
                    ..map_config(channels)
                },
            )
            .unwrap();
            assert_eq!(rbpf.linear_state_dim(), linear, "{channels} channels");
            assert_eq!(rbpf.particles()[0].linear_state.len(), linear);
            assert_eq!(rbpf.linear_covariance().shape(), (linear, linear));
            let (mean, covariance) = rbpf.estimate();
            assert_eq!(mean.len(), reported);
            assert_eq!(covariance.shape(), (reported, reported));
            // No IMU bias states: the six reported rows are zero, mean and covariance.
            for row in REPORTED_IMU_BIAS..REPORTED_BASE_DIM {
                assert!(mean[row] == 0.0 && !mean[row].is_sign_negative());
                assert!(covariance.row(row).iter().all(|v| *v == 0.0));
            }
        }
    }

    /// The initial cloud spreads the requested metres east as well as north (#331), and the
    /// altitude prior is conditional rather than spread.
    #[test]
    fn rbpf_initial_cloud_spreads_the_requested_metres_in_both_directions() {
        const REQUESTED_HORIZONTAL_STD_M: f64 = 10.0;
        const REQUESTED_VERTICAL_STD_M: f64 = 5.0;

        let nominal = spread_test_nominal_state();
        let rbpf = RaoBlackwellizedParticleFilter::new(
            nominal,
            RbpfConfig {
                num_particles: 50_000,
                position_init_std_m: Vector3::new(
                    REQUESTED_HORIZONTAL_STD_M,
                    REQUESTED_HORIZONTAL_STD_M,
                    REQUESTED_VERTICAL_STD_M,
                ),
                seed: 331,
                ..RbpfConfig::default()
            },
        )
        .unwrap();

        let ground: Vec<Vector2<f64>> = rbpf
            .particles
            .iter()
            .map(|particle| position_error_ground_meters(&nominal, &particle.position_error))
            .collect();
        for (axis, name) in [(0, "north"), (1, "east")] {
            let observed = root_mean_square(ground.iter().map(|v| v[axis]));
            assert!(
                (observed - REQUESTED_HORIZONTAL_STD_M).abs() / REQUESTED_HORIZONTAL_STD_M < 0.03,
                "{name} spread is {observed:.3} m for a requested {REQUESTED_HORIZONTAL_STD_M} m \
                 at {SPREAD_TEST_LATITUDE_DEG} deg latitude (#331)"
            );
        }
        assert_approx_eq!(
            rbpf.linear_covariance()[(LinearLayout::ALTITUDE, LinearLayout::ALTITUDE)],
            REQUESTED_VERTICAL_STD_M.powi(2),
            1e-12
        );
        let (_, covariance) = rbpf.estimate();
        assert_approx_eq!(covariance[(2, 2)], REQUESTED_VERTICAL_STD_M.powi(2), 1e-9);
    }

    /// Inflating the cloud multiplies its covariance by the requested factor (#340), in the
    /// spread and the conditional half alike, without moving the mean.
    #[test]
    fn inflating_the_cloud_multiplies_its_covariance_by_the_factor() {
        const FACTOR: f64 = 4.0;

        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            spread_test_nominal_state(),
            RbpfConfig {
                num_particles: 2_000,
                seed: 340,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        for (i, particle) in rbpf.particles.iter_mut().enumerate() {
            particle.linear_state[LinearLayout::VELOCITY] = 0.1 * ((i % 5) as f64 - 2.0);
        }

        let (mean_before, covariance_before) = rbpf.estimate();
        rbpf.inflate_particle_spread(FACTOR);
        let (mean_after, covariance_after) = rbpf.estimate();

        for index in 0..mean_before.len() {
            assert_approx_eq!(
                mean_after[index],
                mean_before[index],
                1e-12 * mean_before[index].abs().max(1.0)
            );
        }
        for index in 0..covariance_before.nrows() {
            let expected = FACTOR * covariance_before[(index, index)];
            assert_approx_eq!(
                covariance_after[(index, index)],
                expected,
                1e-9 * expected.abs().max(1e-12)
            );
        }
        for inert in [1.0, 0.5, f64::NAN] {
            let (mean, covariance) = rbpf.estimate();
            rbpf.inflate_particle_spread(inert);
            let (mean_now, covariance_now) = rbpf.estimate();
            assert_approx_eq!(mean_now[0], mean[0], 1e-15);
            assert_approx_eq!(covariance_now[(0, 0)], covariance[(0, 0)], 1e-18);
        }
    }

    /// Velocity and attitude report their conditional covariance, not just the particles'
    /// spread: fresh from `new`, every particle holds the same conditional mean, and the whole
    /// prior is conditional.
    #[test]
    fn rbpf_reports_the_conditional_velocity_and_attitude_covariance() {
        const VELOCITY_INIT_STD: f64 = 1.5;
        const ATTITUDE_INIT_STD: f64 = 0.2;
        const EAST_VELOCITY: usize = LinearLayout::VELOCITY + 1;

        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            spread_test_nominal_state(),
            RbpfConfig {
                num_particles: 300,
                velocity_init_std_mps: VELOCITY_INIT_STD,
                attitude_init_std_rad: ATTITUDE_INIT_STD,
                seed: 12,
                ..RbpfConfig::default()
            },
        )
        .unwrap();

        // Level, so the Euler angles answer to the tilt one for one.
        let (_, fresh) = rbpf.estimate();
        for (axis, attitude) in ATTITUDE_STATE_INDICES.into_iter().enumerate() {
            let velocity = 3 + axis;
            assert_approx_eq!(fresh[(velocity, velocity)], VELOCITY_INIT_STD.powi(2), 1e-9);
            assert_approx_eq!(fresh[(attitude, attitude)], ATTITUDE_INIT_STD.powi(2), 1e-9);
        }

        let mut total = 0.0;
        for (i, particle) in rbpf.particles.iter_mut().enumerate() {
            particle.weight = 1.0 + (i % 4) as f64;
            total += particle.weight;
            particle.linear_state[EAST_VELOCITY] = 0.5 * ((i % 5) as f64 - 2.0);
        }
        for particle in &mut rbpf.particles {
            particle.weight /= total;
        }
        let mean: f64 = rbpf
            .particles
            .iter()
            .map(|p| p.weight * p.linear_state[EAST_VELOCITY])
            .sum();
        let spread: f64 = rbpf
            .particles
            .iter()
            .map(|p| p.weight * (p.linear_state[EAST_VELOCITY] - mean).powi(2))
            .sum();
        let conditional = rbpf.linear_covariance[(EAST_VELOCITY, EAST_VELOCITY)];

        let (_, covariance) = rbpf.estimate();
        assert_approx_eq!(covariance[(4, 4)], spread + conditional, 1e-9);
        assert!(spread > 0.0 && conditional > 0.0);
    }

    /// The particles move once per measurement epoch (Algorithm 1, steps 2 and 8), while the
    /// estimate keeps up with the nominal in between.
    ///
    /// Inertial samples alone mechanize the nominal and accumulate the time update; no particle
    /// moves and no draw is taken. The reported covariance still grows over those samples,
    /// because the estimate applies the pending update deterministically. The first measurement
    /// then flushes it, and the particles move.
    #[test]
    fn the_time_update_runs_once_per_measurement_epoch() {
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            spread_test_nominal_state(),
            RbpfConfig {
                num_particles: 400,
                seed: 8,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        let before: Vec<Vector2<f64>> = rbpf.particles.iter().map(|p| p.position_error).collect();
        let (_, covariance_before) = rbpf.estimate();
        for _ in 0..20 {
            rbpf.predict(&stationary_imu(), 0.1).unwrap();
        }
        assert!(rbpf.pending.is_some());
        for (particle, before) in rbpf.particles.iter().zip(&before) {
            assert_eq!(
                particle.position_error, *before,
                "a particle moved between epochs"
            );
        }
        let (_, covariance_pending) = rbpf.estimate();
        assert!(
            covariance_pending[(0, 0)] > covariance_before[(0, 0)],
            "two seconds of velocity uncertainty must show in the reported latitude variance \
             before any measurement arrives"
        );

        rbpf.update(&GPSPositionMeasurement {
            latitude: SPREAD_TEST_LATITUDE_DEG,
            longitude: SPREAD_TEST_LONGITUDE_DEG,
            altitude: SPREAD_TEST_ALTITUDE_M,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 5.0,
        })
        .unwrap();
        assert!(
            rbpf.pending.is_none(),
            "a measurement epoch applies the time update"
        );
    }

    /// The estimate between epochs is the distribution the flush realises.
    ///
    /// Before a flush the pending update is applied deterministically: the means through the
    /// transition, the covariance as `A C Aᵀ + Q`. The flush instead draws each particle's
    /// position from `N` and moves its linear means through `L`. The two must describe the
    /// same distribution, so the reported position and velocity variances agree to the
    /// sampling error of the draw.
    #[test]
    fn the_estimate_between_epochs_matches_the_flushed_cloud() {
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            spread_test_nominal_state(),
            RbpfConfig {
                num_particles: 20_000,
                position_init_std_m: Vector3::new(1e-3, 1e-3, 1.0),
                velocity_init_std_mps: 2.0,
                seed: 21,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        for _ in 0..30 {
            rbpf.predict(&stationary_imu(), 0.1).unwrap();
        }
        let (_, pending) = rbpf.estimate();
        rbpf.flush_time_update().unwrap();
        let (_, flushed) = rbpf.estimate();
        for (index, name) in [(0, "latitude"), (1, "longitude"), (3, "north velocity")] {
            let ratio = flushed[(index, index)] / pending[(index, index)];
            assert!(
                (ratio - 1.0).abs() < 0.05,
                "{name}: the flushed cloud reports {ratio:.4} of the variance the pending \
                 estimate did"
            );
        }
    }

    /// A map channel's temporal variation `V` is a first-order Gauss-Markov process, exactly,
    /// whatever the length of the epoch it is propagated over.
    ///
    /// - Started from nothing, its variance after one correlation time is
    ///   `sigma^2 (1 - e^-2)` whether that time is one epoch of 2000 samples or 200 epochs of
    ///   ten -- the accumulated noise carries the state's own decay (departure 5).
    /// - Started from its stationary distribution, which is where `new` puts it, it stays.
    /// - The nominal `V` decays by `e^-1` per correlation time; the offset `c` neither decays
    ///   nor gains variance.
    #[test]
    fn a_map_channels_temporal_variation_is_a_gauss_markov_process() {
        const SIGMA: f64 = 3.0;
        const TAU_S: f64 = 20.0;

        let propagate = |samples: usize, dt: f64, samples_per_epoch: usize, from_zero: bool| {
            let mut rbpf = RaoBlackwellizedParticleFilter::new(
                spread_test_nominal_state(),
                RbpfConfig {
                    num_particles: 16,
                    map_bias_channels: 1,
                    map_bias_initial: vec![7.0],
                    map_bias_init_std: vec![5.0],
                    map_variation_std: vec![SIGMA],
                    map_variation_time_constant_s: vec![TAU_S],
                    seed: 1,
                    ..RbpfConfig::default()
                },
            )
            .unwrap();
            let variation = LinearLayout::variation(0);
            if from_zero {
                rbpf.linear_covariance[(variation, variation)] = 0.0;
            }
            rbpf.nominal_variation[0] = 4.0;
            for sample in 1..=samples {
                rbpf.predict(&stationary_imu(), dt).unwrap();
                if sample % samples_per_epoch == 0 {
                    rbpf.flush_time_update().unwrap();
                }
            }
            rbpf.flush_time_update().unwrap();
            rbpf
        };

        let expected = SIGMA * SIGMA * (1.0 - (-2.0_f64).exp());
        for (samples, dt, per_epoch) in [(2_000, 0.01, 2_000), (2_000, 0.01, 10), (200, 0.1, 1)] {
            let rbpf = propagate(samples, dt, per_epoch, true);
            let (variation, offset) = (LinearLayout::variation(0), LinearLayout::offset(0));
            assert_approx_eq!(
                rbpf.linear_covariance[(variation, variation)],
                expected,
                1e-9 * expected
            );
            assert_approx_eq!(rbpf.nominal_variation[0], 4.0 * (-1.0_f64).exp(), 1e-9);
            assert_approx_eq!(rbpf.nominal_offset[0], 7.0, 1e-15);
            assert_approx_eq!(
                rbpf.linear_covariance[(offset, offset)],
                5.0_f64.mul_add(5.0, -SIGMA * SIGMA),
                1e-9
            );
        }

        let stationary = propagate(500, 0.1, 7, false);
        let variation = LinearLayout::variation(0);
        assert_approx_eq!(
            stationary.linear_covariance[(variation, variation)],
            SIGMA * SIGMA,
            1e-9
        );
    }

    /// Feed a stationary filter with a vertical accelerometer error `seconds` of inertial data,
    /// and, if `with_barometer`, a barometric altitude at the truth once a second.
    fn stationary_run_with_vertical_accel_error(
        accel_error: f64,
        seconds: usize,
        with_barometer: bool,
    ) -> RaoBlackwellizedParticleFilter {
        let truth = spread_test_nominal_state();
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            truth,
            RbpfConfig {
                num_particles: 64,
                seed: 4,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        let mut imu = stationary_imu();
        imu.accel[2] += accel_error;
        let baro = RelativeAltitudeMeasurement {
            relative_altitude: 0.0,
            reference_altitude: truth.altitude,
            ..Default::default()
        };
        for step in 0..seconds * 10 {
            if with_barometer && step % 10 == 0 {
                assert!(rbpf.update(&baro).unwrap().accepted);
            }
            rbpf.predict(&imu, 0.1).unwrap();
        }
        rbpf
    }

    /// The barometer loop holds the vertical channel, and its acceleration correction learns
    /// the accelerometer error.
    ///
    /// Unaided, a 0.05 m/s^2 vertical accelerometer error puts the altitude kilometres off in
    /// five minutes -- `0.5 * 0.05 * 300^2 = 2250 m` before the channel's own instability.
    /// Aided by a barometer reading the truth, the third-order loop keeps it within a metre,
    /// and `â` converges on the error, which is what the loop's integrator is for. No
    /// measurement epoch happens in either run: the loop lives in the mechanization.
    #[test]
    fn the_barometer_loop_holds_the_vertical_channel() {
        const ACCEL_ERROR: f64 = 0.05;
        let truth_altitude = SPREAD_TEST_ALTITUDE_M;

        let unaided = stationary_run_with_vertical_accel_error(ACCEL_ERROR, 300, false);
        assert!(
            (unaided.nominal_state().altitude - truth_altitude).abs() > 1000.0,
            "the unaided vertical channel should have run away; it is {:.1} m off",
            unaided.nominal_state().altitude - truth_altitude
        );

        let aided = stationary_run_with_vertical_accel_error(ACCEL_ERROR, 300, true);
        let altitude_error = aided.nominal_state().altitude - truth_altitude;
        assert!(
            altitude_error.abs() < 1.0,
            "the barometer loop left the altitude {altitude_error:.3} m off"
        );
        assert!(
            (aided.baro_accel_correction() - ACCEL_ERROR).abs() < 0.1 * ACCEL_ERROR,
            "the loop's acceleration correction is {:.4} m/s^2 against an accelerometer error \
             of {ACCEL_ERROR}",
            aided.baro_accel_correction()
        );
        assert!(aided.nominal_state().velocity_vertical.abs() < 0.05);
    }

    /// The barometer loop's error model is the loop's own feedback, with the right signs.
    ///
    /// Two checks, in both frames, since vertical velocity changes sign between them and the
    /// loop's rows carry that sign:
    ///
    /// - **The transition.** With the loop running, `F` must differ from the unaided `F` by
    ///   exactly the loop terms: `-k₁ dt` into altitude, `-up k₂ dt` into vertical velocity and
    ///   `+k₃ dt` into the acceleration correction, from both the altitude error and the
    ///   barometer error, and `-up dt` from the correction into vertical velocity.
    /// - **The mechanization.** Two nominals an altitude error apart, run one sample against
    ///   the same barometer, must end apart by what those terms say the loop does to them.
    #[test]
    fn the_barometer_loops_error_model_matches_its_mechanization() {
        const ALTITUDE_ERROR_M: f64 = 0.5;
        const DT: f64 = 0.1;

        for is_enu in [true, false] {
            let up = if is_enu { 1.0 } else { -1.0 };
            let mut nominal = spread_test_nominal_state();
            nominal.is_enu = is_enu;
            let build = |state: StrapdownState| {
                let mut rbpf = RaoBlackwellizedParticleFilter::new(
                    state,
                    RbpfConfig {
                        num_particles: 4,
                        ..RbpfConfig::default()
                    },
                )
                .unwrap();
                rbpf.baro.altitude = Some(SPREAD_TEST_ALTITUDE_M + 2.0);
                rbpf.baro.accel_correction = 0.01;
                rbpf
            };
            let gravity = earth::gravity(&SPREAD_TEST_LATITUDE_DEG, &SPREAD_TEST_ALTITUDE_M);
            let imu = IMUData {
                accel: Vector3::new(0.0, 0.0, up * gravity),
                gyro: Vector3::zeros(),
            };

            let mut base = build(nominal);
            let (k1, k2, k3) = base.baro_loop_gains();
            let aided = base.error_transition(&imu.accel, &imu.gyro, DT);
            base.baro.altitude = None;
            let unaided = base.error_transition(&imu.accel, &imu.gyro, DT);
            base.baro.altitude = Some(SPREAD_TEST_ALTITUDE_M + 2.0);
            let loop_terms = &aided - &unaided;
            let row = |linear: usize| SAMPLED_STATE_DIM + linear;
            let (altitude, vertical, baro_error, accel) = (
                row(LinearLayout::ALTITUDE),
                row(LinearLayout::VERTICAL_VELOCITY),
                row(LinearLayout::BARO_ERROR),
                row(LinearLayout::ACCEL_CORRECTION),
            );
            for source in [altitude, baro_error] {
                assert_approx_eq!(loop_terms[(altitude, source)], -k1 * DT, 1e-15);
                assert_approx_eq!(loop_terms[(vertical, source)], -up * k2 * DT, 1e-15);
                assert_approx_eq!(loop_terms[(accel, source)], k3 * DT, 1e-15);
            }
            assert_approx_eq!(loop_terms[(vertical, accel)], -up * DT, 1e-15);
            assert_eq!(
                loop_terms.iter().filter(|term| **term != 0.0).count(),
                7,
                "the loop must add exactly its seven terms to the transition"
            );

            // The mechanization: the nominal a metre-scale error higher is pulled down harder,
            // its vertical velocity pushed the same way, and its correction integrates more.
            let mut perturbed_state = nominal;
            perturbed_state.altitude += ALTITUDE_ERROR_M;
            let mut perturbed = build(perturbed_state);
            base.predict(&imu, DT).unwrap();
            perturbed.predict(&imu, DT).unwrap();
            let altitude_gap = perturbed.nominal_state().altitude - base.nominal_state().altitude;
            let vertical_gap = perturbed.nominal_state().velocity_vertical
                - base.nominal_state().velocity_vertical;
            let accel_gap = perturbed.baro_accel_correction() - base.baro_accel_correction();
            assert_approx_eq!(
                altitude_gap - ALTITUDE_ERROR_M,
                -k1 * ALTITUDE_ERROR_M * DT,
                1e-6
            );
            assert_approx_eq!(accel_gap, k3 * ALTITUDE_ERROR_M * DT, 1e-12);
            // The unaided mechanization adds only a gravity-gradient term, ~1e-7 m/s here.
            assert_approx_eq!(vertical_gap, -up * k2 * ALTITUDE_ERROR_M * DT, 1e-5);
        }
    }

    /// A barometer reading is the loop's input, not a measurement epoch: it neither reweights
    /// the cloud nor applies the pending time update.
    #[test]
    fn a_barometer_reading_aids_the_loop_without_an_epoch() {
        let mut rbpf = filter_with_map_channels(0, 3);
        for (i, particle) in rbpf.particles.iter_mut().enumerate() {
            particle.weight = (1 + i % 3) as f64;
        }
        let total: f64 = rbpf.particles.iter().map(|p| p.weight).sum();
        for particle in &mut rbpf.particles {
            particle.weight /= total;
        }
        rbpf.predict(&stationary_imu(), 0.1).unwrap();
        let weights: Vec<f64> = rbpf.particles.iter().map(|p| p.weight).collect();

        let outcome = rbpf
            .update(&RelativeAltitudeMeasurement {
                relative_altitude: 1.5,
                reference_altitude: 100.0,
                ..Default::default()
            })
            .unwrap();
        assert!(outcome.accepted);
        assert_eq!(rbpf.baro.altitude, Some(101.5));
        assert!(
            rbpf.pending.is_some(),
            "a barometer reading is not a measurement epoch"
        );
        assert_eq!(
            rbpf.particles.iter().map(|p| p.weight).collect::<Vec<_>>(),
            weights
        );
    }

    /// The map states stay decoupled from the navigation block: nothing propagates between
    /// them and the process noise is diagonal, so a GNSS fix, whose Jacobian has no map
    /// column, must not touch them, and a fix on the map bias alone must touch nothing else.
    #[test]
    fn the_map_states_stay_decoupled_from_the_navigation_block() {
        let mut rbpf = filter_with_map_channels(2, 4242);
        let imu = IMUData {
            accel: Vector3::new(0.0, 0.0, -earth::gravity(&40.1, &100.0)),
            gyro: Vector3::new(0.01, -0.02, 0.03),
        };
        for _ in 0..8 {
            rbpf.predict(&imu, 0.05).unwrap();
        }
        rbpf.flush_time_update().unwrap();
        let map_base = LinearLayout::MAP_BASE;
        let width = rbpf.linear_state_dim();
        let map_before: Vec<f64> = (map_base..width)
            .map(|state| rbpf.linear_covariance[(state, state)])
            .collect();

        rbpf.update(&GPSPositionAndVelocityMeasurement {
            latitude: 0.7_f64.to_degrees(),
            longitude: (-1.3_f64).to_degrees(),
            altitude: 100.0,
            northward_velocity: 0.1,
            eastward_velocity: -0.2,
            horizontal_noise_std: 3.0,
            vertical_noise_std: 5.0,
            velocity_noise_std: 0.3,
        })
        .unwrap();

        assert_map_block_is_decoupled(&rbpf, "a GNSS update");
        for (state, before) in (map_base..width).zip(map_before) {
            let after = rbpf.linear_covariance[(state, state)];
            assert!(
                after == before,
                "P[({state},{state})] went {before:e} -> {after:e} across a GNSS update"
            );
        }

        // The GNSS update ended in a resample, whose survivors need not be zero-mean in tilt;
        // folding that mean into the nominal transports the tilt covariance, which is
        // recentring's doing rather than the map fix's. Fold it first.
        rbpf.recenter();
        let navigation_before = rbpf
            .linear_covariance
            .view((0, 0), (map_base, map_base))
            .into_owned();
        let observed = LinearLayout::offset(1);
        let observed_before = rbpf.linear_covariance[(observed, observed)];
        rbpf.update(&BiasOnlyMeasurement {
            observed: 1.0,
            noise_std: 1.0,
        })
        .unwrap();

        assert_map_block_is_decoupled(&rbpf, "a map-bias update");
        assert_eq!(
            rbpf.linear_covariance
                .view((0, 0), (map_base, map_base))
                .into_owned(),
            navigation_before,
            "a map-bias update changed the navigation covariance"
        );
        assert!(rbpf.linear_covariance[(observed, observed)] < observed_before);
    }

    /// Assert that the (map, navigation) cross-block of the conditional covariance is zero.
    fn assert_map_block_is_decoupled(rbpf: &RaoBlackwellizedParticleFilter, what: &str) {
        let covariance = &rbpf.linear_covariance;
        for map in LinearLayout::MAP_BASE..covariance.nrows() {
            for navigation in 0..LinearLayout::MAP_BASE {
                assert!(
                    covariance[(map, navigation)] == 0.0 && covariance[(navigation, map)] == 0.0,
                    "P[({map},{navigation})] = {:e} after {what}",
                    covariance[(map, navigation)]
                );
            }
        }
    }

    /// A heading measurement reaches the attitude states (#341): every particle shares the
    /// nominal attitude, so reweighting alone cannot move yaw.
    #[test]
    fn rbpf_magnetometer_update_moves_yaw_toward_the_measurement() {
        // Level and pointing north, with the magnetometer seeing a field rotated 30 deg
        // away: in NED, yaw = atan2(-m_y, m_x), so this is a heading of -30 deg.
        let truth_yaw = -std::f64::consts::FRAC_PI_6;
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                attitude: Rotation3::from_euler_angles(0.0, 0.0, 0.0),
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 200,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        let mag = MagnetometerYawMeasurement {
            mag_x: truth_yaw.cos(),
            mag_y: -truth_yaw.sin(),
            mag_z: 0.0,
            noise_std: 0.05,
            apply_declination: false,
            is_enu: false,
            ..MagnetometerYawMeasurement::default()
        };
        let before = rbpf.estimate().0[8];
        assert_approx_eq!(before, 0.0, 1e-12);
        for _ in 0..20 {
            rbpf.update(&mag).unwrap();
        }
        let after = rbpf.estimate().0[8];
        let closed = (after - before) / (truth_yaw - before);
        assert!(
            closed > 0.9,
            "the magnetometer closed {:.1}% of a {:.1} deg heading offset (#341)",
            closed * 100.0,
            (truth_yaw - before).to_degrees()
        );
    }

    /// A yaw measurement at a tilt moves the reported yaw by the Euler-chart Kalman gain (#349).
    ///
    /// Every `get_jacobian` writes its attitude columns against Euler angles; this filter
    /// carries a nav-frame tilt, reached through `∂Φ/∂ε = E⁻¹`. The expected step is the scalar
    /// Kalman update in Euler space, `Σ_yaw / (Σ_yaw + R) * r`, with `Σ = E⁻¹ P E⁻ᵀ`.
    #[test]
    fn a_heading_fix_at_a_tilt_takes_the_euler_chart_gain() {
        const ROLL: f64 = 1.2;
        const PITCH: f64 = 0.2;
        const YAW: f64 = 0.4;
        const RESIDUAL: f64 = 0.02;
        const NOISE_STD: f64 = 0.1;

        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                attitude: Rotation3::from_euler_angles(ROLL, PITCH, YAW),
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 4,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        let (_, prior) = rbpf.estimate();
        let prior_yaw_variance = prior[(8, 8)];
        rbpf.update(&YawMeasurement {
            yaw: YAW + RESIDUAL,
            noise_std: NOISE_STD,
        })
        .unwrap();

        let expected_step =
            prior_yaw_variance / (prior_yaw_variance + NOISE_STD.powi(2)) * RESIDUAL;
        let step = rbpf.estimate().0[8] - YAW;
        assert!(
            (step / expected_step - 1.0).abs() < 0.02,
            "yaw moved {step:.6} rad for an expected {expected_step:.6}"
        );
    }

    /// Each map channel's total bias `V + c`, and its variance, come back through
    /// [`RaoBlackwellizedParticleFilter::estimate`], against moments computed straight off the
    /// particles and the shared covariance.
    #[test]
    fn rbpf_estimate_reports_each_map_bias_and_its_variance() {
        const INIT_STD: [f64; 2] = [3.0, 40.0];
        const SEED: [f64; 2] = [635.0, -17_500.0];

        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 400,
                map_bias_channels: 2,
                map_bias_initial: SEED.to_vec(),
                map_bias_init_std: INIT_STD.to_vec(),
                map_variation_std: vec![1.0, 5.0],
                map_variation_time_constant_s: vec![300.0; 2],
                seed: 7,
                ..RbpfConfig::default()
            },
        )
        .unwrap();

        let (seeded_mean, seeded_cov) = rbpf.estimate();
        for channel in 0..2 {
            let index = REPORTED_BASE_DIM + channel;
            assert_approx_eq!(seeded_mean[index], SEED[channel], 1e-9);
            assert_approx_eq!(seeded_cov[(index, index)], INIT_STD[channel].powi(2), 1e-9);
        }

        let mut total = 0.0;
        for (i, particle) in rbpf.particles.iter_mut().enumerate() {
            particle.weight = 1.0 + (i % 5) as f64;
            total += particle.weight;
            particle.linear_state[LinearLayout::offset(0)] = (i % 7) as f64 - 3.0;
            particle.linear_state[LinearLayout::variation(1)] = 10.0 * ((i % 3) as f64 - 1.0);
        }
        for particle in &mut rbpf.particles {
            particle.weight /= total;
        }

        let (mean, covariance) = rbpf.estimate();
        for (channel, seed) in SEED.iter().enumerate() {
            let (variation, offset) = (
                LinearLayout::variation(channel),
                LinearLayout::offset(channel),
            );
            let error = |p: &RbpfParticle| p.linear_state[variation] + p.linear_state[offset];
            let error_mean: f64 = rbpf.particles.iter().map(|p| p.weight * error(p)).sum();
            let spread: f64 = rbpf
                .particles
                .iter()
                .map(|p| p.weight * (error(p) - error_mean).powi(2))
                .sum();
            let p = &rbpf.linear_covariance;
            let conditional =
                p[(variation, variation)] + p[(offset, offset)] + 2.0 * p[(variation, offset)];
            let index = REPORTED_BASE_DIM + channel;
            assert_approx_eq!(mean[index], seed + error_mean, 1e-9);
            assert_approx_eq!(covariance[(index, index)], spread + conditional, 1e-9);
            assert!(spread > 0.0);
            assert_approx_eq!(rbpf.map_bias_estimate()[channel], mean[index], 1e-9);
        }
    }

    /// The reported attitude is a mean on the circle, on the principal branch (#341, #314).
    #[test]
    fn rbpf_attitude_estimate_is_a_circular_mean_on_the_principal_branch() {
        let nominal = StrapdownState {
            attitude: Rotation3::from_euler_angles(0.0, 0.0, std::f64::consts::PI),
            ..StrapdownState::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            nominal,
            RbpfConfig {
                num_particles: 2,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        // Level at yaw +pi, so tilts of -0.1 and +0.3 rad about the vertical put the two
        // particles either side of the cut, skewed so the circular and linear means differ.
        let yaw_tilt = LinearLayout::TILT + 2;
        rbpf.particles[0].linear_state[yaw_tilt] = -0.1;
        rbpf.particles[1].linear_state[yaw_tilt] = 0.3;
        let yaws: Vec<f64> = rbpf
            .particles
            .iter()
            .map(|p| rbpf.reported_state(&p.position_error, &p.linear_state)[8])
            .collect();
        assert!(
            yaws[0] > 0.0 && yaws[1] < 0.0,
            "setup should straddle the cut: {yaws:?}"
        );

        let (mean, cov) = rbpf.estimate();
        let yaw = mean[8];
        assert!((-std::f64::consts::PI..=std::f64::consts::PI).contains(&yaw));
        assert_approx_eq!(yaw, crate::wrap_to_pi(std::f64::consts::PI + 0.1), 1e-12);
        let conditional = RbpfConfig::default().attitude_init_std_rad.powi(2);
        assert_approx_eq!(cov[(8, 8)], 0.04 + conditional, 1e-12);
    }

    /// A tight cloud must be unaffected by averaging on the circle rather than the line.
    #[test]
    fn rbpf_circular_mean_agrees_with_the_linear_mean_for_a_tight_cloud() {
        let angles = [0.30, 0.31, 0.29, 0.305, 0.295];
        let weights = [0.1, 0.3, 0.2, 0.25, 0.15];
        let linear: f64 = angles.iter().zip(weights).map(|(a, w)| a * w).sum();
        let circular = circular_mean(angles.iter().zip(weights));
        assert_approx_eq!(circular, linear, 1e-6);
    }

    /// A cloud with no mean direction returns a finite angle rather than failing.
    #[test]
    fn rbpf_circular_mean_of_an_undirected_cloud_is_finite() {
        let angles = [0.0, std::f64::consts::PI];
        let mean = circular_mean(angles.iter().zip([0.5, 0.5]));
        assert!(mean.is_finite());
    }

    /// A measurement of altitude plus a map bias read from the end of the reported state, the
    /// way the `geonav` map models read theirs.
    #[derive(Debug)]
    struct BiasedAltitudeMeasurement {
        observed: f64,
        noise_std: f64,
        bias_from_end: Option<usize>,
        /// Widths this model has been handed, for asserting what a caller passed it.
        seen_state_len: std::cell::RefCell<Vec<usize>>,
    }

    impl BiasedAltitudeMeasurement {
        fn new(observed: f64, bias_from_end: Option<usize>) -> Self {
            Self {
                observed,
                noise_std: 1.0,
                bias_from_end,
                seen_state_len: std::cell::RefCell::new(Vec::new()),
            }
        }

        /// The resolved bias index, which must land after the fifteen reported navigation and
        /// bias states; `None` when the vector has no room for one.
        fn bias_index(&self, state: &DVector<f64>) -> Option<usize> {
            let offset = self.bias_from_end?;
            let index = state.len().checked_sub(offset)?;
            (offset > 0 && index >= REPORTED_BASE_DIM).then_some(index)
        }
    }

    impl MeasurementModel for BiasedAltitudeMeasurement {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn get_dimension(&self) -> usize {
            1
        }
        fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
            Ok(DVector::from_vec(vec![self.observed]))
        }
        fn get_noise(&self) -> DMatrix<f64> {
            DMatrix::from_element(1, 1, self.noise_std.powi(2))
        }
        fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
            self.seen_state_len.borrow_mut().push(state.len());
            let bias = self.bias_index(state).map_or(0.0, |index| state[index]);
            DVector::from_vec(vec![state[2] + bias])
        }
        fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
            let mut h = DMatrix::<f64>::zeros(1, state.len());
            h[(0, 2)] = 1.0;
            if let Some(offset) = self.bias_from_end {
                let index = self
                    .bias_index(state)
                    .ok_or(StrapdownError::DimensionMismatch {
                        what: "map bias state",
                        expected: REPORTED_BASE_DIM + offset,
                        got: state.len(),
                    })?;
                h[(0, index)] = 1.0;
            }
            Ok(h)
        }
    }

    /// [`BiasedAltitudeMeasurement`] with an edge to its map: a particle above
    /// `edge_altitude` is predicted as NaN, the way a geophysical model predicts for a particle
    /// that has left its map tile while the ensemble mean has not.
    #[derive(Debug)]
    struct EdgedAltitudeMeasurement {
        inner: BiasedAltitudeMeasurement,
        edge_altitude: f64,
    }

    impl MeasurementModel for EdgedAltitudeMeasurement {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn get_dimension(&self) -> usize {
            1
        }
        fn get_measurement(&self, state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
            self.inner.get_measurement(state)
        }
        fn get_noise(&self) -> DMatrix<f64> {
            self.inner.get_noise()
        }
        fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
            if state[2] > self.edge_altitude {
                DVector::from_element(1, f64::NAN)
            } else {
                self.inner.get_expected_measurement(state)
            }
        }
        fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
            self.inner.get_jacobian(state)
        }
    }

    /// A direct measurement of the last map channel's total bias, and nothing else.
    #[derive(Debug)]
    struct BiasOnlyMeasurement {
        observed: f64,
        noise_std: f64,
    }

    impl MeasurementModel for BiasOnlyMeasurement {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn get_dimension(&self) -> usize {
            1
        }
        fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
            Ok(DVector::from_vec(vec![self.observed]))
        }
        fn get_noise(&self) -> DMatrix<f64> {
            DMatrix::from_element(1, 1, self.noise_std.powi(2))
        }
        fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
            DVector::from_vec(vec![state[state.len() - 1]])
        }
        fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
            let mut h = DMatrix::<f64>::zeros(1, state.len());
            h[(0, state.len() - 1)] = 1.0;
            Ok(h)
        }
    }

    /// A direct measurement of Euler yaw: the Euler-chart row every heading model writes.
    #[derive(Debug)]
    struct YawMeasurement {
        yaw: f64,
        noise_std: f64,
    }

    impl MeasurementModel for YawMeasurement {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn get_dimension(&self) -> usize {
            1
        }
        fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
            Ok(DVector::from_vec(vec![self.yaw]))
        }
        fn get_noise(&self) -> DMatrix<f64> {
            DMatrix::from_element(1, 1, self.noise_std.powi(2))
        }
        fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
            DVector::from_vec(vec![state[8]])
        }
        fn get_jacobian(&self, _state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
            let mut h = DMatrix::<f64>::zeros(1, 9);
            h[(0, 8)] = 1.0;
            Ok(h)
        }
        fn wrap_residual(&self, residual: &mut DVector<f64>) {
            residual[0] = crate::wrap_to_pi(residual[0]);
        }
    }

    /// Recentring moves every mean error into its nominal and leaves the cloud zero-mean,
    /// without changing any particle's reported state.
    #[test]
    fn rbpf_recentring_moves_every_mean_into_its_nominal() {
        let mut rbpf = filter_with_map_channels(2, 7);
        for (i, particle) in rbpf.particles.iter_mut().enumerate() {
            let jitter = 1e-3 * ((i % 5) as f64 - 2.0);
            particle.linear_state[LinearLayout::ALTITUDE] += 2.0 + jitter;
            particle.linear_state[LinearLayout::VELOCITY] += 0.3;
            particle.linear_state[LinearLayout::TILT + 2] += 0.01 + jitter;
            particle.linear_state[LinearLayout::BARO_ERROR] += 1.5;
            particle.linear_state[LinearLayout::ACCEL_CORRECTION] -= 0.02;
            particle.linear_state[LinearLayout::variation(0)] += 25.0;
            particle.linear_state[LinearLayout::offset(1)] -= 4.0;
        }
        let before: Vec<DVector<f64>> = rbpf
            .particles
            .iter()
            .map(|p| rbpf.reported_state(&p.position_error, &p.linear_state))
            .collect();

        rbpf.recenter();

        let (mean_position, mean_linear) = rbpf.mean_errors();
        assert!(mean_position.norm() < 1e-15);
        assert!(
            mean_linear.amax() < 1e-12,
            "not zero-mean after recentring: {mean_linear}"
        );
        assert_approx_eq!(rbpf.baro_error_estimate(), 1.5, 1e-12);
        assert_approx_eq!(rbpf.baro_accel_correction(), -0.02, 1e-12);
        assert_approx_eq!(rbpf.nominal_variation[0], 25.0, 1e-12);
        assert_approx_eq!(rbpf.nominal_offset[1], -4.0, 1e-12);

        // What a measurement model sees must not have moved. Attitude composes rather than
        // adds, so it agrees to second order in the jitter rather than exactly.
        for (particle, before) in rbpf.particles.iter().zip(&before) {
            let after = rbpf.reported_state(&particle.position_error, &particle.linear_state);
            for index in 0..after.len() {
                let tolerance = if ATTITUDE_STATE_INDICES.contains(&index) {
                    1e-7
                } else {
                    1e-9
                };
                assert_approx_eq!(after[index], before[index], tolerance);
            }
        }
    }

    /// The same, reached through the public update path.
    #[test]
    fn rbpf_map_states_are_recentred_through_a_measurement_update() {
        let mut rbpf = filter_with_map_channels(1, 11);
        let offset = LinearLayout::offset(0);
        for particle in &mut rbpf.particles {
            particle.linear_state[offset] += 12.0;
        }
        let measurement = BiasedAltitudeMeasurement::new(100.0 + 12.0, Some(1));
        assert!(rbpf.update(&measurement).unwrap().accepted);

        let (_, mean_linear) = rbpf.mean_errors();
        assert!(
            mean_linear[offset].abs() < 1e-9,
            "not zero-mean: {}",
            mean_linear[offset]
        );
        assert!(
            (rbpf.map_bias_estimate()[0] - 12.0).abs() < 0.5,
            "the bias estimate should survive recentring near its 12.0 truth, got {}",
            rbpf.map_bias_estimate()[0]
        );
    }

    /// The innovation gate must score a measurement against the states it declares: a map
    /// model reads its bias by index from the end of the vector, so the gate must hand it the
    /// whole reported state.
    #[test]
    fn rbpf_gate_scores_the_map_bias() {
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                attitude: Rotation3::from_euler_angles(0.0, 0.0, 0.9),
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 128,
                map_bias_channels: 1,
                map_bias_initial: vec![12.0],
                map_bias_init_std: vec![0.0],
                map_variation_std: vec![0.0],
                map_variation_time_constant_s: vec![300.0],
                position_init_std_m: Vector3::new(0.01, 0.01, 0.0),
                attitude_init_std_rad: 0.0,
                velocity_init_std_mps: 0.0,
                seed: 5,
                ..RbpfConfig::default()
            },
        )
        .unwrap();

        let measurement = BiasedAltitudeMeasurement::new(100.0 + 12.0, Some(1));
        let outcome = rbpf.evaluate_ensemble_gate(&measurement).unwrap();

        assert_eq!(measurement.seen_state_len.borrow().as_slice(), &[16]);
        assert!(
            outcome.nis < 1e-6,
            "consistent fix should gate at ~zero NIS, got {}",
            outcome.nis
        );
        assert!(outcome.accepted);
    }

    /// Ordinary GNSS and barometer readings survive a filter that carries map channels; the
    /// navigation models' nine-column Jacobians are padded to the reported width.
    #[test]
    fn rbpf_gps_update_survives_a_filter_carrying_map_channels() {
        let mut rbpf = filter_with_map_channels(2, 23);
        let gps = GPSPositionMeasurement {
            latitude: 0.7_f64.to_degrees(),
            longitude: (-1.3_f64).to_degrees(),
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
        };
        assert!(rbpf.update(&gps).unwrap().accepted);
        let baro = RelativeAltitudeMeasurement {
            relative_altitude: 0.0,
            reference_altitude: 100.0,
            ..Default::default()
        };
        assert!(rbpf.update(&baro).unwrap().accepted);
        let (mean, covariance) = rbpf.estimate();
        assert!(mean.iter().chain(covariance.iter()).all(|v| v.is_finite()));
    }

    /// A map bias is estimated by the Kalman half of the update, and exactly.
    ///
    /// With every particle at the same position and conditional mean every particle scores the
    /// same residual, so the whole update is the conditional one: a scalar Kalman filter on a
    /// constant, whose posterior after `k` fixes has a closed form,
    ///
    /// $$ P_k = \left(P_0^{-1} + k R^{-1}\right)^{-1}, \qquad
    ///    b_k = P_k \left(P_0^{-1} b_0 + k R^{-1} b^\ast\right). $$
    ///
    /// `V` is given no variance, so the offset `c` carries the whole prior.
    #[test]
    fn rbpf_map_bias_takes_the_exact_kalman_posterior() {
        const SEED: f64 = 600.0;
        const PRIOR_STD: f64 = 230.0;
        const TRUE_BIAS: f64 = 830.0;
        const NOISE_STD: f64 = 140.0;
        const FIXES: usize = 25;

        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 64,
                position_init_std_m: Vector3::new(1e-9, 1e-9, 0.0),
                map_bias_channels: 1,
                map_bias_initial: vec![SEED],
                map_bias_init_std: vec![PRIOR_STD],
                map_variation_std: vec![0.0],
                map_variation_time_constant_s: vec![300.0],
                seed: 17,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        let mut fix = BiasedAltitudeMeasurement::new(100.0 + TRUE_BIAS, Some(1));
        fix.noise_std = NOISE_STD;
        for _ in 0..FIXES {
            assert!(rbpf.update(&fix).unwrap().accepted);
        }

        let prior_precision = PRIOR_STD.powi(-2);
        let fix_precision = NOISE_STD.powi(-2);
        let posterior_variance = 1.0 / (FIXES as f64).mul_add(fix_precision, prior_precision);
        let posterior_mean = posterior_variance
            * (prior_precision * SEED + FIXES as f64 * fix_precision * TRUE_BIAS);

        let (mean, covariance) = rbpf.estimate();
        assert_approx_eq!(mean[15], posterior_mean, 1e-6);
        assert_approx_eq!(covariance[(15, 15)], posterior_variance, 1e-6);
    }

    /// Particles are weighted by the likelihood with the linear states integrated out: two
    /// particles differing only in their altitude estimate, scored against altitude plus a
    /// wide-prior bias, must weigh as `N(r; 0, C P Cᵀ + R)`, not `N(r; 0, R)`.
    #[test]
    fn rbpf_weights_a_map_fix_by_the_marginal_likelihood() {
        const PRIOR_STD: f64 = 3.0;
        const NOISE_STD: f64 = 1.0;

        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 2,
                position_init_std_m: Vector3::new(1e-9, 1e-9, 0.0),
                map_bias_channels: 1,
                map_bias_initial: vec![0.0],
                map_bias_init_std: vec![PRIOR_STD],
                map_variation_std: vec![0.0],
                map_variation_time_constant_s: vec![300.0],
                // Keep both particles in view rather than resampling on the skew.
                effective_sample_threshold: 0.0,
                seed: 5,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        rbpf.particles[0].linear_state[LinearLayout::ALTITUDE] = 1.0;
        rbpf.particles[1].linear_state[LinearLayout::ALTITUDE] = -2.0;
        let mut fix = BiasedAltitudeMeasurement::new(100.0, Some(1));
        fix.noise_std = NOISE_STD;
        rbpf.update(&fix).unwrap();

        // Residuals z - (altitude + bias) are -1 and +2.
        let marginal_variance = PRIOR_STD.mul_add(PRIOR_STD, NOISE_STD * NOISE_STD);
        let expected_ratio = ((4.0 - 1.0) / (2.0 * marginal_variance)).exp();
        let ratio = rbpf.particles[0].weight / rbpf.particles[1].weight;
        assert!(
            (ratio - expected_ratio).abs() < 1e-6,
            "weight ratio {ratio:.6} against the marginal likelihood's {expected_ratio:.6}"
        );
    }

    /// A particle that has left the map must not poison the Kalman step: some particles off
    /// the map, and nothing goes non-finite; every particle off the map, and nothing moves.
    #[test]
    fn rbpf_bias_update_skips_particles_that_have_left_the_map() {
        const NOMINAL_ALTITUDE_M: f64 = 100.0;
        let edge = NOMINAL_ALTITUDE_M + 3.0;

        let mut rbpf = filter_with_map_channels(1, 29);
        for (i, particle) in rbpf.particles.iter_mut().enumerate() {
            particle.linear_state[LinearLayout::ALTITUDE] = if i % 3 == 0 { 5.0 } else { -1.0 };
        }
        let partly = EdgedAltitudeMeasurement {
            inner: BiasedAltitudeMeasurement::new(NOMINAL_ALTITUDE_M + 2.0, Some(1)),
            edge_altitude: edge,
        };
        rbpf.update(&partly).unwrap();
        for (i, particle) in rbpf.particles.iter().enumerate() {
            assert!(
                particle.linear_state.iter().all(|v| v.is_finite())
                    && particle.position_error.iter().all(|v| v.is_finite()),
                "particle {i} went non-finite"
            );
        }
        let (mean, covariance) = rbpf.estimate();
        assert!(mean.iter().chain(covariance.iter()).all(|v| v.is_finite()));

        let mut rbpf = filter_with_map_channels(1, 31);
        let covariance_before = rbpf.linear_covariance.clone();
        let nowhere = EdgedAltitudeMeasurement {
            inner: BiasedAltitudeMeasurement::new(NOMINAL_ALTITUDE_M + 2.0, Some(1)),
            edge_altitude: f64::NEG_INFINITY,
        };
        rbpf.mpf_update(&nowhere).unwrap();
        assert_eq!(rbpf.linear_covariance, covariance_before);
        let offset = LinearLayout::offset(0);
        assert!(rbpf.particles.iter().all(|p| p.linear_state[offset] == 0.0));
    }

    /// An edit to a configuration, for tabulating the ones that must be refused.
    type ConfigEdit = dyn Fn(&mut RbpfConfig);

    /// The configuration must describe every map channel and the barometer loop usably.
    #[test]
    fn rbpf_refuses_a_configuration_it_cannot_carry() {
        let build = |edit: &ConfigEdit| {
            let mut config = RbpfConfig {
                num_particles: 8,
                ..map_config(2)
            };
            edit(&mut config);
            RaoBlackwellizedParticleFilter::new(StrapdownState::default(), config)
        };
        assert!(build(&|_| {}).is_ok());
        let cases: [(&ConfigEdit, &str); 10] = [
            (&|c| c.map_bias_initial = vec![0.0], "map_bias_initial"),
            (&|c| c.map_bias_init_std = vec![1.0], "map_bias_init_std"),
            (&|c| c.map_variation_std = vec![0.1; 3], "map_variation_std"),
            (
                &|c| c.map_bias_init_std = vec![1.0, -1.0],
                "map_bias_init_std",
            ),
            (
                &|c| c.map_variation_std = vec![0.1, f64::NAN],
                "map_variation_std",
            ),
            (
                &|c| c.map_bias_initial = vec![f64::INFINITY, 0.0],
                "map_bias_initial",
            ),
            (
                &|c| c.map_variation_time_constant_s = vec![300.0, 0.0],
                "map_variation_time_constant_s",
            ),
            (
                &|c| c.baro_loop_time_constant_s = 0.0,
                "baro_loop_time_constant_s",
            ),
            (&|c| c.baro_error_std_m = -1.0, "baro_error_std_m"),
            (
                &|c| c.baro_error_time_constant_s = f64::NAN,
                "baro_error_time_constant_s",
            ),
        ];
        for (edit, expected_field) in cases {
            match build(edit) {
                Err(StrapdownError::InvalidConfiguration { field, .. }) => {
                    assert_eq!(field, expected_field);
                }
                other => panic!("expected `{expected_field}` to be refused, got {other:?}"),
            }
        }
        // An infinite correlation time is a random constant, not an error.
        assert!(build(&|c| c.map_variation_time_constant_s = vec![f64::INFINITY; 2]).is_ok());
        assert!(build(&|c| c.baro_error_time_constant_s = f64::INFINITY).is_ok());
    }

    /// ZARU observes gyro biases, which this filter does not carry, so it is refused rather
    /// than applied as a no-op.
    #[test]
    fn zaru_is_refused() {
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            spread_test_nominal_state(),
            RbpfConfig {
                num_particles: 16,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        let result = rbpf.update(&ZaruMeasurement::from_gyro([0.0, 0.0, 1e-3]));
        assert!(
            matches!(result, Err(StrapdownError::DimensionMismatch { .. })),
            "{result:?}"
        );
    }

    /// A zero position innovation covariance -- the paper's zero `Qⁿ` with nothing uncertain to
    /// couple into position -- propagates without NaN and without drawing any noise.
    ///
    /// The particles still move deterministically through `F`, so "no noise" is checked the
    /// sharp way: two filters holding identical particles but seeded differently must end
    /// bit-identical, which they can only do if every draw was multiplied by a zero root.
    #[test]
    fn a_degenerate_position_innovation_propagates_cleanly() {
        let build = |seed: u64| {
            RaoBlackwellizedParticleFilter::new(
                spread_test_nominal_state(),
                RbpfConfig {
                    num_particles: 32,
                    position_init_std_m: Vector3::new(10.0, 10.0, 0.0),
                    velocity_init_std_mps: 0.0,
                    attitude_init_std_rad: 0.0,
                    velocity_process_noise_std_mps: 0.0,
                    attitude_process_noise_std_rad: 0.0,
                    seed,
                    ..RbpfConfig::default()
                },
            )
            .unwrap()
        };
        let mut first = build(2);
        let mut second = build(3);
        second.particles.clone_from(&first.particles);
        for _ in 0..10 {
            first.predict(&stationary_imu(), 0.1).unwrap();
            second.predict(&stationary_imu(), 0.1).unwrap();
            first.flush_time_update().unwrap();
            second.flush_time_update().unwrap();
        }
        for (a, b) in first.particles.iter().zip(&second.particles) {
            assert!(a.linear_state.iter().all(|v| v.is_finite()));
            assert_eq!(a.position_error, b.position_error, "noise was drawn");
            assert_eq!(a.linear_state, b.linear_state, "noise was drawn");
        }
        assert!(first.linear_covariance.iter().all(|v| v.is_finite()));

        let (inverse, root) =
            position_innovation_factors(&DMatrix::zeros(2, 2), Vector2::new(1.6e-7, 3.1e-7))
                .unwrap();
        assert!(inverse.iter().chain(root.iter()).all(|v| *v == 0.0));
        let nonfinite = DMatrix::from_element(2, 2, f64::NAN);
        assert!(position_innovation_factors(&nonfinite, Vector2::new(1.0, 1.0)).is_err());
    }

    /// The factors of `N` are a pseudo-inverse and a square root of it, at the ~1e-22 rad^2
    /// scale the time update meets them at.
    #[test]
    fn the_position_innovation_factors_invert_and_root_a_tiny_covariance() {
        let radians_per_meter = Vector2::new(1.57e-7, 2.2e-7);
        let meters = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
        let scale =
            DMatrix::from_diagonal(&DVector::from_column_slice(radians_per_meter.as_slice()));
        let n = &scale * meters * &scale;
        let (inverse, root) = position_innovation_factors(&n, radians_per_meter).unwrap();
        assert!((&inverse * &n - DMatrix::<f64>::identity(2, 2)).amax() < 1e-9);
        assert!((&root * root.transpose() - &n).amax() < 1e-9 * n.amax());
    }

    #[test]
    fn rbpf_updates_and_normalizes_weights() {
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                attitude: Rotation3::identity(),
                is_enu: true,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 100,
                position_init_std_m: Vector3::new(5.0, 5.0, 2.0),
                seed: 7,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        rbpf.update(&GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        })
        .unwrap();
        let weight_sum: f64 = rbpf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);
        let (mean, _) = rbpf.estimate();
        assert!(mean.iter().take(3).all(|v| v.is_finite()));
    }

    /// How many standard deviations of its own posterior the filter's final error may be.
    const CONSISTENCY_SIGMAS: f64 = 3.0;

    /// The reported position covariance as (north, east, up) standard deviations in ground
    /// metres, converted with the WGS84 radii at the estimate.
    fn posterior_position_std_m(mean: &DVector<f64>, cov: &DMatrix<f64>) -> Vector3<f64> {
        let latitude_rad = mean[0];
        let altitude_m = mean[2];
        let (meridian_radius, transverse_radius, _) =
            earth::principal_radii(&latitude_rad.to_degrees(), &altitude_m);
        Vector3::new(
            cov[(0, 0)].sqrt() * (meridian_radius + altitude_m),
            cov[(1, 1)].sqrt() * (transverse_radius + altitude_m) * latitude_rad.cos(),
            cov[(2, 2)].sqrt(),
        )
    }

    /// Run one of the three scenarios -- stationary, or 10 m/s north or east -- with a GNSS fix
    /// every sample, from a nominal displaced by `offset_m` metres, and assert the solution is
    /// within three sigma of the posterior the filter reports, and that the posterior has not
    /// collapsed (#295).
    fn run_scenario(velocity_north: f64, velocity_east: f64, offset_m: f64) {
        let lat_deg: f64 = 40.0;
        let lon_deg: f64 = -105.0;
        let alt_m: f64 = 1000.0;
        let g = earth::gravity(&lat_deg, &alt_m);
        let initial_state = StrapdownState {
            latitude: lat_deg.to_radians(),
            longitude: lon_deg.to_radians(),
            altitude: alt_m,
            velocity_north,
            velocity_east,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };
        let sample_rate_hz = 5;
        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            300,
            sample_rate_hz,
            Vector3::new(0.0, 0.0, g),
            Vector3::zeros(),
            true,
            true,
            false,
        );
        let mut nominal = initial_state;
        let delta = (offset_m * earth::METERS_TO_DEGREES).to_radians();
        nominal.latitude += delta;
        nominal.longitude -= delta;

        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            nominal,
            RbpfConfig {
                num_particles: 2_000,
                position_init_std_m: Vector3::new(10.0, 10.0, 5.0),
                seed: 123,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        let dt = 1.0 / sample_rate_hz as f64;
        for (imu, gps) in imu_data.iter().zip(&gps_measurements) {
            rbpf.predict(imu, dt).unwrap();
            rbpf.update(gps).unwrap();
        }
        let weight_sum: f64 = rbpf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);

        let (mean, cov) = rbpf.estimate();
        let truth = true_states.last().unwrap();
        if velocity_north > 0.0 {
            assert!(mean[0] > initial_state.latitude);
        }
        if velocity_east > 0.0 {
            assert!(mean[1] > initial_state.longitude);
        }
        let reported = posterior_position_std_m(&mean, &cov);
        for (axis, name) in [(0, "north"), (1, "east"), (2, "up")] {
            assert!(
                reported[axis] > 0.05,
                "reported {name} sigma {:.3e} m: the posterior has collapsed (#295)",
                reported[axis]
            );
        }
        let horizontal_error_m =
            earth::haversine_distance(mean[0], mean[1], truth.latitude, truth.longitude);
        let altitude_error_m = (mean[2] - truth.altitude).abs();
        let velocity_error_mps = ((mean[3] - truth.velocity_north).powi(2)
            + (mean[4] - truth.velocity_east).powi(2)
            + (mean[5] - truth.velocity_vertical).powi(2))
        .sqrt();
        let max_horizontal_error_m = CONSISTENCY_SIGMAS * reported[0].hypot(reported[1]);
        assert!(
            horizontal_error_m <= max_horizontal_error_m,
            "horizontal error {horizontal_error_m:.3} m exceeds 3 sigma ({max_horizontal_error_m:.3} m)"
        );
        assert!(
            altitude_error_m <= CONSISTENCY_SIGMAS * reported[2],
            "altitude error {altitude_error_m:.3} m exceeds 3 sigma ({:.3} m)",
            CONSISTENCY_SIGMAS * reported[2]
        );
        assert!(
            velocity_error_mps <= 1.0,
            "velocity error {velocity_error_mps:.3} m/s"
        );
    }

    #[test]
    fn rbpf_runs_on_scenario_stationary() {
        run_scenario(0.0, 0.0, 20.0);
    }

    #[test]
    fn rbpf_runs_on_scenario_constant_velocity_north() {
        run_scenario(10.0, 0.0, -25.0);
    }

    #[test]
    fn rbpf_runs_on_scenario_constant_velocity_east() {
        run_scenario(0.0, 10.0, 25.0);
    }
}
