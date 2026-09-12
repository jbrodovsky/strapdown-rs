//! High-level inertial navigation engine.
//!
//! [`InsEngine`] is the entry point most users should reach for. It owns a
//! [`NavigationFilter`] (the 15-state [`ErrorStateKalmanFilter`] by default), advances it at
//! the IMU rate, corrects it whenever an aiding measurement arrives, and reports the current
//! estimate as a [`NavSolution`] in the units a consumer of a navigation solution expects --
//! degrees, metres and metres per second rather than the radians the filter state carries.
//!
//! ```rust
//! use strapdown::engine::{GnssFix, InsEngine};
//! use strapdown::{ImuSample, kalman::InitialState};
//! use nalgebra::Vector3;
//!
//! # fn main() -> Result<(), strapdown::StrapdownError> {
//! let mut engine = InsEngine::builder()
//!     .with_initial_state(InitialState::new(
//!         40.0, -75.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, None,
//!     ))
//!     .with_lever_arm([1.5, 0.0, -0.8])
//!     .build()?;
//!
//! // High-rate inertial propagation.
//! let sample = ImuSample::from_rates(
//!     &strapdown::IMUData { accel: Vector3::new(0.0, 0.0, -9.81), gyro: Vector3::zeros() },
//!     0.01,
//! );
//! for _ in 0..100 {
//!     engine.predict(&sample)?;
//! }
//!
//! // Low-rate, asynchronous aiding.
//! engine.update_gnss(&GnssFix::position(40.0, -75.0, 100.0, 5.0, 10.0))?;
//!
//! let solution = engine.nav_solution();
//! assert!((solution.latitude - 40.0).abs() < 1e-3);
//! # Ok(())
//! # }
//! ```
//!
//! # Antenna lever arm
//!
//! A GNSS antenna is bolted somewhere on the vehicle and an IMU somewhere else. The filter
//! estimates the IMU's position, but the receiver reports the antenna's, and the two differ by
//! the body-frame offset $r_{ant}^b$ rotated into the navigation frame. Feeding the raw fix
//! to the filter therefore injects a standing position error equal to that offset, plus a
//! velocity error whenever the vehicle rotates.
//!
//! $$
//! p_{GNSS} = p_{IMU} + C_b^n r_{ant}^b
//! $$
//!
//! $$
//! v_{GNSS} = v_{IMU} + C_b^n \left( \omega_{ib}^b \times r_{ant}^b \right)
//! $$
//!
//! [`InsEngine::update_gnss`] inverts both relations before handing the measurement to the
//! filter, so the filter only ever sees quantities referred to the IMU centre. The rotation
//! comes from the current attitude estimate and $\omega_{ib}^b$ from the most recent
//! [`predict`](InsEngine::predict), bias-corrected with the filter's own gyro bias estimate.
//!
//! The lever arm is measured **from the IMU centre to the antenna phase centre, in the body
//! frame**, in metres. Its axes are the IMU's own: x forward, y right, z down for the
//! usual aerospace body frame. A zero lever arm disables the compensation exactly (it is an
//! identity, not an approximation), which is the default.
//!
//! # Frame convention
//!
//! The engine is NED by default, like the rest of the crate. ENU is available by setting
//! [`InsEngineConfig::is_enu`] or by supplying an ENU [`InitialState`]; supplying both and
//! disagreeing is rejected at [`build`](InsEngineBuilder::build) rather than silently
//! resolved, because a mismatched frame is the failure mode this crate has lost the most
//! time to (see issues #266, #286 and #296).

use std::fmt::{self, Debug, Display};

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use serde::{Deserialize, Serialize};

use crate::earth::principal_radii;
use crate::kalman::{ErrorStateKalmanFilter, InitialState};
use crate::measurements::{GPSPositionMeasurement, GPSVelocityMeasurement, MeasurementModel};
// `DEFAULT_PROCESS_NOISE` is a tuning constant for the 15-state filters rather than a
// simulation-only value; it lives in `sim` for historical reasons. Reusing it here keeps the
// engine's default tuning identical to the one the ESKF integration suite validates.
use crate::gating::{InnovationGate, UpdateOutcome};
use crate::sim::DEFAULT_PROCESS_NOISE;
use crate::{ImuSample, InputModel, NavigationFilter, StrapdownError};

/// Number of states in the default 15-state error-state filter.
const FULL_STATE_DIMENSION: usize = 15;
/// Number of states any filter driven by this engine must expose at minimum.
///
/// Position, velocity and attitude. Bias states are optional and reported only when present.
const MINIMUM_STATE_DIMENSION: usize = 9;
/// Largest lever arm the builder will accept, in metres.
///
/// Generous for any vehicle the body-frame small-offset model applies to, and small enough
/// that a value entered in the wrong unit (centimetres read as metres, or a latitude pasted
/// into the wrong field) is rejected rather than quietly biasing every fix.
const MAX_LEVER_ARM_M: f64 = 100.0;

/// Configuration for an [`InsEngine`].
///
/// Serialisable so a scenario file can describe an engine end to end. Every field has a
/// default, so `InsEngineConfig::default()` is a valid NED engine with no lever arm and the
/// crate's default process noise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InsEngineConfig {
    /// Local-level frame convention: `false` (the default) is NED, `true` is ENU.
    pub is_enu: bool,
    /// Body-frame offset from the IMU centre to the GNSS antenna phase centre, metres.
    ///
    /// `[0.0, 0.0, 0.0]` (the default) disables lever-arm compensation.
    pub lever_arm: [f64; 3],
    /// Process noise covariance diagonal, 15 elements. `None` uses
    /// [`DEFAULT_PROCESS_NOISE`].
    pub process_noise_diagonal: Option<Vec<f64>>,
    /// Initial error-state covariance diagonal, 15 elements. `None` uses the same defaults
    /// as [`crate::sim::initialize_eskf`].
    pub initial_covariance_diagonal: Option<Vec<f64>>,
    /// Initial accelerometer bias estimate, m/s^2, body frame.
    pub initial_accel_bias: [f64; 3],
    /// Initial gyroscope bias estimate, rad/s, body frame.
    pub initial_gyro_bias: [f64; 3],
}

impl Default for InsEngineConfig {
    fn default() -> Self {
        Self {
            is_enu: false,
            lever_arm: [0.0; 3],
            process_noise_diagonal: None,
            initial_covariance_diagonal: None,
            initial_accel_bias: [0.0; 3],
            initial_gyro_bias: [0.0; 3],
        }
    }
}

/// Default initial error-state covariance diagonal for the 15-state ESKF.
///
/// Matches [`crate::sim::initialize_eskf`]: position and velocity error variances, then
/// attitude, then accelerometer and gyro bias.
const DEFAULT_INITIAL_COVARIANCE: [f64; FULL_STATE_DIMENSION] = [
    1e-6, 1e-6, 1e-4, // position error (rad^2, rad^2, m^2)
    1e-3, 1e-3, 1e-3, // velocity error (m^2/s^2)
    1e-5, 1e-5, 1e-5, // attitude error (rad^2)
    1e-6, 1e-6, 1e-6, // accelerometer bias error
    1e-8, 1e-8, 1e-8, // gyroscope bias error
];

/// Builder for [`InsEngine`].
///
/// Every setter is optional; `InsEngine::builder().build()` yields a NED engine at the null
/// island origin with no lever arm, which is useful mostly as a starting point for tests.
///
/// # Example
///
/// ```rust
/// use strapdown::engine::{InsEngine, InsEngineConfig};
/// use strapdown::kalman::InitialState;
///
/// # fn main() -> Result<(), strapdown::StrapdownError> {
/// let config = InsEngineConfig {
///     lever_arm: [2.0, 0.0, -1.0],
///     ..InsEngineConfig::default()
/// };
/// let engine = InsEngine::builder()
///     .with_config(config)
///     .with_initial_state(InitialState::new(
///         51.5, -0.1, 35.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, None,
///     ))
///     .build()?;
///
/// assert_eq!(engine.lever_arm(), nalgebra::Vector3::new(2.0, 0.0, -1.0));
/// # Ok(())
/// # }
/// ```
pub struct InsEngineBuilder {
    config: InsEngineConfig,
    initial_state: Option<InitialState>,
    filter: Option<Box<dyn NavigationFilter>>,
    /// Whether the caller named a frame, as opposed to inheriting the NED default.
    ///
    /// Needed to tell "the user asked for NED" from "the user said nothing", which is what
    /// makes the frame-conflict check in [`InsEngineBuilder::build`] fire only on a genuine
    /// contradiction.
    frame_set: bool,
}

impl Debug for InsEngineBuilder {
    /// `NavigationFilter` deliberately has no `Debug` supertrait -- requiring one would force
    /// it on every downstream filter -- so a caller-supplied filter is elided.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InsEngineBuilder")
            .field("config", &self.config)
            .field("initial_state", &self.initial_state)
            .field(
                "filter",
                &self.filter.as_ref().map(|_| "<dyn NavigationFilter>"),
            )
            .field("frame_set", &self.frame_set)
            .finish()
    }
}

impl Default for InsEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InsEngineBuilder {
    /// Start from the default configuration: NED, no lever arm, default noise.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: InsEngineConfig::default(),
            initial_state: None,
            filter: None,
            frame_set: false,
        }
    }

    /// Replace the whole configuration.
    ///
    /// Counts as naming a frame, so a subsequent [`Self::with_initial_state`] whose frame
    /// disagrees with `config.is_enu` is a build error rather than a silent override.
    #[must_use]
    pub fn with_config(mut self, config: InsEngineConfig) -> Self {
        self.config = config;
        self.frame_set = true;
        self
    }

    /// Set the initial navigation state.
    ///
    /// When no frame was named separately, the state's own `is_enu` becomes the engine's
    /// frame.
    #[must_use]
    pub const fn with_initial_state(mut self, initial_state: InitialState) -> Self {
        if !self.frame_set {
            self.config.is_enu = initial_state.is_enu;
        }
        self.initial_state = Some(initial_state);
        self
    }

    /// Set the local-level frame convention explicitly: `true` for ENU, `false` for NED.
    #[must_use]
    pub const fn with_frame(mut self, is_enu: bool) -> Self {
        self.config.is_enu = is_enu;
        self.frame_set = true;
        self
    }

    /// Set the body-frame antenna offset in metres, IMU centre to antenna phase centre.
    #[must_use]
    pub const fn with_lever_arm(mut self, lever_arm: [f64; 3]) -> Self {
        self.config.lever_arm = lever_arm;
        self
    }

    /// Set the process noise covariance diagonal (15 elements).
    #[must_use]
    pub fn with_process_noise(mut self, diagonal: Vec<f64>) -> Self {
        self.config.process_noise_diagonal = Some(diagonal);
        self
    }

    /// Set the initial error-state covariance diagonal (15 elements).
    #[must_use]
    pub fn with_initial_covariance(mut self, diagonal: Vec<f64>) -> Self {
        self.config.initial_covariance_diagonal = Some(diagonal);
        self
    }

    /// Set the initial IMU bias estimates: accelerometer in m/s^2, gyro in rad/s.
    #[must_use]
    pub const fn with_imu_biases(mut self, accel_bias: [f64; 3], gyro_bias: [f64; 3]) -> Self {
        self.config.initial_accel_bias = accel_bias;
        self.config.initial_gyro_bias = gyro_bias;
        self
    }

    /// Drive a caller-supplied filter instead of the default 15-state ESKF.
    ///
    /// The filter is taken as built: the initial state, covariance, process-noise and bias
    /// fields of the configuration are *not* applied to it, since they were already spent
    /// when the caller constructed it. The frame and lever arm still apply -- those belong to
    /// the vehicle, not the filter.
    ///
    /// The filter's state vector must be at least nine elements
    /// (`[lat, lon, alt, v_n, v_e, v_v, roll, pitch, yaw]`); bias states are reported in the
    /// [`NavSolution`] only when it carries all fifteen.
    #[must_use]
    pub fn with_filter(mut self, filter: Box<dyn NavigationFilter>) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Build the engine.
    ///
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] if the lever arm is non-finite or longer than
    /// 100 m, if a supplied noise or covariance diagonal is not 15 non-negative finite
    /// elements, if a supplied initial state's frame contradicts an explicitly named frame,
    /// or if a caller-supplied filter exposes fewer than nine states.
    pub fn build(self) -> Result<InsEngine, StrapdownError> {
        let lever_arm = validate_lever_arm(&self.config.lever_arm)?;
        let is_enu = self.resolve_frame()?;

        let filter = match self.filter {
            Some(filter) => filter,
            None => Box::new(self.build_default_filter(is_enu)?),
        };

        let state_dimension = filter.get_estimate().len();
        if state_dimension < MINIMUM_STATE_DIMENSION {
            return Err(StrapdownError::InvalidConfiguration {
                field: "filter",
                reason: format!(
                    "state vector must have at least {MINIMUM_STATE_DIMENSION} elements, got {state_dimension}"
                ),
            });
        }

        Ok(InsEngine {
            filter,
            lever_arm,
            is_enu,
            state_dimension,
            elapsed_s: 0.0,
            angular_rate: Vector3::zeros(),
        })
    }

    /// Reconcile the configured frame with a supplied initial state's frame.
    fn resolve_frame(&self) -> Result<bool, StrapdownError> {
        let configured = self.config.is_enu;
        if let Some(state) = self.initial_state.as_ref()
            && self.frame_set
            && state.is_enu != configured
        {
            return Err(StrapdownError::InvalidConfiguration {
                field: "is_enu",
                reason: format!(
                    "configuration selects {} but the initial state is {}; \
                     set one or the other, not both",
                    frame_name(configured),
                    frame_name(state.is_enu)
                ),
            });
        }
        Ok(configured)
    }

    /// Construct the default 15-state ESKF from the configuration.
    fn build_default_filter(&self, is_enu: bool) -> Result<ErrorStateKalmanFilter, StrapdownError> {
        let process_noise = validate_diagonal(
            self.config.process_noise_diagonal.as_deref(),
            &DEFAULT_PROCESS_NOISE,
            "process_noise_diagonal",
        )?;
        let initial_covariance = validate_diagonal(
            self.config.initial_covariance_diagonal.as_deref(),
            &DEFAULT_INITIAL_COVARIANCE,
            "initial_covariance_diagonal",
        )?;

        let mut initial_state = self.initial_state.clone().unwrap_or_default();
        initial_state.is_enu = is_enu;

        let biases = [
            self.config.initial_accel_bias[0],
            self.config.initial_accel_bias[1],
            self.config.initial_accel_bias[2],
            self.config.initial_gyro_bias[0],
            self.config.initial_gyro_bias[1],
            self.config.initial_gyro_bias[2],
        ];

        Ok(ErrorStateKalmanFilter::new(
            &initial_state,
            &biases,
            initial_covariance,
            DMatrix::from_diagonal(&DVector::from_vec(process_noise)),
        ))
    }
}

/// Human-readable frame name, for error messages.
const fn frame_name(is_enu: bool) -> &'static str {
    if is_enu { "ENU" } else { "NED" }
}

/// Check a lever arm is finite and of a plausible magnitude.
fn validate_lever_arm(lever_arm: &[f64; 3]) -> Result<Vector3<f64>, StrapdownError> {
    let vector = Vector3::new(lever_arm[0], lever_arm[1], lever_arm[2]);
    if !vector.iter().all(|component| component.is_finite()) {
        return Err(StrapdownError::InvalidConfiguration {
            field: "lever_arm",
            reason: "components must be finite".to_string(),
        });
    }
    let magnitude = vector.norm();
    if magnitude > MAX_LEVER_ARM_M {
        return Err(StrapdownError::InvalidConfiguration {
            field: "lever_arm",
            reason: format!(
                "magnitude {magnitude:.3} m exceeds the {MAX_LEVER_ARM_M} m limit; \
                 the offset is measured in metres in the body frame"
            ),
        });
    }
    Ok(vector)
}

/// Validate an optional covariance diagonal, falling back to a default.
fn validate_diagonal(
    supplied: Option<&[f64]>,
    fallback: &[f64; FULL_STATE_DIMENSION],
    field: &'static str,
) -> Result<Vec<f64>, StrapdownError> {
    let Some(values) = supplied else {
        return Ok(fallback.to_vec());
    };
    if values.len() != FULL_STATE_DIMENSION {
        return Err(StrapdownError::InvalidConfiguration {
            field,
            reason: format!(
                "expected {FULL_STATE_DIMENSION} elements, got {}",
                values.len()
            ),
        });
    }
    if let Some(bad) = values.iter().find(|v| !v.is_finite() || **v < 0.0) {
        return Err(StrapdownError::InvalidConfiguration {
            field,
            reason: format!("every element must be finite and non-negative, found {bad}"),
        });
    }
    Ok(values.to_vec())
}

// ============================== Aiding input =================================================

/// A GNSS fix as reported by a receiver, referred to the **antenna**.
///
/// [`InsEngine::update_gnss`] moves it to the IMU centre before the filter sees it, so the
/// values here are the ones a receiver actually emits and need no pre-processing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GnssFix {
    /// Latitude, degrees.
    pub latitude: f64,
    /// Longitude, degrees.
    pub longitude: f64,
    /// Altitude above the WGS84 ellipsoid, metres.
    pub altitude: f64,
    /// Ground velocity `[north, east, vertical]` in m/s, or `None` for a position-only fix.
    ///
    /// The vertical component follows the engine's frame: positive down in NED, positive up
    /// in ENU.
    pub velocity: Option<[f64; 3]>,
    /// One-sigma horizontal position accuracy, metres.
    pub horizontal_noise_std: f64,
    /// One-sigma vertical position accuracy, metres.
    pub vertical_noise_std: f64,
    /// One-sigma velocity accuracy, m/s. Ignored when `velocity` is `None`.
    pub velocity_noise_std: f64,
}

impl GnssFix {
    /// A position-only fix.
    #[must_use]
    pub const fn position(
        latitude: f64,
        longitude: f64,
        altitude: f64,
        horizontal_noise_std: f64,
        vertical_noise_std: f64,
    ) -> Self {
        Self {
            latitude,
            longitude,
            altitude,
            velocity: None,
            horizontal_noise_std,
            vertical_noise_std,
            velocity_noise_std: 0.0,
        }
    }

    /// Add a velocity to a fix. `velocity` is `[north, east, vertical]` in m/s.
    #[must_use]
    pub const fn with_velocity(mut self, velocity: [f64; 3], velocity_noise_std: f64) -> Self {
        self.velocity = Some(velocity);
        self.velocity_noise_std = velocity_noise_std;
        self
    }

    /// Reject a fix whose numbers cannot be used, before it reaches the filter.
    fn validate(&self) -> Result<(), StrapdownError> {
        let finite = self.latitude.is_finite()
            && self.longitude.is_finite()
            && self.altitude.is_finite()
            && self
                .velocity
                .is_none_or(|v| v.iter().all(|c| c.is_finite()));
        if !finite {
            return Err(StrapdownError::MeasurementUnavailable {
                model: "GnssFix",
                reason: "fix contains a non-finite component".to_string(),
            });
        }
        if !(-90.0..=90.0).contains(&self.latitude) {
            return Err(StrapdownError::OutOfRange {
                what: "GnssFix latitude (deg)",
                value: self.latitude,
                min: -90.0,
                max: 90.0,
            });
        }
        Ok(())
    }
}

impl Display for GnssFix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GnssFix(lat: {:.7} deg, lon: {:.7} deg, alt: {:.2} m",
            self.latitude, self.longitude, self.altitude
        )?;
        if let Some([north, east, vertical]) = self.velocity {
            write!(f, ", v: [{north:.3}, {east:.3}, {vertical:.3}] m/s")?;
        }
        write!(f, ")")
    }
}

// ============================== Lever-arm geometry ===========================================

/// Resolve a body-frame lever arm into the navigation frame.
///
/// Returns $C_b^n r_{ant}^b$ as `[north, east, vertical]` metres, where the vertical
/// component follows whichever convention `attitude` was built in -- the mechanization's
/// attitude matrix maps the body frame onto (north, east, vertical) in both NED and ENU.
///
/// # Example
/// ```rust
/// use strapdown::engine::lever_arm_position_offset;
/// use nalgebra::{Rotation3, Vector3};
///
/// // Level, heading north: a 2 m forward lever arm points 2 m north.
/// let offset = lever_arm_position_offset(&Rotation3::identity(), &Vector3::new(2.0, 0.0, 0.0));
/// assert!((offset[0] - 2.0).abs() < 1e-12);
/// assert!(offset[1].abs() < 1e-12);
/// ```
#[must_use]
pub fn lever_arm_position_offset(
    attitude: &Rotation3<f64>,
    lever_arm: &Vector3<f64>,
) -> Vector3<f64> {
    attitude * lever_arm
}

/// Velocity the antenna gains from the vehicle's rotation about the IMU centre.
///
/// Returns $C_b^n (\omega_{ib}^b \times r_{ant}^b)$ as `[north, east, vertical]` in m/s.
///
/// `angular_rate` is the body-frame inertial angular rate in rad/s, bias-corrected. The
/// earth and transport rates are left in it deliberately: at a 100 m lever arm the earth rate
/// contributes under 8 mm/s, well inside GNSS velocity noise, and removing them would make
/// this function depend on position for no measurable gain.
///
/// # Example
/// ```rust
/// use strapdown::engine::lever_arm_velocity_offset;
/// use nalgebra::{Rotation3, Vector3};
///
/// // Yawing at 0.1 rad/s with the antenna 2 m forward: the antenna sweeps east at 0.2 m/s.
/// let offset = lever_arm_velocity_offset(
///     &Rotation3::identity(),
///     &Vector3::new(0.0, 0.0, 0.1),
///     &Vector3::new(2.0, 0.0, 0.0),
/// );
/// assert!((offset[1] - 0.2).abs() < 1e-12);
/// ```
#[must_use]
pub fn lever_arm_velocity_offset(
    attitude: &Rotation3<f64>,
    angular_rate: &Vector3<f64>,
    lever_arm: &Vector3<f64>,
) -> Vector3<f64> {
    attitude * angular_rate.cross(lever_arm)
}

/// Shift a geodetic position by a local-level offset in metres.
///
/// `latitude` and `longitude` are radians, `altitude` metres. `offset` is
/// `[north, east, vertical]` in metres and is **subtracted**, which is the direction that
/// takes an antenna-referred fix back to the IMU centre. The metre-to-angle conversion uses
/// the WGS84 principal radii at `latitude`, matching the filter's own position Jacobians.
///
/// # Example
/// ```rust
/// use strapdown::engine::shift_position_by_offset;
/// use nalgebra::Vector3;
///
/// // An antenna 100 m north of the IMU: the IMU is south of the reported fix.
/// let (lat, _lon, _alt) = shift_position_by_offset(
///     40.0_f64.to_radians(), 0.0, 0.0, &Vector3::new(100.0, 0.0, 0.0), false,
/// );
/// assert!(lat < 40.0_f64.to_radians());
/// ```
#[must_use]
pub fn shift_position_by_offset(
    latitude: f64,
    longitude: f64,
    altitude: f64,
    offset: &Vector3<f64>,
    is_enu: bool,
) -> (f64, f64, f64) {
    // `principal_radii` returns (meridian, prime-vertical, transverse) despite the names;
    // the meridian radius is what converts a northward metre into a latitude radian and the
    // prime vertical is what converts an eastward metre into a longitude radian. This is the
    // same pairing `linearize::error_state_transition_jacobian` uses.
    let (meridian_radius, prime_vertical_radius, _) =
        principal_radii(&latitude.to_degrees(), &altitude);
    let cos_latitude = latitude.cos().abs().max(f64::EPSILON);

    let delta_latitude = offset[0] / (meridian_radius + altitude);
    let delta_longitude = offset[1] / ((prime_vertical_radius + altitude) * cos_latitude);
    // `altitude` is height above the ellipsoid -- positive up -- in both frames, so an NED
    // offset (positive down) has to change sign on its way into an altitude.
    let delta_altitude = if is_enu { offset[2] } else { -offset[2] };

    (
        latitude - delta_latitude,
        longitude - delta_longitude,
        altitude - delta_altitude,
    )
}

// ============================== Navigation solution ==========================================

/// The engine's current estimate, in the units a navigation consumer expects.
///
/// Degrees for latitude, longitude and the Euler angles; metres and metres per second for
/// everything else. The filter state carries radians internally; this is the boundary where
/// that stops being the caller's problem.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NavSolution {
    /// Seconds of inertial propagation since the engine was built.
    pub elapsed_s: f64,
    /// Latitude, degrees.
    pub latitude: f64,
    /// Longitude, degrees.
    pub longitude: f64,
    /// Altitude above the WGS84 ellipsoid, metres.
    pub altitude: f64,
    /// North velocity, m/s.
    pub velocity_north: f64,
    /// East velocity, m/s.
    pub velocity_east: f64,
    /// Vertical velocity, m/s: positive down in NED, positive up in ENU.
    pub velocity_vertical: f64,
    /// Roll, degrees.
    pub roll: f64,
    /// Pitch, degrees.
    pub pitch: f64,
    /// Yaw, degrees.
    pub yaw: f64,
    /// Estimated accelerometer bias, m/s^2, body frame. Zero when the filter carries no
    /// bias states.
    pub accel_bias: [f64; 3],
    /// Estimated gyroscope bias, rad/s, body frame. Zero when the filter carries no bias
    /// states.
    pub gyro_bias: [f64; 3],
    /// One-sigma position uncertainty `[north, east, vertical]` in **metres**, converted
    /// from the filter's radian-valued horizontal covariance.
    pub position_std_m: [f64; 3],
    /// One-sigma velocity uncertainty `[north, east, vertical]` in m/s.
    pub velocity_std_mps: [f64; 3],
    /// Local-level frame convention: `false` is NED, `true` is ENU.
    pub is_enu: bool,
}

impl Display for NavSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NavSolution(t: {:.3} s, {:.7} deg, {:.7} deg, {:.2} m, \
             v: [{:.3}, {:.3}, {:.3}] m/s, rpy: [{:.2}, {:.2}, {:.2}] deg, frame: {})",
            self.elapsed_s,
            self.latitude,
            self.longitude,
            self.altitude,
            self.velocity_north,
            self.velocity_east,
            self.velocity_vertical,
            self.roll,
            self.pitch,
            self.yaw,
            frame_name(self.is_enu),
        )
    }
}

impl NavSolution {
    /// The attitude as a body-to-navigation rotation matrix, $C_b^n$.
    #[must_use]
    pub fn attitude(&self) -> Rotation3<f64> {
        Rotation3::from_euler_angles(
            self.roll.to_radians(),
            self.pitch.to_radians(),
            self.yaw.to_radians(),
        )
    }
}

impl From<&NavSolution> for crate::StrapdownState {
    fn from(solution: &NavSolution) -> Self {
        Self {
            latitude: solution.latitude.to_radians(),
            longitude: solution.longitude.to_radians(),
            altitude: solution.altitude,
            velocity_north: solution.velocity_north,
            velocity_east: solution.velocity_east,
            velocity_vertical: solution.velocity_vertical,
            attitude: solution.attitude(),
            is_enu: solution.is_enu,
        }
    }
}

// ============================== The engine ===================================================

/// A running inertial navigation system.
///
/// Build one with [`InsEngine::builder`], drive it with [`predict`](Self::predict) at the IMU
/// rate, correct it with [`update_gnss`](Self::update_gnss) or [`update`](Self::update)
/// whenever aiding arrives, and read [`nav_solution`](Self::nav_solution) whenever a consumer
/// needs the current estimate.
///
/// Aiding is asynchronous by construction: the engine's clock advances only on `predict`, and
/// an update is applied at whatever epoch the last `predict` left it at. Call `predict` up to
/// a measurement's time of validity before applying it.
pub struct InsEngine {
    filter: Box<dyn NavigationFilter>,
    lever_arm: Vector3<f64>,
    is_enu: bool,
    /// Cached length of the filter's state vector, validated at build to be >= 9.
    state_dimension: usize,
    elapsed_s: f64,
    /// Bias-corrected body-frame angular rate from the most recent `predict`, rad/s.
    ///
    /// Zero until the first `predict`, which makes the velocity lever-arm term vanish rather
    /// than guess -- the correct behaviour when no rotation has been observed yet.
    angular_rate: Vector3<f64>,
}

impl Debug for InsEngine {
    /// The filter is elided: `NavigationFilter` has no `Debug` supertrait, deliberately.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InsEngine")
            .field("filter", &"<dyn NavigationFilter>")
            .field("lever_arm", &self.lever_arm)
            .field("is_enu", &self.is_enu)
            .field("state_dimension", &self.state_dimension)
            .field("elapsed_s", &self.elapsed_s)
            .field("angular_rate", &self.angular_rate)
            .finish()
    }
}

impl Display for InsEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InsEngine {{ {} }}", self.nav_solution())
    }
}

impl InsEngine {
    /// Start building an engine.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::engine::InsEngine;
    /// # fn main() -> Result<(), strapdown::StrapdownError> {
    /// let engine = InsEngine::builder().with_lever_arm([1.0, 0.0, -0.5]).build()?;
    /// assert_eq!(engine.elapsed_s(), 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn builder() -> InsEngineBuilder {
        InsEngineBuilder::new()
    }

    /// Propagate the state forward with one inertial sample.
    ///
    /// This is the high-rate path; call it for every IMU sample. The engine's clock advances
    /// by `sample.dt`, and the sample's bias-corrected angular rate is retained for the
    /// velocity lever-arm term of the next GNSS update.
    ///
    /// # Errors
    /// Whatever the underlying filter's `predict` returns, including
    /// [`StrapdownError::OutOfRange`] for a non-positive `dt`.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::engine::InsEngine;
    /// use strapdown::{IMUData, ImuSample};
    /// use nalgebra::Vector3;
    ///
    /// # fn main() -> Result<(), strapdown::StrapdownError> {
    /// let mut engine = InsEngine::builder().build()?;
    /// let sample = ImuSample::from_rates(
    ///     &IMUData { accel: Vector3::new(0.0, 0.0, -9.81), gyro: Vector3::zeros() },
    ///     0.01,
    /// );
    /// engine.predict(&sample)?;
    /// assert!((engine.elapsed_s() - 0.01).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(&mut self, sample: &ImuSample) -> Result<(), StrapdownError> {
        let input: &dyn InputModel = sample;
        self.filter.predict(input, sample.dt)?;
        self.elapsed_s += sample.dt;
        self.angular_rate = self.bias_corrected_angular_rate(sample);
        Ok(())
    }

    /// Propagate from instantaneous rates rather than increments.
    ///
    /// Convenience over [`predict`](Self::predict) for data sources that report rates; the
    /// conversion is [`ImuSample::from_rates`].
    ///
    /// # Errors
    /// As [`predict`](Self::predict).
    pub fn predict_rates(&mut self, imu: &crate::IMUData, dt: f64) -> Result<(), StrapdownError> {
        self.predict(&ImuSample::from_rates(imu, dt))
    }

    /// Correct the state with a GNSS fix reported at the antenna.
    ///
    /// The fix is moved to the IMU centre using the current attitude estimate and the
    /// configured lever arm before the filter sees it; with a zero lever arm this is exactly
    /// an identity. Position and velocity are applied as two sequential updates, so a
    /// position-only fix costs nothing extra.
    ///
    /// # Errors
    /// [`StrapdownError::MeasurementUnavailable`] or [`StrapdownError::OutOfRange`] if the
    /// fix is malformed, otherwise whatever the underlying filter's `update` returns.
    ///
    /// # Returns
    /// The [`UpdateOutcome`] of the *position* leg. When a gate is installed and that leg is
    /// rejected, the velocity leg is skipped too and the state is left untouched.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::engine::{GnssFix, InsEngine};
    /// use strapdown::kalman::InitialState;
    ///
    /// # fn main() -> Result<(), strapdown::StrapdownError> {
    /// let mut engine = InsEngine::builder()
    ///     .with_initial_state(InitialState::new(
    ///         40.0, -75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, None,
    ///     ))
    ///     .build()?;
    /// let fix = GnssFix::position(40.0001, -75.0, 0.0, 5.0, 10.0)
    ///     .with_velocity([1.0, 0.0, 0.0], 0.2);
    /// engine.update_gnss(&fix)?;
    /// assert!(engine.nav_solution().latitude > 40.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_gnss(&mut self, fix: &GnssFix) -> Result<UpdateOutcome, StrapdownError> {
        fix.validate()?;
        let solution = self.nav_solution();
        let attitude = solution.attitude();

        let position_offset = lever_arm_position_offset(&attitude, &self.lever_arm);
        let (latitude, longitude, altitude) = shift_position_by_offset(
            fix.latitude.to_radians(),
            fix.longitude.to_radians(),
            fix.altitude,
            &position_offset,
            self.is_enu,
        );

        let outcome = self.filter.update(&GPSPositionMeasurement {
            latitude: latitude.to_degrees(),
            longitude: longitude.to_degrees(),
            altitude,
            horizontal_noise_std: fix.horizontal_noise_std,
            vertical_noise_std: fix.vertical_noise_std,
        })?;

        // A fix whose position the gate disbelieved has no more credible velocity: both
        // legs come from the same receiver at the same epoch, and a multipath or spoofed
        // fix corrupts them together. Applying the velocity anyway would let exactly the
        // measurement the gate just rejected back into the state through the other door.
        if !outcome.accepted {
            return Ok(outcome);
        }

        if let Some(velocity) = fix.velocity {
            let velocity_offset =
                lever_arm_velocity_offset(&attitude, &self.angular_rate, &self.lever_arm);
            self.filter.update(&GPSVelocityMeasurement {
                northward_velocity: velocity[0] - velocity_offset[0],
                eastward_velocity: velocity[1] - velocity_offset[1],
                vertical_velocity: velocity[2] - velocity_offset[2],
                horizontal_noise_std: fix.velocity_noise_std,
                vertical_noise_std: fix.velocity_noise_std,
            })?;
        }
        Ok(outcome)
    }

    /// Correct the state with any other measurement model.
    ///
    /// The escape hatch for aiding the engine has no dedicated method for -- barometric
    /// altitude, magnetometer yaw, a geophysical anomaly. No lever-arm compensation is
    /// applied: the offset is specific to the GNSS antenna, and any other sensor's mounting
    /// geometry belongs to its own measurement model.
    ///
    /// # Errors
    /// Whatever the underlying filter's `update` returns. A
    /// [recoverable](StrapdownError::is_recoverable) error means this measurement should be
    /// skipped and the run continued.
    ///
    /// # Returns
    /// The [`UpdateOutcome`], so a caller can see the NIS the update was judged on and
    /// whether the correction was applied. An outcome with `accepted == false` is not an
    /// error: the gate did what it was configured to do and the state is unchanged.
    pub fn update(
        &mut self,
        measurement: &dyn MeasurementModel,
    ) -> Result<UpdateOutcome, StrapdownError> {
        self.filter.update(measurement)
    }

    /// Install (or clear, with `None`) the innovation gate the filter applies to every
    /// measurement.
    ///
    /// Off by default, matching the filters themselves. See
    /// [`InnovationGate`] for what the two variants mean and
    /// why a chi-squared gate is usually the right one.
    ///
    /// # Returns
    /// `true` if the underlying filter honours the gate. Every filter in
    /// [`kalman`](crate::kalman) and the RBPF do.
    pub fn set_innovation_gate(&mut self, gate: Option<InnovationGate>) -> bool {
        self.filter.set_innovation_gate(gate)
    }

    /// The current estimate, in degrees, metres and m/s.
    #[must_use]
    pub fn nav_solution(&self) -> NavSolution {
        let state = self.filter.get_estimate();
        let covariance = self.filter.get_certainty();
        let (accel_bias, gyro_bias) = if self.state_dimension >= FULL_STATE_DIMENSION {
            (
                [state[9], state[10], state[11]],
                [state[12], state[13], state[14]],
            )
        } else {
            ([0.0; 3], [0.0; 3])
        };

        NavSolution {
            elapsed_s: self.elapsed_s,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_vertical: state[5],
            roll: state[6].to_degrees(),
            pitch: state[7].to_degrees(),
            yaw: state[8].to_degrees(),
            accel_bias,
            gyro_bias,
            position_std_m: position_std_m(&state, &covariance),
            velocity_std_mps: velocity_std_mps(&covariance),
            is_enu: self.is_enu,
        }
    }

    /// The filter's covariance matrix, in the filter's own units.
    ///
    /// Radians for the horizontal position states; see [`NavSolution::position_std_m`] for
    /// the metre-valued summary most callers want.
    ///
    /// [`NavSolution::position_std_m`]: NavSolution#structfield.position_std_m
    #[must_use]
    pub fn covariance(&self) -> DMatrix<f64> {
        self.filter.get_certainty()
    }

    /// Seconds of inertial propagation since the engine was built.
    #[must_use]
    pub const fn elapsed_s(&self) -> f64 {
        self.elapsed_s
    }

    /// The configured body-frame antenna offset, metres.
    #[must_use]
    pub const fn lever_arm(&self) -> Vector3<f64> {
        self.lever_arm
    }

    /// Replace the antenna offset on a running engine.
    ///
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] on the same grounds as
    /// [`InsEngineBuilder::build`]: non-finite components or a magnitude over 100 m.
    pub fn set_lever_arm(&mut self, lever_arm: [f64; 3]) -> Result<(), StrapdownError> {
        self.lever_arm = validate_lever_arm(&lever_arm)?;
        Ok(())
    }

    /// `true` when the engine is running in ENU, `false` for the NED default.
    #[must_use]
    pub const fn is_enu(&self) -> bool {
        self.is_enu
    }

    /// Borrow the underlying filter, for callers that need more than the engine exposes.
    #[must_use]
    pub fn filter(&self) -> &dyn NavigationFilter {
        self.filter.as_ref()
    }

    /// Subtract the filter's gyro bias estimate from a sample's average angular rate.
    ///
    /// Falls back to the raw rate when the filter carries no bias states. A `dt` of zero
    /// cannot occur here -- `predict` has already rejected it via the filter -- but
    /// [`ImuSample::to_rates`] reports rather than divides, so the fallback is explicit.
    fn bias_corrected_angular_rate(&self, sample: &ImuSample) -> Vector3<f64> {
        let Ok(rates) = sample.to_rates() else {
            return Vector3::zeros();
        };
        if self.state_dimension < FULL_STATE_DIMENSION {
            return rates.gyro;
        }
        let state = self.filter.get_estimate();
        rates.gyro - Vector3::new(state[12], state[13], state[14])
    }
}

/// Convert the horizontal position variances from radians squared to metres squared.
fn position_std_m(state: &DVector<f64>, covariance: &DMatrix<f64>) -> [f64; 3] {
    if covariance.nrows() < MINIMUM_STATE_DIMENSION {
        return [0.0; 3];
    }
    let latitude = state[0];
    let altitude = state[2];
    let (meridian_radius, prime_vertical_radius, _) =
        principal_radii(&latitude.to_degrees(), &altitude);
    let cos_latitude = latitude.cos().abs().max(f64::EPSILON);
    [
        covariance[(0, 0)].max(0.0).sqrt() * (meridian_radius + altitude),
        covariance[(1, 1)].max(0.0).sqrt() * (prime_vertical_radius + altitude) * cos_latitude,
        covariance[(2, 2)].max(0.0).sqrt(),
    ]
}

/// Velocity one-sigma values straight off the covariance diagonal.
fn velocity_std_mps(covariance: &DMatrix<f64>) -> [f64; 3] {
    if covariance.nrows() < MINIMUM_STATE_DIMENSION {
        return [0.0; 3];
    }
    [
        covariance[(3, 3)].max(0.0).sqrt(),
        covariance[(4, 4)].max(0.0).sqrt(),
        covariance[(5, 5)].max(0.0).sqrt(),
    ]
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Vector3;

    use super::*;
    use crate::IMUData;

    /// Latitude used throughout: mid-latitude, so neither radius of curvature degenerates
    /// and a north/east mix-up is visible in the numbers.
    const TEST_LATITUDE_DEG: f64 = 40.0;
    const TEST_LONGITUDE_DEG: f64 = -75.0;
    const TEST_ALTITUDE_M: f64 = 100.0;

    fn test_initial_state(is_enu: Option<bool>) -> InitialState {
        InitialState::new(
            TEST_LATITUDE_DEG,
            TEST_LONGITUDE_DEG,
            TEST_ALTITUDE_M,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            is_enu,
        )
    }

    /// A filter that records what the engine hands it and never moves.
    ///
    /// The point of the engine's GNSS path is the *transformation* applied to a fix before
    /// the filter sees it. Asserting on a real filter's posterior would test the ESKF's gain
    /// as much as the lever arm; recording the measurement tests the compensation exactly.
    #[derive(Debug, Default)]
    struct Recorder {
        estimate: Vec<f64>,
        positions: Vec<GPSPositionMeasurement>,
        velocities: Vec<GPSVelocityMeasurement>,
        predicts: usize,
    }

    impl Recorder {
        fn new(estimate: Vec<f64>) -> Rc<RefCell<Self>> {
            Rc::new(RefCell::new(Self {
                estimate,
                ..Self::default()
            }))
        }
    }

    /// Newtype so the engine can own the filter while the test keeps a handle on it.
    #[derive(Debug)]
    struct SharedRecorder(Rc<RefCell<Recorder>>);

    impl NavigationFilter for SharedRecorder {
        fn predict(&mut self, _input: &dyn InputModel, _dt: f64) -> Result<(), StrapdownError> {
            self.0.borrow_mut().predicts += 1;
            Ok(())
        }
        fn update(
            &mut self,
            measurement: &dyn MeasurementModel,
        ) -> Result<UpdateOutcome, StrapdownError> {
            let mut recorder = self.0.borrow_mut();
            if let Some(position) = measurement
                .as_any()
                .downcast_ref::<GPSPositionMeasurement>()
            {
                recorder.positions.push(position.clone());
            } else if let Some(velocity) = measurement
                .as_any()
                .downcast_ref::<GPSVelocityMeasurement>()
            {
                recorder.velocities.push(velocity.clone());
            }
            // This recorder exists to capture what the engine hands the filter, not to
            // filter: it applies no correction, so there is no innovation to score. A
            // zero NIS reports "nothing was inconsistent", which is the truthful answer
            // for a filter whose state never moves.
            Ok(UpdateOutcome::accepted(0.0, measurement.get_dimension()))
        }
        fn get_estimate(&self) -> DVector<f64> {
            DVector::from_vec(self.0.borrow().estimate.clone())
        }
        fn get_certainty(&self) -> DMatrix<f64> {
            DMatrix::identity(
                self.0.borrow().estimate.len(),
                self.0.borrow().estimate.len(),
            )
        }
    }

    /// A 15-element estimate at the test origin with a caller-supplied attitude and gyro bias.
    fn recorder_estimate(yaw_rad: f64, gyro_bias: [f64; 3]) -> Vec<f64> {
        vec![
            TEST_LATITUDE_DEG.to_radians(),
            TEST_LONGITUDE_DEG.to_radians(),
            TEST_ALTITUDE_M,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            yaw_rad,
            0.0,
            0.0,
            0.0,
            gyro_bias[0],
            gyro_bias[1],
            gyro_bias[2],
        ]
    }

    fn engine_with_recorder(
        recorder: &Rc<RefCell<Recorder>>,
        lever_arm: [f64; 3],
        is_enu: bool,
    ) -> InsEngine {
        InsEngine::builder()
            .with_frame(is_enu)
            .with_lever_arm(lever_arm)
            .with_filter(Box::new(SharedRecorder(Rc::clone(recorder))))
            .build()
            .unwrap()
    }

    // ------------------------------ Builder ------------------------------------------------

    #[test]
    fn default_builder_is_ned_at_the_origin() {
        let engine = InsEngine::builder().build().unwrap();
        assert!(!engine.is_enu());
        assert_eq!(engine.lever_arm(), Vector3::zeros());
        assert_approx_eq!(engine.elapsed_s(), 0.0);
        assert!(!engine.nav_solution().is_enu);
    }

    #[test]
    fn builder_adopts_the_frame_of_the_initial_state() {
        let engine = InsEngine::builder()
            .with_initial_state(test_initial_state(Some(true)))
            .build()
            .unwrap();
        assert!(engine.is_enu(), "an ENU initial state should select ENU");
    }

    #[test]
    fn builder_rejects_a_frame_that_contradicts_the_initial_state() {
        let error = InsEngine::builder()
            .with_frame(false)
            .with_initial_state(test_initial_state(Some(true)))
            .build()
            .unwrap_err();
        match error {
            StrapdownError::InvalidConfiguration { field, reason } => {
                assert_eq!(field, "is_enu");
                assert!(reason.contains("NED") && reason.contains("ENU"), "{reason}");
            }
            other => panic!("expected InvalidConfiguration, got {other:?}"),
        }
    }

    #[test]
    fn builder_accepts_a_frame_that_agrees_with_the_initial_state() {
        let engine = InsEngine::builder()
            .with_frame(true)
            .with_initial_state(test_initial_state(Some(true)))
            .build()
            .unwrap();
        assert!(engine.is_enu());
    }

    #[test]
    fn builder_rejects_an_implausible_lever_arm() {
        let error = InsEngine::builder()
            // A lever arm entered in centimetres, or a coordinate in the wrong field.
            .with_lever_arm([150.0, 0.0, 0.0])
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::InvalidConfiguration {
                field: "lever_arm",
                ..
            }
        ));

        let error = InsEngine::builder()
            .with_lever_arm([f64::NAN, 0.0, 0.0])
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::InvalidConfiguration {
                field: "lever_arm",
                ..
            }
        ));
    }

    #[test]
    fn builder_rejects_a_malformed_noise_diagonal() {
        let error = InsEngine::builder()
            .with_process_noise(vec![1e-6; 9])
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::InvalidConfiguration {
                field: "process_noise_diagonal",
                ..
            }
        ));

        let mut negative = vec![1e-6; FULL_STATE_DIMENSION];
        negative[4] = -1.0;
        let error = InsEngine::builder()
            .with_initial_covariance(negative)
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::InvalidConfiguration {
                field: "initial_covariance_diagonal",
                ..
            }
        ));
    }

    #[test]
    fn builder_rejects_a_filter_with_too_few_states() {
        let recorder = Recorder::new(vec![0.0; 6]);
        let error = InsEngine::builder()
            .with_filter(Box::new(SharedRecorder(recorder)))
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::InvalidConfiguration {
                field: "filter",
                ..
            }
        ));
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = InsEngineConfig {
            is_enu: true,
            lever_arm: [1.0, -2.0, 0.5],
            process_noise_diagonal: Some(vec![1e-7; FULL_STATE_DIMENSION]),
            ..InsEngineConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: InsEngineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, parsed);
    }

    // ------------------------------ Lever-arm geometry -------------------------------------

    #[test]
    fn a_forward_lever_arm_points_north_when_level_and_heading_north() {
        let offset =
            lever_arm_position_offset(&Rotation3::identity(), &Vector3::new(3.0, 0.0, 0.0));
        assert_approx_eq!(offset[0], 3.0, 1e-12);
        assert_approx_eq!(offset[1], 0.0, 1e-12);
        assert_approx_eq!(offset[2], 0.0, 1e-12);
    }

    #[test]
    fn a_forward_lever_arm_points_east_when_heading_east() {
        let attitude = Rotation3::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let offset = lever_arm_position_offset(&attitude, &Vector3::new(3.0, 0.0, 0.0));
        assert_approx_eq!(offset[0], 0.0, 1e-12);
        assert_approx_eq!(offset[1], 3.0, 1e-12);
    }

    #[test]
    fn the_velocity_offset_is_the_rotated_cross_product() {
        let attitude = Rotation3::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let angular_rate = Vector3::new(0.0, 0.0, 0.25);
        let lever_arm = Vector3::new(4.0, 0.0, 0.0);
        let offset = lever_arm_velocity_offset(&attitude, &angular_rate, &lever_arm);
        // omega x r = [0, 1, 0] m/s in body axes; yawed 90 deg that is 1 m/s to the south.
        assert_approx_eq!(offset[0], -1.0, 1e-12);
        assert_approx_eq!(offset[1], 0.0, 1e-12);
    }

    #[test]
    fn a_zero_lever_arm_leaves_the_position_untouched() {
        let latitude = TEST_LATITUDE_DEG.to_radians();
        let (lat, lon, alt) = shift_position_by_offset(
            latitude,
            TEST_LONGITUDE_DEG.to_radians(),
            TEST_ALTITUDE_M,
            &Vector3::zeros(),
            false,
        );
        assert_eq!(lat, latitude);
        assert_eq!(lon, TEST_LONGITUDE_DEG.to_radians());
        assert_eq!(alt, TEST_ALTITUDE_M);
    }

    #[test]
    fn the_altitude_offset_changes_sign_with_the_frame() {
        let latitude = TEST_LATITUDE_DEG.to_radians();
        let offset = Vector3::new(0.0, 0.0, 2.0);
        let (_, _, ned_altitude) =
            shift_position_by_offset(latitude, 0.0, TEST_ALTITUDE_M, &offset, false);
        let (_, _, enu_altitude) =
            shift_position_by_offset(latitude, 0.0, TEST_ALTITUDE_M, &offset, true);
        // NED: the offset points down, so the antenna is below the IMU and the IMU is higher.
        assert_approx_eq!(ned_altitude, TEST_ALTITUDE_M + 2.0, 1e-12);
        // ENU: the offset points up, so the antenna is above the IMU.
        assert_approx_eq!(enu_altitude, TEST_ALTITUDE_M - 2.0, 1e-12);
    }

    #[test]
    fn a_northward_offset_moves_the_position_south_by_that_many_metres() {
        let latitude = TEST_LATITUDE_DEG.to_radians();
        let (shifted_latitude, _, _) = shift_position_by_offset(
            latitude,
            TEST_LONGITUDE_DEG.to_radians(),
            TEST_ALTITUDE_M,
            &Vector3::new(25.0, 0.0, 0.0),
            false,
        );
        let (meridian_radius, _, _) = principal_radii(&TEST_LATITUDE_DEG, &TEST_ALTITUDE_M);
        let expected = latitude - 25.0 / (meridian_radius + TEST_ALTITUDE_M);
        assert_approx_eq!(shifted_latitude, expected, 1e-15);
        assert!(shifted_latitude < latitude);
    }

    // ------------------------------ Engine behaviour ---------------------------------------

    #[test]
    fn predict_advances_the_clock_and_reaches_the_filter() {
        let recorder = Recorder::new(recorder_estimate(0.0, [0.0; 3]));
        let mut engine = engine_with_recorder(&recorder, [0.0; 3], false);
        let sample = ImuSample::from_rates(
            &IMUData {
                accel: Vector3::new(0.0, 0.0, -9.81),
                gyro: Vector3::zeros(),
            },
            0.01,
        );
        for _ in 0..50 {
            engine.predict(&sample).unwrap();
        }
        assert_eq!(recorder.borrow().predicts, 50);
        assert_approx_eq!(engine.elapsed_s(), 0.5, 1e-12);
        assert_approx_eq!(engine.nav_solution().elapsed_s, 0.5, 1e-12);
    }

    #[test]
    fn a_zero_lever_arm_passes_the_fix_through_unchanged() {
        let recorder = Recorder::new(recorder_estimate(0.0, [0.0; 3]));
        let mut engine = engine_with_recorder(&recorder, [0.0; 3], false);
        let fix = GnssFix::position(
            TEST_LATITUDE_DEG,
            TEST_LONGITUDE_DEG,
            TEST_ALTITUDE_M,
            5.0,
            10.0,
        )
        .with_velocity([1.0, 2.0, 3.0], 0.2);
        engine.update_gnss(&fix).unwrap();

        let recorded = recorder.borrow();
        let position = &recorded.positions[0];
        assert_eq!(position.latitude, TEST_LATITUDE_DEG);
        assert_eq!(position.longitude, TEST_LONGITUDE_DEG);
        assert_eq!(position.altitude, TEST_ALTITUDE_M);
        let velocity = &recorded.velocities[0];
        assert_eq!(velocity.northward_velocity, 1.0);
        assert_eq!(velocity.eastward_velocity, 2.0);
        assert_eq!(velocity.vertical_velocity, 3.0);
    }

    #[test]
    fn the_fix_handed_to_the_filter_is_referred_to_the_imu_centre() {
        let lever_arm = [10.0, 0.0, 2.0];
        let recorder = Recorder::new(recorder_estimate(0.0, [0.0; 3]));
        let mut engine = engine_with_recorder(&recorder, lever_arm, false);
        engine
            .update_gnss(&GnssFix::position(
                TEST_LATITUDE_DEG,
                TEST_LONGITUDE_DEG,
                TEST_ALTITUDE_M,
                5.0,
                10.0,
            ))
            .unwrap();

        let (meridian_radius, _, _) = principal_radii(&TEST_LATITUDE_DEG, &TEST_ALTITUDE_M);
        let expected_latitude = (TEST_LATITUDE_DEG.to_radians()
            - 10.0 / (meridian_radius + TEST_ALTITUDE_M))
            .to_degrees();

        let recorded = recorder.borrow();
        let position = &recorded.positions[0];
        assert_approx_eq!(position.latitude, expected_latitude, 1e-12);
        // Level and heading north, so the offset has no east component at all.
        assert_approx_eq!(position.longitude, TEST_LONGITUDE_DEG, 1e-14);
        // 2 m of *down* lever arm puts the antenna below the IMU.
        assert_approx_eq!(position.altitude, TEST_ALTITUDE_M + 2.0, 1e-9);
        // The receiver's own accuracy is passed through untouched.
        assert_approx_eq!(position.horizontal_noise_std, 5.0);
        assert_approx_eq!(position.vertical_noise_std, 10.0);
    }

    #[test]
    fn the_compensation_follows_the_attitude_estimate() {
        let recorder = Recorder::new(recorder_estimate(std::f64::consts::FRAC_PI_2, [0.0; 3]));
        let mut engine = engine_with_recorder(&recorder, [10.0, 0.0, 0.0], false);
        engine
            .update_gnss(&GnssFix::position(
                TEST_LATITUDE_DEG,
                TEST_LONGITUDE_DEG,
                TEST_ALTITUDE_M,
                5.0,
                10.0,
            ))
            .unwrap();

        let recorded = recorder.borrow();
        let position = &recorded.positions[0];
        // Heading east, so the whole offset lands in longitude.
        assert_approx_eq!(position.latitude, TEST_LATITUDE_DEG, 1e-12);
        assert!(
            position.longitude < TEST_LONGITUDE_DEG,
            "a 10 m eastward antenna offset should move the IMU estimate west"
        );
    }

    #[test]
    fn the_velocity_compensation_uses_the_bias_corrected_rate() {
        let gyro_bias = [0.0, 0.0, 0.05];
        let recorder = Recorder::new(recorder_estimate(0.0, gyro_bias));
        let mut engine = engine_with_recorder(&recorder, [4.0, 0.0, 0.0], false);

        // Raw yaw rate of 0.3 rad/s against a 0.05 rad/s bias: the true rate is 0.25.
        let sample = ImuSample::from_rates(
            &IMUData {
                accel: Vector3::zeros(),
                gyro: Vector3::new(0.0, 0.0, 0.3),
            },
            0.01,
        );
        engine.predict(&sample).unwrap();
        engine
            .update_gnss(
                &GnssFix::position(
                    TEST_LATITUDE_DEG,
                    TEST_LONGITUDE_DEG,
                    TEST_ALTITUDE_M,
                    5.0,
                    10.0,
                )
                .with_velocity([0.0, 1.0, 0.0], 0.2),
            )
            .unwrap();

        let recorded = recorder.borrow();
        let velocity = &recorded.velocities[0];
        // omega x r = 0.25 * 4 = 1.0 m/s east, which is the whole of the reported velocity.
        assert_approx_eq!(velocity.eastward_velocity, 0.0, 1e-12);
        assert_approx_eq!(velocity.northward_velocity, 0.0, 1e-12);
    }

    #[test]
    fn a_position_only_fix_produces_no_velocity_update() {
        let recorder = Recorder::new(recorder_estimate(0.0, [0.0; 3]));
        let mut engine = engine_with_recorder(&recorder, [1.0, 1.0, 1.0], false);
        engine
            .update_gnss(&GnssFix::position(
                TEST_LATITUDE_DEG,
                TEST_LONGITUDE_DEG,
                TEST_ALTITUDE_M,
                5.0,
                10.0,
            ))
            .unwrap();
        assert_eq!(recorder.borrow().positions.len(), 1);
        assert!(recorder.borrow().velocities.is_empty());
    }

    #[test]
    fn a_malformed_fix_is_rejected_before_the_filter_sees_it() {
        let recorder = Recorder::new(recorder_estimate(0.0, [0.0; 3]));
        let mut engine = engine_with_recorder(&recorder, [0.0; 3], false);

        let error = engine
            .update_gnss(&GnssFix::position(f64::NAN, 0.0, 0.0, 5.0, 10.0))
            .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::MeasurementUnavailable { .. }
        ));

        let error = engine
            .update_gnss(&GnssFix::position(91.0, 0.0, 0.0, 5.0, 10.0))
            .unwrap_err();
        assert!(matches!(error, StrapdownError::OutOfRange { .. }));
        assert!(recorder.borrow().positions.is_empty());
    }

    #[test]
    fn set_lever_arm_validates_like_the_builder() {
        let mut engine = InsEngine::builder().build().unwrap();
        engine.set_lever_arm([1.0, 2.0, 3.0]).unwrap();
        assert_eq!(engine.lever_arm(), Vector3::new(1.0, 2.0, 3.0));
        assert!(engine.set_lever_arm([1e6, 0.0, 0.0]).is_err());
        // A rejected value must not have been applied.
        assert_eq!(engine.lever_arm(), Vector3::new(1.0, 2.0, 3.0));
    }

    // ------------------------------ Navigation solution ------------------------------------

    #[test]
    fn the_nav_solution_reports_degrees_and_metres() {
        let engine = InsEngine::builder()
            .with_initial_state(test_initial_state(None))
            .build()
            .unwrap();
        let solution = engine.nav_solution();
        assert_approx_eq!(solution.latitude, TEST_LATITUDE_DEG, 1e-9);
        assert_approx_eq!(solution.longitude, TEST_LONGITUDE_DEG, 1e-9);
        assert_approx_eq!(solution.altitude, TEST_ALTITUDE_M, 1e-9);
        // 1e-6 rad^2 of latitude variance is kilometres, not micro-anything: the point of
        // reporting metres is that this number is legible without the radii in hand.
        let (meridian_radius, _, _) = principal_radii(&TEST_LATITUDE_DEG, &TEST_ALTITUDE_M);
        assert_approx_eq!(
            solution.position_std_m[0],
            1e-3 * (meridian_radius + TEST_ALTITUDE_M),
            1e-6
        );
        assert_approx_eq!(solution.velocity_std_mps[0], (1e-3_f64).sqrt(), 1e-12);
    }

    #[test]
    fn a_nav_solution_converts_back_to_a_strapdown_state() {
        let engine = InsEngine::builder()
            .with_initial_state(test_initial_state(Some(true)))
            .build()
            .unwrap();
        let solution = engine.nav_solution();
        let state = crate::StrapdownState::from(&solution);
        assert_approx_eq!(state.latitude, TEST_LATITUDE_DEG.to_radians(), 1e-12);
        assert_approx_eq!(state.longitude, TEST_LONGITUDE_DEG.to_radians(), 1e-12);
        assert!(state.is_enu);
    }

    #[test]
    fn a_stationary_engine_is_pulled_toward_a_repeated_fix() {
        // The default ESKF, not a recorder: this is the end-to-end path.
        let mut engine = InsEngine::builder()
            .with_initial_state(test_initial_state(None))
            .build()
            .unwrap();
        let sample = ImuSample::from_rates(
            &IMUData {
                // At rest in NED the accelerometer senses specific force of +g upward,
                // i.e. -g along the down axis.
                accel: Vector3::new(0.0, 0.0, -crate::earth::G0),
                gyro: Vector3::zeros(),
            },
            0.01,
        );
        let target_latitude = TEST_LATITUDE_DEG + 1e-4;
        let fix = GnssFix::position(
            target_latitude,
            TEST_LONGITUDE_DEG,
            TEST_ALTITUDE_M,
            5.0,
            10.0,
        );

        let start_error = (engine.nav_solution().latitude - target_latitude).abs();
        for _ in 0..100 {
            for _ in 0..10 {
                engine.predict(&sample).unwrap();
            }
            engine.update_gnss(&fix).unwrap();
        }
        let end_error = (engine.nav_solution().latitude - target_latitude).abs();
        assert!(
            end_error < start_error / 10.0,
            "expected the estimate to converge on the fix: {start_error} -> {end_error}"
        );
    }

    #[test]
    fn display_says_enough_to_identify_the_solution() {
        let engine = InsEngine::builder()
            .with_initial_state(test_initial_state(None))
            .build()
            .unwrap();
        let rendered = format!("{engine}");
        assert!(rendered.contains("NED"), "{rendered}");
        assert!(rendered.contains("40.0"), "{rendered}");
        let fix = GnssFix::position(1.0, 2.0, 3.0, 4.0, 5.0).with_velocity([6.0, 7.0, 8.0], 0.1);
        assert!(format!("{fix}").contains("6.000"));
    }
}
