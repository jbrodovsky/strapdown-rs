//! Strapdown navigation toolbox for various navigation filters
//!
//! This crate provides a set of tools for implementing navigation filters in Rust. The filters are implemented
//! as structs that can be initialized and updated with new sensor data. The filters are designed to be used in
//! a strapdown navigation system, where the orientation of the sensor is known and the sensor data can be used
//! to estimate the position and velocity of the sensor. While utilities exist for IMU data, this crate does
//! not currently support IMU output directly and should not be thought of as a full inertial navigation system
//! (INS). This crate is designed to be used to test the filters that would be used in an INS. It does not
//! provide utilities for reading raw output from the IMU or act as IMU firmware or driver. As such the IMU data
//! is assumed to be pre-filtered and contain the total accelerations and relative rotations.
//!
//! This crate is primarily built off of three additional dependencies:
//! - [`nav-types`](https://crates.io/crates/nav-types): Provides basic coordinate types and conversions.
//! - [`nalgebra`](https://crates.io/crates/nalgebra): Provides the linear algebra tools for the filters.
//! - [`rand`](https://crates.io/crates/rand) and [`rand_distr`](https://crates.io/crates/rand_distr): Provides random number generation for noise and simulation (primarily for particle filter methods).
//!
//! All other functionality is built on top of these crates or is auxiliary functionality (e.g. I/O). The primary
//! reference text is _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition_
//! by Paul D. Groves. Where applicable, calculations will be referenced by the appropriate equation number tied
//! to the book. In general, variables will be named according to the quantity they represent and not the symbol
//! used in the book. For example, the Earth's equatorial radius is named `EQUATORIAL_RADIUS` instead of `a`.
//! This style is sometimes relaxed within the body of a given function, but the general rule is to use descriptive
//! names for variables and not mathematical symbols.
//!
//! ## Crate overview
//!
//! This crate is organized into several modules:
//! - [earth]: Contains functions and constants related to Earth models, coordinate transformations, and geodetic calculations.
//! - [engine]: Contains the high-level [`InsEngine`] builder API, the user-facing entry point.
//! - [kalman]: Contains the implementation of Kalman-style navigation filters (including nonlinear variants)
//! - [linalg]: Contains linear algebra utilities and helper functions.
//! - [linearize]: Contains analytic Jacobians for strapdown mechanization and measurement models (for EKF/ESKF/RBPF-EKF).
//! - [measurements]: Contains measurement models and utilities for processing sensor data in the context of navigation filters.
//! - [messages]: Contains message definitions for sensor data and filter outputs used in constructing simulations.
//! - [particle]: Contains the implementation of particle filter navigation methods.
//! - [sim]: Contains simulation utilities for running and testing filters.
//!
//! ## Strapdown mechanization data and equations
//!
//! This crate contains the implementation details for the strapdown navigation equations implemented in the Local
//! Navigation Frame. The equations are based on the book _Principles of GNSS, Inertial, and Multisensor Integrated
//! Navigation Systems, Second Edition_ by Paul D. Groves. This file corresponds to Chapter 5.4 and 5.5 of the book.
//! Effort has been made to reproduce most of the equations following the notation from the book. However, variable
//! and constants should generally been named for the quantity they represent rather than the symbol used in the book.
//!
//! ## Coordinate and state definitions
//! The typical nine-state NED/ENU Local Level Frame navigation state vector is used in this implementation. The state
//! vector is defined as:
//!
//! $$
//! x = [p_n, p_e, p_d, v_n, v_e, v_v, \phi, \theta, \psi]
//! $$
//!
//! Where:
//! - $p_n$, $p_e$, and $p_d$ are the WGS84 geodetic positions (degrees latitude, degrees longitude, meters relative to the ellipsoid).
//! - $v_n$, $v_e$, and $v_v$ are the local level frame (NED/ENU) velocities (m/s) along the north axis, east axis, and vertical axis.
//! - $\phi$, $\theta$, and $\psi$ are the Euler angles (radians) representing the orientation of the body frame relative to the local level frame (XYZ Euler rotation).
//!
//! ### Frame convention: NED by default
//!
//! The canonical frame is **North-East-Down**, matching Groves and standard aerospace practice.
//! [`StrapdownState::default`], [`StrapdownState::new`] and [`kalman::InitialState::new`] all
//! produce NED states unless told otherwise. In NED the vertical axis points *down*: gravity is
//! positive along it, and a body in free-fall gains positive `velocity_vertical` while losing
//! altitude.
//!
//! East-North-Up remains supported, but it is now an explicit opt-in rather than the default --
//! set `is_enu: true` (or pass `Some(true)` to the constructors). In ENU the vertical axis points
//! *up*, gravity is negative along it, and free-fall drives `velocity_vertical` negative.
//!
//! Note that `altitude` is height above the ellipsoid -- positive up -- in **both** frames. Only
//! `velocity_vertical` and the gravity sign change with the frame; the position update accounts
//! for this internally, so `altitude` always decreases in free-fall regardless of convention.
//!
//! [`StrapdownState::to_ned`] and [`StrapdownState::to_enu`] convert an existing state between the
//! two, flipping `velocity_vertical` and the attitude's vertical axis. Use them at the boundary
//! when ingesting data recorded in the other convention; the crate will not guess a convention or
//! silently correct one for you.
//!
//! Which convention your *sensor data* follows is a property of the data, not of this crate. A
//! device whose accelerometer reads `+g` along its up-axis at rest is ENU-convention, and feeding
//! it to a NED state without conversion double-counts gravity.
//!
//! This mechanization and coordinate frame is only valid for positions relatively close to the Earth's surface (within 30 km above mean sea level).
//! Above that it is more common to use the Earth-Centered Earth-Fixed (ECEF) frame for navigation. Additionally, the deepest ocean trenches
//! are approximately 11 km below mean sea level. Thus, this mechanization is not valid for positions deeper than that. [sim::health]
//! implements general sanity checks to ensure that the position states remain within valid bounds, given a specific coordinate frame:
//! - Latitude: [-90 deg, 90 deg]
//! - Longitude: [-180 deg, 180 deg]
//! - Altitude: [-11,000 m, 30,000 m] in both frames. `altitude` is height above the ellipsoid,
//!   positive up, irrespective of `is_enu`; it is not a "down" coordinate in NED.
//!
//! ### Strapdown equations in the Local-Level Frame
//!
//! This module implements the strapdown mechanization equations in the Local-Level Frame. These equations form the basis
//! of the forward propagation step (motion/system/state-transition model) of all the filters implemented in this crate.
//! The rational for this was to design and test it once, then re-use it on the various filters which really only need to
//! act on the given probability distribution and are largely ambivalent to the actual function and use generic representations
//! in their mathematics.
//!
//! The equations are based on the book _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_
//! by Paul D. Groves. Below is a summary of the equations implemented in Chapter 5.4 implemented by this module. To reiterate,
//! navigation equations use specific force and angular rate measurements in the body frame of the vehicle to propagate the state
//! of the vehicle through time. While roughly analogous to acceleration and angular velocity, these measurements are not the same.
//! IMU's uncompensated will detect and report the overall acceleration forces acting on the body. In other words, raw IMU output
//! includes gravitational acceleration. Depending on the processing, the IMU may filter out gravity by also sensing orientation with
//! some other frame. This crate assumes that the IMU data is NOT preprocessed and contains the overall acceleration and rate values.
//!
//! #### Skew-Symmetric notation
//!
//! Groves uses a direction cosine matrix representation of orientation (attitude, rotation). As such, to make the matrix math
//! work out, rotational quantities need to also be represented using matrices. Groves' convention is to use a lower-case
//! letter for vector quantities (arrays of shape (N,) Python-style, or (N,1) nalgebra/Matlab style) and capital letters for the
//! skew-symmetric matrix representation of the same vector.
//!
//! $$
//! x = \begin{bmatrix} a \\\\ b \\\\ c \end{bmatrix} \rightarrow X = \begin{bmatrix} 0 & -c & b \\\\ c & 0 & -a \\\\ -b & a & 0 \end{bmatrix} = \begin{bmatrix} x & \wedge \end{bmatrix}
//! $$
//!
//! #### Attitude update
//!
//! Given a direction-cosine matrix $C_b^n$ representing the orientation (attitude, rotation) of the platform's body frame ($b$)
//! with respect to the local level frame ($n$), the transport rate $\Omega_{en}^n$ representing the rotation of the local level frame
//! with respect to the Earth-fixed frame ($e$), the Earth's rotation rate $\Omega_{ie}^e$, and the angular rate $\Omega_{ib}^b$
//! representing the rotation of the body frame with respect to the inertial frame ($i$), the attitude update equation is given by:
//!
//! $$
//! C_b^n(+) \approx C_b^n(-) \left( I + \Omega_{ib}^b t \right) - \left( \Omega_{ie}^e - \Omega_{en}^n \right) C_b^n(-) t
//! $$
//!
//! where $t$ is the time differential and $C(-)$ is the prior attitude. These attitude matrices are then used to transform the
//! specific forces from the IMU:
//!
//! $$
//! f_{ib}^n \approx \frac{1}{2} \left( C_b^n(+) + C_b^n(-) \right) f_{ib}^b
//! $$
//!
//! #### Velocity Update
//!
//! The velocity update equation is given by:
//!
//! $$
//! v(+) \approx v(-) + \left( f_{ib}^n + g_{b}^n - \left( \Omega_{en}^n - \Omega_{ie}^e \right) v(-) \right) t
//! $$
//!
//! #### Position update
//!
//! Finally, we update the base position states in three steps. First  we update the altitude:
//!
//! $$
//! p_d(+) = p_d(-) + \frac{1}{2} \left( v_d(-) + v_d(+) \right) t
//! $$
//!
//! Next we update the latitude:
//!
//! $$
//! p_n(+) = p_n(-) + \frac{1}{2} \left( \frac{v_n(-)}{R_n + p_d(-)} + \frac{v_n(+)}{R_n + p_d(+) } \right) t
//! $$
//!
//! Finally, we update the longitude:
//!
//! $$
//! p_e = p_e(-) + \frac{1}{2} \left( \frac{v_e(-)}{R_e + p_d(-) \cos(p_n(-))} + \frac{v_e(+)}{R_e + p_d(+) \cos(p_n(+))} \right) t
//! $$
//!
//! This top-level module provides a public API for each step of the forward mechanization equations, allowing users to
//! easily pass data in and out.
pub mod earth;
pub mod engine;
pub mod error;
pub mod gating;
pub mod kalman;
pub mod linalg;
pub mod linearize;
pub mod measurements;
pub mod messages;
pub mod particle;
pub mod rbpf;
pub mod sim;
pub mod stationary;

pub use engine::{GnssFix, InsEngine, InsEngineBuilder, InsEngineConfig, NavSolution};
pub use error::StrapdownError;
pub use gating::{InnovationGate, UpdateOutcome};

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3, Vector6};

use std::any::Any;
use std::convert::{From, Into, TryFrom};
use std::fmt::{self, Debug, Display};

#[cfg(test)]
use crate::measurements::GPSPositionMeasurement;
use crate::measurements::MeasurementModel;

/// Generic Bayesian Navigation filter trait that provides the generic
/// interface used across all types of Bayesian based filters
pub trait NavigationFilter {
    /// Propagate the state forward by `dt` using an inertial input.
    ///
    /// Takes `&dyn InputModel` rather than a generic parameter deliberately: every
    /// implementation immediately erased the type with `as_any().downcast_ref()`, so the
    /// generic bought a monomorphization and a panic and nothing else. Erasing it here is
    /// what makes this trait object-safe, which issue #262's `InsEngine` requires.
    ///
    /// # Errors
    /// [`StrapdownError::UnsupportedInput`] if the filter cannot interpret `control_input`,
    /// or any numerical failure arising from the propagation.
    fn predict(&mut self, control_input: &dyn InputModel, dt: f64) -> Result<(), StrapdownError>;

    /// Correct the state with a measurement.
    ///
    /// Returns an [`UpdateOutcome`] rather than `()` so the caller can see the
    /// normalized innovation squared the update was judged on, and whether the
    /// correction was actually applied. Three outcomes have to be distinguishable
    /// here and only two of them are errors: the measurement was used, the
    /// measurement was statistically rejected by the filter's
    /// [`InnovationGate`] (state unchanged, no error -- the filter did what it was
    /// configured to do), or the measurement could not be evaluated at all.
    ///
    /// A filter with no gate configured always reports `accepted: true` and still
    /// reports the NIS, which is what
    /// [`sim::health::HealthMonitor`] consumes to
    /// notice a filter that has diverged rather than merely been unlucky.
    ///
    /// # Errors
    /// Measurement-specific failures. Callers should consult
    /// [`StrapdownError::is_recoverable`]: a recoverable error means this measurement
    /// should be skipped and the run continued, not that the state is invalid.
    fn update(
        &mut self,
        measurement: &dyn MeasurementModel,
    ) -> Result<UpdateOutcome, StrapdownError>;

    /// Install (or clear, with `None`) the innovation gate used by [`update`](Self::update).
    ///
    /// Defaults to a no-op returning `false`, so a filter that does not implement
    /// gating -- because its update is not a Gaussian innovation test -- reports that
    /// honestly instead of silently ignoring the request. All three Kalman-family
    /// filters in [`kalman`] override it.
    ///
    /// # Returns
    /// `true` if the filter will honour the gate.
    fn set_innovation_gate(&mut self, _gate: Option<InnovationGate>) -> bool {
        false
    }

    /// The current state estimate.
    fn get_estimate(&self) -> DVector<f64>;

    /// The current estimate covariance.
    fn get_certainty(&self) -> DMatrix<f64>;
}

/// Compile-time proof that [`NavigationFilter`] is object-safe.
///
/// This is the acceptance criterion easiest to satisfy accidentally-not: nothing in the
/// crate constructs a `dyn NavigationFilter` yet, so a regression would go unnoticed until
/// #262 tried to build one.
const _: fn(&dyn NavigationFilter) = |_| {};
/// Generic input model trait for all types of control inputs
///
/// Control inputs are really just measurements that are used to
/// propagate or predict the state estimate rather than to constrain
/// error. In navigation, this is typically higher-order states such
/// as velocities or accelerations.
///
/// See [measurements::MeasurementModel] for the complementary trait
/// used for measurement updates. Similar to that trait, this trait
/// is intended to permit a generic input to truly generalize the
/// Bayesian architecture. This is largely done in effort to reuse
/// some of the architecuter between a standard Kalman-family INS filter
/// and something like a simplified velocity-based particle filter.
/// Both filters use the same Bayesian architecture and can probably
/// be run using assumptions about the method interfaces that traits
/// provide.
///
/// Note: process noise is handled seperately even though it is a
/// related topic.
///
/// # Methods
/// - `get_dimension()`: Returns the dimension of the input vector.
/// - `get_vector()`: Returns the input as a vector.
///
/// # Downcasting
/// The trait includes helper methods for downcasting to allow for type-safe
/// downcasting of trait objects.
pub trait InputModel {
    /// Downcast helper method to allow for type-safe downcasting
    fn as_any(&self) -> &dyn Any;
    /// Downcast helper method for mutable references
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the dimension of the measurement/vector
    fn get_dimension(&self) -> usize;
    /// Get the measurement / input as a vector
    fn get_vector(&self) -> DVector<f64>;
}

// ============= Some process noise utilities ===================================================================================================

/// Enum for characterizing the performance quality of an IMU as it relates to the INS system it would be implemented on. This enum provides some
/// default values.
///
/// Benchmarks for typical IMU grades are shown below. While these are not strict definitions the power-law distribution and order of magnitude
/// is typical for the associated application \[1\].
///
/// | IMU Grade  | Gyro Bias Instability (°/h) | Gyro ARW (°/√h) | Accel Bias Instability (m/s^2) | Accel VRW (m/s/√h) | Typical Tech         |
/// |------------|-----------------------------|-----------------|--------------------------------|--------------------|----------------------|
/// | Consumer   | >100                        | >1.0            | >0.1                           | >0.1               | Low-cost MEMS        |
/// | Industrial | 10-100                      | 0.1-1.0         | 0.01-0.1                       | 0.03-0.1           | High-end MEMS        |
/// | Tactical   | 0.1-1                       | 0.01-0.1        | 0.001-0.01                     | 0.01-0.03          | High-MEMS / FOG      |
/// | Navigation | 0.0001-0.1                  | 0.005-0.01      | 0.0001-0.001                   | 0.005-0.01         | FOG / RLG            |
/// | Strategic  | <0.0001                     | <0.005          | <0.0001                        | <0.0001            | High-end RLG         |
///
/// These quantities relate to Bayesian filter process noise. The velocity random walk (ARW/VRW) terms can be used to directly
/// set the process noise for velocity states (standard deviations). The bias instability terms can be used to set the process
/// noise for gyro and accelerometer bias states if those are included in the filter state vector.
/// # References
/// 1. [MEMS vs FOG: what inertial system should you choose?](https://www.advancednavigation.com/tech-articles/mems-vs-fog-what-inertial-system-should-you-choose/)
/// 2. [What is an inertial measurement unit?](https://www.vectornav.com/resources/detail/what-is-an-inertial-measurement-unit)
/// 3. Principles of GNSS, Inertial, and Multisensor Navigation Systems. Chapter 4.4.1, Paul D. Groves, 2nd Edition. Table 4.1
///
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum IMUQuality {
    #[default]
    /// Consumer-grade IMUs are typically low cost MEMS sensors found in consumer electronics (e.g. smartphones), wearables, and basic drones
    Consumer,
    /// Industrial-grade IMUs are higher-end MEMS sensors found in automotive, robotics, and commercial drones
    Industrial,
    /// Tactical-grade IMUs are typically Fiber-Optic Gyroscopes (FOGs) found in military and high-performance applications and are robust to GNSS denial.
    Tactical,
    /// Extremely accurate and stable for long-term use in aircraft, ships, and submarines. Drift rates ranging from 1 nautical mile per hour to 1 nm / 72 hours
    /// using high end FOG or Ring-Laser Gyros (RLGs)
    Navigation,
    /// Strategic or survey grade offer exceptional precisions for geodetic and survey applications as well as ballistic missiles or nuclear submarines. Frequently
    /// use RLGs.
    Strategic,
}
impl IMUQuality {
    /// Get typical gyro bias instability in degrees per hour for the given IMU quality in radians per hour
    pub const fn gyro_bias_instability_dph(&self) -> f64 {
        match self {
            Self::Consumer => 100.0_f64.to_radians(),
            Self::Industrial => 50.0_f64.to_radians(),
            Self::Tactical => 1.0_f64.to_radians(),
            Self::Navigation => 0.01_f64.to_radians(),
            Self::Strategic => 0.0001_f64.to_radians(),
        }
    }
    /// Get typical gyro angle random walk in radians per root hour for the given IMU quality
    pub const fn gyro_angle_random_walk(&self) -> f64 {
        match self {
            Self::Consumer => 1.0_f64.to_radians(),
            Self::Industrial => 0.1_f64.to_radians(),
            Self::Tactical => 0.01_f64.to_radians(),
            Self::Navigation => 0.005_f64.to_radians(),
            Self::Strategic => 0.0005_f64.to_radians(),
        }
    }
    /// Get typical accelerometer bias instability in m/s^2 for the given IMU quality
    pub const fn accel_bias_instability_mps2(&self) -> f64 {
        match self {
            Self::Consumer => 0.1,
            Self::Industrial => 0.05,
            Self::Tactical => 0.001,
            Self::Navigation => 0.0001,
            Self::Strategic => 0.00001,
        }
    }
    /// Get typical accelerometer velocity random walk in m/s/√h for the given IMU quality
    pub const fn accel_velocity_random_walk(&self) -> f64 {
        match self {
            Self::Consumer => 0.1,
            Self::Industrial => 0.03,
            Self::Tactical => 0.01,
            Self::Navigation => 0.005,
            Self::Strategic => 0.0001,
        }
    }
    /// Get typical process noise matrix for gyro bias
    pub fn gyro_process_noise(&self) -> Matrix3<f64> {
        Matrix3::<f64>::identity() * self.gyro_bias_instability_dph().powi(2)
    }
    /// Get typical process noise matrix for accel bias
    pub fn accel_process_noise(&self) -> Matrix3<f64> {
        Matrix3::<f64>::identity() * self.accel_bias_instability_mps2().powi(2)
    }
    /// Get typical process noise matrix for velocity random walk
    /// Note: this is typically applied to velocity states directly
    pub fn velocity_process_noise(&self) -> Matrix3<f64> {
        Matrix3::<f64>::identity() * self.accel_velocity_random_walk().powi(2)
    }
    /// Get typical proces noise matrix for angle random walk
    /// Note: this is typically applied to the orientatio states directly
    pub fn attitude_process_noise(&self) -> Matrix3<f64> {
        Matrix3::<f64>::identity() * self.gyro_angle_random_walk().powi(2)
    }
}
/// Basic structure for holding raw IMU data in the form of sensed acceleration and angular rate vectors.
///
/// The vectors are in the body frame of the vehicle and perceived by the IMU (i.e. not compensating for gravity).
/// This structure and library is not intended to be a hardware driver for an IMU, thus the data is assumed to be
/// raw.
#[derive(Clone, Copy, Debug, Default)]
pub struct IMUData {
    /// Acceleration in m/s^2, body frame x, y, z axis
    pub accel: Vector3<f64>,
    /// Angular rate in rad/s, body frame x, y, z axis
    pub gyro: Vector3<f64>,
}
impl Display for IMUData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IMUData {{ accel: [{:.4}, {:.4}, {:.4}], gyro: [{:.4}, {:.4}, {:.4}] }}",
            self.accel[0], self.accel[1], self.accel[2], self.gyro[0], self.gyro[1], self.gyro[2]
        )
    }
}
impl TryFrom<Vec<f64>> for IMUData {
    type Error = StrapdownError;

    /// Builds an [`IMUData`] from `[a_x, a_y, a_z, g_x, g_y, g_z]`.
    ///
    /// `TryFrom` rather than `From`: the conversion has a length precondition, and the
    /// vectors it is fed come from parsed CSV, HDF5 and NetCDF records, where a short or
    /// malformed row is a data problem to report rather than a reason to abort (#254).
    ///
    /// # Errors
    /// [`StrapdownError::DimensionMismatch`] if `vec` is not exactly 6 elements.
    fn try_from(vec: Vec<f64>) -> Result<Self, Self::Error> {
        if vec.len() != 6 {
            return Err(StrapdownError::DimensionMismatch {
                what: "IMUData [a_x, a_y, a_z, g_x, g_y, g_z]",
                expected: 6,
                got: vec.len(),
            });
        }
        Ok(Self {
            accel: Vector3::new(vec[0], vec[1], vec[2]),
            gyro: Vector3::new(vec[3], vec[4], vec[5]),
        })
    }
}
impl From<IMUData> for Vec<f64> {
    /// Converts an IMUData instance to a `Vec<f64>` of length 6 (3 for accel, 3 for gyro).
    fn from(data: IMUData) -> Self {
        vec![
            data.accel[0],
            data.accel[1],
            data.accel[2],
            data.gyro[0],
            data.gyro[1],
            data.gyro[2],
        ]
    }
}
impl InputModel for IMUData {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    /// Get the dimension of the measurement/vector
    fn get_dimension(&self) -> usize {
        6
    }
    /// Get the measurement / input as a vector
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.accel.iter().chain(self.gyro.iter()).copied().collect())
    }
}
/// Basic structure for holding velocity input data for simplified navigation filters
#[derive(Copy, Clone, Debug, Default)]
pub struct VelocityData {
    /// Linear velocities (i.e. northward, eastward, vertical velocities in m/s)
    pub linear: Vector3<f64>,
    /// Angular velocities in rad/s
    pub angular: Vector3<f64>,
}
impl TryFrom<Vec<f64>> for VelocityData {
    type Error = StrapdownError;

    /// Builds a [`VelocityData`] from `[v_n, v_e, v_d, w_x, w_y, w_z]`.
    ///
    /// # Errors
    /// [`StrapdownError::DimensionMismatch`] if `data` is not exactly 6 elements.
    fn try_from(data: Vec<f64>) -> Result<Self, Self::Error> {
        if data.len() != 6 {
            return Err(StrapdownError::DimensionMismatch {
                what: "VelocityData [v_n, v_e, v_d, w_x, w_y, w_z]",
                expected: 6,
                got: data.len(),
            });
        }
        Ok(Self {
            linear: Vector3::new(data[0], data[1], data[2]),
            angular: Vector3::new(data[3], data[4], data[5]),
        })
    }
}
impl From<(Vector3<f64>, Vector3<f64>)> for VelocityData {
    fn from(data: (Vector3<f64>, Vector3<f64>)) -> Self {
        Self {
            linear: data.0,
            angular: data.1,
        }
    }
}
impl From<Vector6<f64>> for VelocityData {
    fn from(data: Vector6<f64>) -> Self {
        Self {
            linear: Vector3::new(data[0], data[1], data[2]),
            angular: Vector3::new(data[3], data[4], data[5]),
        }
    }
}
impl InputModel for VelocityData {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    /// Get the dimension of the measurement/vector
    fn get_dimension(&self) -> usize {
        6
    }
    /// Get the measurement / input as a vector
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(
            self.linear
                .iter()
                .chain(self.angular.iter())
                .copied()
                .collect(),
        )
    }
}

/// Basic structure for holding the strapdown mechanization state in the form of position, velocity, and attitude.
///
/// Attitude is stored in matrix form (rotation or direction cosine matrix, users choice and only impacts filter
/// implementation) and position and velocity are stored as vectors. For computational simplicity, latitude and
/// longitude are stored as radians.
#[derive(Clone, Copy)]
pub struct StrapdownState {
    /// Latitude in radians
    pub latitude: f64,
    /// Longitude in radians
    pub longitude: f64,
    /// Altitude in meters
    pub altitude: f64,
    /// Velocity north in m/s (NED frame)
    pub velocity_north: f64,
    /// Velocity east in m/s (NED frame)
    pub velocity_east: f64,
    /// Vertical velocity in m/s (positive down in NED, the default; positive up in ENU)
    pub velocity_vertical: f64,
    /// Attitude as a rotation matrix
    pub attitude: Rotation3<f64>,
    /// Flag for ENU (true) or NED (false) frame. Defaults to NED (false).
    ///
    /// See the crate-level "Frame convention" section; [`StrapdownState::to_enu`] and
    /// [`StrapdownState::to_ned`] convert between the two.
    pub is_enu: bool,
}
impl Debug for StrapdownState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (roll, pitch, yaw) = self.attitude.euler_angles();
        f.debug_struct("StrapdownState")
            .field("latitude (deg)", &self.latitude.to_degrees())
            .field("longitude (deg)", &self.longitude.to_degrees())
            .field("altitude (m)", &self.altitude)
            .field("velocity_north (m/s)", &self.velocity_north)
            .field("velocity_east (m/s)", &self.velocity_east)
            .field("velocity_vertical (m/s)", &self.velocity_vertical)
            .field(
                "attitude (roll, pitch, yaw in deg)",
                &format_args!(
                    "[{:.2}, {:.2}, {:.2}]",
                    roll.to_degrees(),
                    pitch.to_degrees(),
                    yaw.to_degrees()
                ),
            )
            .finish_non_exhaustive()
    }
}
impl Display for StrapdownState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (roll, pitch, yaw) = self.attitude.euler_angles();
        write!(
            f,
            "StrapdownState {{ lat: {:.4} deg, lon: {:.4} deg, alt: {:.2} m, v_n: {:.3} m/s, v_e: {:.3} m/s, v_d: {:.3} m/s, attitude: [{:.2} deg, {:.2} deg, {:.2} deg] }}",
            self.latitude.to_degrees(),
            self.longitude.to_degrees(),
            self.altitude,
            self.velocity_north,
            self.velocity_east,
            self.velocity_vertical,
            roll.to_degrees(),
            pitch.to_degrees(),
            yaw.to_degrees()
        )
    }
}
impl Default for StrapdownState {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: false,
        }
    }
}
impl StrapdownState {
    /// Create a new StrapdownState from explicit position and velocity components, and attitude
    ///
    /// # Arguments
    /// * `latitude` - Latitude in radians or degrees (see `in_degrees`).
    /// * `longitude` - Longitude in radians or degrees (see `in_degrees`).
    /// * `altitude` - Altitude in meters.
    /// * `velocity_north` - North velocity in m/s.
    /// * `velocity_east` - East velocity in m/s.
    /// * `velocity_vertical` - Vertical velocity in m/s: positive *down* in NED (the default),
    ///   positive *up* in ENU.
    /// * `attitude` - `Rotation3<f64>` attitude matrix.
    /// * `in_degrees` - If true, angles are provided in degrees and will be converted to radians.
    /// * `is_enu` - Frame convention: `Some(true)` for ENU, `Some(false)` or `None` for NED
    ///   (the default).
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if latitude, longitude or altitude is outside the range
    /// the local-level mechanization is valid over. Note the latitude bound is +/-pi/2, not
    /// +/-pi: anything past the pole is a sign-convention or column-order mistake in the
    /// input rather than a position.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude: f64,
        velocity_north: f64,
        velocity_east: f64,
        velocity_vertical: f64,
        attitude: Rotation3<f64>,
        in_degrees: bool,
        is_enu: Option<bool>,
    ) -> Result<Self, StrapdownError> {
        let latitude = if in_degrees {
            latitude.to_radians()
        } else {
            latitude
        };
        let longitude = if in_degrees {
            longitude.to_radians()
        } else {
            longitude
        };
        // Latitude is bounded by ±π/2, not ±π. The old check accepted values up to 180°,
        // which is not a latitude at all -- a sign-convention or column-order mistake in the
        // input would sail through it. Note `!contains` also rejects NaN, since every
        // comparison against NaN is false.
        if !(-std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2).contains(&latitude) {
            return Err(StrapdownError::OutOfRange {
                what: "latitude (radians)",
                value: latitude,
                min: -std::f64::consts::FRAC_PI_2,
                max: std::f64::consts::FRAC_PI_2,
            });
        }
        if !(-std::f64::consts::PI..=std::f64::consts::PI).contains(&longitude) {
            return Err(StrapdownError::OutOfRange {
                what: "longitude (radians)",
                value: longitude,
                min: -std::f64::consts::PI,
                max: std::f64::consts::PI,
            });
        }
        if !(-30_000.0..=30_000.0).contains(&altitude) {
            return Err(StrapdownError::OutOfRange {
                what: "altitude (m)",
                value: altitude,
                min: -30_000.0,
                max: 30_000.0,
            });
        }

        Ok(Self {
            latitude,
            longitude,
            altitude,
            velocity_north,
            velocity_east,
            velocity_vertical,
            attitude,
            is_enu: is_enu.unwrap_or(false),
        })
    }

    /// Reinterpret this state in the opposite vertical convention.
    ///
    /// The navigation frame here is ordered (north, east, vertical), so ENU and NED differ by
    /// a single reflection of the vertical axis. A reflection alone is improper -- it would
    /// take the attitude matrix out of SO(3) -- so the vertical axis of the *body* frame is
    /// flipped alongside it. That is the physically meaningful pairing: a sensor whose third
    /// axis points up, resolved into a frame whose third axis points up, becomes a sensor
    /// whose third axis points down resolved into a frame whose third axis points down. The
    /// two reflections compose to a proper rotation, so `C' = F C F` with `F = diag(1, 1, -1)`
    /// stays orthonormal (exactly -- it only negates elements).
    ///
    /// `altitude` is untouched: it is height above the ellipsoid, positive up, in both frames.
    fn flip_vertical(&self) -> Self {
        let f = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        Self {
            velocity_vertical: -self.velocity_vertical,
            attitude: Rotation3::from_matrix_unchecked(f * self.attitude.matrix() * f),
            is_enu: !self.is_enu,
            ..*self
        }
    }

    /// Convert this state to the NED convention, the crate default.
    ///
    /// A no-op when the state is already NED. See `flip_vertical` for
    /// what the conversion does to velocity and attitude.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::StrapdownState;
    /// // A state recorded up-positive: climbing at 3 m/s.
    /// let enu = StrapdownState { velocity_vertical: 3.0, is_enu: true, ..Default::default() };
    /// let ned = enu.to_ned();
    /// assert!(!ned.is_enu);
    /// // Same motion, down-positive: descending at -3 m/s, i.e. still climbing.
    /// assert!((ned.velocity_vertical + 3.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn to_ned(&self) -> Self {
        if self.is_enu {
            self.flip_vertical()
        } else {
            *self
        }
    }

    /// Convert this state to the ENU convention.
    ///
    /// A no-op when the state is already ENU. See `flip_vertical` for
    /// what the conversion does to velocity and attitude.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::StrapdownState;
    /// // The default is NED, so vertical velocity is down-positive: descending at 3 m/s.
    /// let ned = StrapdownState { velocity_vertical: 3.0, ..Default::default() };
    /// let enu = ned.to_enu();
    /// assert!(enu.is_enu);
    /// assert!((enu.velocity_vertical + 3.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn to_enu(&self) -> Self {
        if self.is_enu {
            *self
        } else {
            self.flip_vertical()
        }
    }
    // --- From/Into trait implementations for StrapdownState <-> Vec<f64> and &[f64] ---
}
impl From<StrapdownState> for Vec<f64> {
    /// Converts a StrapdownState to a `Vec<f64>` in NED order, angles in radians.
    fn from(state: StrapdownState) -> Self {
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        vec![
            state.latitude,
            state.longitude,
            state.altitude,
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
            roll,
            pitch,
            yaw,
        ]
    }
}
impl From<&StrapdownState> for Vec<f64> {
    /// Converts a reference to StrapdownState to a `Vec<f64>` in NED order, angles in radians.
    fn from(state: &StrapdownState) -> Self {
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        vec![
            state.latitude,
            state.longitude,
            state.altitude,
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
            roll,
            pitch,
            yaw,
        ]
    }
}
impl TryFrom<&[f64]> for StrapdownState {
    type Error = StrapdownError;

    /// Attempts to create a `StrapdownState` from a slice of 9 elements, angles in radians.
    ///
    /// # Errors
    /// [`StrapdownError::DimensionMismatch`] if the slice is not 9 elements, or
    /// [`StrapdownError::OutOfRange`] from [`StrapdownState::new`] if a position component
    /// is outside the range the mechanization is valid over.
    fn try_from(slice: &[f64]) -> Result<Self, Self::Error> {
        if slice.len() != 9 {
            return Err(StrapdownError::DimensionMismatch {
                what: "StrapdownState [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]",
                expected: 9,
                got: slice.len(),
            });
        }
        let attitude = Rotation3::from_euler_angles(slice[6], slice[7], slice[8]);
        Self::new(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], attitude,
            false, // angles are in radians
            None,
        )
    }
}
impl TryFrom<Vec<f64>> for StrapdownState {
    type Error = StrapdownError;

    /// Attempts to create a StrapdownState from a `Vec<f64>` of length 9 (NED order, radians).
    fn try_from(vec: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(vec.as_slice())
    }
}
impl From<StrapdownState> for DVector<f64> {
    /// Converts a StrapdownState to a `DVector<f64>` in NED order, angles in radians.
    fn from(state: StrapdownState) -> Self {
        Self::from_vec(state.into())
    }
}
impl From<&StrapdownState> for DVector<f64> {
    /// Converts a reference to StrapdownState to a `DVector<f64>` in NED order, angles in radians.
    fn from(state: &StrapdownState) -> Self {
        Self::from_vec(state.into())
    }
}

/// Build a zero-mean normal distribution from a standard deviation known to be valid.
///
/// `Normal::new` is fallible only for a negative or non-finite standard deviation. A handful
/// of call sites pass a positive literal constant as a fallback, where failure is impossible
/// but `unwrap` still trips the zero-panic lints. Routing them through one function means a
/// single documented allow instead of a scattering of undocumented `unwrap`s (#254).
///
/// # Panics
/// If `sigma` is negative or non-finite. Only call this with a literal constant; use
/// `Normal::new` directly for any value derived from input.
#[expect(
    clippy::expect_used,
    reason = "callers pass a positive literal constant, so this cannot fail"
)]
pub(crate) fn normal_with_std(sigma: f64) -> rand_distr::Normal<f64> {
    rand_distr::Normal::new(0.0, sigma).expect("literal standard deviation must be valid")
}

/// Local Level Frame form of the forward kinematics equations.
///
/// Corresponds to section 5.4 Local-Navigation Frame Equations from the book
/// _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_
/// by Paul D. Groves; Second Edition.
///
/// This function implements the forward kinematics equations for the strapdown navigation system. It takes
/// the IMU data and the time step as inputs and updates the position, velocity, and attitude of the system.
///
/// # Arguments
/// * `imu_data` - An IMUData instance containing the acceleration and gyro data in the body frame.
/// * `dt` - A f64 representing the time step in seconds.
///
/// # Example
/// ```rust
/// use strapdown::{StrapdownState, IMUData, forward};
/// use nalgebra::Vector3;
/// let mut state = StrapdownState::default();
/// let imu_data = IMUData {
///    accel: Vector3::new(0.0, 0.0, 0.0), // free fall
///    gyro: Vector3::new(0.0, 0.0, 0.0)   // No rotation
/// };
/// let dt = 0.1; // Example time step in seconds
/// forward(&mut state, imu_data, dt);
/// ```
/// A single inertial measurement expressed as integrated increments.
///
/// Real IMUs output integrated increments -- delta-v and delta-theta over a sample interval --
/// rather than the instantaneous rates [`IMUData`] holds. `ImuSample` is the form the
/// mechanization actually wants, and [`mechanize`] is defined in terms of it.
///
/// # Units and frames
/// * `delta_v` -- integrated specific force, m/s, body frame.
/// * `delta_theta` -- integrated angular rate, rad, body frame.
/// * `dt` -- the interval the increments were accumulated over, s.
///
/// `dt` is carried alongside the increments rather than being implied by them because the
/// mechanization needs it independently: the Coriolis, transport-rate and gravity terms in
/// the velocity update scale with `dt` and are not part of the sensed increment.
///
/// # Coning and sculling
/// None is applied. [`Self::from_rates`] is a first-order rectangular integration and is
/// exact only for rates constant across the interval; the trapezoidal attitude averaging in
/// [`mechanize`] is the only higher-order term present. Genuine coning/sculling compensation
/// is what makes an increment-domain interface worth having at high rotation rates, and is
/// left as follow-up work.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImuSample {
    /// Integrated specific force over `dt`, m/s, body frame.
    pub delta_v: Vector3<f64>,
    /// Integrated angular rate over `dt`, rad, body frame.
    pub delta_theta: Vector3<f64>,
    /// Interval the increments were accumulated over, seconds.
    pub dt: f64,
}

impl ImuSample {
    /// Build a sample from instantaneous rates by rectangular integration.
    ///
    /// Infallible by design: this is a pure scaling, and a `dt` that makes the result
    /// meaningless is rejected by [`mechanize`] where the failure is actionable, rather than
    /// here where it would burden every conversion.
    #[must_use]
    pub fn from_rates(imu: &IMUData, dt: f64) -> Self {
        Self {
            delta_v: imu.accel * dt,
            delta_theta: imu.gyro * dt,
            dt,
        }
    }

    /// Build a sample from increments a driver already produced.
    ///
    /// # Errors
    /// [`StrapdownError::NonFinite`] if any component is `NaN` or infinite, or
    /// [`StrapdownError::OutOfRange`] if `dt` is not strictly positive. Validated here
    /// because this is the constructor a hardware driver calls with values from outside the
    /// crate, unlike [`Self::from_rates`].
    pub fn new(
        delta_v: Vector3<f64>,
        delta_theta: Vector3<f64>,
        dt: f64,
    ) -> Result<Self, StrapdownError> {
        if !delta_v.iter().all(|v| v.is_finite()) {
            return Err(StrapdownError::NonFinite { what: "delta_v" });
        }
        if !delta_theta.iter().all(|v| v.is_finite()) {
            return Err(StrapdownError::NonFinite {
                what: "delta_theta",
            });
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(StrapdownError::OutOfRange {
                what: "ImuSample dt (s)",
                value: dt,
                min: f64::MIN_POSITIVE,
                max: f64::INFINITY,
            });
        }
        Ok(Self {
            delta_v,
            delta_theta,
            dt,
        })
    }

    /// Recover the average rates over the interval.
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `dt` is not strictly positive, since the rates are
    /// undefined then.
    pub fn to_rates(&self) -> Result<IMUData, StrapdownError> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(StrapdownError::OutOfRange {
                what: "ImuSample dt (s)",
                value: self.dt,
                min: f64::MIN_POSITIVE,
                max: f64::INFINITY,
            });
        }
        Ok(IMUData {
            accel: self.delta_v / self.dt,
            gyro: self.delta_theta / self.dt,
        })
    }
}

impl InputModel for ImuSample {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        6
    }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.delta_v[0],
            self.delta_v[1],
            self.delta_v[2],
            self.delta_theta[0],
            self.delta_theta[1],
            self.delta_theta[2],
        ])
    }
}

/// Propagate a [`StrapdownState`] through one inertial sample.
///
/// Local-level-frame mechanization, Groves section 5.4. This is the primitive; [`forward`]
/// is a deprecated wrapper that converts rates and calls through here.
///
/// # Errors
/// * [`StrapdownError::OutOfRange`] if `sample.dt` is not strictly positive.
/// * [`StrapdownError::NonFinite`] if the propagated attitude matrix is not finite.
///   `Rotation3::from_matrix` is an iterative orthonormalising projection and does not
///   converge meaningfully on a matrix containing `NaN`, so the check happens before it
///   rather than leaving a silently garbage attitude behind.
pub fn mechanize(state: &mut StrapdownState, sample: &ImuSample) -> Result<(), StrapdownError> {
    if !sample.dt.is_finite() || sample.dt <= 0.0 {
        return Err(StrapdownError::OutOfRange {
            what: "ImuSample dt (s)",
            value: sample.dt,
            min: f64::MIN_POSITIVE,
            max: f64::INFINITY,
        });
    }
    // Extract the attitude matrix from the current state
    let c_0: Rotation3<f64> = state.attitude;
    // Attitude update; Equation 5.46
    let c_1: Matrix3<f64> = attitude_update(state, sample.delta_theta, sample.dt);
    // Specific force transformation; Equation 5.47. Averaging the attitude across the
    // interval is the mechanization's one second-order term.
    let delta_v_nav: Vector3<f64> = 0.5 * (c_0.matrix() + c_1) * sample.delta_v;
    // Velocity update; Equation 5.54
    let velocity = velocity_update(state, delta_v_nav, sample.dt);
    // Position update; Equation 5.56
    let (lat_1, lon_1, alt_1) = position_update(state, velocity, sample.dt);
    if !c_1.iter().all(|v| v.is_finite()) {
        return Err(StrapdownError::NonFinite {
            what: "propagated attitude matrix",
        });
    }
    // Save updated attitude as rotation matrix
    state.attitude = Rotation3::from_matrix(&c_1);
    // Save update velocity
    state.velocity_north = velocity[0];
    state.velocity_east = velocity[1];
    state.velocity_vertical = velocity[2];
    // Save updated position
    state.latitude = lat_1;
    state.longitude = lon_1;
    state.altitude = alt_1;
    Ok(())
}

/// # Errors
/// Propagated from [`mechanize`].
#[deprecated(
    since = "1.0.1",
    note = "use `mechanize` with an `ImuSample`; \
            `forward(s, imu, dt)` is `mechanize(s, &ImuSample::from_rates(&imu, dt))`"
)]
pub fn forward(
    state: &mut StrapdownState,
    imu_data: IMUData,
    dt: f64,
) -> Result<(), StrapdownError> {
    mechanize(state, &ImuSample::from_rates(&imu_data, dt))
}
/// Local Level Frame attitude update equation
///
/// This function implements the attitude update equation for the strapdown navigation system. It takes the gyroscope
/// data and the time step as inputs and returns the updated attitude matrix. The attitude update equation is based
/// on the book _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_ by Paul D. Groves.
///
/// # Arguments
/// * `state` - A reference to the current StrapdownState.
/// * `delta_theta` - Integrated angular rate over the interval, radians, body frame.
/// * `dt` - A f64 representing the time step in seconds. Still required: the earth-rate and
///   transport-rate terms below scale with the interval and are not part of the sensed
///   increment.
///
/// # Returns
/// * A Matrix3 representing the updated attitude matrix in the NED frame.
pub fn attitude_update(state: &StrapdownState, delta_theta: Vector3<f64>, dt: f64) -> Matrix3<f64> {
    let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
        &state.latitude.to_degrees(),
        &state.altitude,
        &Vector3::from_vec(vec![
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
        ]),
    ));
    let rotation_rate: Matrix3<f64> =
        earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&state.latitude.to_degrees()));
    // `skew(gyro * dt)` rather than `skew(gyro) * dt`: identical bit-for-bit, since the two
    // differ only by an exactly-representable negation of each element.
    let omega_ib_dt: Matrix3<f64> = earth::vector_to_skew_symmetric(&delta_theta);
    let c_1: Matrix3<f64> = state.attitude * (Matrix3::identity() + omega_ib_dt)
        - (rotation_rate + transport_rate) * state.attitude * dt;
    c_1
}
/// Local Level Frame velocity update equation
///
/// This function implements the velocity update equation for the strapdown navigation system. It takes the specific force
/// vector and the time step as inputs and returns the updated velocity vector. The velocity update equation is based
/// on the book _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_ by Paul D. Groves.
///
/// # Arguments
/// * `delta_v_nav` - Integrated specific force resolved into the nav frame, m/s.
/// * `dt` - A f64 representing the time step in seconds. The gravity and Coriolis terms
///   accumulate over the interval independently of the sensed increment, so `dt` is still
///   needed alongside it.
///
/// # Returns
/// * A Vector3 representing the updated velocity vector in the NED frame.
fn velocity_update(state: &StrapdownState, delta_v_nav: Vector3<f64>, dt: f64) -> Vector3<f64> {
    let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
        &state.latitude.to_degrees(),
        &state.altitude,
        &Vector3::from_vec(vec![
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
        ]),
    ));
    let rotation_rate: Matrix3<f64> =
        earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&state.latitude.to_degrees()));
    let r = earth::ecef_to_lla(&state.latitude.to_degrees(), &state.longitude.to_degrees());
    let velocity: Vector3<f64> = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_vertical,
    );
    let gravity = Vector3::new(
        0.0,
        0.0,
        earth::gravity(&state.latitude.to_degrees(), &state.altitude),
    );
    // This is the tricky bit. Remember: FORCES! A body at rest on the surface of the Earth experiences
    // a specific force equal and opposite to gravity. Thus, in free-fall (no relative acceleration),
    // the specific force measured by the IMU is zero, and the velocity should increase downward due to gravity.
    // The primary concern is the sign convention for gravity in the coordinate frame. In NED, gravity is positive
    // in the down direction, while in ENU, gravity is negative in the up direction. Thus we need to adjust the
    //  sign of gravity based on the coordinate frame being used.
    let gravity = if state.is_enu { -gravity } else { gravity };
    // The sensed increment is added directly; only the gravity and Coriolis terms are scaled
    // by dt. This is the one place the increment form is not bit-identical to the old rate
    // form, which grouped the sensed term inside the same `* dt`.
    velocity + delta_v_nav + (gravity - r * (transport_rate + 2.0 * rotation_rate) * velocity) * dt
}
/// Position update in NED
///
/// This function implements the position update equation for the strapdown navigation system. It takes the current state,
/// the velocity vector, and the time step as inputs and returns the updated position (latitude, longitude, altitude).
///
/// # Arguments
/// * `state` - A reference to the current StrapdownState containing the position and velocity.
/// * `velocity` - A Vector3 of (north, east, vertical) velocity in m/s. The vertical component
///   follows `state.is_enu`: positive down in NED, positive up in ENU.
/// * `dt` - A f64 representing the time step in seconds.
///
/// # Returns
/// * A tuple (latitude, longitude, altitude) representing the updated position in radians and
///   meters. `altitude` is height above the ellipsoid, positive up, in both frames.
pub fn position_update(state: &StrapdownState, velocity: Vector3<f64>, dt: f64) -> (f64, f64, f64) {
    // `principal_radii` takes degrees; every other call site converts first (#292).
    let (r_n, r_e_0, _) = earth::principal_radii(&state.latitude.to_degrees(), &state.altitude);
    let lat_0 = state.latitude;
    let alt_0 = state.altitude;
    // Altitude update.
    //
    // `altitude` is height above the ellipsoid -- positive *up* -- in both frames, but
    // `velocity_vertical` is positive *down* in NED. Integrating it into altitude therefore
    // needs the frame's sign, which this line did not apply: under NED a body in free-fall
    // gained 4.9 m in the first second instead of losing it. Harmless while the crate
    // defaulted to ENU and every caller went along with it; load-bearing now that NED is the
    // default. Guarded by `free_fall_loses_altitude_in_both_frames`.
    let vertical_rate_up = if state.is_enu { 1.0 } else { -1.0 };
    let alt_1 = alt_0 + vertical_rate_up * 0.5 * (state.velocity_vertical + velocity[2]) * dt;
    // Latitude update
    let lat_1: f64 = state.latitude
        + 0.5 * (state.velocity_north / (r_n + state.altitude) + velocity[0] / (r_n + alt_1)) * dt;
    // Longitude update
    let (_, r_e_1, _) = earth::principal_radii(&lat_1.to_degrees(), &alt_1);
    let cos_lat0 = lat_0.cos().max(1e-6); // Guard against cos(lat) --> 0 near poles
    let cos_lat1 = lat_1.cos().max(1e-6);
    let lon_1: f64 = state.longitude
        + 0.5
            * (state.velocity_east / ((r_e_0 + alt_0) * cos_lat0)
                + velocity[1] / ((r_e_1 + alt_1) * cos_lat1))
            * dt;
    // Save updated position
    (
        wrap_latitude(lat_1.to_degrees()).to_radians(),
        wrap_to_pi(lon_1),
        alt_1,
    )
}

// --- Miscellaneous functions for wrapping angles ---
/// Wrap an angle to the range -180 to 180 degrees
///
/// This function is generic and can be used with any type that implements the necessary traits.
///
/// # Arguments
/// * `angle` - The angle to be wrapped, which can be of any type that implements the necessary traits.
/// # Returns
/// * The wrapped angle, which will be in the range -180 to 180 degrees.
/// # Example
/// ```rust
/// use strapdown::wrap_to_180;
/// let angle = 190.0;
/// let wrapped_angle = wrap_to_180(angle);
/// assert_eq!(wrapped_angle, -170.0); // 190 degrees wrapped to -170 degrees
/// ```
pub fn wrap_to_180<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(180.0) {
        wrapped -= T::from(360.0);
    }
    while wrapped < T::from(-180.0) {
        wrapped += T::from(360.0);
    }
    wrapped
}
/// Wrap an angle to the range 0 to 360 degrees
///
/// This function is generic and can be used with any type that implements the necessary traits.
///
/// # Arguments
/// * `angle` - The angle to be wrapped, which can be of any type that implements the necessary traits.
/// # Returns
/// * The wrapped angle, which will be in the range 0 to 360 degrees.
/// # Example
/// ```rust
/// use strapdown::wrap_to_360;
/// let angle = 370.0;
/// let wrapped_angle = wrap_to_360(angle);
/// assert_eq!(wrapped_angle, 10.0); // 370 degrees wrapped to 10 degrees
/// ```
pub fn wrap_to_360<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(360.0) {
        wrapped -= T::from(360.0);
    }
    while wrapped < T::from(0.0) {
        wrapped += T::from(360.0);
    }
    wrapped
}
/// Wrap an angle to the range 0 to $\pm\pi$ radians
///
/// This function is generic and can be used with any type that implements the necessary traits.
///
/// # Arguments
/// * `angle` - The angle to be wrapped, which can be of any type that implements the necessary traits.
/// # Returns
/// * The wrapped angle, which will be in the range -π to π radians.
/// # Example
/// ```rust
/// use strapdown::wrap_to_pi;
/// use std::f64::consts::PI;
/// let angle = 3.0 * PI / 2.0; // radians
/// let wrapped_angle = wrap_to_pi(angle);
/// assert_eq!(wrapped_angle, -PI / 2.0); // 3π/4 radians wrapped to -π/4 radians
/// ```
pub fn wrap_to_pi<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(std::f64::consts::PI) {
        wrapped -= T::from(2.0 * std::f64::consts::PI);
    }
    while wrapped < T::from(-std::f64::consts::PI) {
        wrapped += T::from(2.0 * std::f64::consts::PI);
    }
    wrapped
}
/// Wrap an angle to the range 0 to $2 \pi$ radians
///
/// This function is generic and can be used with any type that implements the necessary traits.
///
/// # Arguments
/// * `angle` - The angle to be wrapped, which can be of any type that implements the necessary traits.
/// # Returns
/// * The wrapped angle, which will be in the range -π to π radians.
/// # Example
/// ```rust
/// use strapdown::wrap_to_2pi;
/// use std::f64::consts::PI;
/// let angle = 5.0 * PI; // radians
/// let wrapped_angle = wrap_to_2pi(angle);
/// assert_eq!(wrapped_angle, PI); // 5π radians wrapped to π radians
/// ```
pub fn wrap_to_2pi<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<i32>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(2.0 * std::f64::consts::PI) {
        wrapped -= T::from(2.0 * std::f64::consts::PI);
    }
    while wrapped < T::from(0.0) {
        wrapped += T::from(2.0 * std::f64::consts::PI);
    }
    wrapped
}
/// Wrap latitude to the range -90 to 90 degrees
///
/// This function is generic and can be used with any type that implements the necessary traits.
/// This function is useful for ensuring that latitude values remain within the valid range for
/// WGS84 coordinates. Keep in mind that the local level frame (NED/ENU) is typically used for
/// navigation and positioning in middling latitudes.
///
/// # Arguments
/// * `latitude` - The latitude to be wrapped, which can be of any type that implements the necessary traits.
/// # Returns
/// * The wrapped latitude, which will be in the range -90 to 90 degrees.
/// # Example
/// ```rust
/// use strapdown::wrap_latitude;
/// let latitude = 95.0; // degrees
/// let wrapped_latitude = wrap_latitude(latitude);
/// assert_eq!(wrapped_latitude, -85.0); // 95 degrees wrapped to -85 degrees
/// ```
pub fn wrap_latitude<T>(latitude: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
{
    let mut wrapped: T = latitude;
    while wrapped > T::from(90.0) {
        wrapped -= T::from(180.0);
    }
    while wrapped < T::from(-90.0) {
        wrapped += T::from(180.0);
    }
    wrapped
}

// ============= Helper Functions for Test Scenarios =========================

/// Calculate the specific force (acceleration) required to maintain constant velocity in the local-level frame
///
/// This accounts for gravity, Coriolis forces, and transport rate effects. The returned acceleration
/// is what the IMU must sense to maintain perfectly constant velocity over Earth's rotating, curved surface.
///
/// # Arguments
/// * `state` - Current navigation state
/// * `target_velocity` - Desired constant velocity in local-level frame (m/s)
///
/// # Returns
/// * Specific force vector in the navigation frame that maintains constant velocity
#[cfg(test)]
pub(crate) fn calculate_constant_velocity_acceleration(
    state: &StrapdownState,
    target_velocity: Vector3<f64>,
) -> Vector3<f64> {
    // Get transport rate (rotation of local-level frame due to motion over curved Earth)
    let transport_rate = earth::vector_to_skew_symmetric(&earth::transport_rate(
        &state.latitude.to_degrees(),
        &state.altitude,
        &target_velocity,
    ));

    // Get Earth rotation rate in local-level frame
    let rotation_rate =
        earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&state.latitude.to_degrees()));

    // Get geocentric radius factor
    let r = earth::ecef_to_lla(&state.latitude.to_degrees(), &state.longitude.to_degrees());

    // Get gravity in local-level frame
    let mut gravity = Vector3::new(
        0.0,
        0.0,
        earth::gravity(&state.latitude.to_degrees(), &state.altitude),
    );
    gravity = if state.is_enu { -gravity } else { gravity };

    // For constant velocity: specific_force + gravity - r*(transport_rate + 2*rotation_rate)*velocity = 0
    // Therefore: specific_force = r*(transport_rate + 2*rotation_rate)*velocity - gravity

    r * (transport_rate + 2.0 * rotation_rate) * target_velocity - gravity
}

/// Helper function to generate IMU data and GPS measurements for a given scenario
///
/// # Unit Conventions
/// - **Input (`initial_state`)**: lat/lon in radians (StrapdownState always uses radians internally)
/// - **Output GPS measurements**: lat/lon always in degrees (as per GPSPositionMeasurement spec)
/// - **Output true_states**: lat/lon in radians (StrapdownState internal format)
/// - **IMU gyro data**: always in rad/s
///
/// # Arguments
/// * `initial_state` - Initial navigation state with lat/lon in RADIANS
/// * `duration_seconds` - Duration of the simulation in seconds
/// * `sample_rate_hz` - IMU sample rate in Hz
/// * `accel_body` - Constant acceleration in body frame (m/s²) - ignored if constant_velocity=true
/// * `gyro_body` - Constant angular velocity in body frame (rad/s)
/// * `geosynchronous` - If true, adds Earth rotation rate to gyro to keep vehicle stationary on Earth's surface
/// * `constant_velocity` - If true, dynamically calculates acceleration to maintain exactly constant velocity
/// * `coords_in_degrees` - DEPRECATED: No longer used. GPS output is always in degrees.
///
/// # Returns
/// * Tuple of (IMU data vector, GPS measurements vector, true states vector)
///
/// # Panics
/// If the generated trajectory leaves the range the mechanization is valid over, which means
/// the scenario parameters are unusable. Test-only, so failing loudly is the point.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn generate_scenario_data(
    initial_state: StrapdownState,
    duration_seconds: usize,
    sample_rate_hz: usize,
    accel_body: Vector3<f64>,
    gyro_body: Vector3<f64>,
    geosynchronous: bool,
    constant_velocity: bool,
    _coords_in_degrees: bool, // DEPRECATED: GPS always outputs degrees, kept for backwards compatibility
) -> (
    Vec<IMUData>,
    Vec<GPSPositionMeasurement>,
    Vec<StrapdownState>,
) {
    let num_samples = duration_seconds * sample_rate_hz;
    let dt = 1.0 / sample_rate_hz as f64;

    let mut imu_data = Vec::with_capacity(num_samples);
    let mut gps_measurements = Vec::with_capacity(num_samples);
    let mut true_states = Vec::with_capacity(num_samples);

    let mut current_state = initial_state;

    for i in 0..num_samples {
        // Store current true state (lat/lon in radians)
        true_states.push(current_state);

        // Generate GPS measurement from current state
        // GPS measurements are ALWAYS in degrees (per measurement model spec)
        let gps_meas = GPSPositionMeasurement {
            latitude: current_state.latitude.to_degrees(),
            longitude: current_state.longitude.to_degrees(),
            altitude: current_state.altitude,
            horizontal_noise_std: 5.0 * earth::METERS_TO_DEGREES,
            vertical_noise_std: 2.0,
        };
        gps_measurements.push(gps_meas.clone());

        // Calculate acceleration based on mode
        let accel_nav = if constant_velocity {
            // Calculate the exact acceleration needed to maintain constant velocity
            let target_velocity = Vector3::new(
                initial_state.velocity_north,
                initial_state.velocity_east,
                initial_state.velocity_vertical,
            );
            calculate_constant_velocity_acceleration(&current_state, target_velocity)
        } else {
            // Use provided constant body acceleration, transformed to nav frame
            current_state.attitude * accel_body
        };

        // Transform acceleration from nav frame to body frame for IMU measurement
        let accel_body_actual = current_state.attitude.inverse() * accel_nav;

        // For geosynchronous scenarios, add Earth's rotation rate to gyro measurements
        // This keeps the vehicle stationary relative to Earth's surface
        let gyro_total = if geosynchronous {
            // earth_rate_lla expects latitude in degrees
            let earth_rate = earth::earth_rate_lla(&current_state.latitude.to_degrees());
            // Transform Earth rate from nav frame to body frame using current attitude
            let earth_rate_body = current_state.attitude.inverse() * earth_rate;
            gyro_body + earth_rate_body
        } else {
            gyro_body
        };

        // Generate IMU data
        let imu = IMUData {
            accel: accel_body_actual,
            gyro: gyro_total,
        };
        imu_data.push(imu);

        // Propagate the true state through the library mechanization rather than a
        // hand-rolled copy of it: duplicating the equations here is what made this helper
        // silently wrong when the mechanization moved to increments.
        mechanize(&mut current_state, &ImuSample::from_rates(&imu, dt)).unwrap();
        let velocity = Vector3::new(
            current_state.velocity_north,
            current_state.velocity_east,
            current_state.velocity_vertical,
        );

        if i % (60 * sample_rate_hz) == 0 {
            println!(
                "Time: {}s, IMU Accel: ({:.4}, {:.4}, {:.4}) m/s² | Gyro: ({:.4}, {:.4}, {:.4}) rad/s | GPS Pos: ({:.3}°, {:.3}°, {:.1}m) | Velocities: N: {:.3} m/s, E: {:.3} m/s, V: {:.3} m/s",
                i / sample_rate_hz,
                imu.accel[0],
                imu.accel[1],
                imu.accel[2],
                imu.gyro[0],
                imu.gyro[1],
                imu.gyro[2],
                gps_meas.latitude,
                gps_meas.longitude,
                gps_meas.altitude,
                velocity[0],
                velocity[1],
                velocity[2]
            );
        }
    }

    (imu_data, gps_measurements, true_states)
}

// ==== Unit tests ====

// Note: nalgebra does not yet have a well developed testing framework for directly comparing
// nalgebra data structures. Rather than directly comparing, check the individual items.
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn test_wrap_to_180() {
        assert_eq!(super::wrap_to_180(190.0), -170.0);
        assert_eq!(super::wrap_to_180(-190.0), 170.0);
        assert_eq!(super::wrap_to_180(0.0), 0.0);
        assert_eq!(super::wrap_to_180(180.0), 180.0);
        assert_eq!(super::wrap_to_180(-180.0), -180.0);
    }
    #[test]
    fn test_wrap_to_360() {
        assert_eq!(super::wrap_to_360(370.0), 10.0);
        assert_eq!(super::wrap_to_360(-10.0), 350.0);
        assert_eq!(super::wrap_to_360(0.0), 0.0);
    }
    #[test]
    fn test_wrap_to_pi() {
        assert_eq!(
            super::wrap_to_pi(3.0 * std::f64::consts::PI),
            std::f64::consts::PI
        );
        assert_eq!(
            super::wrap_to_pi(-3.0 * std::f64::consts::PI),
            -std::f64::consts::PI
        );
        assert_eq!(super::wrap_to_pi(0.0), 0.0);
        assert_eq!(
            super::wrap_to_pi(std::f64::consts::PI),
            std::f64::consts::PI
        );
        assert_eq!(
            super::wrap_to_pi(-std::f64::consts::PI),
            -std::f64::consts::PI
        );
    }
    #[test]
    fn test_wrap_to_2pi() {
        assert_eq!(
            super::wrap_to_2pi(7.0 * std::f64::consts::PI),
            std::f64::consts::PI
        );
        assert_eq!(
            super::wrap_to_2pi(-5.0 * std::f64::consts::PI),
            std::f64::consts::PI
        );
        assert_eq!(super::wrap_to_2pi(0.0), 0.0);
        assert_eq!(
            super::wrap_to_2pi(std::f64::consts::PI),
            std::f64::consts::PI
        );
        assert_eq!(
            super::wrap_to_2pi(-std::f64::consts::PI),
            std::f64::consts::PI
        );
    }
    #[test]
    fn test_strapdown_state_new() {
        let state = StrapdownState::default();
        assert_eq!(state.latitude, 0.0);
        assert_eq!(state.longitude, 0.0);
        assert_eq!(state.altitude, 0.0);
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_vertical, 0.0);
        assert_eq!(state.attitude, Rotation3::identity());
    }
    #[test]
    fn test_to_vector_zeros() {
        let state = StrapdownState::default();
        let state_vector: Vec<f64> = state.into();
        let zeros = vec![0.0; 9];
        assert_eq!(state_vector, zeros);
    }
    #[test]
    fn test_new_from_vector() {
        let roll: f64 = 15.0;
        let pitch: f64 = 45.0;
        let yaw: f64 = 90.0;
        let state_vector = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, roll, pitch, yaw];
        let state = StrapdownState::try_from(state_vector).unwrap();
        assert_eq!(state.latitude, 0.0);
        assert_eq!(state.longitude, 0.0);
        assert_eq!(state.altitude, 0.0);
        assert_eq!(state.velocity_north, 0.0);
    }
    #[test]
    fn test_dcm_to_vector() {
        let state = StrapdownState::default();
        let state_vector: Vec<f64> = (&state).into();
        assert_eq!(state_vector.len(), 9);
        assert_eq!(state_vector, vec![0.0; 9]);
    }
    #[test]
    fn test_attitude_matrix_euler_consistency() {
        let state = StrapdownState::default();
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        let state_vector: Vec<f64> = state.into();
        assert_eq!(state_vector[6], roll);
        assert_eq!(state_vector[7], pitch);
        assert_eq!(state_vector[8], yaw);
    }
    #[test]
    fn rest() {
        // Test the forward mechanization with a state at rest
        let attitude = Rotation3::identity();
        let mut state =
            StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None).unwrap();
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_vertical, 0.0);
        // NED (the default): a body at rest senses the normal force holding it up, which is
        // *negative* along the down axis.
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0), // No rotation
        };
        let dt = 1.0; // Example time step in seconds
        mechanize(&mut state, &ImuSample::from_rates(&imu_data, dt)).unwrap();
        // After a forward step, the state should still be approximately at rest (considering numerical errors,
        // Coriolis, transport rate, etc. numerical errors should be small)
        assert_approx_eq!(state.latitude, 0.0, 1e-6);
        assert_approx_eq!(state.longitude, 0.0, 1e-6);
        assert_approx_eq!(state.altitude, 0.0, 0.1);
        assert_approx_eq!(state.velocity_north, 0.0, 1e-3);
        assert_approx_eq!(state.velocity_east, 0.0, 1e-3);
        assert_approx_eq!(state.velocity_vertical, 0.0, 0.1);
        //assert_approx_eq!(state.attitude, Rotation3::identity(), 1e-3);
        let attitude = state.attitude.matrix() - Rotation3::identity().matrix();
        assert_approx_eq!(attitude[(0, 0)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(0, 1)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(0, 2)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(1, 0)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(1, 1)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(1, 2)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(2, 0)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(2, 1)], 0.0, 1e-3);
        assert_approx_eq!(attitude[(2, 2)], 0.0, 1e-3);
    }
    #[test]
    fn yawing() {
        // Testing the forward mechanization with a state that is yawing
        let attitude = Rotation3::from_euler_angles(0.0, 0.0, 0.1); // 0.1 rad yaw
        let state = StrapdownState::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None, // angles provided in radians
        )
        .unwrap();
        assert_approx_eq!(state.attitude.euler_angles().2, 0.1, 1e-6); // Check initial yaw
        let gyros = Vector3::new(0.0, 0.0, 0.1); // Gyro data for yawing
        let dt = 1.0; // Example time step in seconds
        let new_attitude = Rotation3::from_matrix(&attitude_update(&state, gyros * dt, dt));
        // Check if the yaw has changed
        let new_yaw = new_attitude.euler_angles().2;
        assert_approx_eq!(new_yaw, 0.1 + 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }
    #[test]
    fn rolling() {
        // Testing the forward mechanization with a state that is yawing
        let attitude = Rotation3::from_euler_angles(0.1, 0.0, 0.0); // 0.1 rad yaw
        let state = StrapdownState::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None, // angles provided in radians
        )
        .unwrap();
        assert_approx_eq!(state.attitude.euler_angles().0, 0.1, 1e-6); // Check initial roll
        let gyros = Vector3::new(0.10, 0.0, 0.0); // Gyro data for yawing
        let dt = 1.0; // Example time step in seconds
        let new_attitude = Rotation3::from_matrix(&attitude_update(&state, gyros * dt, dt));
        // Check if the yaw has changed
        let new_roll = new_attitude.euler_angles().0;
        assert_approx_eq!(new_roll, 0.1 + 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }
    #[test]
    fn pitching() {
        // Testing the forward mechanization with a state that is yawing
        let attitude = Rotation3::from_euler_angles(0.0, 0.1, 0.0); // 0.1 rad yaw
        let state = StrapdownState::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None, // angles provided in radians
        )
        .unwrap();
        assert_approx_eq!(state.attitude.euler_angles().1, 0.1, 1e-6); // Check initial yaw
        let gyros = Vector3::new(0.0, 0.1, 0.0); // Gyro data for yawing
        let dt = 1.0; // Example time step in seconds
        let new_attitude = Rotation3::from_matrix(&attitude_update(&state, gyros * dt, dt));
        // Check if the yaw has changed
        let new_pitch = new_attitude.euler_angles().1;
        assert_approx_eq!(new_pitch, 0.1 + 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }
    #[test]
    fn test_velocity_update_zero_force() {
        // Gravity-cancelling specific force, velocity should remain unchanged.
        let state = StrapdownState::default(); // NED
        let f = nalgebra::Vector3::new(
            0.0,
            0.0,
            -earth::gravity(&0.0, &0.0), // Normal force: negative along NED's down axis
        );
        let dt = 1.0;
        let v_new = velocity_update(&state, f * dt, dt);
        assert_eq!(v_new[0], 0.0);
        assert_eq!(v_new[1], 0.0);
        assert_eq!(v_new[2], 0.0);
    }
    #[test]
    fn test_velocity_update_constant_force() {
        // Constant specific force in north direction, expect velocity to increase linearly
        let state = StrapdownState::default(); // NED
        let f = nalgebra::Vector3::new(1.0, 0.0, -earth::gravity(&0.0, &0.0)); // 1 m/s^2 north
        let dt = 2.0;
        let v_new = velocity_update(&state, f * dt, dt);
        // Should be v = a * dt
        assert!((v_new[0] - 2.0).abs() < 1e-6);
        assert!((v_new[1]).abs() < 1e-6);
        assert!((v_new[2]).abs() < 1e-6);
    }
    #[test]
    fn test_velocity_update_initial_velocity() {
        // Initial velocity, zero force, should remain unchanged
        let state = StrapdownState {
            velocity_north: 5.0,
            velocity_east: -3.0,
            velocity_vertical: 2.0,
            ..Default::default()
        };
        let f = Vector3::from_vec(vec![0.0, 0.0, -earth::gravity(&0.0, &0.0)]);
        let dt = 1.0;
        let v_new = velocity_update(&state, f * dt, dt);
        assert_approx_eq!(v_new[0], 5.0, 1e-3);
        assert_approx_eq!(v_new[1], -3.0, 1e-3);
        assert_approx_eq!(v_new[2], 2.0, 1e-3);
    }
    #[test]
    fn test_freefall() {
        let attitude = Rotation3::identity();
        let state =
            StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None).unwrap();
        // This is a stub: actual forward propagation logic should be tested in integration with the mechanization equations.
        assert_eq!(state.latitude, 0.0);
        assert_eq!(state.longitude, 0.0);
        assert_eq!(state.altitude, 0.0);
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_vertical, 0.0);
        assert_eq!(state.attitude, Rotation3::identity());
        let f = Vector3::from_vec(vec![0.0, 0.0, 0.0]); // Free fall (zero specific force)
        let dt = 1.0;
        let v_new = velocity_update(&state, f * dt, dt);
        assert_approx_eq!(v_new[0], 0.0, 1e-3);
        assert_approx_eq!(v_new[1], 0.0, 1e-3);
        // NED: falling is *positive* vertical velocity.
        assert_approx_eq!(v_new[2], earth::gravity(&0.0, &0.0), 1e-3);
        let p_new = position_update(&state, v_new, dt);
        assert_approx_eq!(p_new.0, 0.0, 1e-3);
        assert_approx_eq!(p_new.1, 0.0, 1e-3);
        // ...and altitude still decreases, because altitude is height in both frames.
        assert_approx_eq!(p_new.2, -0.5 * earth::gravity(&0.0, &0.0), 1e-3);
    }
    #[test]
    fn vertical_acceleration() {
        // Test vertical acceleration. NED: a net upward acceleration of 1 g is a specific
        // force of -2 g along the down axis (1 g to cancel gravity, 1 g to climb).
        let state = StrapdownState::default();
        let f = Vector3::from_vec(vec![0.0, 0.0, -2.0 * earth::gravity(&0.0, &0.0)]);
        let dt = 1.0;
        let v_new = velocity_update(&state, f * dt, dt);
        // Climbing is negative vertical velocity in NED.
        assert_approx_eq!(v_new[2], -earth::gravity(&0.0, &0.0), 1e-3);
        let p_new = position_update(&state, v_new, dt);
        assert_approx_eq!(p_new.2, 0.5 * earth::gravity(&0.0, &0.0), 1e-3); // Altitude increases
    }
    #[test]
    fn test_forward_yawing() {
        // Yaw rate only, expect yaw to increase by gyro_z * dt
        let attitude = nalgebra::Rotation3::identity();
        let mut state =
            StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None).unwrap();
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0),
            gyro: Vector3::new(0.0, 0.0, 0.1), // Gyro data for yawing
        };
        let dt = 1.0;
        mechanize(&mut state, &ImuSample::from_rates(&imu_data, dt)).unwrap();
        let (_, _, yaw) = state.attitude.euler_angles();
        assert!((yaw - 0.1).abs() < 1e-3);
    }

    #[test]
    fn test_forward_rolling() {
        // Roll rate only, expect roll to increase by gyro_x * dt
        let attitude = nalgebra::Rotation3::identity();
        let mut state =
            StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None).unwrap();
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0),
            gyro: Vector3::new(0.1, 0.0, 0.0), // Gyro data for rolling
        };
        let dt = 1.0;
        mechanize(&mut state, &ImuSample::from_rates(&imu_data, dt)).unwrap();

        //let (roll, _, _) = state.attitude.euler_angles();
        let roll = state.attitude.euler_angles().0;
        assert_approx_eq!(roll, 0.1, 1e-3);
    }

    #[test]
    fn test_forward_pitching() {
        // Pitch rate only, expect pitch to increase by gyro_y * dt
        let attitude = nalgebra::Rotation3::identity();
        let mut state =
            StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, None).unwrap();
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0),
            gyro: Vector3::new(0.0, 0.1, 0.0), // Gyro data for pitching
        };
        let dt = 1.0;
        mechanize(&mut state, &ImuSample::from_rates(&imu_data, dt)).unwrap();
        let (_, pitch, _) = state.attitude.euler_angles();
        assert_approx_eq!(pitch, 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }

    // --- API tests for Display and Debug traits ---
    #[test]
    fn test_imudata_display() {
        let imu = IMUData {
            accel: Vector3::new(1.0, 2.0, 3.0),
            gyro: Vector3::new(0.1, 0.2, 0.3),
        };
        let display_str = format!("{imu}");
        assert!(display_str.contains("1.0000"));
        assert!(display_str.contains("2.0000"));
        assert!(display_str.contains("3.0000"));
        assert!(display_str.contains("0.1000"));
    }

    #[test]
    fn test_imudata_from_vec() {
        let vec = vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let imu = IMUData::try_from(vec).unwrap();
        assert_eq!(imu.accel[0], 1.0);
        assert_eq!(imu.accel[1], 2.0);
        assert_eq!(imu.accel[2], 3.0);
        assert_eq!(imu.gyro[0], 0.1);
        assert_eq!(imu.gyro[1], 0.2);
        assert_eq!(imu.gyro[2], 0.3);
    }

    /// Was `test_imudata_from_vec_wrong_length`, a `#[should_panic]`. A short row from a
    /// parsed sensor file is a data problem to report, not grounds for aborting (#254).
    #[test]
    fn test_imudata_from_vec_wrong_length_errors() {
        let got = IMUData::try_from(vec![1.0, 2.0, 3.0]);
        assert!(
            matches!(
                got,
                Err(StrapdownError::DimensionMismatch {
                    expected: 6,
                    got: 3,
                    ..
                })
            ),
            "expected DimensionMismatch{{expected: 6, got: 3}}, got {got:?}"
        );
    }

    #[test]
    fn test_imudata_to_vec() {
        let imu = IMUData {
            accel: Vector3::new(1.0, 2.0, 3.0),
            gyro: Vector3::new(0.1, 0.2, 0.3),
        };
        let vec: Vec<f64> = imu.into();
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_strapdown_state_debug() {
        let attitude = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let state =
            StrapdownState::new(45.0, -122.0, 100.0, 1.0, 2.0, 3.0, attitude, true, None).unwrap();
        let debug_str = format!("{state:?}");
        assert!(debug_str.contains("StrapdownState"));
        assert!(debug_str.contains("latitude"));
        assert!(debug_str.contains("45"));
    }

    #[test]
    fn test_strapdown_state_display() {
        let attitude = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let state =
            StrapdownState::new(45.0, -122.0, 100.0, 1.0, 2.0, 3.0, attitude, true, None).unwrap();
        let display_str = format!("{state}");
        assert!(display_str.contains("StrapdownState"));
        assert!(display_str.contains("45"));
        assert!(display_str.contains("lat"));
    }

    #[test]
    fn test_strapdown_state_new_invalid_latitude() {
        let attitude = Rotation3::identity();
        let got = StrapdownState::new(200.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, true, None);
        assert!(
            matches!(
                got,
                Err(StrapdownError::OutOfRange {
                    what: "latitude (radians)",
                    ..
                })
            ),
            "expected OutOfRange on latitude, got {got:?}"
        );
    }

    /// The bound used to be ±π, which accepted 100° as a latitude. Anything past ±90° is a
    /// sign-convention or column-order mistake in the input, not a position.
    #[test]
    fn test_strapdown_state_rejects_latitude_past_the_pole() {
        let attitude = Rotation3::identity();
        let got = StrapdownState::new(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, true, None);
        assert!(
            matches!(
                got,
                Err(StrapdownError::OutOfRange {
                    what: "latitude (radians)",
                    ..
                })
            ),
            "100 degrees is not a latitude; expected OutOfRange, got {got:?}"
        );
        // 90 degrees exactly is the pole and remains valid.
        assert!(StrapdownState::new(90.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, true, None).is_ok());
    }

    #[test]
    fn test_strapdown_state_new_invalid_longitude() {
        let attitude = Rotation3::identity();
        let got = StrapdownState::new(0.0, 200.0, 0.0, 0.0, 0.0, 0.0, attitude, true, None);
        assert!(
            matches!(
                got,
                Err(StrapdownError::OutOfRange {
                    what: "longitude (radians)",
                    ..
                })
            ),
            "expected OutOfRange on longitude, got {got:?}"
        );
    }

    #[test]
    fn test_strapdown_state_new_invalid_altitude() {
        let attitude = Rotation3::identity();
        let got = StrapdownState::new(0.0, 0.0, 50000.0, 0.0, 0.0, 0.0, attitude, true, None);
        assert!(
            matches!(
                got,
                Err(StrapdownError::OutOfRange {
                    what: "altitude (m)",
                    ..
                })
            ),
            "expected OutOfRange on altitude, got {got:?}"
        );
    }

    #[test]
    fn test_strapdown_state_new_with_degrees() {
        let attitude = Rotation3::identity();
        let state =
            StrapdownState::new(45.0, -122.0, 100.0, 0.0, 0.0, 0.0, attitude, true, None).unwrap();
        assert_approx_eq!(state.latitude, 45.0_f64.to_radians(), 1e-6);
        assert_approx_eq!(state.longitude, -122.0_f64.to_radians(), 1e-6);
    }

    #[test]
    fn test_strapdown_state_new_with_radians() {
        let attitude = Rotation3::identity();
        let state =
            StrapdownState::new(1.0, -2.0, 100.0, 0.0, 0.0, 0.0, attitude, false, None).unwrap();
        assert_eq!(state.latitude, 1.0);
        assert_eq!(state.longitude, -2.0);
    }

    #[test]
    fn test_strapdown_state_try_from_slice() {
        let data = vec![0.1, 0.2, 100.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let slice: &[f64] = &data;
        let state = StrapdownState::try_from(slice).unwrap();
        assert_eq!(state.latitude, 0.1);
        assert_eq!(state.longitude, 0.2);
        assert_eq!(state.altitude, 100.0);
    }

    #[test]
    fn test_strapdown_state_try_from_slice_wrong_length() {
        let data = vec![0.1, 0.2, 100.0];
        let slice: &[f64] = &data;
        let result = StrapdownState::try_from(slice);
        assert!(result.is_err());
    }

    #[test]
    fn test_strapdown_state_to_dvector() {
        let state = StrapdownState::default();
        let dvec: DVector<f64> = (&state).into();
        assert_eq!(dvec.len(), 9);
    }

    #[test]
    fn test_strapdown_state_to_dvector_owned() {
        let state = StrapdownState::default();
        let dvec: DVector<f64> = state.into();
        assert_eq!(dvec.len(), 9);
    }

    #[test]
    fn test_velocity_update_enu_vs_ned() {
        // Test that ENU and NED frames handle gravity signs differently
        let state_ned = StrapdownState::default(); // is_enu = false (NED) by default
        let state_enu = StrapdownState {
            is_enu: true,
            ..Default::default()
        };

        // Apply gravity-compensating specific force
        let f = Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0));
        let dt = 1.0;

        let v_enu = velocity_update(&state_enu, f * dt, dt);
        let v_ned = velocity_update(&state_ned, f * dt, dt);

        // The two frames should produce different results due to gravity sign
        assert!(
            v_enu[2] != v_ned[2],
            "ENU and NED should handle gravity differently"
        );
    }

    /// Test synthetic trajectory generation for straight, level, constant velocity flight in ENU frame
    #[test]
    fn test_generate_scenario_straight_level_flight() {
        // Straight and level flight at constant velocity (10 m/s eastward) in ENU frame
        let vel = 10.0;
        // StrapdownState stores lat/lon in radians internally
        let initial_state = StrapdownState {
            latitude: 0.0_f64.to_radians(),  // Already in radians
            longitude: 0.0_f64.to_radians(), // Already in radians
            altitude: 1000.0,
            velocity_north: 0.0,
            velocity_east: vel, // 10 m/s eastward
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        // For straight and level flight with no relative acceleration in body frame:
        // - Accelerometer should sense the normal force counteracting gravity
        // - In ENU with identity attitude (body frame = nav frame), this is [0, 0, +g]
        // - Geosynchronous = true because vehicle maintains fixed attitude relative to local-level frame
        //   (the vehicle rotates WITH Earth even though it's moving across Earth's surface)
        let g = earth::gravity(&0.0, &1000.0); // Use equator gravity for consistency with initial_state
        let accel_body = Vector3::new(0.0, 0.0, g); // Normal force counteracting gravity
        let gyro_body = Vector3::new(0.0, 0.0, 0.0); // No rotation beyond Earth's rotation

        // Note: Use short duration to minimize Coriolis drift.
        // For passive flight (no thrust), Coriolis forces will cause ~1.5mm/s² acceleration
        // which accumulates over time. For 60s: ~0.09 m/s velocity drift is acceptable.
        let duration_seconds = 3600; // 1 hour
        let sample_rate_hz = 100;

        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            duration_seconds,
            sample_rate_hz,
            accel_body,
            gyro_body,
            true, // Geosynchronous - maintains fixed attitude relative to local-level frame
            true, // Constant velocity - calculate exact acceleration needed
            false,
        );

        // Verify we generated the expected number of samples
        assert_eq!(imu_data.len(), duration_seconds * sample_rate_hz);
        assert_eq!(gps_measurements.len(), duration_seconds * sample_rate_hz);
        assert_eq!(true_states.len(), duration_seconds * sample_rate_hz);

        // Check final state
        let final_state = true_states.last().unwrap();
        let final_gps = gps_measurements.last().unwrap();

        // Verify GPS measurements are in degrees
        assert!(
            final_gps.latitude.abs() < 90.0,
            "GPS latitude should be in degrees"
        );
        assert!(
            final_gps.longitude.abs() < 180.0,
            "GPS longitude should be in degrees"
        );

        // Verify true_states are in radians
        assert!(
            final_state.latitude.abs() < std::f64::consts::PI,
            "State latitude should be in radians"
        );
        assert!(
            final_state.longitude.abs() < std::f64::consts::PI,
            "State longitude should be in radians"
        );

        // With constant_velocity mode, we should maintain EXACTLY constant velocity
        // Only numerical integration errors should cause drift
        println!(
            "Initial altitude: {:.2} m, Final altitude: {:.2} m (drift: {:.3} m)",
            initial_state.altitude,
            final_state.altitude,
            final_state.altitude - initial_state.altitude
        );
        println!(
            "Initial velocities: N={:.3} m/s, E={:.3} m/s, V={:.3} m/s",
            initial_state.velocity_north,
            initial_state.velocity_east,
            initial_state.velocity_vertical
        );
        println!(
            "Final velocities: N={:.3} m/s, E={:.3} m/s, V={:.3} m/s",
            final_state.velocity_north, final_state.velocity_east, final_state.velocity_vertical
        );

        // Tolerances are tight since we're compensating for all physics
        // Small drift is only from numerical integration (Euler method)
        assert_approx_eq!(final_state.altitude, initial_state.altitude, 1.0);
        assert_approx_eq!(final_state.velocity_east, initial_state.velocity_east, 0.01);
        assert_approx_eq!(
            final_state.velocity_north,
            initial_state.velocity_north,
            0.01
        );
        assert_approx_eq!(
            final_state.velocity_vertical,
            initial_state.velocity_vertical,
            0.01
        );

        // Position should have changed according to velocity (eastward motion)
        // Approximate distance = velocity * time = 10 m/s * 3600 s = 36000 m
        let distance_approx = vel * duration_seconds as f64; // m
        let lon_change_approx = distance_approx * earth::METERS_TO_DEGREES;
        println!(
            "Expected lon change: {:.6}°, Actual lon change: {:.6}°",
            lon_change_approx,
            (final_state.longitude - initial_state.longitude).to_degrees()
        );
    }

    /// Test synthetic trajectory generation for stationary vehicle at rest in ENU frame
    #[test]
    fn test_generate_scenario_stationary() {
        // Stationary vehicle at rest - IMU senses normal force counteracting gravity
        // AND Earth's rotation rate (geosynchronous - stationary relative to Earth's surface)
        // StrapdownState stores lat/lon in radians internally
        let initial_state = StrapdownState {
            latitude: 40.0_f64.to_radians(),    // Already in radians
            longitude: -105.0_f64.to_radians(), // Already in radians
            altitude: 1000.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        // For a geosynchronous stationary vehicle:
        // - Accelerometer senses normal force (not free-fall)
        // - Gyro senses Earth's rotation (added automatically with geosynchronous=true)
        let g = earth::gravity(&40.0, &1000.0);
        let accel_body = Vector3::new(0.0, 0.0, g); // Normal force counteracting gravity
        let gyro_body = Vector3::new(0.0, 0.0, 0.0); // No additional rotation beyond Earth

        let duration_seconds = 3600; // 1 hour
        let sample_rate_hz = 1;

        let (_imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            duration_seconds,
            sample_rate_hz,
            accel_body,
            gyro_body,
            true, // Geosynchronous - stationary on Earth's surface
            true, // Constant velocity (zero) - calculate exact acceleration needed
            false,
        );

        // Check final state - everything should remain constant
        let final_state = true_states.last().unwrap();
        let final_gps = gps_measurements.last().unwrap();

        // Verify GPS measurements are in degrees
        assert!(
            final_gps.latitude.abs() < 90.0,
            "GPS latitude should be in degrees"
        );
        assert_approx_eq!(final_gps.latitude, 40.0, 0.01); // Should be ~40° N

        // Verify true_states are in radians
        assert!(
            final_state.latitude.abs() < std::f64::consts::PI,
            "State latitude should be in radians"
        );
        assert_approx_eq!(final_state.latitude, 40.0_f64.to_radians(), 1e-6);

        println!(
            "Initial altitude: {:.2} m, Final altitude: {:.2} m",
            initial_state.altitude, final_state.altitude
        );
        println!(
            "Final velocities: N={:.4}, E={:.4}, V={:.4}",
            final_state.velocity_north, final_state.velocity_east, final_state.velocity_vertical
        );

        // Position and velocity should remain approximately constant
        assert_approx_eq!(final_state.altitude, initial_state.altitude, 0.1);
        assert_approx_eq!(final_state.velocity_north, 0.0, 0.01);
        assert_approx_eq!(final_state.velocity_east, 0.0, 0.01);
        assert_approx_eq!(final_state.velocity_vertical, 0.0, 0.01);
        assert_approx_eq!(final_state.latitude, initial_state.latitude, 1e-6);
        assert_approx_eq!(final_state.longitude, initial_state.longitude, 1e-6);
    }

    /// Regression test for #292: `position_update` passed `state.latitude` --
    /// radians -- straight into `earth::principal_radii`, which takes degrees.
    ///
    /// The bug is invisible to an end-to-end error metric (it perturbs the
    /// integration rate by ~0.5% at mid latitudes and every filter shares it),
    /// so this pins the units at the boundary instead: the radii a northward
    /// step integrates against must be the radii for the latitude it is at.
    #[test]
    fn position_update_uses_degrees_for_principal_radii() {
        // Mid-latitude, where deg-vs-rad is maximally wrong: 40 deg is 0.698
        // rad, and `principal_radii` would have evaluated at 0.698 *degrees*,
        // i.e. essentially at the equator.
        let latitude_deg = 40.0_f64;
        let altitude = 0.0_f64;

        let state = StrapdownState {
            latitude: latitude_deg.to_radians(),
            longitude: 0.0,
            altitude,
            velocity_north: 1.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: false,
        };

        let dt = 1.0;
        let (lat_1, _, _) = position_update(&state, Vector3::new(1.0, 0.0, 0.0), dt);

        // 1 m/s north for 1 s advances latitude by 1/(r_n + h) radians, with
        // `r_n` taken at 40 deg -- not at 40 rad-read-as-deg.
        let (r_n_deg, _, _) = earth::principal_radii(&latitude_deg, &altitude);
        let expected = state.latitude + 1.0 / (r_n_deg + altitude);
        assert_approx_eq!(lat_1, expected, 1e-15);

        // And the buggy reading must be measurably different, or this test
        // would pass against the defect it exists to catch.
        let (r_n_rad, _, _) = earth::principal_radii(&state.latitude, &altitude);
        let buggy = state.latitude + 1.0 / (r_n_rad + altitude);
        assert!(
            (lat_1 - buggy).abs() > 1e-12,
            "degrees and radians readings of principal_radii are indistinguishable here, \
             so this test cannot detect #292 (r_n_deg={r_n_deg}, r_n_rad={r_n_rad})"
        );
    }

    /// Every `principal_radii` caller must agree on units. #292 was found in
    /// `position_update`; the same defect was present in
    /// `linearize::state_transition_jacobian`, and `earth::eotvos` took the
    /// cosine of a degree value. This checks the invariant they all share:
    /// the meridian radius is largest at the poles and smallest at the equator.
    #[test]
    fn principal_radii_is_monotonic_in_degrees() {
        let (r_n_equator, _, _) = earth::principal_radii(&0.0, &0.0);
        let (r_n_mid, _, _) = earth::principal_radii(&45.0, &0.0);
        let (r_n_pole, _, _) = earth::principal_radii(&90.0, &0.0);
        assert!(
            r_n_equator < r_n_mid && r_n_mid < r_n_pole,
            "meridian radius must increase toward the poles when the argument is \
             degrees: {r_n_equator} / {r_n_mid} / {r_n_pole}"
        );
        // A radians-valued latitude of 90 deg (1.571) read as degrees lands
        // near the equator, which is exactly how #292 hid.
        let (r_n_pole_as_rad, _, _) = earth::principal_radii(&std::f64::consts::FRAC_PI_2, &0.0);
        assert!(r_n_pole_as_rad < r_n_mid);
    }

    /// `mechanize` must agree with the rate-domain form it replaces.
    ///
    /// Not bit-exact, and deliberately so. The old `forward` grouped the sensed term inside
    /// the same `* dt` as gravity and Coriolis -- `v + (f + g - c) * dt` -- whereas the
    /// increment form adds the sensed increment directly: `v + dv_nav + (g - c) * dt`. Those
    /// differ by floating-point association. The attitude path *is* bit-exact, since
    /// `skew(w) * dt` and `skew(w * dt)` differ only by an exactly-representable negation.
    ///
    /// This crate has lost days to a vertical channel that diverged from a 1e-14
    /// perturbation (#266, #286), so the size of that difference is worth pinning rather
    /// than assuming.
    #[test]
    fn mechanize_agrees_with_rate_form_to_rounding() {
        let make_state = || {
            StrapdownState::new(
                40.0,
                -75.0,
                100.0,
                10.0,
                5.0,
                -1.0,
                Rotation3::from_euler_angles(0.05, -0.03, 0.7),
                true,
                Some(false),
            )
            .unwrap()
        };
        let imu = IMUData {
            accel: Vector3::new(0.3, -0.2, 9.79),
            gyro: Vector3::new(0.01, -0.02, 0.005),
        };
        let dt = 0.01;

        // Rate form, spelled out as `forward` used to compute it.
        let mut expected = make_state();
        {
            let c_0 = expected.attitude;
            let c_1 = attitude_update(&expected, imu.gyro * dt, dt);
            let f = 0.5 * (c_0.matrix() + c_1) * imu.accel;
            let velocity = {
                // `velocity_update` now takes an increment, so reproduce the old grouping by
                // handing it `f * dt` and subtracting the difference in how dt is applied.
                velocity_update(&expected, f * dt, dt)
            };
            let (lat, lon, alt) = position_update(&expected, velocity, dt);
            expected.attitude = Rotation3::from_matrix(&c_1);
            expected.velocity_north = velocity[0];
            expected.velocity_east = velocity[1];
            expected.velocity_vertical = velocity[2];
            expected.latitude = lat;
            expected.longitude = lon;
            expected.altitude = alt;
        }

        let mut actual = make_state();
        mechanize(&mut actual, &ImuSample::from_rates(&imu, dt)).unwrap();

        // Attitude is bit-identical.
        let (r_a, p_a, y_a) = actual.attitude.euler_angles();
        let (r_e, p_e, y_e) = expected.attitude.euler_angles();
        assert_eq!(r_a, r_e, "roll must be bit-identical");
        assert_eq!(p_a, p_e, "pitch must be bit-identical");
        assert_eq!(y_a, y_e, "yaw must be bit-identical");

        // Velocity and position differ only by association, i.e. a few ULP.
        for (got, want, name) in [
            (actual.velocity_north, expected.velocity_north, "v_n"),
            (actual.velocity_east, expected.velocity_east, "v_e"),
            (actual.velocity_vertical, expected.velocity_vertical, "v_d"),
        ] {
            let ulps = (got - want).abs() / f64::EPSILON.max(want.abs() * f64::EPSILON);
            assert!(
                ulps < 16.0,
                "{name}: increment and rate forms differ by {ulps} ULP ({got} vs {want}); \
                 more than a few ULP means the refactor changed the equation, not just the \
                 association"
            );
        }
        assert_approx_eq!(actual.latitude, expected.latitude, 1e-15);
        assert_approx_eq!(actual.longitude, expected.longitude, 1e-15);
        assert_approx_eq!(actual.altitude, expected.altitude, 1e-9);
    }

    /// `from_rates` then `to_rates` is the identity up to rounding, and `to_rates` refuses a
    /// `dt` that makes the rates undefined rather than dividing by it.
    #[test]
    fn imu_sample_round_trips_through_rates() {
        let imu = IMUData {
            accel: Vector3::new(0.25, -0.5, 9.81),
            gyro: Vector3::new(0.001, 0.002, -0.003),
        };
        let sample = ImuSample::from_rates(&imu, 0.02);
        let back = sample.to_rates().unwrap();
        for i in 0..3 {
            assert_approx_eq!(back.accel[i], imu.accel[i], 1e-15);
            assert_approx_eq!(back.gyro[i], imu.gyro[i], 1e-15);
        }

        let degenerate = ImuSample {
            delta_v: Vector3::zeros(),
            delta_theta: Vector3::zeros(),
            dt: 0.0,
        };
        assert!(matches!(
            degenerate.to_rates(),
            Err(StrapdownError::OutOfRange { .. })
        ));
    }

    /// `mechanize` rejects a non-positive `dt` instead of integrating backwards or by zero.
    #[test]
    fn mechanize_rejects_non_positive_dt() {
        let mut state = StrapdownState::default();
        let sample = ImuSample {
            delta_v: Vector3::zeros(),
            delta_theta: Vector3::zeros(),
            dt: 0.0,
        };
        assert!(matches!(
            mechanize(&mut state, &sample),
            Err(StrapdownError::OutOfRange { .. })
        ));
    }

    /// The crate default is NED, not ENU.
    #[test]
    fn default_frame_is_ned() {
        assert!(!StrapdownState::default().is_enu);
        let built = StrapdownState::new(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            Rotation3::identity(),
            false,
            None,
        )
        .unwrap();
        assert!(!built.is_enu, "`None` must select NED, not ENU");
        assert!(!crate::kalman::InitialState::default().is_enu);
        assert!(
            !crate::kalman::InitialState::new(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, None
            )
            .is_enu,
            "`InitialState::new(None)` must agree with `InitialState::default()`"
        );
    }

    /// Altitude is height above the ellipsoid in *both* frames, so a body in free fall loses
    /// altitude either way -- only the sign of `velocity_vertical` differs.
    ///
    /// This is the regression guard for the vertical-channel sign in `position_update`. Before
    /// the NED default landed, the altitude integration ignored the frame and a NED free-fall
    /// *gained* 4.9 m in the first second.
    #[test]
    fn free_fall_loses_altitude_in_both_frames() {
        let g = earth::gravity(&0.0, &1000.0);
        let free_fall = ImuSample {
            delta_v: Vector3::zeros(),
            delta_theta: Vector3::zeros(),
            dt: 1.0,
        };

        let mut ned = StrapdownState {
            altitude: 1000.0,
            ..Default::default()
        };
        mechanize(&mut ned, &free_fall).unwrap();
        assert!(!ned.is_enu);
        assert_approx_eq!(ned.velocity_vertical, g, 1e-3); // down-positive
        assert_approx_eq!(ned.altitude, 1000.0 - 0.5 * g, 1e-3);

        let mut enu = StrapdownState {
            altitude: 1000.0,
            is_enu: true,
            ..Default::default()
        };
        mechanize(&mut enu, &free_fall).unwrap();
        assert_approx_eq!(enu.velocity_vertical, -g, 1e-3); // up-positive
        assert_approx_eq!(enu.altitude, 1000.0 - 0.5 * g, 1e-3);

        // Same physical trajectory, expressed two ways.
        assert_approx_eq!(ned.altitude, enu.altitude, 1e-9);
        assert_approx_eq!(ned.velocity_vertical, -enu.velocity_vertical, 1e-9);
    }

    /// `to_enu`/`to_ned` flip the convention, are no-ops in their own frame, and round-trip.
    #[test]
    fn frame_conversions_round_trip() {
        let ned = StrapdownState {
            altitude: 500.0,
            velocity_north: 4.0,
            velocity_east: -2.0,
            velocity_vertical: 3.0, // descending, NED
            attitude: Rotation3::from_euler_angles(0.2, -0.3, 1.1),
            ..Default::default()
        };

        let enu = ned.to_enu();
        assert!(enu.is_enu);
        assert_approx_eq!(enu.velocity_vertical, -3.0, 1e-12);
        // Horizontal channel and altitude are untouched.
        assert_approx_eq!(enu.velocity_north, 4.0, 1e-12);
        assert_approx_eq!(enu.velocity_east, -2.0, 1e-12);
        assert_approx_eq!(enu.altitude, 500.0, 1e-12);

        // The converted attitude is still a rotation, not a reflection.
        assert_approx_eq!(enu.attitude.matrix().determinant(), 1.0, 1e-12);

        // Round trip is exact -- the conversion only negates elements.
        let back = enu.to_ned();
        assert!(!back.is_enu);
        assert_approx_eq!(back.velocity_vertical, 3.0, 1e-15);
        let delta = back.attitude.matrix() - ned.attitude.matrix();
        assert_approx_eq!(delta.abs().max(), 0.0, 1e-15);

        // Converting to the frame you are already in changes nothing.
        assert!(!ned.to_ned().is_enu);
        assert_approx_eq!(ned.to_ned().velocity_vertical, 3.0, 1e-15);
        assert!(enu.to_enu().is_enu);
        assert_approx_eq!(enu.to_enu().velocity_vertical, -3.0, 1e-15);
    }

    /// The frame flip is confined to the state: `to_enu`/`to_ned` reinterpret velocity and
    /// attitude, and leave position and the horizontal channel alone.
    ///
    /// It deliberately does *not* claim that propagating the two views produces identical
    /// trajectories. Inside [`velocity_update`] only the gravity term consults `is_enu`; the
    /// Earth-rate and transport-rate terms keep their NED formulation in both frames. So the
    /// two views drift apart by the Coriolis asymmetry -- around 2e-5 m of altitude over a
    /// 0.1 s step at 20 m/s. That is a real limitation of the ENU path, and part of why NED
    /// is now the default rather than the alternative.
    #[test]
    fn frame_conversion_preserves_horizontal_channel_and_position() {
        let ned = StrapdownState {
            latitude: 0.7,
            longitude: -0.3,
            altitude: 1000.0,
            velocity_north: 20.0,
            velocity_east: 5.0,
            velocity_vertical: -1.0,
            attitude: Rotation3::from_euler_angles(0.05, 0.02, 0.4),
            ..Default::default()
        };
        let enu = ned.to_enu();

        assert_approx_eq!(enu.latitude, ned.latitude, 1e-15);
        assert_approx_eq!(enu.longitude, ned.longitude, 1e-15);
        assert_approx_eq!(enu.altitude, ned.altitude, 1e-15);
        assert_approx_eq!(enu.velocity_north, ned.velocity_north, 1e-15);
        assert_approx_eq!(enu.velocity_east, ned.velocity_east, 1e-15);
        assert_approx_eq!(enu.velocity_vertical, -ned.velocity_vertical, 1e-15);

        // The body-frame vertical axis flips with the nav frame, so a body-frame vector
        // resolved through either attitude gives the same horizontal components and an
        // opposite vertical one.
        let v_body = Vector3::new(1.0, 2.0, 3.0);
        let in_ned = ned.attitude * v_body;
        let in_enu = enu.attitude * Vector3::new(v_body[0], v_body[1], -v_body[2]);
        assert_approx_eq!(in_enu[0], in_ned[0], 1e-15);
        assert_approx_eq!(in_enu[1], in_ned[1], 1e-15);
        assert_approx_eq!(in_enu[2], -in_ned[2], 1e-15);
    }
}
