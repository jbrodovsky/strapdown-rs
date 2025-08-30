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
//! ## Strapdown mechanization data and equations
//!
//! This crate contains the implementation details for the strapdown navigation equations implemented in the Local
//! Navigation Frame. The equations are based on the book _Principles of GNSS, Inertial, and Multisensor Integrated
//! Navigation Systems, Second Edition_ by Paul D. Groves. This file corresponds to Chapter 5.4 and 5.5 of the book.
//! Effort has been made to reproduce most of the equations following the notation from the book. However, variable
//! and constants should generally been named for the quantity they represent rather than the symbol used in the book.
//!
//! ## Coordinate and state definitions
//! The typical nine-state NED/ENU navigation state vector is used in this implementation. The state vector is defined as:
//!
//! $$
//! x = [p_n, p_e, p_d, v_n, v_e, v_d, \phi, \theta, \psi]
//! $$
//!
//! Where:
//! - $p_n$, $p_e$, and $p_d$ are the WGS84 geodetic positions (degrees latitude, degrees longitude, meters relative to the ellipsoid).
//! - $v_n$, $v_e$, and $v_d$ are the local level frame (NED/ENU) velocities (m/s) along the north axis, east axis, and vertical axis.
//! - $\phi$, $\theta$, and $\psi$ are the Euler angles (radians) representing the orientation of the body frame relative to the local level frame (XYZ Euler rotation).
//!
//! The coordinate convention and order is in NED.
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
//! by Paul D. Groves. Below is a summary of the equations implemented in Chapter 5.4 implemented by this module.
//!
//! #### Skew-Symmetric notation
//!
//! Groves uses a direction cosine matrix representation of orientation (attitude, rotation). As such, to make the matrix math
//! work out, rotational quantities need to also be represented using matrices. Groves' convention is to use a lower-case
//! letter for vector quantities (arrays of shape (N,) Python-style, or (N,1) nalgebra/Matlab style) and capital letters for the
//! skew-symmetric matrix representation of the same vector.
//!
//! $$
//! x = \begin{bmatrix} a \\\\ b \\\\ c \end{bmatrix} \rightarrow X = \begin{bmatrix} 0 & -c & b \\\\ c & 0 & -a \\\\ -b & a & 0 \end{bmatrix}
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
pub mod filter;
pub mod linalg;
pub mod sim;

use nalgebra::{DVector, Matrix3, Rotation3, Vector3};

use std::convert::{From, Into, TryFrom};
use std::fmt::{Debug, Display};

/// Basic structure for holding IMU data in the form of acceleration and angular rate vectors.
///
/// The vectors are the body frame of the vehicle and represent relative movement. This structure and library is not intended
/// to be a hardware driver for an IMU, thus the data is assumed to be pre-processed and ready for use in the
/// mechanization equations (the IMU processing has already filtered out gravitational acceleration).
#[derive(Clone, Copy, Debug, Default)]
pub struct IMUData {
    pub accel: Vector3<f64>, // Acceleration in m/s^2, body frame x, y, z axis
    pub gyro: Vector3<f64>,  // Angular rate in rad/s, body frame x, y, z axis
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
impl From<Vec<f64>> for IMUData {
    fn from(vec: Vec<f64>) -> Self {
        if vec.len() != 6 {
            panic!(
                "IMUData must be initialized with a vector of length 6 (3 for accel, 3 for gyro)"
            );
        }
        IMUData {
            accel: Vector3::new(vec[0], vec[1], vec[2]),
            gyro: Vector3::new(vec[3], vec[4], vec[5]),
        }
    }
}
impl Into<Vec<f64>> for IMUData {
    fn into(self) -> Vec<f64> {
        vec![
            self.accel[0],
            self.accel[1],
            self.accel[2],
            self.gyro[0],
            self.gyro[1],
            self.gyro[2],
        ]
    }
}
/// Basic structure for holding the strapdown mechanization state in the form of position, velocity, and attitude.
///
/// Attitude is stored in matrix form (rotation or direction cosine matrix) and position and velocity are stored as
/// vectors. The order or the states depends on the coordinate system used. The struct does not care, but the
/// coordinate system used will determine which functions you should use. Default is NED but nonetheless must be
/// assigned. For computational simplicity, latitude and longitude are stored as radians.
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
    /// Velocity down in m/s (NED frame)
    pub velocity_down: f64,
    /// Attitude as a rotation matrix (unchanged)
    pub attitude: Rotation3<f64>,
    /// Coordinate convention used for the state vector (NED or ENU; NED is true by default)
    pub coordinate_convention: bool, // true for NED, false for ENU
}

impl Debug for StrapdownState {
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
            self.velocity_down,
            roll.to_degrees(),
            pitch.to_degrees(),
            yaw.to_degrees()
        )
    }
}
impl Default for StrapdownState {
    fn default() -> Self {
        StrapdownState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_down: 0.0,
            attitude: Rotation3::identity(),
            coordinate_convention: true, // NED by default
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
    /// * `velocity_down` - Down velocity in m/s.
    /// * `attitude` - Rotation3<f64> attitude matrix.
    /// * `in_degrees` - If true, angles are provided in degrees and will be converted to radians.
    /// * `ned` - If true, the coordinate convention is NED (North, East, Down), otherwise ENU (East, North, Up).
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude: f64,
        velocity_north: f64,
        velocity_east: f64,
        velocity_down: f64,
        attitude: Rotation3<f64>,
        in_degrees: bool,
        ned: bool,
    ) -> StrapdownState {
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
        assert!(
            latitude >= -std::f64::consts::PI && latitude <= std::f64::consts::PI,
            "Latitude must be in the range [-π, π]"
        );
        assert!(
            longitude >= -std::f64::consts::PI && longitude <= std::f64::consts::PI,
            "Longitude must be in the range [-π, π]"
        );
        assert!(
            altitude >= -10_000.0,
            "Altitude must be greater than -10,000 meters (to avoid unrealistic values)"
        );

        StrapdownState {
            latitude,
            longitude,
            altitude,
            velocity_north,
            velocity_east,
            velocity_down,
            attitude,
            coordinate_convention: ned,
        }
    }
    // --- From/Into trait implementations for StrapdownState <-> Vec<f64> and &[f64] ---
}
impl From<StrapdownState> for Vec<f64> {
    /// Converts a StrapdownState to a Vec<f64> in NED order, angles in radians.
    fn from(state: StrapdownState) -> Self {
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        vec![
            state.latitude,
            state.longitude,
            state.altitude,
            state.velocity_north,
            state.velocity_east,
            state.velocity_down,
            roll,
            pitch,
            yaw,
        ]
    }
}
impl From<&StrapdownState> for Vec<f64> {
    /// Converts a reference to StrapdownState to a Vec<f64> in NED order, angles in radians.
    fn from(state: &StrapdownState) -> Self {
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        vec![
            state.latitude,
            state.longitude,
            state.altitude,
            state.velocity_north,
            state.velocity_east,
            state.velocity_down,
            roll,
            pitch,
            yaw,
        ]
    }
}
impl TryFrom<&[f64]> for StrapdownState {
    type Error = &'static str;
    /// Attempts to create a StrapdownState from a slice of 9 elements (NED order, radians).
    fn try_from(slice: &[f64]) -> Result<Self, Self::Error> {
        if slice.len() != 9 {
            return Err("Slice must have length 9 for StrapdownState");
        }
        let attitude = Rotation3::from_euler_angles(slice[6], slice[7], slice[8]);
        Ok(StrapdownState::new(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], attitude,
            false, // angles are in radians
            true,  // NED convention
        ))
    }
}
impl TryFrom<Vec<f64>> for StrapdownState {
    type Error = &'static str;
    /// Attempts to create a StrapdownState from a Vec<f64> of length 9 (NED order, radians).
    fn try_from(vec: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(vec.as_slice())
    }
}
impl From<StrapdownState> for DVector<f64> {
    /// Converts a StrapdownState to a DVector<f64> in NED order, angles in radians.
    fn from(state: StrapdownState) -> Self {
        DVector::from_vec(state.into())
    }
}
impl From<&StrapdownState> for DVector<f64> {
    /// Converts a reference to StrapdownState to a DVector<f64> in NED order, angles in radians.
    fn from(state: &StrapdownState) -> Self {
        DVector::from_vec(state.into())
    }
}

/// NED form of the forward kinematics equations. Corresponds to section 5.4 Local-Navigation Frame Equations
/// from the book _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_
/// by Paul D. Groves; Second Edition.
///
/// This function implements the forward kinematics equations for the strapdown navigation system. It takes
/// the IMU data and the time step as inputs and updates the position, velocity, and attitude of the system.
/// The IMU data is assumed to be pre-processed and ready for use in the mechanization equations (i.e. the
/// gravity vector has already been filtered out and the data represents relative motion).
///
/// # Arguments
/// * `imu_data` - A reference to an IMUData instance containing the acceleration and gyro data in the body frame.
/// * `dt` - A f64 representing the time step in seconds.
///
/// # Example
/// ```rust
/// use strapdown::{StrapdownState, IMUData, forward};
/// use nalgebra::Vector3;
/// let mut state = StrapdownState::default();
/// let imu_data = IMUData {
///    accel: Vector3::new(0.0, 0.0, -9.81), // free fall acceleration in m/s^2
///    gyro: Vector3::new(0.0, 0.0, 0.0) // No rotation
/// };
/// let dt = 0.1; // Example time step in seconds
/// forward(&mut state, imu_data, dt);
/// ```
pub fn forward(state: &mut StrapdownState, imu_data: IMUData, dt: f64) {
    // Extract the attitude matrix from the current state
    let c_0: Rotation3<f64> = state.attitude;
    // Attitude update; Equation 5.46
    let c_1: Matrix3<f64> = attitude_update(state, imu_data.gyro, dt);
    // Specific force transformation; Equation 5.47
    let f: Vector3<f64> = 0.5 * (c_0.matrix() + c_1) * imu_data.accel;
    // Velocity update; Equation 5.54
    let velocity = velocity_update(state, f, dt);
    // Position update; Equation 5.56
    let (lat_1, lon_1, alt_1) = position_update(state, velocity, dt);
    // Save updated attitude as rotation
    state.attitude = Rotation3::from_matrix(&c_1);
    // Save update velocity
    state.velocity_north = velocity[0];
    state.velocity_east = velocity[1];
    state.velocity_down = velocity[2];
    // Save updated position
    state.latitude = lat_1;
    state.longitude = lon_1;
    state.altitude = alt_1;
}
/// NED Attitude update equation
///
/// This function implements the attitude update equation for the strapdown navigation system. It takes the gyroscope
/// data and the time step as inputs and returns the updated attitude matrix. The attitude update equation is based
/// on the book _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_ by Paul D. Groves.
///
/// # Arguments
/// * `gyros` - A Vector3 representing the gyroscope data in rad/s in the body frame x, y, z axis.
/// * `dt` - A f64 representing the time step in seconds.
///
/// # Returns
/// * A Matrix3 representing the updated attitude matrix in the NED frame.
fn attitude_update(state: &StrapdownState, gyros: Vector3<f64>, dt: f64) -> Matrix3<f64> {
    let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
        &state.latitude.to_degrees(),
        &state.altitude,
        &Vector3::from_vec(vec![
            state.velocity_north,
            state.velocity_east,
            state.velocity_down,
        ]),
    ));
    let rotation_rate: Matrix3<f64> =
        earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&state.latitude.to_degrees()));
    let omega_ib: Matrix3<f64> = earth::vector_to_skew_symmetric(&gyros);
    let c_1: Matrix3<f64> = state.attitude * (Matrix3::identity() + omega_ib * dt)
        - (rotation_rate + transport_rate) * state.attitude * dt;
    c_1
}
/// Velocity update in NED
///
/// This function implements the velocity update equation for the strapdown navigation system. It takes the specific force
/// vector and the time step as inputs and returns the updated velocity vector. The velocity update equation is based
/// on the book _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_ by Paul D. Groves.
///
/// # Arguments
/// * `f` - A Vector3 representing the specific force vector in m/s^2 in the NED frame.
/// * `dt` - A f64 representing the time step in seconds.
///
/// # Returns
/// * A Vector3 representing the updated velocity vector in the NED frame.
fn velocity_update(state: &StrapdownState, specific_force: Vector3<f64>, dt: f64) -> Vector3<f64> {
    let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
        &state.latitude.to_degrees(),
        &state.altitude,
        &Vector3::from_vec(vec![
            state.velocity_north,
            state.velocity_east,
            state.velocity_down,
        ]),
    ));
    let rotation_rate: Matrix3<f64> =
        earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&state.latitude.to_degrees()));
    let r = earth::ecef_to_lla(&state.latitude.to_degrees(), &state.longitude.to_degrees());
    let velocity: Vector3<f64> = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_down,
    );
    let gravity = Vector3::new(
        0.0,
        0.0,
        earth::gravity(&state.latitude.to_degrees(), &state.altitude),
    );
    velocity
        + (specific_force - gravity - r * (transport_rate + 2.0 * rotation_rate) * velocity) * dt
}
/// Position update in NED
///
/// This function implements the position update equation for the strapdown navigation system. It takes the current state,
/// the velocity vector, and the time step as inputs and returns the updated position (latitude, longitude, altitude).
///
/// # Arguments
/// * `state` - A reference to the current StrapdownState containing the position and velocity.
/// * `velocity` - A Vector3 representing the velocity vector in m/s in the NED frame.
/// * `dt` - A f64 representing the time step in seconds.
///
/// # Returns
/// * A tuple (latitude, longitude, altitude) representing the updated position in radians and meters.
pub fn position_update(state: &StrapdownState, velocity: Vector3<f64>, dt: f64) -> (f64, f64, f64) {
    let (r_n, r_e_0, _) = earth::principal_radii(&state.latitude, &state.altitude);
    let lat_0 = state.latitude;
    let alt_0 = state.altitude;
    // Altitude update
    let alt_1 = alt_0 + 0.5 * (state.velocity_down + velocity[2]) * dt;
    // Latitude update
    let lat_1: f64 = state.latitude
        + 0.5 * (state.velocity_north / (r_n + alt_0) + velocity[1] / (r_n + state.altitude)) * dt;
    // Longitude update
    let (_, r_e_1, _) = earth::principal_radii(&lat_1, &state.altitude);
    let lon_1: f64 = state.longitude
        + 0.5
            * (state.velocity_east / ((r_e_0 + alt_0) * lat_0.cos())
                + velocity[1] / ((r_e_1 + state.altitude) * lat_1.cos()))
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
// Note: nalgebra does not yet have a well developed testing framework for directly comparing
// nalgebra data structures. Rather than directly comparing, check the individual items.
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_strapdown_state_new() {
        let state = StrapdownState::default();
        assert_eq!(state.latitude, 0.0);
        assert_eq!(state.longitude, 0.0);
        assert_eq!(state.altitude, 0.0);
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_down, 0.0);
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
    // Test the forward mechanization (basic structure, not full dynamics)
    fn test_forward_freefall_stub() {
        let attitude = Rotation3::identity();
        let state = StrapdownState::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, true, // NED convention
        );
        // This is a stub: actual forward propagation logic should be tested in integration with the mechanization equations.
        assert_eq!(state.latitude, 0.0);
        assert_eq!(state.longitude, 0.0);
        assert_eq!(state.altitude, 0.0);
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_down, 0.0);
        assert_eq!(state.attitude, Rotation3::identity());
    }
    #[test]
    fn rest() {
        // Test the forward mechanization with a state at rest
        let attitude = Rotation3::identity();
        let mut state = StrapdownState::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, true, // NED convention
        );
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_down, 0.0);
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0), // No rotation
        };
        let dt = 1.0; // Example time step in seconds
        forward(&mut state, imu_data, dt);
        // After a forward step, the state should still be at rest
        assert_approx_eq!(state.latitude, 0.0, 1e-6);
        assert_approx_eq!(state.longitude, 0.0, 1e-6);
        assert_approx_eq!(state.altitude, 0.0, 0.1);
        assert_approx_eq!(state.velocity_north, 0.0, 1e-3);
        assert_approx_eq!(state.velocity_east, 0.0, 1e-3);
        assert_approx_eq!(state.velocity_down, 0.0, 0.1);
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
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, // angles provided in radians
            true,  // NED convention
        );
        assert_approx_eq!(state.attitude.euler_angles().2, 0.1, 1e-6); // Check initial yaw
        let gyros = Vector3::new(0.0, 0.0, 0.1); // Gyro data for yawing
        let dt = 1.0; // Example time step in seconds
        let new_attitude = Rotation3::from_matrix(&attitude_update(&state, gyros, dt));
        // Check if the yaw has changed
        let new_yaw = new_attitude.euler_angles().2;
        assert_approx_eq!(new_yaw, 0.1 + 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }
    #[test]
    fn rolling() {
        // Testing the forward mechanization with a state that is yawing
        let attitude = Rotation3::from_euler_angles(0.1, 0.0, 0.0); // 0.1 rad yaw
        let state = StrapdownState::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, // angles provided in radians
            true,  // NED convention
        );
        assert_approx_eq!(state.attitude.euler_angles().0, 0.1, 1e-6); // Check initial roll
        let gyros = Vector3::new(0.10, 0.0, 0.0); // Gyro data for yawing
        let dt = 1.0; // Example time step in seconds
        let new_attitude = Rotation3::from_matrix(&attitude_update(&state, gyros, dt));
        // Check if the yaw has changed
        let new_roll = new_attitude.euler_angles().0;
        assert_approx_eq!(new_roll, 0.1 + 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }
    #[test]
    fn pitching() {
        // Testing the forward mechanization with a state that is yawing
        let attitude = Rotation3::from_euler_angles(0.0, 0.1, 0.0); // 0.1 rad yaw
        let state = StrapdownState::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, // angles provided in radians
            true,  // NED convention
        );
        assert_approx_eq!(state.attitude.euler_angles().1, 0.1, 1e-6); // Check initial yaw
        let gyros = Vector3::new(0.0, 0.1, 0.0); // Gyro data for yawing
        let dt = 1.0; // Example time step in seconds
        let new_attitude = Rotation3::from_matrix(&attitude_update(&state, gyros, dt));
        // Check if the yaw has changed
        let new_pitch = new_attitude.euler_angles().1;
        assert_approx_eq!(new_pitch, 0.1 + 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }
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
    fn test_velocity_update_zero_force() {
        // Zero specific force, velocity should remain unchanged
        let state = StrapdownState::default();
        let f = nalgebra::Vector3::new(
            0.0,
            0.0,
            earth::gravity(&0.0, &0.0), // Gravity vector in NED
        );
        let dt = 1.0;
        let v_new = velocity_update(&state, f, dt);
        assert_eq!(v_new[0], 0.0);
        assert_eq!(v_new[1], 0.0);
        assert_eq!(v_new[2], 0.0);
    }
    #[test]
    fn test_velocity_update_constant_force() {
        // Constant specific force in north direction, expect velocity to increase linearly
        let state = StrapdownState::default();
        let f = nalgebra::Vector3::new(1.0, 0.0, earth::gravity(&0.0, &0.0)); // 1 m/s^2 north
        let dt = 2.0;
        let v_new = velocity_update(&state, f, dt);
        // Should be v = a * dt
        assert!((v_new[0] - 2.0).abs() < 1e-6);
        assert!((v_new[1]).abs() < 1e-6);
        assert!((v_new[2]).abs() < 1e-6);
    }
    #[test]
    fn test_velocity_update_initial_velocity() {
        // Initial velocity, zero force, should remain unchanged
        let mut state = StrapdownState::default();
        state.velocity_north = 5.0;
        state.velocity_east = -3.0;
        state.velocity_down = 2.0;
        let f = Vector3::from_vec(vec![0.0, 0.0, earth::gravity(&0.0, &0.0)]);
        let dt = 1.0;
        let v_new = velocity_update(&state, f, dt);
        assert_approx_eq!(v_new[0], 5.0, 1e-3);
        assert_approx_eq!(v_new[1], -3.0, 1e-3);
        assert_approx_eq!(v_new[2], 2.0, 1e-3);
    }
    #[test]
    fn vertical_acceleration() {
        // Test vertical acceleration
        let mut state = StrapdownState::default();
        state.velocity_north = 0.0;
        state.velocity_east = 0.0;
        state.velocity_down = 0.0;
        let f = Vector3::from_vec(vec![0.0, 0.0, 2.0 * earth::gravity(&0.0, &0.0)]); // Downward acceleration
        let dt = 1.0;
        let v_new = velocity_update(&state, f, dt);
        assert_approx_eq!(v_new[2], earth::gravity(&0.0, &0.0), 1e-3);
    }
    #[test]
    fn test_forward_yawing() {
        // Yaw rate only, expect yaw to increase by gyro_z * dt
        let attitude = nalgebra::Rotation3::identity();
        let mut state = StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, true);
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0),
            gyro: Vector3::new(0.0, 0.0, 0.1), // Gyro data for yawing
        };
        let dt = 1.0;
        forward(&mut state, imu_data, dt);
        let (_, _, yaw) = state.attitude.euler_angles();
        assert!((yaw - 0.1).abs() < 1e-3);
    }

    #[test]
    fn test_forward_rolling() {
        // Roll rate only, expect roll to increase by gyro_x * dt
        let attitude = nalgebra::Rotation3::identity();
        let mut state = StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, true);
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0),
            gyro: Vector3::new(0.1, 0.0, 0.0), // Gyro data for rolling
        };
        let dt = 1.0;
        forward(&mut state, imu_data, dt);

        //let (roll, _, _) = state.attitude.euler_angles();
        let roll = state.attitude.euler_angles().0;
        assert_approx_eq!(roll, 0.1, 1e-3);
    }

    #[test]
    fn test_forward_pitching() {
        // Pitch rate only, expect pitch to increase by gyro_y * dt
        let attitude = nalgebra::Rotation3::identity();
        let mut state = StrapdownState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, true);
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0),
            gyro: Vector3::new(0.0, 0.1, 0.0), // Gyro data for pitching
        };
        let dt = 1.0;
        forward(&mut state, imu_data, dt);
        let (_, pitch, _) = state.attitude.euler_angles();
        assert_approx_eq!(pitch, 0.1, 1e-3); // 0.1 rad initial + 0.1 rad
    }
}
