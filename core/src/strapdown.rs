//! Strapdown navigation toolbox for various navigation filters
//!
//! This crate provides a set of tools for implementing navigation filters in Rust. The filters are implemented
//! as structs that can be initialized and updated with new sensor data. The filters are designed to be used in
//! a strapdown navigation system, where the orientation of the sensor is known and the sensor data can be used
//! to estimate the position and velocity of the sensor. While utilities exist for IMU data, this crate does
//! not currently support IMU output directly and should not be thought of as a full inertial navigation system
//! (INS). This crate is designed to be used to test the filters that would be used in an INS. It does not
//! provide utilities for reading raw output from the IMU or act as an IMU firmware or driver.
//!
//! As such the IMU data is assumed to be _relative_ accelerations and rotations with the orientation and gravity
//! vector pre-filtered. Additional signals that can be derived using IMU data, such as gravity or magnetic vector
//! and anomalies, should come be provided to this toolbox as a seperate sensor channel. In other words, to
//! calculate the gravity vector the IMU output should be parsed to separately output the overall acceleration
//! and rotation of the sensor whereas the navigation filter will use the gravity and orientation corrected
//! acceleration and rotation to estimate the position
//!
//! Primarily built off of two crate dependencies:
//! - [`nav-types`](https://crates.io/crates/nav-types): Provides basic coordinate types and conversions.
//! - [`nalgebra`](https://crates.io/crates/nalgebra): Provides the linear algebra tools for the filters.
//!
//! All other functionality is built on top of these crates. The primary reference text is _Principles of GNSS,
//! Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition_ by Paul D. Groves. Where applicable,
//! calculations will be referenced by the appropriate equation number tied to the book. In general, variables
//! will be named according to the quantity they represent and not the symbol used in the book. For example,
//! the Earth's equatorial radius is named `EQUATORIAL_RADIUS` instead of `a`. This style is sometimes relaxed
//! within the body of a given function, but the general rule is to use descriptive names for variables and not
//! mathematical symbols.
//!
//! # Strapdown mechanization data and equations
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
//! The coordinate convention and order is in NED. ENU implementations are to be added in the future.
//!
//! ## Strapdown equations in the Local-Level Frame
//! This crates implements the strapdown mechanization equations in the Local-Level Frame. These equations form the basis
//! of the forward propagation step (motion/system/state-transition model) of all the filters implemented in this crate.
//! The rational for this was to design and test it once, then re-used on the various filters which really only need to
//! act on the given probability distribution and are largely ambivilent to the actual function and use generic representations
//! in thier mathematics.
//!
//! The equations are based on the book _Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition_
//! by Paul D. Groves. Below is a summary of the equations implemented in Chapter 5.4 implemented by this module.
//!
//! ### Skew-Symmetric notation
//!
//! Groves uses a direction cosine matrix representation of orientation (attitude, rotation). As such, to make the matrix math
//! work out, rotational quantities need to also be represented using matricies. As such, Groves' convention is to use a lower-case
//! letter for vector quantities (arrays of shape (N,) Python-style, or (N,1) nalgebra/Matlab style) and capital letters for the
//! skew-symmetric matrix representation of the same vector.
//!
//! $$
//! x = \begin{bmatrix} a \\\\ b \\\\ c \end{bmatrix} \rightarrow X = \begin{bmatrix} 0 & -c & b \\\\ c & 0 & -a \\\\ -b & a & 0 \end{bmatrix}
//! $$
//!
//! ### Attitude update
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
//! where $t$ is the time differential and $C(-)$ is the prior attitude. These attitude matricies are then used to transform the
//! specific forces from the IMU:
//!
//! $$
//! f_{ib}^n \approx \frac{1}{2} \left( C_b^n(+) + C_b^n(-) \right) f_{ib}^b
//! $$
//!
//! ### Velocity Update
//!
//! The velocity update equation is given by:
//!
//! $$
//! v(+) \approx v(-) + \left( f_{ib}^n + g_{b}^n - \left( \Omega_{en}^n - \Omega_{ie}^e \right) v(-) \right) t
//! $$
//!
//! ### Position update
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

pub mod earth;
pub mod filter;
pub mod linalg;
pub mod sim;

 // might be overkill to need this entire create just for this
use nalgebra::{Matrix3, Rotation3, SVector, Vector3};
use std::fmt::{Debug, Display};

/// Basic structure for holding IMU data in the form of acceleration and angular rate vectors.
///
/// The vectors are the body frame of the vehicle and represent relative movement. This structure and library is not intended
/// to be a hardware driver for an IMU, thus the data is assumed to be pre-processed and ready for use in the
/// mechanization equations (the IMU processing has already filtered out gravitational acceleration).
#[derive(Clone, Copy, Debug)]
pub struct IMUData {
    pub accel: Vector3<f64>, // Acceleration in m/s^2, body frame x, y, z axis
    pub gyro: Vector3<f64>,  // Angular rate in rad/s, body frame x, y, z axis
}
impl Default for IMUData {
    fn default() -> Self {
        Self::new()
    }
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
impl IMUData {
    /// Create a new IMUData instance with all zeros
    pub fn new() -> IMUData {
        IMUData {
            accel: Vector3::zeros(),
            gyro: Vector3::zeros(),
        }
    }
    /// Create a new IMUData instance from acceleration and gyro vectors
    ///
    /// The vectors are in the body frame of the vehicle and represent relative movement.
    /// The acceleration vector is in m/s^2 and the gyro vector is in rad/s.
    ///
    /// # Arguments
    /// * `accel` - A Vector3 representing the acceleration in m/s^2 in the body frame x, y, z axis.
    /// * `gyro` - A Vector3 representing the angular rate in rad/s in the body frame x, y, z axis.
    ///
    /// # Returns
    /// * An IMUData instance containing the acceleration and gyro vectors.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::IMUData;
    /// use nalgebra::Vector3;
    /// let imu_data = IMUData::new_from_vector(
    ///    Vector3::new(0.0, 0.0, -9.81), // free fall acceleration in m/s^2
    ///    Vector3::new(0.0, 0.0, 0.0) // No rotation
    /// );
    /// ```
    pub fn new_from_vector(accel: Vector3<f64>, gyro: Vector3<f64>) -> IMUData {
        IMUData { accel, gyro }
    }
    /// Create a new IMUData instance from acceleration and gyro vectors in Vec<f64> format
    ///
    /// The vectors are in the body frame of the vehicle and represent relative movement.
    /// The acceleration vector is in m/s^2 and the gyro vector is in rad/s.
    ///
    /// # Arguments
    /// * `accel` - A Vector3 representing the acceleration in m/s^2 in the body frame x, y, z axis.
    /// * `gyro` - A Vector3 representing the angular rate in rad/s in the body frame x, y, z axis.
    ///
    /// # Returns
    /// * An IMUData instance containing the acceleration and gyro vectors.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::IMUData;
    /// let imu_data = IMUData::new_from_vec(
    ///    vec![0.0, 0.0, -9.81], // free fall acceleration in m/s^2
    ///    vec![0.0, 0.0, 0.0] // No rotation
    /// );
    /// ```
    pub fn new_from_vec(accel: Vec<f64>, gyro: Vec<f64>) -> IMUData {
        IMUData {
            accel: Vector3::new(accel[0], accel[1], accel[2]),
            gyro: Vector3::new(gyro[0], gyro[1], gyro[2]),
        }
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
        Self::new()
    }
}

impl StrapdownState {
    /// Create a new StrapdownState with all zeros
    pub fn new() -> StrapdownState {
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
    pub fn new_from_components(
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
        StrapdownState {
            latitude: if in_degrees {
                latitude.to_radians()
            } else {
                latitude
            },
            longitude: if in_degrees {
                longitude.to_radians()
            } else {
                longitude
            },
            altitude,
            velocity_north,
            velocity_east,
            velocity_down,
            attitude,
            coordinate_convention: ned,
        }
    }
    /// Create a StrapdownState from a canonical state vector (NED order: lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw)
    ///
    /// # Arguments
    /// * `state` - SVector<f64, 9> in the order [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
    /// * `in_degrees` - If true, angles are provided in degrees and will be converted to radians.
    pub fn new_from_vector(state: SVector<f64, 9>, in_degrees: bool) -> StrapdownState {
        let attitude = Rotation3::from_euler_angles(
            if in_degrees {
                state[6].to_radians()
            } else {
                state[6]
            },
            if in_degrees {
                state[7].to_radians()
            } else {
                state[7]
            },
            if in_degrees {
                state[8].to_radians()
            } else {
                state[8]
            },
        );
        StrapdownState {
            latitude: if in_degrees {
                state[0].to_radians()
            } else {
                state[0]
            },
            longitude: if in_degrees {
                state[1].to_radians()
            } else {
                state[1]
            },
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_down: state[5],
            attitude,
            coordinate_convention: true,
        }
    }
    /// Convert the StrapdownState to a one dimensional vector, nalgebra style
    ///
    /// # Arguments
    /// * `in_degrees` - If true, angles are returned in degrees.
    ///
    /// # Returns
    /// * SVector<f64, 9> in the order [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
    pub fn to_vector(&self, in_degrees: bool) -> SVector<f64, 9> {
        SVector::from_vec(self.to_vec(in_degrees))
    }
    /// Convert the StrapdownState to a one dimensional vector, native Vec<f64> style
    ///
    /// # Arguments
    /// * `in_degrees` - If true, angles are returned in degrees.
    ///
    /// # Returns
    /// * Vec<f64> in the order [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
    pub fn to_vec(&self, in_degrees: bool) -> Vec<f64> {
        let (roll, pitch, yaw) = self.attitude.euler_angles();
        if self.coordinate_convention {
            vec![
                if in_degrees {
                    self.latitude.to_degrees()
                } else {
                    self.latitude
                },
                if in_degrees {
                    self.longitude.to_degrees()
                } else {
                    self.longitude
                },
                self.altitude,
                self.velocity_north,
                self.velocity_east,
                self.velocity_down,
                if in_degrees { roll.to_degrees() } else { roll },
                if in_degrees {
                    pitch.to_degrees()
                } else {
                    pitch
                },
                if in_degrees { yaw.to_degrees() } else { yaw },
            ]
        } else {
            vec![
                if in_degrees {
                    self.longitude.to_degrees()
                } else {
                    self.longitude
                },
                if in_degrees {
                    self.latitude.to_degrees()
                } else {
                    self.latitude
                },
                self.altitude,
                self.velocity_east,
                self.velocity_north,
                -self.velocity_down, // Note: Down is negative in ENU
                if in_degrees { roll.to_degrees() } else { roll },
                if in_degrees {
                    pitch.to_degrees()
                } else {
                    pitch
                },
                if in_degrees { yaw.to_degrees() } else { yaw },
            ]
        }
    }
    pub fn get_velocity(&self) -> Vector3<f64> {
        if self.coordinate_convention {
            Vector3::new(self.velocity_north, self.velocity_east, self.velocity_down)
        } else {
            // ENU convention: North is East, East is North, Down is negative
            Vector3::new(self.velocity_east, self.velocity_north, -self.velocity_down)
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
    /// use strapdown::{StrapdownState, IMUData};
    /// use nalgebra::Vector3;
    /// let mut state = StrapdownState::new();
    /// let imu_data = IMUData::new_from_vector(
    ///    Vector3::new(0.0, 0.0, -9.81), // free fall acceleration in m/s^2
    ///    Vector3::new(0.0, 0.0, 0.0) // No rotation
    /// );
    /// let dt = 0.1; // Example time step in seconds
    /// state.forward(&imu_data, dt);
    /// ```
    pub fn forward(&mut self, imu_data: &IMUData, dt: f64) {
        // Extract the attitude matrix from the current state
        let c_0: Rotation3<f64> = self.attitude;
        // Attitude update; Equation 5.46
        let c_1: Matrix3<f64> = self.attitude_update(&imu_data.gyro, dt);
        // Specific force transformation; Equation 5.47
        let f: Vector3<f64> = 0.5 * (c_0.matrix() + c_1) * imu_data.accel;
        // Velocity update; Equation 5.54
        //let v_0: Vector3<f64> = self.get_velocity();
        let v_n_0 = self.velocity_north;
        let v_e_0 = self.velocity_east;
        let v_d_0 = self.velocity_down;
        //let v_1: Vector3<f64> = self.velocity_update(&f, dt);
        let v = self.velocity_update(&f, dt);
        let v_n_1 = v[0];
        let v_e_1 = v[1];
        let v_d_1 = v[2];
        // Position update; Equation 5.56
        let (r_n, r_e_0, _) = earth::principal_radii(&self.latitude, &self.altitude);
        let lat_0 = self.latitude;
        let alt_0 = self.altitude;
        // Altitude update
        self.altitude += 0.5 * (v_d_0 + v_d_1) * dt;
        // Latitude update
        let lat_1: f64 =
            self.latitude + 0.5 * (v_n_0 / (r_n + alt_0) + v_n_1 / (r_n + self.longitude)) * dt;
        // Longitude update
        let (_, r_e_1, _) = earth::principal_radii(&lat_1, &self.altitude);
        let lon_1: f64 = self.longitude
            + 0.5
                * (v_e_0 / ((r_e_0 + alt_0) * lat_0.cos())
                    + v_e_1 / ((r_e_1 + self.altitude) * lat_1.cos()))
                * dt;
        // Save updated position
        self.latitude = lat_1;
        self.longitude = lon_1;
        // Save updated attitude as rotation
        self.attitude = Rotation3::from_matrix(&c_1);
        // Save update velocity
        self.velocity_north = v_n_1;
        self.velocity_east = v_e_1;
        self.velocity_down = v_d_1;
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
    fn attitude_update(&self, gyros: &Vector3<f64>, dt: f64) -> Matrix3<f64> {
        let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
            &self.latitude,
            &self.altitude,
            &Vector3::from_vec(vec![
                self.velocity_north,
                self.velocity_east,
                self.velocity_down,
            ]),
        ));
        let rotation_rate: Matrix3<f64> =
            earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&self.latitude));
        let omega_ib: Matrix3<f64> = earth::vector_to_skew_symmetric(gyros);
        let c_1: Matrix3<f64> = self.attitude * (Matrix3::identity() + omega_ib * dt)
            - (rotation_rate + transport_rate) * self.attitude * dt;
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
    fn velocity_update(&self, f: &Vector3<f64>, dt: f64) -> Vector3<f64> {
        let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
            &self.latitude,
            &self.altitude,
            &Vector3::from_vec(vec![
                self.velocity_north,
                self.velocity_east,
                self.velocity_down,
            ]),
        ));
        let rotation_rate: Matrix3<f64> =
            earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&self.latitude));
        let r = earth::ecef_to_lla(&self.latitude, &self.longitude);
        // let grav: Vector3<f64> = earth::gravitation(&self.position[0], &self.position[1], &self.position[2]);
        let velocity: Vector3<f64> =
            Vector3::new(self.velocity_north, self.velocity_east, self.velocity_down);
        velocity + (f - r * (transport_rate + 2.0 * rotation_rate) * velocity) * dt
    }
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

/// tester function for building bindings
pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

// Note: nalgebra does not yet have a well developed testing framework for directly comparing
// nalgebra data structures. Rather than directly comparing, check the individual items.
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_strapdown_state_new() {
        let state = StrapdownState::new();
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
        let state = StrapdownState::new();
        let state_vector = state.to_vector(true);
        let zeros: SVector<f64, 9> = SVector::zeros();
        assert_eq!(state_vector, zeros);
    }
    #[test]
    fn test_new_from_vector() {
        let roll: f64 = 15.0;
        let pitch: f64 = 45.0;
        let yaw: f64 = 90.0;
        let state_vector: SVector<f64, 9> = nalgebra::vector![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            roll.to_radians(),
            pitch.to_radians(),
            yaw.to_radians()
        ];
        let state: StrapdownState = StrapdownState::new_from_vector(state_vector, false);
        assert_eq!(state.latitude, 0.0);
        assert_eq!(state.longitude, 0.0);
        assert_eq!(state.altitude, 0.0);
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_down, 0.0);
        let eulers = state.attitude.euler_angles();
        assert_approx_eq!(eulers.0.to_degrees(), roll, 1e-6);
        assert_approx_eq!(eulers.1.to_degrees(), pitch, 1e-6);
        assert_approx_eq!(eulers.2.to_degrees(), yaw, 1e-6);
    }
    #[test]
    fn test_dcm_to_vector() {
        let attitude = Rotation3::from_euler_angles(0.0, 0.0, 0.0);
        let state: StrapdownState = StrapdownState::new_from_components(
            0.0,
            (1.0_f64).to_degrees(),
            2.0,
            0.0,
            15.0,
            45.0,
            attitude,
            true, // angles provided in degrees
            true, // NED convention
        );
        let state_vector = state.to_vector(true);
        assert_eq!(state_vector[0], 0.0);
        assert_eq!(state_vector[1], (1.0_f64).to_degrees());
        assert_eq!(state_vector[2], 2.0);
        assert_eq!(state_vector[3], 0.0);
        assert_eq!(state_vector[4], 15.0);
        assert_eq!(state_vector[5], 45.0);
        assert_eq!(state_vector[6], 0.0);
        assert_eq!(state_vector[7], 0.0);
        assert_eq!(state_vector[8], 0.0);
    }
    #[test]
    // Test rotation (attitude) matrix creation and euler extraction
    fn test_attitude_matrix_euler_consistency() {
        let roll: f64 = 0.1;
        let pitch: f64 = 0.2;
        let yaw: f64 = 0.3;
        let attitude = Rotation3::from_euler_angles(roll, pitch, yaw);
        let (r, p, y) = attitude.euler_angles();
        assert_approx_eq!(r, roll, 1e-10);
        assert_approx_eq!(p, pitch, 1e-10);
        assert_approx_eq!(y, yaw, 1e-10);
    }

    #[test]
    // Test the forward mechanization (basic structure, not full dynamics)
    fn test_forward_freefall_stub() {
        let attitude = Rotation3::identity();
        let state = StrapdownState::new_from_components(
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
    fn test_hover_stub() {
        let attitude = Rotation3::identity();
        let state = StrapdownState::new_from_components(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, attitude, false, true, // NED convention
        );
        // This is a stub: actual hover logic should be tested in integration with the mechanization equations.
        assert_eq!(state.velocity_north, 0.0);
        assert_eq!(state.velocity_east, 0.0);
        assert_eq!(state.velocity_down, 0.0);
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
}
