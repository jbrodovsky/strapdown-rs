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
//! Primarily built off of three crate dependencies:
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

use std::fmt::{Debug, Display};
use angle::Deg; // might be overkill to need this entire create just for this
use nalgebra::{Matrix3, Rotation3, SVector, Vector3};

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
            self.accel[0], self.accel[1], self.accel[2],
            self.gyro[0], self.gyro[1], self.gyro[2]
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
    pub position: Vector3<f64>, // latitude (rad), longitude (rad), altitude (m)
    pub velocity: Vector3<f64>, // velocity in NED/ENU frame (m/s)
    pub attitude: Rotation3<f64>, // attitude in the form of a rotation matrix (direction cosine matrix; roll-pitch-yaw Euler angles)
//    ned: bool,                    // NED or ENU; true for NED, false for ENU
}
impl Debug for StrapdownState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (roll, pitch, yaw) = self.attitude.euler_angles();
        write!(
            f,
            "StrapdownState {{ position: [{:.4}, {:.4}, {:.4}], velocity: [{:.4}, {:.4}, {:.4}], attitude: [{:.4}, {:.4}, {:.4}] }}",
            self.position[0].to_degrees(),
            self.position[1].to_degrees(),
            self.position[2],
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
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
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            attitude: Rotation3::identity(),
            //ned: true,
        }
    }
    /// Create a new StrapdownState from a position, velocity, and attitude vectors.
    /// 
    /// This function initializes a new StrapdownState instance with the given position, velocity, and attitude vectors.
    /// The position is in the form of latitude (degrees), longitude (degrees), and altitude (meters). The corresponding
    /// velocities are in the NED/ENU frame (meters per second) and the attitude is given as roll, pitch, and yaw angles 
    /// in degrees. 
    /// 
    /// # Arguments
    /// * `position` - A Vector3 representing the position in the form of latitude (degrees), longitude (degrees), and altitude (meters).
    /// * `velocity` - A Vector3 representing the velocity in the NED/ENU frame (meters per second).
    /// * `attitude` - A Vector3 representing the attitude in the form of roll, pitch, and yaw angles (degrees).
    ///  /// # Returns
    /// * A StrapdownState instance containing the position, velocity, and attitude.
    /// # Example
    /// ```rust
    /// use strapdown::StrapdownState;
    /// use nalgebra::Vector3;
    /// let position = Vector3::new(37.7749, -122.4194, 0.0); // San Francisco coordinates
    /// let velocity = Vector3::new(0.0, 0.0, 0.0); // No initial velocity
    /// let attitude = Vector3::new(0.0, 0.0, 0.0); // No initial attitude
    /// let state = StrapdownState::new_from(position, velocity, attitude);
    /// ```
    pub fn new_from(
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        attitude: Vector3<f64>,
    ) -> StrapdownState {
        StrapdownState {
            attitude: Rotation3::from_euler_angles(
                attitude[0].to_radians(),
                attitude[1].to_radians(),
                attitude[2].to_radians(),
            ),
            velocity,
            position,
            //ned: true,
        }
    }
    /// Create a StrapdownState from a vector of states
    ///
    /// Creates a StrapdownState object from a cannoincal strapdown state vector. The vector is in the 
    /// form of: $\left(p_n, p_e, p_d, v_n, v_e, v_d, \phi, \theta, \psi\right)$ where the angles for 
    /// attitude (roll, pitch, yaw), latitude ($p_n$), and longitude ($p_e$) are in radians.
    /// 
    /// # Arguments
    /// * `state` - A SVector of shape (9,) representing the strapdown state vector.
    ///
    /// # Returns
    /// * A StrapdownState instance containing the position, velocity, and attitude.
    /// 
    /// # Example
    /// ```rust
    /// use strapdown::StrapdownState;
    /// use nalgebra::SVector;
    /// 
    /// let state_vector: SVector<f64, 9> = SVector::from_vec(vec![37.7749, -122.4194, 0.0, 10.0, 0.0, 0.0, 0.0, 45.0, 0.0]); // Example state vector
    /// let strapdown_state = StrapdownState::new_from_vector(state_vector);
    /// ```
    pub fn new_from_vector(state: SVector<f64, 9>) -> StrapdownState {
        StrapdownState {
            attitude: Rotation3::from_euler_angles(state[6], state[7], state[8]),
            velocity: Vector3::new(state[3], state[4], state[5]),
            position: Vector3::new(state[0], state[1], state[2]),
            //ned: is_ned,
        }
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
    /// 
    /// # Example
    /// ```
    /// use strapdown::StrapdownState;
    /// use nalgebra::{Vector3, Matrix3};
    /// 
    /// let state = StrapdownState::new();
    /// let gyros = Vector3::new(0.1, 0.0, 0.0); // Example gyroscope data in rad/s
    /// let dt = 0.1; // Example time step in seconds
    /// let updated_attitude: Matrix3<f64> = state.attitude_update(&gyros, dt);
    /// ```
    fn attitude_update(&self, gyros: &Vector3<f64>, dt: f64) -> Matrix3<f64> {
        let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
            &self.position[0],
            &self.position[2],
            &self.velocity,
        ));
        let rotation_rate: Matrix3<f64> =
            earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&self.position[0]));
        let omega_ib: Matrix3<f64> = earth::vector_to_skew_symmetric(gyros);
        let c_1: Matrix3<f64> = &self.attitude * (Matrix3::identity() + omega_ib * dt)
            - (rotation_rate + transport_rate) * &self.attitude * dt;
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
    /// 
    /// # Example
    /// ```
    /// use strapdown::StrapdownState;
    /// use nalgebra::{Vector3, Matrix3};
    /// let state = StrapdownState::new();
    /// let f_1 = Vector3::new(0.0, 0.0, -9.81); // Example specific force vector in m/s^2; This is gravitational freefall
    /// let dt = 0.1; // Example time step in seconds
    /// let updated_velocity: Vector3<f64> = state.velocity_update(&f_1, dt);
    /// ```
    fn velocity_update(&self, f_1: &Vector3<f64>, dt: f64) -> Vector3<f64> {
        let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
            &self.position[0],
            &self.position[2],
            &self.velocity,
        ));
        let rotation_rate: Matrix3<f64> =
            earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&self.position[0]));
        let r = earth::ecef_to_lla(&self.position[0], &self.position[1]);
        // let grav: Vector3<f64> = earth::gravitation(&self.position[0], &self.position[1], &self.position[2]);
        let v_1: Vector3<f64> = &self.velocity
            + (f_1 - r * (transport_rate + 2.0 * rotation_rate) * &self.velocity) * dt;
        v_1
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
        let f_1: Vector3<f64> = 0.5 * (c_0.matrix() + c_1) * imu_data.accel;
        // Velocity update; Equation 5.54
        let v_0: Vector3<f64> = self.velocity;
        let v_1: Vector3<f64> = self.velocity_update(&f_1, dt);
        // Position update; Equation 5.56
        let (r_n, r_e_0, _) = earth::principal_radii(&self.position[0], &self.position[2]);
        let lat_0: f64 = self.position[0].to_radians();
        let alt_0: f64 = self.position[2];
        // Altitude update
        self.position[2] += 0.5 * (v_0[2] + v_1[2]) * dt;
        // Latitude update
        let lat_1: f64 = &self.position[0].to_radians()
            + 0.5 * (v_0[0] / (r_n + alt_0) + v_1[0] / (r_n + &self.position[2])) * dt;
        // Longitude update
        let (_, r_e_1, _) = earth::principal_radii(&lat_1, &self.position[2]);
        let lon_1: f64 = self.position[1].to_radians()
            + 0.5
                * (v_0[1] / ((r_e_0 + alt_0) * lat_0.cos())
                    + v_1[1] / ((r_e_1 + &self.position[2]) * &lat_1.cos()))
                * dt;
        // Update position to degrees
        self.position[0] = lat_1.to_degrees();
        self.position[1] = lon_1.to_degrees();
        // Update attitude to rotation
        self.attitude = Rotation3::from_matrix(&c_1);
        // Update velocity
        self.velocity = v_1;
    }
    /// Convert the StrapdownState to a one dimensional vector, nalgebra style
    /// 
    /// StrapdownState internally stores the attitude as a direction cosine matrix (DCM) and the position 
    /// and velocity as vectors. Outside of this object, it is useful to have the navigation state in a 
    /// traditional cannonical vector form for use in various filters and algorithms. This function converts
    /// the internal state to a one dimensional vector in the form of: $\left[p_n, p_e, p_d, v_n, v_e, v_d, \phi, \theta, \psi \right]$
    /// 
    /// # Arguments
    /// * `in_degrees` - A boolean indicating whether to return the angles in degrees (true) or radians (false).
    /// 
    /// # Returns
    /// * An SVector of shape (9,) representing the strapdown state vector.
    pub fn to_vector(&self, in_degrees: bool) -> SVector<f64, 9> {
        let mut state: SVector<f64, 9> = SVector::zeros();
        state[2] = self.position[2];
        state[3] = self.velocity[0];
        state[4] = self.velocity[1];
        state[5] = self.velocity[2];
        let (roll, pitch, yaw) = &self.attitude.euler_angles();
        if in_degrees {
            state[0] = self.position[0].to_degrees();
            state[1] = self.position[1].to_degrees();
            state[6] = Deg(roll.to_degrees()).0;
            state[7] = Deg(pitch.to_degrees()).0;
            state[8] = Deg(yaw.to_degrees()).0;
        } else {
            state[6] = *roll;
            state[7] = *pitch;
            state[8] = *yaw;
        }
        state
    }
    /// Convert the StrapdownState to a one dimensional vector, native vec (list) style
    /// 
    /// StrapdownState internally stores the attitude as a direction cosine matrix (DCM) and the position 
    /// and velocity as vectors. Outside of this object, it is useful to have the navigation state in a 
    /// traditional cannonical vector form for use in various filters and algorithms. This function converts
    /// the internal state to a one dimensional vector in the form of: $\left[p_n, p_e, p_d, v_n, v_e, v_d, \phi, \theta, \psi \right]$
    /// 
    /// # Arguments
    /// * `in_degrees` - A boolean indicating whether to return the angles in degrees (true) or radians (false).
    /// 
    /// # Returns
    /// * An SVector of shape (9,) representing the strapdown state vector.
    pub fn to_vec(&self, in_degrees: bool) -> Vec<f64> {
        let mut state: Vec<f64> = vec![0.0; 9];
        state[2] = self.position[2];
        state[3] = self.velocity[0];
        state[4] = self.velocity[1];
        state[5] = self.velocity[2];
        let (roll, pitch, yaw) = &self.attitude.euler_angles();
        if in_degrees {
            state[0] = self.position[0].to_degrees();
            state[1] = self.position[1].to_degrees();
            state[6] = Deg(roll.to_degrees()).0;
            state[7] = Deg(pitch.to_degrees()).0;
            state[8] = Deg(yaw.to_degrees()).0;
        } else {
            state[6] = *roll;
            state[7] = *pitch;
            state[8] = *yaw;
        }
        state
    }   
}
// Miscellaneous functions for wrapping angles

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
/// let angle = 3.0 * PI / 4.0; // radians
/// let wrapped_angle = wrap_to_pi(angle);
/// assert_eq!(wrapped_angle, -PI / 4.0); // 3π/4 radians wrapped to -π/4 radians
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
        assert_eq!(state.position, Vector3::zeros());
        assert_eq!(state.velocity, Vector3::zeros());
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
        let state_vector: SVector<f64, 9> = nalgebra::vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, roll.to_radians(), pitch.to_radians(), yaw.to_radians()];
        let state: StrapdownState = StrapdownState::new_from_vector(state_vector);
        assert_eq!(state.position, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(state.velocity, Vector3::new(0.0, 0.0, 0.0));
        let eulers = state.attitude.euler_angles();
        assert_approx_eq!(eulers.0.to_degrees(), roll, 1e-6);
        assert_approx_eq!(eulers.1.to_degrees(), pitch, 1e-6);
        assert_approx_eq!(eulers.2.to_degrees(), yaw, 1e-6);
    }
    #[test]
    fn test_dcm_to_vector() {
        let state: StrapdownState = StrapdownState::new_from(
            Vector3::new(0.0, 1.0, 2.0),
            Vector3::new(0.0, 15.0, 45.0),
            Vector3::new(0.0, 0.0, 0.0),
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
    // Test rotation
    fn test_attitude_update_no_motion() {
        let gyros = Vector3::new(0.0, 0.0, 0.0);
        let state: StrapdownState = StrapdownState::new();
        let dt = 1.0;
        let new_attitude = state.attitude_update(&gyros, dt);
        assert_approx_eq!(new_attitude[(0, 0)], 1.0, 1e-3);
        assert_approx_eq!(new_attitude[(0, 1)], 0.0, 1e-3);
        assert_approx_eq!(new_attitude[(0, 2)], 0.0, 1e-3);
        assert_approx_eq!(new_attitude[(1, 0)], 0.0, 1e-3);
        assert_approx_eq!(new_attitude[(1, 1)], 1.0, 1e-3);
        assert_approx_eq!(new_attitude[(1, 2)], 0.0, 1e-3);
        assert_approx_eq!(new_attitude[(2, 0)], 0.0, 1e-3);
        assert_approx_eq!(new_attitude[(2, 1)], 0.0, 1e-3);
        assert_approx_eq!(new_attitude[(2, 2)], 1.0, 1e-3);
    }
    #[test]
    fn test_attitude_update_yawing() {
        let gyros = Vector3::new(0.0, 0.0, 0.1);
        let state: StrapdownState = StrapdownState::new();
        let dt = 1.0;
        let new_attitude = state.attitude_update(&gyros, dt);
        let eulers = Rotation3::from_matrix(&new_attitude).euler_angles();
        assert_approx_eq!(eulers.0, 0.0, 1e-3);
        assert_approx_eq!(eulers.1, 0.0, 1e-3);
        assert_approx_eq!(eulers.2, 0.1, 1e-3);
    }
    #[test]
    fn test_attitude_update_rolling() {
        let gyros = Vector3::new(0.1, 0.0, 0.0);
        let state: StrapdownState = StrapdownState::new();
        let dt = 1.0;
        let new_attitude = state.attitude_update(&gyros, dt);
        let eulers = Rotation3::from_matrix(&new_attitude).euler_angles();
        assert_approx_eq!(eulers.0, 0.1, 1e-3);
        assert_approx_eq!(eulers.1, 0.0, 1e-3);
        assert_approx_eq!(eulers.2, 0.0, 1e-3);
    }
    #[test]
    fn test_attitude_update_pitching() {
        let gyros = Vector3::new(0.0, 0.1, 0.0);
        let state: StrapdownState = StrapdownState::new();
        let dt = 1.0;
        let new_attitude = state.attitude_update(&gyros, dt);
        let eulers = Rotation3::from_matrix(&new_attitude).euler_angles();
        assert_approx_eq!(eulers.0, 0.0, 1e-3);
        assert_approx_eq!(eulers.1, 0.1, 1e-3);
        assert_approx_eq!(eulers.2, 0.0, 1e-3);
    }

    #[test]
    // Test the forward mechanization
    fn test_forward_freefall() {
        let mut state: StrapdownState = StrapdownState::new_from(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
        );
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -earth::gravity(&0.0, &0.0)), // Currently configured as relative body-frame acceleration
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        state.forward(&imu_data, 1.0);
        assert_approx_eq!(state.velocity[0], 0.00, 1e-6);
        assert_approx_eq!(state.velocity[1], 0.0004, 1e-3);
        assert_approx_eq!(state.velocity[2], -earth::gravity(&0.0, &0.0), 1e-3);

        assert_approx_eq!(state.position[0], 0.0);
        assert_approx_eq!(state.position[1], 0.0);
        assert_approx_eq!(state.position[2], -earth::gravity(&0.0, &0.0) / 2.0, 1e-3);

        let attitude = state.attitude.matrix();
        assert_approx_eq!(&attitude[(0, 0)], 1.0, 1e-4);
        assert_approx_eq!(&attitude[(0, 1)], 0.0, 1e-4);
        assert_approx_eq!(&attitude[(0, 2)], 0.0, 1e-4);
        assert_approx_eq!(&attitude[(1, 0)], 0.0, 1e-4);
        assert_approx_eq!(&attitude[(1, 1)], 1.0, 1e-4);
        assert_approx_eq!(&attitude[(1, 2)], 0.0, 1e-4);
        assert_approx_eq!(&attitude[(2, 0)], 0.0, 1e-4);
        assert_approx_eq!(&attitude[(2, 1)], 0.0, 1e-4);
        assert_approx_eq!(&attitude[(2, 2)], 1.0, 1e-4);
    }
    #[test]
    fn test_hover() {
        let mut state: StrapdownState = StrapdownState::new_from(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
        );
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0), // Currently configured as relative body-frame acceleration
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        state.forward(&imu_data, 1.0);
        assert_approx_eq!(state.velocity[0], 0.0, 0.1);
        assert_approx_eq!(state.velocity[1], 0.0, 0.1);
        assert_approx_eq!(state.velocity[2], 0.0, 0.1);
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
        assert_eq!(super::wrap_to_pi(3.0 * std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_pi(-3.0 * std::f64::consts::PI), -std::f64::consts::PI);
        assert_eq!(super::wrap_to_pi(0.0), 0.0);
        assert_eq!(super::wrap_to_pi(std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_pi(-std::f64::consts::PI), -std::f64::consts::PI);
    }
    #[test]
    fn test_wrap_to_2pi() {
        assert_eq!(super::wrap_to_2pi(7.0 * std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_2pi(-5.0 * std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_2pi(0.0), 0.0);
        assert_eq!(super::wrap_to_2pi(std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_2pi(-std::f64::consts::PI), std::f64::consts::PI);
    }
}