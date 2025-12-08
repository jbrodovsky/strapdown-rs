//! Inertial Navigation Filters
//!
//! This module contains implementations of various inertial navigation filters, including
//! Kalman filters and particle filters. These filters are used to estimate the state of a
//! strapdown inertial navigation system based on IMU measurements and other sensor data.
//! The filters use the strapdown equations (provided by the StrapdownState) to propagate
//! the state in the local level frame.
//!
//! Currently, this module contains an implementation of a full-state Unscented Kalman Filter
//! (UKF) and a full-state particle filter. For completeness, an Extended Kalman Filter (EKF)
//! should be included, however, a strapdown EKF INS is typically implemented as an error state
//! filter, which would require a slightly different architecture.
//!
//! Contained in this module is also a simple standard position measurement model for both
//! the UKF and particle filter. This model is used to update the state based on position
//! measurements in the local level frame (i.e. a GPS fix).
use crate::earth::METERS_TO_DEGREES;
use crate::linalg::{matrix_square_root, robust_spd_solve, symmetrize};
use crate::{IMUData, StrapdownState, forward, wrap_to_2pi, wrap_to_180, wrap_to_360};

use std::any::Any;
use std::convert::From;
use std::fmt::{self, Debug, Display};

use nalgebra::{DMatrix, DVector, Rotation3};
use rand;
use rand_distr::Distribution;
// ==== Measurement Models ========================================================
// Below is a set of generic measurement models for the UKF and particle filter.
// These models provide a vec of "expected measurements" based on the sigma points
// location. When used in a filter, you should simultaneously iterate through the
// list of sigma points or particles and the expected measurements in order to
// calculate the innovation matrix or the particle weighting.
// ================================================================================
/// Generic measurement model trait for all filters
pub trait MeasurementModel: Any {
    /// Allow runtime downcasting of boxed trait objects.
    fn as_any(&self) -> &dyn Any;
    /// Allow mutable runtime downcasting of boxed trait objects.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the dimensionality of the measurement vector.
    fn get_dimension(&self) -> usize;
    /// Get the measurement vector
    fn get_vector(&self) -> DVector<f64>;
    /// Get the measurement noise covariance matrix
    fn get_noise(&self) -> DMatrix<f64>;
    /// Get the measurement sigma points, performs the mapping between the state space
    /// and the measurement space.
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64>;
}
/// GPS position measurement model
#[derive(Clone, Debug, Default)]
pub struct GPSPositionMeasurement {
    // <-- Check this model for degree/radian consistency
    /// latitude in degrees
    pub latitude: f64,
    /// longitude in degrees
    pub longitude: f64,
    /// altitude in meters
    pub altitude: f64,
    /// noise standard deviation in meters
    pub horizontal_noise_std: f64,
    /// vertical noise standard deviation in meters
    pub vertical_noise_std: f64,
}
impl Display for GPSPositionMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPSPositionMeasurement(lat: {}, lon: {}, alt: {}, horiz_noise: {}, vert_noise: {})",
            self.latitude,
            self.longitude,
            self.altitude,
            self.horizontal_noise_std,
            self.vertical_noise_std
        )
    }
}
impl MeasurementModel for GPSPositionMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        3 // latitude, longitude, altitude
    }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.latitude.to_radians(),
            self.longitude.to_radians(),
            self.altitude,
        ])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
            self.vertical_noise_std.powi(2),
        ]))
    }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(3, state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[0];
            measurement_sigma_points[(1, i)] = sigma_point[1];
            measurement_sigma_points[(2, i)] = sigma_point[2];
        }
        measurement_sigma_points
    }
}
/// GPS Velocity measurement model
#[derive(Clone, Debug, Default)]
pub struct GPSVelocityMeasurement {
    /// Northward velocity in m/s
    pub northward_velocity: f64,
    /// Eastward velocity in m/s
    pub eastward_velocity: f64,
    /// Downward velocity in m/s
    pub downward_velocity: f64,
    /// noise standard deviation in m/s
    pub horizontal_noise_std: f64,
    /// vertical noise standard deviation in m/s
    pub vertical_noise_std: f64,
}
impl Display for GPSVelocityMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPSVelocityMeasurement(north: {}, east: {}, down: {}, horiz_noise: {}, vert_noise: {})",
            self.northward_velocity,
            self.eastward_velocity,
            self.downward_velocity,
            self.horizontal_noise_std,
            self.vertical_noise_std
        )
    }
}
impl MeasurementModel for GPSVelocityMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        3 // northward, eastward, downward velocity
    }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.northward_velocity,
            self.eastward_velocity,
            self.downward_velocity,
        ])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.horizontal_noise_std.powi(2),
            self.horizontal_noise_std.powi(2),
            self.vertical_noise_std.powi(2),
        ]))
    }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(3, state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[3];
            measurement_sigma_points[(1, i)] = sigma_point[4];
            measurement_sigma_points[(2, i)] = sigma_point[5];
        }
        measurement_sigma_points
    }
}

/// GPS Position and Velocity measurement model
#[derive(Clone, Debug, Default)]
pub struct GPSPositionAndVelocityMeasurement {
    /// latitude in degrees
    pub latitude: f64,
    /// longitude in degrees
    pub longitude: f64,
    /// altitude in meters
    pub altitude: f64,
    /// Northward velocity in m/s
    pub northward_velocity: f64,
    /// Eastward velocity in m/s
    pub eastward_velocity: f64,
    /// Downward velocity in m/s
    // pub downward_velocity: f64, // GPS speed measurements do not typically provide vertical velocity
    /// noise standard deviation in meters for position
    pub horizontal_noise_std: f64,
    /// vertical noise standard deviation in meters for position
    pub vertical_noise_std: f64,
    /// noise standard deviation in m/s for velocity
    pub velocity_noise_std: f64,
}
impl MeasurementModel for GPSPositionAndVelocityMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        5 // latitude, longitude, altitude, northward velocity, eastward velocity
    }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.latitude.to_radians(),
            self.longitude.to_radians(),
            self.altitude,
            self.northward_velocity,
            self.eastward_velocity,
        ])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
            self.vertical_noise_std.powi(2),
            self.velocity_noise_std.powi(2),
            self.velocity_noise_std.powi(2),
        ]))
    }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(5, state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[0];
            measurement_sigma_points[(1, i)] = sigma_point[1];
            measurement_sigma_points[(2, i)] = sigma_point[2];
            measurement_sigma_points[(3, i)] = sigma_point[3];
            measurement_sigma_points[(4, i)] = sigma_point[4];
        }
        measurement_sigma_points
    }
}

/// A relative relative altitude measurement derived from barometric pressure.
/// Note that this measurement model is an altitude measurement derived from
/// a barometric altimeter and not a direct calculation of altitude from the
/// barometric pressure.
#[derive(Clone, Debug, Default)]
pub struct RelativeAltitudeMeasurement {
    /// Measured relative altitude in meters
    pub relative_altitude: f64,
    /// Reference pressure in Pa
    pub reference_altitude: f64,
}
impl Display for RelativeAltitudeMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RelativeAltitudeMeasurement(rel_alt: {}, ref_alt: {})",
            self.relative_altitude, self.reference_altitude
        )
    }
}
impl MeasurementModel for RelativeAltitudeMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        1 // relative altitude
    }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.relative_altitude + self.reference_altitude])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![5.0])) // 1 mm noise
    }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points =
            DMatrix::<f64>::zeros(self.get_dimension(), state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[2];
        }
        measurement_sigma_points
    }
}
/// Basic strapdown state parameters for the UKF and particle filter initialization.
#[derive(Clone, Debug, Default)]
pub struct InitialState {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub northward_velocity: f64,
    pub eastward_velocity: f64,
    pub downward_velocity: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub in_degrees: bool,
    pub is_enu: bool,
}
impl InitialState {
    /// Creates a new InitialState struct and checks the validity of the input parameters.
    ///
    /// # Arguments
    /// * `latitude` - The initial latitude of the strapdown state.
    /// * `longitude` - The initial longitude of the strapdown state.
    /// * `altitude` - The initial altitude of the strapdown state [-10,000, +30,000] meter above MSL.
    /// * `northward_velocity` - The initial northward velocity of the strapdown state.
    /// * `eastward_velocity` - The initial eastward velocity of the strapdown state.
    /// * `downward_velocity` - The initial downward velocity of the strapdown state.
    /// * `roll` - The initial roll of the strapdown state.
    /// * `pitch` - The initial pitch of the strapdown state.
    /// * `yaw` - The initial yaw of the strapdown state.
    /// * `in_degrees` - Whether the input angles are in degrees or radians.
    /// * `is_enu` - Whether the local level frame is East-North-Up (ENU) or North-East-Down (NED). If None, defaults to ENU.
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude: f64,
        northward_velocity: f64,
        eastward_velocity: f64,
        downward_velocity: f64,
        mut roll: f64,
        mut pitch: f64,
        mut yaw: f64,
        in_degrees: bool,
        is_enu: Option<bool>,
    ) -> Self {
        let latitude = if in_degrees {
            assert!(
                latitude >= -90.0 && latitude <= 90.0,
                "Latitude must be between -90 and +90 degrees"
            );
            latitude
        } else {
            let lat_deg = latitude.to_degrees();
            assert!(
                lat_deg >= -90.0 && lat_deg <= 90.0,
                "Latitude must be between -90 and +90 degrees"
            );
            latitude
        };
        let longitude = if in_degrees {
            assert!(
                wrap_to_180(longitude) >= -180.0 && wrap_to_180(longitude) <= 180.0,
                "Longitude must be between -180 and +180 degrees"
            );
            wrap_to_180(longitude)
        } else {
            let lon_deg = longitude.to_degrees();
            assert!(
                lon_deg >= -180.0 && lon_deg <= 180.0,
                "Longitude must be between -180 and +180 degrees"
            );
            longitude
        };
        let is_enu = is_enu.unwrap_or(true);
        let altitude = if is_enu {
            assert!(
                altitude >= -10000.0 && altitude <= 30000.0,
                "Altitude must be between -10,000 and +30,000 meters in ENU"
            );
            altitude
        } else {
            assert!(
                altitude >= -30000.0 && altitude <= 10000.0,
                "Altitude must be between 10,000 and -30,000 meters in NED"
            );
            altitude
        };
        if in_degrees {
            roll = wrap_to_360(roll).to_radians();
            pitch = wrap_to_360(pitch).to_radians();
            yaw = wrap_to_360(yaw).to_radians();
        } else {
            roll = wrap_to_2pi(roll);
            pitch = wrap_to_2pi(pitch);
            yaw = wrap_to_2pi(yaw);
        }
        InitialState {
            latitude,
            longitude,
            altitude,
            northward_velocity,
            eastward_velocity,
            downward_velocity,
            roll,
            pitch,
            yaw,
            in_degrees,
            is_enu,
        }
    }
}

/// Navigation filter trait for common functions across all navigation filters
pub trait NavigationFilter {
    /// Predicts the state using the strapdown equations and IMU measurements.
    ///
    /// The IMU measurements are used to update the strapdown state in the local level frame.
    /// The IMU measurements are assumed to be in the body frame.
    ///
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the state with (e.g. relative accelerations (m/s^2) and angular rates (rad/s)).
    /// * `dt` - The time step for the propagation.
    fn predict(&mut self, imu_data: IMUData, dt: f64);
    /// Perform the measurement update step.
    ///
    /// This method updates the navigation filter based on the measurement provided. The
    /// measurement model is specific to a given implementation of a Bayesian filter and
    /// must be provided by the user as the model determines the shape and quantities of
    /// the measurement vector, measurement covariance matrix, and where applicable and
    /// the measurement sigma points. Measurement models are implemented as traits to
    /// allow for the filter to consume them generically.
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M);
    /// Get the current state estimate as determined by the underlying logic of the filter.
    fn get_estimate(&self) -> DVector<f64>;
    /// Get the state estimate's certainty  as determined by the underlying logic of the
    /// filter.
    fn get_certainty(&self) -> DMatrix<f64>;
}
/// Strapdown Unscented Kalman Filter Inertial Navigation Filter
///
/// This filter uses the Unscented Kalman Filter (UKF) algorithm to estimate the state of a
/// strapdown inertial navigation system. It uses the strapdown equations to propagate the state
/// in the local level frame based on IMU measurements in the body frame. The filter also uses
/// a generic measurement model to update the state based on measurements in the local level frame.
///
/// Because of the generic nature of both the UKF and this toolbox, the filter requires the user to
/// implement the measurement model(s). The measurement model must calculate the measurement sigma points
/// ($\mathcal{Z} = h(\mathcal{X})$) and the measurement noise matrix ($R$) for the filter. Some basic
/// GNSS-based are provided in this module (position, velocity, position and velocity, barometric altitude).
/// In a given scenario's implementation, the user should then call these measurement models. Please see the
/// `sim` module for a reference implementation of a full state UKF INS with a position and velocity GPS-based
/// measurement model and barometric altitude measurement model.
///
/// Note that, internally, angles are always stored in radians (both for the attitude and the position),
/// however, the user can choose to convert them to degrees when retrieving the state vector and the UKF
/// and underlying strapdown state can be constructed from data in degrees by using the boolean `in_degrees`
/// toggle where applicable. Generally speaking, the design of this crate is such that methods that expect
/// a WGS84 coordinate (e.g. latitude or longitude) will expect the value in degrees, whereas trigonometric
/// functions (e.g. sine, cosine, tangent) will expect the value in radians.
#[derive(Clone)]
pub struct UnscentedKalmanFilter {
    mean_state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    lambda: f64,
    state_size: usize,
    weights_mean: DVector<f64>,
    weights_cov: DVector<f64>,
    is_enu: bool,
}
impl Debug for UnscentedKalmanFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UKF")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("lambda", &self.lambda)
            .field("state_size", &self.state_size)
            .finish()
    }
}
impl Display for UnscentedKalmanFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnscentedKalmanFilter")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("lambda", &self.lambda)
            .field("state_size", &self.state_size)
            .finish()
    }
}
impl UnscentedKalmanFilter {
    /// Creates a new UnscentedKalmanFilter with the given initial state, biases, covariance, process noise,
    /// any additional other states, and UnscentedKalmanFilter hyper parameters.
    ///
    /// # Arguments
    /// * `position` - The initial position of the strapdown state.
    /// * `velocity` - The initial velocity of the strapdown state.
    /// * `attitude` - The initial attitude of the strapdown state.
    /// * `imu_biases` - The initial IMU biases.
    /// * `other_states` - Any additional states the filter is estimating (ex: measurement or sensor bias).
    /// * `covariance_diagonal` - The initial covariance diagonal.
    /// * `process_noise_diagonal` - The process noise diagonal.
    /// * `alpha` - The alpha parameter for the UKF.
    /// * `beta` - The beta parameter for the UKF.
    /// * `kappa` - The kappa parameter for the UKF.
    /// * `in_degrees` - Whether the input vectors are in degrees or radians.
    ///
    /// # Returns
    /// * A new UnscentedKalmanFilter struct.
    pub fn new(
        initial_state: InitialState,
        imu_biases: Vec<f64>,
        other_states: Option<Vec<f64>>,
        covariance_diagonal: Vec<f64>,
        process_noise: DMatrix<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> UnscentedKalmanFilter {
        assert!(
            process_noise.nrows() == process_noise.ncols(),
            "Process noise matrix must be square"
        );
        let mut mean = if initial_state.in_degrees {
            vec![
                initial_state.latitude.to_radians(),
                initial_state.longitude.to_radians(),
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.downward_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        } else {
            vec![
                initial_state.latitude,
                initial_state.longitude,
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.downward_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        };
        mean.extend(imu_biases);
        if let Some(ref other_states) = other_states {
            mean.extend(other_states.iter().cloned());
        }
        assert!(
            mean.len() >= 15,
            "Expected a canonical state vector of at least 15 states (position, velocity, attitude, imu biases)"
        );
        assert!(
            mean.len() == covariance_diagonal.len(),
            "{}",
            &format!(
                "Mean vector and covariance diagonal must be of the same size (mean: {}, covariance_diagonal: {})",
                mean.len(),
                covariance_diagonal.len()
            )
        );
        let state_size = mean.len();
        let mean_state = DVector::from_vec(mean);
        let covariance = DMatrix::<f64>::from_diagonal(&DVector::from_vec(covariance_diagonal));
        assert!(
            covariance.shape() == (state_size, state_size),
            "Covariance matrix must be square"
        );
        assert!(
            covariance.shape() == process_noise.shape(),
            "Covariance and process noise must be of the same size"
        );
        let lambda = alpha * alpha * (state_size as f64 + kappa) - state_size as f64;
        let mut weights_mean = DVector::zeros(2 * state_size + 1);
        let mut weights_cov = DVector::zeros(2 * state_size + 1);
        weights_mean[0] = lambda / (state_size as f64 + lambda);
        weights_cov[0] = lambda / (state_size as f64 + lambda) + (1.0 - alpha * alpha + beta);
        for i in 1..(2 * state_size + 1) {
            let w = 1.0 / (2.0 * (state_size as f64 + lambda));
            weights_mean[i] = w;
            weights_cov[i] = w;
        }
        UnscentedKalmanFilter {
            mean_state,
            covariance,
            process_noise,
            lambda,
            state_size,
            weights_mean,
            weights_cov,
            is_enu: initial_state.is_enu,
        }
    }
    pub fn get_sigma_points(&self) -> DMatrix<f64> {
        let p = (self.state_size as f64 + self.lambda) * self.covariance.clone();
        let sqrt_p = matrix_square_root(&p);
        let mu = self.mean_state.clone();
        let mut pts = DMatrix::<f64>::zeros(self.state_size, 2 * self.state_size + 1);
        pts.column_mut(0).copy_from(&mu);
        for i in 0..sqrt_p.ncols() {
            pts.column_mut(i + 1).copy_from(&(&mu + sqrt_p.column(i)));
            pts.column_mut(i + 1 + self.state_size)
                .copy_from(&(&mu - sqrt_p.column(i)));
        }
        pts
    }
    fn robust_kalman_gain(
        &mut self,
        cross_covariance: &DMatrix<f64>,
        s: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        // Solve S Kᵀ = P_xzᵀ  => K = (S^{-1} P_xz)ᵀ
        let kt = robust_spd_solve(&symmetrize(s), &cross_covariance.transpose());
        kt.transpose()
    }
}
impl NavigationFilter for UnscentedKalmanFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        // Propagate the strapdown state using the strapdown equations
        let mut sigma_points = self.get_sigma_points();
        for i in 0..sigma_points.ncols() {
            let mut sigma_point_vec = sigma_points.column(i).clone_owned();
            let mut state = StrapdownState {
                latitude: sigma_point_vec[0],
                longitude: sigma_point_vec[1],
                altitude: sigma_point_vec[2],
                velocity_north: sigma_point_vec[3],
                velocity_east: sigma_point_vec[4],
                velocity_down: sigma_point_vec[5],
                attitude: Rotation3::from_euler_angles(
                    sigma_point_vec[6],
                    sigma_point_vec[7],
                    sigma_point_vec[8],
                ),
                is_enu: self.is_enu,
            };
            let accel_biases = if self.state_size >= 15 {
                DVector::from_vec(vec![
                    sigma_point_vec[9],
                    sigma_point_vec[10],
                    sigma_point_vec[11],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let gyro_biases = if self.state_size >= 15 {
                DVector::from_vec(vec![
                    sigma_point_vec[12],
                    sigma_point_vec[13],
                    sigma_point_vec[14],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let imu_data = IMUData {
                accel: imu_data.accel - &accel_biases,
                gyro: imu_data.gyro - &gyro_biases,
            };
            // println!("propagating: lat {}  lon {}", state.latitude.to_degrees(), state.longitude.to_degrees());
            forward(&mut state, imu_data, dt);
            // Update the sigma point with the new state
            sigma_point_vec[0] = state.latitude;
            sigma_point_vec[1] = state.longitude;
            sigma_point_vec[2] = state.altitude;
            sigma_point_vec[3] = state.velocity_north;
            sigma_point_vec[4] = state.velocity_east;
            sigma_point_vec[5] = state.velocity_down;
            sigma_point_vec[6] = state.attitude.euler_angles().0; // Roll
            sigma_point_vec[7] = state.attitude.euler_angles().1; // Pitch
            sigma_point_vec[8] = state.attitude.euler_angles().2; // Yaw
            sigma_points.set_column(i, &sigma_point_vec);
        }
        // Update the mean state as mu_bar
        let mut mu_bar = DVector::<f64>::zeros(self.state_size);
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            mu_bar += self.weights_mean[i] * sigma_point;
        }
        // Update the covariance as P_bar
        let mut p_bar = DMatrix::<f64>::zeros(self.state_size, self.state_size);
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            let diff = sigma_point - &mu_bar;
            p_bar += self.weights_cov[i] * &diff * &diff.transpose();
        }
        // Add process noise to the covariance
        p_bar += &self.process_noise;
        // Update the mean state and covariance
        self.mean_state = mu_bar;
        self.covariance = symmetrize(&p_bar);
    }
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        let measurement_sigma_points = measurement.get_sigma_points(&self.get_sigma_points());
        // Calculate expected measurement
        let mut z_hat = DVector::<f64>::zeros(measurement.get_dimension());
        for (i, sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            z_hat += self.weights_mean[i] * sigma_point;
        }
        // Calculate innovation matrix S
        let mut s = DMatrix::<f64>::zeros(measurement.get_dimension(), measurement.get_dimension());
        //for i in 0..measurement_sigma_points.len() {
        for (i, sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            let diff = sigma_point - &z_hat;
            s += self.weights_cov[i] * &diff * &diff.transpose();
        }
        s += measurement.get_noise();
        // Calculate the cross-covariance
        let sigma_points = self.get_sigma_points();
        let mut cross_covariance =
            DMatrix::<f64>::zeros(self.state_size, measurement.get_dimension());
        for (i, measurement_sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            let measurement_diff = measurement_sigma_point - &z_hat;
            let state_diff = sigma_points.column(i) - &self.mean_state;
            cross_covariance += self.weights_cov[i] * state_diff * measurement_diff.transpose();
        }
        // // Calculate the Kalman gain
        // let s_inv = match s.clone().try_inverse() {
        //     Some(inv) => inv,
        //     None => panic!("Innovation matrix is singular"),
        // };
        // let k = &cross_covariance * &s_inv;
        // // check that the kalman gain and measurement diff are compatible to multiply
        // if k.ncols() != measurement.get_dimension() {
        //     panic!("Kalman gain and measurement differential are not compatible");
        // }
        // K = P_xz * S^{-1} without forming S^{-1}
        let k = self.robust_kalman_gain(&cross_covariance, &s);
        // Update the mean and covariance
        self.mean_state += &k * (measurement.get_vector() - &z_hat);
        // wrap attitude angles to 2pi
        // TODO: #30 Refactor attitude angles to use a more robust representation
        self.mean_state[6] = wrap_to_2pi(self.mean_state[6]);
        self.mean_state[7] = wrap_to_2pi(self.mean_state[7]);
        self.mean_state[8] = wrap_to_2pi(self.mean_state[8]);
        self.covariance -= &k * &s * &k.transpose();
        // Re-symmetrize to fight round-off
        self.covariance = 0.5 * (&self.covariance + self.covariance.transpose());
    }
    fn get_estimate(&self) -> DVector<f64> {
        self.mean_state.clone()
    }
    fn get_certainty(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Particle {
    /// The strapdown state of the particle
    pub nav_state: StrapdownState,
    /// Any additional other states
    pub other_states: Option<DVector<f64>>,
    /// State length dimension
    pub state_size: usize,
    /// The weight of the particle
    pub weight: f64,
}
impl Display for Particle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Particle")
            .field("latitude", &self.nav_state.latitude.to_degrees())
            .field("longitude", &self.nav_state.longitude.to_degrees())
            .field("altitude", &self.nav_state.altitude)
            .field("velocity_north", &self.nav_state.velocity_north)
            .field("velocity_east", &self.nav_state.velocity_east)
            .field("velocity_down", &self.nav_state.velocity_down)
            .field("roll", &self.nav_state.attitude.euler_angles().0)
            .field("pitch", &self.nav_state.attitude.euler_angles().1)
            .field("yaw", &self.nav_state.attitude.euler_angles().2)
            .field("weight", &self.weight)
            .finish()
    }
}
impl Particle {
    /// Create a new particle with the given strapdown state, other states, and weight.
    pub fn new(
        nav_state: StrapdownState,
        other_states: Option<DVector<f64>>,
        weight: f64,
    ) -> Particle {
        let state_size = 9 + match &other_states {
            Some(states) => states.len(),
            None => 0,
        };
        Particle {
            nav_state,
            other_states,
            state_size,
            weight,
        }
    }
}
impl From<(DVector<f64>, f64)> for Particle {
    fn from(tuple: (DVector<f64>, f64)) -> Self {
        let (state_vector, weight) = tuple;
        assert!(
            state_vector.len() >= 9,
            "State vector must be at least 9 elements long"
        );
        let nav_state = StrapdownState {
            latitude: state_vector[0],
            longitude: state_vector[1],
            altitude: state_vector[2],
            velocity_north: state_vector[3],
            velocity_east: state_vector[4],
            velocity_down: state_vector[5],
            attitude: Rotation3::from_euler_angles(
                state_vector[6],
                state_vector[7],
                state_vector[8],
            ),
            is_enu: true,
        };
        let other_states = if state_vector.len() > 9 {
            Some(state_vector.rows(9, state_vector.len() - 9).clone_owned())
        } else {
            None
        };
        Particle::new(nav_state, other_states, weight)
    }
}
/// ParticleAveragingStrategy defines the method used to compute the navigation solution
/// from the set of particles in a particle filter. This strategy determines how the
/// filter aggregates individual particle states into a single state estimate and covariance.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum ParticleAveragingStrategy {
    /// Computes the weighted mean and covariance across all particles, using each particle's weight
    /// as its contribution. This is the default and most statistically consistent approach for particle filters.
    #[default]
    WeightedAverage,
    /// Computes the arithmetic mean and covariance across all particles, ignoring particle weights.
    /// This can be useful for diagnostic purposes or when all particles have equal weights.
    UnweightedAverage,
    /// Selects the state of the particle with the highest weight as the navigation solution, with
    /// zero covariance. This approach is sometimes used in multi-modal or highly non-Gaussian scenarios.
    HighestWeight,
}
impl ParticleAveragingStrategy {
    /// Get the weighted averaged state and covariance
    fn weighted_average_state(pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>) {
        let state_size = pf.state_size;
        let mut mean = DVector::<f64>::zeros(state_size);

        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            mean[0] += particle.weight * particle.nav_state.latitude;
            mean[1] += particle.weight * particle.nav_state.longitude;
            mean[2] += particle.weight * particle.nav_state.altitude;
            mean[3] += particle.weight * particle.nav_state.velocity_north;
            mean[4] += particle.weight * particle.nav_state.velocity_east;
            mean[5] += particle.weight * particle.nav_state.velocity_down;
            mean[6] += particle.weight * euler.0; // roll
            mean[7] += particle.weight * euler.1; // pitch
            mean[8] += particle.weight * euler.2; // yaw

            if let Some(ref other) = particle.other_states {
                for (i, val) in other.iter().enumerate() {
                    if 9 + i < state_size {
                        mean[9 + i] += particle.weight * val;
                    }
                }
            }
        }

        // Compute covariance
        let mut cov = DMatrix::<f64>::zeros(state_size, state_size);
        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            let mut state_vec = vec![
                particle.nav_state.latitude,
                particle.nav_state.longitude,
                particle.nav_state.altitude,
                particle.nav_state.velocity_north,
                particle.nav_state.velocity_east,
                particle.nav_state.velocity_down,
                euler.0,
                euler.1,
                euler.2,
            ];
            if let Some(ref other) = particle.other_states {
                state_vec.extend(other.iter());
            }
            // Ensure state vector matches state_size
            if state_vec.len() != state_size {
                // This might happen if particles have inconsistent sizes or pf.state_size is wrong
                // For safety, resize or panic. Here we assume consistency but handle potential mismatch if needed.
                // Ideally, particle construction enforces this.
            }

            let state = DVector::from_vec(state_vec);
            let diff = state - &mean;
            cov += particle.weight * &diff * &diff.transpose();
        }

        (mean, cov)
    }
    // Get the unweighted averaged state and covariance
    fn unweighted_average_state(pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>) {
        let n = pf.particles.len() as f64;
        let state_size = pf.state_size;
        let mut mean = DVector::<f64>::zeros(state_size);

        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            mean[0] += particle.nav_state.latitude / n;
            mean[1] += particle.nav_state.longitude / n;
            mean[2] += particle.nav_state.altitude / n;
            mean[3] += particle.nav_state.velocity_north / n;
            mean[4] += particle.nav_state.velocity_east / n;
            mean[5] += particle.nav_state.velocity_down / n;
            mean[6] += euler.0 / n; // roll
            mean[7] += euler.1 / n; // pitch
            mean[8] += euler.2 / n; // yaw

            if let Some(ref other) = particle.other_states {
                for (i, val) in other.iter().enumerate() {
                    if 9 + i < state_size {
                        mean[9 + i] += val / n;
                    }
                }
            }
        }

        // Compute covariance
        let mut cov = DMatrix::<f64>::zeros(state_size, state_size);
        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            let mut state_vec = vec![
                particle.nav_state.latitude,
                particle.nav_state.longitude,
                particle.nav_state.altitude,
                particle.nav_state.velocity_north,
                particle.nav_state.velocity_east,
                particle.nav_state.velocity_down,
                euler.0,
                euler.1,
                euler.2,
            ];
            if let Some(ref other) = particle.other_states {
                state_vec.extend(other.iter());
            }
            let state = DVector::from_vec(state_vec);
            let diff = state - &mean;
            cov += (1.0 / n) * &diff * &diff.transpose();
        }

        (mean, cov)
    }
    // Get the highest weight particle's state and zero covariance
    fn highest_weight_state(pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>) {
        let best_particle = pf
            .particles
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
            .expect("Particle filter has no particles");

        let euler = best_particle.nav_state.attitude.euler_angles();
        let mut state_vec = vec![
            best_particle.nav_state.latitude,
            best_particle.nav_state.longitude,
            best_particle.nav_state.altitude,
            best_particle.nav_state.velocity_north,
            best_particle.nav_state.velocity_east,
            best_particle.nav_state.velocity_down,
            euler.0,
            euler.1,
            euler.2,
        ];
        if let Some(ref other_states) = best_particle.other_states {
            state_vec.extend(other_states.iter());
        }
        let mean = DVector::from_vec(state_vec);

        // For single particle, covariance is zero
        let cov = DMatrix::<f64>::zeros(pf.state_size, pf.state_size);

        (mean, cov)
    }
}
/// Strategy for resampling particles in the particle filter
#[derive(Clone, Debug, Default)]
pub enum ParticleResamplingStrategy {
    #[default]
    Naive,
    Systematic,
    Multinomial,
    Residual,
    Stratified,
    Adaptive,
}
impl ParticleResamplingStrategy {
    // Resample particles based on the selected strategy
    fn resample(&self, particles: Vec<Particle>) -> Vec<Particle> {
        match self {
            ParticleResamplingStrategy::Naive => {
                // Naive resampling implementation (to be filled in)
                particles
            }
            ParticleResamplingStrategy::Systematic => {
                // Systematic resampling implementation (to be filled in)
                particles
            }
            ParticleResamplingStrategy::Multinomial => {
                // Multinomial resampling implementation (to be filled in)
                particles
            }
            ParticleResamplingStrategy::Residual => Self::residual_resample(particles),
            ParticleResamplingStrategy::Stratified => {
                // Stratified resampling implementation (to be filled in)
                particles
            }
            ParticleResamplingStrategy::Adaptive => {
                // Adaptive resampling implementation (to be filled in)
                particles
            }
        }
    }
    fn residual_resample(particles: Vec<Particle>) -> Vec<Particle> {
        let n = particles.len();
        let mut new_particles = Vec::<Particle>::with_capacity(n);
        let weights: Vec<f64> = particles.iter().map(|p| p.weight).collect();
        let mut num_copies = vec![0usize; n];
        let mut residual: Vec<f64> = vec![0.0; n];
        //let mut total_residual: f64 = 0.0;
        // Integer part
        for (i, &w) in weights.iter().enumerate() {
            let copies = (w * n as f64).floor() as usize;
            num_copies[i] = copies;
            residual[i] = w * n as f64 - copies as f64;
            //total_residual += residual[i];
        }
        // Copy integer part
        for (i, &copies) in num_copies.iter().enumerate() {
            for _ in 0..copies {
                let mut new_particle = particles[i].clone();
                new_particle.weight = 1.0 / n as f64;
                new_particles.push(new_particle);
            }
        }
        // Residual part
        let residual_particles = n - new_particles.len();
        if residual_particles > 0 {
            // Normalize residuals
            let sum_residual: f64 = residual.iter().sum();
            let mut positions = Vec::with_capacity(residual_particles);
            let step = sum_residual / residual_particles as f64;
            let mut u = rand::random::<f64>() * step;
            for _ in 0..residual_particles {
                positions.push(u);
                u += step;
            }
            let mut i = 0;
            let mut j = 0;
            let mut cumsum = residual[0];
            while j < residual_particles {
                while positions[j] > cumsum {
                    i += 1;
                    cumsum += residual[i];
                }
                let mut new_particle = particles[i].clone();
                new_particle.weight = 1.0 / n as f64;
                new_particles.push(new_particle);
                j += 1;
            }
        }
        new_particles
    }
}
/// Particle filter for strapdown inertial navigation
///
/// This filter uses a particle filter algorithm to estimate the state of a strapdown inertial navigation system.
/// Similarly to the UKF, it uses thin wrappers around the StrapdownState's forward function to propagate the state.
/// The particle filter is a little more generic in implementation than the UKF, as all it fundamentally is is a set
/// of particles and several related functions to propagate, update, and resample the particles.
///
/// Process noise is primarily applied to bias states (accelerometer and gyroscope biases) to model random walk
/// characteristics. Navigation states receive minimal process noise (typically 3-4 orders of magnitude smaller)
/// to maintain particle diversity and prevent filter collapse, while allowing uncertainty to propagate naturally
/// through the nonlinear strapdown dynamics.
#[derive(Clone, Default)]
pub struct ParticleFilter {
    /// The particles in the particle filter
    pub particles: Vec<Particle>,
    /// Process noise standard deviations for all states
    /// Bias states (indices 9-14) should have typical IMU random walk values (~1e-7 for accel, ~1e-9 for gyro).
    /// Navigation states (indices 0-8) should have minimal values (~1e-12 for angles, ~1e-9 for velocities/altitude)
    /// to maintain particle diversity without introducing excessive drift.
    pub process_noise: DVector<f64>,
    /// Strategy for determining navigation solution
    pub averaging_strategy: ParticleAveragingStrategy,
    /// Stategy for resampling
    pub resampling_strategy: ParticleResamplingStrategy,
    /// State vector dimension
    pub state_size: usize,
}
impl Debug for ParticleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // let strategy = ParticleAveragingStrategy::WeightedAverage;
        let mean = self.get_estimate();
        let cov = self.get_certainty();

        // Calculate effective particle count
        let effective_particles = 1.0
            / self
                .particles
                .iter()
                .map(|p| p.weight * p.weight)
                .sum::<f64>();

        // Find weight range
        let min_weight = self
            .particles
            .iter()
            .map(|p| p.weight)
            .fold(f64::INFINITY, f64::min);
        let max_weight = self.particles.iter().map(|p| p.weight).fold(0.0, f64::max);

        f.debug_struct("ParticleFilter")
            .field("num_particles", &self.particles.len())
            .field("effective_particles", &effective_particles)
            .field(
                "weight_range",
                &format_args!("[{:.4e}, {:.4e}]", min_weight, max_weight),
            )
            .field(
                "mean_position",
                &format_args!(
                    "({:.6}°, {:.6}°, {:.2}m)",
                    mean[0].to_degrees(),
                    mean[1].to_degrees(),
                    mean[2]
                ),
            )
            .field(
                "mean_velocity",
                &format_args!("({:.3}, {:.3}, {:.3}) m/s", mean[3], mean[4], mean[5]),
            )
            .field(
                "mean_attitude",
                &format_args!("({:.3}, {:.3}, {:.3}) rad", mean[6], mean[7], mean[8]),
            )
            .field(
                "position_cov_diag",
                &format_args!(
                    "({:.4e}, {:.4e}, {:.4e})",
                    cov[(0, 0)],
                    cov[(1, 1)],
                    cov[(2, 2)]
                ),
            )
            .finish()
    }
}
impl ParticleFilter {
    /// Create a new particle filter with the given particles and default process noise
    ///
    /// # Arguments
    /// * `particles` - The particles to use for the particle filter.
    pub fn new(
        particles: Vec<Particle>,
        process_noise_std: Option<DVector<f64>>,
        estimation_strategy: Option<ParticleAveragingStrategy>,
        resampling_method: Option<ParticleResamplingStrategy>,
    ) -> Self {
        // Default process noise standard deviations
        // Bias process noise models IMU random walk; navigation noise maintains particle diversity
        let state_size = particles[0].state_size;
        let process_noise: DVector<f64> = match process_noise_std {
            Some(pn) => pn,
            None => {
                let mut noise = vec![
                    1e-12, // lat: minimal noise to prevent particle collapse
                    1e-12, // lon: minimal noise
                    1e-9,  // alt: small noise for vertical channel
                    1e-9,  // vn: minimal noise
                    1e-9,  // ve: minimal noise
                    1e-9,  // vd: minimal noise
                    1e-12, // roll: minimal noise
                    1e-12, // pitch: minimal noise
                    1e-12, // yaw: minimal noise
                ];
                // Set bias process noise if biases are estimated (state_size > 9)
                if state_size > 9 {
                    // Accelerometer biases: typical random walk ~1e-7 m/s²
                    noise.push(1e-7); // X axis
                    noise.push(1e-7); // Y axis
                    noise.push(1e-7); // Z axis
                    if state_size > 12 {
                        // Gyroscope biases: typical random walk ~1e-9 rad/s
                        noise.push(1e-9); // X axis
                        noise.push(1e-9); // Y axis
                        noise.push(1e-9); // Z axis
                    }
                }
                // Pad with zeros for any additional states beyond biases
                while noise.len() < state_size {
                    noise.push(0.0);
                }
                DVector::from_vec(noise)
            }
        };
        assert_eq!(
            process_noise.len(),
            particles[0].state_size,
            "Process noise must have {} elements",
            particles[0].state_size
        );
        let state_size = particles[0].state_size;
        ParticleFilter {
            particles,
            process_noise,
            averaging_strategy: estimation_strategy
                .unwrap_or(ParticleAveragingStrategy::WeightedAverage),
            resampling_strategy: resampling_method.unwrap_or(ParticleResamplingStrategy::Residual),
            state_size,
        }
    }
    /// Initialization function for creating the particle field according to a normal distribution
    ///
    /// # Arguments
    /// * `mean` - an `InitialState` struct defining the mean state for the particle filter navigation states
    /// * `other_states` - a DVector<f64> defining the mean values for any additional other states to be estimated
    /// * `navigation_covariance` - a DVector<f64> defining the diagonal covariance for initial particle sampling
    /// * `other_covariance` - a DVector<f64> defining the diagonal covariance for bias state initial sampling
    /// * `num_particles` - the number of particles to initialize
    /// * `process_noise` - Optional process noise for all states (15 elements). Bias states (9-14) should have
    ///   typical IMU random walk values. Navigation states (0-8) should be minimal to maintain diversity without
    ///   excessive drift. If None, defaults to appropriate values.
    /// * `averaging_strategy` - Optional strategy for computing navigation solution from particles
    /// * `resampling_method` - Optional strategy for resampling particles
    /// # Returns
    /// * A new ParticleFilter instance
    ///
    /// # Example
    ///
    /// ```rust
    /// use nalgebra::DVector;
    /// use strapdown::filter::{ParticleFilter, InitialState};
    /// let mean = InitialState {
    ///    latitude: 37.7749,
    ///    longitude: -122.4194,
    ///    altitude: 10.0,
    ///    northward_velocity: 0.0,
    ///    eastward_velocity: 0.0,
    ///    downward_velocity: 0.0,
    ///    roll: 0.0,
    ///    pitch: 0.0,
    ///    yaw: 0.0,
    ///    in_degrees: true,
    ///    is_enu: true,
    /// };
    /// let other_states = DVector::from_vec(vec![0.0; 5]);
    /// let navigation_covariance = DVector::from_vec(vec![1.0; 9]);
    /// let other_covariance = DVector::from_vec(vec![1.0; 5]);
    /// let num_particles = 100;
    /// let process_noise = DVector::from_vec(vec![0.1; 14]);
    /// let pf = ParticleFilter::new_about(mean, other_states, navigation_covariance, other_covariance, num_particles, Some(process_noise), None, None);
    /// ```
    pub fn new_about(
        mean: InitialState,
        other_states: DVector<f64>,
        navigation_covariance: DVector<f64>,
        other_covariance: DVector<f64>,
        num_particles: usize,
        process_noise_std: Option<DVector<f64>>,
        estimation_strategy: Option<ParticleAveragingStrategy>,
        resampling_method: Option<ParticleResamplingStrategy>,
    ) -> ParticleFilter {
        use rand_distr::{Distribution, Normal};
        let mut particles = Vec::with_capacity(num_particles);
        let position_std = (
            navigation_covariance[0].sqrt(),
            navigation_covariance[1].sqrt(),
            navigation_covariance[2].sqrt(),
        );
        let velocity_std = (
            navigation_covariance[3].sqrt(),
            navigation_covariance[4].sqrt(),
            navigation_covariance[5].sqrt(),
        );
        let attitude_std = (
            navigation_covariance[6].sqrt(),
            navigation_covariance[7].sqrt(),
            navigation_covariance[8].sqrt(),
        );
        let other_std: Vec<f64> = other_covariance.iter().map(|x| x.sqrt()).collect();
        let lat_normal = Normal::new(0.0, position_std.0).unwrap();
        let lon_normal = Normal::new(0.0, position_std.1).unwrap();
        let alt_normal = Normal::new(0.0, position_std.2).unwrap();
        let vn_normal = Normal::new(0.0, velocity_std.0).unwrap();
        let ve_normal = Normal::new(0.0, velocity_std.1).unwrap();
        let vd_normal = Normal::new(0.0, velocity_std.2).unwrap();
        let roll_normal = Normal::new(0.0, attitude_std.0).unwrap();
        let pitch_normal = Normal::new(0.0, attitude_std.1).unwrap();
        let yaw_normal = Normal::new(0.0, attitude_std.2).unwrap();
        let mut rng = rand::rng();
        for _ in 0..num_particles {
            let latitude = if mean.in_degrees {
                mean.latitude.to_radians() + lat_normal.sample(&mut rng)
            } else {
                mean.latitude + lat_normal.sample(&mut rng)
            };
            let longitude = if mean.in_degrees {
                mean.longitude.to_radians() + lon_normal.sample(&mut rng)
            } else {
                mean.longitude + lon_normal.sample(&mut rng)
            };
            let altitude = mean.altitude + alt_normal.sample(&mut rng);
            let velocity_north = mean.northward_velocity + vn_normal.sample(&mut rng);
            let velocity_east = mean.eastward_velocity + ve_normal.sample(&mut rng);
            let velocity_down = mean.downward_velocity + vd_normal.sample(&mut rng);
            let roll = mean.roll + roll_normal.sample(&mut rng);
            let pitch = mean.pitch + pitch_normal.sample(&mut rng);
            let yaw = mean.yaw + yaw_normal.sample(&mut rng);
            let attitude = Rotation3::from_euler_angles(roll, pitch, yaw);
            let mut other_state_vec = Vec::with_capacity(other_states.len());
            for (i, &state) in other_states.iter().enumerate() {
                let other_normal = Normal::new(0.0, other_std[i]).unwrap();
                other_state_vec.push(state + other_normal.sample(&mut rng));
            }
            let particle = Particle::new(
                StrapdownState {
                    latitude,
                    longitude,
                    altitude,
                    velocity_north,
                    velocity_east,
                    velocity_down,
                    attitude,
                    is_enu: true,
                },
                Some(DVector::from_vec(other_state_vec)),
                1.0 / num_particles as f64,
            );
            particles.push(particle);
        }
        ParticleFilter::new(
            particles,
            process_noise_std,
            estimation_strategy,
            resampling_method,
        )
    }
    /// Helper function that converts the particles' state vector to a matrix
    pub fn particles_to_matrix(&self) -> DMatrix<f64> {
        let n_particles = self.particles.len();
        let state_size = self.state_size;
        let mut data = Vec::with_capacity(n_particles * state_size);
        for particle in &self.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            data.push(particle.nav_state.latitude);
            data.push(particle.nav_state.longitude);
            data.push(particle.nav_state.altitude);
            data.push(particle.nav_state.velocity_north);
            data.push(particle.nav_state.velocity_east);
            data.push(particle.nav_state.velocity_down);
            data.push(euler.0); // roll
            data.push(euler.1); // pitch
            data.push(euler.2); // yaw
            if let Some(ref other_states) = particle.other_states {
                for val in other_states.iter() {
                    data.push(*val);
                }
            }
        }
        DMatrix::from_vec(state_size, n_particles, data)
    }
    /// Set the weights of the particles (e.g., after a measurement update)
    ///
    /// # Arguments
    /// * `weights` - The weights to set for the particles.
    pub fn set_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.particles.len());
        for (particle, &w) in self.particles.iter_mut().zip(weights.iter()) {
            particle.weight = w;
        }
    }
    /// Normalize the weights of the particles. This is typically done after a measurement update
    /// to ensure that the weights sum to 1.0 and can be treated like a probability distribution.
    pub fn normalize_weights(&mut self) {
        let sum: f64 = self.particles.iter().map(|p| p.weight).sum();
        if sum > 0.0 && sum.is_finite() {
            for particle in &mut self.particles {
                particle.weight /= sum;
            }
        } else {
            // If weights sum to zero or are invalid, reset to uniform
            log::warn!(
                "Particle weights sum to {} (invalid), resetting to uniform distribution",
                sum
            );
            let uniform_weight = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.weight = uniform_weight;
            }
        }
    }
}
impl NavigationFilter for ParticleFilter {
    /// Propagate all particles forward using the strapdown equations with process noise
    ///
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the particles with.
    /// * `dt` - Time step in seconds
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        let mut rng = rand::rng();

        for particle in &mut self.particles {
            let accel_biases = if particle.other_states.is_some()
                && particle.other_states.as_ref().unwrap().len() >= 6
            {
                DVector::from_vec(vec![
                    particle.other_states.as_ref().unwrap()[0],
                    particle.other_states.as_ref().unwrap()[1],
                    particle.other_states.as_ref().unwrap()[2],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let gyro_biases = if particle.other_states.is_some()
                && particle.other_states.as_ref().unwrap().len() >= 6
            {
                DVector::from_vec(vec![
                    particle.other_states.as_ref().unwrap()[3],
                    particle.other_states.as_ref().unwrap()[4],
                    particle.other_states.as_ref().unwrap()[5],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let imu_data = IMUData {
                accel: imu_data.accel - accel_biases,
                gyro: imu_data.gyro - gyro_biases,
            };
            // Deterministic propagation
            forward(&mut particle.nav_state, imu_data, dt);
            // Add process noise to maintain particle diversity and model bias random walk
            // Navigation noise is minimal (<<< bias noise) to prevent collapse
            // while allowing uncertainty propagation through strapdown dynamics
            // Scale noise by sqrt(dt) for proper time scaling
            let dt_sqrt = dt.sqrt();

            // Add minimal noise to navigation states to prevent particle collapse
            particle.nav_state.latitude += rand_distr::Normal::new(0.0, self.process_noise[0])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.longitude += rand_distr::Normal::new(0.0, self.process_noise[1])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.altitude += rand_distr::Normal::new(0.0, self.process_noise[2])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.velocity_north +=
                rand_distr::Normal::new(0.0, self.process_noise[3])
                    .unwrap()
                    .sample(&mut rng)
                    * dt_sqrt;
            particle.nav_state.velocity_east += rand_distr::Normal::new(0.0, self.process_noise[4])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.velocity_down += rand_distr::Normal::new(0.0, self.process_noise[5])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let roll_noise = rand_distr::Normal::new(0.0, self.process_noise[6])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let pitch_noise = rand_distr::Normal::new(0.0, self.process_noise[7])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let yaw_noise = rand_distr::Normal::new(0.0, self.process_noise[8])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let (roll, pitch, yaw) = particle.nav_state.attitude.euler_angles();
            particle.nav_state.attitude = Rotation3::from_euler_angles(
                roll + roll_noise,
                pitch + pitch_noise,
                yaw + yaw_noise,
            );

            // Add process noise to bias states to model IMU random walk
            if let Some(ref mut other_states) = particle.other_states {
                // Bias states: accelerometer (indices 0-2) and gyroscope (indices 3-5)
                for i in 0..other_states.len().min(6) {
                    let noise = rand_distr::Normal::new(0.0, self.process_noise[9 + i])
                        .unwrap()
                        .sample(&mut rng)
                        * dt_sqrt;
                    other_states[i] += noise;
                }
            }
        }
    }
    /// Update the weights of the particles based on a measurement
    ///
    /// This method updates the particle weights based on the likelihood of the measurement
    /// given each particle's state. The measurement model is used to compute the expected
    /// measurement for each particle, and the particle weights are updated based on the
    /// measurement innovation using a Gaussian likelihood function. After updating, the
    /// weights are normalized to sum to 1.0.
    ///
    /// The likelihood is computed as p(z|x) ∝ exp(-0.5 * (z - h(x))^T * R^-1 * (z - h(x))),
    /// where z is the measurement, h(x) is the expected measurement from the particle state,
    /// and R is the measurement noise covariance.
    ///
    /// # Arguments
    /// * `measurement` - The measurement to update the particle weights with.
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        let particle_matrix = self.particles_to_matrix();
        let measurement_sigma_points = measurement.get_sigma_points(&particle_matrix);
        // Compute expected measurement for each particle
        let mut z_hats = DMatrix::<f64>::zeros(measurement.get_dimension(), self.particles.len());
        for (i, measurement_sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            z_hats.set_column(i, &measurement_sigma_point);
        }

        // Compute log-likelihoods to avoid numerical underflow
        let mut log_likelihoods = Vec::with_capacity(self.particles.len());

        // Update weights based on measurement likelihood
        for (i, particle) in self.particles.iter_mut().enumerate() {
            let z_hat = z_hats.column(i);
            let innovation = measurement.get_vector() - z_hat;
            let sigmas = measurement.get_noise();

            // Compute log-likelihood to avoid numerical underflow
            let sigma_inv = match sigmas.clone().try_inverse() {
                Some(inv) => inv,
                None => {
                    log::warn!(
                        "Measurement covariance matrix is singular, using very small likelihood"
                    );
                    particle.weight = 1e-300;
                    log_likelihoods.push(-690.0); // log(1e-300)
                    continue;
                }
            };

            let sigma_det = sigmas.determinant();
            if sigma_det <= 0.0 {
                log::warn!(
                    "Measurement covariance determinant is non-positive: {}",
                    sigma_det
                );
                particle.weight = 1e-300;
                log_likelihoods.push(-690.0);
                continue;
            }

            let mahalanobis = innovation.transpose() * sigma_inv * innovation;
            let log_likelihood = -0.5
                * (measurement.get_dimension() as f64 * (2.0 * std::f64::consts::PI).ln()
                    + sigma_det.ln()
                    + mahalanobis[(0, 0)]);

            log_likelihoods.push(log_likelihood);
        }

        // Convert log-likelihoods to weights using the log-sum-exp trick
        let max_log_likelihood = log_likelihoods
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        for (i, particle) in self.particles.iter_mut().enumerate() {
            // Subtract max to avoid overflow, then exponentiate
            particle.weight = (log_likelihoods[i] - max_log_likelihood).exp();
        }

        // Normalize weights to sum to 1.0
        self.normalize_weights();
        // Note: Resampling is done separately via the resample() method
        // This allows getting the estimate with updated weights before resampling
    }

    fn get_estimate(&self) -> DVector<f64> {
        match self.averaging_strategy {
            ParticleAveragingStrategy::WeightedAverage => {
                let (mean, _cov) = ParticleAveragingStrategy::weighted_average_state(self);
                mean
            }
            ParticleAveragingStrategy::UnweightedAverage => {
                let (mean, _cov) = ParticleAveragingStrategy::unweighted_average_state(self);
                mean
            }
            ParticleAveragingStrategy::HighestWeight => {
                let (mean, _cov) = ParticleAveragingStrategy::highest_weight_state(self);
                mean
            }
        }
    }
    fn get_certainty(&self) -> DMatrix<f64> {
        match self.averaging_strategy {
            ParticleAveragingStrategy::WeightedAverage => {
                let (_mean, cov) = ParticleAveragingStrategy::weighted_average_state(self);
                cov
            }
            ParticleAveragingStrategy::UnweightedAverage => {
                let (_mean, cov) = ParticleAveragingStrategy::unweighted_average_state(self);
                cov
            }
            ParticleAveragingStrategy::HighestWeight => {
                let (_mean, cov) = ParticleAveragingStrategy::highest_weight_state(self);
                cov
            }
        }
    }
}

impl ParticleFilter {
    /// Resample particles based on their weights
    ///
    /// This method resamples the particles according to the configured resampling strategy.
    /// After resampling, all particles have equal weights (1/N). Resampling should typically
    /// be done after update() and get_estimate() to prevent weight degeneracy.
    ///
    /// The effective sample size (N_eff = 1 / sum(w_i^2)) can be used to determine when
    /// resampling is needed. A common threshold is N_eff < N/2.
    pub fn resample(&mut self) {
        self.particles = self.resampling_strategy.resample(self.particles.clone());
    }

    /// Calculate the effective sample size (ESS)
    ///
    /// ESS = 1 / sum(w_i^2) where w_i are the normalized weights.
    /// A low ESS indicates weight degeneracy and that resampling may be beneficial.
    /// Common threshold is to resample when ESS < N/2.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_of_squares: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_of_squares > 0.0 {
            1.0 / sum_of_squares
        } else {
            0.0
        }
    }
}

/// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth;
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Vector3;

    const IMU_BIASES: [f64; 6] = [0.0; 6];
    const N: usize = 15;
    const COVARIANCE_DIAGONAL: [f64; N] = [1e-9; N];
    const PROCESS_NOISE_DIAGONAL: [f64; N] = [1e-9; N];

    const ALPHA: f64 = 1e-3;
    const BETA: f64 = 2.0;
    const KAPPA: f64 = 0.0;
    const UKF_PARAMS: InitialState = InitialState {
        latitude: 0.0,
        longitude: 0.0,
        altitude: 0.0,
        northward_velocity: 0.0,
        eastward_velocity: 0.0,
        downward_velocity: 0.0,
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
        in_degrees: false,
        is_enu: true,
    };

    #[test]
    fn ukf_construction() {
        let measurement_bias = vec![0.0; 3]; // Example measurement bias
        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            Some(measurement_bias.clone()),
            vec![1e-3; 18],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-3; 18])),
            ALPHA,
            BETA,
            KAPPA,
        );
        assert_eq!(ukf.mean_state.len(), 18);
        let wms = ukf.weights_mean;
        let wcs = ukf.weights_cov;
        assert_eq!(wms.len(), (2 * ukf.state_size) + 1);
        assert_eq!(wcs.len(), (2 * ukf.state_size) + 1);
        // Check that the weights are correct
        let lambda = ALPHA.powi(2) * (18.0 + KAPPA) - 18.0;
        assert_eq!(lambda, ukf.lambda);
        let wm_0 = lambda / (18.0 + lambda);
        let wc_0 = wm_0 + (1.0 - ALPHA.powi(2)) + BETA;
        let w_i = 1.0 / (2.0 * (18.0 + lambda));
        assert_approx_eq!(wms[0], wm_0, 1e-6);
        assert_approx_eq!(wcs[0], wc_0, 1e-6);
        for i in 1..wms.len() {
            assert_approx_eq!(wms[i], w_i, 1e-6);
            assert_approx_eq!(wcs[i], w_i, 1e-6);
        }
    }
    #[test]
    fn ukf_get_sigma_points() {
        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );
        let sigma_points = ukf.get_sigma_points();
        assert_eq!(sigma_points.ncols(), (2 * ukf.state_size) + 1);

        let mu = ukf.get_sigma_points() * ukf.weights_mean;
        assert_eq!(mu.nrows(), ukf.state_size);
        assert_eq!(mu.ncols(), 1);
        assert_approx_eq!(mu[0], 0.0, 1e-6);
        assert_approx_eq!(mu[1], 0.0, 1e-6);
        assert_approx_eq!(mu[2], 0.0, 1e-6);
        assert_approx_eq!(mu[3], 0.0, 1e-6);
        assert_approx_eq!(mu[4], 0.0, 1e-6);
        assert_approx_eq!(mu[5], 0.0, 1e-6);
        assert_approx_eq!(mu[6], 0.0, 1e-6);
        assert_approx_eq!(mu[7], 0.0, 1e-6);
        assert_approx_eq!(mu[8], 0.0, 1e-6);
    }
    #[test]
    fn ukf_propagate() {
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            vec![0.0; 6],
            None,         //Some(measurement_bias.clone()),
            vec![0.0; N], // Absolute certainty use for testing the process
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            1e-3,
            2.0,
            0.0,
        );
        let dt = 1.0;
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0), // No rotation
        };
        ukf.predict(imu_data, dt);
        assert!(
            ukf.mean_state.len() == 15 //+ measurement_bias.len()
        );
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1e-3,
            vertical_noise_std: 1e-3,
        };
        ukf.update(&measurement);
        // Check that the state has not changed
        assert_approx_eq!(ukf.mean_state[0], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[1], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[2], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.1);
    }
    #[test]
    fn test_gps_position_measurement_display() {
        let measurement = GPSPositionMeasurement {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
        };
        let display_str = format!("{}", measurement);
        assert!(display_str.contains("45"));
        assert!(display_str.contains("-75"));
        assert!(display_str.contains("100"));
    }
    #[test]
    fn test_gps_position_measurement_model() {
        let measurement = GPSPositionMeasurement {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
        };

        // Test dimension
        assert_eq!(measurement.get_dimension(), 3);

        // Test vector conversion
        let vec = measurement.get_vector();
        assert_eq!(vec.len(), 3);
        assert_approx_eq!(vec[0], 45.0_f64.to_radians(), 1e-6);
        assert_approx_eq!(vec[1], (-75.0_f64).to_radians(), 1e-6);
        assert_approx_eq!(vec[2], 100.0, 1e-6);

        // Test noise matrix
        let noise = measurement.get_noise();
        assert_eq!(noise.nrows(), 3);
        assert_eq!(noise.ncols(), 3);
        assert!(noise[(0, 0)] > 0.0);
        assert!(noise[(1, 1)] > 0.0);
        assert!(noise[(2, 2)] > 0.0);

        // Test sigma points mapping
        let state_sigma = DMatrix::from_vec(
            9,
            3,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
                8.1, 9.1, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2,
            ],
        );
        let meas_sigma = measurement.get_sigma_points(&state_sigma);
        assert_eq!(meas_sigma.nrows(), 3);
        assert_eq!(meas_sigma.ncols(), 3);
        assert_approx_eq!(meas_sigma[(0, 0)], 1.0, 1e-6);
        assert_approx_eq!(meas_sigma[(1, 0)], 2.0, 1e-6);
        assert_approx_eq!(meas_sigma[(2, 0)], 3.0, 1e-6);
    }
    #[test]
    fn test_gps_velocity_measurement_display() {
        let measurement = GPSVelocityMeasurement {
            northward_velocity: 10.0,
            eastward_velocity: 5.0,
            downward_velocity: -2.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 1.0,
        };
        let display_str = format!("{}", measurement);
        assert!(display_str.contains("10"));
        assert!(display_str.contains("5"));
    }
    #[test]
    fn test_gps_velocity_measurement_model() {
        let measurement = GPSVelocityMeasurement {
            northward_velocity: 10.0,
            eastward_velocity: 5.0,
            downward_velocity: -2.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 1.0,
        };

        // Test dimension
        assert_eq!(measurement.get_dimension(), 3);

        // Test vector
        let vec = measurement.get_vector();
        assert_eq!(vec.len(), 3);
        assert_approx_eq!(vec[0], 10.0, 1e-6);
        assert_approx_eq!(vec[1], 5.0, 1e-6);
        assert_approx_eq!(vec[2], -2.0, 1e-6);

        // Test noise matrix
        let noise = measurement.get_noise();
        assert_eq!(noise.nrows(), 3);
        assert_eq!(noise.ncols(), 3);

        // Test sigma points mapping (maps to velocity components)
        let state_sigma = DMatrix::from_vec(
            9,
            2,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
                8.1, 9.1,
            ],
        );
        let meas_sigma = measurement.get_sigma_points(&state_sigma);
        assert_eq!(meas_sigma.nrows(), 3);
        assert_eq!(meas_sigma.ncols(), 2);
        assert_approx_eq!(meas_sigma[(0, 0)], 4.0, 1e-6); // state[3]
        assert_approx_eq!(meas_sigma[(1, 0)], 5.0, 1e-6); // state[4]
        assert_approx_eq!(meas_sigma[(2, 0)], 6.0, 1e-6); // state[5]
    }
    #[test]
    fn test_gps_position_velocity_measurement_model() {
        let measurement = GPSPositionAndVelocityMeasurement {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 5.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
            velocity_noise_std: 0.5,
        };

        // Test dimension
        assert_eq!(measurement.get_dimension(), 5);

        // Test vector
        let vec = measurement.get_vector();
        assert_eq!(vec.len(), 5);
        assert_approx_eq!(vec[0], 45.0_f64.to_radians(), 1e-6);
        assert_approx_eq!(vec[1], (-75.0_f64).to_radians(), 1e-6);
        assert_approx_eq!(vec[2], 100.0, 1e-6);
        assert_approx_eq!(vec[3], 10.0, 1e-6);
        assert_approx_eq!(vec[4], 5.0, 1e-6);

        // Test noise matrix
        let noise = measurement.get_noise();
        assert_eq!(noise.nrows(), 5);
        assert_eq!(noise.ncols(), 5);

        // Test sigma points mapping
        let state_sigma = DMatrix::from_vec(
            9,
            2,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
                8.1, 9.1,
            ],
        );
        let meas_sigma = measurement.get_sigma_points(&state_sigma);
        assert_eq!(meas_sigma.nrows(), 5);
        assert_eq!(meas_sigma.ncols(), 2);
        assert_approx_eq!(meas_sigma[(0, 0)], 1.0, 1e-6); // lat
        assert_approx_eq!(meas_sigma[(1, 0)], 2.0, 1e-6); // lon
        assert_approx_eq!(meas_sigma[(2, 0)], 3.0, 1e-6); // alt
        assert_approx_eq!(meas_sigma[(3, 0)], 4.0, 1e-6); // v_n
        assert_approx_eq!(meas_sigma[(4, 0)], 5.0, 1e-6); // v_e
    }
    #[test]
    fn test_measurement_as_any() {
        let measurement = GPSPositionMeasurement {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
        };

        // Test as_any downcast
        let any_ref = measurement.as_any();
        assert!(any_ref.downcast_ref::<GPSPositionMeasurement>().is_some());

        // Test as_any_mut downcast
        let mut measurement_mut = measurement.clone();
        let any_mut = measurement_mut.as_any_mut();
        assert!(any_mut.downcast_mut::<GPSPositionMeasurement>().is_some());
    }
    #[test]
    fn test_particle_filter_new() {
        let nav_state = StrapdownState::default();
        let particle = Particle::new(nav_state, None, 1.0);
        let particles = vec![particle];
        let pf = ParticleFilter::new(particles, None, None, None);

        assert_eq!(pf.particles.len(), 1);
        assert_eq!(pf.state_size, 9);
        // Check default process noise
        assert_eq!(pf.process_noise.len(), 9);
        assert!(matches!(
            pf.averaging_strategy,
            ParticleAveragingStrategy::WeightedAverage
        ));
        assert!(matches!(
            pf.resampling_strategy,
            ParticleResamplingStrategy::Residual
        ));
    }
    #[test]
    fn test_particle_filter_new_about() {
        let mean = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };
        let other_states = DVector::from_vec(vec![0.0; 2]);
        let nav_cov = DVector::from_vec(vec![0.01; 9]); // Small variance
        let other_cov = DVector::from_vec(vec![0.01; 2]);
        let num_particles = 100;

        let pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            num_particles,
            Some(DVector::from_vec(vec![0.0; 11])),
            None,
            None,
        );

        assert_eq!(pf.particles.len(), num_particles);
        assert_eq!(pf.state_size, 11); // 9 nav + 2 other

        // Check that mean of particles is close to initial mean
        let estimate = pf.get_estimate();
        assert_approx_eq!(estimate[2], 100.0, 1.0); // Altitude
        assert_approx_eq!(estimate[3], 10.0, 1.0); // Vn
    }
    #[test]
    fn test_particle_filter_methods() {
        let nav_state = StrapdownState::default();
        let p1 = Particle::new(nav_state.clone(), None, 0.2);
        let p2 = Particle::new(nav_state, None, 0.8);
        let mut pf = ParticleFilter::new(vec![p1, p2], None, None, None);

        // Test set_weights
        pf.set_weights(&[0.5, 0.5]);
        assert_eq!(pf.particles[0].weight, 0.5);
        assert_eq!(pf.particles[1].weight, 0.5);

        // Test normalize_weights
        pf.set_weights(&[2.0, 2.0]);
        pf.normalize_weights();
        assert_approx_eq!(pf.particles[0].weight, 0.5, 1e-6);
        assert_approx_eq!(pf.particles[1].weight, 0.5, 1e-6);

        // Test particles_to_matrix
        let matrix = pf.particles_to_matrix();
        assert_eq!(matrix.nrows(), 9);
        assert_eq!(matrix.ncols(), 2);
    }
    #[test]
    fn test_particle_averaging_strategies() {
        let mut s1 = StrapdownState::default();
        s1.altitude = 10.0;
        let mut s2 = StrapdownState::default();
        s2.altitude = 20.0;

        let p1 = Particle::new(s1, None, 0.25);
        let p2 = Particle::new(s2, None, 0.75);

        let mut pf = ParticleFilter::new(vec![p1, p2], None, None, None);

        // Weighted Average: 0.25 * 10 + 0.75 * 20 = 2.5 + 15 = 17.5
        pf.averaging_strategy = ParticleAveragingStrategy::WeightedAverage;
        let est_weighted = pf.get_estimate();
        assert_approx_eq!(est_weighted[2], 17.5, 1e-6);

        // Unweighted Average: (10 + 20) / 2 = 15
        pf.averaging_strategy = ParticleAveragingStrategy::UnweightedAverage;
        let est_unweighted = pf.get_estimate();
        assert_approx_eq!(est_unweighted[2], 15.0, 1e-6);

        // Highest Weight: 20.0 (since p2 has 0.75)
        pf.averaging_strategy = ParticleAveragingStrategy::HighestWeight;
        let est_highest = pf.get_estimate();
        assert_approx_eq!(est_highest[2], 20.0, 1e-6);
    }
    #[test]
    fn test_resampling_residual() {
        let s = StrapdownState::default();
        // 3 particles with weights 0.1, 0.1, 0.8
        // Should result in particles roughly proportional to weights
        let p1 = Particle::new(s.clone(), None, 0.1);
        let p2 = Particle::new(s.clone(), None, 0.1);
        let p3 = Particle::new(s.clone(), None, 0.8);

        let strategy = ParticleResamplingStrategy::Residual;
        let resampled = strategy.resample(vec![p1, p2, p3]);

        assert_eq!(resampled.len(), 3);
        // All weights should be reset to 1/N
        for p in resampled {
            assert_approx_eq!(p.weight, 1.0 / 3.0, 1e-6);
        }
    }
    #[test]
    fn test_particle_filter_predict_update() {
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-9; 9]);
        // Include 6 bias states: 3 accel biases + 3 gyro biases
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);
        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            100,
            Some(DVector::from_vec(vec![
                1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-7, 1e-7, 1e-7, 1e-9,
                1e-9, 1e-9,
            ])),
            None,
            Some(ParticleResamplingStrategy::Residual),
        );

        // Predict
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        pf.predict(imu_data, 0.1);

        // Update
        let meas = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1.0,
            vertical_noise_std: 1.0,
        };
        pf.update(&meas);

        // Check that we still have particles and they are normalized
        assert_eq!(pf.particles.len(), 100);
        let sum_weights: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(sum_weights, 1.0, 1e-6);
    }
    #[test]
    fn test_particle_filter_altitude_measurement_correction() {
        // Test that altitude measurements properly correct altitude drift
        let mean = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let nav_cov = DVector::from_vec(vec![1e-6, 1e-6, 10.0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]);
        // Include 6 bias states: 3 accel biases + 3 gyro biases
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);
        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            200,
            Some(DVector::from_vec(vec![
                1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-7, 1e-7, 1e-7, 1e-9,
                1e-9, 1e-9,
            ])),
            None,
            Some(ParticleResamplingStrategy::Residual),
        );

        // Run several prediction steps to allow some drift
        // Use slightly incorrect gravity compensation to induce altitude drift
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 9.5), // Slightly off from 9.81 to cause drift
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        for _ in 0..10 {
            pf.predict(imu_data, 0.1);
        }

        // Apply GPS position measurement with correct altitude
        let meas = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };

        // Check particle altitudes and weights before update
        let mut pre_altitudes: Vec<_> = pf
            .particles
            .iter()
            .map(|p| (p.nav_state.altitude, p.weight))
            .collect();
        pre_altitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!(
            "Before update - first 5 particles (alt, weight): {:?}",
            &pre_altitudes[..5.min(pre_altitudes.len())]
        );
        println!(
            "Before update - last 5 particles (alt, weight): {:?}",
            &pre_altitudes[pre_altitudes.len().saturating_sub(5)..]
        );

        pf.update(&meas);

        // Check particle altitudes and weights after update
        let mut post_altitudes: Vec<_> = pf
            .particles
            .iter()
            .map(|p| (p.nav_state.altitude, p.weight))
            .collect();
        post_altitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by weight descending
        println!("After update - top 10 by weight (alt, weight):");
        for (i, (alt, wt)) in post_altitudes.iter().take(10).enumerate() {
            println!("  {}: alt={:.2}m, weight={:.6e}", i + 1, alt, wt);
        }

        let post_update_estimate = pf.get_estimate();
        let post_update_altitude = post_update_estimate[2];

        // Now resample for the next iteration
        pf.resample();

        // The update should pull the estimate toward the measurement,
        // though it may not always reduce error in a single step due to
        // the stochastic nature of particle filters
        // Just verify that the altitude remains within reasonable bounds
        assert!(
            post_update_altitude > 90.0 && post_update_altitude < 110.0,
            "Altitude {} is outside reasonable bounds after correction",
            post_update_altitude
        );

        // Verify the estimate is closer to 100m than to the extremes
        let dist_to_true = (post_update_altitude - 100.0).abs();
        assert!(
            dist_to_true < 5.0,
            "Altitude {} is too far from true value 100.0 (error: {:.2}m)",
            post_update_altitude,
            dist_to_true
        );
    }

    #[test]
    fn test_particle_filter_weight_degeneracy() {
        // Test that weights degenerate without resampling but are maintained with resampling
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-6; 9]);

        // First test: without resampling, effective sample size should decrease
        // Include 6 bias states: 3 accel biases + 3 gyro biases
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);
        let mut pf_no_resample = ParticleFilter::new_about(
            mean.clone(),
            other_states.clone(),
            nav_cov.clone(),
            other_cov.clone(),
            100,
            Some(DVector::from_vec(vec![
                1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-7, 1e-7, 1e-7, 1e-9,
                1e-9, 1e-9,
            ])),
            None,
            Some(ParticleResamplingStrategy::Residual),
        );

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        let initial_ess = pf_no_resample.effective_sample_size();

        for _ in 0..5 {
            pf_no_resample.predict(imu_data, 0.1);

            let meas = GPSPositionMeasurement {
                latitude: 0.0,
                longitude: 0.0,
                altitude: 0.0,
                horizontal_noise_std: 5.0,
                vertical_noise_std: 2.0,
            };

            pf_no_resample.update(&meas);
            // Note: no resample() call
        }

        let final_ess_no_resample = pf_no_resample.effective_sample_size();

        // Without resampling, ESS should decrease significantly
        assert!(
            final_ess_no_resample < initial_ess * 0.5,
            "ESS should decrease without resampling: initial={:.1}, final={:.1}",
            initial_ess,
            final_ess_no_resample
        );

        // Second test: with resampling, effective sample size should stay high
        let mut pf_with_resample = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            100,
            Some(DVector::from_vec(vec![
                1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-7, 1e-7, 1e-7, 1e-9,
                1e-9, 1e-9,
            ])),
            None,
            Some(ParticleResamplingStrategy::Residual),
        );

        for _ in 0..5 {
            pf_with_resample.predict(imu_data, 0.1);

            let meas = GPSPositionMeasurement {
                latitude: 0.0,
                longitude: 0.0,
                altitude: 0.0,
                horizontal_noise_std: 5.0,
                vertical_noise_std: 2.0,
            };

            pf_with_resample.update(&meas);
            pf_with_resample.resample(); // Resample after each update
        }

        let final_ess_with_resample = pf_with_resample.effective_sample_size();

        // With resampling, ESS should remain high (close to N)
        assert!(
            final_ess_with_resample > 50.0,
            "ESS should remain high with resampling: {:.1}",
            final_ess_with_resample
        );

        // Weights should be normalized in both cases
        let sum_no_resample: f64 = pf_no_resample.particles.iter().map(|p| p.weight).sum();
        let sum_with_resample: f64 = pf_with_resample.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(sum_no_resample, 1.0, 1e-6);
        assert_approx_eq!(sum_with_resample, 1.0, 1e-6);
    }

    #[test]
    fn test_particle_filter_measurement_rejection() {
        // Test that outlier measurements don't completely destroy the filter
        let mean = InitialState {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: true,
            is_enu: true,
        };

        let nav_cov = DVector::from_vec(vec![1e-6, 1e-6, 1.0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]);
        // Include 6 bias states: 3 accel biases + 3 gyro biases
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);
        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            200,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            Some(ParticleResamplingStrategy::Residual),
        );

        let initial_estimate = pf.get_estimate();

        // Apply a severely corrupted measurement (1000m altitude error)
        let bad_meas = GPSPositionMeasurement {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 1100.0, // 1000m error
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };

        pf.update(&bad_meas);

        let post_bad_estimate = pf.get_estimate();

        // Estimate should not jump all the way to the bad measurement
        let altitude_change = (post_bad_estimate[2] - initial_estimate[2]).abs();
        assert!(
            altitude_change < 500.0,
            "Filter jumped {} meters toward bad measurement, should be more conservative",
            altitude_change
        );

        // Now apply several good measurements to recover
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        for _ in 0..10 {
            pf.predict(imu_data, 0.1);

            let good_meas = GPSPositionMeasurement {
                latitude: 45.0,
                longitude: -75.0,
                altitude: 100.0,
                horizontal_noise_std: 5.0,
                vertical_noise_std: 2.0,
            };

            pf.update(&good_meas);
        }

        let recovered_estimate = pf.get_estimate();

        // Should recover toward true altitude
        assert!(
            (recovered_estimate[2] - 100.0).abs() < 50.0,
            "Filter failed to recover, altitude = {} (expected ~100)",
            recovered_estimate[2]
        );
    }

    #[test]
    fn test_particle_filter_process_noise_scaling() {
        // Test that process noise scales properly with time step
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-9; 9]);

        // Create two filters with same initial conditions
        // Include 6 bias states: 3 accel biases + 3 gyro biases
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);
        let mut pf_small_dt = ParticleFilter::new_about(
            mean.clone(),
            other_states.clone(),
            nav_cov.clone(),
            other_cov.clone(),
            100,
            Some(DVector::from_vec(vec![
                1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-6, 1e-6, 1e-6, 1e-6,
                1e-6, 1e-6,
            ])),
            None,
            None,
        );

        let mut pf_large_dt = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            100,
            Some(DVector::from_vec(vec![
                1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-6, 1e-6, 1e-6, 1e-6,
                1e-6, 1e-6,
            ])),
            None,
            None,
        );

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        // Run same total time with different step sizes
        for _ in 0..10 {
            pf_small_dt.predict(imu_data, 0.01);
        }

        pf_large_dt.predict(imu_data, 0.1);

        let cov_small_dt = pf_small_dt.get_certainty();
        let cov_large_dt = pf_large_dt.get_certainty();

        // Covariances should be similar in magnitude
        // (process noise is scaled by sqrt(dt) in predict)
        let ratio = cov_large_dt[(2, 2)] / cov_small_dt[(2, 2)];
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "Covariance ratio {} indicates improper time scaling",
            ratio
        );
    }

    #[test]
    fn test_particle_filter_consistency_with_ukf() {
        // Compare particle filter and UKF on same scenario
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        // Create UKF
        let mut ukf = UnscentedKalmanFilter::new(
            initial_state.clone(),
            vec![0.0; 6],
            None,
            vec![
                1e-6, 1e-6, 1.0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 15])),
            1e-3,
            2.0,
            0.0,
        );

        // Create PF with similar initial conditions
        let mut pf = ParticleFilter::new_about(
            initial_state,
            DVector::from_vec(vec![0.0; 6]), // IMU biases
            DVector::from_vec(vec![1e-6, 1e-6, 1.0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]),
            DVector::from_vec(vec![1e-6; 6]),
            500, // More particles for better comparison
            Some(DVector::from_vec(vec![
                1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-7, 1e-7, 1e-7, 1e-9,
                1e-9, 1e-9,
            ])),
            None,
            Some(ParticleResamplingStrategy::Residual),
        );

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        // Run both filters
        for _ in 0..5 {
            ukf.predict(imu_data, 0.1);
            pf.predict(imu_data, 0.1);

            let meas = GPSPositionMeasurement {
                latitude: 0.0,
                longitude: 0.0,
                altitude: 100.0,
                horizontal_noise_std: 5.0,
                vertical_noise_std: 2.0,
            };

            ukf.update(&meas);
            pf.update(&meas);
        }

        let ukf_estimate = ukf.get_estimate();
        let pf_estimate = pf.get_estimate();

        // Estimates should be reasonably close
        let altitude_diff = (ukf_estimate[2] - pf_estimate[2]).abs();
        assert!(
            altitude_diff < 10.0,
            "UKF and PF altitude estimates differ by {} (UKF: {}, PF: {})",
            altitude_diff,
            ukf_estimate[2],
            pf_estimate[2]
        );
    }

    #[test]
    fn test_relative_altitude_measurement() {
        let meas = RelativeAltitudeMeasurement {
            relative_altitude: 10.0,
            reference_altitude: 100.0,
        };

        assert_eq!(meas.get_dimension(), 1);
        let vec = meas.get_vector();
        assert_approx_eq!(vec[0], 110.0, 1e-6);

        let noise = meas.get_noise();
        assert_eq!(noise.nrows(), 1);

        let state_sigma =
            DMatrix::from_vec(9, 1, vec![0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let meas_sigma = meas.get_sigma_points(&state_sigma);
        assert_approx_eq!(meas_sigma[(0, 0)], 50.0, 1e-6);

        let display = format!("{}", meas);
        assert!(display.contains("10"));
        assert!(display.contains("100"));
    }

    #[test]
    fn test_particle_display_and_from() {
        let s = StrapdownState::default();
        let p = Particle::new(s, None, 0.5);
        let display = format!("{}", p);
        assert!(display.contains("weight"));
        assert!(display.contains("0.5"));

        let vec = DVector::from_vec(vec![0.0; 9]);
        let p_from = Particle::from((vec, 0.1));
        assert_eq!(p_from.state_size, 9);
        assert_eq!(p_from.weight, 0.1);
    }

    // ========== NEW TESTS FOR IMPROVED COVERAGE ==========

    #[test]
    fn test_gps_velocity_as_any_mut() {
        let mut measurement = GPSVelocityMeasurement {
            northward_velocity: 10.0,
            eastward_velocity: 5.0,
            downward_velocity: -2.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 1.0,
        };

        // Test as_any_mut downcast (line 144)
        let any_mut = measurement.as_any_mut();
        assert!(any_mut.downcast_mut::<GPSVelocityMeasurement>().is_some());
    }

    #[test]
    fn test_gps_position_velocity_as_any_mut() {
        let mut measurement = GPSPositionAndVelocityMeasurement {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 5.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
            velocity_noise_std: 0.5,
        };

        // Test as_any_mut downcast (line 201)
        let any_mut = measurement.as_any_mut();
        assert!(
            any_mut
                .downcast_mut::<GPSPositionAndVelocityMeasurement>()
                .is_some()
        );
    }

    #[test]
    fn test_relative_altitude_as_any_mut() {
        let mut measurement = RelativeAltitudeMeasurement {
            relative_altitude: 10.0,
            reference_altitude: 100.0,
        };

        // Test as_any_mut downcast (line 262)
        let any_mut = measurement.as_any_mut();
        assert!(
            any_mut
                .downcast_mut::<RelativeAltitudeMeasurement>()
                .is_some()
        );
    }

    #[test]
    fn test_initial_state_new_degrees() {
        // Test InitialState::new with degrees
        let state = InitialState::new(
            45.0,       // latitude in degrees
            -75.0,      // longitude in degrees
            100.0,      // altitude
            10.0,       // northward_velocity
            5.0,        // eastward_velocity
            0.0,        // downward_velocity
            90.0,       // roll in degrees
            45.0,       // pitch in degrees
            180.0,      // yaw in degrees
            true,       // in_degrees
            Some(true), // is_enu
        );

        assert_eq!(state.latitude, 45.0);
        assert_eq!(state.longitude, -75.0);
        assert_eq!(state.altitude, 100.0);
        assert!(state.in_degrees);
        assert!(state.is_enu);
        // Angles should be converted to radians
        assert_approx_eq!(state.roll, 90.0_f64.to_radians(), 1e-6);
        assert_approx_eq!(state.pitch, 45.0_f64.to_radians(), 1e-6);
        assert_approx_eq!(state.yaw, 180.0_f64.to_radians(), 1e-6);
    }

    #[test]
    fn test_initial_state_new_radians() {
        // Test InitialState::new with radians
        let state = InitialState::new(
            0.785398,  // ~45 degrees in radians
            -1.308997, // ~-75 degrees in radians
            100.0,
            10.0,
            5.0,
            0.0,
            1.5708, // ~90 degrees in radians
            0.7854, // ~45 degrees in radians
            3.1416, // ~180 degrees in radians
            false,  // in_degrees
            Some(true),
        );

        assert!(!state.in_degrees);
        assert!(state.is_enu);
        // Angles should be wrapped to 2pi
        assert!(state.roll >= 0.0 && state.roll < 2.0 * std::f64::consts::PI);
        assert!(state.pitch >= 0.0 && state.pitch < 2.0 * std::f64::consts::PI);
        assert!(state.yaw >= 0.0 && state.yaw < 2.0 * std::f64::consts::PI);
    }

    #[test]
    fn test_initial_state_new_ned() {
        // Test InitialState::new with NED frame
        let state = InitialState::new(
            45.0,
            -75.0,
            -100.0, // negative altitude for NED
            10.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            Some(false), // is_enu = false
        );

        assert!(!state.is_enu);
        assert_eq!(state.altitude, -100.0);
    }

    #[test]
    #[should_panic(expected = "Latitude must be between -90 and +90 degrees")]
    fn test_initial_state_new_invalid_latitude_degrees() {
        // Test latitude validation in degrees
        InitialState::new(
            95.0, // invalid latitude
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            Some(true),
        );
    }

    #[test]
    #[should_panic(expected = "Latitude must be between -90 and +90 degrees")]
    fn test_initial_state_new_invalid_latitude_radians() {
        // Test latitude validation in radians
        InitialState::new(
            2.0, // > 90 degrees in radians
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
            Some(true),
        );
    }

    #[test]
    #[should_panic(expected = "Altitude must be between -10,000 and +30,000 meters in ENU")]
    fn test_initial_state_new_invalid_altitude_enu() {
        // Test altitude validation in ENU
        InitialState::new(
            0.0,
            0.0,
            35000.0, // too high for ENU
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            Some(true),
        );
    }

    #[test]
    #[should_panic(expected = "Altitude must be between 10,000 and -30,000 meters in NED")]
    fn test_initial_state_new_invalid_altitude_ned() {
        // Test altitude validation in NED
        InitialState::new(
            0.0,
            0.0,
            15000.0, // too high for NED (should be negative)
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            Some(false),
        );
    }

    #[test]
    fn test_ukf_debug_display() {
        // Test Debug and Display implementations for UKF
        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        // Test Debug
        let debug_str = format!("{:?}", ukf);
        assert!(debug_str.contains("UKF"));
        assert!(debug_str.contains("mean_state"));

        // Test Display
        let display_str = format!("{}", ukf);
        assert!(display_str.contains("UnscentedKalmanFilter"));
        assert!(display_str.contains("covariance"));
    }

    #[test]
    fn test_ukf_predict_with_biases() {
        // Test UKF predict with non-zero biases
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], // non-zero biases
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        ukf.predict(imu_data, 0.1);

        // Just verify prediction completed without panic
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn test_ukf_update_with_cross_covariance() {
        // Test UKF update to cover cross-covariance calculation
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        // First predict to move state
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        ukf.predict(imu_data, 0.1);

        // Update with GPS position measurement
        let measurement = GPSPositionMeasurement {
            latitude: 0.001,
            longitude: 0.001,
            altitude: 10.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };

        ukf.update(&measurement);

        // Verify update completed
        assert!(ukf.mean_state.len() > 0);
    }

    #[test]
    fn test_particle_from_with_other_states() {
        // Test Particle::from with state vector > 9
        let state_vec = DVector::from_vec(vec![
            0.1, 0.2, 0.3, // position
            1.0, 2.0, 3.0, // velocity
            0.0, 0.0, 0.0, // attitude
            10.0, 11.0, // other states
        ]);

        let particle = Particle::from((state_vec, 0.5));

        assert_eq!(particle.state_size, 11);
        assert_eq!(particle.weight, 0.5);
        assert!(particle.other_states.is_some());
        assert_eq!(particle.other_states.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_particle_averaging_unweighted_with_other_states() {
        // Test unweighted averaging with other_states
        let mut s1 = StrapdownState::default();
        s1.altitude = 10.0;
        let mut s2 = StrapdownState::default();
        s2.altitude = 20.0;

        let p1 = Particle::new(s1, Some(DVector::from_vec(vec![1.0, 2.0])), 0.5);
        let p2 = Particle::new(s2, Some(DVector::from_vec(vec![3.0, 4.0])), 0.5);

        let mut pf = ParticleFilter::new(
            vec![p1, p2],
            Some(DVector::from_vec(vec![0.0; 11])),
            None,
            None,
        );

        pf.averaging_strategy = ParticleAveragingStrategy::UnweightedAverage;
        let estimate = pf.get_estimate();

        // Average altitude: (10 + 20) / 2 = 15
        assert_approx_eq!(estimate[2], 15.0, 1e-6);
        // Average other states: (1 + 3) / 2 = 2, (2 + 4) / 2 = 3
        assert_approx_eq!(estimate[9], 2.0, 1e-6);
        assert_approx_eq!(estimate[10], 3.0, 1e-6);
    }

    #[test]
    fn test_particle_averaging_highest_weight_with_other_states() {
        // Test highest weight averaging with other_states
        let mut s1 = StrapdownState::default();
        s1.altitude = 10.0;
        let mut s2 = StrapdownState::default();
        s2.altitude = 20.0;

        let p1 = Particle::new(s1, Some(DVector::from_vec(vec![1.0, 2.0])), 0.3);
        let p2 = Particle::new(s2, Some(DVector::from_vec(vec![3.0, 4.0])), 0.7);

        let mut pf = ParticleFilter::new(
            vec![p1, p2],
            Some(DVector::from_vec(vec![0.0; 11])),
            None,
            None,
        );

        pf.averaging_strategy = ParticleAveragingStrategy::HighestWeight;
        let estimate = pf.get_estimate();

        // Should get p2's state (highest weight)
        assert_approx_eq!(estimate[2], 20.0, 1e-6);
        assert_approx_eq!(estimate[9], 3.0, 1e-6);
        assert_approx_eq!(estimate[10], 4.0, 1e-6);
    }

    #[test]
    fn test_particle_resampling_strategies() {
        // Test various resampling strategies
        let s = StrapdownState::default();
        let p1 = Particle::new(s.clone(), None, 0.3);
        let p2 = Particle::new(s.clone(), None, 0.3);
        let p3 = Particle::new(s.clone(), None, 0.4);
        let particles = vec![p1, p2, p3];

        // Test Naive
        let strategy = ParticleResamplingStrategy::Naive;
        let resampled = strategy.resample(particles.clone());
        assert_eq!(resampled.len(), 3);

        // Test Systematic
        let strategy = ParticleResamplingStrategy::Systematic;
        let resampled = strategy.resample(particles.clone());
        assert_eq!(resampled.len(), 3);

        // Test Multinomial
        let strategy = ParticleResamplingStrategy::Multinomial;
        let resampled = strategy.resample(particles.clone());
        assert_eq!(resampled.len(), 3);

        // Test Stratified
        let strategy = ParticleResamplingStrategy::Stratified;
        let resampled = strategy.resample(particles.clone());
        assert_eq!(resampled.len(), 3);

        // Test Adaptive
        let strategy = ParticleResamplingStrategy::Adaptive;
        let resampled = strategy.resample(particles.clone());
        assert_eq!(resampled.len(), 3);
    }

    #[test]
    fn test_particle_filter_debug() {
        // Test ParticleFilter Debug implementation
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-6; 9]);
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);

        let pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            100,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            None,
        );

        let debug_str = format!("{:?}", pf);
        assert!(debug_str.contains("ParticleFilter"));
        assert!(debug_str.contains("num_particles"));
        assert!(debug_str.contains("effective_particles"));
        assert!(debug_str.contains("weight_range"));
        assert!(debug_str.contains("mean_position"));
    }

    #[test]
    fn test_particle_filter_normalize_weights_invalid() {
        // Test normalize_weights with invalid sum
        let s = StrapdownState::default();
        let p1 = Particle::new(s.clone(), None, 0.0);
        let p2 = Particle::new(s.clone(), None, 0.0);

        let mut pf = ParticleFilter::new(vec![p1, p2], None, None, None);

        pf.normalize_weights();

        // Weights should be reset to uniform
        assert_approx_eq!(pf.particles[0].weight, 0.5, 1e-6);
        assert_approx_eq!(pf.particles[1].weight, 0.5, 1e-6);
    }

    #[test]
    fn test_particle_filter_predict_without_biases() {
        // Test particle filter predict when other_states is None or < 6
        let s = StrapdownState::default();
        let p = Particle::new(s, None, 1.0);

        let mut pf =
            ParticleFilter::new(vec![p], Some(DVector::from_vec(vec![0.0; 9])), None, None);

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        pf.predict(imu_data, 0.1);

        assert_eq!(pf.particles.len(), 1);
    }

    #[test]
    fn test_particle_filter_update_singular_covariance() {
        // Test particle filter update with singular measurement covariance
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-9; 9]);
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);

        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            10,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            None,
        );

        // Create a measurement model with zero noise (singular covariance)
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 0.0,
            vertical_noise_std: 0.0,
        };

        pf.update(&measurement);

        // Should handle singular matrix gracefully
        assert_eq!(pf.particles.len(), 10);
    }

    #[test]
    fn test_particle_filter_update_negative_determinant() {
        // Test handling of non-positive determinant
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-9; 9]);
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);

        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            10,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            None,
        );

        // Use a very small noise that might cause numerical issues
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1e-100,
            vertical_noise_std: 1e-100,
        };

        pf.update(&measurement);

        // Should handle gracefully
        assert_eq!(pf.particles.len(), 10);
    }

    #[test]
    fn test_particle_filter_get_certainty_strategies() {
        // Test get_certainty with different averaging strategies
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-6; 9]);
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);

        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            50,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            None,
        );

        // Test UnweightedAverage certainty
        pf.averaging_strategy = ParticleAveragingStrategy::UnweightedAverage;
        let cov_unweighted = pf.get_certainty();
        assert_eq!(cov_unweighted.nrows(), 15);
        assert_eq!(cov_unweighted.ncols(), 15);

        // Test HighestWeight certainty
        pf.averaging_strategy = ParticleAveragingStrategy::HighestWeight;
        let cov_highest = pf.get_certainty();
        assert_eq!(cov_highest.nrows(), 15);
        assert_eq!(cov_highest.ncols(), 15);
        // For single particle, covariance should be zero
        assert_approx_eq!(cov_highest[(0, 0)], 0.0, 1e-6);
    }

    #[test]
    fn test_particle_filter_effective_sample_size_zero() {
        // Test effective_sample_size when sum_of_squares is 0
        let s = StrapdownState::default();
        let p1 = Particle::new(s.clone(), None, 0.0);
        let p2 = Particle::new(s.clone(), None, 0.0);

        let pf = ParticleFilter::new(vec![p1, p2], None, None, None);

        let ess = pf.effective_sample_size();
        assert_eq!(ess, 0.0);
    }

    #[test]
    fn test_initial_state_default_is_enu() {
        // Test InitialState::new with None for is_enu (should default to true)
        let state = InitialState::new(
            0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true,
            None, // Should default to true
        );

        assert!(state.is_enu);
    }

    #[test]
    fn test_ukf_with_additional_states() {
        // Test UKF construction with additional states beyond 15
        let measurement_bias = vec![1.0, 2.0, 3.0];
        let total_states = 15 + measurement_bias.len();

        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            Some(measurement_bias),
            vec![1e-6; total_states],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; total_states])),
            ALPHA,
            BETA,
            KAPPA,
        );

        assert_eq!(ukf.state_size, total_states);
        assert_eq!(ukf.mean_state.len(), total_states);
    }

    #[test]
    fn test_particle_filter_with_velocity_measurement() {
        // Test particle filter with velocity measurement
        let mean = InitialState::default();
        let nav_cov = DVector::from_vec(vec![1e-6; 9]);
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);

        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            50,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            None,
        );

        // Update with velocity measurement
        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 1.0,
            eastward_velocity: 0.5,
            downward_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };

        pf.update(&vel_meas);

        // Verify update completed
        let sum_weights: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(sum_weights, 1.0, 1e-6);
    }

    #[test]
    fn test_ukf_with_velocity_measurement() {
        // Test UKF with velocity measurement
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };

        ukf.update(&vel_meas);

        // Verify update completed
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn test_ukf_with_position_velocity_measurement() {
        // Test UKF with combined position and velocity measurement
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let meas = GPSPositionAndVelocityMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };

        ukf.update(&meas);

        // Verify update completed
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn test_particle_filter_with_altitude_measurement() {
        // Test particle filter with relative altitude measurement
        let mean = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let nav_cov = DVector::from_vec(vec![1e-6, 1e-6, 1.0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]);
        let other_states = DVector::from_vec(vec![0.0; 6]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);

        let mut pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            50,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            None,
        );

        let alt_meas = RelativeAltitudeMeasurement {
            relative_altitude: 5.0,
            reference_altitude: 95.0,
        };

        pf.update(&alt_meas);

        let sum_weights: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(sum_weights, 1.0, 1e-6);
    }

    #[test]
    fn test_ukf_with_altitude_measurement() {
        // Test UKF with relative altitude measurement
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let alt_meas = RelativeAltitudeMeasurement {
            relative_altitude: 5.0,
            reference_altitude: 95.0,
        };

        ukf.update(&alt_meas);

        // Should pull altitude toward 100m
        assert!(ukf.mean_state[2] > 90.0 && ukf.mean_state[2] < 110.0);
    }

    #[test]
    fn test_particle_new_with_other_states() {
        // Test Particle::new with other_states to verify state_size calculation
        let nav_state = StrapdownState::default();
        let other = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let particle = Particle::new(nav_state, Some(other.clone()), 0.5);

        assert_eq!(particle.state_size, 12); // 9 nav + 3 other
        assert!(particle.other_states.is_some());
        assert_eq!(particle.other_states.unwrap().len(), 3);
    }

    #[test]
    fn test_initial_state_longitude_wrapping() {
        // Test longitude wrapping in degrees
        let state = InitialState::new(
            0.0,
            370.0, // Should wrap to 10.0
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            Some(true),
        );

        // Longitude should be wrapped
        assert!(state.longitude >= -180.0 && state.longitude <= 180.0);
    }

    #[test]
    fn test_particle_filter_new_about_with_degrees() {
        // Test new_about with in_degrees = true
        let mean = InitialState {
            latitude: 45.0,
            longitude: -75.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: true, // degrees
            is_enu: true,
        };

        let other_states = DVector::from_vec(vec![0.0; 6]);
        let nav_cov = DVector::from_vec(vec![1e-6; 9]);
        let other_cov = DVector::from_vec(vec![1e-6; 6]);

        let pf = ParticleFilter::new_about(
            mean,
            other_states,
            nav_cov,
            other_cov,
            50,
            Some(DVector::from_vec(vec![1e-9; 15])),
            None,
            None,
        );

        assert_eq!(pf.particles.len(), 50);

        // Particles should be initialized around the mean
        let estimate = pf.get_estimate();
        // Estimate should be in radians internally
        assert!(estimate[0].abs() < 2.0); // latitude in radians
        assert!(estimate[1].abs() < 3.0); // longitude in radians
    }
}
