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
use crate::earth::{METERS_TO_DEGREES, relative_barometric_altitude};
use crate::linalg::matrix_square_root;
use crate::{IMUData, StrapdownState, forward, wrap_to_2pi};

use std::fmt::Debug;

use nalgebra::{DMatrix, DVector, Rotation3};
use rand;

/// Basic strapdown state parameters for the UKF and particle filter initialization.
#[derive(Clone, Debug, Default)]
pub struct StrapdownParams {
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
}
/// Generic measurement model trait for all filters
pub trait MeasurementModel {
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
pub struct GPSPositionMeasurement {  // <-- Check this model for degree/radian consistency
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
impl MeasurementModel for GPSPositionMeasurement {
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
impl MeasurementModel for GPSVelocityMeasurement {
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
/// A releative barometric altitude measurement model that describes the relative altitude
/// based on a pressure measurement. This is typically used in barometric altimeters to
/// estimate the altitude relative to a reference pressure.
#[derive(Clone, Debug, Default)]
pub struct RelativeBarometricAltitudeMeasurement {
    /// Measured pressure in Pa
    pub pressure: f64,
    /// Reference pressure in Pa
    pub reference_pressure: f64,
}
impl MeasurementModel for RelativeBarometricAltitudeMeasurement {
    fn get_dimension(&self) -> usize {
        1 // relative altitude
    }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![relative_barometric_altitude(
            self.pressure, self.reference_pressure,
        )])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![1e-3])) // 1 mm noise
    }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(1, state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = relative_barometric_altitude(
                sigma_point[2], self.reference_pressure,
            );
        }
        measurement_sigma_points
    }
}
/// Strapdown Unscented Kalman Filter Inertial Navigation Filter
///
/// This filter uses the Unscented Kalman Filter (UKF) algorithm to estimate the state of a
/// strapdown inertial navigation system. It uses the strapdown equations to propagate the state
/// in the local level frame based on IMU measurements in the body frame. The filter also uses
/// a generic position measurement model to update the state based on position measurements in
/// the local level frame.
///
/// Because of the generic nature of both the UKF and this toolbox, the filter requires the user to
/// implement the measurement model. The measurement model must calculate the measurement sigma points
/// ($\mathcal{Z} = h(\mathcal{X})$) and the innovation matrix ($S$) for the filter. Some basic
/// GNSS-based are provided in this module.
///
/// Note that, internally, angles are always stored in radians (both for the attitude and the position),
/// however, the user can choose to convert them to degrees when retrieving the state vector and the UKF
/// and underlying strapdown state can be constructed from data in degrees by using the boolean `in_degrees`
/// toggle where applicable.
pub struct UKF {
    mean_state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    lambda: f64,
    state_size: usize,
    weights_mean: DVector<f64>,
    weights_cov: DVector<f64>,
}
impl Debug for UKF {
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
impl UKF {
    /// Creates a new UKF with the given initial state, biases, covariance, process noise,
    /// any additional other states, and UKF hyperparameters.
    ///
    /// # Arguments
    /// * `position` - The initial position of the strapdown state.
    /// * `velocity` - The initial velocity of the strapdown state.
    /// * `attitude` - The initial attitude of the strapdown state.
    /// * `imu_biases` - The initial IMU biases.
    /// * `other_states` - Any addtional states the filter is estimating (ex: measurement or sensor bias).
    /// * `covariance_diagonal` - The initial covariance diagonal.
    /// * `process_noise_diagonal` - The process noise diagonal.
    /// * `alpha` - The alpha parameter for the UKF.
    /// * `beta` - The beta parameter for the UKF.
    /// * `kappa` - The kappa parameter for the UKF.
    /// * `in_degrees` - Whether the input vectors are in degrees or radians.
    ///
    /// # Returns
    /// * A new UKF struct.
    pub fn new(
        strapdown_state: StrapdownParams,
        imu_biases: Vec<f64>,
        other_states: Option<Vec<f64>>,
        covariance_diagonal: Vec<f64>,
        process_noise: DMatrix<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> UKF {
        assert!(
            process_noise.nrows() == process_noise.ncols(),
            "Process noise matrix must be square"
        );
        let mut mean = if strapdown_state.in_degrees {
            vec![
                strapdown_state.latitude.to_radians(),
                strapdown_state.longitude.to_radians(),
                strapdown_state.altitude,
                strapdown_state.northward_velocity,
                strapdown_state.eastward_velocity,
                strapdown_state.downward_velocity,
                strapdown_state.roll,
                strapdown_state.pitch,
                strapdown_state.yaw,
            ]
        } else {
            vec![
                strapdown_state.latitude,
                strapdown_state.longitude,
                strapdown_state.altitude,
                strapdown_state.northward_velocity,
                strapdown_state.eastward_velocity,
                strapdown_state.downward_velocity,
                strapdown_state.roll,
                strapdown_state.pitch,
                strapdown_state.yaw,
            ]
        };
        mean.extend(imu_biases);
        if let Some(ref other_states) = other_states {
            mean.extend(other_states.iter().cloned());
        }
        assert!(
            mean.len() >= 15,
            "Expected a cannonical state vector of at least 15 states (position, velocity, attitude, imu biases)"
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
        UKF {
            mean_state,
            covariance,
            process_noise,
            lambda,
            state_size,
            weights_mean,
            weights_cov,
        }
    }
    /// Predicts the state using the strapdown equations and IMU measurements.
    ///
    /// The IMU measurements are used to update the strapdown state in the local level frame.
    /// The IMU measurements are assumed to be in the body frame.
    ///
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the state with (e.g. relative accelerations (m/s^2) and angular rates (rad/s)).
    /// * `dt` - The time step for the propagation.
    ///
    /// # Returns
    /// * none
    pub fn predict(&mut self, imu_data: IMUData, dt: f64) {
        // Propagate the strapdown state using the strapdown equations
        let mut sigma_points = self.get_sigma_points();
        for i in 0..sigma_points.ncols() {
            let mut sigma_point_vec = sigma_points.column(i).clone_owned();
            let mut state = StrapdownState { 
                latitude: sigma_point_vec[0], 
                longitude: sigma_point_vec[1], 
                altitude: sigma_point_vec[2], 
                velocity_north: sigma_point_vec[3], 
                velocity_east:  sigma_point_vec[4], 
                velocity_down: sigma_point_vec[5], 
                attitude: Rotation3::from_euler_angles(
                    sigma_point_vec[6], 
                    sigma_point_vec[7], 
                    sigma_point_vec[8]
                ),
                coordinate_convention: true,
             };
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
        self.covariance = p_bar;
    }
    /// Get the UKF mean state.
    pub fn get_mean(&self) -> DVector<f64> {
        self.mean_state.clone()
    }
    /// Get the UKF covariance.
    pub fn get_covariance(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }
    /// Convert a Vec<SigmaPoint> to a DMatrix<f64>
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
    /// Perform the Kalman measurement update step.
    ///
    /// This method updates the state and covariance based on the measurement and measurement
    /// sigma points. The measurement model is specific to a given implementation of the UKF
    /// and must be provided by the user as the model determines the shape and quantities of
    /// the measurement vector and the measurement sigma points. Measurement models should be
    /// implemented as traits and applied to the UKF as needed.
    ///
    /// This module contains some standard GNSS-aided measurements models (`position_measurement_model`,
    /// `velocity_measurement_model`, and `position_and_velocity_measurement_model`) that can be
    /// used. See the `sim` module for a canonical example of a GPS-aided INS implementation
    /// that uses these models.
    ///
    /// **Note**: Canonical INS implementations use a position measurement model. Typically,
    /// position is reported in _degrees_ for latitude and longitude, and in meters for altitude.
    /// Internally, the UKF stores the latitude and longitude in _radians_, and the measurement models make no
    /// assumptions about the units of the position measurements. However, the user should
    /// ensure that the provided measurement to this function is in the same units as the
    /// measurement model.
    ///
    /// # Arguments
    /// * `measurement` - The measurement vector to update the state with.
    /// * `measurement_sigma_points` - The measurement sigma points to use for the update.
    pub fn update<M: MeasurementModel>(
        &mut self,
        measurement: M,
    ) {
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
        let mut cross_covariance = DMatrix::<f64>::zeros(self.state_size, measurement.get_dimension());
        for (i, measurement_sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            let measurement_diff = measurement_sigma_point - &z_hat;
            let state_diff = sigma_points.column(i) - &self.mean_state;
            cross_covariance += self.weights_cov[i] * state_diff * measurement_diff.transpose();
        }
        // Calculate the Kalman gain
        let s_inv = match s.clone().try_inverse() {
            Some(inv) => inv,
            None => panic!("Innovation matrix is singular"),
        };
        let k = cross_covariance * s_inv;
        // check that the kalman gain and measurement diff are compatible to multiply
        if k.ncols() != measurement.get_dimension() {
            panic!("Kalman gain and measurement differential are not compatible");
        }
        self.mean_state += &k * (measurement.get_vector() - &z_hat);
        // wrap attitude angles to 2pi
        // TODO: #30 Refactor attitude angles to use a more robust representation
        self.mean_state[6] = wrap_to_2pi(self.mean_state[6]);
        self.mean_state[7] = wrap_to_2pi(self.mean_state[7]);
        self.mean_state[8] = wrap_to_2pi(self.mean_state[8]);
        self.covariance -= &k * s * &k.transpose();
    }
}
#[derive(Clone, Debug, Default)]
pub struct Particle {
    /// The strapdown state of the particle
    pub nav_state: StrapdownState,
    /// The weight of the particle
    pub weight: f64,
}

/// Particle filter for strapdown inertial navigation
///
/// This filter uses a particle filter algorithm to estimate the state of a strapdown inertial navigation system.
/// Similarly to the UKF, it uses thin wrappers around the StrapdownState's forward function to propagate the state.
/// The particle filter is a little more generic in implementation than the UKF, as all it fundamentally is is a set
/// of particles and several related functions to propagate, update, and resample the particles.
pub struct ParticleFilter {
    /// The particles in the particle filter
    pub particles: Vec<Particle>,
}
impl ParticleFilter {
    /// Create a new particle filter with the given particles
    ///
    /// # Arguments
    /// * `particles` - The particles to use for the particle filter.
    pub fn new(particles: Vec<Particle>) -> Self {
        ParticleFilter { particles }
    }
    /// Propagate all particles forward using the strapdown equations
    ///
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the particles with.
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64) {
        for particle in &mut self.particles {
            //particle.forward(*imu_data, dt, None);
            forward(&mut particle.nav_state, *imu_data, dt);
        }
    }
    /// Update the weights of the particles based on a measurement
    ///
    /// Generic measurement update function for the particle filter. This function requires the user to provide
    /// a measurement vector and a list of expected measurements for each particle. This list of expected measurements
    /// is the result of a measurement model that is specific to the filter implementation. This model determines
    /// the shape and quantities of the measurement vector and the expected measurements sigma points. This module
    /// contains some standard GNSS-aided measurements models (`position_measurement_model`,
    /// `velocity_measurement_model`, and `position_and_velocity_measurement_model`) that can be used.
    ///
    /// **Note**: Canonical INS implementations use a position measurement model. Typically,
    /// position is reported in _degrees_ for latitude and longitude, and in meters for altitude.
    /// Internally, the particle filter stores the latitude and longitude in _radians_, and the measurement models
    /// make no assumptions about the units of the position measurements. However, the user should
    /// ensure that the provided measurement to this function is in the same units as the
    /// measurement model.
    pub fn update(&mut self, measurement: &DVector<f64>, expected_measurements: &[DVector<f64>]) {
        assert_eq!(self.particles.len(), expected_measurements.len());
        let mut weights = Vec::with_capacity(self.particles.len());
        for expected in expected_measurements.iter() {
            // Calculate the Mahalanobis distance
            let diff = measurement - expected;
            let weight = (-0.5 * diff.transpose() * diff).exp().sum(); //TODO: #22 modify this to use any and/or a user specified probability distribution
            weights.push(weight);
        }
        // self.set_weights(weights.as_slice());
        self.normalize_weights();
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
        if sum > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= sum;
            }
        }
    }
    /// Residual resampling (systematic resampling)
    pub fn residual_resample(&mut self) {
        let n = self.particles.len();
        let mut new_particles = Vec::with_capacity(n);
        let weights: Vec<f64> = self.particles.iter().map(|p| p.weight).collect();
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
                new_particles.push(self.particles[i].clone());
            }
        }
        // Residual part
        let residual_particles = n - new_particles.len();
        if residual_particles > 0 {
            // Normalize residuals
            let sum_residual: f64 = residual.iter().sum();
            //let cumsum = 0.0;
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
                new_particles.push(self.particles[i].clone());
                j += 1;
            }
        }
        // Reset weights
        let uniform_weight = 1.0 / n as f64;
        for particle in &mut new_particles {
            particle.weight = uniform_weight;
        }
        self.particles = new_particles;
    }
}
/// Tests
#[cfg(test)]
mod tests {
    use crate::earth;
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Vector3;

    const IMU_BIASES: [f64; 6] = [0.0; 6];
    const N: usize = 15;
    const COVARIANCE_DIAGONAL: [f64; N] = [1e-9; N];
    const PROCESS_NOISE_DIAGONAL: [f64; N] = [1e-9; N];

    const ALPHA: f64 = 1e-3;
    const BETA: f64 = 2.0;
    const KAPPA: f64 = 0.0;
    const UKF_PARAMS: StrapdownParams = StrapdownParams {
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
    };

    #[test]
    fn ukf_construction() {
        let measurement_bias = vec![0.0; 3]; // Example measurement bias
        let ukf = UKF::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            Some(measurement_bias.clone()),
            vec![1e-3; 18],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-3; 18])),
            ALPHA,
            BETA,
            KAPPA,
        );
        assert_eq!(
            ukf.mean_state.len(),
            18
        );
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
        let ukf = UKF::new(
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
        let mut ukf = UKF::new(
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
        ukf.update(measurement);
        // Check that the state has not changed
        assert_approx_eq!(ukf.mean_state[0], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[1], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[2], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.1);
    }
    //#[test]
    //fn ukf_debug() {
    //    let imu_biases = vec![0.0, 0.0, 0.0];
    //    let measurement_bias = vec![1.0, 1.0, 1.0];
    //    let n = 9 + imu_biases.len() + measurement_bias.len();
    //    let covariance_diagonal = vec![1e-3; n];
    //    let process_noise_diagonal = vec![1e-3; n];
    //    let alpha = 1e-3;
    //    let beta = 2.0;
    //    let kappa = 1e-3;
    //    let ukf_params = StrapdownParams {
    //        latitude: 0.0,
    //        longitude: 0.0,
    //        altitude: 0.0,
    //        northward_velocity: 0.0,
    //        eastward_velocity: 0.0,
    //        downward_velocity: 0.0,
    //        roll: 0.0,
    //        pitch: 0.0,
    //        yaw: 0.0,
    //        in_degrees: false,
    //    };
    //    let ukf = UKF::new(
    //        ukf_params,
    //        imu_biases.clone(),
    //        Some(measurement_bias.clone()),
    //        covariance_diagonal,
    //        DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal)),
    //        alpha,
    //        beta,
    //        kappa,
    //    );
    //    let debug_str = format!("{:?}", ukf);
    //    assert!(debug_str.contains("mean_state"));
    //}
    //#[test]
    //fn test_ukf_hover() {
    //     let imu_data = IMUData::new_from_vec(vec![0.0, 0.0, earth::gravity(&0.0, &0.0)], vec![0.0, 0.0, 0.0]);
    //     let position = vec![0.0, 0.0, 0.0];
    //     let velocity = [0.0, 0.0, 0.0];
    //     let attitude = [0.0, 0.0, 0.0];
    //     let imu_biases = vec![0.0, 0.0, 0.0];
    //     let measurement_bias = vec![0.0, 0.0, 0.0];
    //     let covariance_diagonal = vec![1e-9; 9 + imu_biases.len() + measurement_bias.len()];
    //     let process_noise_diagonal = vec![1e-9; 9 + imu_biases.len() + measurement_bias.len()];
    //     let alpha = 1e-3;
    //     let beta = 2.0;
    //     let kappa = 0.0;
    //     let ukf_params = StrapdownParams {
    //         latitude: position[0],
    //         longitude: position[1],
    //         altitude: position[2],
    //         northward_velocity: velocity[0],
    //         eastward_velocity: velocity[1],
    //         downward_velocity: velocity[2],
    //         roll: attitude[0],
    //         pitch: attitude[1],
    //         yaw: attitude[2],
    //         in_degrees: false,
    //     };
    //     let mut ukf = UKF::new(
    //         ukf_params,
    //         imu_biases.clone(),
    //         Some(measurement_bias.clone()),
    //         covariance_diagonal,
    //         DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal)),
    //         alpha,
    //         beta,
    //         kappa,
    //     );
    //     let dt = 1.0;
    //     let measurement_sigma_points = ukf.position_measurement_model(true);
    //     let measurement_noise = ukf.position_measurement_noise(true);
    //     let measurement = DVector::from_vec(position.clone());
    //     for _i in 0..60 {
    //         ukf.predict(&imu_data, dt);
    //         ukf.update(&measurement, &measurement_sigma_points, &measurement_noise);
    //     }
    //     assert_approx_eq!(ukf.mean_state[0], position[0], 1e-3);
    //     assert_approx_eq!(ukf.mean_state[1], position[1], 1e-3);
    //     assert_approx_eq!(ukf.mean_state[2], position[2], 0.01);
    //     assert_approx_eq!(ukf.mean_state[3], velocity[0], 0.01);
    //     assert_approx_eq!(ukf.mean_state[4], velocity[1], 0.01);
    //     assert_approx_eq!(ukf.mean_state[5], velocity[2], 0.01);
    // }
}
