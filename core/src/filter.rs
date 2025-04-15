//! Inertial Navigation Filters
//! This module contains implementations of various inertial navigation filters, including
//! Kalman filters and particle filters. These filters are used to estimate the state of a
//! strapdown inertial navigation system based on IMU measurements and other sensor data.
//! The filters use the strapdown equations (provided by the StrapdownState) to propagate 
//! the state in the local level frame. The filters also use a generic position measurement
//! model to update the state based on position measurements in the local level frame.

use nalgebra::{SMatrix, SVector, DMatrix, DVector, Vector3, Matrix3, ArrayStorage};
use crate::{IMUData, StrapdownState};

pub struct SigmaPoint {
    nav_state: StrapdownState,
    other_states: Vec<f64>   // I'm keeping this as generic as possible. 
}

impl SigmaPoint {
    fn new(nav_state: StrapdownState, other_states: Vec<f64>) -> SigmaPoint {
        SigmaPoint {
            nav_state,
            other_states
        }
    }
    fn get_state(&self) -> DVector<f64> {
        let mut state = self.nav_state.to_vector(false).as_slice().to_vec();
        state.extend(self.other_states.as_slice());
        DVector::from_vec(state)
    }
    fn get_state_size(&self) -> usize {
        9 + self.other_states.len()
    }
    fn forward(&mut self, imu_data: &IMUData, dt: f64) {
        // Propagate the strapdown state using the strapdown equations
        self.nav_state.forward(imu_data, dt);
    }
    fn to_vector(&self) -> DVector<f64> {
        let mut state = self.nav_state.to_vector(false).as_slice().to_vec();
        state.extend(self.other_states.as_slice());
        return DVector::from_vec(state);
    }
    fn from_vector(state: DVector<f64>) -> SigmaPoint {
        let nav_state = StrapdownState::new_from_vector(
            SMatrix::from_iterator(state.view_range(0..9, 0).iter().cloned()),
        );
        let other_states = state.view_range(9..state.len(), 0);
        let sigma_point = SigmaPoint::new(nav_state, other_states.as_slice().to_vec());
        return sigma_point;
    }
}
/// Strapdown Unscented Kalman Filter Inertial Navigation Filter
/// This filter uses the Unscented Kalman Filter (UKF) algorithm to estimate the state of a 
/// strapdown inertial navigation system. It uses the strapdown equations to propagate the state
/// in the local level frame based on IMU measurements in the body frame. The filter also uses
/// a generic position measurement model to update the state based on position measurements in 
/// the local level frame.
/// 
pub struct UKF {
    mean_state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    measurement_noise: DMatrix<f64>,
    alpha: f64,
    lambda: f64,
    state_size: usize,
    weights_mean: DVector<f64>,
    weights_cov: DVector<f64>,
}

impl UKF {
    /// Creates a new UKF with the given initial state, biases, covariance, process noise,
    /// measurement noise, and UKF parameters.
    pub fn new(
        position: Vec<f64>,
        velocity: Vec<f64>,
        attitude: Vec<f64>,
        imu_biases: Vec<f64>,
        measurement_bias: Vec<f64>,
        covariance_diagonal: Vec<f64>,
        process_noise_diagonal: Vec<f64>,
        measurement_noise_diagonal: Vec<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> UKF {
        let mut mean: Vec<f64> = position;
        mean.extend(velocity);
        mean.extend(attitude);
        mean.extend(imu_biases);
        mean.extend(measurement_bias);
        assert!(
            mean.len() == covariance_diagonal.len(),
            "Mean state and covariance diagonal must be of the same size"
        );
        let state_size = mean.len() as usize;
        let mean_state = DVector::from_vec(mean);
        let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));
        let measurement_noise = DMatrix::from_diagonal(&DVector::from_vec(measurement_noise_diagonal));
        let covariance = DMatrix::<f64>::from_diagonal(&DVector::from_vec(covariance_diagonal));
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
            measurement_noise,
            alpha,
            lambda,
            state_size, 
            weights_mean,
            weights_cov,
        }
    }
    /// Propagates the state using the strapdown equations and IMU measurements.
    /// The IMU measurements are used to update the strapdown state in the local level frame.
    /// The IMU measurements are assumed to be in the body frame.
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the state with.
    /// * `dt` - The time step for the propagation.
    /// # Returns
    /// * none
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64) {
        // Propagate the strapdown state using the strapdown equations
        let mut sigma_points = self.get_sigma_points(&self.mean_state, &self.covariance);
        for sigma_point in &mut sigma_points {
            // Propagate the strapdown state using the strapdown equations
            sigma_point.forward(imu_data, dt);
        }
        let mut mu_bar = DVector::<f64>::zeros(self.state_size);
        // Update the mean state through a naive loop
        for i in 0..(2 * self.state_size + 1) {
            mu_bar += self.weights_mean[i] * &sigma_points[i].get_state();
        }
        let mut cov_bar = DMatrix::<f64>::zeros(self.state_size, self.state_size);
        // Update the covariance through a naive loop
        for i in 0..(2*self.state_size + 1) {
            let sigma_point = &sigma_points[i];
            let weight_cov = self.weights_cov[i];
            let diff = &sigma_point.get_state() - &mu_bar;
            cov_bar += weight_cov * &diff * &diff.transpose();
        }
        self.mean_state = mu_bar;
        self.covariance = cov_bar + &self.process_noise;        
    }
    // --- Working notes ---------------------------------------------------------
    // Issue here is that this is a generic strapdown INS filter, so while we can 
    // maker reasonable assumptions about the generic propagation function of the
    // UKF, (i.e. the strapdown equations), we don't neccessarily know what the
    // measurement model is on this generic level.
    /// --------------------------------------------------------------------------

    /// Perform the Kalman measurement update
    /// This method updates the state and covariance based on the measurement and innovation.
    /// The innovation matrix must be calculated based on the measurement model.
    pub fn update(&mut self, measurement: DVector<f64>, measurement_sigma_points: Vec<DVector<f64>>, innovation: DMatrix<f64>) {
        // Assert that the measurement is the correct size as the measurement noise diagonal
        assert!(
            measurement.len() == self.measurement_noise.nrows(),
            "Measurement and measurement noise must be of the same size"
        );
        // Calculate expected measurement
        let mut z_hat = DVector::<f64>::zeros(measurement.len());
        for i in 0..measurement_sigma_points.len() {
            let sigma_point = &measurement_sigma_points[i];
            let weight_mean = self.weights_mean[i];
            z_hat += weight_mean * sigma_point;
        }
        // Calculate innovation matrix S
        let mut s = DMatrix::<f64>::zeros(measurement.len(), measurement.len());
        for i in 0..measurement_sigma_points.len() {
            let diff = &measurement_sigma_points[i] - &z_hat;
            s += self.weights_cov[i] * &diff * &diff.transpose();
        }
        s += &self.measurement_noise;
        // Calculate the cross-covariance
        let sigma_points = self.get_sigma_points(&self.mean_state, &self.covariance);
        let mut cross_covariance = DMatrix::<f64>::zeros(self.state_size, measurement.len());
        for i in 0..measurement_sigma_points.len() {
            let measurement_diff = &measurement_sigma_points[i] - &z_hat;
            let sigma_point = &sigma_points[i].to_vector();
            let state_diff = sigma_point - &self.mean_state;
            cross_covariance += self.weights_cov[i] * state_diff * measurement_diff.transpose();
        }
        // Calculate the Kalman gain
        let s_inv = match innovation.clone().try_inverse() {
            Some(inv) => inv,
            None => panic!("Innovation matrix is singular"),
        };
        let k = cross_covariance * s_inv;
        // check that the kalman gain and measurmente diff are compatible to multiply
        if k.ncols() != measurement.len() {
            panic!("Kalman gain and measurement differential are not compatible");
        }
        // Perform Kalman update
        self.mean_state += &k * (measurement - &z_hat);
        self.covariance -= &k * innovation * &k.transpose();        
    }
    /// Get the UKF mean state.
    /// The mean state is the current navigation state with the additional imu and measurement biases appended.
    pub fn get_mean(&self) -> DVector<f64> {
        return self.mean_state.clone();      
    }
    // Working notes:
    //  Because of my choice to seperately define the strapdown propagation, I'll need to rework the 
    //  the UKF propagation step to resemble something more like a limited particle filter rather than
    //  the more traditional purely linear algebra vector-matrix operations of a traditional KF/EKF/UKF.

    /// Calculate the sigma points for the UKF.
    /// The sigma points are calculated based on the current state and covariance.
    pub fn get_sigma_points(&self, mean: &DVector<f64>, covariance: &DMatrix<f64>) -> Vec<SigmaPoint> {
        
        //let mut sigma_points = DMatrix::<f64>::zeros(self.state_size, 2 * self.state_size + 1);
        //sigma_points.column_mut(0).copy_from(&self.mean_state);
        let mut sqrt_cov = (self.state_size as f64 + self.lambda) * covariance.clone(); 
        sqrt_cov = sqrt_cov.cholesky().unwrap().l();

        let mut sigma_points = Vec::<SigmaPoint>::with_capacity(2*self.state_size + 1);
        // Add the mean state as the first sigma point
        sigma_points[0] = SigmaPoint::from_vector(self.mean_state.clone());
        for i in 0..self.state_size {
            let sqrt_cov_i = sqrt_cov.column(i);
            let left = mean + sqrt_cov_i;
            let right = mean - sqrt_cov_i;
            sigma_points[i] = SigmaPoint::from_vector(left);
            sigma_points[i + self.state_size] = SigmaPoint::from_vector(right);
        }
        return sigma_points;
    }
}

// TODO: #84 Add particle filter