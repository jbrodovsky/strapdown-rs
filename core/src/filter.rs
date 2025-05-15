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
use std::fmt::Debug;
use rand;
use rand_distr::{Distribution, Normal};
use nalgebra::{SMatrix, DMatrix, DVector};
use crate::{IMUData, StrapdownState};

/// Helper struct for UKF implementation
/// 
/// This struct is used to represent a sigma point in the UKF. It contains the strapdown state
/// and other states. The strapdown state is used to propagate the state using the strapdown
/// navigation equations
#[derive(Clone, Debug)]
pub struct SigmaPoint {
    /// The basic local level frame strapdown state
    nav_state: StrapdownState,
    /// The other states to be estimated by the UKF
    other_states: Vec<f64>,
}
impl SigmaPoint {
    /// Create new sigma point from a strapdown state vector plus whatever other states the UKF
    /// estimates.
    /// 
    /// # Arguments
    /// 
    /// * `nav_state` - The strapdown state to use for the sigma point.
    /// * `other_states` - The other states to use for the sigma point.
    /// # Returns
    /// * A new SigmaPoint struct.
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use strapdown::StrapdownState;
    /// use nalgebra::SVector;
    /// 
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros());
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let sigma_point = SigmaPoint::new(nav_state, other_states);
    /// ```
    /// 
    pub fn new(nav_state: StrapdownState, other_states: Vec<f64>) -> SigmaPoint {
        SigmaPoint {
            nav_state,
            other_states
        }
    }
    /// Get the strapdown state of the sigma point as an nalgebra vector.
    /// 
    /// # Returns
    /// * A vector containing the strapdown state and other states.
    /// 
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use strapdown::StrapdownState;
    /// use nalgebra::SVector;
    /// 
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros());
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let sigma_point = SigmaPoint::new(nav_state, other_states);
    /// 
    /// let state = sigma_point.get_state();
    /// assert_eq!(state.len(), 12);
    /// ```
    pub fn get_state(&self) -> DVector<f64> {
        let mut state = self.nav_state.to_vector(false).as_slice().to_vec();
        state.extend(self.other_states.iter());
        return DVector::from_vec(state);
    }
    /// Forward mechanization (propagation) of the strapdown state.
    /// 
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the state with.
    /// * `dt` - The time step for the propagation.
    /// # Returns
    /// * none
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use strapdown::StrapdownState;
    /// use strapdown::IMUData;
    /// use nalgebra::SVector;
    /// 
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros());
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let mut sigma_point = SigmaPoint::new(nav_state, other_states);
    /// 
    /// let imu_data = IMUData::new_from_vec(vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
    /// let dt = 0.1;
    /// 
    /// sigma_point.forward(&imu_data, dt);
    /// ```
    pub fn forward(&mut self, imu_data: &IMUData, dt: f64) {
        // Propagate the strapdown state using the strapdown equations
        self.nav_state.forward(imu_data, dt);
    }
    /// Convert the sigma point to an nalgebra vector.
    /// 
    /// # Returns
    /// * A vector containing the strapdown state and other states.
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use strapdown::StrapdownState;
    /// use nalgebra::SVector;
    /// 
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros());
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let sigma_point = SigmaPoint::new(nav_state, other_states);
    /// 
    /// let state = sigma_point.to_vector();
    /// assert_eq!(state.len(), 12);
    /// ```
    pub fn to_vector(&self) -> DVector<f64> {
        let mut state = self.nav_state.to_vector(false).as_slice().to_vec();
        state.extend(self.other_states.as_slice());
        return DVector::from_vec(state);
    }
    /// Convert an nalgebra vector to a sigma point.
    /// 
    /// # Arguments
    /// * `state` - The vector to convert to a sigma point.
    /// # Returns
    /// * A new SigmaPoint struct.
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use nalgebra::DVector;
    /// 
    /// let state = DVector::from_vec(vec![0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]);
    /// let sigma_point = SigmaPoint::from_vector(state);
    /// ```
    pub fn from_vector(state: DVector<f64>) -> SigmaPoint {
        let nav_state = StrapdownState::new_from_vector(
            SMatrix::from_iterator(state.view_range(0..9, 0).iter().cloned()),
        );
        let other_states = state.view_range(9..state.len(), 0);
        let sigma_point = SigmaPoint::new(nav_state, other_states.as_slice().to_vec());
        return sigma_point;
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
/// ($\mathcal{Z} = h(\mathcal{X})$) and the innovation matrix ($S$) for the filter.
pub struct UKF {
    mean_state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    measurement_noise: DMatrix<f64>,
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
            .field("measurement_noise", &self.measurement_noise)
            .field("lambda", &self.lambda)
            .field("state_size", &self.state_size)
            .finish()
    }
}

impl UKF {
    /// Creates a new UKF with the given initial state, biases, covariance, process noise,
    /// measurement noise, and UKF parameters.
    /// 
    /// # Arguments
    /// * `position` - The initial position of the strapdown state.
    /// * `velocity` - The initial velocity of the strapdown state.
    /// * `attitude` - The initial attitude of the strapdown state.
    /// * `imu_biases` - The initial IMU biases.
    /// * `measurement_bias` - The initial measurement biases.
    /// * `covariance_diagonal` - The initial covariance diagonal.
    /// * `process_noise_diagonal` - The process noise diagonal.
    /// * `measurement_noise_diagonal` - The measurement noise diagonal.
    /// * `alpha` - The alpha parameter for the UKF.
    /// * `beta` - The beta parameter for the UKF.
    /// * `kappa` - The kappa parameter for the UKF.
    /// # Returns
    /// * A new UKF struct.
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
            lambda,
            state_size, 
            weights_mean,
            weights_cov,
        }
    }
    /// Propagates the state using the strapdown equations and IMU measurements.
    /// 
    /// The IMU measurements are used to update the strapdown state in the local level frame.
    /// The IMU measurements are assumed to be in the body frame.
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the state with.
    /// * `dt` - The time step for the propagation.
    /// # Returns
    /// * none
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64) {
        // Propagate the strapdown state using the strapdown equations
        let mut sigma_points = self.get_sigma_points();
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
    /// Perform the Kalman measurement update step.
    /// 
    /// This method updates the state and covariance based on the measurement and measurement 
    /// sigma points. The measurement model is specific to a given implementation of the UKF
    /// and must be provided by the user. This model determines the shape and quantities of
    /// the measurement vector and the measurement sigma points. See the `sim` module for 
    /// the canonical example of a GPS-aided INS implementation. 
    /// 
    /// # Arguments
    /// * `measurement` - The measurement vector to update the state with.
    /// * `measurement_sigma_points` - The measurement sigma points to use for the update.
    /// 
    /// # Returns
    /// * none
    pub fn update(&mut self, measurement: &DVector<f64>, measurement_sigma_points: &Vec<DVector<f64>>) {
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
        let sigma_points = self.get_sigma_points();
        let mut cross_covariance = DMatrix::<f64>::zeros(self.state_size, measurement.len());
        for i in 0..measurement_sigma_points.len() {
            let measurement_diff = &measurement_sigma_points[i] - &z_hat;
            //let sigma_point = &sigma_points[i].to_vector();
            let state_diff = &sigma_points[i].to_vector() - &self.mean_state;
            cross_covariance += self.weights_cov[i] * state_diff * measurement_diff.transpose();
        }
        // Calculate the Kalman gain
        let s_inv = match s.clone().try_inverse() {
            Some(inv) => inv,
            None => panic!("Innovation matrix is singular"),
        };
        let k = cross_covariance * s_inv;
        // check that the kalman gain and measurement diff are compatible to multiply
        if k.ncols() != measurement.len() {
            panic!("Kalman gain and measurement differential are not compatible");
        }
        // Perform Kalman update
        self.mean_state += &k * (measurement - &z_hat);
        self.covariance -= &k * s * &k.transpose();        
    }
    /// Get the UKF mean state.
    pub fn get_mean(&self) -> DVector<f64> {
        return self.mean_state.clone();      
    }
    /// Get the UKF covariance.
    pub fn get_covariance(&self) -> DMatrix<f64> {
        return self.covariance.clone();
    }
    /// Calculate the sigma points for the UKF propagation based on the current state and covariance.
    pub fn get_sigma_points(&self) -> Vec<SigmaPoint> {
        //let mut sigma_points = DMatrix::<f64>::zeros(self.state_size, 2 * self.state_size + 1);
        //sigma_points.column_mut(0).copy_from(&self.mean_state);
        // let mean = self.get_mean();
        let mut sqrt_cov = (self.state_size as f64 + self.lambda) * self.covariance.clone(); 
        sqrt_cov = sqrt_cov.cholesky().unwrap().l();
        let mut sigma_points = Vec::<SigmaPoint>::with_capacity(2*self.state_size + 1);
        // Add the mean state as the first sigma point
        sigma_points.push(SigmaPoint::from_vector(self.mean_state.clone()));
        let mut left: Vec<SigmaPoint> = Vec::with_capacity(self.state_size);
        let mut right: Vec<SigmaPoint> = Vec::with_capacity(self.state_size);
        for i in 0..self.state_size {
            let sqrt_cov_i = sqrt_cov.column(i);
            left.push( SigmaPoint::from_vector(self.mean_state.clone() + sqrt_cov_i));
            right.push(SigmaPoint::from_vector(self.mean_state.clone() - sqrt_cov_i));
        }
        sigma_points.append(&mut left);
        sigma_points.append(&mut right);
        return sigma_points;
    }
}
/// UKF GPS or Position-based measurement model
/// 
/// Standard GPS-aided INS measurement model. This model is used to update the state based on
/// position measurements in the local level frame. The model assumes that the position 
/// measurements are in the local level frame and that the strapdown state is in the body frame.
/// 
/// # Arguments
/// * `sigma_points` - the strapdown UKF's sigma points
/// 
/// # Returns
/// * A vector of measurement sigma points
pub fn ukf_position_measurement_model(sigma_points: &Vec<SigmaPoint>) -> Vec<DVector<f64>> {
    let mut measurement_sigma_points = Vec::<DVector<f64>>::with_capacity(sigma_points.len());
    for sigma_point in sigma_points {
        let state = sigma_point.get_state();
        // Extract the position from the state
        //let position = state.view_range(0..3, 0);
        // Convert to a DVector
        //let measurement = DVector::from_vec(position.as_slice().to_vec());
        measurement_sigma_points.push(state.view_range(0..3, 0).as_slice().to_vec().into());
    }
    return measurement_sigma_points;
}
/// UKF velocity-based measurement model
/// 
/// Velocity measurement model. This model is used to update the state based on
/// velocity measurements in the local level frame. The model assumes that the velocity 
/// measurements are in the local level frame and that the strapdown state is in the 
/// body frame.
/// 
/// # Arguments
/// * `sigma_points` - the strapdown UKF's sigma points
/// 
/// # Returns
/// * A vector of measurement sigma points
pub fn ukf_velocity_measurement_model(sigma_points: &Vec<SigmaPoint>) -> Vec<DVector<f64>> {
    let mut measurement_sigma_points = Vec::<DVector<f64>>::with_capacity(sigma_points.len());
    for sigma_point in sigma_points {
        let state = sigma_point.get_state();
        // Extract the position from the state
        //let position = state.view_range(0..3, 0);
        // Convert to a DVector
        //let measurement = DVector::from_vec(position.as_slice().to_vec());
        measurement_sigma_points.push(state.view_range(3..6, 0).as_slice().to_vec().into());
    }
    return measurement_sigma_points;
}
/// UKF GPS or Position-based measurement model
/// 
/// GPS-aided INS measurement model for a combined position-velocity measuremet. 
/// This model is used to update the state based on both position and velocity 
/// measurements in the local level frame. The model assumes that the position 
/// measurements are in the local level frame (latitude, longitude, altitude)
/// and that the velocity is orient along the northward, eastward, and downward
/// axes.
/// 
/// # Arguments
/// * `sigma_points` - the strapdown UKF's sigma points
/// 
/// # Returns
/// * A vector of measurement sigma points
pub fn ukf_position_velocity_measurement_model(sigma_points: &Vec<SigmaPoint>) -> Vec<DVector<f64>> {
    let mut measurement_sigma_points = Vec::<DVector<f64>>::with_capacity(sigma_points.len());
    for sigma_point in sigma_points {
        let state = sigma_point.get_state();
        // Extract the position from the state
        //let position = state.view_range(0..3, 0);
        // Convert to a DVector
        //let measurement = DVector::from_vec(position.as_slice().to_vec());
        measurement_sigma_points.push(state.view_range(0..6, 0).as_slice().to_vec().into());
    }
    return measurement_sigma_points;
}
/// Particle for the particle filter
#[derive(Clone)]
pub struct Particle {
    pub nav_state: StrapdownState,
    pub accel_bias: Vec<f64>,
    pub gyro_bias: Vec<f64>,
    pub other_states: Vec<f64>,
    pub weight: f64,
}
impl Particle {
    pub fn new(nav_state: StrapdownState, accel_bias: Vec<f64>, gyro_bias: Vec<f64>, other_states: Vec<f64>, weight: f64) -> Self {
        Particle {
            nav_state,
            accel_bias,
            gyro_bias,
            other_states,
            weight,
        }
    }
    /// Propagate the particle using the strapdown equations with an optional bias random walk
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64, noise: Option<Vec<f64>>) {
        let mut rng = rand::rng();
        self.nav_state.forward(imu_data, dt);
        // Optionally, you could add bias random walk here if desired
        let mut mean = self.nav_state.to_vector(true);
        // unpack noise if some, otherwise uses zeros
        let noise_vect  = match noise {
            None => vec![0.0; mean.len() + self.accel_bias.len() + self.gyro_bias.len() + self.other_states.len()],
            Some(n) => n,
        };
        assert!(noise_vect.len() == mean.len() + self.accel_bias.len() + self.gyro_bias.len() + self.other_states.len(), "Noise vector must be the same size as the state vector");
        // Sample from zero mean gaussian noise with stddev of noise_vect
        let mut jitter = Vec::with_capacity(mean.len()+ self.accel_bias.len() + self.gyro_bias.len() + self.other_states.len());
        for stddev in noise_vect.iter() {
            let normal = Normal::new(0.0, *stddev).unwrap();
            jitter.push(normal.sample(&mut rng));
        }
        let jitter = DVector::from_vec(jitter);
        mean += jitter;
        self.nav_state = StrapdownState::new_from_vector(mean);
        self.accel_bias = mean.as_slice()[9..12].to_vec();
        self.gyro_bias = mean.as_slice()[12..15].to_vec();
        self.other_states = mean.as_slice()[15..].to_vec();        
    }
}
/// Particle filter for strapdown inertial navigation
pub struct ParticleFilter {
    pub particles: Vec<Particle>,
}
impl ParticleFilter {
    /// Create a new particle filter with the given particles
    pub fn new(particles: Vec<Particle>) -> Self {
        ParticleFilter { particles }
    }
    /// Propagate all particles forward using the strapdown equations
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64) {
        for particle in &mut self.particles {
            particle.propagate(imu_data, dt, None);
        }
    }
    /// Set the weights of the particles (e.g., after a measurement update)
    pub fn set_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.particles.len());
        for (particle, &w) in self.particles.iter_mut().zip(weights.iter()) {
            particle.weight = w;
        }
    }
    /// Normalize the weights of the particles
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
        let mut total_residual = 0.0;
        // Integer part
        for (i, &w) in weights.iter().enumerate() {
            let copies = (w * n as f64).floor() as usize;
            num_copies[i] = copies;
            residual[i] = w * n as f64 - copies as f64;
            total_residual += residual[i];
        }
        // Copy integer part
        for (i, &copies) in num_copies.iter().enumerate() {
            for _ in 0..copies {
                new_particles.push(self.particles[i].clone());
            }
        }
        // Residual part
        let mut residual_particles = n - new_particles.len();
        if residual_particles > 0 {
            // Normalize residuals
            let sum_residual: f64 = residual.iter().sum();
            let mut cumsum = 0.0;
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
    use super::*;
    use nalgebra::{SVector, Vector3};
    use assert_approx_eq::assert_approx_eq;
    use crate::earth;

    // Test sigma point functionality
    #[test]
    fn test_sigma_point() {        
        let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros());
        let other_states = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut sigma_point = SigmaPoint::new(nav_state.clone(), other_states.clone());
        assert_eq!(sigma_point.get_state().len(), 15);
        
        let state = sigma_point.get_state();
        let state_vector = sigma_point.to_vector();
        assert_eq!(state.len(), state_vector.len());
        let sigma2 = SigmaPoint::from_vector(state_vector);
        assert_eq!(sigma_point.get_state(), sigma2.get_state());

        // test forward propagation
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0), // Currently configured as relative body-frame acceleration
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        let dt = 1.0;
        sigma_point.forward(&imu_data, dt);

        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];

        assert_approx_eq!(sigma_point.nav_state.position[0], position[0], 1e-3);
        assert_approx_eq!(sigma_point.nav_state.position[1], position[1], 1e-3);
        assert_approx_eq!(sigma_point.nav_state.position[2], position[2], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[0], velocity[0], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[1], velocity[1], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[2], velocity[2], 0.1);

        let debug_str = format!("{:?}", sigma_point);
        assert!(debug_str.contains("nav_state"));
    }

    #[test]
     fn test_ukf_construction() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![0.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let covariance_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
        let process_noise_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
        let measurement_noise_diagonal = vec![1e-3; 3];
        let alpha = 1e-3;
        let beta = 2.0;
        let kappa = 1e-3;
        let ukf = UKF::new(
            position.clone(),
            velocity.clone(),
            attitude.clone(),
            imu_biases.clone(),
            measurement_bias.clone(),
            covariance_diagonal,
            process_noise_diagonal,
            measurement_noise_diagonal,
            alpha,
            beta,
            kappa,
        );
        assert_eq!(ukf.mean_state.len(), position.len() + velocity.len() + attitude.len() + imu_biases.len() + measurement_bias.len());
    }
    #[test]
    #[should_panic]
    fn test_ukf_construction_panic() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![1.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let covariance_diagonal = vec![1e-3; 9];
        let process_noise_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
        let measurement_noise_diagonal = vec![1e-3; 3];
        let alpha = 1e-3;
        let beta = 2.0;
        let kappa = 1e-3;
        UKF::new(
            position.clone(),
            velocity.clone(),
            attitude.clone(),
            imu_biases.clone(),
            measurement_bias.clone(),
            covariance_diagonal,
            process_noise_diagonal,
            measurement_noise_diagonal,
            alpha,
            beta,
            kappa,
        );
    }
    #[test]
    fn test_ukf_get_sigma_points() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![1.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let covariance_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
        let process_noise_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
        let measurement_noise_diagonal = vec![1e-3; 3];
        let alpha = 1e-3;
        let beta = 2.0;
        let kappa = 1e-3;
        let ukf = UKF::new(
            position.clone(),
            velocity.clone(),
            attitude.clone(),
            imu_biases.clone(),
            measurement_bias.clone(),
            covariance_diagonal,
            process_noise_diagonal,
            measurement_noise_diagonal,
            alpha,
            beta,
            kappa,
        );
        let sigma_points = ukf.get_sigma_points();
        assert_eq!(sigma_points.len(), (2 * ukf.state_size) + 1);
    }
    #[test]
    fn test_ukf_propagate() {
        let imu_data = IMUData::new_from_vec(
            vec![0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0]
        );
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![0.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![0.0, 0.0, 0.0];
        let covariance_diagonal = vec![1e-12; 9 + imu_biases.len() + measurement_bias.len()];
        let process_noise_diagonal = vec![1e-12; 9 + imu_biases.len() + measurement_bias.len()];
        let measurement_noise_diagonal = vec![1e-12; 3];
        let alpha = 1e-3;
        let beta = 2.0;
        let kappa = 1e-3;
        let mut ukf = UKF::new(
            position.clone(),
            velocity.clone(),
            attitude.clone(),
            imu_biases.clone(),
            measurement_bias.clone(),
            covariance_diagonal,
            process_noise_diagonal,
            measurement_noise_diagonal,
            alpha,
            beta,
            kappa,
        );
        let dt = 1.0;
        ukf.propagate(&imu_data, dt);
        assert!(ukf.mean_state.len() == position.len() + velocity.len() + attitude.len() + imu_biases.len() + measurement_bias.len());
        assert_approx_eq!(ukf.mean_state[0], position[0], 1e-3);
        assert_approx_eq!(ukf.mean_state[1], position[1], 1e-3);
        assert_approx_eq!(ukf.mean_state[2], position[2], 0.1);
        assert_approx_eq!(ukf.mean_state[3], velocity[0], 0.1);
        assert_approx_eq!(ukf.mean_state[4], velocity[1], 0.1);
        assert_approx_eq!(ukf.mean_state[5], velocity[2], 0.1);
    }
    #[test]
    fn test_ukf_debug() {
        let ukf = UKF::new(
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0e-3; 15],
            vec![1.0e-3; 15],
            vec![1e-3; 3],
            1e-3,
            2.0,
            1e-3,
        );
        let debug_str = format!("{:?}", ukf);
        assert!(debug_str.contains("mean_state"));
    }
    #[test]
    fn test_ukf_hover() {
        let imu_data = IMUData::new_from_vec(
            vec![0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0]
        );
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![0.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![0.0, 0.0, 0.0];
        let covariance_diagonal = vec![1e-12; 9 + imu_biases.len() + measurement_bias.len()];
        let process_noise_diagonal = vec![1e-12; 9 + imu_biases.len() + measurement_bias.len()];
        let measurement_noise_diagonal = vec![1e-12; 3];
        let alpha = 1e-3;
        let beta = 2.0;
        let kappa = 1e-3;
        let mut ukf = UKF::new(
            position.clone(),
            velocity.clone(),
            attitude.clone(),
            imu_biases.clone(),
            measurement_bias.clone(),
            covariance_diagonal,
            process_noise_diagonal,
            measurement_noise_diagonal,
            alpha,
            beta,
            kappa,
        );
        let dt = 1.0;
        let measurement_sigmas = ukf_position_measurement_model(&ukf.get_sigma_points());
        let measurement = DVector::from_vec(position.clone());
        for _i in 0..60 {
            ukf.propagate(&imu_data, dt);
            ukf.update(&measurement, &measurement_sigmas);
        }

        assert_approx_eq!(ukf.mean_state[0], position[0], 1e-3);
        assert_approx_eq!(ukf.mean_state[1], position[1], 1e-3);
        assert_approx_eq!(ukf.mean_state[2], position[2], 0.01);
        assert_approx_eq!(ukf.mean_state[3], velocity[0], 0.01);
        assert_approx_eq!(ukf.mean_state[4], velocity[1], 0.01);
        assert_approx_eq!(ukf.mean_state[5], velocity[2], 0.01);
    }
    
    #[test]
    fn test_particle_construction() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![0.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let other_states = vec![1.0, 2.0, 3.0];
        let weight = 1.0;

        let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros());
        let particle = Particle::new(nav_state.clone(), imu_biases.clone(), imu_biases.clone(), other_states.clone(), weight);
        assert_eq!(particle.nav_state.position.len(), position.len());
    }
    
    #[test]
    fn test_particle_filter_construction() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![1.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let other_states = vec![1.0, 2.0, 3.0];
        let weight = 1.0;

        let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros());
        let particle = Particle::new(nav_state.clone(), imu_biases.clone(), imu_biases.clone(), other_states.clone(), weight);
        let particles = vec![particle; 10];

        let pf = ParticleFilter::new(particles);
        assert_eq!(pf.particles.len(), 10);
    }
    
}