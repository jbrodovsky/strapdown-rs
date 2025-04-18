//! Inertial Navigation Filters
//! 
//! This module contains implementations of various inertial navigation filters, including
//! Kalman filters and particle filters. These filters are used to estimate the state of a
//! strapdown inertial navigation system based on IMU measurements and other sensor data.
//! The filters use the strapdown equations (provided by the StrapdownState) to propagate 
//! the state in the local level frame. The filters also use a generic position measurement
//! model to update the state based on position measurements in the local level frame.

use rand;
use nalgebra::{SMatrix, DMatrix, DVector};
use crate::{IMUData, StrapdownState};

/// Helper struct for UKF implementation
/// 
/// This struct is used to represent a sigma point in the UKF. It contains the strapdown state
/// and other states. The strapdown state is used to propagate the state using the strapdown
/// navigation equations
pub struct SigmaPoint {
    nav_state: StrapdownState,
    other_states: Vec<f64>   // I'm keeping this as generic as possible. 
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
        state.extend(self.other_states.as_slice());
        DVector::from_vec(state)
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
    // UKF, (i.e. the strapdown equations), we don't necessarily know what the
    // measurement model is on this generic level.
    // ---------------------------------------------------------------------------

    /// Perform the Kalman measurement update
    /// 
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
        // check that the kalman gain and measurement diff are compatible to multiply
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
    //  Because of my choice to separately define the strapdown propagation, I'll need to rework the 
    //  the UKF propagation step to resemble something more like a limited particle filter rather than
    //  the more traditional purely linear algebra vector-matrix operations of a traditional KF/EKF/UKF.

    /// Calculate the sigma points for the UKF.
    /// 
    /// The sigma points are calculated based on the current state and covariance.
    /// # Arguments
    /// * `mean` - The mean state to use for the sigma points.
    /// * `covariance` - The covariance to use for the sigma points.
    /// # Returns
    /// * A vector of sigma points.
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
    /// Propagate the particle using the strapdown equations
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64) {
        self.nav_state.forward(imu_data, dt);
        // Optionally, you could add bias random walk here if desired
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
            particle.propagate(imu_data, dt);
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

/// Strapdown Extended Kalman Filter Inertial Navigation Filter
/// 
/// This filter uses the Extended Kalman Filter (EKF) algorithm to estimate the state of a 
/// strapdown inertial navigation system. It uses the strapdown equations to propagate the state
/// in the local level frame based on IMU measurements in the body frame. The filter also uses
/// a generic position measurement model to update the state based on position measurements in 
/// the local level frame.
/// 
/// Because of the generic nature of this toolbox, the filter requires the user to
/// implement the measurement model and the measurement Jacobian matrix. These functions must be
/// provided as inputs to the update method.
pub struct EKF {
    nav_state: StrapdownState,
    state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    measurement_noise: DMatrix<f64>,
    state_size: usize,
}

impl EKF {
    /// Creates a new EKF with the given initial state, biases, covariance, process noise,
    /// and measurement noise.
    /// 
    /// Create a new instance of the EKF with the given initial state, biases, covariance,
    /// process noise, and measurement noise. Additional states that may be used for the filter
    /// are included in the "other states" vector.
    /// # Arguments
    /// * `position` - The initial position of the EKF (degrees latitude, degrees longitude, meters).
    /// * `velocity` - The initial velocity of the EKF (meters per second).
    /// * `attitude` - The initial attitude of the EKF (Eulers angles, roll, pitch, yaw).
    /// * `imu_biases` - The initial IMU biases of the EKF (accelerometer and gyroscope biases).
    /// * `measurement_bias` - The initial measurement biases of the EKF (position measurement biases).
    /// * `other_states` - The initial other states of the EKF (e.g. magnetometer biases, other sensor biases, etc).
    /// * `covariance_diagonal` - The initial covariance diagonal of the EKF (the diagonal elements of the covariance matrix).
    /// * `process_noise_diagonal` - The process noise diagonal of the EKF (the diagonal elements of the process noise matrix).
    /// * `measurement_noise_diagonal` - The measurement noise diagonal of the EKF (the diagonal elements of the measurement noise matrix).
    /// # Returns
    /// * A new EKF struct.
    /// # Example
    /// ```rust
    /// use strapdown::filter::EKF;
    /// use strapdown::StrapdownState;
    /// use nalgebra::SVector;
    /// 
    /// let position = vec![0.0, 0.0, 0.0];
    /// let velocity = vec![0.0, 0.0, 0.0];
    /// let attitude = vec![1.0, 0.0, 0.0];
    /// let imu_biases = vec![0.0, 0.0, 0.0];
    /// let measurement_bias = vec![1.0, 1.0, 1.0];
    /// let covariance_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
    /// let process_noise_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
    /// let measurement_noise_diagonal = vec![1e-3; 3];
    /// let ekf = EKF::new(
    ///    position.clone(),
    ///    velocity.clone(),
    ///    attitude.clone(),
    ///    imu_biases.clone(),
    ///    measurement_bias.clone(),
    ///    covariance_diagonal,
    ///    process_noise_diagonal,
    ///    measurement_noise_diagonal,
    /// );
    /// assert_eq!(ekf.state.len(), position.len() + velocity.len() + attitude.len() + imu_biases.len() + measurement_bias.len());
    /// ```
    pub fn new(
        position: Vec<f64>,
        velocity: Vec<f64>,
        attitude: Vec<f64>,
        imu_biases: Vec<f64>,
        measurement_bias: Vec<f64>,
        other_states: Vec<f64>,
        covariance_diagonal: Vec<f64>,
        process_noise_diagonal: Vec<f64>,
        measurement_noise_diagonal: Vec<f64>,
    ) -> EKF {
        let mut mean: Vec<f64> = position.clone();
        mean.extend(velocity.clone());
        mean.extend(attitude.clone());
        mean.extend(imu_biases.clone());
        mean.extend(measurement_bias.clone());
        mean.extend(other_states.clone());
        
        assert!(
            mean.len() == covariance_diagonal.len(),
            "Mean state and covariance diagonal must be of the same size"
        );
        
        let state_size = mean.len();
        let state = DVector::from_vec(mean);
        
        // Create the navigation state from position, velocity, and attitude
        let mut nav_state_vec = Vec::new();
        nav_state_vec.extend(position);
        nav_state_vec.extend(velocity);
        nav_state_vec.extend(attitude);
        
        let nav_state = StrapdownState::new_from_vector(
            SMatrix::from_iterator(nav_state_vec.iter().cloned())
        );
        
        let covariance = DMatrix::from_diagonal(&DVector::from_vec(covariance_diagonal));
        let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));
        let measurement_noise = DMatrix::from_diagonal(&DVector::from_vec(measurement_noise_diagonal));
        
        EKF {
            nav_state,
            state,
            covariance,
            process_noise,
            measurement_noise,
            state_size,
        }
    }    
    /// Update the internal state vector from the nav_state and other_states
    fn update_state_vector(&mut self) {
        let mut mean = self.nav_state.to_vector(false).as_slice().to_vec();
        let other_states = self.state.rows(9, self.state.len() - 9).as_slice().to_vec();
        mean.extend(other_states);
        self.state = DVector::from_vec(mean);
    }    
    /// Update the nav_state and other_states from the state vector
    fn update_from_state_vector(&mut self) {
        // First 9 elements go to nav_state (position, velocity, attitude)
        let nav_state_vec = SMatrix::from_iterator(
            self.state.rows(0, 9).iter().cloned()
        );
        self.nav_state = StrapdownState::new_from_vector(nav_state_vec);
        
        // Remaining elements go to other_states
        // self.other_states = self.state.rows(9, self.state_size - 9).as_slice().to_vec();
    }    
    /// Propagates the state using the strapdown equations and IMU measurements.
    /// 
    /// The IMU measurements are used to update the strapdown state in the local level frame.
    /// The IMU measurements are assumed to be in the body frame.
    /// 
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the state with.
    /// * `dt` - The time step for the propagation.
    /// * `state_transition_jacobian` - The Jacobian of the state transition function.
    /// # Returns
    /// * none
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64, state_transition_jacobian: &DMatrix<f64>) {
        // Propagate the strapdown state using the strapdown equations
        self.nav_state.forward(imu_data, dt);        
        // Update the state vector
        self.update_state_vector();        
        // Propagate the covariance
        // P_k = F_k * P_{k-1} * F_k^T + Q_k
        self.covariance = state_transition_jacobian * &self.covariance * state_transition_jacobian.transpose() + &self.process_noise;
    }    
    /// Perform the Kalman measurement update
    /// 
    /// This method updates the state and covariance based on the measurement, expected measurement,
    /// and measurement Jacobian.
    /// 
    /// # Arguments
    /// * `measurement` - The actual measurement vector.
    /// * `expected_measurement` - The expected measurement based on the current state.
    /// * `measurement_jacobian` - The Jacobian of the measurement model with respect to the state.
    /// # Returns
    /// * none
    pub fn update(&mut self, measurement: &DVector<f64>, expected_measurement: &DVector<f64>, measurement_jacobian: &DMatrix<f64>) {
        // Assert that the measurement is the correct size as the measurement noise diagonal
        assert!(
            measurement.len() == self.measurement_noise.nrows(),
            "Measurement and measurement noise must be of the same size"
        );        
        // Calculate innovation vector
        let innovation = measurement - expected_measurement;
        // Calculate innovation covariance
        // S = H * P * H^T + R
        let innovation_covariance = measurement_jacobian * &self.covariance * measurement_jacobian.transpose() + &self.measurement_noise;        
        // Calculate Kalman gain
        // K = P * H^T * S^-1
        let innovation_inverse = match innovation_covariance.clone().try_inverse() {
            Some(inv) => inv,
            None => panic!("Innovation covariance matrix is singular"),
        };
        let kalman_gain = &self.covariance * measurement_jacobian.transpose() * innovation_inverse;
        // Update state
        // x_k = x_k + K * (z - h(x_k))
        self.state += &kalman_gain * innovation;        
        // Update covariance
        // P_k = (I - K * H) * P_k
        let identity = DMatrix::identity(self.state_size, self.state_size);
        self.covariance = (&identity - &kalman_gain * measurement_jacobian) * &self.covariance;
        // Update nav_state and other_states from the state vector
        self.update_from_state_vector();
    }
    
    /// Get the EKF state.
    /// The state is the current navigation state vector with the additional imu and measurement biases appended along with any remaining states.
    pub fn get_state(&self) -> DVector<f64> {
        return self.state.clone();
    }
    /// Get the navigation state component of the EKF state.
    pub fn get_nav_state(&self) -> StrapdownState {
        return self.nav_state.clone();
    }    
    /// Get the covariance matrix.
    pub fn get_covariance(&self) -> DMatrix<f64> {
        return self.covariance.clone();
    }
}

/// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StrapdownState;
    use nalgebra::{SMatrix, DVector};
    use std::f64::consts::PI;

    #[test]
    fn test_ukf_construction() {
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
        assert_eq!(ukf.mean_state.len(), position.len() + velocity.len() + attitude.len() + imu_biases.len() + measurement_bias.len());
    }

    #[test]
    fn test_ekf_construction() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![1.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let other_states = vec![0.0, 0.0, 0.0];
        let covariance_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
        let process_noise_diagonal = vec![1e-3; 9 + imu_biases.len() + measurement_bias.len()];
        let measurement_noise_diagonal = vec![1e-3; 3];

        let ekf = EKF::new(
            position.clone(),
            velocity.clone(),
            attitude.clone(),
            imu_biases.clone(),
            measurement_bias.clone(),
            other_states.clone(),
            covariance_diagonal,
            process_noise_diagonal,
            measurement_noise_diagonal,
        );
        assert_eq!(ekf.state.len(), position.len() + velocity.len() + attitude.len() + imu_biases.len() + measurement_bias.len());
    }
}