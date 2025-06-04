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
use crate::linalg::matrix_square_root;
use crate::{IMUData, StrapdownState};
use nalgebra::{DMatrix, DVector, SVector, Vector3};
use rand;
use rand_distr::{Distribution, Normal};
use std::fmt::Debug;

/// Helper struct for UKF and Particle Filter implementations
///
/// This struct is used to represent a sigma point in the UKF or a particle in the particle filter.
/// Functionally, these two data structures are the same, but they are used in different contexts.
/// It contains the strapdown state and other states. The strapdown state is used to propagate the
/// state using the strapdown navigation equations. `other_states` is used to represent any other
/// states that are being estimated by the filter. In the canonical 15-state INS filter, these
/// other states would be the six IMU biases, however, this struct is generic and can be used for
/// any other states that are being estimated by the filter and can be stored as a floating point
/// value.
///
/// Note that the `weight` field is primarily used in the particle filter, but is included here
/// for brevity as a particle is pretty much just a sigma point with only one weighting scheme.
/// Note that the UKF uses two different weighting schemes, one for the mean and one for the covariance.
#[derive(Clone, Debug)]
pub struct SigmaPoint {
    /// State dimension
    n: usize,
    /// The basic local level frame strapdown state
    nav_state: StrapdownState,
    /// The other states to be estimated by the UKF
    other_states: Vec<f64>,
    /// The weight of the sigma point, primarily used in the Particle Filter
    weight: f64,
}
impl SigmaPoint {
    /// Create new sigma point from a strapdown state vector plus whatever other states the UKF
    /// estimates.
    ///
    /// # Arguments
    ///
    /// * `nav_state` - The strapdown state to use for the sigma point.
    /// * `other_states` - The other states to use for the sigma point.
    /// * `weight` - The weight of the sigma point.
    ///
    /// # Returns
    /// * A new SigmaPoint struct.
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use strapdown::StrapdownState;
    /// use nalgebra::SVector;
    ///
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let sigma_point = SigmaPoint::new(nav_state, other_states, Some(1.0));
    /// ```
    pub fn new(
        nav_state: StrapdownState,
        other_states: Vec<f64>,
        weight: Option<f64>,
    ) -> SigmaPoint {
        let n: usize = nav_state.to_vector(false).len() + other_states.len();
        SigmaPoint {
            n,
            nav_state,
            other_states,
            weight: weight.unwrap_or(1.0),
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
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let sigma_point = SigmaPoint::new(nav_state, other_states, None);
    ///
    /// let state = sigma_point.get_state(false);
    /// assert_eq!(state.len(), 12);
    /// ```
    pub fn get_state(&self, in_degrees: bool) -> DVector<f64> {
        let mut state = self.nav_state.to_vector(in_degrees).as_slice().to_vec();
        state.extend(self.other_states.iter());
        DVector::from_vec(state)
    }
    /// Get the weight of the sigma point.
    ///
    /// # Returns
    /// * The weight of the sigma point.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use strapdown::StrapdownState;
    /// use nalgebra::SVector;
    ///
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let sigma_point = SigmaPoint::new(nav_state, other_states, Some(1.0));
    ///
    /// let weight = sigma_point.get_weight();
    /// assert_eq!(weight, 1.0);
    pub fn get_weight(&self) -> f64 {
        self.weight
    }

    /// Forward mechanization (propagation) of the strapdown state. Serves
    /// as a thin wrapper around the strapdown state `forward` function.
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
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let mut sigma_point = SigmaPoint::new(nav_state, other_states, None);
    ///
    /// let imu_data = IMUData::new_from_vec(vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
    /// let dt = 0.1;
    ///
    /// sigma_point.forward(&imu_data, dt, None);
    /// ```
    pub fn forward(&mut self, imu_data: &IMUData, dt: f64, noise: Option<Vec<f64>>) {
        // Propagate the strapdown state using the strapdown equations
        self.nav_state.forward(imu_data, dt);
        let noise_vect = match noise {
            None => return, // UKF mode, does not use noise in propagation
            Some(v) => v,   // Particle filter mode, uses noise in propagation
        };
        let mut rng = rand::rng();
        let mut jitter = Vec::with_capacity(self.n);
        for stddev in noise_vect.iter() {
            let normal = Normal::new(0.0, *stddev).unwrap();
            jitter.push(normal.sample(&mut rng));
        }
        // Convert jitter to DVector for addition
        let jitter_vec = DVector::from_vec(jitter);

        // Get current state and add jitter
        let state = self.get_state(false);
        let perturbed_state = state + jitter_vec;

        // Update the state with perturbed values
        // First 9 elements go to nav_state
        let nav_part = perturbed_state.rows(0, 9);
        self.nav_state = StrapdownState::new_from_vector(
            SVector::from_iterator(nav_part.iter().cloned()),
            false,
        );
        // Remaining elements go to other_states
        let other_len = perturbed_state.len() - 9;
        if other_len > 0 {
            self.other_states = perturbed_state.rows(9, other_len).iter().cloned().collect();
        }
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
    /// let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
    /// let other_states = vec![0.0, 0.0, 0.0];
    /// let sigma_point = SigmaPoint::new(nav_state, other_states, None);
    ///
    /// let state = sigma_point.to_vector(true);
    /// assert_eq!(state.len(), 12);
    /// ```
    pub fn to_vector(&self, in_degrees: bool) -> DVector<f64> {
        let mut state = self.nav_state.to_vector(in_degrees).as_slice().to_vec();
        state.extend(self.other_states.as_slice());
        DVector::from_vec(state)
    }
    /// Convert an nalgebra vector to a sigma point.
    ///
    /// # Arguments
    /// * `state` - The vector to convert to a sigma point.
    /// * `weight` - The weight of the sigma point, if any.
    /// # `in_degrees` - Whether the input vector is in degrees or radians.
    ///
    /// # Returns
    /// * A new SigmaPoint struct.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::filter::SigmaPoint;
    /// use nalgebra::DVector;
    ///
    /// let state = DVector::from_vec(vec![0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]);
    /// let sigma_point = SigmaPoint::from_vector(state, Some(1.0), false);
    /// ```
    pub fn from_vector(state: DVector<f64>, weight: Option<f64>, in_degrees: bool) -> SigmaPoint {
        assert!(
            state.len() >= 9,
            "Expected a cannonical state vector of at least 9 states."
        );
        let position: Vector3<f64> = Vector3::new(state[0], state[1], state[2]);
        let velocity: Vector3<f64> = Vector3::new(state[3], state[4], state[5]);
        let attitude: Vector3<f64> = Vector3::new(state[6], state[7], state[8]);
        let other_states = state
            .rows(9, state.len() - 9)
            .iter()
            .cloned()
            .collect::<Vec<f64>>();
        let nav_state: StrapdownState =
            StrapdownState::new_from(position, velocity, attitude, in_degrees);
        SigmaPoint::new(nav_state, other_states, weight)
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
    /// * `measurement_noise_diagonal` - The measurement noise diagonal.
    /// * `alpha` - The alpha parameter for the UKF.
    /// * `beta` - The beta parameter for the UKF.
    /// * `kappa` - The kappa parameter for the UKF.
    /// * `in_degrees` - Whether the input vectors are in degrees or radians.
    ///
    /// # Returns
    /// * A new UKF struct.
    pub fn new(
        position: Vec<f64>,
        velocity: Vec<f64>,
        attitude: Vec<f64>,
        imu_biases: Vec<f64>,
        other_states: Vec<f64>,
        covariance_diagonal: Vec<f64>,
        process_noise: DMatrix<f64>,
        measurement_noise: DMatrix<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
        in_degrees: bool,
    ) -> UKF {
        assert!(
            measurement_noise.nrows() == measurement_noise.ncols(),
            "Measurement noise matrix must be square"
        );
        assert!(
            process_noise.nrows() == process_noise.ncols(),
            "Process noise matrix must be square"
        );

        // If in_degrees, convert the position and attitude to radians
        let position = if in_degrees {
            vec![
                position[0].to_radians(),
                position[1].to_radians(),
                position[2],
            ]
        } else {
            position
        };
        let attitude = if in_degrees {
            vec![
                attitude[0].to_radians(),
                attitude[1].to_radians(),
                attitude[2].to_radians(),
            ]
        } else {
            attitude
        };

        let mut mean: Vec<f64> = position;
        mean.extend(velocity);
        mean.extend(attitude);
        mean.extend(imu_biases);
        mean.extend(other_states);
        assert!(
            mean.len() >= 9,
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
    ///
    /// # Arguments
    /// * `imu_data` - The IMU measurements to propagate the state with (e.g. relative accelerations (m/s^2) and angular rates (rad/s)).
    /// * `dt` - The time step for the propagation.
    ///
    /// # Returns
    /// * none
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64) {
        // Propagate the strapdown state using the strapdown equations
        let mut sigma_points = self.get_sigma_points();
        for sigma_point in &mut sigma_points {
            sigma_point.forward(imu_data, dt, None);
        }
        let mut mu_bar = DVector::<f64>::zeros(self.state_size);
        // Update the mean state through a naive loop
        for (i, sigma_point) in sigma_points.iter().enumerate() {
            mu_bar += self.weights_mean[i] * sigma_point.get_state(false);
        }
        let mut cov_bar = DMatrix::<f64>::zeros(self.state_size, self.state_size);
        // Update the covariance through a naive loop
        for (i, sigma_point) in sigma_points.iter().enumerate() {
            //let sigma_point = &sigma_points[i];
            let weight_cov = self.weights_cov[i];
            let diff = sigma_point.get_state(false) - &mu_bar;
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
    /// the measurement vector and the measurement sigma points. This module contains some
    /// standard GNSS-aided measurements models (`position_measurement_model`, `velocity_measurement_model`,
    /// and `position_and_velocity_measurement_model`) that can be used. See the `sim` module for
    /// a canonical example of a GPS-aided INS implementation that uses these models.
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
    pub fn update(
        &mut self,
        measurement: &DVector<f64>,
        measurement_sigma_points: &[DVector<f64>],
    ) {
        // Assert that the measurement is the correct size as the measurement noise diagonal
        assert!(
            measurement.len() == self.measurement_noise.nrows(),
            "Measurement and measurement noise must be of the same size"
        );
        // Calculate expected measurement
        let mut z_hat = DVector::<f64>::zeros(measurement.len());
        for (i, sigma_point) in measurement_sigma_points.iter().enumerate() {
            z_hat += self.weights_mean[i] * sigma_point;
        }
        // Calculate innovation matrix S
        let mut s = DMatrix::<f64>::zeros(measurement.len(), measurement.len());
        //for i in 0..measurement_sigma_points.len() {
        for (i, sigma_point) in measurement_sigma_points.iter().enumerate() {
            let diff = sigma_point - &z_hat;
            s += self.weights_cov[i] * &diff * &diff.transpose();
        }
        s += &self.measurement_noise;
        // print out the data in s
        // println!("Innovation: ");
        // let (nrows, ncols) = s.shape();
        // for r in 0..nrows{
        //     print!("[ ");
        //     for c in 0..ncols {
        //         print!("{:.8} ", s[(r,c)]);
        //     }
        //     println!(" ]");
        // }
        // Calculate the cross-covariance
        let sigma_points = self.get_sigma_points();
        let mut cross_covariance = DMatrix::<f64>::zeros(self.state_size, measurement.len());
        for (i, sigma_point) in measurement_sigma_points.iter().enumerate() {
            let measurement_diff = sigma_point - &z_hat;
            let state_diff = &sigma_points[i].to_vector(false) - &self.mean_state;
            cross_covariance += self.weights_cov[i] * state_diff * measurement_diff.transpose();
        }
        // println!("Cross-covariance: ");
        // let (nrows, ncols) = cross_covariance.shape();
        // for r in 0..nrows{
        //     print!("[ ");
        //     for c in 0..ncols {
        //         print!("{:.8} ", cross_covariance[(r,c)]);
        //     }
        //     println!(" ]");
        // }
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
        // println!("Kalman Gain: ");
        // let (nrows, ncols) = k.shape();
        // for r in 0..nrows{
        //     print!("[ ");
        //     for c in 0..ncols {
        //         print!("{:.8} ", k[(r,c)]);
        //     }
        //     println!(" ]");
        // }
        // Perform Kalman update
        //println!("Expected measurement: {:.4}, {:.4}, {:.4}", z_hat[0], z_hat[1], z_hat[2]);
        //let z_diff = measurement - &z_hat;
        //println!("Difference between measurement and expected measurement: \n [{:.4}, {:.4}, {:.4}]", z_diff[0], z_diff[1], z_diff[2]);
        //let mean_update = &k * (z_diff);
        //println!("Mean update: \n [{:.4}, {:.4}, {:.4}]", mean_update[0], mean_update[1], mean_update[2]);
        self.mean_state += &k * (measurement - &z_hat);
        self.covariance -= &k * s * &k.transpose();
    }
    /// Get the UKF mean state.
    pub fn get_mean(&self) -> DVector<f64> {
        self.mean_state.clone()
    }
    /// Get the UKF covariance.
    pub fn get_covariance(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }
    /// Calculate the sigma points for the UKF based on the current state and covariance.
    pub fn get_sigma_points(&self) -> Vec<SigmaPoint> {
        //dbg!("UKF::get_sigma_points -> state_size: {}", self.state_size);
        let scaled_covariance = (self.state_size as f64 + self.lambda) * self.covariance.clone();
        let sqrt_cov = matrix_square_root(&scaled_covariance);
        let mut sigma_points = Vec::<SigmaPoint>::with_capacity(2 * self.state_size + 1);
        // Add the mean state as the first sigma point, note that the UKF uses two sets of weights
        let mean = self.mean_state.clone();
        sigma_points.push(SigmaPoint::from_vector(mean.clone(), Some(0.0), false));

        let mut left: Vec<SigmaPoint> = Vec::with_capacity(self.state_size);
        let mut right: Vec<SigmaPoint> = Vec::with_capacity(self.state_size);
        for i in 0..self.state_size {
            let sqrt_cov_i = sqrt_cov.column(i);
            left.push(SigmaPoint::from_vector(
                &mean + sqrt_cov_i,
                Some(0.0),
                false,
            ));
            right.push(SigmaPoint::from_vector(
                &mean - sqrt_cov_i,
                Some(0.0),
                false,
            ));
        }
        sigma_points.extend(left);
        sigma_points.extend(right);
        sigma_points
    }
}
/// Particle filter for strapdown inertial navigation
pub struct ParticleFilter {
    pub particles: Vec<SigmaPoint>,
}
impl ParticleFilter {
    /// Create a new particle filter with the given particles
    pub fn new(particles: Vec<SigmaPoint>) -> Self {
        ParticleFilter { particles }
    }
    /// Propagate all particles forward using the strapdown equations
    pub fn propagate(&mut self, imu_data: &IMUData, dt: f64) {
        for particle in &mut self.particles {
            particle.forward(imu_data, dt, None);
        }
    }
    /// Update the weights of the particles based on a measurement
    pub fn update(&mut self, measurement: &DVector<f64>) {
        
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
// ==== Measurement Models ==========
// Below is a set of generic measurement models for the UKF and particle filter.
// These models provide a vec of "expected measurements" based on the sigma points
// location. When used in a filter, you should simulatanously iterate through the
// list of sigma points or particles and the expected measurements in order to
// calculate the innovation matrix or the particle weighting.
// ===============================================================================================

/// GPS or Position-based measurement model
///
/// Standard GNSS-aided loosely-coupled INS measurement model. This model is used to update the state
/// based on position measurements in the local level frame. The model assumes that the position
/// measurements are in the local level frame and that the strapdown state is in the body frame.
///
/// # Arguments
/// * `sigma_points` - the strapdown UKF's sigma points
/// * `with_altitude` - whether to include altitude in the measurement
///
/// # Returns
/// * A vector of measurement sigma points either (N x 3) or (N x 2) depending on the `with_altitude` flag
pub fn position_measurement_model(
    sigma_points: &Vec<SigmaPoint>,
    with_altitude: bool,
) -> Vec<DVector<f64>> {
    let mut measurement_sigma_points = Vec::<DVector<f64>>::with_capacity(sigma_points.len());
    for sigma_point in sigma_points {
        let state = sigma_point.get_state(false);
        let mut measurement_sigma_point = DVector::<f64>::zeros(if with_altitude { 3 } else { 2 });
        if with_altitude {
            measurement_sigma_point[0] = state[0]; // Latitude
            measurement_sigma_point[1] = state[1]; // Longitude
            measurement_sigma_point[2] = state[2]; // Altitude
        } else {
            measurement_sigma_point[0] = state[0]; // Latitude
            measurement_sigma_point[1] = state[1]; // Longitude
        }
        measurement_sigma_points.push(measurement_sigma_point);
    }
    measurement_sigma_points
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
/// * `with_altitude` - whether to include altitude in the measurement
///
/// # Returns
/// * A vector of measurement sigma points either (N x 3) or (N x 2) depending on the `with_altitude` flag
pub fn velocity_measurement_model(
    sigma_points: &Vec<SigmaPoint>,
    with_altitude: bool,
) -> Vec<DVector<f64>> {
    let mut measurement_sigma_points = Vec::<DVector<f64>>::with_capacity(sigma_points.len());
    for sigma_point in sigma_points {
        let state = sigma_point.get_state(false);
        let mut measurement_sigma_point = DVector::<f64>::zeros(if with_altitude { 3 } else { 2 });
        if with_altitude {
            measurement_sigma_point[0] = state[3]; // Northward
            measurement_sigma_point[1] = state[4]; // Eastward
            measurement_sigma_point[2] = state[5]; // downward
        } else {
            measurement_sigma_point[0] = state[3]; // Northward
            measurement_sigma_point[1] = state[4]; // Eastward
        }
        measurement_sigma_points.push(measurement_sigma_point);
    }
    measurement_sigma_points
}
/// GPS or Position-based measurement model
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
/// * `with_altitude` - whether to include altitude in the measurement
///
/// # Returns
/// * A vector of measurement sigma points either (N x 6) or (N x 4) depending on the `with_altitude` flag
pub fn position_and_velocity_measurement_model(
    sigma_points: &Vec<SigmaPoint>,
    with_altitude: bool,
) -> Vec<DVector<f64>> {
    let mut measurement_sigma_points = Vec::<DVector<f64>>::with_capacity(sigma_points.len());
    for sigma_point in sigma_points {
        let state = sigma_point.get_state(false);
        if with_altitude {
            // Position and velocity with altitude
            let measurement_sigma_point = DVector::<f64>::from_vec(vec![
                state[0], // Latitude
                state[1], // Longitude
                state[2], // Altitude
                state[3], // Northward velocity
                state[4], // Eastward velocity
                state[5], // Downward velocity
            ]);
            measurement_sigma_points.push(measurement_sigma_point);
        } else {
            // Position and velocity without altitude
            let measurement_sigma_point = DVector::<f64>::from_vec(vec![
                state[0], // Latitude
                state[1], // Longitude
                state[3], // Northward velocity
                state[4], // Eastward velocity
            ]);
            measurement_sigma_points.push(measurement_sigma_point);
        }
    }
    measurement_sigma_points
}

/// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::{SVector, Vector3};
    // Test sigma point functionality
    #[test]
    fn test_sigma_point() {
        let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
        let other_states = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut sigma_point = SigmaPoint::new(nav_state, other_states.clone(), Some(1.0));
        assert_eq!(sigma_point.get_state(false).len(), 15);
        assert_eq!(sigma_point.get_weight(), 1.0);

        let state = sigma_point.get_state(false);
        let state_vector = sigma_point.to_vector(false);
        assert_eq!(state.len(), state_vector.len());
        let sigma2 = SigmaPoint::from_vector(state_vector, None, false);
        assert_eq!(sigma_point.get_state(false), sigma2.get_state(false));
        // test forward propagation
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0), // Currently configured as relative body-frame acceleration
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        let dt = 1.0;
        sigma_point.forward(&imu_data, dt, None);

        let position = [0.0, 0.0, 0.0];
        let velocity = [0.0, 0.0, 0.0];

        assert_approx_eq!(sigma_point.nav_state.position[0], position[0], 1e-3);
        assert_approx_eq!(sigma_point.nav_state.position[1], position[1], 1e-3);
        assert_approx_eq!(sigma_point.nav_state.position[2], position[2], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[0], velocity[0], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[1], velocity[1], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[2], velocity[2], 0.1);

        // Test forward propagation with noise
        let noise = vec![
            0.01, 0.01, 0.01, // Position noise
            0.01, 0.01, 0.01, // Velocity noise
            0.01, 0.01, 0.01, // Attitude noise
            0.01, 0.01, 0.01, // IMU biases noise
            0.01, 0.01, 0.01, // Measurement bias noise
        ];
        sigma_point.forward(&imu_data, dt, Some(noise));
        // Check that the state has changed
        assert!(sigma_point.nav_state.position[0] != position[0]);
        assert!(sigma_point.nav_state.position[1] != position[1]);
        assert!(sigma_point.nav_state.position[2] != position[2]);
        assert!(sigma_point.nav_state.velocity[0] != velocity[0]);
        assert!(sigma_point.nav_state.velocity[1] != velocity[1]);
        assert!(sigma_point.nav_state.velocity[2] != velocity[2]);
        // Check that the state is still in the same ballpark
        assert_approx_eq!(sigma_point.nav_state.position[0], position[0], 0.1);
        assert_approx_eq!(sigma_point.nav_state.position[1], position[1], 0.1);
        assert_approx_eq!(sigma_point.nav_state.position[2], position[2], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[0], velocity[0], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[1], velocity[1], 0.1);
        assert_approx_eq!(sigma_point.nav_state.velocity[2], velocity[2], 0.1);

        let debug_str = format!("{:?}", sigma_point,);
        assert!(debug_str.contains("nav_state"));
    }
    #[test]
    fn test_ukf_construction() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![0.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let n = 9 + imu_biases.len() + measurement_bias.len();
        let covariance_diagonal = vec![1e-3; n];
        let process_noise_diagonal = vec![1e-3; n];
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
            DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal)),
            DMatrix::from_diagonal(&DVector::from_vec(measurement_noise_diagonal)),
            alpha,
            beta,
            kappa,
            false,
        );
        assert_eq!(
            ukf.mean_state.len(),
            position.len()
                + velocity.len()
                + attitude.len()
                + imu_biases.len()
                + measurement_bias.len()
        );
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
        let n = 9 + imu_biases.len() + measurement_bias.len();
        let process_noise_diagonal = DVector::from_vec(vec![1e-3; n]);
        let measurement_noise_diagonal = DVector::from_vec(vec![1e-3; 3]);
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
            DMatrix::from_diagonal(&process_noise_diagonal),
            DMatrix::from_diagonal(&measurement_noise_diagonal),
            alpha,
            beta,
            kappa,
            false,
        );
    }
    #[test]
    fn test_ukf_get_sigma_points() {
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![1.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let n = 9 + imu_biases.len() + measurement_bias.len();
        let covariance_diagonal = vec![1e-3; n];
        let process_noise_diagonal = DVector::from_vec(vec![1e-3; n]);
        let measurement_noise_diagonal = DVector::from_vec(vec![1e-3; 3]);
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
            DMatrix::from_diagonal(&process_noise_diagonal),
            DMatrix::from_diagonal(&measurement_noise_diagonal),
            alpha,
            beta,
            kappa,
            false,
        );
        let sigma_points = ukf.get_sigma_points();
        assert_eq!(sigma_points.len(), (2 * ukf.state_size) + 1);
    }
    #[test]
    fn test_ukf_propagate() {
        let imu_data = IMUData::new_from_vec(vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let position = vec![0.0, 0.0, 0.0];
        let velocity = vec![0.0, 0.0, 0.0];
        let attitude = vec![1.0, 0.0, 0.0];
        let imu_biases = vec![0.0, 0.0, 0.0];
        let measurement_bias = vec![1.0, 1.0, 1.0];
        let n = 9 + imu_biases.len() + measurement_bias.len();
        let covariance_diagonal = vec![1e-3; n];
        let process_noise_diagonal = DVector::from_vec(vec![1e-3; n]);
        let measurement_noise_diagonal = DVector::from_vec(vec![1e-3; 3]);
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
            DMatrix::from_diagonal(&process_noise_diagonal),
            DMatrix::from_diagonal(&measurement_noise_diagonal),
            alpha,
            beta,
            kappa,
            false,
        );
        let dt = 1.0;
        ukf.propagate(&imu_data, dt);
        assert!(
            ukf.mean_state.len()
                == position.len()
                    + velocity.len()
                    + attitude.len()
                    + imu_biases.len()
                    + measurement_bias.len()
        );
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
            DMatrix::from_diagonal(&DVector::from_vec(vec![1.0e-3; 15])),
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-3; 3])),
            1e-3,
            2.0,
            1e-3,
            false,
        );
        let debug_str = format!("{:?}", ukf);
        assert!(debug_str.contains("mean_state"));
    }
    #[test]
    fn test_ukf_hover() {
        let imu_data = IMUData::new_from_vec(vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
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
            DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal)),
            DMatrix::from_diagonal(&DVector::from_vec(measurement_noise_diagonal)),
            alpha,
            beta,
            kappa,
            false,
        );
        let dt = 1.0;
        let measurement_sigmas = position_measurement_model(&ukf.get_sigma_points(), true);
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
        let position = [0.0, 0.0, 0.0];
        let other_states = vec![1.0, 2.0, 3.0];
        let weight = 1.0;
        let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
        let particle = SigmaPoint::new(nav_state, other_states.clone(), Some(weight));
        assert_eq!(particle.nav_state.position.len(), position.len());
    }
    #[test]
    fn test_particle_filter_construction() {
        let other_states = vec![1.0, 2.0, 3.0];
        let weight = 1.0;

        let nav_state = StrapdownState::new_from_vector(SVector::<f64, 9>::zeros(), false);
        let particle = SigmaPoint::new(nav_state, other_states.clone(), Some(weight));
        let particles = vec![particle; 10];

        let pf = ParticleFilter::new(particles);
        assert_eq!(pf.particles.len(), 10);
    }
}
