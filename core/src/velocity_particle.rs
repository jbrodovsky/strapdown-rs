//! Six-state velocity particle filter for GPS-aided navigation
//!
//! This module implements a basic six-state particle filter for navigation applications
//! that includes position and velocity states: [lat, lon, alt, v_n, v_e, v_d].
//! Unlike the full 9-state or 15-state navigation filters, this simplified particle
//! filter tracks only position and velocity, making it computationally efficient while
//! still providing useful navigation estimates when aided by GPS measurements.
//!
//! # State Vector
//!
//! The six-state particle represents:
//! - Position: [latitude (rad), longitude (rad), altitude (m)]
//! - Velocity: [v_north (m/s), v_east (m/s), v_vertical (m/s)]
//!
//! # Usage
//!
//! ```rust
//! use strapdown::velocity_particle::{VelocityParticle, VelocityParticleFilter};
//! use nalgebra::DVector;
//!
//! // Initialize with position and velocity
//! let initial_state = DVector::from_vec(vec![
//!     45.0_f64.to_radians(), -122.0_f64.to_radians(), 100.0,  // lat, lon, alt
//!     10.0, 5.0, 0.0                                           // v_n, v_e, v_d
//! ]);
//!
//! let mut pf = VelocityParticleFilter::new(
//!     initial_state,
//!     1000,  // number of particles
//!     1.0,   // process noise std (meters/second)
//! );
//!
//! // Propagate particles forward in time
//! pf.predict(0.1); // dt = 0.1 seconds
//!
//! // Update with GPS measurement (position only)
//! // pf.update_with_gps(gps_measurement);
//! ```

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::any::Any;

use crate::earth::principal_radii;
use crate::measurements::MeasurementModel;
use crate::particle::{
    Particle, ParticleAveragingStrategy, ParticleFilter, ParticleResamplingStrategy,
};

/// Six-state velocity particle
///
/// Represents position and velocity states for navigation.
/// State vector: [lat (rad), lon (rad), alt (m), v_n (m/s), v_e (m/s), v_d (m/s)]
#[derive(Clone, Debug)]
pub struct VelocityParticle {
    /// State vector [lat, lon, alt, v_n, v_e, v_d]
    state: DVector<f64>,
    /// Particle weight for importance sampling
    weight: f64,
}

impl Particle for VelocityParticle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn new(initial_state: &DVector<f64>, weight: f64) -> Self {
        assert_eq!(
            initial_state.len(),
            6,
            "VelocityParticle requires 6-state vector [lat, lon, alt, v_n, v_e, v_d]"
        );
        VelocityParticle {
            state: initial_state.clone(),
            weight,
        }
    }

    fn state(&self) -> DVector<f64> {
        self.state.clone()
    }

    fn update_weight<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // Compute expected measurement from particle's state
        let expected = measurement.get_expected_measurement(&self.state);
        let actual = measurement.get_measurement(&self.state);

        // Measurement residual
        let innovation = actual - expected;

        // Measurement noise covariance
        let noise_cov = measurement.get_noise();

        // Compute Gaussian likelihood: exp(-0.5 * innovation^T * R^{-1} * innovation)
        // For diagonal covariance, this simplifies to a product of 1D Gaussians
        let mut log_likelihood = 0.0;
        for i in 0..innovation.len() {
            let variance = noise_cov[(i, i)];
            if variance > 0.0 {
                let std_dev = variance.sqrt();
                // Gaussian PDF in log space
                let normalized_innovation = innovation[i] / std_dev;
                log_likelihood += -0.5 * normalized_innovation.powi(2)
                    - std_dev.ln()
                    - 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }

        // Update weight by multiplying with likelihood (in log space: add)
        let likelihood = log_likelihood.exp();
        self.weight *= likelihood;
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}

impl VelocityParticle {
    /// Get the particle's state vector
    pub fn get_state(&self) -> &DVector<f64> {
        &self.state
    }

    /// Set the particle's state vector
    pub fn set_state(&mut self, state: DVector<f64>) {
        assert_eq!(
            state.len(),
            6,
            "State must be 6-element vector [lat, lon, alt, v_n, v_e, v_d]"
        );
        self.state = state;
    }

    /// Propagate particle state forward using constant velocity motion model
    ///
    /// Updates position based on current velocity: position += velocity * dt
    /// Adds process noise to both position and velocity for particle diversity.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step in seconds
    /// * `process_noise_std` - Standard deviation of process noise
    /// * `rng` - Random number generator
    pub fn propagate(&mut self, dt: f64, process_noise_std: f64, rng: &mut StdRng) {
        // Extract current state
        let lat = self.state[0]; // radians
        let lon = self.state[1]; // radians
        let alt = self.state[2]; // meters
        let v_n = self.state[3]; // m/s north
        let v_e = self.state[4]; // m/s east
        let v_d = self.state[5]; // m/s down (vertical)

        // Get principal radii at current position
        let lat_deg = lat.to_degrees();
        let (r_n, r_e, _) = principal_radii(&lat_deg, &alt);

        // Position update using constant velocity model
        // Latitude change (northward displacement)
        let delta_lat = (v_n * dt) / r_n;

        // Longitude change (eastward displacement)
        let delta_lon = if lat.cos().abs() > 1e-8 {
            (v_e * dt) / (r_e * lat.cos())
        } else {
            0.0 // At poles, longitude is undefined
        };

        // Altitude change (vertical displacement, down is positive in NED)
        let delta_alt = -v_d * dt; // Negative because down is positive in NED frame

        // Generate process noise for position and velocity
        // Using Box-Muller transform for Gaussian noise
        let normal = Normal::new(0.0, 1.0).unwrap();

        let noise_lat_m = if process_noise_std > 0.0 {
            normal.sample(rng) * process_noise_std
        } else {
            0.0
        };
        let noise_lon_m = if process_noise_std > 0.0 {
            normal.sample(rng) * process_noise_std
        } else {
            0.0
        };
        let noise_alt = if process_noise_std > 0.0 {
            normal.sample(rng) * process_noise_std
        } else {
            0.0
        };

        // Velocity noise (smaller than position noise, scaled by dt)
        let noise_v_n = if process_noise_std > 0.0 {
            normal.sample(rng) * (process_noise_std / dt.sqrt())
        } else {
            0.0
        };
        let noise_v_e = if process_noise_std > 0.0 {
            normal.sample(rng) * (process_noise_std / dt.sqrt())
        } else {
            0.0
        };
        let noise_v_d = if process_noise_std > 0.0 {
            normal.sample(rng) * (process_noise_std / dt.sqrt())
        } else {
            0.0
        };

        // Convert position noise from meters to radians
        let noise_lat_rad = noise_lat_m / r_n;
        let noise_lon_rad = if lat.cos().abs() > 1e-8 {
            noise_lon_m / (r_e * lat.cos())
        } else {
            0.0
        };

        // Update state with propagation and noise
        self.state[0] = lat + delta_lat + noise_lat_rad;
        self.state[1] = lon + delta_lon + noise_lon_rad;
        self.state[2] = alt + delta_alt + noise_alt;
        self.state[3] = v_n + noise_v_n;
        self.state[4] = v_e + noise_v_e;
        self.state[5] = v_d + noise_v_d;

        // Ensure latitude is within [-pi/2, pi/2]
        self.state[0] =
            self.state[0].clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);

        // Wrap longitude to [-pi, pi]
        while self.state[1] > std::f64::consts::PI {
            self.state[1] -= 2.0 * std::f64::consts::PI;
        }
        while self.state[1] < -std::f64::consts::PI {
            self.state[1] += 2.0 * std::f64::consts::PI;
        }
    }
}

/// Six-state velocity particle filter
///
/// A particle filter that tracks position and velocity using a constant velocity
/// motion model. Designed for GPS-aided navigation applications where GPS provides
/// position measurements to constrain the filter.
///
/// # Features
///
/// - 6-state representation (3 position + 3 velocity)
/// - Constant velocity motion model for prediction
/// - GPS position measurement updates
/// - Configurable process noise for drift rate tuning
/// - Multiple resampling strategies
///
/// # Example
///
/// ```rust
/// use strapdown::velocity_particle::VelocityParticleFilter;
/// use nalgebra::DVector;
///
/// let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let mut pf = VelocityParticleFilter::new(initial_state, 1000, 1.0);
///
/// // Propagate and update cycle
/// pf.predict(0.1);
/// // pf.update_with_gps(gps_measurement);
/// // pf.resample_if_needed();
/// ```
pub struct VelocityParticleFilter {
    /// Underlying particle filter
    filter: ParticleFilter<VelocityParticle>,
    /// Process noise standard deviation (meters)
    process_noise_std: f64,
    /// Random number generator for particle propagation
    rng: StdRng,
}

impl VelocityParticleFilter {
    /// Create a new velocity particle filter
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Initial state [lat (rad), lon (rad), alt (m), v_n, v_e, v_d (m/s)]
    /// * `num_particles` - Number of particles to use
    /// * `process_noise_std` - Standard deviation of process noise in meters
    ///
    /// # Returns
    ///
    /// A new `VelocityParticleFilter` instance
    pub fn new(initial_state: DVector<f64>, num_particles: usize, process_noise_std: f64) -> Self {
        Self::new_with_seed(initial_state, num_particles, process_noise_std, rand::random())
    }

    /// Create a new velocity particle filter with a specific random seed
    ///
    /// This is useful for reproducible tests.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Initial state [lat (rad), lon (rad), alt (m), v_n, v_e, v_d (m/s)]
    /// * `num_particles` - Number of particles to use
    /// * `process_noise_std` - Standard deviation of process noise in meters
    /// * `seed` - Random seed for deterministic behavior
    ///
    /// # Returns
    ///
    /// A new `VelocityParticleFilter` instance
    pub fn new_with_seed(
        initial_state: DVector<f64>,
        num_particles: usize,
        process_noise_std: f64,
        seed: u64,
    ) -> Self {
        assert_eq!(
            initial_state.len(),
            6,
            "Initial state must be 6-element vector [lat, lon, alt, v_n, v_e, v_d]"
        );
        assert!(num_particles > 0, "Number of particles must be positive");
        assert!(
            process_noise_std >= 0.0,
            "Process noise standard deviation must be non-negative"
        );

        let filter = ParticleFilter::<VelocityParticle>::new(
            &initial_state,
            num_particles,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5, // Resample when effective sample size < 50% of particles
        );

        let rng = StdRng::seed_from_u64(seed);

        VelocityParticleFilter {
            filter,
            process_noise_std,
            rng,
        }
    }

    /// Predict step: propagate all particles using constant velocity model
    ///
    /// Updates each particle's position based on its current velocity and adds
    /// process noise to maintain particle diversity.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step in seconds
    pub fn predict(&mut self, dt: f64) {
        assert!(dt > 0.0, "Time step must be positive");

        // Propagate each particle
        for particle in self.filter.particles_mut() {
            particle.propagate(dt, self.process_noise_std, &mut self.rng);
        }
    }

    /// Update particle weights based on measurement
    ///
    /// Computes the likelihood of each particle given the measurement and updates
    /// the weights accordingly. After calling this method, you should typically
    /// call `normalize_weights()`.
    ///
    /// # Arguments
    ///
    /// * `measurement` - Measurement model (e.g., GPS position)
    pub fn update_weights<M: MeasurementModel>(&mut self, measurement: &M) {
        for particle in self.filter.particles_mut() {
            particle.update_weight(measurement);
        }
    }

    /// Normalize particle weights to sum to 1.0
    pub fn normalize_weights(&mut self) {
        self.filter.normalize_weights();
    }

    /// Resample particles if needed (based on effective sample size)
    ///
    /// Returns true if resampling was performed
    pub fn resample_if_needed(&mut self) -> bool {
        let n_eff = self.filter.effective_sample_size();
        let threshold = 0.5 * self.filter.num_particles() as f64;

        if n_eff < threshold {
            self.filter.resample();
            true
        } else {
            false
        }
    }

    /// Resample particles unconditionally
    pub fn resample(&mut self) {
        self.filter.resample();
    }

    /// Get the current state estimate (weighted mean)
    ///
    /// Returns state vector [lat (rad), lon (rad), alt (m), v_n, v_e, v_d (m/s)]
    pub fn get_estimate(&self) -> DVector<f64> {
        self.filter.get_estimate()
    }

    /// Get the state covariance estimate
    ///
    /// Returns 6x6 covariance matrix
    pub fn get_covariance(&self) -> DMatrix<f64> {
        self.filter.get_certainty()
    }

    /// Get effective sample size
    ///
    /// N_eff = 1 / sum(w_i^2)
    pub fn effective_sample_size(&self) -> f64 {
        self.filter.effective_sample_size()
    }

    /// Get the number of particles
    pub fn num_particles(&self) -> usize {
        self.filter.num_particles()
    }

    /// Get read-only access to particles
    pub fn particles(&self) -> &[VelocityParticle] {
        self.filter.particles()
    }

    /// Get the configured process noise standard deviation
    pub fn process_noise_std(&self) -> f64 {
        self.process_noise_std
    }

    /// Set the process noise standard deviation
    ///
    /// This controls particle spread and can be tuned to achieve desired drift rates
    /// (e.g., 1 km/h, 1 nm/day).
    ///
    /// # Arguments
    ///
    /// * `std_dev` - New process noise standard deviation in meters
    pub fn set_process_noise_std(&mut self, std_dev: f64) {
        assert!(
            std_dev >= 0.0,
            "Process noise standard deviation must be non-negative"
        );
        self.process_noise_std = std_dev;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_velocity_particle_creation() {
        let state = DVector::from_vec(vec![0.7854, -2.1293, 100.0, 10.0, 5.0, 0.0]);
        let particle = VelocityParticle::new(&state, 0.01);

        assert_eq!(particle.state().len(), 6);
        assert_approx_eq!(particle.state()[0], 0.7854, 1e-6);
        assert_approx_eq!(particle.state()[3], 10.0, 1e-6);
        assert_approx_eq!(particle.weight(), 0.01, 1e-10);
    }

    #[test]
    #[should_panic(expected = "VelocityParticle requires 6-state vector")]
    fn test_particle_wrong_dimension() {
        let state = DVector::from_vec(vec![0.0, 0.0, 0.0]); // Wrong size
        let _particle = VelocityParticle::new(&state, 1.0);
    }

    #[test]
    fn test_particle_propagation_constant_velocity_north() {
        let mut particle = VelocityParticle::new(
            &DVector::from_vec(vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0]), // 10 m/s north
            1.0,
        );
        let dt = 1.0; // 1 second
        let mut rng = StdRng::seed_from_u64(42);

        let initial_lat = particle.state()[0];
        particle.propagate(dt, 0.0, &mut rng); // No noise
        let final_lat = particle.state()[0];

        // Latitude should increase (moving north)
        assert!(final_lat > initial_lat);

        // Velocity should remain approximately the same (no noise)
        assert_approx_eq!(particle.state()[3], 10.0, 1e-6);
    }

    #[test]
    fn test_particle_propagation_constant_velocity_east() {
        let mut particle = VelocityParticle::new(
            &DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0]), // 10 m/s east
            1.0,
        );
        let dt = 1.0;
        let mut rng = StdRng::seed_from_u64(42);

        let initial_lon = particle.state()[1];
        particle.propagate(dt, 0.0, &mut rng); // No noise

        // Longitude should increase (moving east)
        assert!(particle.state()[1] > initial_lon);

        // Velocity should remain approximately the same
        assert_approx_eq!(particle.state()[4], 10.0, 1e-6);
    }

    #[test]
    fn test_particle_filter_creation() {
        let initial_state = DVector::from_vec(vec![0.7854, -2.1293, 100.0, 10.0, 5.0, 0.0]);
        let pf = VelocityParticleFilter::new(initial_state.clone(), 100, 1.0);

        assert_eq!(pf.num_particles(), 100);
        assert_approx_eq!(pf.process_noise_std(), 1.0, 1e-10);

        // All particles should start with uniform weights
        let expected_weight = 1.0 / 100.0;
        for particle in pf.particles() {
            assert_approx_eq!(particle.weight(), expected_weight, 1e-10);
        }
    }

    #[test]
    fn test_particle_filter_prediction() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0, 10.0, 5.0, 0.0]);
        let mut pf = VelocityParticleFilter::new(initial_state, 50, 0.1);

        let dt = 1.0;
        pf.predict(dt);

        let estimate = pf.get_estimate();

        // After propagation, mean position should have moved north and east
        assert!(estimate[0] > 0.0); // Latitude increased
        assert!(estimate[1] > 0.0); // Longitude increased

        // Mean velocity should be close to initial (with small noise)
        assert!((estimate[3] - 10.0).abs() < 2.0); // v_n close to 10.0
        assert!((estimate[4] - 5.0).abs() < 2.0); // v_e close to 5.0
    }

    #[test]
    fn test_effective_sample_size() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let pf = VelocityParticleFilter::new(initial_state, 100, 1.0);

        // With uniform weights, N_eff should equal N
        let n_eff = pf.effective_sample_size();
        assert!((n_eff - 100.0).abs() < 1.0); // Allow small numerical error
    }

    #[test]
    fn test_process_noise_setter() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut pf = VelocityParticleFilter::new(initial_state, 100, 1.0);

        pf.set_process_noise_std(2.5);
        assert_approx_eq!(pf.process_noise_std(), 2.5, 1e-10);
    }

    #[test]
    #[should_panic(expected = "Process noise standard deviation must be non-negative")]
    fn test_negative_process_noise() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let _pf = VelocityParticleFilter::new(initial_state, 100, -1.0);
    }
}
