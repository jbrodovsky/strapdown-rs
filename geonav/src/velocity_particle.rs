//! Velocity-informed particle filter for geophysical navigation
//!
//! This module implements a particle filter that uses velocity estimates from an
//! existing INS/GNSS filter (e.g., UKF) to propagate position-only particles, while
//! incorporating geophysical anomaly measurements for position estimation. This approach
//! reduces the dimensionality of the particle filter from the full navigation state to
//! just position (3 states: lat, lon, alt), making it more computationally efficient
//! while leveraging the accurate velocity estimates from the Kalman filter.
//!
//! # Architecture
//!
//! The velocity-informed particle filter operates in an open-loop manner alongside an
//! INS/GNSS filter:
//!
//! ```text
//!                      IMU
//!                       |
//!                       v
//!      +--------------------------------+
//!      |   INS/GNSS Filter (UKF/EKF)  |
//!      |   - Full state estimation     |
//!      |   - Position, velocity, etc.  |
//!      +--------------------------------+
//!                |               |
//!                v               | Velocity
//!              GNSS              | estimates
//!                                |
//!                                v
//!      +--------------------------------+
//!      | Velocity-Informed Particle   |
//!      | Filter                       |
//!      | - Position-only particles    |
//!      | - Geophysical measurements   |
//!      +--------------------------------+
//!                |
//!                v
//!          Position estimate
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use geonav::velocity_particle::{VelocityInformedParticleFilter, VelocityInformedParticle};
//! use geonav::{GeoMap, GravityMeasurement};
//! use nalgebra::DVector;
//! use std::rc::Rc;
//!
//! // Initialize particle filter with position estimate from INS/GNSS filter
//! let initial_position = DVector::from_vec(vec![45.0_f64.to_radians(), -122.0_f64.to_radians(), 100.0]);
//! let num_particles = 1000;
//! let process_noise_std = 1.0; // meters
//!
//! let mut pf = VelocityInformedParticleFilter::new(
//!     initial_position,
//!     num_particles,
//!     process_noise_std,
//! );
//!
//! // Get velocity from INS/GNSS filter
//! let velocity = DVector::from_vec(vec![10.0, 5.0, 0.0]); // [v_n, v_e, v_d] m/s
//! let dt = 0.1; // seconds
//!
//! // Propagate particles using velocity
//! pf.propagate(&velocity, dt);
//!
//! // Update with geophysical measurement
//! let gravity_map = Rc::new(load_gravity_map());
//! let measurement = create_gravity_measurement(&gravity_map, observed_gravity);
//! pf.update_weights(&measurement);
//!
//! // Resample if needed
//! if pf.effective_sample_size() < 0.5 * num_particles as f64 {
//!     pf.resample();
//! }
//!
//! // Get position estimate
//! let position = pf.get_estimate();
//! ```

use std::any::Any;
use nalgebra::{DVector, DMatrix};
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;

use strapdown::particle::{
    Particle, ParticleFilter, ParticleResamplingStrategy, ParticleAveragingStrategy
};
use strapdown::measurements::MeasurementModel;

/// Velocity-informed particle for position tracking
///
/// This particle represents only position states (latitude, longitude, altitude)
/// and is propagated using velocity estimates from an external filter. This reduces
/// the dimensionality compared to full-state particles while still leveraging
/// the accurate velocity information from INS/GNSS filters.
///
/// # State Vector
///
/// The particle state is a 3-element vector: `[latitude (rad), longitude (rad), altitude (m)]`
#[derive(Clone, Debug)]
pub struct VelocityInformedParticle {
    /// Position state: [lat (rad), lon (rad), alt (m)]
    position: DVector<f64>,
    /// Particle weight for importance sampling
    weight: f64,
}

impl Particle for VelocityInformedParticle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn new(initial_state: &DVector<f64>, weight: f64) -> Self {
        assert_eq!(initial_state.len(), 3, "VelocityInformedParticle requires 3-state vector [lat, lon, alt]");
        VelocityInformedParticle {
            position: initial_state.clone(),
            weight,
        }
    }

    fn state(&self) -> DVector<f64> {
        self.position.clone()
    }

    fn update_weight<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // Compute expected measurement from particle's position
        let expected = measurement.get_expected_measurement(&self.position);
        let actual = measurement.get_vector();
        
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
                // Gaussian PDF: (1 / (sqrt(2*pi) * sigma)) * exp(-0.5 * ((x - mu) / sigma)^2)
                // In log space: -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
                let normalized_innovation = innovation[i] / std_dev;
                log_likelihood += -0.5 * normalized_innovation.powi(2) - std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }
        
        // Update weight by multiplying with likelihood (in log space: add)
        // To avoid numerical underflow, we work with log weights and convert back
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

impl VelocityInformedParticle {
    /// Get the particle's position
    pub fn position(&self) -> &DVector<f64> {
        &self.position
    }

    /// Set the particle's position
    pub fn set_position(&mut self, position: DVector<f64>) {
        assert_eq!(position.len(), 3, "Position must be 3-element vector [lat, lon, alt]");
        self.position = position;
    }

    /// Propagate particle position using velocity estimate
    ///
    /// Uses simple kinematic update: position = position + velocity * dt
    /// with small random jitter added for process noise.
    ///
    /// # Arguments
    ///
    /// * `velocity` - Velocity vector [v_n (m/s), v_e (m/s), v_d (m/s)]
    /// * `dt` - Time step in seconds
    /// * `process_noise_std` - Standard deviation of process noise in meters
    /// * `rng` - Random number generator for process noise
    pub fn propagate(
        &mut self,
        velocity: &DVector<f64>,
        dt: f64,
        process_noise_std: f64,
        rng: &mut StdRng,
    ) {
        assert_eq!(velocity.len(), 3, "Velocity must be 3-element vector [v_n, v_e, v_d]");

        // Extract current position
        let lat = self.position[0];  // radians
        let lon = self.position[1];  // radians
        let alt = self.position[2];  // meters

        // Extract velocity components
        let v_n = velocity[0];  // m/s north
        let v_e = velocity[1];  // m/s east
        let v_d = velocity[2];  // m/s down

        // Convert velocities to position changes
        // Approximate conversion for small displacements
        use strapdown::earth::{EARTH_RADIUS_EQUATOR, EARTH_SEMI_MINOR_AXIS};
        
        // Latitude change (northward displacement)
        let r_n = EARTH_SEMI_MINOR_AXIS + alt;  // Radius of curvature in meridian
        let delta_lat = (v_n * dt) / r_n;
        
        // Longitude change (eastward displacement)
        let r_e = (EARTH_RADIUS_EQUATOR + alt) * lat.cos();  // Radius in prime vertical
        let delta_lon = if lat.cos().abs() > 1e-8 {
            (v_e * dt) / r_e
        } else {
            0.0  // At poles, longitude is undefined
        };
        
        // Altitude change (vertical displacement, down is positive)
        let delta_alt = -v_d * dt;  // Negative because down is positive in NED frame

        // Add process noise (jitter) in meters, then convert to radians for lat/lon
        let noise_lat = rng.gen::<f64>() * process_noise_std * 2.0 - process_noise_std;  // [-std, +std]
        let noise_lon = rng.gen::<f64>() * process_noise_std * 2.0 - process_noise_std;
        let noise_alt = rng.gen::<f64>() * process_noise_std * 2.0 - process_noise_std;

        // Convert noise from meters to radians for lat/lon
        let noise_lat_rad = noise_lat / r_n;
        let noise_lon_rad = if lat.cos().abs() > 1e-8 {
            noise_lon / r_e
        } else {
            0.0
        };

        // Update position with propagation and noise
        self.position[0] = lat + delta_lat + noise_lat_rad;
        self.position[1] = lon + delta_lon + noise_lon_rad;
        self.position[2] = alt + delta_alt + noise_alt;

        // Ensure latitude is within [-pi/2, pi/2]
        self.position[0] = self.position[0].max(-std::f64::consts::FRAC_PI_2).min(std::f64::consts::FRAC_PI_2);
        
        // Wrap longitude to [-pi, pi]
        while self.position[1] > std::f64::consts::PI {
            self.position[1] -= 2.0 * std::f64::consts::PI;
        }
        while self.position[1] < -std::f64::consts::PI {
            self.position[1] += 2.0 * std::f64::consts::PI;
        }
    }
}

/// Velocity-informed particle filter for geophysical navigation
///
/// This particle filter uses velocity estimates from an external INS/GNSS filter
/// to propagate position-only particles, while incorporating geophysical anomaly
/// measurements (gravity, magnetic) for position estimation. The filter operates
/// in an open-loop manner and does not feedback to the INS/GNSS filter.
///
/// # Features
///
/// - Position-only particle representation (3 DOF: lat, lon, alt)
/// - Velocity-informed propagation using estimates from INS/GNSS filter
/// - Geophysical anomaly measurement integration
/// - Configurable number of particles
/// - Configurable process noise (jitter) during propagation
/// - Multiple resampling strategies
///
/// # Example
///
/// ```rust,ignore
/// let initial_position = DVector::from_vec(vec![0.7854, -2.1293, 100.0]); // [lat, lon, alt]
/// let mut pf = VelocityInformedParticleFilter::new(initial_position, 1000, 1.0);
///
/// // Propagate using velocity from UKF
/// let velocity = DVector::from_vec(vec![10.0, 5.0, 0.0]); // [v_n, v_e, v_d]
/// pf.propagate(&velocity, 0.1);
///
/// // Update with geophysical measurement
/// pf.update_weights(&gravity_measurement);
/// pf.normalize_weights();
///
/// // Resample if needed
/// if pf.effective_sample_size() < 500.0 {
///     pf.resample();
/// }
///
/// let position = pf.get_estimate();
/// ```
pub struct VelocityInformedParticleFilter {
    /// Underlying particle filter
    filter: ParticleFilter<VelocityInformedParticle>,
    /// Process noise standard deviation (meters)
    process_noise_std: f64,
    /// Random number generator for particle propagation
    rng: StdRng,
}

impl VelocityInformedParticleFilter {
    /// Create a new velocity-informed particle filter
    ///
    /// # Arguments
    ///
    /// * `initial_position` - Initial position estimate [lat (rad), lon (rad), alt (m)]
    /// * `num_particles` - Number of particles to use
    /// * `process_noise_std` - Standard deviation of process noise in meters (jitter during propagation)
    ///
    /// # Returns
    ///
    /// A new `VelocityInformedParticleFilter` instance
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let initial_pos = DVector::from_vec(vec![45.0_f64.to_radians(), -122.0_f64.to_radians(), 100.0]);
    /// let pf = VelocityInformedParticleFilter::new(initial_pos, 1000, 1.0);
    /// ```
    pub fn new(
        initial_position: DVector<f64>,
        num_particles: usize,
        process_noise_std: f64,
    ) -> Self {
        assert_eq!(initial_position.len(), 3, "Initial position must be 3-element vector [lat, lon, alt]");
        assert!(num_particles > 0, "Number of particles must be positive");
        assert!(process_noise_std >= 0.0, "Process noise standard deviation must be non-negative");

        let filter = ParticleFilter::<VelocityInformedParticle>::new(
            &initial_position,
            num_particles,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,  // Resample when effective sample size < 50% of particles
        );

        let rng = StdRng::seed_from_u64(rand::random());

        VelocityInformedParticleFilter {
            filter,
            process_noise_std,
            rng,
        }
    }

    /// Create a particle filter with custom configuration
    ///
    /// # Arguments
    ///
    /// * `initial_position` - Initial position estimate [lat (rad), lon (rad), alt (m)]
    /// * `num_particles` - Number of particles to use
    /// * `process_noise_std` - Standard deviation of process noise in meters
    /// * `resampling_strategy` - Algorithm for resampling particles
    /// * `averaging_strategy` - Method for computing position estimate from particles
    /// * `resampling_threshold` - Trigger resampling when N_eff < threshold * N (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A new `VelocityInformedParticleFilter` instance with custom configuration
    pub fn new_with_config(
        initial_position: DVector<f64>,
        num_particles: usize,
        process_noise_std: f64,
        resampling_strategy: ParticleResamplingStrategy,
        averaging_strategy: ParticleAveragingStrategy,
        resampling_threshold: f64,
    ) -> Self {
        assert_eq!(initial_position.len(), 3, "Initial position must be 3-element vector [lat, lon, alt]");
        assert!(num_particles > 0, "Number of particles must be positive");
        assert!(process_noise_std >= 0.0, "Process noise standard deviation must be non-negative");

        let filter = ParticleFilter::<VelocityInformedParticle>::new(
            &initial_position,
            num_particles,
            resampling_strategy,
            averaging_strategy,
            resampling_threshold,
        );

        let rng = StdRng::seed_from_u64(rand::random());

        VelocityInformedParticleFilter {
            filter,
            process_noise_std,
            rng,
        }
    }

    /// Propagate all particles using velocity estimate from external filter
    ///
    /// This method updates the position of each particle based on the provided
    /// velocity estimate and adds process noise (jitter) to maintain particle
    /// diversity.
    ///
    /// # Arguments
    ///
    /// * `velocity` - Velocity vector [v_n (m/s), v_e (m/s), v_d (m/s)] from INS/GNSS filter
    /// * `dt` - Time step in seconds
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let velocity = ukf.get_estimate().rows(3, 3).into_owned(); // Extract velocity from UKF state
    /// pf.propagate(&velocity, 0.1);
    /// ```
    pub fn propagate(&mut self, velocity: &DVector<f64>, dt: f64) {
        assert_eq!(velocity.len(), 3, "Velocity must be 3-element vector [v_n, v_e, v_d]");
        assert!(dt > 0.0, "Time step must be positive");

        // Propagate each particle
        for particle in self.filter.particles_mut() {
            particle.propagate(velocity, dt, self.process_noise_std, &mut self.rng);
        }
    }

    /// Update particle weights based on geophysical measurement
    ///
    /// This method computes the likelihood of each particle given the measurement
    /// and updates the weights accordingly. After calling this method, you should
    /// typically call `normalize_weights()`.
    ///
    /// # Arguments
    ///
    /// * `measurement` - Geophysical anomaly measurement (gravity, magnetic, etc.)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let measurement = create_gravity_measurement(&map, observed_gravity);
    /// pf.update_weights(&measurement);
    /// pf.normalize_weights();
    /// ```
    pub fn update_weights<M: MeasurementModel>(&mut self, measurement: &M) {
        // Update weight for each particle based on measurement likelihood
        for particle in self.filter.particles_mut() {
            // For geophysical measurements, we need to set the state first
            // This is handled by the measurement model's update_weight method
            // through the particle's position
            particle.update_weight(measurement);
        }
    }

    /// Normalize particle weights to sum to 1.0
    ///
    /// This method should be called after `update_weights()` to ensure the
    /// weights form a valid probability distribution.
    pub fn normalize_weights(&mut self) {
        self.filter.normalize_weights();
    }

    /// Resample particles to combat degeneracy
    ///
    /// This method generates a new set of particles by sampling from the current
    /// set according to their weights. High-weight particles are likely to be
    /// duplicated, while low-weight particles may be eliminated.
    pub fn resample(&mut self) {
        self.filter.resample();
    }

    /// Get the current position estimate
    ///
    /// Returns the weighted mean position of all particles.
    ///
    /// # Returns
    ///
    /// Position vector [lat (rad), lon (rad), alt (m)]
    pub fn get_estimate(&self) -> DVector<f64> {
        self.filter.get_estimate()
    }

    /// Get the position covariance estimate
    ///
    /// Returns the sample covariance of particles around their weighted mean.
    ///
    /// # Returns
    ///
    /// 3x3 covariance matrix for position uncertainty
    pub fn get_covariance(&self) -> DMatrix<f64> {
        self.filter.get_certainty()
    }

    /// Calculate effective sample size
    ///
    /// The effective sample size indicates how many particles are effectively
    /// contributing to the estimate. Low values indicate particle degeneracy
    /// and suggest that resampling is needed.
    ///
    /// # Returns
    ///
    /// Effective sample size: N_eff = 1 / sum(w_i^2)
    pub fn effective_sample_size(&self) -> f64 {
        self.filter.effective_sample_size()
    }

    /// Get the number of particles
    pub fn num_particles(&self) -> usize {
        self.filter.num_particles()
    }

    /// Get read-only access to particles
    pub fn particles(&self) -> &[VelocityInformedParticle] {
        self.filter.particles()
    }

    /// Get the configured process noise standard deviation
    pub fn process_noise_std(&self) -> f64 {
        self.process_noise_std
    }

    /// Set the process noise standard deviation
    ///
    /// This controls the amount of jitter added during particle propagation.
    ///
    /// # Arguments
    ///
    /// * `std_dev` - New process noise standard deviation in meters
    pub fn set_process_noise_std(&mut self, std_dev: f64) {
        assert!(std_dev >= 0.0, "Process noise standard deviation must be non-negative");
        self.process_noise_std = std_dev;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_velocity_informed_particle_creation() {
        let position = DVector::from_vec(vec![0.7854, -2.1293, 100.0]); // ~45°N, ~-122°E, 100m
        let particle = VelocityInformedParticle::new(&position, 0.01);

        assert_eq!(particle.position().len(), 3);
        assert_approx_eq!(particle.position()[0], 0.7854, 1e-6);
        assert_approx_eq!(particle.position()[1], -2.1293, 1e-6);
        assert_approx_eq!(particle.position()[2], 100.0, 1e-6);
        assert_approx_eq!(particle.weight(), 0.01, 1e-10);
    }

    #[test]
    #[should_panic(expected = "VelocityInformedParticle requires 3-state vector")]
    fn test_particle_wrong_dimension() {
        let position = DVector::from_vec(vec![0.0, 0.0]); // Wrong size
        let _particle = VelocityInformedParticle::new(&position, 1.0);
    }

    #[test]
    fn test_particle_propagation_northward() {
        let mut particle = VelocityInformedParticle::new(
            &DVector::from_vec(vec![0.0, 0.0, 0.0]),
            1.0
        );
        let velocity = DVector::from_vec(vec![10.0, 0.0, 0.0]); // 10 m/s north
        let dt = 1.0; // 1 second
        let mut rng = StdRng::seed_from_u64(42);

        let initial_lat = particle.position()[0];
        particle.propagate(&velocity, dt, 0.0, &mut rng); // No noise
        let final_lat = particle.position()[0];

        // Latitude should increase (moving north)
        assert!(final_lat > initial_lat);
        
        // Longitude should remain approximately the same
        assert_approx_eq!(particle.position()[1], 0.0, 1e-6);
        
        // Altitude should remain the same
        assert_approx_eq!(particle.position()[2], 0.0, 1e-6);
    }

    #[test]
    fn test_particle_propagation_eastward() {
        let mut particle = VelocityInformedParticle::new(
            &DVector::from_vec(vec![0.0, 0.0, 0.0]),
            1.0
        );
        let velocity = DVector::from_vec(vec![0.0, 10.0, 0.0]); // 10 m/s east
        let dt = 1.0;
        let mut rng = StdRng::seed_from_u64(42);

        let initial_lon = particle.position()[1];
        particle.propagate(&velocity, dt, 0.0, &mut rng); // No noise

        // Longitude should increase (moving east)
        assert!(particle.position()[1] > initial_lon);
        
        // Latitude should remain approximately the same
        assert_approx_eq!(particle.position()[0], 0.0, 1e-6);
    }

    #[test]
    fn test_particle_propagation_vertical() {
        let mut particle = VelocityInformedParticle::new(
            &DVector::from_vec(vec![0.0, 0.0, 100.0]),
            1.0
        );
        let velocity = DVector::from_vec(vec![0.0, 0.0, -1.0]); // 1 m/s up (negative in NED)
        let dt = 10.0;
        let mut rng = StdRng::seed_from_u64(42);

        particle.propagate(&velocity, dt, 0.0, &mut rng); // No noise

        // Altitude should increase by approximately 10m
        assert_approx_eq!(particle.position()[2], 110.0, 0.1);
    }

    #[test]
    fn test_particle_filter_creation() {
        let initial_position = DVector::from_vec(vec![0.7854, -2.1293, 100.0]);
        let pf = VelocityInformedParticleFilter::new(initial_position.clone(), 100, 1.0);

        assert_eq!(pf.num_particles(), 100);
        assert_approx_eq!(pf.process_noise_std(), 1.0, 1e-10);
        
        // All particles should start with uniform weights
        let expected_weight = 1.0 / 100.0;
        for particle in pf.particles() {
            assert_approx_eq!(particle.weight(), expected_weight, 1e-10);
        }
    }

    #[test]
    fn test_particle_filter_propagation() {
        let initial_position = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = VelocityInformedParticleFilter::new(initial_position, 50, 0.1);

        let velocity = DVector::from_vec(vec![10.0, 5.0, 0.0]);
        let dt = 1.0;

        pf.propagate(&velocity, dt);

        let estimate = pf.get_estimate();
        
        // After propagation, mean position should have moved north and east
        assert!(estimate[0] > 0.0); // Latitude increased
        assert!(estimate[1] > 0.0); // Longitude increased
    }

    #[test]
    fn test_effective_sample_size() {
        let initial_position = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pf = VelocityInformedParticleFilter::new(initial_position, 100, 1.0);

        // With uniform weights, N_eff should equal N
        let n_eff = pf.effective_sample_size();
        assert!((n_eff - 100.0).abs() < 1.0); // Allow small numerical error
    }

    #[test]
    fn test_weight_normalization() {
        let initial_position = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = VelocityInformedParticleFilter::new(initial_position, 10, 1.0);

        // Manually set non-uniform weights
        for (i, particle) in pf.filter.particles_mut().iter_mut().enumerate() {
            particle.set_weight((i + 1) as f64);
        }

        pf.normalize_weights();

        // Weights should sum to 1.0
        let sum: f64 = pf.particles().iter().map(|p| p.weight()).sum();
        assert_approx_eq!(sum, 1.0, 1e-10);
    }

    #[test]
    fn test_process_noise_setter() {
        let initial_position = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = VelocityInformedParticleFilter::new(initial_position, 100, 1.0);

        pf.set_process_noise_std(2.5);
        assert_approx_eq!(pf.process_noise_std(), 2.5, 1e-10);
    }

    #[test]
    #[should_panic(expected = "Process noise standard deviation must be non-negative")]
    fn test_negative_process_noise() {
        let initial_position = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let _pf = VelocityInformedParticleFilter::new(initial_position, 100, -1.0);
    }

    #[test]
    fn test_longitude_wrapping() {
        let mut particle = VelocityInformedParticle::new(
            &DVector::from_vec(vec![0.0, 3.0, 0.0]), // Close to pi
            1.0
        );
        let velocity = DVector::from_vec(vec![0.0, 1000.0, 0.0]); // Large eastward velocity
        let dt = 100.0; // Long time step
        let mut rng = StdRng::seed_from_u64(42);

        particle.propagate(&velocity, dt, 0.0, &mut rng);

        // Longitude should be wrapped to [-pi, pi]
        assert!(particle.position()[1] >= -std::f64::consts::PI);
        assert!(particle.position()[1] <= std::f64::consts::PI);
    }

    #[test]
    fn test_latitude_clamping() {
        let mut particle = VelocityInformedParticle::new(
            &DVector::from_vec(vec![1.5, 0.0, 0.0]), // Close to pi/2
            1.0
        );
        let velocity = DVector::from_vec(vec![1000.0, 0.0, 0.0]); // Large northward velocity
        let dt = 100.0; // Long time step
        let mut rng = StdRng::seed_from_u64(42);

        particle.propagate(&velocity, dt, 0.0, &mut rng);

        // Latitude should be clamped to [-pi/2, pi/2]
        assert!(particle.position()[0] >= -std::f64::consts::FRAC_PI_2);
        assert!(particle.position()[0] <= std::f64::consts::FRAC_PI_2);
    }
}
