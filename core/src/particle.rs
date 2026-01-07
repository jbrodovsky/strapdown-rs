//! Particle filter implementation for nonlinear Bayesian state estimation.
//!
//! This module provides a particle filter (also known as Sequential Monte Carlo) implementation
//! for strapdown inertial navigation. Unlike Kalman filters which assume Gaussian distributions,
//! particle filters can represent arbitrary probability distributions through weighted samples.
//! This makes them well-suited for highly nonlinear systems and non-Gaussian noise characteristics.
//! However, they typically require more computational resources due to the need for many particles
//! to accurately approximate the posterior distribution. As such, they are often used in scenarios
//! where Kalman filters struggle, such as in the presence of multimodal distributions or severe
//! nonlinearities with state dynamics and dimensions that are manageable.
//!
//! A full 15-state canonical strapdown inertial navigation system is feasible with particle filters
//! for moderate particle counts (e.g., 1000-5000 particles) on modern hardware with *reasonably
//! good* IMU quality. For more challenging scenarios, such as low-quality IMUs or more complex
//! motion dynamics, hybrid approaches like Rao-Blackwellized Particle Filters (RBPF) can
//! be employed. RBPFs use particles to represent a subset of the state (e.g., position and velocity)
//! while using Kalman filters for the remaining states (e.g., attitude and sensor biases), reducing
//! the dimensionality of the particle filter and improving efficiency.
//!
//! # Key Components
//!
//! - [`Particle`]: A trait defining the required methods for particle state representation
//! - [`ParticleFilter`]: The main particle filter struct managing particles and weights
//! - [`ParticleAveragingStrategy`]: Methods for extracting state estimates from particles
//! - [`ParticleResamplingStrategy`]: Algorithms for combating particle degeneracy
//!
//! # Usage Example
//!
//! To use the particle filter, you must first implement the `Particle` trait for your custom
//! particle type:
//!
//! ```rust
//! use strapdown::particle::{Particle, ParticleFilter, ParticleResamplingStrategy, ParticleAveragingStrategy};
//! use strapdown::measurements::MeasurementModel;
//! use nalgebra::DVector;
//! use std::any::Any;
//!
//! // Define your custom particle type
//! struct MyParticle {
//!     state: DVector<f64>,  // [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
//!     weight: f64,
//! }
//!
//! impl Particle for MyParticle {
//!     fn as_any(&self) -> &dyn Any { self }
//!     fn as_any_mut(&mut self) -> &mut dyn Any { self }
//!     
//!     fn new(initial_state: &DVector<f64>, weight: f64) -> Self {
//!         MyParticle {
//!             state: initial_state.clone(),
//!             weight,
//!         }
//!     }
//!     
//!     fn state(&self) -> DVector<f64> {
//!         self.state.clone()
//!     }
//!     
//!     fn update_weight<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
//!         // Compute likelihood p(z|x) and update weight
//!         let expected = measurement.get_expected_measurement(&self.state);
//!         let actual = measurement.get_measurement(&self.state);
//!         let diff = actual - expected;
//!         
//!         // Simple Gaussian likelihood (customize for your application)
//!         let likelihood = (-0.5 * diff.norm_squared() as f64).exp();
//!         self.weight *= likelihood;
//!     }
//!     
//!     fn weight(&self) -> f64 { self.weight }
//!     fn set_weight(&mut self, weight: f64) { self.weight = weight; }
//! }
//!
//! // Create and use the particle filter
//! let initial_state = DVector::from_vec(vec![0.0; 9]);
//! let mut pf = ParticleFilter::<MyParticle>::new(
//!     &initial_state,
//!     1000,  // number of particles
//!     ParticleResamplingStrategy::Systematic,
//!     ParticleAveragingStrategy::WeightedMean,
//!     0.5,   // resampling threshold
//! );
//!
//! // Propagate particles (user responsibility)
//! for particle in pf.particles_mut() {
//!     // Apply your motion model with process noise
//!     // particle.propagate(imu_data, dt);
//! }
//!
//! // Update particle weights based on measurements
//! // for particle in pf.particles_mut() {
//! //     particle.update_weight(&gps_measurement);
//! // }
//! // pf.normalize_weights();
//!
//! // Resample if needed
//! // let n_eff = pf.effective_sample_size();
//! // if n_eff < threshold {
//! //     pf.resample();
//! // }
//!
//! // Get state estimate
//! let state_estimate = pf.get_estimate();
//! let covariance = pf.get_certainty();
//! ```
//!

use std::any::Any;

use crate::measurements::MeasurementModel;

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::prelude::*;

/// Trait defining the interface for particle state representation
///
/// Users must implement this trait for their custom particle types to use with
/// the generic ParticleFilter. The trait provides methods for particle initialization,
/// state extraction, and weight management.
pub trait Particle: Any {
    /// Downcast helper method to allow for type-safe downcasting
    fn as_any(&self) -> &dyn Any;
    /// Downcast helper method for mutable references
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Creates a new particle with the given initial state and weight
    ///
    /// # Arguments
    /// * `initial_state` - The initial state vector for the particle
    /// * `weight` - The initial weight (typically 1/N for N particles)
    fn new(initial_state: &DVector<f64>, weight: f64) -> Self
    where
        Self: Sized;

    /// Returns the state vector of the particle
    fn state(&self) -> DVector<f64>;

    /// Updates the particle weight based on the provided measurement model and actual measurement
    ///
    /// This method should compute the likelihood of the measurement given the particle's
    /// current state and update the weight accordingly. The specific implementation
    /// depends on the measurement model and noise characteristics.
    fn update_weight<M: MeasurementModel + ?Sized>(&mut self, measurement: &M);

    /// Returns the current weight of the particle
    fn weight(&self) -> f64;

    /// Sets the weight of the particle
    fn set_weight(&mut self, weight: f64);
}

/// Strategy for extracting state estimates from particles
///
/// Different averaging strategies can be used depending on the application:
/// - Mean: Simple arithmetic mean of particle states
/// - WeightedMean: Weighted average using particle weights
/// - HighestWeight: State of the particle with highest weight
#[derive(Clone, Copy, Debug, Default)]
pub enum ParticleAveragingStrategy {
    /// Arithmetic mean of all particle states (ignores weights)
    Mean,
    /// Weighted mean using particle weights
    #[default]
    WeightedMean,
    /// State of the particle with the highest weight (maximum a posteriori estimate)
    HighestWeight,
}

/// Strategy for resampling particles to combat degeneracy
///
/// Particle filters suffer from degeneracy where most particles have negligible weights.
/// Resampling addresses this by generating new particles from the current set based on
/// their weights.
#[derive(Clone, Copy, Debug, Default)]
pub enum ParticleResamplingStrategy {
    /// Multinomial resampling: draw N independent samples from weighted distribution
    Multinomial,
    /// Systematic resampling: deterministic selection with random offset
    #[default]
    Systematic,
    /// Stratified resampling: divide [0,1] into N strata and sample once per stratum
    Stratified,
    /// Residual resampling: deterministic selection of high-weight particles, then random sampling for remainder
    Residual,
}

/// Generic particle filter for Bayesian state estimation
///
/// This struct manages a set of weighted particles and provides methods for
/// prediction (propagation), update (weight adjustment), and resampling.
/// It implements the NavigationFilter trait to be compatible with the existing
/// filter architecture.
///
/// # Type Parameters
/// * `P` - The particle type, must implement the Particle trait
///
/// # Example
/// ```ignore
/// // Define a custom particle type
/// struct MyParticle { state: DVector<f64>, weight: f64 }
///
/// impl Particle for MyParticle {
///     // ... implement required methods
/// }
///
/// // Create particle filter
/// let initial_state = DVector::from_vec(vec![0.0; 9]);
/// let pf = ParticleFilter::<MyParticle>::new(
///     &initial_state,
///     1000, // number of particles
///     ParticleResamplingStrategy::Systematic,
///     ParticleAveragingStrategy::WeightedMean,
///     0.5, // resampling threshold
/// );
/// ```
pub struct ParticleFilter<P: Particle> {
    /// Collection of particles
    particles: Vec<P>,
    /// Resampling strategy
    resampling_strategy: ParticleResamplingStrategy,
    /// Averaging strategy for state estimation
    averaging_strategy: ParticleAveragingStrategy,
    /// Effective sample size threshold for triggering resampling (0.0 to 1.0)
    /// Resampling occurs when N_eff < threshold * N
    /// Note: Currently stored but not used in resampling logic
    _resampling_threshold: f64,
    /// Random number generator for resampling
    rng: StdRng,
}

impl<P: Particle> ParticleFilter<P> {
    /// Create a new particle filter
    ///
    /// # Arguments
    /// * `initial_state` - Initial state estimate
    /// * `num_particles` - Number of particles to use
    /// * `resampling_strategy` - Strategy for resampling particles
    /// * `averaging_strategy` - Strategy for computing state estimate
    /// * `resampling_threshold` - Threshold for triggering resampling (0.0 to 1.0)
    ///
    /// # Returns
    /// A new ParticleFilter instance with uniformly weighted particles
    pub fn new(
        initial_state: &DVector<f64>,
        num_particles: usize,
        resampling_strategy: ParticleResamplingStrategy,
        averaging_strategy: ParticleAveragingStrategy,
        resampling_threshold: f64,
    ) -> Self {
        assert!(num_particles > 0, "Number of particles must be positive");
        assert!(
            (0.0..=1.0).contains(&resampling_threshold),
            "Resampling threshold must be between 0.0 and 1.0"
        );

        let initial_weight = 1.0 / num_particles as f64;
        let particles: Vec<P> = (0..num_particles)
            .map(|_| P::new(initial_state, initial_weight))
            .collect();

        Self {
            particles,
            resampling_strategy,
            averaging_strategy,
            _resampling_threshold: resampling_threshold,
            rng: StdRng::seed_from_u64(rand::rng().random()),
        }
    }

    /// Get the number of particles
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    /// Calculate effective sample size
    ///
    /// N_eff = 1 / sum(w_i^2)
    ///
    /// This metric indicates how many particles are effectively contributing
    /// to the state estimate. Low N_eff indicates particle degeneracy.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_squared_weights: f64 = self.particles.iter().map(|p| p.weight().powi(2)).sum();

        if sum_squared_weights > 0.0 {
            1.0 / sum_squared_weights
        } else {
            0.0
        }
    }

    /// Normalize particle weights to sum to 1.0
    pub fn normalize_weights(&mut self) {
        let sum: f64 = self.particles.iter().map(|p| p.weight()).sum();

        if sum > 0.0 {
            for particle in &mut self.particles {
                particle.set_weight(particle.weight() / sum);
            }
        } else {
            // If all weights are zero, reset to uniform
            let uniform_weight = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.set_weight(uniform_weight);
            }
        }
    }

    /// Resample particles based on the configured strategy
    ///
    /// This creates a new set of particles by sampling from the current
    /// set according to their weights. High-weight particles are likely
    /// to be duplicated, while low-weight particles may be eliminated.
    pub fn resample(&mut self) {
        let num_particles = self.particles.len();
        let weights: Vec<f64> = self.particles.iter().map(|p| p.weight()).collect();

        // Get indices of resampled particles
        let indices = match self.resampling_strategy {
            ParticleResamplingStrategy::Multinomial => {
                self.multinomial_resample(&weights, num_particles)
            }
            ParticleResamplingStrategy::Systematic => {
                self.systematic_resample(&weights, num_particles)
            }
            ParticleResamplingStrategy::Stratified => {
                self.stratified_resample(&weights, num_particles)
            }
            ParticleResamplingStrategy::Residual => self.residual_resample(&weights, num_particles),
        };

        // Create new particle set from resampled indices
        let new_particles: Vec<P> = indices
            .iter()
            .map(|&idx| {
                let state = self.particles[idx].state();
                P::new(&state, 1.0 / num_particles as f64)
            })
            .collect();

        self.particles = new_particles;
    }

    /// Multinomial resampling
    ///
    /// Draw N independent samples from the weighted distribution
    fn multinomial_resample(&mut self, weights: &[f64], num_samples: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(num_samples);
        let cumsum: Vec<f64> = weights
            .iter()
            .scan(0.0, |acc, &w| {
                *acc += w;
                Some(*acc)
            })
            .collect();

        for _ in 0..num_samples {
            let u: f64 = self.rng.random();
            let idx = cumsum
                .iter()
                .position(|&c| c >= u)
                .unwrap_or(weights.len() - 1);
            indices.push(idx);
        }

        indices
    }

    /// Systematic resampling
    ///
    /// More efficient and lower variance than multinomial
    fn systematic_resample(&mut self, weights: &[f64], num_samples: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(num_samples);
        let step = 1.0 / num_samples as f64;
        let start: f64 = self.rng.random::<f64>() * step;

        let mut cumsum = 0.0;
        let mut i = 0;

        for j in 0..num_samples {
            let u = start + (j as f64) * step;

            while cumsum < u && i < weights.len() {
                cumsum += weights[i];
                i += 1;
            }

            // Ensure i is at least 1 before subtracting, and clamp to valid range
            let idx = if i > 0 { i - 1 } else { 0 };
            indices.push(idx.min(weights.len() - 1));
        }

        indices
    }

    /// Stratified resampling
    ///
    /// Divide [0,1] into N strata and sample once per stratum
    fn stratified_resample(&mut self, weights: &[f64], num_samples: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(num_samples);
        let step = 1.0 / num_samples as f64;

        let mut cumsum = 0.0;
        let mut i = 0;

        for j in 0..num_samples {
            let u = (j as f64 + self.rng.random::<f64>()) * step;

            while cumsum < u && i < weights.len() {
                cumsum += weights[i];
                i += 1;
            }

            // Ensure i is at least 1 before subtracting, and clamp to valid range
            let idx = if i > 0 { i - 1 } else { 0 };
            indices.push(idx.min(weights.len() - 1));
        }

        indices
    }

    /// Residual resampling
    ///
    /// Deterministically selects particles based on integer parts of N*weights,
    /// then randomly samples remaining particles. This minimizes variance while
    /// ensuring particles with high weights are always included.
    fn residual_resample(&mut self, weights: &[f64], num_samples: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(num_samples);

        // Calculate expected number of copies for each particle
        let n_weights: Vec<f64> = weights.iter().map(|&w| w * num_samples as f64).collect();

        // Deterministic selection: take integer part of each weight
        let mut residual_weights = Vec::with_capacity(weights.len());
        for (i, &nw) in n_weights.iter().enumerate() {
            let n_copies = nw.floor() as usize;
            for _ in 0..n_copies {
                indices.push(i);
            }
            // Store fractional part for residual sampling
            residual_weights.push(nw - nw.floor());
        }

        // Normalize residual weights
        let residual_sum: f64 = residual_weights.iter().sum();
        if residual_sum > 0.0 {
            for w in &mut residual_weights {
                *w /= residual_sum;
            }
        }

        // Random sampling for remaining particles
        let remaining = num_samples - indices.len();
        if remaining > 0 {
            // Use systematic resampling for the residuals
            let step = 1.0 / remaining as f64;
            let start: f64 = self.rng.random::<f64>() * step;

            let mut cumsum = 0.0;
            let mut i = 0;

            for j in 0..remaining {
                let u = start + (j as f64) * step;

                while cumsum < u && i < residual_weights.len() {
                    cumsum += residual_weights[i];
                    i += 1;
                }

                let idx = if i > 0 { i - 1 } else { 0 };
                indices.push(idx.min(weights.len() - 1));
            }
        }

        indices
    }

    /// Compute state estimate from particles using the configured averaging strategy
    fn compute_state_estimate(&self) -> DVector<f64> {
        assert!(
            !self.particles.is_empty(),
            "Cannot compute state estimate from empty particle set"
        );

        match self.averaging_strategy {
            ParticleAveragingStrategy::Mean => {
                // Simple mean
                let state_dim = self.particles[0].state().len();
                let mut mean = DVector::zeros(state_dim);

                for particle in &self.particles {
                    mean += particle.state();
                }

                mean / self.particles.len() as f64
            }
            ParticleAveragingStrategy::WeightedMean => {
                // Weighted mean
                let state_dim = self.particles[0].state().len();
                let mut weighted_mean = DVector::zeros(state_dim);

                for particle in &self.particles {
                    weighted_mean += particle.state() * particle.weight();
                }

                weighted_mean
            }
            ParticleAveragingStrategy::HighestWeight => {
                // Return state of particle with highest weight (MAP estimate)
                let max_particle = self
                    .particles
                    .iter()
                    .max_by(|a, b| a.weight().partial_cmp(&b.weight()).unwrap())
                    .expect("Particle set is not empty");

                max_particle.state()
            }
        }
    }

    /// Compute covariance estimate from particles
    ///
    /// This computes the sample covariance of the particles around their
    /// weighted mean, providing an estimate of state uncertainty.
    fn compute_covariance(&self) -> DMatrix<f64> {
        assert!(
            !self.particles.is_empty(),
            "Cannot compute covariance from empty particle set"
        );

        let mean = self.compute_state_estimate();
        let state_dim = mean.len();
        let mut cov = DMatrix::zeros(state_dim, state_dim);

        for particle in &self.particles {
            let diff = particle.state() - &mean;
            cov += particle.weight() * &diff * diff.transpose();
        }

        cov
    }

    /// Get read-only access to the particles
    pub fn particles(&self) -> &[P] {
        &self.particles
    }

    /// Get mutable access to the particles
    pub fn particles_mut(&mut self) -> &mut [P] {
        &mut self.particles
    }

    /// Get the current state estimate
    ///
    /// Returns the state estimate computed from particles using the configured
    /// averaging strategy (mean, weighted mean, or highest weight).
    pub fn get_estimate(&self) -> DVector<f64> {
        self.compute_state_estimate()
    }

    /// Get the state covariance estimate
    ///
    /// Returns the sample covariance computed from the particles around their mean.
    pub fn get_certainty(&self) -> DMatrix<f64> {
        self.compute_covariance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurements::GPSPositionMeasurement;

    /// Simple test particle implementation for unit testing
    #[derive(Clone)]
    struct SimpleParticle {
        state: DVector<f64>,
        weight: f64,
    }

    impl Particle for SimpleParticle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn new(initial_state: &DVector<f64>, weight: f64) -> Self {
            SimpleParticle {
                state: initial_state.clone(),
                weight,
            }
        }

        fn state(&self) -> DVector<f64> {
            self.state.clone()
        }

        fn update_weight<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
            // Simple likelihood: exp(-distance^2 / (2 * sigma^2))
            let expected = measurement.get_expected_measurement(&self.state);
            let actual = measurement.get_measurement(&self.state);
            let diff = actual - expected;
            let noise = measurement.get_noise();

            // Simple Gaussian likelihood assuming diagonal covariance
            let mut likelihood = 1.0;
            for i in 0..diff.len() {
                let sigma_sq = noise[(i, i)];
                if sigma_sq > 0.0 {
                    likelihood *= (-0.5 * diff[i].powi(2) / sigma_sq).exp();
                }
            }

            self.weight *= likelihood;
        }

        fn weight(&self) -> f64 {
            self.weight
        }

        fn set_weight(&mut self, weight: f64) {
            self.weight = weight;
        }
    }

    #[test]
    fn test_particle_filter_construction() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        assert_eq!(pf.num_particles(), 100);

        // Check that all particles are initialized with uniform weight
        let expected_weight = 1.0 / 100.0;
        for particle in pf.particles() {
            assert!((particle.weight() - expected_weight).abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "Number of particles must be positive")]
    fn test_particle_filter_zero_particles() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let _pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            0,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );
    }

    #[test]
    #[should_panic(expected = "Resampling threshold must be between 0.0 and 1.0")]
    fn test_particle_filter_invalid_threshold() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let _pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            1.5, // Invalid threshold
        );
    }

    #[test]
    fn test_effective_sample_size() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        // With uniform weights, N_eff should equal N
        let n_eff = pf.effective_sample_size();
        assert!((n_eff - 100.0).abs() < 1.0); // Allow small numerical error
    }

    #[test]
    fn test_normalize_weights() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            10,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        // Set non-uniform weights
        for (i, particle) in pf.particles_mut().iter_mut().enumerate() {
            particle.set_weight((i + 1) as f64);
        }

        pf.normalize_weights();

        // Check that weights sum to 1.0
        let sum: f64 = pf.particles().iter().map(|p| p.weight()).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights_all_zero() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            10,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        // Set all weights to zero
        for particle in pf.particles_mut() {
            particle.set_weight(0.0);
        }

        pf.normalize_weights();

        // Should reset to uniform weights
        let expected_weight = 1.0 / 10.0;
        for particle in pf.particles() {
            assert!((particle.weight() - expected_weight).abs() < 1e-10);
        }
    }

    #[test]
    fn test_multinomial_resampling() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Multinomial,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        pf.resample();

        // After resampling, should still have 100 particles with uniform weights
        assert_eq!(pf.num_particles(), 100);
        let expected_weight = 1.0 / 100.0;
        for particle in pf.particles() {
            assert!((particle.weight() - expected_weight).abs() < 1e-10);
        }
    }

    #[test]
    fn test_systematic_resampling() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        pf.resample();

        assert_eq!(pf.num_particles(), 100);
    }

    #[test]
    fn test_stratified_resampling() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Stratified,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        pf.resample();

        assert_eq!(pf.num_particles(), 100);
    }

    #[test]
    fn test_residual_resampling() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Residual,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        pf.resample();

        assert_eq!(pf.num_particles(), 100);
    }

    #[test]
    fn test_compute_state_estimate_mean() {
        let initial_state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            10,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::Mean,
            0.5,
        );

        let estimate = pf.get_estimate();

        // All particles have the same state, so mean should equal initial state
        assert!((estimate[0] - 1.0).abs() < 1e-10);
        assert!((estimate[1] - 2.0).abs() < 1e-10);
        assert!((estimate[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_state_estimate_weighted_mean() {
        let initial_state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            10,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        let estimate = pf.get_estimate();

        // With uniform weights, weighted mean equals simple mean
        assert!((estimate[0] - 1.0).abs() < 1e-10);
        assert!((estimate[1] - 2.0).abs() < 1e-10);
        assert!((estimate[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_state_estimate_highest_weight() {
        let initial_state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            10,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::HighestWeight,
            0.5,
        );

        // Set different weights - make one particle have highest weight
        for (i, particle) in pf.particles_mut().iter_mut().enumerate() {
            if i == 5 {
                particle.set_weight(0.5);
                // Give it a different state
                *particle = SimpleParticle::new(&DVector::from_vec(vec![10.0, 20.0, 30.0]), 0.5);
            } else {
                particle.set_weight(0.05);
            }
        }

        let estimate = pf.get_estimate();

        // Should return the state of particle with highest weight
        assert!((estimate[0] - 10.0).abs() < 1e-10);
        assert!((estimate[1] - 20.0).abs() < 1e-10);
        assert!((estimate[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_covariance() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        let cov = pf.get_certainty();

        // With all particles at the same state, covariance should be near zero
        assert_eq!(cov.nrows(), 3);
        assert_eq!(cov.ncols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(cov[(i, j)].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_weight_update_and_normalization() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            50,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        // Create a GPS position measurement
        let measurement = GPSPositionMeasurement {
            latitude: 0.1,
            longitude: 0.1,
            altitude: 10.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };

        // Update weights for each particle
        for particle in pf.particles_mut() {
            particle.update_weight(&measurement);
        }

        // Normalize weights
        pf.normalize_weights();

        // Weights should be normalized
        let sum: f64 = pf.particles().iter().map(|p| p.weight()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_resampling_strategy_default() {
        let strategy = ParticleResamplingStrategy::default();
        assert!(matches!(strategy, ParticleResamplingStrategy::Systematic));
    }

    #[test]
    fn test_averaging_strategy_default() {
        let strategy = ParticleAveragingStrategy::default();
        assert!(matches!(strategy, ParticleAveragingStrategy::WeightedMean));
    }

    #[test]
    fn test_particle_filter_with_different_state_dimensions() {
        // Test with 9-state vector
        let initial_state = DVector::from_vec(vec![0.0; 9]);
        let pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            50,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        assert_eq!(pf.num_particles(), 50);
        let estimate = pf.get_estimate();
        assert_eq!(estimate.len(), 9);
    }

    #[test]
    fn test_effective_sample_size_degeneracy() {
        let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mut pf = ParticleFilter::<SimpleParticle>::new(
            &initial_state,
            100,
            ParticleResamplingStrategy::Systematic,
            ParticleAveragingStrategy::WeightedMean,
            0.5,
        );

        // Create degenerate weight distribution (one particle has all weight)
        for (i, particle) in pf.particles_mut().iter_mut().enumerate() {
            if i == 0 {
                particle.set_weight(0.99);
            } else {
                particle.set_weight(0.01 / 99.0);
            }
        }

        let n_eff = pf.effective_sample_size();

        // N_eff should be much less than N for degenerate distribution
        assert!(n_eff < 10.0, "Expected N_eff < 10, got {}", n_eff);
    }
}

// ============= Velocity-based Particle Filter ===================================

/// Position-only particle for velocity-based navigation
///
/// This particle represents only position states (latitude, longitude, altitude)
/// and is propagated using velocity inputs from an external source (e.g., INS filter).
/// This reduces dimensionality compared to full-state particles while still leveraging
/// accurate velocity information.
///
/// # State Vector
///
/// The particle state is a 3-element vector: `[latitude (rad), longitude (rad), altitude (m)]`
#[derive(Clone, Debug)]
pub struct VelocityParticle {
    /// State vector [lat, lon, alt]
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
            3,
            "VelocityParticle requires 3-state vector [lat, lon, alt]"
        );
        VelocityParticle {
            state: initial_state.clone(),
            weight,
        }
    }

    fn state(&self) -> DVector<f64> {
        self.state.clone()
    }

    fn update_weight<M: crate::measurements::MeasurementModel + ?Sized>(
        &mut self,
        measurement: &M,
    ) {
        // Compute expected measurement from particle's state
        let expected = measurement.get_expected_measurement(&self.state);
        let actual = measurement.get_measurement(&self.state);

        // Measurement residual
        let innovation = actual - expected;

        // Measurement noise covariance
        let noise_cov = measurement.get_noise();

        // Compute Gaussian likelihood: exp(-0.5 * innovation^T * R^{-1} * innovation)
        let mut log_likelihood = 0.0;
        for i in 0..innovation.len() {
            let variance = noise_cov[(i, i)];
            if variance > 0.0 {
                let std_dev = variance.sqrt();
                let normalized_innovation = innovation[i] / std_dev;
                log_likelihood += -0.5 * normalized_innovation.powi(2)
                    - std_dev.ln()
                    - 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }

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
    /// Propagate particle state forward using supplied velocities
    ///
    /// Updates position based on supplied velocity: position += velocity * dt
    /// Adds process noise for particle diversity.
    ///
    /// # Arguments
    ///
    /// * `v_n` - Northward velocity (m/s)
    /// * `v_e` - Eastward velocity (m/s)
    /// * `v_d` - Vertical velocity (m/s, positive down in NED)
    /// * `dt` - Time step in seconds
    /// * `process_noise_std` - Standard deviation of process noise (meters)
    /// * `rng` - Random number generator
    pub fn propagate(
        &mut self,
        v_n: f64,
        v_e: f64,
        v_d: f64,
        dt: f64,
        process_noise_std: f64,
        rng: &mut StdRng,
    ) {
        use rand_distr::{Distribution, Normal};

        // Extract current state
        let lat = self.state[0]; // radians
        let lon = self.state[1]; // radians
        let alt = self.state[2]; // meters

        // Get principal radii at current position
        let lat_deg = lat.to_degrees();
        let (r_n, r_e, _) = crate::earth::principal_radii(&lat_deg, &alt);

        // Position update using supplied velocities
        let delta_lat = (v_n * dt) / r_n;
        let delta_lon = if lat.cos().abs() > 1e-8 {
            (v_e * dt) / (r_e * lat.cos())
        } else {
            0.0
        };
        let delta_alt = -v_d * dt; // Negative because down is positive in NED frame

        // Generate process noise for position
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

        // Ensure latitude is within [-pi/2, pi/2]
        self.state[0] =
            self.state[0].clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);

        // Wrap longitude to [-pi, pi]
        self.state[1] = crate::wrap_to_pi(self.state[1]);
    }
}

/// Velocity input model for particle filter prediction
///
/// Represents 3-axis velocity inputs [v_north, v_east, v_vertical] in m/s
#[derive(Clone, Debug)]
pub struct VelocityInput {
    /// Northward velocity (m/s)
    pub v_north: f64,
    /// Eastward velocity (m/s)
    pub v_east: f64,
    /// Vertical velocity (m/s, positive down in NED)
    pub v_vertical: f64,
}

impl crate::InputModel for VelocityInput {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_dimension(&self) -> usize {
        3
    }

    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.v_north, self.v_east, self.v_vertical])
    }
}

/// Velocity-based particle filter for GPS-aided navigation
///
/// A particle filter that tracks position-only using supplied velocity inputs.
/// Designed for scenarios where velocity is provided by an external source
/// (e.g., an INS/GNSS filter) and we want position estimation with GPS aiding.
///
/// # Features
///
/// - 3-state position representation [lat, lon, alt]
/// - Velocity-driven prediction (velocities supplied as input)
/// - GPS position measurement updates
/// - Configurable process noise for drift rate tuning
///
/// # Example
///
/// ```rust
/// use strapdown::particle::{VelocityParticleFilter, VelocityInput};
/// use strapdown::NavigationFilter;
/// use nalgebra::DVector;
///
/// let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]); // [lat, lon, alt] in rad, rad, m
/// let mut pf = VelocityParticleFilter::new(initial_state, 1000, 1.0);
///
/// // Propagate with velocity input
/// let vel_input = VelocityInput { v_north: 10.0, v_east: 5.0, v_vertical: 0.0 };
/// pf.predict(&vel_input, 0.1); // dt = 0.1s
///
/// // Update with GPS measurement
/// // pf.update(&gps_measurement);
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
    /// Create a new velocity-based particle filter
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Initial state [lat (rad), lon (rad), alt (m)]
    /// * `num_particles` - Number of particles to use
    /// * `process_noise_std` - Standard deviation of process noise in meters
    ///
    /// # Returns
    ///
    /// A new `VelocityParticleFilter` instance
    pub fn new(initial_state: DVector<f64>, num_particles: usize, process_noise_std: f64) -> Self {
        Self::new_with_seed(
            initial_state,
            num_particles,
            process_noise_std,
            rand::random(),
        )
    }

    /// Create a new velocity-based particle filter with a specific random seed
    ///
    /// This is useful for reproducible tests.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Initial state [lat (rad), lon (rad), alt (m)]
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
            3,
            "Initial state must be 3-element vector [lat, lon, alt]"
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
            0.5,
        );

        let rng = StdRng::seed_from_u64(seed);

        VelocityParticleFilter {
            filter,
            process_noise_std,
            rng,
        }
    }

    /// Get effective sample size
    pub fn effective_sample_size(&self) -> f64 {
        self.filter.effective_sample_size()
    }

    /// Get the number of particles
    pub fn num_particles(&self) -> usize {
        self.filter.num_particles()
    }

    /// Get the configured process noise standard deviation
    pub fn process_noise_std(&self) -> f64 {
        self.process_noise_std
    }

    /// Set the process noise standard deviation
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
}

impl crate::NavigationFilter for VelocityParticleFilter {
    /// Predict step: propagate all particles using supplied velocities
    ///
    /// # Arguments
    ///
    /// * `control_input` - VelocityInput containing [v_north, v_east, v_vertical]
    /// * `dt` - Time step in seconds
    fn predict<C: crate::InputModel>(&mut self, control_input: &C, dt: f64) {
        let vel_input = control_input
            .as_any()
            .downcast_ref::<VelocityInput>()
            .expect("VelocityParticleFilter.predict expects a VelocityInput InputModel");

        assert!(dt > 0.0, "Time step must be positive");

        // Propagate each particle
        for particle in self.filter.particles_mut() {
            particle.propagate(
                vel_input.v_north,
                vel_input.v_east,
                vel_input.v_vertical,
                dt,
                self.process_noise_std,
                &mut self.rng,
            );
        }
    }

    /// Update particle weights based on measurement
    ///
    /// # Arguments
    ///
    /// * `measurement` - Measurement model (e.g., GPS position)
    fn update<M: crate::measurements::MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        for particle in self.filter.particles_mut() {
            particle.update_weight(measurement);
        }
        self.normalize_weights();
        self.resample_if_needed();
    }

    /// Get the current state estimate (weighted mean)
    ///
    /// Returns state vector [lat (rad), lon (rad), alt (m)]
    fn get_estimate(&self) -> DVector<f64> {
        self.filter.get_estimate()
    }

    /// Get the state covariance estimate
    ///
    /// Returns 3x3 covariance matrix
    fn get_certainty(&self) -> DMatrix<f64> {
        self.filter.get_certainty()
    }
}
