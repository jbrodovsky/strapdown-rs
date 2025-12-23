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
//!         let actual = measurement.get_vector();
//!         let diff = actual - expected;
//!         
//!         // Simple Gaussian likelihood (customize for your application)
//!         let likelihood = (-0.5 * diff.norm_squared()).exp();
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
use rand::prelude::*;
use rand::{thread_rng, SeedableRng};

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
    resampling_threshold: f64,
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
            resampling_threshold,
            rng: StdRng::seed_from_u64(thread_rng().random()),
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
        let sum_squared_weights: f64 = self
            .particles
            .iter()
            .map(|p| p.weight().powi(2))
            .sum();
        
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
            ParticleResamplingStrategy::Residual => {
                self.residual_resample(&weights, num_particles)
            }
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
            let idx = cumsum.iter().position(|&c| c >= u).unwrap_or(weights.len() - 1);
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
        assert!(!self.particles.is_empty(), "Cannot compute state estimate from empty particle set");
        
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
                let max_particle = self.particles
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
        assert!(!self.particles.is_empty(), "Cannot compute covariance from empty particle set");
        
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
            let actual = measurement.get_vector();
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
