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
//! # Library Design Philosophy
//!
//! This module is designed as a **template-style library** that provides building blocks for
//! implementing custom particle filter-based navigation systems. Since there is no canonical
//! form of a particle filter-based INS, we provide generic components that users can adapt:
//!
//! ## Key Components
//!
//! - [`Particle`]: A trait defining the required methods for particle state representation
//! - [`ParticleFilter`]: A trait for marking navigation systems as particle filter-based
//! - **Standalone resampling functions**: Generic algorithms that operate on particle weights
//!   - [`multinomial_resample`]: Independent sampling from weighted distribution
//!   - [`systematic_resample`]: Deterministic selection with random offset
//!   - [`stratified_resample`]: Stratified sampling for lower variance
//!   - [`residual_resample`]: Hybrid deterministic/random sampling

use std::any::Any;

use nalgebra::DVector;
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

    /// Sets the state vector of the particle
    fn set_state(&mut self, state: DVector<f64>);

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

// ============= Generic Resampling Functions ===================================

/// Multinomial resampling algorithm
///
/// Draws N independent samples from the weighted particle distribution.
/// This is the simplest resampling method but has highest variance.
///
/// # Arguments
/// * `weights` - Non-negative weights for each particle. Callers are responsible
///   for ensuring the weights sum to 1.0 before calling this function.
/// * `num_samples` - Number of samples to draw
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of indices representing which particles to keep/duplicate
///
/// # Example
/// ```
/// use strapdown::particle::multinomial_resample;
/// use rand::SeedableRng;
///
/// let weights = vec![0.1, 0.3, 0.4, 0.2];
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let indices = multinomial_resample(&weights, 4, &mut rng);
/// assert_eq!(indices.len(), 4);
/// ```
pub fn multinomial_resample<R: Rng>(
    weights: &[f64],
    num_samples: usize,
    rng: &mut R,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(num_samples);
    let cumsum: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, &w| {
            *acc += w;
            Some(*acc)
        })
        .collect();

    for _ in 0..num_samples {
        let u: f64 = rng.random();
        let idx = cumsum
            .iter()
            .position(|&c| c >= u)
            .unwrap_or(weights.len() - 1);
        indices.push(idx);
    }

    indices
}

/// Systematic resampling algorithm
///
/// More efficient and lower variance than multinomial resampling.
/// Uses a single random number and deterministic spacing to select particles.
///
/// # Arguments
/// * `weights` - Non-negative weights for each particle. Callers are responsible
///   for ensuring the weights sum to 1.0 before calling this function.
/// * `num_samples` - Number of samples to draw
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of indices representing which particles to keep/duplicate
///
/// # Example
/// ```
/// use strapdown::particle::systematic_resample;
/// use rand::SeedableRng;
///
/// let weights = vec![0.1, 0.3, 0.4, 0.2];
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let indices = systematic_resample(&weights, 4, &mut rng);
/// assert_eq!(indices.len(), 4);
/// ```
pub fn systematic_resample<R: Rng>(weights: &[f64], num_samples: usize, rng: &mut R) -> Vec<usize> {
    let mut indices = Vec::with_capacity(num_samples);
    let step = 1.0 / num_samples as f64;
    let start: f64 = rng.random::<f64>() * step;

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

/// Stratified resampling algorithm
///
/// Divides [0,1] into N equal strata and samples once per stratum.
/// Provides lower variance than multinomial resampling while maintaining
/// randomness compared to systematic resampling.
///
/// # Arguments
/// * `weights` - Non-negative weights for each particle. Callers are responsible
///   for ensuring the weights sum to 1.0 before calling this function.
/// * `num_samples` - Number of samples to draw
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of indices representing which particles to keep/duplicate
///
/// # Example
/// ```
/// use strapdown::particle::stratified_resample;
/// use rand::SeedableRng;
///
/// let weights = vec![0.1, 0.3, 0.4, 0.2];
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let indices = stratified_resample(&weights, 4, &mut rng);
/// assert_eq!(indices.len(), 4);
/// ```
pub fn stratified_resample<R: Rng>(weights: &[f64], num_samples: usize, rng: &mut R) -> Vec<usize> {
    let mut indices = Vec::with_capacity(num_samples);
    let step = 1.0 / num_samples as f64;

    let mut cumsum = 0.0;
    let mut i = 0;

    for j in 0..num_samples {
        let u = (j as f64 + rng.random::<f64>()) * step;

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

/// Residual resampling algorithm
///
/// Deterministically selects particles based on integer parts of N*weights,
/// then randomly samples remaining particles. This minimizes variance while
/// ensuring particles with high weights are always included.
///
/// # Arguments
/// * `weights` - Non-negative weights for each particle. Callers are responsible
///   for ensuring the weights sum to 1.0 before calling this function.
/// * `num_samples` - Number of samples to draw
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of indices representing which particles to keep/duplicate
///
/// # Example
/// ```
/// use strapdown::particle::residual_resample;
/// use rand::SeedableRng;
///
/// let weights = vec![0.1, 0.3, 0.4, 0.2];
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let indices = residual_resample(&weights, 4, &mut rng);
/// assert_eq!(indices.len(), 4);
/// ```
pub fn residual_resample<R: Rng>(weights: &[f64], num_samples: usize, rng: &mut R) -> Vec<usize> {
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
        let start: f64 = rng.random::<f64>() * step;

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

// ============= ParticleFilter Trait ==========================================

/// Trait for marking a navigation system as particle filter-based
///
/// This trait provides a common interface for all particle filter implementations.
/// It is designed to be a thin wrapper around the generic resampling functions
/// provided by this module.
pub trait ParticleFilter {
    /// Resample particles based on their weights
    ///
    /// This method should implement resampling logic to combat particle degeneracy.
    /// Implementations should use the generic resampling functions provided by this
    /// module (systematic_resample, multinomial_resample, etc.) as appropriate.
    ///
    /// Note: Weights should be normalized before resampling so that they
    /// sum to 1.0. Call weight normalization first if needed.
    fn resample(&mut self);
}

// ==================== Unit Tests =============================================
#[cfg(test)]
mod tests {
    use super::*;
    
    
    

    // ============= Tests for standalone resampling functions ==================

    #[test]
    fn test_standalone_multinomial_resample() {
        let weights = vec![0.1, 0.3, 0.4, 0.2];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let indices = multinomial_resample(&weights, 100, &mut rng);

        assert_eq!(indices.len(), 100);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < weights.len());
        }

        // With 100 samples, distribution should roughly match weights
        let mut counts = vec![0; weights.len()];
        for &idx in &indices {
            counts[idx] += 1;
        }

        // Check that higher weight particles are sampled more often
        assert!(counts[2] > counts[0]); // 0.4 > 0.1
        assert!(counts[1] > counts[0]); // 0.3 > 0.1
    }

    #[test]
    fn test_standalone_systematic_resample() {
        let weights = vec![0.25, 0.25, 0.25, 0.25]; // Uniform weights
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let indices = systematic_resample(&weights, 100, &mut rng);

        assert_eq!(indices.len(), 100);

        // With uniform weights, each particle should be selected roughly equally
        let mut counts = vec![0; weights.len()];
        for &idx in &indices {
            counts[idx] += 1;
        }

        // Each should be selected about 25 times
        for count in counts {
            let diff = if count > 25 { count - 25 } else { 25 - count };
            assert!(diff <= 5); // Allow some variance
        }
    }

    #[test]
    fn test_standalone_stratified_resample() {
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let indices = stratified_resample(&weights, 50, &mut rng);

        assert_eq!(indices.len(), 50);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < weights.len());
        }
    }

    #[test]
    fn test_standalone_residual_resample() {
        let weights = vec![0.1, 0.3, 0.4, 0.2];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let indices = residual_resample(&weights, 10, &mut rng);

        assert_eq!(indices.len(), 10);

        // Count occurrences
        let mut counts = vec![0; weights.len()];
        for &idx in &indices {
            counts[idx] += 1;
        }

        // Particle with weight 0.4 should appear at least 4 times (deterministic part)
        // Particle with weight 0.3 should appear at least 3 times
        assert!(counts[2] >= 4); // 0.4 * 10 = 4.0
        assert!(counts[1] >= 3); // 0.3 * 10 = 3.0
    }

    #[test]
    fn test_resampling_functions_preserve_particle_count() {
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let num_samples = 100;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Test all resampling methods
        let multinomial_indices = multinomial_resample(&weights, num_samples, &mut rng);
        assert_eq!(multinomial_indices.len(), num_samples);

        let systematic_indices = systematic_resample(&weights, num_samples, &mut rng);
        assert_eq!(systematic_indices.len(), num_samples);

        let stratified_indices = stratified_resample(&weights, num_samples, &mut rng);
        assert_eq!(stratified_indices.len(), num_samples);

        let residual_indices = residual_resample(&weights, num_samples, &mut rng);
        assert_eq!(residual_indices.len(), num_samples);
    }
}
