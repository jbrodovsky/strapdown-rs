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
//! - [`BasicParticleFilter`]: A complete implementation using the library components
//! - [`ParticleAveragingStrategy`]: Methods for extracting state estimates from particles
//! - [`ParticleResamplingStrategy`]: Configuration for resampling algorithms

use std::any::Any;
use std::vec;

use crate::{StrapdownState, NavigationFilter, forward, IMUData};
use crate::measurements::MeasurementModel;

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use rand::SeedableRng;
use rand::prelude::*;
use rand_distr::Normal;


/// Default particle filter process noise and covariance for MEMS-grade IMU
///
/// This constant provides typical jitter/noise levels for a particle filter using
/// consumer-grade MEMS IMU data. Position noise is specified in radians for lat/lon
/// (approximately 10m horizontal uncertainty at Earth's surface).
/// 
/// Position noise calculation: 10m * 9e-6 deg/m ≈ 9e-5 degrees, convert to radians and square
pub const MEMS_PF_JITTER: [f64; 15] =[
    2.4685e-12, // lat variance (rad²) ≈ (10m * 9e-6 deg/m * π/180)²
    2.4685e-12, // lon variance (rad²)
    1e-3, // alt variance (m²)
    1e-6, 1e-6, 1e-6, // Velocity noise (vn, ve, vv)
    1e-5, 1e-5, 1e-5, // Attitude noise (roll, pitch, yaw)
    1e-7, 1e-7, 1e-7, // Accelerometer bias noise (bx, by, bz)
    1e-8, 1e-8, 1e-8, // Gyroscope bias noise (bx, by, bz)
];

/// Common particle-filter default vectors and initialization covariances for reuse
///
/// All arrays are in the same 15-state ordering used by the particle filter:
/// [lat, lon, alt, v_n, v_e, v_v, roll, pitch, yaw, abx, aby, abz, gbx, gby, gbz]
pub mod pf_defaults {
    //! Exposed defaults for process noise (variance) and example initialization covariances.
    //! Units: lat/lon in radians (variance), alt in meters², velocities in (m/s)²,
    //! attitudes in rad², biases in their native squared units.

    /// MEMS-grade default jitter (re-export of `MEMS_PF_JITTER` for convenience)
    pub const MEMS_PF_JITTER: [f64; 15] = super::MEMS_PF_JITTER;

    /// Very small process noise used for synthetic/perfect-IMU tests (variance)
    pub const PROCESS_NOISE_PERFECT_IMU: [f64; 15] = [
        1e-12, 1e-12, 1e-12, // lat, lon, alt
        1e-8, 1e-8, 1e-8,    // v_n, v_e, v_v
        1e-10, 1e-10, 1e-10, // roll, pitch, yaw
        1e-12, 1e-12, 1e-12, // accel bias
        1e-12, 1e-12, 1e-12, // gyro bias
    ];

    /// Small nonzero process noise used in stationary/eastward tests (variance)
    pub const PROCESS_NOISE_SMALL: [f64; 15] = [
        1e-12, 1e-12, 1e-8,  // lat, lon, alt
        1e-8, 1e-8, 1e-8,    // velocities
        1e-8, 1e-8, 1e-8,    // attitude
        1e-10, 1e-10, 1e-10, // accel bias
        1e-10, 1e-10, 1e-10, // gyro bias
    ];

    /// Example initialization covariance for a moving vehicle test (variance)
    /// Horizontal position given as variance corresponding to ~5 m sigma.
    pub const INIT_COV_MOVING: [f64; 15] = [
        // lat/lon: (5 m -> degrees -> radians)² approximated by using METERS_TO_DEGREES elsewhere
        2.4685e-12 / 4.0, 2.4685e-12 / 4.0, // lat, lon (approx 5 m)
        25.0,    // alt variance (5 m sigma)
        0.1, 0.1, 1e-8, // v_n, v_e, v_v (tight vertical)
        0.01, 0.01, 0.01, // attitude
        0.01, 0.01, 0.01, // accel bias
        0.001, 0.001, 0.001, // gyro bias
    ];

    /// Example initialization covariance for a stationary test (variance)
    pub const INIT_COV_STATIONARY: [f64; 15] = [
        2.4685e-12, 2.4685e-12, 25.0, // lat, lon (~10 m), alt (5 m sigma)
        1e-8, 1e-8, 1e-8, // very tight velocities
        0.01, 0.01, 0.01, // attitude
        0.01, 0.01, 0.01, // accel bias
        0.001, 0.001, 0.001, // gyro bias
    ];
}

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
#[derive(Clone, Copy, Debug, Default)]
pub struct StrapdownParticle {
    state: StrapdownState,
    accel_bias: Vector3<f64>,
    gyro_bias: Vector3<f64>,
    weight: f64,
}

impl Particle for StrapdownParticle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn new(initial_state: &DVector<f64>, weight: f64) -> Self
    where
        Self: Sized,
    {
        let state = StrapdownState {
            latitude: initial_state[0],
            longitude: initial_state[1],
            altitude: initial_state[2],
            velocity_north: initial_state[3],
            velocity_east: initial_state[4],
            velocity_vertical: initial_state[5],
            attitude: Rotation3::from_euler_angles(
                initial_state[6],
                initial_state[7],
                initial_state[8],
            ),
            is_enu: true,
        };
        StrapdownParticle {
            state,
            accel_bias: Vector3::new(initial_state[9], initial_state[10], initial_state[11]),
            gyro_bias: Vector3::new(initial_state[12], initial_state[13], initial_state[14]),
            weight,
        }
    }

    fn state(&self) -> DVector<f64> {
        DVector::from_vec(
            vec![
                self.state.latitude,
                self.state.longitude,
                self.state.altitude,
                self.state.velocity_north,
                self.state.velocity_east,
                self.state.velocity_vertical,
                self.state.attitude.euler_angles().0,
                self.state.attitude.euler_angles().1,
                self.state.attitude.euler_angles().2,
                self.accel_bias[0],
                self.accel_bias[1],
                self.accel_bias[2],
                self.gyro_bias[0],
                self.gyro_bias[1],
                self.gyro_bias[2],
            ]
        )
    }

    fn set_state(&mut self, state: DVector<f64>) {
        self.state.latitude = state[0];
        self.state.longitude = state[1];
        self.state.altitude = state[2];
        self.state.velocity_north = state[3];
        self.state.velocity_east = state[4];
        self.state.velocity_vertical = state[5];
        self.state.attitude = Rotation3::from_euler_angles(state[6], state[7], state[8]);
        self.accel_bias = Vector3::new(state[9], state[10], state[11]);
        self.gyro_bias = Vector3::new(state[12], state[13], state[14]);
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}

pub struct StrapdownParticleFilter {
    /// The particles in the filter
    particles: Vec<StrapdownParticle>,
    /// number of particles
    _n: usize,
    /// Random number generator for resampling
    rng: rand::rngs::StdRng,
    /// Process noise to apply during prediction and proposal distribution sampling
    process_noise: DVector<f64>,

}

impl StrapdownParticleFilter {
    pub fn new(
        particles: Vec<StrapdownParticle>,
        process_noise: DVector<f64>,
        seed: u64,
    ) -> Self {
        StrapdownParticleFilter {
            particles: particles.clone(),
            _n: particles.len(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            process_noise,
        }
    }

    pub fn new_about(
        mean: DVector<f64>,
        covariance: DVector<f64>,
        process_noise: DVector<f64>,
        num_particles: usize,
        seed: u64,
    ) -> Self {
        assert!(mean.len() == 15, "Mean vector must be of length 15, received {}", mean.len());
        assert!(covariance.len() == 15, "Covariance vector must be of length 15, received {}", covariance.len());

        let mut particles = Vec::with_capacity(num_particles);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        let noise_dist: Vec<Normal<f64>> = covariance
            .iter()
            .map(|&variance| Normal::new(0.0, variance.sqrt()).unwrap())
            .collect();
       
        for _ in 0..num_particles {
            let particle_state = &mean + DVector::from_iterator(
                15,
                noise_dist.iter().map(|dist| dist.sample(&mut rng))
            );
            particles.push(StrapdownParticle::new(&particle_state.into(), 1.0 / num_particles as f64));
        }

        StrapdownParticleFilter {
            particles,
            _n: num_particles,
            rng,
            process_noise,
        }
    }

    pub fn particles_mut(&mut self) -> &mut Vec<StrapdownParticle> {
        &mut self.particles
    }

    pub fn particles(&self) -> &Vec<StrapdownParticle> {
        &self.particles
    }
}

impl ParticleFilter for StrapdownParticleFilter {
    fn resample(&mut self) {
        // Extract and normalize weights
        let mut weights: Vec<f64> = self.particles.iter().map(|p| p.weight()).collect();
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }
        let indices = systematic_resample(&weights, self.particles.len(), &mut self.rng);

        let new_particles: Vec<StrapdownParticle> = indices
            .iter()
            .map(|&idx| {
                let state = self.particles[idx].state();
                StrapdownParticle::new(&state, 1.0 / self.particles.len() as f64)
            })
            .collect();
        self.particles = new_particles;
    }
}

impl NavigationFilter for StrapdownParticleFilter {
    fn predict<C: crate::InputModel>(&mut self, control_input: &C, dt: f64) {
        // Downcast control input into imu data
        let imu_data = control_input
            .as_any()
            .downcast_ref::<crate::IMUData>()
            .expect("Control input is not of type IMUData");

        for particle in &mut self.particles {
            // Apply bias correction to IMU measurements
            let corrected_imu = IMUData {
                accel: imu_data.accel - particle.accel_bias,
                gyro: imu_data.gyro - particle.gyro_bias,
            };

            // Use the tested forward() function to propagate particle state
            // This ensures consistency with the rest of the codebase
            forward(&mut particle.state, corrected_imu, dt);

            // Add process noise to the state
            // Create Normal distributions for each state component's process noise
            if self.process_noise.iter().any(|&v| v > 0.0) {
                let jitter_dists: Vec<Normal<f64>> = self.process_noise
                    .iter()
                    .map(|&variance| Normal::new(0.0, variance.sqrt()).unwrap())
                    .collect();

                let noise: DVector<f64> = DVector::from_iterator(
                    15,
                    jitter_dists.iter().map(|dist| dist.sample(&mut self.rng))
                );

                // Get current state, add noise, and set back
                let mut noisy_state = particle.state();
                noisy_state += noise;
                particle.set_state(noisy_state);
            }
        }
    }
    /// Update particle weights based on measurement likelihoods using multivariate Gaussian PDF
    /// 
    /// # Arguments
    /// * `measurement` - Measurement model to use for weight updates
    /// 
    /// # Note
    /// Innovation is computed as (actual measurement - expected measurement from particle state)
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // Get measurement noise covariance (already in correct units: radians² for lat/lon)
        let noise_cov = measurement.get_noise();
        let noise_cov_det = noise_cov.determinant();
        let noise_cov_inv = noise_cov.try_inverse().expect("Noise covariance matrix is not invertible");
        
        // Precompute normalization constant for multivariate Gaussian PDF
        let dim = measurement.get_dimension() as f64;
        let normalization = 1.0 / ((2.0 * std::f64::consts::PI).powf(dim / 2.0) * noise_cov_det.sqrt());
        
        // Update weights with measurement likelihoods
        for particle in &mut self.particles {
            // Compute innovation: (actual measurement - expected measurement)
            // Both are in the same units (radians for lat/lon, meters for alt)
            let z = measurement.get_measurement(&particle.state());
            let h = measurement.get_expected_measurement(&particle.state());
            let innovation = z - h;
            
            // Compute Mahalanobis distance: innovation^T * Σ^(-1) * innovation
            let mahalanobis = (innovation.transpose() * &noise_cov_inv * &innovation)[(0, 0)];
            
            // Compute likelihood using multivariate Gaussian PDF
            // p(z|x) = (1 / sqrt((2π)^k * |Σ|)) * exp(-0.5 * mahalanobis)
            let likelihood = normalization * (-0.5 * mahalanobis).exp();
            
            // Bayesian weight update: w = w * p(z|x)
            particle.set_weight(likelihood * particle.weight());
        }
        
        // Normalize weights to sum to 1.0
        let weight_sum: f64 = self.particles.iter().map(|p| p.weight()).sum();
        if weight_sum > 0.0 {
            for particle in &mut self.particles {
                particle.set_weight(particle.weight() / weight_sum);
            }
        } else {
            // If all weights are zero (numerical underflow), reset to uniform
            let uniform_weight = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.set_weight(uniform_weight);
            }
        }
    } 

    fn get_certainty(&self) -> DMatrix<f64> {
        let mut covariance = DMatrix::zeros(15, 15);
        let estimate = self.get_estimate();

        for particle in &self.particles {
            let nav_state = particle.state();
            let diff = &nav_state - &estimate;
            let weight = particle.weight();
            covariance += weight * (&diff * diff.transpose());
        }
        covariance        
    }

    fn get_estimate(&self) -> DVector<f64> {
        let mut estimate = DVector::zeros(15);
        let mut _total_weight = 0.0;

        for particle in &self.particles {
            let weight = particle.weight();
            let nav_state = particle.state();
            estimate += nav_state * weight;
            _total_weight += weight;
        }
        estimate        
    }
}

// ==================== Unit Tests =============================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurements::GPSPositionMeasurement;
    use crate::{IMUData, StrapdownState, earth, generate_scenario_data};
    use nalgebra::Rotation3;

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

    // ============= Unit Tests for StrapdownParticleFilter ======================

    #[test]
    fn test_strapdown_particle_filter_initialization() {
        let initial_mean = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            100,
            42,
        );

        assert_eq!(filter.particles().len(), 100);
        
        // Check that initial weights sum to 1.0
        let weight_sum: f64 = filter.particles().iter().map(|p| p.weight()).sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
        
        // Check that all particles have equal weight
        for particle in filter.particles() {
            assert!((particle.weight() - 0.01).abs() < 1e-10);
        }
    }

    #[test]
    fn test_strapdown_particle_filter_particles_distributed_around_mean() {
        let lat_mean = 40.0_f64.to_radians();
        let lon_mean = -105.0_f64.to_radians();
        let alt_mean = 1000.0;
        
        let initial_mean = DVector::from_vec(vec![
            lat_mean, lon_mean, alt_mean,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            1000,
            42,
        );

        // Calculate empirical mean from particles
        let mut lat_sum = 0.0;
        let mut lon_sum = 0.0;
        let mut alt_sum = 0.0;
        
        for particle in filter.particles() {
            let state = particle.state();
            lat_sum += state[0];
            lon_sum += state[1];
            alt_sum += state[2];
        }
        
        let lat_empirical = lat_sum / 1000.0;
        let lon_empirical = lon_sum / 1000.0;
        let alt_empirical = alt_sum / 1000.0;
        
        // Empirical mean should be close to the specified mean
        assert!((lat_empirical - lat_mean).abs() < 1e-4);
        assert!((lon_empirical - lon_mean).abs() < 1e-4);
        assert!((alt_empirical - alt_mean).abs() < 10.0);
    }

    #[test]
    fn test_strapdown_particle_filter_predict_step() {
        let initial_mean = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let mut filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            50,
            42,
        );

        let initial_estimate = filter.get_estimate();
        
        // Apply a prediction step with zero IMU input
        let imu = IMUData {
            accel: Vector3::zeros(),
            gyro: Vector3::zeros(),
        };
        
        filter.predict(&imu, 1.0);
        
        let after_predict = filter.get_estimate();
        
        // State should have changed due to process noise
        assert_ne!(initial_estimate, after_predict);
    }

    #[test]
    fn test_strapdown_particle_filter_update_step() {
        let lat = 40.0_f64.to_radians();
        let lon = -105.0_f64.to_radians();
        let alt = 1000.0;
        
        let initial_mean = DVector::from_vec(vec![
            lat, lon, alt,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let mut filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            50,
            42,
        );
        
        let weights_0 = filter.particles().iter().map(|p| p.weight()).collect::<Vec<f64>>();
        // All weights should be equal initially
        let first_weight = weights_0[0];
        let all_same = weights_0.iter().all(|&w| (w - first_weight).abs() < 1e-10);
        assert!(all_same, "Initial weights should all be equal. Initial weights: {:?}", weights_0);
        // Create a GPS measurement
        let gps = GPSPositionMeasurement {
            latitude: lat.to_degrees(),
            longitude: lon.to_degrees(),
            altitude: alt,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };        
        // Apply update
        filter.update(&gps);
        
        // Check that weights have been updated (should no longer all be equal)
        let weights: Vec<f64> = filter.particles().iter().map(|p| p.weight()).collect();
        let first_weight = weights[0];
        let all_same = weights.iter().all(|&w| (w - first_weight).abs() < 1e-10);

        assert!(!all_same, "Weights should have been updated and differ from each other. Updated Weights: {:?}", weights);
    }

    #[test]
    fn test_strapdown_particle_filter_resample() {
        let initial_mean = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let mut filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            50,
            42,
        );

        // Manually set non-uniform weights
        {
            let particles = filter.particles_mut();
            let num_particles = particles.len();
            particles[0].set_weight(0.5);
            particles[1].set_weight(0.3);
            for i in 2..num_particles {
                particles[i].set_weight(0.2 / (num_particles - 2) as f64);
            }
        }
        
        // Resample
        filter.resample();
        
        // After resampling, all weights should be equal again
        let weight_sum: f64 = filter.particles().iter().map(|p| p.weight()).sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
        
        for particle in filter.particles() {
            assert!((particle.weight() - 1.0 / 50.0).abs() < 1e-10);
        }
        
        // Particle count should be preserved
        assert_eq!(filter.particles().len(), 50);
    }

    #[test]
    fn test_strapdown_particle_filter_get_estimate() {
        let lat = 40.0_f64.to_radians();
        let lon = -105.0_f64.to_radians();
        let alt = 1000.0;
        
        let initial_mean = DVector::from_vec(vec![
            lat, lon, alt,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(vec![
            1e-10, 1e-10, 1e-10, // Very tight initial distribution
            1e-10, 1e-10, 1e-10,
            1e-10, 1e-10, 1e-10,
            1e-10, 1e-10, 1e-10,
            1e-10, 1e-10, 1e-10,
        ]);
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let filter = StrapdownParticleFilter::new_about(
            initial_mean.clone(),
            initial_covariance,
            process_noise,
            100,
            42,
        );

        let estimate = filter.get_estimate();
        
        // With very tight distribution, estimate should be very close to mean
        for i in 0..15 {
            assert!((estimate[i] - initial_mean[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_strapdown_particle_filter_get_certainty() {
        let initial_mean = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            100,
            42,
        );

        let covariance = filter.get_certainty();
        
        // Covariance should be a 15x15 matrix
        assert_eq!(covariance.nrows(), 15);
        assert_eq!(covariance.ncols(), 15);
        
        // Diagonal elements should be non-negative (variances)
        for i in 0..15 {
            assert!(covariance[(i, i)] >= 0.0);
        }
        
        // Covariance should be symmetric
        for i in 0..15 {
            for j in 0..15 {
                assert!((covariance[(i, j)] - covariance[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_strapdown_particle_filter_full_cycle() {
        let initial_state = StrapdownState {
            latitude: 40.0_f64.to_radians(),
            longitude: -105.0_f64.to_radians(),
            altitude: 1000.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let initial_mean = DVector::from_vec(vec![
            initial_state.latitude,
            initial_state.longitude,
            initial_state.altitude,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        // Use much smaller process noise for stationary scenario with perfect measurements
        let process_noise = DVector::from_vec(vec![
            1e-10, 1e-10, 1e-6, // Very small position noise
            1e-10, 1e-10, 1e-10, // Very small velocity noise  
            1e-8, 1e-8, 1e-8,   // Small attitude noise
            1e-10, 1e-10, 1e-10, // Very small accel bias noise
            1e-10, 1e-10, 1e-10, // Very small gyro bias noise
        ]);

        let mut filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            200, // Use more particles for better statistics
            42,
        );

        let gravity = earth::gravity(&initial_state.latitude, &initial_state.altitude);
        
        // Run a few iterations of predict-update-resample cycle
        for i in 0..10 {
            // For a stationary vehicle in ENU frame, the accelerometer senses the reaction force
            // from the ground, which is +gravity in the up direction (opposing gravity's pull)
            let imu = IMUData {
                accel: Vector3::new(0.0, 0.0, gravity), // Positive in ENU up direction
                gyro: Vector3::zeros(),
            };
            
            let gps = GPSPositionMeasurement {
                latitude: initial_state.latitude,
                longitude: initial_state.longitude,
                altitude: initial_state.altitude,
                horizontal_noise_std: 5.0 * earth::METERS_TO_DEGREES,
                vertical_noise_std: 2.0,
            };
            
            filter.predict(&imu, 1.0);
            filter.update(&gps);
            filter.resample();
            
            if i % 3 == 0 {
                let est = filter.get_estimate();
                eprintln!("Iteration {}: lat={:.6}, lon={:.6}, alt={:.2}", 
                         i, est[0], est[1], est[2]);
            }
        }
        
        // Filter should still have correct number of particles
        assert_eq!(filter.particles().len(), 200);
        
        // Weights should still sum to 1.0
        let weight_sum: f64 = filter.particles().iter().map(|p| p.weight()).sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
        
        // With corrected weight handling and small process noise, estimate should be accurate
        let estimate = filter.get_estimate();
        eprintln!("Final estimate: lat={:.6} (error={:.6}), lon={:.6} (error={:.6}), alt={:.2} (error={:.2})",
                 estimate[0], estimate[0] - initial_state.latitude,
                 estimate[1], estimate[1] - initial_state.longitude,
                 estimate[2], estimate[2] - initial_state.altitude);
        
        // Verify we get a reasonable estimate (not NaN or infinity)
        for i in 0..15 {
            assert!(estimate[i].is_finite(), "Estimate element {} is not finite: {}", i, estimate[i]);
        }
        
        // With proper weight normalization and small process noise, accuracy should be excellent
        // Note: Some small drift is expected due to GPS measurement noise and limited particles
        assert!((estimate[0] - initial_state.latitude).abs() < 1e-3, 
               "Latitude error too large: {} rad ({} m)", 
               (estimate[0] - initial_state.latitude).abs(),
               (estimate[0] - initial_state.latitude).abs() * 6371000.0);
        assert!((estimate[1] - initial_state.longitude).abs() < 1e-3,
               "Longitude error too large: {} rad", 
               (estimate[1] - initial_state.longitude).abs());
        assert!((estimate[2] - initial_state.altitude).abs() < 5.0,
               "Altitude error too large: {} m", 
               (estimate[2] - initial_state.altitude).abs());
        
        // Verify covariance is also finite
        let covariance = filter.get_certainty();
        for i in 0..15 {
            for j in 0..15 {
                assert!(covariance[(i, j)].is_finite(), 
                       "Covariance element ({}, {}) is not finite", i, j);
            }
        }
    }

    #[test]
    fn test_strapdown_particle_new_constructor() {
        // Test the direct constructor with pre-made particles
        let mut particles = Vec::new();
        let initial_state = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        
        for _ in 0..100 {
            particles.push(StrapdownParticle::new(&initial_state, 0.01));
        }
        
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let filter = StrapdownParticleFilter::new(particles, process_noise, 42);
        
        assert_eq!(filter.particles().len(), 100);
    }

    #[test]
    #[should_panic(expected = "Mean vector must be of length 15")]
    fn test_strapdown_particle_filter_wrong_mean_size() {
        let initial_mean = DVector::from_vec(vec![1.0, 2.0, 3.0]); // Wrong size
        let initial_covariance = DVector::from_vec(MEMS_PF_JITTER.to_vec());
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let _filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            50,
            42,
        );
    }

    #[test]
    #[should_panic(expected = "Covariance vector must be of length 15")]
    fn test_strapdown_particle_filter_wrong_covariance_size() {
        let initial_mean = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let initial_covariance = DVector::from_vec(vec![1.0, 2.0, 3.0]); // Wrong size
        let process_noise = DVector::from_vec(MEMS_PF_JITTER.to_vec());

        let _filter = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            50,
            42,
        );
    }

    #[test]
    fn test_strapdown_particle_set_and_get_state() {
        let initial_state = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        
        let mut particle = StrapdownParticle::new(&initial_state, 1.0);
        let retrieved_state = particle.state();
        
        // States should match
        for i in 0..15 {
            assert!((retrieved_state[i] - initial_state[i]).abs() < 1e-10);
        }
        
        // Modify state
        let new_state = DVector::from_vec(vec![
            41.0_f64.to_radians(),
            -104.0_f64.to_radians(),
            2000.0,
            1.0, 2.0, 3.0,
            0.1, 0.2, 0.3,
            0.01, 0.02, 0.03,
            0.001, 0.002, 0.003,
        ]);
        
        particle.set_state(new_state.clone());
        let retrieved_new_state = particle.state();
        
        // Modified state should match
        for i in 0..15 {
            assert!((retrieved_new_state[i] - new_state[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_strapdown_particle_weight_operations() {
        let initial_state = DVector::from_vec(vec![
            40.0_f64.to_radians(),
            -105.0_f64.to_radians(),
            1000.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        
        let mut particle = StrapdownParticle::new(&initial_state, 0.5);
        
        assert_eq!(particle.weight(), 0.5);
        
        particle.set_weight(0.7);
        assert_eq!(particle.weight(), 0.7);
    }

    // ============= Scenario Tests for StrapdownParticleFilter ==================

    /// Test particle filter tracking a stationary vehicle (geosynchronous)
    ///
    /// # Test Rationale
    ///
    /// This test demonstrates particle filter performance on a stationary target with:
    /// - Perfect IMU data (sensing gravity, no rotation)
    /// - GPS position + horizontal velocity measurements (NO vertical velocity)
    /// - Short 10-second duration to limit drift from unobserved vertical velocity
    ///
    /// ## Why Altitude Drifts
    ///
    /// Even with perfect IMU and zero initial velocity uncertainty, altitude tracking degrades because:
    /// 1. **GPS doesn't measure vertical velocity** - the measurement model only observes
    ///    `[lat, lon, alt, v_north, v_east]`, not `v_vertical`
    /// 2. **Nonlinear dynamics** - particles starting at slightly different altitudes experience
    ///    slightly different gravity (varies ~3 µm/s² per meter), causing velocity divergence
    /// 3. **No observability** - without vertical velocity measurements, the filter cannot
    ///    correct accumulated vertical velocity errors
    ///
    /// ## Solutions for Production Use
    ///
    /// - Use barometric altitude for smoother vertical tracking (but still no velocity)
    /// - Add vertical velocity measurements (from GPS Doppler, if available)
    /// - Use UKF/EKF which better handles partially observed states
    /// - Tighten vertical process noise to constrain drift (at cost of slower response)
    /// - Accept limited vertical accuracy and rely on frequent GPS updates
    #[test]
    fn test_particle_filter_stationary_scenario() {
        use crate::{generate_scenario_data, StrapdownState};
        use crate::measurements::GPSPositionAndVelocityMeasurement;
        use nalgebra::{Rotation3, Vector3};

        // Stationary vehicle at 40°N, 105°W
        let initial_state = StrapdownState {
            latitude: 40.0_f64.to_radians(),
            longitude: -105.0_f64.to_radians(),
            altitude: 1000.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let g = earth::gravity(&40.0, &1000.0);
        let accel_body = Vector3::new(0.0, 0.0, g);
        let gyro_body = Vector3::zeros();

        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            10, // 10 seconds (short duration to limit drift from unobserved vertical velocity)
            1,  // 1 Hz
            accel_body,
            gyro_body,
            true,  // Geosynchronous
            true,  // Constant velocity (zero)
            false  // GPS output is always in degrees, no longer used
        );

        // Convert GPS position measurements to position+velocity measurements
        // Note: GPS does NOT measure vertical velocity, so altitude will drift over time
        let gps_pos_vel_measurements: Vec<GPSPositionAndVelocityMeasurement> = gps_measurements
            .iter()
            .map(|gps| GPSPositionAndVelocityMeasurement {
                latitude: gps.latitude,
                longitude: gps.longitude,
                altitude: gps.altitude,
                northward_velocity: 0.0,  // Stationary
                eastward_velocity: 0.0,   // Stationary
                horizontal_noise_std: gps.horizontal_noise_std,
                vertical_noise_std: gps.vertical_noise_std,
                velocity_noise_std: 0.1,  // 0.1 m/s velocity noise
            })
            .collect();

        // Initialize particle filter with some initial uncertainty
        // Note: Particle filter expects radians for lat/lon internally
        let initial_mean = DVector::from_vec(vec![
            initial_state.latitude,  // Already in radians
            initial_state.longitude, // Already in radians
            initial_state.altitude,
            0.0, 0.0, 0.0, // velocities
            0.0, 0.0, 0.0, // euler angles
            0.0, 0.0, 0.0, // accel bias
            0.0, 0.0, 0.0, // gyro bias
        ]);

        let initial_covariance = DVector::from_vec(vec![
            (10.0 * earth::METERS_TO_DEGREES).to_radians().powi(2), 
            (10.0 * earth::METERS_TO_DEGREES).to_radians().powi(2), 
            5.0_f64.powi(2),
            1e-8, 1e-8, 1e-8, // Very low velocity uncertainty for stationary scenario
            0.1_f64.powi(2), 0.1_f64.powi(2), 0.1_f64.powi(2),
            0.01, 0.01, 0.01,
            0.001, 0.001, 0.001,
        ]);

        // Use small but non-zero process noise to account for modeling errors
        let process_noise = DVector::from_vec(vec![
            1e-12, 1e-12, 1e-8,  // Very small position noise
            1e-8, 1e-8, 1e-8,    // Small velocity noise
            1e-8, 1e-8, 1e-8,    // Small attitude noise
            1e-10, 1e-10, 1e-10, // Very small accel bias noise
            1e-10, 1e-10, 1e-10, // Very small gyro bias noise
        ]);

        let mut pf = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            500, // particles
            42,  // random seed
        );

        // Run filter for 10 seconds with resampling
        for i in 0..10 {
            // Predict step with IMU data
            pf.predict(&imu_data[i], 1.0);
            
            // Debug: Check a few particles' states
            if i == 0 {
                for j in 0..3 {
                    let p = &pf.particles()[j];
                    println!("  Particle {}: is_enu={}, alt={:.2}m, v_vert={:.3}m/s", 
                             j, p.state.is_enu, p.state.altitude, p.state.velocity_vertical);
                }
            }
            
            // Update with GPS position+velocity measurement every second
            pf.update(&gps_pos_vel_measurements[i]);
            
            // Resample every 5 steps to prevent particle degeneracy
            if i % 5 == 0 {
                pf.resample();
            }

            if i % 5 == 0 {
                let state = pf.get_estimate();
                println!("  Time {}s: Estimated position: ({:.6}°, {:.6}°, {:.2}m)",
                         i,
                         state[0].to_degrees(),
                         state[1].to_degrees(),
                         state[2]);
            }
        }

        // Check final estimate
        let final_estimate = pf.get_estimate();
        let final_true_state = &true_states.last().unwrap();

        println!("Stationary test:");
        println!("  True final position: ({:.6}°, {:.6}°, {:.2}m)",
                 final_true_state.latitude.to_degrees(),
                 final_true_state.longitude.to_degrees(),
                 final_true_state.altitude);
        println!("  Estimated position: ({:.6}°, {:.6}°, {:.2}m)",
                 final_estimate[0].to_degrees(),
                 final_estimate[1].to_degrees(),
                 final_estimate[2]);
        println!("  Position error: ({:.3}m, {:.3}m, {:.3}m)",
                 (final_estimate[0] - final_true_state.latitude) * earth::DEGREES_TO_METERS,
                 (final_estimate[1] - final_true_state.longitude) * earth::DEGREES_TO_METERS,
                 final_estimate[2] - final_true_state.altitude);

        // Assert reasonable tracking accuracy
        // Note: Vertical accuracy is limited because GPS does not measure vertical velocity
        // Horizontal velocity measurements help constrain horizontal position drift
        assert!((final_estimate[0] - final_true_state.latitude).abs() * earth::DEGREES_TO_METERS < 50.0,
                "Latitude error: {:.1}m", 
                (final_estimate[0] - final_true_state.latitude).abs() * earth::DEGREES_TO_METERS);
        assert!((final_estimate[1] - final_true_state.longitude).abs() * earth::DEGREES_TO_METERS < 50.0,
                "Longitude error: {:.1}m",
                (final_estimate[1] - final_true_state.longitude).abs() * earth::DEGREES_TO_METERS);
        // Looser vertical constraint due to unobserved vertical velocity
        assert!((final_estimate[2] - final_true_state.altitude).abs() < 200.0,
                "Altitude error: {:.1}m (vertical velocity is unobserved)",
                (final_estimate[2] - final_true_state.altitude).abs());
    }

    /// Test particle filter tracking eastward constant velocity flight
    ///
    /// # Test Rationale
    ///
    /// This test demonstrates particle filter performance on a moving target with:
    /// - Perfect IMU data with dynamically calculated acceleration to maintain constant velocity
    /// - GPS position + horizontal velocity measurements (NO vertical velocity)
    /// - 10-second duration (short to limit drift from unobserved vertical velocity)
    ///
    /// ## Key Differences from Stationary Test
    ///
    /// - Vehicle moves eastward at 10 m/s along the equator
    /// - GPS provides velocity measurements (10 m/s east, 0 m/s north)
    /// - Horizontal velocity is observable, constraining horizontal position drift
    /// - Vertical velocity remains unobserved, so altitude still drifts
    ///
    /// ## Expected Behavior
    ///
    /// - Horizontal tracking should be excellent (within 50m) due to velocity measurements
    /// - Vertical tracking degrades over time due to unobserved vertical velocity
    /// - Looser altitude tolerance (200m) reflects this observability limitation
    #[test]
    fn test_particle_filter_eastward_constant_velocity() {
        use crate::{generate_scenario_data, StrapdownState};
        use crate::measurements::GPSPositionAndVelocityMeasurement;
        use nalgebra::{Rotation3, Vector3};

        // Vehicle moving eastward at 10 m/s at equator
        let vel = 10.0;
        let initial_state = StrapdownState {
            latitude: 0.0_f64.to_radians(),
            longitude: 0.0_f64.to_radians(),
            altitude: 1000.0,
            velocity_north: 0.0,
            velocity_east: vel,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let g = earth::gravity(&0.0, &1000.0);
        let accel_body = Vector3::new(0.0, 0.0, g);
        let gyro_body = Vector3::zeros();

        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            10,  // 10 seconds (very short to demonstrate tracking capability)
            1,   // 1 Hz
            accel_body,
            gyro_body,
            true,  // Geosynchronous
            true,  // Constant velocity (dynamic acceleration calculation)
            false  // GPS output always in degrees
        );

        // Convert GPS position measurements to position+velocity measurements
        let gps_pos_vel_measurements: Vec<GPSPositionAndVelocityMeasurement> = gps_measurements
            .iter()
            .map(|gps| GPSPositionAndVelocityMeasurement {
                latitude: gps.latitude,
                longitude: gps.longitude,
                altitude: gps.altitude,
                northward_velocity: 0.0,  // Constant eastward velocity
                eastward_velocity: vel,   // 10 m/s east
                horizontal_noise_std: gps.horizontal_noise_std,
                vertical_noise_std: gps.vertical_noise_std,
                velocity_noise_std: 0.1,  // 0.1 m/s velocity noise
            })
            .collect();

        // Initialize particle filter
        let initial_mean = DVector::from_vec(vec![
            initial_state.latitude,
            initial_state.longitude,
            initial_state.altitude,
            0.0, vel, 0.0, // velocities (initial velocity estimate)
            0.0, 0.0, 0.0, // euler angles
            0.0, 0.0, 0.0, // accel bias
            0.0, 0.0, 0.0, // gyro bias
        ]);

        let initial_covariance = DVector::from_vec(vec![
            (10.0 * earth::METERS_TO_DEGREES).to_radians().powi(2), 
            (10.0 * earth::METERS_TO_DEGREES).to_radians().powi(2), 
            5.0_f64.powi(2),
            0.1, 0.1, 1e-8, // Horizontal velocity uncertainty, very low vertical
            0.1_f64.powi(2), 0.1_f64.powi(2), 0.1_f64.powi(2),
            0.01, 0.01, 0.01,
            0.001, 0.001, 0.001,
        ]);

        let process_noise = DVector::from_vec(vec![
            1e-12, 1e-12, 1e-8,  // Very small position noise
            1e-8, 1e-8, 1e-8,    // Small velocity noise
            1e-8, 1e-8, 1e-8,    // Small attitude noise
            1e-10, 1e-10, 1e-10, // Very small accel bias noise
            1e-10, 1e-10, 1e-10, // Very small gyro bias noise
        ]);

        let mut pf = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            500, // particles
            42,  // random seed
        );

        // Run filter for 60 seconds with resampling
        for i in 0..10 {
            pf.predict(&imu_data[i], 1.0);
            pf.update(&gps_pos_vel_measurements[i]);
            
            // Resample every 5 steps
            if i % 5 == 0 {
                pf.resample();
            }

            if i % 3 == 0 {
                let state = pf.get_estimate();
                println!("  Time {}s: Estimated position: ({:.6}°, {:.6}°, {:.2}m), velocity: ({:.2}, {:.2}, {:.2}) m/s",
                         i,
                         state[0].to_degrees(),
                         state[1].to_degrees(),
                         state[2],
                         state[3], state[4], state[5]);
            }
        }

        // Check final estimate
        let final_estimate = pf.get_estimate();
        let final_true_state = &true_states.last().unwrap();

        println!("Eastward flight test:");
        println!("  True final position: ({:.6}°, {:.6}°, {:.2}m)",
                 final_true_state.latitude.to_degrees(),
                 final_true_state.longitude.to_degrees(),
                 final_true_state.altitude);
        println!("  Estimated position: ({:.6}°, {:.6}°, {:.2}m)",
                 final_estimate[0].to_degrees(),
                 final_estimate[1].to_degrees(),
                 final_estimate[2]);
        println!("  True final velocity: ({:.3}, {:.3}, {:.3}) m/s",
                 final_true_state.velocity_north,
                 final_true_state.velocity_east,
                 final_true_state.velocity_vertical);
        println!("  Estimated velocity: ({:.3}, {:.3}, {:.3}) m/s",
                 final_estimate[3], final_estimate[4], final_estimate[5]);
        println!("  Position error: ({:.3}m, {:.3}m, {:.3}m)",
                 (final_estimate[0] - final_true_state.latitude) * earth::DEGREES_TO_METERS,
                 (final_estimate[1] - final_true_state.longitude) * earth::DEGREES_TO_METERS,
                 final_estimate[2] - final_true_state.altitude);

        // Assert reasonable tracking accuracy
        // Horizontal tracking should be good due to velocity measurements
        assert!((final_estimate[0] - final_true_state.latitude).abs() * earth::DEGREES_TO_METERS < 50.0,
                "Latitude error: {:.1}m", 
                (final_estimate[0] - final_true_state.latitude).abs() * earth::DEGREES_TO_METERS);
        assert!((final_estimate[1] - final_true_state.longitude).abs() * earth::DEGREES_TO_METERS < 50.0,
                "Longitude error: {:.1}m",
                (final_estimate[1] - final_true_state.longitude).abs() * earth::DEGREES_TO_METERS);
        // Looser vertical constraint due to unobserved vertical velocity
        assert!((final_estimate[2] - final_true_state.altitude).abs() < 200.0,
                "Altitude error: {:.1}m (vertical velocity is unobserved)",
                (final_estimate[2] - final_true_state.altitude).abs());
        
        // Velocity tracking should be good for horizontal components
        // Note: Some velocity drift is expected due to coupling with unobserved vertical velocity
        assert!((final_estimate[3] - final_true_state.velocity_north).abs() < 2.0,
                "North velocity error: {:.2}m/s", 
                (final_estimate[3] - final_true_state.velocity_north).abs());
        assert!((final_estimate[4] - final_true_state.velocity_east).abs() < 2.0,
                "East velocity error: {:.2}m/s",
                (final_estimate[4] - final_true_state.velocity_east).abs());
    }

    /// Test particle filter tracking northward constant velocity flight
    ///
    /// # Test Rationale
    ///
    /// This test demonstrates particle filter performance on a moving target with:
    /// - Perfect IMU data with dynamically calculated acceleration to maintain constant velocity
    /// - GPS position + horizontal velocity measurements (NO vertical velocity)
    /// - 10-second duration (short to limit drift from unobserved vertical velocity)
    ///
    /// # Key Observability Insight
    ///
    /// GPSPositionAndVelocityMeasurement only measures 5 states:
    /// - [latitude, longitude, altitude, v_north, v_east]
    /// - DOES NOT measure v_vertical
    ///
    /// This means the vertical velocity channel is UNOBSERVED and will drift over time.
    /// The drift is not a bug in the filter - it's a fundamental limitation of the
    /// measurement model. Real-world systems need additional altitude rate sensors
    /// (barometer, radar altimeter, etc.) to constrain vertical velocity.
    ///
    /// # Test Configuration
    ///
    /// We set very low initial vertical velocity uncertainty (1e-8) to start particles
    /// with nearly identical vertical velocity, reducing initial spread. The short 10s
    /// duration limits how much drift can accumulate.
    #[test]
    fn test_particle_filter_northward_constant_velocity() {
        use crate::{generate_scenario_data, StrapdownState};
        use crate::measurements::GPSPositionAndVelocityMeasurement;
        use nalgebra::{Rotation3, Vector3};

        // Vehicle moving northward at 10 m/s at equator
        let vel = 10.0;
        let initial_state = StrapdownState {
            latitude: 0.0_f64.to_radians(),
            longitude: 0.0_f64.to_radians(),
            altitude: 1000.0,
            velocity_north: vel,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let g = earth::gravity(&0.0, &1000.0);
        let accel_body = Vector3::new(0.0, 0.0, g);
        let gyro_body = Vector3::zeros();

        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            10,  // 10 seconds (very short to demonstrate tracking capability)
            1,   // 1 Hz
            accel_body,
            gyro_body,
            true,  // Geosynchronous
            true,  // Constant velocity (dynamic acceleration calculation)
            false  // GPS output always in degrees
        );

        // Initialize particle filter
        let initial_mean = DVector::from_vec(vec![
            initial_state.latitude,
            initial_state.longitude,
            initial_state.altitude,
            vel, 0.0, 0.0, // velocities
            0.0, 0.0, 0.0, // euler angles
            0.0, 0.0, 0.0, // accel bias
            0.0, 0.0, 0.0, // gyro bias
        ]);

        // Initial covariance: tight on position and vertical velocity
        // Vertical velocity uncertainty is very low (1e-8) since it's unobserved by GPS
        let initial_covariance = DVector::from_vec(vec![
            (5.0 * earth::METERS_TO_DEGREES).powi(2), (5.0 * earth::METERS_TO_DEGREES).powi(2), 5.0_f64.powi(2),
            0.1, 0.1, 1e-8,  // Horizontal velocity uncertainty 0.1, vertical 1e-8
            0.1_f64.powi(2), 0.1_f64.powi(2), 0.1_f64.powi(2),
            0.01, 0.01, 0.01,
            0.001, 0.001, 0.001,
        ]);

        // Minimal process noise since IMU data is perfect in this synthetic test
        let process_noise = DVector::from_vec(vec![
            1e-12, 1e-12, 1e-12,  // Position process noise
            1e-8, 1e-8, 1e-8,     // Velocity process noise
            1e-10, 1e-10, 1e-10,  // Attitude process noise
            1e-12, 1e-12, 1e-12,  // Accel bias process noise
            1e-12, 1e-12, 1e-12,  // Gyro bias process noise
        ]);

        let mut pf = StrapdownParticleFilter::new_about(
            initial_mean,
            initial_covariance,
            process_noise,
            500,
            42,
        );

        // Convert GPS measurements to position+velocity measurements
        let gps_pos_vel_measurements: Vec<GPSPositionAndVelocityMeasurement> = gps_measurements
            .iter()
            .zip(true_states.iter())
            .map(|(gps, state)| GPSPositionAndVelocityMeasurement {
                latitude: gps.latitude,
                longitude: gps.longitude,
                altitude: gps.altitude,
                northward_velocity: state.velocity_north,
                eastward_velocity: state.velocity_east,
                horizontal_noise_std: gps.horizontal_noise_std,
                vertical_noise_std: gps.vertical_noise_std,
                velocity_noise_std: 0.1,  // 0.1 m/s velocity noise
            })
            .collect();

        // Print first measurement for verification
        if !gps_pos_vel_measurements.is_empty() {
            let first_meas = &gps_pos_vel_measurements[0];
            let first_imu = &imu_data[0];
            let first_state = &true_states[0];
            println!("Time: 0s, IMU Accel: ({:.4}, {:.4}, {:.4}) m/s² | Gyro: ({:.4}, {:.4}, {:.4}) rad/s | GPS Pos: ({:.3}°, {:.3}°, {:.1}m) | Velocities: N: {:.3} m/s, E: {:.3} m/s, V: {:.3} m/s",
                     first_imu.accel[0], first_imu.accel[1], first_imu.accel[2],
                     first_imu.gyro[0], first_imu.gyro[1], first_imu.gyro[2],
                     first_meas.latitude, first_meas.longitude, first_meas.altitude,
                     first_state.velocity_north, first_state.velocity_east, first_state.velocity_vertical);
        }

        // Run filter with GPS updates every second for 10 seconds
        for i in 0..10 {
            pf.predict(&imu_data[i], 1.0);
            pf.update(&gps_pos_vel_measurements[i]);
            
            // Resample every 5 steps
            if i % 5 == 0 {
                pf.resample();
            }

            if i % 3 == 0 {
                let state = pf.get_estimate();
                println!("  Time {}s: Estimated position: ({:.6}°, {:.6}°, {:.2}m), velocity: ({:.2}, {:.2}, {:.2}) m/s",
                         i,
                         state[0].to_degrees(),
                         state[1].to_degrees(),
                         state[2],
                         state[3], state[4], state[5]);
            }
        }

        // Check final estimate
        let final_estimate = pf.get_estimate();
        let final_true_state = &true_states.last().unwrap();

        println!("Northward flight test:");
        println!("  True final position: ({:.6}°, {:.6}°, {:.2}m)",
                 final_true_state.latitude.to_degrees(),
                 final_true_state.longitude.to_degrees(),
                 final_true_state.altitude);
        println!("  Estimated position: ({:.6}°, {:.6}°, {:.2}m)",
                 final_estimate[0].to_degrees(),
                 final_estimate[1].to_degrees(),
                 final_estimate[2]);
        println!("  True final velocity: ({:.3}, {:.3}, {:.3}) m/s",
                 final_true_state.velocity_north,
                 final_true_state.velocity_east,
                 final_true_state.velocity_vertical);
        println!("  Estimated velocity: ({:.3}, {:.3}, {:.3}) m/s",
                 final_estimate[3], final_estimate[4], final_estimate[5]);
        println!("  Position error: ({:.3}m, {:.3}m, {:.3}m)",
                 (final_estimate[0] - final_true_state.latitude) * earth::DEGREES_TO_METERS,
                 (final_estimate[1] - final_true_state.longitude) * earth::DEGREES_TO_METERS,
                 final_estimate[2] - final_true_state.altitude);

        // Assert reasonable tracking accuracy
        // Horizontal tracking should be good due to velocity measurements
        assert!((final_estimate[0] - final_true_state.latitude).abs() * earth::DEGREES_TO_METERS < 50.0,
                "Latitude error: {:.1}m", 
                (final_estimate[0] - final_true_state.latitude).abs() * earth::DEGREES_TO_METERS);
        assert!((final_estimate[1] - final_true_state.longitude).abs() * earth::DEGREES_TO_METERS < 50.0,
                "Longitude error: {:.1}m",
                (final_estimate[1] - final_true_state.longitude).abs() * earth::DEGREES_TO_METERS);
        // Looser vertical constraint due to unobserved vertical velocity
        assert!((final_estimate[2] - final_true_state.altitude).abs() < 200.0,
                "Altitude error: {:.1}m (vertical velocity is unobserved)",
                (final_estimate[2] - final_true_state.altitude).abs());
        
        // Velocity tracking should be good for horizontal components
        // Note: Some velocity drift is expected due to coupling with unobserved vertical velocity
        assert!((final_estimate[3] - final_true_state.velocity_north).abs() < 2.0,
                "North velocity error: {:.2}m/s", 
                (final_estimate[3] - final_true_state.velocity_north).abs());
        assert!((final_estimate[4] - final_true_state.velocity_east).abs() < 2.0,
                "East velocity error: {:.2}m/s",
                (final_estimate[4] - final_true_state.velocity_east).abs());
    }

}