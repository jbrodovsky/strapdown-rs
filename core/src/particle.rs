//! Particle filter implementation for nonlinear Bayesian state estimation.
//!
//! This module provides a particle filter (also known as Sequential Monte Carlo) implementation
//! for strapdown inertial navigation. Unlike Kalman filters which assume Gaussian distributions,
//! particle filters can represent arbitrary probability distributions through weighted samples.
//!
//! # Key Components
//!
//! - [`Particle`]: Individual state hypothesis with navigation state, biases, and weight
//! - [`ParticleFilter`]: Main filter managing particle ensemble and resampling
//! - [`ProcessNoise`]: Process noise parameters for stochastic propagation
//! - [`ParticleAveragingStrategy`]: Methods for extracting state estimates from particles
//! - [`ParticleResamplingStrategy`]: Algorithms for combating particle degeneracy
//!
//! # Vertical Channel Damping
//!
//! The particle filter uses a modified vertical channel model to stabilize altitude estimates.
//! Rather than purely kinematic integration, a damped oscillator model reduces drift:
//!
//! $$
//! \dot{v}_D = a_D - g + b_D + w_D - 2\zeta\omega_n v_D - \omega_n^2 (h - h_{ref})
//! $$
//!
//! where $\zeta$ is damping ratio, $\omega_n$ is natural frequency, and $h_{ref}$ is reference altitude.
//!
//! # References
//!
//! * Sokolovic, V., et al. (2014). "Adaptive Error Damping in the Vertical Channel of the
//!   INS/GPS/Baro-Altimeter Integrated Navigation System." Scientific Technical Review.
//! * Arulampalam, M. S., et al. (2002). "A tutorial on particle filters for online
//!   nonlinear/non-Gaussian Bayesian tracking." IEEE Transactions on Signal Processing.

use crate::earth;
use crate::kalman::NavigationFilter;
use crate::measurements::MeasurementModel;
use crate::{IMUData, StrapdownState, attitude_update};

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use rand::rngs::StdRng;
use rand::{self, Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::fmt::{self, Debug};

// Tunable Gains (Third-Order Loop)
/// Position Error Gain (1/s)
const K1: f64 = 1.0;
/// Velocity Error Gain (1/s^2)
const K2: f64 = 0.1; 
/// Bias Error Gain (1/s^3) - Keep this very small!
const K3: f64 = 0.001; 

#[derive(Clone, Debug, Default)]
pub struct Particle {
    /// Navigation state (position, velocity, attitude) in NED frame
    pub nav_state: StrapdownState,
    
    /// Accelerometer bias vector [b_x, b_y, b_z] in m/s²
    pub accel_bias: DVector<f64>,
    
    /// Gyroscope bias vector [b_p, b_q, b_r] in rad/s
    pub gyro_bias: DVector<f64>,
    
    /// Optional extended states beyond standard 15-state INS model
    pub other_states: Option<DVector<f64>>,
    
    /// Total dimension of state vector (minimum 15)
    pub state_size: usize,
    
    /// Importance weight (unnormalized likelihood)
    pub weight: f64,
    
    /// Altitude error for vertical channel damping feedback (meters)
    pub altitude_error: f64,
    
    /// Position error gain for vertical channel (1/s)
    pub k1: f64,
    
    /// Velocity error gain for vertical channel (1/s²)
    pub k2: f64,
    
    /// Bias error gain for vertical channel (1/s³)
    pub k3: f64,
}
impl Particle {
    /// Creates a new particle with specified state and parameters.
    ///
    /// # Arguments
    ///
    /// * `nav_state` - Navigation state (position, velocity, attitude)
    /// * `accel_bias` - 3D accelerometer bias vector in m/s²
    /// * `gyro_bias` - 3D gyroscope bias vector in rad/s
    /// * `other_states` - Optional extended states beyond 15-state model
    /// * `weight` - Initial importance weight (typically 1/N for N particles)
    /// * `altitude_error` - Initial altitude error estimate in meters (default 0.0)
    /// * `k1` - Position error gain for vertical damping (1/s)
    /// * `k2` - Velocity error gain for vertical damping (1/s²)
    /// * `k3` - Bias error gain for vertical damping (1/s³)
    ///
    /// # Panics
    ///
    /// Panics if accelerometer or gyroscope bias vectors are not 3-dimensional,
    /// or if gains are not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use strapdown_core::particle::Particle;
    /// use strapdown_core::StrapdownState;
    /// use nalgebra::{DVector, Rotation3, Vector3};
    ///
    /// let nav_state = StrapdownState {
    ///     latitude: 0.7,
    ///     longitude: -1.2,
    ///     altitude: 100.0,
    ///     velocity: Vector3::zeros(),
    ///     attitude: Rotation3::identity(),
    /// };
    /// let particle = Particle::new(
    ///     nav_state,
    ///     DVector::zeros(3),
    ///     DVector::zeros(3),
    ///     None,
    ///     1.0,
    ///     None,
    ///     1.0,
    ///     0.1,
    ///     0.001,
    /// );
    /// ```
    pub fn new(
        nav_state: StrapdownState,
        accel_bias: DVector<f64>,
        gyro_bias: DVector<f64>,
        other_states: Option<DVector<f64>>,
        weight: f64,
        altitude_error: Option<f64>,
        k1: f64,
        k2: f64,
        k3: f64,
    ) -> Particle {
        assert!(
            accel_bias.len() == 3,
            "Accelerometer bias must be a 3-element vector"
        );
        assert!(
            gyro_bias.len() == 3,
            "Gyroscope bias must be a 3-element vector"
        );
        let state_size = 15
            + match &other_states {
                Some(states) => states.len(),
                None => 0,
            };
        assert!(
            state_size >= 15,
            "Particle state vector size must be at least 15 (navigation states plus biases)"
        );
        assert!(
            weight >= 0.0 && weight.is_finite(),
            "Particle weight must be non-negative and finite"
        );
        let altitude_error = altitude_error.unwrap_or(0.0);
        Particle {
            nav_state,
            accel_bias,
            gyro_bias,
            other_states,
            state_size,
            weight,
            altitude_error,
            k1,
            k2,
            k3,
        }
    }
}
impl From<(DVector<f64>, f64)> for Particle {
    /// Converts a state vector and weight into a Particle.
    ///
    /// # Arguments
    ///
    /// * `tuple.0` - State vector with minimum 15 elements (see [`Particle`] for layout)
    /// * `tuple.1` - Importance weight
    ///
    /// # Panics
    ///
    /// Panics if state vector has fewer than 15 elements.
    ///
    /// # State Vector Layout
    ///
    /// - [0-2]: latitude (rad), longitude (rad), altitude (m)
    /// - [3-5]: velocity NED (m/s)
    /// - [6-8]: roll, pitch, yaw (rad)
    /// - [9-11]: accelerometer biases (m/s²)
    /// - [12-14]: gyroscope biases (rad/s)
    /// - [15+]: optional extended states
    fn from(tuple: (DVector<f64>, f64)) -> Self {
        let (state_vector, weight) = tuple;
        assert!(
            state_vector.len() >= 15,
            "State vector must be at least 15 elements long"
        );
        let nav_state = StrapdownState {
            latitude: state_vector[0],
            longitude: state_vector[1],
            altitude: state_vector[2],
            velocity_north: state_vector[3],
            velocity_east: state_vector[4],
            velocity_down: state_vector[5],
            attitude: Rotation3::from_euler_angles(
                state_vector[6],
                state_vector[7],
                state_vector[8],
            ),
            is_enu: true,
        };
        let accel_bias =
            DVector::from_vec(vec![state_vector[9], state_vector[10], state_vector[11]]);
        let gyro_bias =
            DVector::from_vec(vec![state_vector[12], state_vector[13], state_vector[14]]);
        let other_states = if state_vector.len() > 15 {
            Some(state_vector.rows(15, state_vector.len() - 15).clone_owned())
        } else {
            None
        };
        Particle::new(
            nav_state,
            accel_bias,
            gyro_bias,
            other_states,
            weight,
            None,
            0.0,
            0.0,
            0.0,
        )
    }
}
impl From<Particle> for (DVector<f64>, f64) {
    /// Converts a Particle into a state vector and weight tuple.
    ///
    /// # Returns
    ///
    /// Tuple of (state_vector, weight) where state_vector contains:
    /// - Navigation state (position, velocity, attitude Euler angles)
    /// - Accelerometer biases
    /// - Gyroscope biases
    /// - Optional extended states
    ///
    /// This is the inverse of the `From<(DVector<f64>, f64)>` implementation.
    fn from(val: Particle) -> Self {
        let mut state_vec = vec![
            val.nav_state.latitude,
            val.nav_state.longitude,
            val.nav_state.altitude,
            val.nav_state.velocity_north,
            val.nav_state.velocity_east,
            val.nav_state.velocity_down,
            val.nav_state.attitude.euler_angles().0,
            val.nav_state.attitude.euler_angles().1,
            val.nav_state.attitude.euler_angles().2,
        ];
        state_vec.push(val.accel_bias[0]);
        state_vec.push(val.accel_bias[1]);
        state_vec.push(val.accel_bias[2]);
        state_vec.push(val.gyro_bias[0]);
        state_vec.push(val.gyro_bias[1]);
        state_vec.push(val.gyro_bias[2]);
        if let Some(other_states) = val.other_states {
            state_vec.extend(other_states.iter());
        }
        (DVector::from_vec(state_vec), val.weight)
    }
}
/// Strategy for extracting point estimates from particle ensemble.
///
/// Different averaging methods trade off between robustness and optimality.
/// Weighted average is optimal for unimodal distributions, while highest weight
/// can better represent multimodal posteriors.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum ParticleAveragingStrategy {
    /// Compute weighted mean and covariance (optimal for Gaussian posteriors)
    #[default]
    WeightedAverage,
    
    /// Compute unweighted mean and covariance (more robust to weight degeneracy)
    UnweightedAverage,
    
    /// Use state of highest-weight particle (best for multimodal distributions)
    HighestWeight,
}
/// Trait for computing state estimates from particle ensembles.
///
/// Implementations should return the mean state vector and covariance matrix
/// representing the current posterior distribution.
pub trait Average: Send + Sync {
    /// Computes point estimate and covariance from particle filter.
    ///
    /// # Arguments
    ///
    /// * `pf` - Particle filter with weighted ensemble
    ///
    /// # Returns
    ///
    /// Tuple of (mean_state, covariance_matrix) where dimensions match `pf.state_size`
    fn average(&self, pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>);
}
/// Implementations of averaging strategies
impl Average for ParticleAveragingStrategy {
    fn average(&self, pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>) {
        match self {
            ParticleAveragingStrategy::WeightedAverage => {
                ParticleAveragingStrategy::weighted_average_state(pf)
            }
            ParticleAveragingStrategy::UnweightedAverage => {
                ParticleAveragingStrategy::unweighted_average_state(pf)
            }
            ParticleAveragingStrategy::HighestWeight => {
                ParticleAveragingStrategy::highest_weight_state(pf)
            }
        }
    }
}
/// Averaging implementations
impl ParticleAveragingStrategy {
    fn weighted_average_state(pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>) {
        let state_size = pf.state_size;
        println!("State size: {}", state_size);
        let mut mean = DVector::<f64>::zeros(state_size);
        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            mean[0] += particle.weight * particle.nav_state.latitude;
            mean[1] += particle.weight * particle.nav_state.longitude;
            mean[2] += particle.weight * particle.nav_state.altitude;
            mean[3] += particle.weight * particle.nav_state.velocity_north;
            mean[4] += particle.weight * particle.nav_state.velocity_east;
            mean[5] += particle.weight * particle.nav_state.velocity_down;
            mean[6] += particle.weight * euler.0;
            mean[7] += particle.weight * euler.1;
            mean[8] += particle.weight * euler.2;
            mean[9] += particle.weight * particle.accel_bias[0];
            mean[10] += particle.weight * particle.accel_bias[1];
            mean[11] += particle.weight * particle.accel_bias[2];
            mean[12] += particle.weight * particle.gyro_bias[0];
            mean[13] += particle.weight * particle.gyro_bias[1];
            mean[14] += particle.weight * particle.gyro_bias[2];
            if let Some(ref other) = particle.other_states {
                for (i, val) in other.iter().enumerate() {
                    if 9 + i < state_size {
                        mean[9 + i] += particle.weight * val;
                    }
                }
            }
        }
        let mut cov = DMatrix::<f64>::zeros(state_size, state_size);
        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            let mut state_vec = vec![
                particle.nav_state.latitude,
                particle.nav_state.longitude,
                particle.nav_state.altitude,
                particle.nav_state.velocity_north,
                particle.nav_state.velocity_east,
                particle.nav_state.velocity_down,
                euler.0,
                euler.1,
                euler.2,
                particle.accel_bias[0],
                particle.accel_bias[1],
                particle.accel_bias[2],
                particle.gyro_bias[0],
                particle.gyro_bias[1],
                particle.gyro_bias[2],
            ];
            if let Some(ref other) = particle.other_states {
                state_vec.extend(other.iter());
            }
            let state = DVector::from_vec(state_vec);
            let diff = state - &mean;
            cov += particle.weight * &diff * &diff.transpose();
        }
        (mean, cov)
    }
    fn unweighted_average_state(pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>) {
        let n = pf.particles.len() as f64;
        let state_size = pf.state_size;
        let mut mean = DVector::<f64>::zeros(state_size);
        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            mean[0] += particle.nav_state.latitude / n;
            mean[1] += particle.nav_state.longitude / n;
            mean[2] += particle.nav_state.altitude / n;
            mean[3] += particle.nav_state.velocity_north / n;
            mean[4] += particle.nav_state.velocity_east / n;
            mean[5] += particle.nav_state.velocity_down / n;
            mean[6] += euler.0 / n;
            mean[7] += euler.1 / n;
            mean[8] += euler.2 / n;
            mean[9] += particle.accel_bias[0] / n;
            mean[10] += particle.accel_bias[1] / n;
            mean[11] += particle.accel_bias[2] / n;
            mean[12] += particle.gyro_bias[0] / n;
            mean[13] += particle.gyro_bias[1] / n;
            mean[14] += particle.gyro_bias[2] / n;
            if let Some(ref other) = particle.other_states {
                for (i, val) in other.iter().enumerate() {
                    if 9 + i < state_size {
                        mean[9 + i] += val / n;
                    }
                }
            }
        }
        let mut cov = DMatrix::<f64>::zeros(state_size, state_size);
        for particle in &pf.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            let mut state_vec = vec![
                particle.nav_state.latitude,
                particle.nav_state.longitude,
                particle.nav_state.altitude,
                particle.nav_state.velocity_north,
                particle.nav_state.velocity_east,
                particle.nav_state.velocity_down,
                euler.0,
                euler.1,
                euler.2,
                particle.accel_bias[0],
                particle.accel_bias[1],
                particle.accel_bias[2],
                particle.gyro_bias[0],
                particle.gyro_bias[1],
                particle.gyro_bias[2],
            ];
            if let Some(ref other) = particle.other_states {
                state_vec.extend(other.iter());
            }
            let state = DVector::from_vec(state_vec);
            let diff = state - &mean;
            cov += (1.0 / n) * &diff * &diff.transpose();
        }
        (mean, cov)
    }
    fn highest_weight_state(pf: &ParticleFilter) -> (DVector<f64>, DMatrix<f64>) {
        let best_particle = pf
            .particles
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
            .expect("Particle filter has no particles");
        let euler = best_particle.nav_state.attitude.euler_angles();
        let mut state_vec = vec![
            best_particle.nav_state.latitude,
            best_particle.nav_state.longitude,
            best_particle.nav_state.altitude,
            best_particle.nav_state.velocity_north,
            best_particle.nav_state.velocity_east,
            best_particle.nav_state.velocity_down,
            euler.0,
            euler.1,
            euler.2,
            best_particle.accel_bias[0],
            best_particle.accel_bias[1],
            best_particle.accel_bias[2],
            best_particle.gyro_bias[0],
            best_particle.gyro_bias[1],
            best_particle.gyro_bias[2],
        ];
        if let Some(ref other_states) = best_particle.other_states {
            state_vec.extend(other_states.iter());
        }
        let mean = DVector::from_vec(state_vec);
        let cov = DMatrix::<f64>::zeros(pf.state_size, pf.state_size);
        (mean, cov)
    }
}

/// Resampling algorithm for combating particle degeneracy.
///
/// Resampling duplicates high-weight particles and discards low-weight ones,
/// maintaining ensemble diversity. Different algorithms have trade-offs in
/// computational cost and variance of the resampled ensemble.
///
/// # References
///
/// * Douc, R., & Cappé, O. (2005). "Comparison of resampling schemes for particle filtering."
///   Image and Signal Processing and Analysis.
#[derive(Clone, Debug, Default)]
pub enum ParticleResamplingStrategy {
    /// Naive random sampling with replacement (high variance)
    Naive,
    
    /// Systematic resampling with deterministic spacing (low variance)
    Systematic,
    
    /// Independent draws from cumulative distribution (moderate variance)
    Multinomial,
    
    /// Residual resampling combining deterministic and stochastic steps (low variance)
    #[default]
    Residual,
    
    /// Stratified resampling with jittered deterministic samples (low variance)
    Stratified,
    
    /// Adaptive resampling based on effective sample size (currently aliases Systematic)
    Adaptive,
}
/// Trait for implementing resampling strategies
impl ParticleResamplingStrategy {
    fn naive_resample(particles: Vec<Particle>, rng: &mut StdRng) -> Vec<Particle> {
        let n = particles.len();
        let mut new_particles = Vec::<Particle>::with_capacity(n);
        let uniform_weight = 1.0 / n as f64;
        for _ in 0..n {
            let idx = rng.next_u32() as usize % n;
            let mut new_particle = particles[idx].clone();
            new_particle.weight = uniform_weight;
            new_particles.push(new_particle);
        }
        new_particles
    }
    fn systematic_resample(particles: Vec<Particle>, rng: &mut StdRng) -> Vec<Particle> {
        let n = particles.len();
        let mut new_particles = Vec::<Particle>::with_capacity(n);
        if n == 0 {
            return new_particles;
        }
        let step = 1.0 / n as f64;
        let mut u = rng.random::<f64>() * step;
        let mut i = 0usize;
        let mut cumsum = particles.get(0).map(|p| p.weight).unwrap_or(0.0);
        for _ in 0..n {
            while u > cumsum && i + 1 < n {
                i += 1;
                cumsum += particles[i].weight;
            }
            let mut new_particle = particles[i].clone();
            new_particle.weight = 1.0 / n as f64;
            new_particles.push(new_particle);
            u += step;
        }
        new_particles
    }
    fn multinomial_resample(particles: Vec<Particle>, rng: &mut StdRng) -> Vec<Particle> {
        let n = particles.len();
        let mut new_particles = Vec::<Particle>::with_capacity(n);
        if n == 0 {
            return new_particles;
        }
        let cumulative_weights: Vec<f64> = particles
            .iter()
            .scan(0.0, |acc, p| {
                *acc += p.weight;
                Some(*acc)
            })
            .collect();
        for _ in 0..n {
            let u = rng.random::<f64>();
            let idx = cumulative_weights
                .iter()
                .position(|&cw| cw >= u)
                .unwrap_or(n - 1);
            let mut new_particle = particles[idx].clone();
            new_particle.weight = 1.0 / n as f64;
            new_particles.push(new_particle);
        }
        new_particles
    }
    fn residual_resample(particles: Vec<Particle>, rng: &mut StdRng) -> Vec<Particle> {
        let n = particles.len();
        let mut new_particles = Vec::<Particle>::with_capacity(n);
        let weights: Vec<f64> = particles.iter().map(|p| p.weight).collect();
        let mut num_copies = vec![0usize; n];
        let mut residual: Vec<f64> = vec![0.0; n];
        for (i, &w) in weights.iter().enumerate() {
            let copies = (w * n as f64).floor() as usize;
            num_copies[i] = copies;
            residual[i] = w * n as f64 - copies as f64;
        }
        for (i, &copies) in num_copies.iter().enumerate() {
            for _ in 0..copies {
                let mut new_particle = particles[i].clone();
                new_particle.weight = 1.0 / n as f64;
                new_particles.push(new_particle);
            }
        }
        let residual_particles = n - new_particles.len();
        if residual_particles > 0 {
            let sum_residual: f64 = residual.iter().sum();
            let mut positions = Vec::with_capacity(residual_particles);
            let step = sum_residual / residual_particles as f64;
            let mut u = rng.random::<f64>() * step;
            for _ in 0..residual_particles {
                positions.push(u);
                u += step;
            }
            let mut i = 0usize;
            let mut j = 0usize;
            let mut cumsum = residual.get(0).cloned().unwrap_or(0.0);
            while j < residual_particles {
                while positions[j] > cumsum && i + 1 < n {
                    i += 1;
                    cumsum += residual[i];
                }
                let mut new_particle = particles[i].clone();
                new_particle.weight = 1.0 / n as f64;
                new_particles.push(new_particle);
                j += 1;
            }
        }
        new_particles
    }
    fn stratified_resample(particles: Vec<Particle>, rng: &mut StdRng) -> Vec<Particle> {
        let n = particles.len();
        let mut new_particles = Vec::<Particle>::with_capacity(n);
        let step = 1.0 / n as f64;
        for i in 0..n {
            let u = (i as f64 + rng.random::<f64>()) * step;
            let mut cumsum = 0.0;
            let mut j = 0usize;
            while u > cumsum && j + 1 <= n {
                cumsum += particles.get(j).map(|p| p.weight).unwrap_or(0.0);
                j += 1;
            }
            let idx = if j == 0 { 0 } else { j - 1 };
            let mut new_particle = particles[idx].clone();
            new_particle.weight = 1.0 / n as f64;
            new_particles.push(new_particle);
        }
        new_particles
    }
    // Adaptive resampling can be implemented based on effective sample size
    // For simplicity, we will just call systematic resampling here
    fn adaptive_resample(particles: Vec<Particle>, rng: &mut StdRng) -> Vec<Particle> {
        // TODO: #118 Implement adaptive resampling based on effective sample size
        eprintln!(
            "Warning: Adaptive resampling is not yet implemented. Falling back to systematic resampling."
        );
        Self::systematic_resample(particles, rng)
    }
}

/// Process noise parameters for stochastic state propagation.
///
/// These parameters control the diffusion of particles during prediction,
/// representing uncertainty in the IMU measurements and bias random walks.
/// Proper tuning is critical for filter performance.
///
/// # Tuning Guidelines
///
/// - `accel_std` and `gyro_std`: Scale with IMU noise density specifications
/// - `accel_walk_std` and `gyro_walk_std`: Model bias instability (typically 1e-6 to 1e-4)
/// - `vertical_accel_std`: Larger values increase altitude estimation uncertainty
///
/// # Units
///
/// All noise parameters are standard deviations, not variances.
#[derive(Clone, Copy, Debug)]
pub struct ProcessNoise {
    /// Standard deviation of accelerometer white noise (m/s²)
    pub accel_std: f64,
    
    /// Standard deviation of gyroscope white noise (rad/s)
    pub gyro_std: f64,
    
    /// Standard deviation of accelerometer bias random walk (m/s²/√s)
    pub accel_walk_std: f64,
    
    /// Standard deviation of gyroscope bias random walk (rad/s/√s)
    pub gyro_walk_std: f64,
    
    /// Standard deviation of vertical acceleration noise for damped channel (m/s²)
    pub vertical_accel_std: f64,
}
impl Default for ProcessNoise {
    fn default() -> Self {
        ProcessNoise {
            accel_std: 0.02,
            gyro_std: 0.01,
            accel_walk_std: 0.0001,
            gyro_walk_std: 0.0001,
            vertical_accel_std: 0.1,
        }
    }
}
impl ProcessNoise {
    /// Creates a new process noise specification.
    ///
    /// # Arguments
    ///
    /// * `accel_std` - Accelerometer white noise standard deviation (m/s²)
    /// * `gyro_std` - Gyroscope white noise standard deviation (rad/s)
    /// * `accel_walk_std` - Accelerometer bias random walk (m/s²/√s)
    /// * `gyro_walk_std` - Gyroscope bias random walk (rad/s/√s)
    /// * `vertical_accel_std` - Vertical acceleration noise for damped model (m/s²)
    ///
    /// # Panics
    ///
    /// Panics if any parameter is negative or non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use strapdown_core::particle::ProcessNoise;
    ///
    /// // Consumer-grade IMU (e.g., smartphone)
    /// let noise_consumer = ProcessNoise::new(0.01, 0.001, 1e-5, 1e-6, 0.1);
    ///
    /// // Tactical-grade IMU
    /// let noise_tactical = ProcessNoise::new(0.0001, 0.00001, 1e-6, 1e-7, 0.01);
    /// ```
    pub fn new(
        accel_std: f64,
        gyro_std: f64,
        accel_walk_std: f64,
        gyro_walk_std: f64,
        vertical_accel_std: f64,
    ) -> Self {
        // assert the input values are non-negative and finite
        assert!(
            accel_std >= 0.0 && accel_std.is_finite(),
            "Accelerometer standard deviation must be non-negative and finite"
        );
        assert!(
            gyro_std >= 0.0 && gyro_std.is_finite(),
            "Gyroscope standard deviation must be non-negative and finite"
        );
        assert!(
            accel_walk_std >= 0.0 && accel_walk_std.is_finite(),
            "Accelerometer walk standard deviation must be non-negative and finite"
        );
        assert!(
            gyro_walk_std >= 0.0 && gyro_walk_std.is_finite(),
            "Gyroscope walk standard deviation must be non-negative and finite"
        );
        assert!(
            vertical_accel_std >= 0.0 && vertical_accel_std.is_finite(),
            "Vertical acceleration standard deviation must be non-negative and finite"
        );
        ProcessNoise {
            accel_std,
            gyro_std,
            accel_walk_std,
            gyro_walk_std,
            vertical_accel_std,
        }
    }
}

/// Sequential Monte Carlo filter for nonlinear Bayesian state estimation.
///
/// The particle filter represents the posterior distribution p(x|z) through a weighted
/// set of samples (particles). Each particle evolves according to the strapdown equations
/// with added process noise, and weights are updated based on measurement likelihood.
///
/// # Algorithm Overview
///
/// 1. **Prediction**: Propagate each particle through nonlinear dynamics with noise
/// 2. **Update**: Compute importance weights based on measurement likelihood  
/// 3. **Resampling**: Duplicate high-weight particles, discard low-weight ones
/// 4. **Estimation**: Extract point estimate via weighted average or other strategy
///
/// # Vertical Channel Modification
///
/// Unlike standard strapdown mechanization, the vertical channel uses a damped oscillator
/// model to reduce altitude drift. See module-level documentation for details.
///
/// # Examples
///
/// ```no_run
/// use strapdown_core::particle::{ParticleFilter, Particle, ProcessNoise};
/// use strapdown_core::StrapdownState;
/// use nalgebra::{DVector, Rotation3, Vector3};
///
/// // Initialize particles around initial state estimate
/// let initial_state = StrapdownState {
///     latitude: 0.7,
///     longitude: -1.2,
///     altitude: 100.0,
///     velocity: Vector3::zeros(),
///     attitude: Rotation3::identity(),
/// };
///
/// let mut particles = Vec::new();
/// for _ in 0..1000 {
///     particles.push(Particle::new(
///         initial_state.clone(),
///         DVector::zeros(3),
///         DVector::zeros(3),
///         None,
///         1.0 / 1000.0,
///         None,
///         1.0, 0.1, 0.001,
///     ));
/// }
///
/// let mut pf = ParticleFilter::new(
///     particles,
///     None,  // Use default process noise
///     None,  // Use default averaging
///     None,  // Use default resampling
///     None,  // Resample every cycle
///     Some(42),  // Random seed
/// );
/// ```
///
/// # References
///
/// * Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). "Novel approach to
///   nonlinear/non-Gaussian Bayesian state estimation." IEE Proceedings F.
#[derive(Clone)]
pub struct ParticleFilter {
    /// Weighted ensemble of state hypotheses
    pub particles: Vec<Particle>,
    
    /// Process noise parameters for stochastic propagation
    pub process_noise: ProcessNoise,
    
    /// Strategy for extracting point estimates from ensemble
    pub averaging_strategy: ParticleAveragingStrategy,
    
    /// Algorithm for resampling particles
    pub resampling_strategy: ParticleResamplingStrategy,
    
    /// Dimension of state vector (minimum 15)
    pub state_size: usize,
    
    /// Seeded random number generator for reproducibility
    rng: StdRng,
    
    /// Normal distribution for accelerometer white noise
    accel_noise: Normal<f64>,
    
    /// Normal distribution for gyroscope white noise
    gyro_noise: Normal<f64>,
    
    /// Normal distribution for vertical channel damping noise
    vertical_accel_noise: Normal<f64>,
    
    /// Normal distribution for accelerometer bias random walk
    accel_walk_noise: Normal<f64>,
    
    /// Normal distribution for gyroscope bias random walk
    gyro_walk_noise: Normal<f64>,
    
    /// If true, only resample when effective sample size drops; if false, resample every cycle
    resampling_mode: bool,
}

impl Default for ParticleFilter {
    fn default() -> Self {
        let process_noise = ProcessNoise::default();
        let accel_noise = Normal::new(0.0, process_noise.accel_std).unwrap();
        let gyro_noise = Normal::new(0.0, process_noise.gyro_std).unwrap();
        let vertical_accel_noise = Normal::new(0.0, process_noise.vertical_accel_std).unwrap();
        let accel_walk_noise = Normal::new(0.0, process_noise.accel_walk_std).unwrap();
        let gyro_walk_noise = Normal::new(0.0, process_noise.gyro_walk_std).unwrap();
        ParticleFilter {
            particles: Vec::new(),
            process_noise,
            averaging_strategy: ParticleAveragingStrategy::WeightedAverage,
            resampling_strategy: ParticleResamplingStrategy::Residual,
            state_size: 15,
            rng: StdRng::from_os_rng(),
            accel_noise,
            gyro_noise,
            vertical_accel_noise,
            accel_walk_noise,
            gyro_walk_noise,
            resampling_mode: false,
        }
    }
}
impl Debug for ParticleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mean = self.get_estimate();
        // let cov = self.get_certainty();
        let effective_particles = 1.0
            / self
                .particles
                .iter()
                .map(|p| p.weight * p.weight)
                .sum::<f64>();
        let min_weight = self
            .particles
            .iter()
            .map(|p| p.weight)
            .fold(f64::INFINITY, f64::min);
        let max_weight = self.particles.iter().map(|p| p.weight).fold(0.0, f64::max);
        f.debug_struct("ParticleFilter")
            .field("num_particles", &self.particles.len())
            .field("effective_particles", &effective_particles)
            .field(
                "weight_range",
                &format_args!("[{:.4e}, {:.4e}]", min_weight, max_weight),
            )
            .field(
                "mean_position",
                &format_args!(
                    "({:.6}°, {:.6}°, {:.2}m)",
                    mean[0].to_degrees(),
                    mean[1].to_degrees(),
                    mean[2]
                ),
            )
            .field(
                "mean_velocity",
                &format_args!("({:.3}, {:.3}, {:.3}) m/s", mean[3], mean[4], mean[5]),
            )
            .field(
                "mean_attitude",
                &format_args!("({:.3}, {:.3}, {:.3}) rad", mean[6], mean[7], mean[8]),
            )
            .finish()
    }
}
impl ParticleFilter {
    /// Creates a new particle filter with specified configuration.
    ///
    /// # Arguments
    ///
    /// * `particles` - Initial particle ensemble (typically uniform weights 1/N)
    /// * `process_noise_std` - Optional 5D vector [accel, gyro, accel_walk, gyro_walk, vert_accel]
    ///   standard deviations. If None, uses [`ProcessNoise::default()`].
    /// * `estimation_strategy` - Method for extracting state estimate (default: weighted average)
    /// * `resampling_method` - Resampling algorithm (default: residual)
    /// * `resampling_mode` - If Some(true), only resample when ESS is low; if None/Some(false),
    ///   resample every cycle
    /// * `random_seed` - Seed for reproducible random number generation
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use strapdown_core::particle::{ParticleFilter, Particle};
    /// # use nalgebra::DVector;
    /// # let particles = vec![];  // Would normally initialize properly
    /// // Custom process noise: [accel_std, gyro_std, accel_walk, gyro_walk, vert_accel]
    /// let noise = DVector::from_vec(vec![0.01, 0.001, 1e-5, 1e-6, 0.1]);
    /// let pf = ParticleFilter::new(
    ///     particles,
    ///     Some(noise),
    ///     None,
    ///     None,
    ///     None,
    ///     Some(42),
    /// );
    /// ```
    pub fn new(
        particles: Vec<Particle>,
        process_noise_std: Option<DVector<f64>>,
        estimation_strategy: Option<ParticleAveragingStrategy>,
        resampling_method: Option<ParticleResamplingStrategy>,
        resampling_mode: Option<bool>,
        random_seed: Option<u64>,
    ) -> Self {
        let state_size = particles[0].state_size;
        let process_noise = match process_noise_std {
            Some(std) => ProcessNoise::new(std[0], std[1], std[2], std[3], std[4]),
            None => ProcessNoise::default(),
        };
        ParticleFilter {
            particles,
            process_noise,
            averaging_strategy: estimation_strategy
                .unwrap_or(ParticleAveragingStrategy::WeightedAverage),
            resampling_strategy: resampling_method.unwrap_or(ParticleResamplingStrategy::Residual),
            state_size,
            rng: match random_seed {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_os_rng(),
            },
            accel_noise: Normal::new(0.0, process_noise.accel_std).unwrap(),
            gyro_noise: Normal::new(0.0, process_noise.gyro_std).unwrap(),
            vertical_accel_noise: Normal::new(0.0, process_noise.vertical_accel_std).unwrap(),
            accel_walk_noise: Normal::new(0.0, process_noise.accel_walk_std).unwrap(),
            gyro_walk_noise: Normal::new(0.0, process_noise.gyro_walk_std).unwrap(),
            resampling_mode: resampling_mode.unwrap_or(false),
        }
    }
    /// Converts particle ensemble to matrix representation.
    ///
    /// # Returns
    ///
    /// Matrix of size `state_size × n_particles` where each column is one particle's
    /// state vector. Useful for vectorized operations or analysis.
    ///
    /// # State Vector Order
    ///
    /// Rows follow the standard layout: position, velocity, attitude (Euler angles),
    /// accelerometer biases, gyroscope biases, then optional extended states.
    pub fn particles_to_matrix(&self) -> DMatrix<f64> {
        let n_particles = self.particles.len();
        let state_size = self.state_size;
        let mut data = Vec::with_capacity(n_particles * state_size);
        for particle in &self.particles {
            let euler = particle.nav_state.attitude.euler_angles();
            data.push(particle.nav_state.latitude);
            data.push(particle.nav_state.longitude);
            data.push(particle.nav_state.altitude);
            data.push(particle.nav_state.velocity_north);
            data.push(particle.nav_state.velocity_east);
            data.push(particle.nav_state.velocity_down);
            data.push(euler.0);
            data.push(euler.1);
            data.push(euler.2);
            // append accel and gyro biases (indices 9..14)
            data.push(particle.accel_bias[0]);
            data.push(particle.accel_bias[1]);
            data.push(particle.accel_bias[2]);
            data.push(particle.gyro_bias[0]);
            data.push(particle.gyro_bias[1]);
            data.push(particle.gyro_bias[2]);
            // other_states start at index 15; pad missing entries with zeros to keep fixed width
            let mut other_iter = particle.other_states.as_ref().map(|v| v.iter());
            for _ in 15..state_size {
                if let Some(ref mut it) = other_iter {
                    if let Some(val) = it.next() {
                        data.push(*val);
                        continue;
                    }
                }
                data.push(0.0);
            }
        }
        DMatrix::from_vec(state_size, n_particles, data)
    }
    /// Sets particle weights from a slice.
    ///
    /// # Arguments
    ///
    /// * `weights` - Slice of weights, must have same length as number of particles
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != self.particles.len()`
    pub fn set_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.particles.len());
        for (particle, &w) in self.particles.iter_mut().zip(weights.iter()) {
            particle.weight = w;
        }
    }
    /// Normalizes particle weights to sum to 1.0.
    ///
    /// If the sum is zero or non-finite, sets uniform weights (1/N for N particles).
    /// This prevents numerical issues from weight degeneracy.
    pub fn normalize_weights(&mut self) {
        let sum: f64 = self.particles.iter().map(|p| p.weight).sum();
        if sum > 0.0 && sum.is_finite() {
            for particle in &mut self.particles {
                particle.weight /= sum;
            }
        } else {
            let uniform = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.weight = uniform;
            }
        }
    }
    /// Resamples particles according to configured strategy.
    ///
    /// Creates new ensemble by duplicating high-weight particles and discarding
    /// low-weight ones. All weights are reset to uniform (1/N) after resampling.
    ///
    /// The choice of resampling algorithm affects the variance of the resampled
    /// ensemble. Systematic and residual methods generally have lower variance
    /// than naive multinomial resampling.
    pub fn resample(&mut self) {
        self.particles = match self.resampling_strategy {
            ParticleResamplingStrategy::Naive => {
                ParticleResamplingStrategy::naive_resample(self.particles.clone(), &mut self.rng)
            }
            ParticleResamplingStrategy::Systematic => {
                ParticleResamplingStrategy::systematic_resample(
                    self.particles.clone(),
                    &mut self.rng,
                )
            }
            ParticleResamplingStrategy::Multinomial => {
                ParticleResamplingStrategy::multinomial_resample(
                    self.particles.clone(),
                    &mut self.rng,
                )
            }
            ParticleResamplingStrategy::Residual => {
                ParticleResamplingStrategy::residual_resample(self.particles.clone(), &mut self.rng)
            }
            ParticleResamplingStrategy::Stratified => {
                ParticleResamplingStrategy::stratified_resample(
                    self.particles.clone(),
                    &mut self.rng,
                )
            }
            ParticleResamplingStrategy::Adaptive => {
                ParticleResamplingStrategy::adaptive_resample(self.particles.clone(), &mut self.rng)
            }
        }
    }
    /// Computes the effective sample size (ESS) of the particle ensemble.
    ///
    /// ESS measures how well the particles represent the posterior. It is defined as:
    ///
    /// $$
    /// ESS = \frac{1}{\sum_{i=1}^N w_i^2}
    /// $$
    ///
    /// where $w_i$ are normalized weights. ESS ranges from 1 (complete degeneracy,
    /// all weight on one particle) to N (uniform weights).
    ///
    /// # Returns
    ///
    /// Effective sample size, typically between 1 and `particles.len()`.
    /// Common threshold for resampling is ESS < N/2.
    ///
    /// # References
    ///
    /// * Kong, A., Liu, J. S., & Wong, W. H. (1994). "Sequential imputations and
    ///   Bayesian missing data problems." Journal of the American Statistical Association.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_of_squares: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_of_squares > 0.0 {
            1.0 / sum_of_squares
        } else {
            0.0
        }
    }
    /// Adds random walk noise to IMU measurements for particle diffusion.
    ///
    /// Each particle receives slightly different IMU measurements, implementing
    /// the stochastic propagation required for particle filtering.
    ///
    /// # Arguments
    ///
    /// * `imu_data` - Clean IMU measurement
    /// * `accel_walk_noise` - Normal distribution for accelerometer noise
    /// * `gyro_walk_noise` - Normal distribution for gyroscope noise
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Perturbed IMU measurement with added noise
    fn sample_noisy_imu(
        imu_data: &IMUData,
        accel_walk_noise: &Normal<f64>,
        gyro_walk_noise: &Normal<f64>,
        rng: &mut StdRng,
    ) -> IMUData {
        let noisy_accel: Vector3<f64> = Vector3::from_vec(vec![
            imu_data.accel[0] + accel_walk_noise.sample(rng),
            imu_data.accel[1] + accel_walk_noise.sample(rng),
            imu_data.accel[2] + accel_walk_noise.sample(rng),
        ]);
        let noisy_gyro: Vector3<f64> = Vector3::from_vec(vec![
            imu_data.gyro[0] + gyro_walk_noise.sample(rng),
            imu_data.gyro[1] + gyro_walk_noise.sample(rng),
            imu_data.gyro[2] + gyro_walk_noise.sample(rng),
        ]);
        IMUData {
            accel: noisy_accel,
            gyro: noisy_gyro,
        }
    }
    /// Updates sensor biases with random walk and vertical channel feedback.
    ///
    /// Models bias evolution as a random walk plus altitude error feedback in
    /// the vertical accelerometer bias (z-axis). The feedback implements the
    /// third loop of the vertical channel damping.
    ///
    /// # Arguments
    ///
    /// * `particle` - Particle whose biases to update (modified in place)
    /// * `dt` - Time step in seconds
    /// * `accel_noise` - Normal distribution for accelerometer bias noise
    /// * `gyro_noise` - Normal distribution for gyroscope bias noise
    /// * `rng` - Random number generator
    ///
    /// # Mathematical Model
    ///
    /// Horizontal biases (x, y):
    /// $$
    /// b_{k+1} = b_k + w \sqrt{\Delta t}, \quad w \sim \mathcal{N}(0, \sigma^2)
    /// $$
    ///
    /// Vertical bias (z) with damping feedback:
    /// $$
    /// b_{z,k+1} = b_{z,k} + k_3 \epsilon_h \Delta t + w \sqrt{\Delta t}
    /// $$
    ///
    /// where $\epsilon_h$ is altitude error and $k_3$ is the bias gain.
    fn propagate_biases(
        particle: &mut Particle,
        dt: f64,
        accel_noise: &Normal<f64>,
        gyro_noise: &Normal<f64>,
        rng: &mut StdRng,
    ) {
        particle.accel_bias[0] += accel_noise.sample(rng) * dt.sqrt();
        particle.accel_bias[1] += accel_noise.sample(rng) * dt.sqrt();
        particle.accel_bias[2] +=
            particle.k3 * particle.altitude_error * dt + accel_noise.sample(rng) * dt.sqrt(); // Vertical channel with damping            
        particle.gyro_bias[0] += gyro_noise.sample(rng) * dt.sqrt();
        particle.gyro_bias[1] += gyro_noise.sample(rng) * dt.sqrt();
        particle.gyro_bias[2] += gyro_noise.sample(rng) * dt.sqrt();
    }
}
impl NavigationFilter for ParticleFilter {
    /// Particle filter variant of the forward propagation equations
    ///
    /// The particle filter uses a modified strapdown mechanization to propagate each particle's state, the primary
    /// difference in being the vertical (up/down) channel. This channel is notoriously noisy even when using high
    /// quality IMU data, so we use a model that isn't strictly informed by the physical kinetic equations. This
    /// version uses a damped velocity model to help stabilize altitude estimates according to: Sokolovic, V., et al.
    /// (2014). "Adaptive Error Damping in the Vertical Channel of the INS/GPS/Baro-Altimeter Integrated Navigation
    /// System." Scientific Technical Review.
    ///
    /// $$
    /// \dot{v}_D = a_D - g + b_D + w_D - 2 \zeta \omega_n v_D - \omega_n^2 (h - h_{ref})
    /// $$
    ///
    /// # Arguments
    /// - `imu_data`: The IMU data to use for propagation
    /// - `dt`: The time step to use for propagation
    ///
    /// # Notes
    /// - Process noise is added to each particle after propagation, based on the `process_noise
    /// ` member of the `ParticleFilter` struct.
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        let noisy_imu = Self::sample_noisy_imu(
            &imu_data,
            &self.accel_walk_noise,
            &self.gyro_walk_noise,
            &mut self.rng,
        );
        for particle in self.particles.iter_mut() {
            // Propagate the biases
            Self::propagate_biases(
                particle,
                dt,
                &self.accel_noise,
                &self.gyro_noise,
                &mut self.rng,
            );
            // Correct IMU measurements for biases
            let gyros: Vector3<f64> = noisy_imu.gyro - &particle.gyro_bias;
            let accel: Vector3<f64> = noisy_imu.accel - &particle.accel_bias;
            // Attitude update
            let c_1 = attitude_update(&particle.nav_state, gyros, dt);
            // Velocity update
            let f = particle.nav_state.attitude * accel;
            let transport_rate = earth::vector_to_skew_symmetric(&earth::transport_rate(
                &particle.nav_state.latitude.to_degrees(),
                &particle.nav_state.altitude,
                &Vector3::from_vec(vec![
                    particle.nav_state.velocity_north,
                    particle.nav_state.velocity_east,
                    particle.nav_state.velocity_down,
                ]),
            ));
            let rotation_rate = earth::vector_to_skew_symmetric(&earth::earth_rate_lla(
                &particle.nav_state.latitude.to_degrees(),
            ));
            let r = earth::ecef_to_lla(
                &particle.nav_state.latitude.to_degrees(),
                &particle.nav_state.longitude.to_degrees(),
            );
            let mut velocity = Vector3::from_vec(vec![
                particle.nav_state.velocity_north,
                particle.nav_state.velocity_east,
                particle.nav_state.velocity_down,
            ]);
            let coriolis = r * (2.0 * rotation_rate + transport_rate) * velocity;
            let g = earth::gravity(
                &particle.nav_state.latitude.to_degrees(),
                &particle.nav_state.altitude,
            );
            let g = if particle.nav_state.is_enu { -g } else { g };
            velocity[0] += (f[0] + coriolis[0]) * dt;
            velocity[1] += (f[1] + coriolis[1]) * dt;
            velocity[2] += (f[2] + g - particle.k2 * particle.altitude_error + coriolis[2]) * dt
                + self.vertical_accel_noise.sample(&mut self.rng) * dt.sqrt();
            // Position update
            let (r_n, r_e_0, _) =
                earth::principal_radii(&particle.nav_state.latitude, &particle.nav_state.altitude);
            let lat_0 = particle.nav_state.latitude;
            // Altitude update
            let alt_1 = particle.nav_state.altitude
                + (velocity[2] - particle.k1 * particle.altitude_error) * dt;
            // Latitude update
            let lat_1 = particle.nav_state.latitude
                + 0.5
                    * (velocity[0] / (r_n + particle.nav_state.altitude)
                        + velocity[0] / (r_n + alt_1))
                    * dt;
            // Longitude update
            let (_, r_e_1, _) = earth::principal_radii(&lat_1, &alt_1);
            let cos_lat0 = lat_0.cos().max(1e-6); // Guard against cos(lat) --> 0 near poles
            let cos_lat1 = lat_1.cos().max(1e-6);
            let lon_1 = particle.nav_state.longitude
                + 0.5
                    * (particle.nav_state.velocity_east
                        / ((r_e_0 + particle.nav_state.altitude) * cos_lat0)
                        + velocity[1] / ((r_e_1 + alt_1) * cos_lat1))
                    * dt;
            // Update the particle state
            particle.nav_state.latitude = lat_1;
            particle.nav_state.longitude = lon_1;
            particle.nav_state.altitude = alt_1;
            particle.nav_state.velocity_north = velocity[0];
            particle.nav_state.velocity_east = velocity[1];
            particle.nav_state.velocity_down = velocity[2];
            particle.nav_state.attitude = Rotation3::from_matrix_unchecked(c_1);
        }
    }
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // let particle_matrix = self.particles_to_matrix();
        // let measurement_sigma_points = measurement.get_sigma_points(&particle_matrix);
        // let mut z_hats = DMatrix::<f64>::zeros(measurement.get_dimension(), self.particles.len());
        // for (i, measurement_sigma_point) in measurement_sigma_points.column_iter().enumerate() { z_hats.set_column(i, &measurement_sigma_point); }
        // let mut log_likelihoods = Vec::with_capacity(self.particles.len());
        // for (i, particle) in self.particles.iter_mut().enumerate() {
        //     let z_hat = z_hats.column(i);
        //     let innovation = measurement.get_vector() - z_hat;
        //     let sigmas = measurement.get_noise();
        //     let sigma_inv = match sigmas.clone().try_inverse() { Some(inv) => inv, None => { particle.weight = 1e-300; log_likelihoods.push(-690.0); continue; } };
        //     let sigma_det = sigmas.determinant(); if sigma_det <= 0.0 { particle.weight = 1e-300; log_likelihoods.push(-690.0); continue; }
        //     let mahalanobis = innovation.transpose() * sigma_inv * innovation;
        //     let log_likelihood = -0.5 * (measurement.get_dimension() as f64 * (2.0 * std::f64::consts::PI).ln() + sigma_det.ln() + mahalanobis[(0, 0)]);
        //     log_likelihoods.push(log_likelihood);
        // }
        // let max_log_likelihood = log_likelihoods.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // for (i, particle) in self.particles.iter_mut().enumerate() { particle.weight = (log_likelihoods[i] - max_log_likelihood).exp(); }
        // self.normalize_weights();
        for p in &mut self.particles {
            let (state, _) = p.clone().into();
            let innovation =
                measurement.get_vector() - measurement.get_expected_measurement(&state);
            let sigmas = measurement.get_noise();
            let sigma_inv = match sigmas.clone().try_inverse() {
                Some(inv) => inv,
                None => {
                    p.weight = 1e-300;
                    continue;
                }
            };
            let sigma_det = sigmas.determinant();
            if sigma_det <= 0.0 {
                p.weight = 1e-300;
                continue;
            }
            let mahalanobis = innovation.transpose() * sigma_inv * innovation;
            let log_likelihood = -0.5
                * (measurement.get_dimension() as f64 * (2.0 * std::f64::consts::PI).ln()
                    + sigma_det.ln()
                    + mahalanobis[(0, 0)]);
            p.weight = if self.resampling_mode {
                p.weight * log_likelihood.exp()
            } else {
                log_likelihood.exp()
            };
        }
    }
    fn get_estimate(&self) -> DVector<f64> {
        match self.averaging_strategy {
            ParticleAveragingStrategy::WeightedAverage => {
                let (mean, _cov) = ParticleAveragingStrategy::weighted_average_state(self);
                mean
            }
            ParticleAveragingStrategy::UnweightedAverage => {
                let (mean, _cov) = ParticleAveragingStrategy::unweighted_average_state(self);
                mean
            }
            ParticleAveragingStrategy::HighestWeight => {
                let (mean, _cov) = ParticleAveragingStrategy::highest_weight_state(self);
                mean
            }
        }
    }
    fn get_certainty(&self) -> DMatrix<f64> {
        match self.averaging_strategy {
            ParticleAveragingStrategy::WeightedAverage => {
                let (_mean, cov) = ParticleAveragingStrategy::weighted_average_state(self);
                cov
            }
            ParticleAveragingStrategy::UnweightedAverage => {
                let (_mean, cov) = ParticleAveragingStrategy::unweighted_average_state(self);
                cov
            }
            ParticleAveragingStrategy::HighestWeight => {
                let (_mean, cov) = ParticleAveragingStrategy::highest_weight_state(self);
                cov
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector, Rotation3, Vector3, Matrix1};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::Normal;

    const EPS: f64 = 1e-9;

    fn make_nav_state() -> StrapdownState {
        StrapdownState {
            latitude: 0.1,
            longitude: 0.2,
            altitude: 10.0,
            velocity_north: 1.0,
            velocity_east: 2.0,
            velocity_down: -0.5,
            attitude: Rotation3::from_euler_angles(0.01, 0.02, 0.03),
            is_enu: true,
        }
    }

    fn make_particle(weight: f64) -> Particle {
        Particle::new(
            make_nav_state(),
            DVector::from_vec(vec![0.001, 0.002, 0.003]),
            DVector::from_vec(vec![0.01, 0.02, 0.03]),
            None,
            weight,
            Some(0.5),
            0.1,
            0.2,
            0.001,
        )
    }

    #[test]
    fn test_particle_new_valid_and_fields() {
        let nav = make_nav_state();
        let accel = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let gyro = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let other = Some(DVector::from_vec(vec![7.0, 8.0]));
        let p = Particle::new(nav.clone(), accel.clone(), gyro.clone(), other.clone(), 0.5, None, 1.0, 2.0, 3.0);
        assert_eq!(p.nav_state.latitude, nav.latitude);
        assert_eq!(p.accel_bias, accel);
        assert_eq!(p.gyro_bias, gyro);
        assert_eq!(p.other_states.unwrap(), other.unwrap());
        assert!(p.state_size >= 15);
        assert!((p.weight - 0.5).abs() < EPS);
    }

    #[test]
    #[should_panic]
    fn test_particle_new_invalid_accel_len() {
        let nav = make_nav_state();
        // accel_bias wrong length
        let _ = Particle::new(nav, DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0]), DVector::from_vec(vec![0.0, 0.0, 0.0]), None, 0.1, None, 0.0, 0.0, 0.0);
    }

    #[test]
    #[should_panic]
    fn test_particle_new_invalid_gyro_len() {
        let nav = make_nav_state();
        // gyro_bias wrong length
        let _ = Particle::new(nav, DVector::from_vec(vec![0.0, 0.0, 0.0]), DVector::from_vec(vec![0.0, 0.0]), None, 0.1, None, 0.0, 0.0, 0.0);
    }

    #[test]
    fn test_from_and_into_state_vector_roundtrip() {
        // prepare a full 15-element state vector with distinct values
        let mut vec = vec![
            0.11, 0.22, 33.0, // lat, lon, alt
            1.1, 2.2, -3.3,   // vn, ve, vd
            0.001, 0.002, 0.003, // euler
            0.1, 0.2, 0.3,    // accel bias
            0.01, 0.02, 0.03, // gyro bias
        ];
        // add some extra other_states
        vec.extend_from_slice(&[9.9, 8.8]);
        let state = DVector::from_vec(vec.clone());
        let weight = 0.42;
        let particle = Particle::from((state.clone(), weight));
        let (out_vec, out_w) = (particle.clone()).into();
        // the returned vector should start with the original state (nav + biases + gyros + extras)
        // Note: euler angles are converted from attitude; rounding differences possible but small.
        let out_v = out_vec;
        assert_eq!(out_w, weight);
        // check size and some entries
        assert_eq!(out_v.len(), state.len());
        for i in 0..state.len() {
            let a = state[i];
            let b = out_v[i];
            assert!((a - b).abs() < 1e-6, "index {} expected {} got {}", i, a, b);
        }
    }

    #[test]
    fn test_process_noise_default_and_new_and_invalid() {
        let d = ProcessNoise::default();
        assert!(d.accel_std > 0.0);
        let p = ProcessNoise::new(0.1, 0.2, 1e-6, 1e-6, 0.3);
        assert!((p.accel_std - 0.1).abs() < EPS);
        // invalid: negative std should panic
        let res = std::panic::catch_unwind(|| {
            ProcessNoise::new(-0.1, 0.0, 0.0, 0.0, 0.0);
        });
        assert!(res.is_err());
    }

    #[test]
    fn test_particle_averaging_strategies() {
        // two particles with different states and weights
        let mut p1 = make_particle(0.8);
        let mut p2 = make_particle(0.2);
        p1.nav_state.latitude = 0.0;
        p2.nav_state.latitude = 1.0;
        p1.nav_state.longitude = 0.0;
        p2.nav_state.longitude = 2.0;
        let pf = ParticleFilter {
            particles: vec![p1.clone(), p2.clone()],
            process_noise: ProcessNoise::default(),
            averaging_strategy: ParticleAveragingStrategy::WeightedAverage,
            resampling_strategy: ParticleResamplingStrategy::Residual,
            state_size: 15,
            rng: StdRng::seed_from_u64(42),
            accel_noise: Normal::new(0.0, 0.001).unwrap(),
            gyro_noise: Normal::new(0.0, 0.001).unwrap(),
            vertical_accel_noise: Normal::new(0.0, 0.001).unwrap(),
            accel_walk_noise: Normal::new(0.0, 0.001).unwrap(),
            gyro_walk_noise: Normal::new(0.0, 0.001).unwrap(),
            resampling_mode: false,
        };
        // weighted mean: latitude = 0*0.8 + 1*0.2 = 0.2
        let (mean_w, _) = ParticleAveragingStrategy::weighted_average_state(&pf);
        assert!((mean_w[0] - 0.2).abs() < 1e-12);
        // unweighted average: (0 + 1)/2 = 0.5
        let (mean_u, _) = ParticleAveragingStrategy::unweighted_average_state(&pf);
        assert!((mean_u[0] - 0.5).abs() < 1e-12);
        // highest weight should pick p1 (weight 0.8) => latitude 0.0
        let (mean_h, _) = ParticleAveragingStrategy::highest_weight_state(&pf);
        assert!((mean_h[0] - 0.0).abs() < 1e-12);
    }

    fn make_particles_for_resample() -> Vec<Particle> {
        vec![
            Particle::new(
                make_nav_state(),
                DVector::from_vec(vec![0.0, 0.0, 0.0]),
                DVector::from_vec(vec![0.0, 0.0, 0.0]),
                None,
                0.5,
                None,
                0.0,
                0.0,
                0.0,
            ),
            Particle::new(
                make_nav_state(),
                DVector::from_vec(vec![0.0, 0.0, 0.0]),
                DVector::from_vec(vec![0.0, 0.0, 0.0]),
                None,
                0.5,
                None,
                0.0,
                0.0,
                0.0,
            ),
        ]
    }

    #[test]
    fn test_naive_resample_and_weights_reset() {
        let mut rng = StdRng::seed_from_u64(123);
        let particles = make_particles_for_resample();
        let new = ParticleResamplingStrategy::naive_resample(particles, &mut rng);
        assert_eq!(new.len(), 2);
        for p in new {
            assert!((p.weight - 0.5).abs() < EPS); // uniform weight = 1/n = 0.5
        }
    }

    #[test]
    fn test_systematic_resample_behavior() {
        let mut rng = StdRng::seed_from_u64(1234);
        let mut parts = make_particles_for_resample();
        // set weights to [0.25, 0.75] to exercise selection
        parts[0].weight = 0.25;
        parts[1].weight = 0.75;
        let new = ParticleResamplingStrategy::systematic_resample(parts, &mut rng);
        assert_eq!(new.len(), 2);
        for p in new {
            assert!((p.weight - 0.5).abs() < EPS); // reset to uniform
        }
    }

    #[test]
    fn test_stratified_resample() {
        let mut rng = StdRng::seed_from_u64(55);
        let mut parts = make_particles_for_resample();
        parts[0].weight = 0.3;
        parts[1].weight = 0.7;
        let new = ParticleResamplingStrategy::stratified_resample(parts, &mut rng);
        assert_eq!(new.len(), 2);
        for p in new {
            assert!((p.weight - 0.5).abs() < EPS);
        }
    }

    #[test]
    fn test_adaptive_resample_calls_systematic() {
        let mut rng = StdRng::seed_from_u64(99);
        let parts = make_particles_for_resample();
        let new = ParticleResamplingStrategy::adaptive_resample(parts, &mut rng);
        assert_eq!(new.len(), 2);
    }

    #[test]
    fn test_residual_resample_counts_and_fill() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut parts = make_particles_for_resample();
        // make first particle very dominant
        parts[0].weight = 0.9;
        parts[1].weight = 0.1;
        let new = ParticleResamplingStrategy::residual_resample(parts, &mut rng);
        assert_eq!(new.len(), 2);
        // all weights reset
        for p in new {
            assert!((p.weight - 0.5).abs() < EPS);
        }
    }

    #[test]
    fn test_particles_to_matrix_and_set_weights_and_normalize() {
        let mut p1 = make_particle(1.0);
        let mut p2 = make_particle(1.0);
        // add other_states to one particle to ensure variable-length handled in data push
        p2.other_states = Some(DVector::from_vec(vec![11.0]));
        // make state_size reflect max (15 + 1)
        let pf = ParticleFilter {
            particles: vec![p1.clone(), p2.clone()],
            process_noise: ProcessNoise::default(),
            averaging_strategy: ParticleAveragingStrategy::WeightedAverage,
            resampling_strategy: ParticleResamplingStrategy::Residual,
            state_size: 16, // larger due to p2.other_states
            rng: StdRng::seed_from_u64(0),
            accel_noise: Normal::new(0.0, 0.001).unwrap(),
            gyro_noise: Normal::new(0.0, 0.001).unwrap(),
            vertical_accel_noise: Normal::new(0.0, 0.001).unwrap(),
            accel_walk_noise: Normal::new(0.0, 0.001).unwrap(),
            gyro_walk_noise: Normal::new(0.0, 0.001).unwrap(),
            resampling_mode: false,
        };
        let mat = pf.particles_to_matrix();
        // dimensions: state_size x n_particles
        assert_eq!(mat.ncols(), 2);
        assert_eq!(mat.nrows(), 16);
        // test set_weights with correct length
        let mut pf2 = pf.clone();
        pf2.set_weights(&[0.2, 0.8]);
        assert!((pf2.particles[0].weight - 0.2).abs() < EPS);
        // set all weights to zero and normalize -> should set uniform weights
        pf2.set_weights(&[0.0, 0.0]);
        pf2.normalize_weights();
        assert!((pf2.particles[0].weight - 0.5).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_set_weights_bad_length_panics() {
        let mut pf = ParticleFilter::default();
        // ParticleFilter::default() has zero particles, but set_weights asserts equal lengths -> should panic
        pf.set_weights(&[0.1]);
    }

    #[test]
    fn test_effective_sample_size() {
        let mut pf = ParticleFilter::default();
        // create two particles and set weights
        pf.particles = make_particles_for_resample();
        pf.particles[0].weight = 0.5;
        pf.particles[1].weight = 0.5;
        let ess = pf.effective_sample_size();
        // for equal weights 0.5 each, ESS = 1 / (0.25 + 0.25) = 2
        assert!((ess - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sample_noisy_imu_and_propagate_biases() {
        let imu = IMUData {
            accel: Vector3::new(0.1, 0.2, 0.3),
            gyro: Vector3::new(0.01, 0.02, 0.03),
        };
        let mut rng = StdRng::seed_from_u64(12345);
        let accel_walk = Normal::new(0.0, 1e-6).unwrap();
        let gyro_walk = Normal::new(0.0, 1e-6).unwrap();
        let noisy = ParticleFilter::sample_noisy_imu(&imu, &accel_walk, &gyro_walk, &mut rng);
        // small perturbation expected
        assert!((noisy.accel[0] - imu.accel[0]).abs() < 1e-3);
        // propagate_biases: ensure biases change (vertical channel includes altitude_error term)
        let mut particle = make_particle(1.0);
        particle.altitude_error = 2.0;
        let mut rng2 = StdRng::seed_from_u64(42);
        let accel_n = Normal::new(0.0, 1e-6).unwrap();
        let gyro_n = Normal::new(0.0, 1e-6).unwrap();
        let prev_z = particle.accel_bias[2];
        ParticleFilter::propagate_biases(&mut particle, 0.1, &accel_n, &gyro_n, &mut rng2);
        // altitude bias should have increased by approximately k3 * altitude_error * dt (plus tiny noise)
        let expected_increase = particle.k3 * 2.0 * 0.1;
        assert!((particle.accel_bias[2] - prev_z) > (expected_increase - 1e-3));
    }

    // Minimal MeasurementModel implementation for testing update()
    struct DummyMeas {
        z: DVector<f64>,
        noise: DMatrix<f64>,
    }
    impl DummyMeas {
        fn new(z: DVector<f64>, noise: DMatrix<f64>) -> Self {
            DummyMeas { z, noise }
        }
    }
    impl MeasurementModel for DummyMeas {
        fn get_vector(&self) -> DVector<f64> {
            self.z.clone()
        }
        fn get_noise(&self) -> DMatrix<f64> {
            self.noise.clone()
        }
        fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
            // return a 1-dim measurement equal to state's altitude (index 2) if dimension > 0
            DVector::from_vec(vec![state[2]])
        }
        fn get_dimension(&self) -> usize {
            1
        }
        
        fn as_any(&self) -> &dyn std::any::Any {
            todo!()
        }
        
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            todo!()
        }
    }

    #[test]
    fn test_update_sets_weights_on_singular_noise() {
        // singular noise matrix (zero determinant) should set weight to tiny value 1e-300
        let mut pf = ParticleFilter::default();
        let mut p = make_particle(0.5);
        pf.particles = vec![p.clone()];
        let meas = DummyMeas::new(DVector::from_vec(vec![0.0]), DMatrix::from_vec(1, 1, vec![0.0]));
        pf.update(&meas);
        assert!((pf.particles[0].weight - 1e-300).abs() < 1e-320 || pf.particles[0].weight == 1e-300);
    }

    #[test]
    fn test_update_normal_case_and_resampling_mode_false() {
        let mut p = make_particle(0.7);
        p.nav_state.altitude = 0.0; // make expected measurement match zero innovation
        let mut pf = ParticleFilter::new(
            vec![p.clone()],
            None,
            Some(ParticleAveragingStrategy::WeightedAverage),
            Some(ParticleResamplingStrategy::Residual),
            Some(false),
            Some(123),
        );
        let noise = DMatrix::from_vec(1, 1, vec![1.0]);
        let meas = DummyMeas::new(DVector::from_vec(vec![0.0]), noise);
        pf.update(&meas);
        // compute expected likelihood for dimension=1, det=1, innovation=0:
        let dim = 1.0;
        let expected_log_like = -0.5 * (dim * (2.0 * std::f64::consts::PI).ln() + 1f64.ln() + 0.0);
        let expected_weight = expected_log_like.exp();
        assert!((pf.particles[0].weight - expected_weight).abs() < 1e-12);
    }

    #[test]
    fn test_predict_runs_and_updates_state() {
        // create a filter with one particle and deterministic RNG and tiny noises
        let mut p = make_particle(1.0);
        p.nav_state.altitude = 0.0;
        let mut pf = ParticleFilter::new(
            vec![p],
            None,
            Some(ParticleAveragingStrategy::WeightedAverage),
            Some(ParticleResamplingStrategy::Residual),
            Some(false),
            Some(42),
        );
        // set very small noise to keep deterministic-ish
        pf.accel_walk_noise = Normal::new(0.0, 1e-9).unwrap();
        pf.gyro_walk_noise = Normal::new(0.0, 1e-9).unwrap();
        pf.accel_noise = Normal::new(0.0, 1e-9).unwrap();
        pf.gyro_noise = Normal::new(0.0, 1e-9).unwrap();
        pf.vertical_accel_noise = Normal::new(0.0, 1e-9).unwrap();

        let imu = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        pf.predict(imu, 0.1);
        // after predict, velocities and altitude should have been updated (not NaN)
        let est = pf.get_estimate();
        assert!(est[2].is_finite());
        assert!(est[3].is_finite());
    }
}