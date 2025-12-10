//! Particle-filter style navigation code extracted from the former `filter.rs`.
use crate::kalman::NavigationFilter;
use crate::measurements::MeasurementModel;
use crate::{IMUData, StrapdownState, attitude_update, velocity_update, position_update};
use crate::earth;

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use rand::{self, RngCore, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::fmt::{self, Debug};

// Tunable Gains (Third-Order Loop)
// These need to be tuned to the response time of your vehicle
const K1: f64 = 1.0;       // Position Error Gain (1/s)
const K2: f64 = 0.1;       // Velocity Error Gain (1/s^2)
const K3: f64 = 0.001;     // Bias Error Gain (1/s^3) - Keep this very small!

#[derive(Clone, Debug, Default)]
pub struct Particle {
    /// The navigation state of the particle
    pub nav_state: StrapdownState,
    /// Accelerometer biases
    pub accel_bias: DVector<f64>,
    /// Gyroscope biases
    pub gyro_bias: DVector<f64>,
    /// Additional states beyond the navigation state and accel/gyro biases
    pub other_states: Option<DVector<f64>>,
    /// The size of the state vector
    pub state_size: usize,
    /// The weight of the particle
    pub weight: f64,
    /// The estimated error in the altitude measurement (used as a measurement, not a state)
    pub altitude_error: f64, 
    /// Vertical position rate gain (first loop)
    pub k1: f64,
    /// Vertical velocity rate gain (second loop)
    pub k2: f64,
    /// Vertical acceleration bias rate gain (third loop)
    pub k3: f64,
}
impl Particle {
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
        let state_size = 15 + match &other_states {
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
        let accel_bias = DVector::from_vec(vec![
            state_vector[9],
            state_vector[10],
            state_vector[11],
        ]);
        let gyro_bias = DVector::from_vec(vec![
            state_vector[12],
            state_vector[13],
            state_vector[14],
        ]);
        let other_states = if state_vector.len() > 15 {
            Some(state_vector.rows(15, state_vector.len() - 15).clone_owned())
        } else {
            None
        };
        Particle::new(nav_state, accel_bias, gyro_bias, other_states, weight, None, 0.0, 0.0, 0.0)
    }
}
impl Into<(DVector<f64>, f64)> for Particle {
    fn into(self) -> (DVector<f64>, f64) {
        let mut state_vec = vec![
            self.nav_state.latitude,
            self.nav_state.longitude,
            self.nav_state.altitude,
            self.nav_state.velocity_north,
            self.nav_state.velocity_east,
            self.nav_state.velocity_down,
            self.nav_state.attitude.euler_angles().0,
            self.nav_state.attitude.euler_angles().1,
            self.nav_state.attitude.euler_angles().2,
        ];
        state_vec.push(self.accel_bias[0]);
        state_vec.push(self.accel_bias[1]);
        state_vec.push(self.accel_bias[2]);
        state_vec.push(self.gyro_bias[0]);
        state_vec.push(self.gyro_bias[1]);
        state_vec.push(self.gyro_bias[2]);
        if let Some(other_states) = self.other_states {
            state_vec.extend(other_states.iter());
        }
        (DVector::from_vec(state_vec), self.weight)
    }
}
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum ParticleAveragingStrategy {
    #[default]
    WeightedAverage,
    UnweightedAverage,
    HighestWeight,
}
/// Trait for implementing averaging strategies
pub trait Average: Send + Sync {
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
        ];
        if let Some(ref other_states) = best_particle.other_states {
            state_vec.extend(other_states.iter());
        }
        let mean = DVector::from_vec(state_vec);
        let cov = DMatrix::<f64>::zeros(pf.state_size, pf.state_size);
        (mean, cov)
    }
}

#[derive(Clone, Debug, Default)]
pub enum ParticleResamplingStrategy {
    Naive,
    Systematic,
    Multinomial,
    #[default]
    Residual,
    Stratified,
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
        let step = 1.0 / n as f64;
        let mut u = rng.next_u64() as f64 * step;
        let mut i = 0;
        let mut cumsum = particles[0].weight;
        for _ in 0..n {
            while u > cumsum {
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
        let cumulative_weights: Vec<f64> = particles
            .iter()
            .scan(0.0, |acc, p| {
                *acc += p.weight;
                Some(*acc)
            })
            .collect();
        for _ in 0..n {
            let u = rand::random::<f64>();
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
            let mut u = rng.next_u64() as f64 * step;
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
            let u = (i as f64 + rng.next_u64() as f64) * step;
            let mut cumsum = 0.0;
            let mut j = 0;
            while u > cumsum {
                cumsum += particles[j].weight;
                j += 1;
            }
            let mut new_particle = particles[j - 1].clone();
            new_particle.weight = 1.0 / n as f64;
            new_particles.push(new_particle);
        }
        new_particles
    }
    // Adaptive resampling can be implemented based on effective sample size
    // For simplicity, we will just call systematic resampling here
    fn adaptive_resample(particles: Vec<Particle>, rng: &mut StdRng) -> Vec<Particle> {
        // TODO: #118 Implement adaptive resampling based on effective sample size
        eprintln!("Warning: Adaptive resampling is not yet implemented. Falling back to systematic resampling.");
        Self::systematic_resample(particles, rng)
    }       
}

#[derive(Clone, Copy, Debug)]
pub struct ProcessNoise {
    /// Standard deviation of accelerometer noise (m/s^2)
    pub accel_std: f64,
    /// Standard deviation of gyroscope noise (rad/s)
    pub gyro_std: f64,
    /// Standard deviation of accelerometer walk noise (m/s^2)
    pub accel_walk_std: f64,
    /// Standard deviation of gyroscope walk noise (rad/s)
    pub gyro_walk_std: f64,
    /// Standard deviation of vertical velocity noise (m/s)
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

#[derive(Clone)]
pub struct ParticleFilter {
    /// The particles in the filter
    pub particles: Vec<Particle>,
    /// The process noise covariance matrix diagonal for the filter (aka jitter) 
    pub process_noise: ProcessNoise,
    /// The strategy for averaging particles to get the state estimate
    pub averaging_strategy: ParticleAveragingStrategy,
    /// The strategy for resampling particles
    pub resampling_strategy: ParticleResamplingStrategy,
    /// The size of the state vector
    pub state_size: usize,
    /// The random number generator
    rng: StdRng,
    /// Accelerometer noise random distribution
    accel_noise: Normal<f64>,
    /// Gyroscope noise random distribution
    gyro_noise: Normal<f64>,
    /// Vertical acceleration noise random distribution
    vertical_accel_noise: Normal<f64>,
    /// Accelerometer walk noise random distribution
    accel_walk_noise: Normal<f64>,
    /// Gyroscope walk noise random distribution
    gyro_walk_noise: Normal<f64>,
    /// Resampling mode (use effective sample size to trigger resampling; defaults to false and resamples every cycle)
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
            Some(std) => ProcessNoise::new(
                std[0],
                std[1],
                std[2],
                std[3],
                std[4],
            ),
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
            if let Some(ref other_states) = particle.other_states {
                for val in other_states.iter() {
                    data.push(*val);
                }
            }
        }
        DMatrix::from_vec(state_size, n_particles, data)
    }
    pub fn set_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.particles.len());
        for (particle, &w) in self.particles.iter_mut().zip(weights.iter()) {
            particle.weight = w;
        }
    }
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
    pub fn resample(&mut self) {
        self.particles = match self.resampling_strategy {
            ParticleResamplingStrategy::Naive => ParticleResamplingStrategy::naive_resample(self.particles.clone(), &mut self.rng),
            ParticleResamplingStrategy::Systematic => ParticleResamplingStrategy::systematic_resample(self.particles.clone(), &mut self.rng),
            ParticleResamplingStrategy::Multinomial => ParticleResamplingStrategy::multinomial_resample(self.particles.clone(), &mut self.rng),
            ParticleResamplingStrategy::Residual => ParticleResamplingStrategy::residual_resample(self.particles.clone(), &mut self.rng),
            ParticleResamplingStrategy::Stratified => ParticleResamplingStrategy::stratified_resample(self.particles.clone(), &mut self.rng),
            ParticleResamplingStrategy::Adaptive => ParticleResamplingStrategy::adaptive_resample(self.particles.clone(), &mut self.rng),
        }
    }
    pub fn effective_sample_size(&self) -> f64 {
        let sum_of_squares: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_of_squares > 0.0 {
            1.0 / sum_of_squares
        } else {
            0.0
        }
    }
    fn sample_noisy_imu(imu_data: &IMUData,
        accel_walk_noise: &Normal<f64>,
        gyro_walk_noise: &Normal<f64>,
        rng: &mut StdRng,) -> IMUData {
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
    fn propagate_biases(
        particle: &mut Particle,
        dt: f64,
        accel_noise: &Normal<f64>,
        gyro_noise: &Normal<f64>,
        rng: &mut StdRng,
    ) {
        particle.accel_bias[0] += accel_noise.sample(rng) * dt.sqrt();
        particle.accel_bias[1] += accel_noise.sample(rng) * dt.sqrt();
        particle.accel_bias[2] += particle.k3 * particle.altitude_error * dt + accel_noise.sample(rng) * dt.sqrt(); // Vertical channel with damping            
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
            let gyros: Vector3<f64> = &noisy_imu.accel - &particle.gyro_bias;
            let accel: Vector3<f64> = &noisy_imu.gyro - &particle.accel_bias;
            // Attitude update
            let c_1 = attitude_update(&mut particle.nav_state, gyros, dt); 
            // Attitude update
            let c_1 = attitude_update(&mut particle.nav_state, gyros, dt); 
            // Velocity update
            let f = particle.nav_state.attitude * accel;
            let transport_rate = earth::vector_to_skew_symmetric(
                &earth::transport_rate(&particle.nav_state.latitude.to_degrees(), 
                &particle.nav_state.altitude, 
                &Vector3::from_vec(vec![
                    particle.nav_state.velocity_north,
                    particle.nav_state.velocity_east,
                    particle.nav_state.velocity_down,
                ])
            ));
            let rotation_rate = earth::vector_to_skew_symmetric(&earth::earth_rate_lla(
                &particle.nav_state.latitude.to_degrees()
            ));
            let r = earth::ecef_to_lla(
                &particle.nav_state.latitude.to_degrees(), 
                &particle.nav_state.longitude.to_degrees()
            );
            let mut velocity = Vector3::from_vec(vec![
                particle.nav_state.velocity_north,
                particle.nav_state.velocity_east,
                particle.nav_state.velocity_down,
            ]);
            let coriolis = r * (2.0 * rotation_rate + &transport_rate) * &velocity;
            let g = earth::gravity(&particle.nav_state.latitude.to_degrees(), &particle.nav_state.altitude);
            let g = if particle.nav_state.is_enu {
                -g
            } else {
                g
            };
            velocity[0] += (f[0] + coriolis[0]) * dt;
            velocity[1] += (f[1] + coriolis[1]) * dt;
            velocity[2] += (f[2] + g - particle.k2 * particle.altitude_error + coriolis[2]) * dt + self.vertical_accel_noise.sample(&mut self.rng) * dt.sqrt();
            // Position update
            let (r_n, r_e_0, _) = earth::principal_radii(&particle.nav_state.latitude, &particle.nav_state.altitude);
            let lat_0 = particle.nav_state.latitude;
            // Altitude update
            let alt_1 = particle.nav_state.altitude + (velocity[2] - particle.k1 * particle.altitude_error) * dt;
            // Latitude update
            let lat_1 = particle.nav_state.latitude + 0.5 * (velocity[0] / (r_n + particle.nav_state.altitude) + velocity[0] / (r_n + alt_1)) * dt;
            // Longitude update
            let (_, r_e_1, _) = earth::principal_radii(&lat_1, &alt_1);
            let cos_lat0 = lat_0.cos().max(1e-6); // Guard against cos(lat) --> 0 near poles
            let cos_lat1 = lat_1.cos().max(1e-6);
            let lon_1 = particle.nav_state.longitude 
                + 0.5 
                    * (particle.nav_state.velocity_east / ((r_e_0 + particle.nav_state.altitude) * cos_lat0) 
                        + velocity[1] / ((r_e_1 + alt_1) * cos_lat1)) * dt;
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
            let innovation = measurement.get_vector() - measurement.get_expected_measurement(&state);
            let sigmas = measurement.get_noise();
            let sigma_inv = match sigmas.clone().try_inverse() { 
                Some(inv) => inv, 
                None => { p.weight = 1e-300; continue; } 
            };
            let sigma_det = sigmas.determinant(); 
            if sigma_det <= 0.0 { 
                p.weight = 1e-300; 
                continue; 
            }
            let mahalanobis = innovation.transpose() * sigma_inv * innovation;
            let log_likelihood = -0.5 * (measurement.get_dimension() as f64 * (2.0 * std::f64::consts::PI).ln() + sigma_det.ln() + mahalanobis[(0, 0)]);
            p.weight = if self.resampling_mode {
                &p.weight * log_likelihood.exp()
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


// ...existing code...

impl ParticleFilter {
    // ...existing code...

    

    // Replace predict body with orchestrator that calls helpers
    

    // ...existing code...
}
// ...existing code...
