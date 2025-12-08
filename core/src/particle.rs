//! Particle-filter style navigation code extracted from the former `filter.rs`.
use crate::kalman::NavigationFilter;
use crate::measurements::MeasurementModel;
use crate::{IMUData, StrapdownState, forward};

use nalgebra::{DMatrix, DVector, Rotation3};
use rand;
use rand_distr::Distribution;
use std::fmt::{self, Debug, Display};

#[derive(Clone, Debug, Default)]
pub struct Particle {
    pub nav_state: StrapdownState,
    pub other_states: Option<DVector<f64>>,
    pub state_size: usize,
    pub weight: f64,
}
impl Display for Particle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Particle")
            .field("latitude", &self.nav_state.latitude.to_degrees())
            .field("longitude", &self.nav_state.longitude.to_degrees())
            .field("altitude", &self.nav_state.altitude)
            .field("velocity_north", &self.nav_state.velocity_north)
            .field("velocity_east", &self.nav_state.velocity_east)
            .field("velocity_down", &self.nav_state.velocity_down)
            .field("roll", &self.nav_state.attitude.euler_angles().0)
            .field("pitch", &self.nav_state.attitude.euler_angles().1)
            .field("yaw", &self.nav_state.attitude.euler_angles().2)
            .field("weight", &self.weight)
            .finish()
    }
}
impl Particle {
    pub fn new(
        nav_state: StrapdownState,
        other_states: Option<DVector<f64>>,
        weight: f64,
    ) -> Particle {
        let state_size = 9 + match &other_states {
            Some(states) => states.len(),
            None => 0,
        };
        Particle {
            nav_state,
            other_states,
            state_size,
            weight,
        }
    }
}
impl From<(DVector<f64>, f64)> for Particle {
    fn from(tuple: (DVector<f64>, f64)) -> Self {
        let (state_vector, weight) = tuple;
        assert!(
            state_vector.len() >= 9,
            "State vector must be at least 9 elements long"
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
        let other_states = if state_vector.len() > 9 {
            Some(state_vector.rows(9, state_vector.len() - 9).clone_owned())
        } else {
            None
        };
        Particle::new(nav_state, other_states, weight)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum ParticleAveragingStrategy {
    WeightedAverage,
    UnweightedAverage,
    HighestWeight,
}
impl Default for ParticleAveragingStrategy {
    fn default() -> Self {
        ParticleAveragingStrategy::WeightedAverage
    }
}

#[derive(Clone, Debug)]
pub enum ParticleResamplingStrategy {
    Naive,
    Systematic,
    Multinomial,
    Residual,
    Stratified,
    Adaptive,
}
impl Default for ParticleResamplingStrategy {
    fn default() -> Self {
        ParticleResamplingStrategy::Residual
    }
}

impl ParticleResamplingStrategy {
    pub fn resample(&self, particles: Vec<Particle>) -> Vec<Particle> {
        match self {
            ParticleResamplingStrategy::Residual => Self::residual_resample(particles),
            _ => particles,
        }
    }
    fn residual_resample(particles: Vec<Particle>) -> Vec<Particle> {
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
                let mut new_particle = particles[i].clone();
                new_particle.weight = 1.0 / n as f64;
                new_particles.push(new_particle);
                j += 1;
            }
        }
        new_particles
    }
}

#[derive(Clone, Default)]
pub struct ParticleFilter {
    pub particles: Vec<Particle>,
    pub process_noise: DVector<f64>,
    pub averaging_strategy: ParticleAveragingStrategy,
    pub resampling_strategy: ParticleResamplingStrategy,
    pub state_size: usize,
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
    ) -> Self {
        let state_size = particles[0].state_size;
        let process_noise: DVector<f64> = match process_noise_std {
            Some(pn) => pn,
            None => {
                let mut noise = vec![1e-12, 1e-12, 1e-9, 1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12];
                if state_size > 9 {
                    noise.push(1e-7);
                    noise.push(1e-7);
                    noise.push(1e-7);
                    if state_size > 12 {
                        noise.push(1e-9);
                        noise.push(1e-9);
                        noise.push(1e-9);
                    }
                }
                while noise.len() < state_size {
                    noise.push(0.0);
                }
                DVector::from_vec(noise)
            }
        };
        ParticleFilter {
            particles,
            process_noise,
            averaging_strategy: estimation_strategy
                .unwrap_or(ParticleAveragingStrategy::WeightedAverage),
            resampling_strategy: resampling_method.unwrap_or(ParticleResamplingStrategy::Residual),
            state_size,
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
        self.particles = self.resampling_strategy.resample(self.particles.clone());
    }
    pub fn effective_sample_size(&self) -> f64 {
        let sum_of_squares: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_of_squares > 0.0 {
            1.0 / sum_of_squares
        } else {
            0.0
        }
    }
}
impl crate::kalman::NavigationFilter for ParticleFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        let mut rng = rand::rng();
        for particle in &mut self.particles {
            let accel_biases = if particle.other_states.is_some()
                && particle.other_states.as_ref().unwrap().len() >= 6
            {
                DVector::from_vec(vec![
                    particle.other_states.as_ref().unwrap()[0],
                    particle.other_states.as_ref().unwrap()[1],
                    particle.other_states.as_ref().unwrap()[2],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let gyro_biases = if particle.other_states.is_some()
                && particle.other_states.as_ref().unwrap().len() >= 6
            {
                DVector::from_vec(vec![
                    particle.other_states.as_ref().unwrap()[3],
                    particle.other_states.as_ref().unwrap()[4],
                    particle.other_states.as_ref().unwrap()[5],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let imu_data = IMUData {
                accel: imu_data.accel - accel_biases,
                gyro: imu_data.gyro - gyro_biases,
            };
            forward(&mut particle.nav_state, imu_data, dt);
            let dt_sqrt = dt.sqrt();
            particle.nav_state.latitude += rand_distr::Normal::new(0.0, self.process_noise[0])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.longitude += rand_distr::Normal::new(0.0, self.process_noise[1])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.altitude += rand_distr::Normal::new(0.0, self.process_noise[2])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.velocity_north +=
                rand_distr::Normal::new(0.0, self.process_noise[3])
                    .unwrap()
                    .sample(&mut rng)
                    * dt_sqrt;
            particle.nav_state.velocity_east += rand_distr::Normal::new(0.0, self.process_noise[4])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            particle.nav_state.velocity_down += rand_distr::Normal::new(0.0, self.process_noise[5])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let roll_noise = rand_distr::Normal::new(0.0, self.process_noise[6])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let pitch_noise = rand_distr::Normal::new(0.0, self.process_noise[7])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let yaw_noise = rand_distr::Normal::new(0.0, self.process_noise[8])
                .unwrap()
                .sample(&mut rng)
                * dt_sqrt;
            let (roll, pitch, yaw) = particle.nav_state.attitude.euler_angles();
            particle.nav_state.attitude = Rotation3::from_euler_angles(
                roll + roll_noise,
                pitch + pitch_noise,
                yaw + yaw_noise,
            );
            if let Some(ref mut other_states) = particle.other_states {
                for i in 0..other_states.len().min(6) {
                    let noise = rand_distr::Normal::new(0.0, self.process_noise[9 + i])
                        .unwrap()
                        .sample(&mut rng)
                        * dt_sqrt;
                    other_states[i] += noise;
                }
            }
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

// Helper averaging implementations (moved here for ParticleFilter use)
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
