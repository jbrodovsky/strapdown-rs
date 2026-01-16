//! Rao-Blackwellized particle filter (RBPF) for inertial navigation.
//!
//! This filter represents position errors with particles and uses a shared
//! linear Kalman filter for velocity/attitude error states. It is intended
//! for map-matching and GNSS-aided navigation where measurements are highly
//! nonlinear in position but linear in the remaining states.

use crate::earth::METERS_TO_DEGREES;
use crate::linalg::{matrix_square_root, symmetrize};
use crate::linearize::state_transition_jacobian;
use crate::measurements::{
    GPSPositionAndVelocityMeasurement, GPSPositionMeasurement, GPSVelocityMeasurement,
    MeasurementModel, RelativeAltitudeMeasurement,
};
use crate::particle::{
    ParticleResamplingStrategy, multinomial_resample, residual_resample, stratified_resample,
    systematic_resample,
};
use crate::{IMUData, StrapdownState, forward};

use nalgebra::{DMatrix, DVector, Vector3};
use rand::prelude::*;
use rand_distr::Normal;

const POSITION_STATE_DIM: usize = 3;
const LINEAR_STATE_DIM_BASE: usize = 6;

/// RBPF configuration parameters.
#[derive(Clone, Debug)]
pub struct RbpfConfig {
    pub num_particles: usize,
    pub resampling_strategy: ParticleResamplingStrategy,
    pub effective_sample_threshold: f64,
    pub position_init_std_m: Vector3<f64>,
    pub velocity_init_std_mps: f64,
    pub attitude_init_std_rad: f64,
    pub position_process_noise_std_m: Vector3<f64>,
    pub velocity_process_noise_std_mps: f64,
    pub attitude_process_noise_std_rad: f64,
    /// Additional linear states appended after velocity/attitude (e.g., map bias states).
    pub extra_state_dim: usize,
    /// Initial standard deviation for extra states (applied uniformly).
    pub extra_state_init_std: f64,
    /// Process noise standard deviation for extra states (random walk, applied uniformly).
    pub extra_state_process_noise_std: f64,
    pub seed: u64,
    pub recenter_after_update: bool,
    /// Apply a pseudo-measurement that vertical velocity is zero.
    pub zero_vertical_velocity: bool,
    /// Standard deviation for the zero-vertical-velocity pseudo-measurement.
    pub zero_vertical_velocity_std_mps: f64,
}

impl Default for RbpfConfig {
    fn default() -> Self {
        Self {
            num_particles: 500,
            resampling_strategy: ParticleResamplingStrategy::Systematic,
            effective_sample_threshold: 0.5,
            position_init_std_m: Vector3::new(10.0, 10.0, 5.0),
            velocity_init_std_mps: 1.0,
            attitude_init_std_rad: 0.1,
            position_process_noise_std_m: Vector3::new(1.0, 1.0, 1.0),
            velocity_process_noise_std_mps: 1e-3,
            attitude_process_noise_std_rad: 0.01,
            extra_state_dim: 0,
            extra_state_init_std: 0.0,
            extra_state_process_noise_std: 0.0,
            seed: 42,
            recenter_after_update: true,
            zero_vertical_velocity: true,
            zero_vertical_velocity_std_mps: 0.1,
        }
    }
}

/// RBPF particle state (position error + linear state).
#[derive(Clone, Debug)]
pub struct RbpfParticle {
    pub position_error: Vector3<f64>,
    pub linear_state: DVector<f64>,
    pub linear_cov: DMatrix<f64>,
    pub weight: f64,
}

/// Rao-Blackwellized particle filter implementation.
pub struct RaoBlackwellizedParticleFilter {
    config: RbpfConfig,
    particles: Vec<RbpfParticle>,
    nominal: StrapdownState,
    rng: StdRng,
    linear_update_applied: bool,
}

impl RaoBlackwellizedParticleFilter {
    fn linear_state_dim(&self) -> usize {
        LINEAR_STATE_DIM_BASE + self.config.extra_state_dim
    }
    /// Create a new RBPF with particles initialized around the nominal state.
    pub fn new(nominal: StrapdownState, config: RbpfConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let linear_dim = LINEAR_STATE_DIM_BASE + config.extra_state_dim;

        let meters_to_rad = METERS_TO_DEGREES.to_radians();
        let pos_std = Vector3::new(
            config.position_init_std_m[0] * meters_to_rad,
            config.position_init_std_m[1] * meters_to_rad,
            config.position_init_std_m[2],
        );

        let normal_lat = Normal::new(0.0, pos_std[0]).unwrap();
        let normal_lon = Normal::new(0.0, pos_std[1]).unwrap();
        let normal_alt = Normal::new(0.0, pos_std[2]).unwrap();

        let mut linear_cov = DMatrix::<f64>::zeros(linear_dim, linear_dim);
        for i in 0..3 {
            linear_cov[(i, i)] = config.velocity_init_std_mps.powi(2);
            linear_cov[(i + 3, i + 3)] = config.attitude_init_std_rad.powi(2);
        }
        if config.extra_state_dim > 0 {
            let var = config.extra_state_init_std.powi(2);
            for i in 0..config.extra_state_dim {
                linear_cov[(LINEAR_STATE_DIM_BASE + i, LINEAR_STATE_DIM_BASE + i)] = var;
            }
        }

        let mut particles = Vec::with_capacity(config.num_particles);
        let weight = 1.0 / config.num_particles as f64;
        let extra_state_normal = if config.extra_state_dim > 0 && config.extra_state_init_std > 0.0
        {
            Some(Normal::new(0.0, config.extra_state_init_std).unwrap())
        } else {
            None
        };
        for _ in 0..config.num_particles {
            let position_error = Vector3::new(
                normal_lat.sample(&mut rng),
                normal_lon.sample(&mut rng),
                normal_alt.sample(&mut rng),
            );
            let mut linear_state = DVector::zeros(linear_dim);
            if let Some(extra_normal) = &extra_state_normal {
                for i in 0..config.extra_state_dim {
                    linear_state[LINEAR_STATE_DIM_BASE + i] = extra_normal.sample(&mut rng);
                }
            }
            particles.push(RbpfParticle {
                position_error,
                linear_state,
                linear_cov: linear_cov.clone(),
                weight,
            });
        }

        Self {
            config,
            particles,
            nominal,
            rng,
            linear_update_applied: false,
        }
    }

    /// Access the nominal INS state.
    pub fn nominal_state(&self) -> &StrapdownState {
        &self.nominal
    }

    /// Predict step using IMU data.
    pub fn predict(&mut self, imu: &IMUData, dt: f64) {
        let f = state_transition_jacobian(&self.nominal, &imu.accel, &imu.gyro, dt);
        let linear_dim = LINEAR_STATE_DIM_BASE + self.config.extra_state_dim;

        let f_nn = f
            .view((0, 0), (POSITION_STATE_DIM, POSITION_STATE_DIM))
            .into_owned();
        let f_nl = f
            .view((0, 3), (POSITION_STATE_DIM, LINEAR_STATE_DIM_BASE))
            .into_owned();
        let f_ln = f
            .view((3, 0), (LINEAR_STATE_DIM_BASE, POSITION_STATE_DIM))
            .into_owned();
        let f_ll = f
            .view((3, 3), (LINEAR_STATE_DIM_BASE, LINEAR_STATE_DIM_BASE))
            .into_owned();

        let mut f_nl_full = DMatrix::<f64>::zeros(POSITION_STATE_DIM, linear_dim);
        f_nl_full
            .view_mut((0, 0), (POSITION_STATE_DIM, LINEAR_STATE_DIM_BASE))
            .copy_from(&f_nl);
        let mut f_ln_full = DMatrix::<f64>::zeros(linear_dim, POSITION_STATE_DIM);
        f_ln_full
            .view_mut((0, 0), (LINEAR_STATE_DIM_BASE, POSITION_STATE_DIM))
            .copy_from(&f_ln);
        let mut f_ll_full = DMatrix::<f64>::identity(linear_dim, linear_dim);
        f_ll_full
            .view_mut((0, 0), (LINEAR_STATE_DIM_BASE, LINEAR_STATE_DIM_BASE))
            .copy_from(&f_ll);

        // Scale process noise with dt to approximate continuous-time random walk.
        let meters_to_rad = METERS_TO_DEGREES.to_radians();
        let pos_noise = Vector3::new(
            self.config.position_process_noise_std_m[0] * meters_to_rad * dt,
            self.config.position_process_noise_std_m[1] * meters_to_rad * dt,
            self.config.position_process_noise_std_m[2] * dt,
        );
        let mut q_n = DMatrix::<f64>::zeros(POSITION_STATE_DIM, POSITION_STATE_DIM);
        for i in 0..POSITION_STATE_DIM {
            q_n[(i, i)] = pos_noise[i].powi(2);
        }

        let mut q_l = DMatrix::<f64>::zeros(linear_dim, linear_dim);
        let vel_noise = self.config.velocity_process_noise_std_mps * dt;
        let att_noise = self.config.attitude_process_noise_std_rad * dt;
        for i in 0..3 {
            q_l[(i, i)] = vel_noise.powi(2);
            q_l[(i + 3, i + 3)] = att_noise.powi(2);
        }
        if self.config.extra_state_dim > 0 {
            let extra_noise = self.config.extra_state_process_noise_std * dt;
            for i in 0..self.config.extra_state_dim {
                q_l[(LINEAR_STATE_DIM_BASE + i, LINEAR_STATE_DIM_BASE + i)] =
                    extra_noise.powi(2);
            }
        }

        // Propagate nominal state with strapdown mechanization.
        forward(&mut self.nominal, *imu, dt);

        let normal = Normal::new(0.0, 1.0).unwrap();

        for particle in &mut self.particles {
            let x_n = particle.position_error;
            let x_l = particle.linear_state.clone();
            let x_n_vec = DVector::from_vec(vec![x_n[0], x_n[1], x_n[2]]);

            let n = &f_nl_full * &particle.linear_cov * f_nl_full.transpose() + &q_n;
            let n = symmetrize(&n);
            let n_inv = n
                .clone()
                .try_inverse()
                .unwrap_or_else(|| DMatrix::identity(POSITION_STATE_DIM, POSITION_STATE_DIM));
            let l = &f_ll_full * &particle.linear_cov * f_nl_full.transpose() * n_inv;
            let mut p_new = &f_ll_full * &particle.linear_cov * f_ll_full.transpose()
                + &q_l
                - &l * &n * l.transpose();
            p_new = symmetrize(&p_new);
            for i in 0..linear_dim {
                p_new[(i, i)] += 1e-9;
            }

            let q_sqrt = matrix_square_root(&n);
            let noise_vec = DVector::from_iterator(
                POSITION_STATE_DIM,
                (0..POSITION_STATE_DIM).map(|_| normal.sample(&mut self.rng)),
            );
            let q_noise = &q_sqrt * noise_vec;

            let x_n_pred_vec = &f_nn * &x_n_vec + &f_nl_full * &x_l + q_noise;
            let x_n_pred = Vector3::new(x_n_pred_vec[0], x_n_pred_vec[1], x_n_pred_vec[2]);
            let z = &x_n_pred_vec - &f_nn * &x_n_vec;
            let mut x_l_pred =
                &f_ll_full * &x_l + &f_ln_full * &x_n_vec + &l * (z - &f_nl_full * &x_l);
            if self.config.extra_state_dim > 0 && self.config.extra_state_process_noise_std > 0.0 {
                for i in 0..self.config.extra_state_dim {
                    let idx = LINEAR_STATE_DIM_BASE + i;
                    let noise = normal.sample(&mut self.rng)
                        * self.config.extra_state_process_noise_std
                        * dt;
                    x_l_pred[idx] += noise;
                }
            }

            particle.position_error = x_n_pred;
            particle.linear_state = x_l_pred;
            particle.linear_cov = p_new;
        }
    }

    /// Update step using a measurement model.
    pub fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        if let Some(pos_meas) = measurement
            .as_any()
            .downcast_ref::<GPSPositionMeasurement>()
        {
            self.update_position_only(pos_meas);
            return;
        }
        if let Some(vel_meas) = measurement
            .as_any()
            .downcast_ref::<GPSVelocityMeasurement>()
        {
            self.update_velocity_only(vel_meas);
            return;
        }
        if let Some(pos_vel) = measurement
            .as_any()
            .downcast_ref::<GPSPositionAndVelocityMeasurement>()
        {
            self.update_position_velocity(pos_vel);
            return;
        }
        if let Some(alt) = measurement
            .as_any()
            .downcast_ref::<RelativeAltitudeMeasurement>()
        {
            self.update_position_only(alt);
            return;
        }

        self.update_weights_generic(measurement);

        if self.config.zero_vertical_velocity {
            self.update_vertical_velocity_constraint();
        }
    }

    /// Return weighted mean and covariance of the full 9-state estimate.
    pub fn estimate(&self) -> (DVector<f64>, DMatrix<f64>) {
        let mut mean = DVector::<f64>::zeros(9);
        for particle in &self.particles {
            let state = self.particle_state_vector(particle);
            mean += state * particle.weight;
        }

        let mut cov = DMatrix::<f64>::zeros(9, 9);
        for particle in &self.particles {
            let state = self.particle_state_vector(particle);
            let diff = &state - &mean;
            cov += particle.weight * (&diff * diff.transpose());
        }
        cov = symmetrize(&cov);
        (mean, cov)
    }

    /// Compute the effective sample size.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.particles.iter().map(|p| p.weight.powi(2)).sum();
        if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 }
    }

    fn update_position_only<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        self.update_weights_generic(measurement);
        if self.config.zero_vertical_velocity {
            self.update_vertical_velocity_constraint();
        }
    }

    fn update_velocity_only(&mut self, measurement: &GPSVelocityMeasurement) {
        let measurement_vec = measurement.get_measurement(&DVector::zeros(9));
        let v_nominal = Vector3::new(
            self.nominal.velocity_north,
            self.nominal.velocity_east,
            self.nominal.velocity_vertical,
        );
        let residual = DVector::from_vec(vec![
            measurement_vec[0] - v_nominal[0],
            measurement_vec[1] - v_nominal[1],
            measurement_vec[2] - v_nominal[2],
        ]);

        let mut h = DMatrix::<f64>::zeros(3, self.linear_state_dim());
        for i in 0..3 {
            h[(i, i)] = 1.0;
        }
        self.update_linear_state(&residual, &h, &measurement.get_noise());

        if self.config.zero_vertical_velocity {
            self.update_vertical_velocity_constraint();
        }
    }

    fn update_position_velocity(&mut self, measurement: &GPSPositionAndVelocityMeasurement) {
        self.update_weights_gps_position(measurement);

        let measurement_vec = measurement.get_measurement(&DVector::zeros(9));
        let v_nominal = Vector3::new(self.nominal.velocity_north, self.nominal.velocity_east, 0.0);
        let residual = DVector::from_vec(vec![
            measurement_vec[3] - v_nominal[0],
            measurement_vec[4] - v_nominal[1],
        ]);

        let mut h = DMatrix::<f64>::zeros(2, self.linear_state_dim());
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;

        let mut r = DMatrix::<f64>::zeros(2, 2);
        let noise = measurement.velocity_noise_std.powi(2);
        r[(0, 0)] = noise;
        r[(1, 1)] = noise;

        self.update_linear_state(&residual, &h, &r);

        if self.config.zero_vertical_velocity {
            self.update_vertical_velocity_constraint();
        }
    }

    fn update_weights_gps_position(&mut self, measurement: &GPSPositionAndVelocityMeasurement) {
        let z_full = measurement.get_measurement(&DVector::zeros(9));
        let z = DVector::from_vec(vec![z_full[0], z_full[1], z_full[2]]);

        let r_full = measurement.get_noise();
        let mut r = DMatrix::<f64>::zeros(3, 3);
        for i in 0..3 {
            r[(i, i)] = r_full[(i, i)];
        }

        let mut log_weights = Vec::with_capacity(self.particles.len());
        let mut max_log = f64::NEG_INFINITY;

        for particle in &self.particles {
            let state = self.particle_state_vector(particle);
            let z_hat = DVector::from_vec(vec![state[0], state[1], state[2]]);
            let residual = &z - z_hat;
            let log_likelihood = gaussian_log_likelihood(&residual, &r);
            let log_w = particle.weight.ln() + log_likelihood;
            log_weights.push(log_w);
            if log_w > max_log {
                max_log = log_w;
            }
        }

        let mut sum = 0.0;
        for (particle, log_w) in self.particles.iter_mut().zip(log_weights.iter()) {
            let w = (log_w - max_log).exp();
            particle.weight = w;
            sum += w;
        }

        if sum > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= sum;
            }
        } else {
            let uniform = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.weight = uniform;
            }
        }

        if self.config.recenter_after_update {
            self.recenter_errors();
        }

        self.maybe_resample();
    }

    fn update_weights_generic<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        let mut log_weights = Vec::with_capacity(self.particles.len());
        let mut max_log = f64::NEG_INFINITY;

        for particle in &self.particles {
            let state = self.particle_state_vector_full(particle);
            let z = measurement.get_measurement(&state);
            let z_hat = measurement.get_expected_measurement(&state);
            let residual = z - z_hat;

            let log_likelihood = gaussian_log_likelihood(&residual, &measurement.get_noise());
            let log_w = particle.weight.ln() + log_likelihood;
            log_weights.push(log_w);
            if log_w > max_log {
                max_log = log_w;
            }
        }

        let mut sum = 0.0;
        for (particle, log_w) in self.particles.iter_mut().zip(log_weights.iter()) {
            let w = (log_w - max_log).exp();
            particle.weight = w;
            sum += w;
        }

        if sum > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= sum;
            }
        } else {
            let uniform = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.weight = uniform;
            }
        }

        if self.config.recenter_after_update {
            self.recenter_errors();
        }

        self.maybe_resample();
    }

    fn update_linear_state(&mut self, residual: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
        let linear_dim = self.linear_state_dim();
        let eye = DMatrix::<f64>::identity(linear_dim, linear_dim);

        for particle in &mut self.particles {
            let s = h * &particle.linear_cov * h.transpose() + r;
            let s_inv = s
                .clone()
                .try_inverse()
                .unwrap_or_else(|| DMatrix::identity(s.nrows(), s.ncols()));
            let k = &particle.linear_cov * h.transpose() * s_inv;

            let innovation = residual - h * &particle.linear_state;
            particle.linear_state = &particle.linear_state + &k * innovation;
            particle.linear_cov = (eye.clone() - &k * h) * &particle.linear_cov;
            particle.linear_cov = symmetrize(&particle.linear_cov);
        }
        self.linear_update_applied = true;
    }

    fn update_vertical_velocity_constraint(&mut self) {
        let residual = DVector::from_vec(vec![0.0 - self.nominal.velocity_vertical]);
        let mut h = DMatrix::<f64>::zeros(1, self.linear_state_dim());
        h[(0, 2)] = 1.0;
        let r =
            DMatrix::<f64>::from_element(1, 1, self.config.zero_vertical_velocity_std_mps.powi(2));
        self.update_linear_state(&residual, &h, &r);
    }

    fn particle_state_vector(&self, particle: &RbpfParticle) -> DVector<f64> {
        let (roll, pitch, yaw) = self.nominal.attitude.euler_angles();
        DVector::from_vec(vec![
            self.nominal.latitude + particle.position_error[0],
            self.nominal.longitude + particle.position_error[1],
            self.nominal.altitude + particle.position_error[2],
            self.nominal.velocity_north + particle.linear_state[0],
            self.nominal.velocity_east + particle.linear_state[1],
            self.nominal.velocity_vertical + particle.linear_state[2],
            roll + particle.linear_state[3],
            pitch + particle.linear_state[4],
            yaw + particle.linear_state[5],
        ])
    }

    fn particle_state_vector_full(&self, particle: &RbpfParticle) -> DVector<f64> {
        let mut state = self.particle_state_vector(particle).as_slice().to_vec();
        if self.config.extra_state_dim > 0 {
            state.extend_from_slice(
                particle
                    .linear_state
                    .rows(LINEAR_STATE_DIM_BASE, self.config.extra_state_dim)
                    .as_slice(),
            );
        }
        DVector::from_vec(state)
    }

    fn recenter_errors(&mut self) {
        let mut mean_pos = Vector3::zeros();
        let mut mean_lin = DVector::<f64>::zeros(self.linear_state_dim());
        for particle in &self.particles {
            mean_pos += particle.position_error * particle.weight;
            mean_lin += &particle.linear_state * particle.weight;
        }

        let apply_linear = self.linear_update_applied;
        let mean_lin_base = mean_lin.rows(0, LINEAR_STATE_DIM_BASE).into_owned();
        let delta_x = if apply_linear {
            DVector::from_vec(vec![
                mean_pos[0],
                mean_pos[1],
                mean_pos[2],
                mean_lin_base[0],
                mean_lin_base[1],
                mean_lin_base[2],
                mean_lin_base[3],
                mean_lin_base[4],
                mean_lin_base[5],
            ])
        } else {
            DVector::from_vec(vec![
                mean_pos[0],
                mean_pos[1],
                mean_pos[2],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ])
        };
        crate::linearize::apply_eskf_correction(&mut self.nominal, &delta_x);

        for particle in &mut self.particles {
            particle.position_error -= mean_pos;
            for i in 0..LINEAR_STATE_DIM_BASE {
                particle.linear_state[i] -= mean_lin_base[i];
            }
        }

        self.linear_update_applied = false;
    }

    fn maybe_resample(&mut self) {
        let n_eff = self.effective_sample_size();
        let threshold = self.config.effective_sample_threshold * self.particles.len() as f64;
        if n_eff >= threshold {
            return;
        }

        let weights: Vec<f64> = self.particles.iter().map(|p| p.weight).collect();
        let indices = match self.config.resampling_strategy {
            ParticleResamplingStrategy::Multinomial => {
                multinomial_resample(&weights, self.particles.len(), &mut self.rng)
            }
            ParticleResamplingStrategy::Systematic => {
                systematic_resample(&weights, self.particles.len(), &mut self.rng)
            }
            ParticleResamplingStrategy::Stratified => {
                stratified_resample(&weights, self.particles.len(), &mut self.rng)
            }
            ParticleResamplingStrategy::Residual => {
                residual_resample(&weights, self.particles.len(), &mut self.rng)
            }
        };

        let mut new_particles = Vec::with_capacity(self.particles.len());
        for idx in indices {
            let mut particle = self.particles[idx].clone();
            particle.weight = 1.0 / self.particles.len() as f64;
            new_particles.push(particle);
        }
        self.particles = new_particles;
    }
}

fn gaussian_log_likelihood(residual: &DVector<f64>, noise: &DMatrix<f64>) -> f64 {
    if residual.iter().any(|v| !v.is_finite()) {
        return f64::NEG_INFINITY;
    }
    let noise_inv = noise
        .clone()
        .try_inverse()
        .unwrap_or_else(|| DMatrix::identity(noise.nrows(), noise.ncols()));
    let quad = residual.transpose() * noise_inv * residual;
    -0.5 * quad[(0, 0)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{earth, generate_scenario_data};
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Rotation3;

    fn run_rbpf_on_scenario(
        nominal: StrapdownState,
        imu_data: &[IMUData],
        gps_measurements: &[GPSPositionMeasurement],
        sample_rate_hz: usize,
    ) -> (DVector<f64>, DMatrix<f64>) {
        assert_eq!(imu_data.len(), gps_measurements.len());
        let dt = 1.0 / sample_rate_hz as f64;

        let config = RbpfConfig {
            num_particles: 10000,
            position_init_std_m: Vector3::new(10.0, 10.0, 5.0),
            seed: 123,
            zero_vertical_velocity: true,
            zero_vertical_velocity_std_mps: 0.05,
            ..RbpfConfig::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config);

        for (imu, gps) in imu_data.iter().zip(gps_measurements.iter()) {
            rbpf.predict(imu, dt);
            rbpf.update(gps);
        }

        let weight_sum: f64 = rbpf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);

        rbpf.estimate()
    }

    fn assert_solution_close_to_truth(
        estimate: &DVector<f64>,
        truth: &StrapdownState,
        max_horizontal_error_m: f64,
        max_alt_error_m: f64,
        max_vel_error_mps: f64,
    ) {
        assert!(estimate.iter().all(|v| v.is_finite()));

        let horizontal_error_m =
            earth::haversine_distance(estimate[0], estimate[1], truth.latitude, truth.longitude);
        let alt_error_m = (estimate[2] - truth.altitude).abs();
        let vel_error_mps = ((estimate[3] - truth.velocity_north).powi(2)
            + (estimate[4] - truth.velocity_east).powi(2)
            + (estimate[5] - truth.velocity_vertical).powi(2))
        .sqrt();

        assert!(
            horizontal_error_m <= max_horizontal_error_m,
            "Horizontal error too large: {horizontal_error_m:.3} m (max {max_horizontal_error_m:.3} m)"
        );
        assert!(
            alt_error_m <= max_alt_error_m,
            "Altitude error too large: {alt_error_m:.3} m (max {max_alt_error_m:.3} m)"
        );
        assert!(
            vel_error_mps <= max_vel_error_mps,
            "Velocity error too large: {vel_error_mps:.3} m/s (max {max_vel_error_mps:.3} m/s)"
        );
    }

    #[test]
    fn rbpf_updates_and_normalizes_weights() {
        let nominal = StrapdownState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let config = RbpfConfig {
            num_particles: 100,
            position_init_std_m: Vector3::new(5.0, 5.0, 2.0),
            seed: 7,
            ..RbpfConfig::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config);

        let meas = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        rbpf.update(&meas);

        let weight_sum: f64 = rbpf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);

        let (mean, _) = rbpf.estimate();
        assert!(mean[0].is_finite());
        assert!(mean[1].is_finite());
        assert!(mean[2].is_finite());
    }

    #[test]
    fn rbpf_runs_on_scenario_stationary() {
        let lat_deg: f64 = 40.0;
        let lon_deg: f64 = -105.0;
        let alt_m: f64 = 1000.0;
        let g = earth::gravity(&lat_deg, &alt_m);

        let initial_state = StrapdownState {
            latitude: lat_deg.to_radians(),
            longitude: lon_deg.to_radians(),
            altitude: alt_m,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let duration_seconds = 3600;
        let sample_rate_hz = 5;

        let accel_body = Vector3::new(0.0, 0.0, g);
        let gyro_body = Vector3::new(0.0, 0.0, 0.0);

        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            duration_seconds,
            sample_rate_hz,
            accel_body,
            gyro_body,
            true,
            true,
            false,
        );

        let mut nominal = initial_state;
        let delta_deg = 20.0 * earth::METERS_TO_DEGREES;
        nominal.latitude += delta_deg.to_radians();
        nominal.longitude -= delta_deg.to_radians();

        let (mean, _cov) =
            run_rbpf_on_scenario(nominal, &imu_data, &gps_measurements, sample_rate_hz);
        let truth = true_states.last().unwrap();

        assert_solution_close_to_truth(&mean, truth, 25.0, 5.0, 0.5);
    }

    #[test]
    fn rbpf_runs_on_scenario_constant_velocity_north() {
        let lat_deg: f64 = 40.0;
        let lon_deg: f64 = -105.0;
        let alt_m: f64 = 1000.0;
        let g = earth::gravity(&lat_deg, &alt_m);

        let v_north = 10.0;
        let initial_state = StrapdownState {
            latitude: lat_deg.to_radians(),
            longitude: lon_deg.to_radians(),
            altitude: alt_m,
            velocity_north: v_north,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let duration_seconds = 3600;
        let sample_rate_hz = 5;

        let accel_body = Vector3::new(0.0, 0.0, g);
        let gyro_body = Vector3::new(0.0, 0.0, 0.0);

        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            duration_seconds,
            sample_rate_hz,
            accel_body,
            gyro_body,
            true,
            true,
            false,
        );

        let mut nominal = initial_state;
        let delta_deg = 25.0 * earth::METERS_TO_DEGREES;
        nominal.latitude -= delta_deg.to_radians();
        nominal.longitude += delta_deg.to_radians();

        let (mean, _cov) =
            run_rbpf_on_scenario(nominal, &imu_data, &gps_measurements, sample_rate_hz);
        let truth = true_states.last().unwrap();

        // Expect northward motion; RBPF estimate should reflect it.
        assert!(mean[0] > initial_state.latitude);
        assert_solution_close_to_truth(&mean, truth, 50.0, 25.0, 1.0);
    }

    #[test]
    fn rbpf_runs_on_scenario_constant_velocity_east() {
        let lat_deg: f64 = 40.0;
        let lon_deg: f64 = -105.0;
        let alt_m: f64 = 1000.0;
        let g = earth::gravity(&lat_deg, &alt_m);

        let v_east = 10.0;
        let initial_state = StrapdownState {
            latitude: lat_deg.to_radians(),
            longitude: lon_deg.to_radians(),
            altitude: alt_m,
            velocity_north: 0.0,
            velocity_east: v_east,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let duration_seconds = 3600;
        let sample_rate_hz = 5;

        let accel_body = Vector3::new(0.0, 0.0, g);
        let gyro_body = Vector3::new(0.0, 0.0, 0.0);

        let (imu_data, gps_measurements, true_states) = generate_scenario_data(
            initial_state,
            duration_seconds,
            sample_rate_hz,
            accel_body,
            gyro_body,
            true,
            true,
            false,
        );

        let mut nominal = initial_state;
        let delta_deg = 25.0 * earth::METERS_TO_DEGREES;
        nominal.latitude += delta_deg.to_radians();
        nominal.longitude -= delta_deg.to_radians();

        let (mean, _cov) =
            run_rbpf_on_scenario(nominal, &imu_data, &gps_measurements, sample_rate_hz);
        let truth = true_states.last().unwrap();

        // Expect eastward motion; RBPF estimate should reflect it.
        assert!(mean[1] > initial_state.longitude);
        assert_solution_close_to_truth(&mean, truth, 50.0, 25.0, 1.0);
    }
}
