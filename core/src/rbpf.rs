//! Rao-Blackwellized particle filter (RBPF) for inertial navigation.
//!
//! This filter represents position errors with particles and uses a shared
//! linear Kalman filter for velocity/attitude error states. It is intended
//! for map-matching and GNSS-aided navigation where measurements are highly
//! nonlinear in position but linear in the remaining states.

use crate::StrapdownError;
use crate::earth::METERS_TO_DEGREES;
use crate::gating::{InnovationGate, UpdateOutcome, normalized_innovation_squared};
use crate::kalman::imu_sample_from_input;
use crate::linalg::{matrix_square_root, symmetrize};
use crate::linearize::state_transition_jacobian;
use crate::measurements::{
    GPSPositionAndVelocityMeasurement, GPSPositionMeasurement, GPSVelocityMeasurement,
    MagnetometerYawMeasurement, MeasurementModel, RelativeAltitudeMeasurement,
};
use crate::particle::{
    ParticleResamplingStrategy, multinomial_resample, residual_resample, stratified_resample,
    systematic_resample,
};
use crate::{ImuSample, InputModel, NavigationFilter, StrapdownState, mechanize};

use nalgebra::{DMatrix, DVector, Vector3};
use rand::prelude::*;
use rand_distr::Normal;

const POSITION_STATE_DIM: usize = 3;
const LINEAR_STATE_DIM_BASE: usize = 6;

/// Index of the yaw error within a particle's linear state.
///
/// The linear state is `[dv_n, dv_e, dv_d, droll, dpitch, dyaw, ..extra]`, so this is the
/// third attitude entry. Named because it is the row a heading measurement selects in
/// [`RaoBlackwellizedParticleFilter::update_yaw_only`], and a bare `5` there is
/// indistinguishable from the velocity indices above it.
const YAW_ERROR_STATE_INDEX: usize = 5;

/// Indices of the three attitude angles within the assembled 9-state vector.
///
/// These are the channels that live on the circle rather than the line, so they are the
/// ones [`RaoBlackwellizedParticleFilter::estimate`] averages with [`circular_mean`] and
/// differences with [`crate::wrap_to_pi`]. Every other channel is an ordinary linear
/// quantity.
const ATTITUDE_STATE_INDICES: [usize; 3] = [6, 7, 8];

/// Weighted mean of angles, computed on the circle.
///
/// The mean direction of the unit vectors at `angles`, weighted by `weights`:
/// `atan2(sum w sin(x), sum w cos(x))`. Unlike a linear mean this is invariant to where
/// each angle is wrapped, so a cloud straddling the +/-pi branch cut averages to the
/// direction between its members rather than to the far side of the circle, and the
/// result is always on `[-pi, pi]`.
///
/// It agrees with the linear mean to second order in the spread, so a tight cloud is
/// unaffected -- on `core/tests/test_data.csv` the two differ by at most 2e-4 deg. The
/// point of using it anyway is that nothing keeps the cloud tight: the spread is set by
/// [`RbpfConfig::attitude_init_std_rad`] and by how badly the filter is doing.
///
/// A cloud with no mean direction -- one spread evenly around the circle, so that the
/// resultant vector is zero -- returns 0 rather than failing, which is `atan2(0, 0)`.
/// That is a degenerate input for which no angle is more correct than another, and the
/// caller sees it in the attitude variance, which is at its maximum there.
fn circular_mean<'a>(angles: impl Iterator<Item = (&'a f64, f64)>) -> f64 {
    let (sin_sum, cos_sum) = angles.fold((0.0, 0.0), |(s, c), (angle, weight)| {
        (s + weight * angle.sin(), c + weight * angle.cos())
    });
    sin_sum.atan2(cos_sum)
}

/// RBPF configuration parameters.
#[derive(Clone, Debug)]
pub struct RbpfConfig {
    /// Number of particles in the cloud; fixed for the life of the filter.
    pub num_particles: usize,
    /// Strategy used to resample the cloud once the effective sample size drops below
    /// the trigger described by [`RbpfConfig::effective_sample_threshold`].
    pub resampling_strategy: ParticleResamplingStrategy,
    /// Resampling trigger as a fraction of `num_particles`: the cloud is resampled
    /// when the effective sample size drops below this fraction of the particle count.
    pub effective_sample_threshold: f64,
    /// Initial position-error standard deviation in metres, as (latitude, longitude,
    /// altitude). Both horizontal entries are scaled at construction by the same
    /// latitude conversion factor ([`crate::earth::METERS_TO_DEGREES`], taken in
    /// radians). That is correct for latitude but not for longitude, where a metre of
    /// easting subtends `1 / ((R_e + h) cos(latitude))` radians -- so the `cos(latitude)`
    /// is missing from the denominator and the radian sigma comes out *too small* by
    /// that factor. The east extent the cloud actually receives is therefore the
    /// requested value **multiplied** by cos(latitude): a requested 10 m spreads about
    /// 7.7 m at 40 degrees, 5.0 m at 60 degrees and 1.7 m at 80 degrees. The cloud is
    /// under-spread east-west, increasingly so towards the poles. Issue #331 tracks the
    /// fix, here and in `position_process_noise_std_m`. The altitude entry is used in
    /// metres directly.
    pub position_init_std_m: Vector3<f64>,
    /// Initial standard deviation of each of the three velocity error states, in m/s
    /// (applied uniformly to north, east and vertical).
    pub velocity_init_std_mps: f64,
    /// Initial standard deviation of each of the three attitude error states, in
    /// radians (applied uniformly to roll, pitch and yaw).
    pub attitude_init_std_rad: f64,
    /// Position proposal scale (m per second of sample period). The predict
    /// step scales it by the IMU sample interval (`pos_noise = std * dt`), so
    /// the effective per-step standard deviation is this value times `dt_s`.
    /// It must cover unmodelled position wander between fixes: under the
    /// reference degraded profile (`Degraded { sigma_pos_m: 3.0 }`, 5 s fixes)
    /// the default 1 m starves the particle cloud (see #267) -- raise it
    /// explicitly in that configuration. Kept at 1 m here because a wider
    /// default proposal measurably degrades clean stationary tracking.
    pub position_process_noise_std_m: Vector3<f64>,
    /// Velocity random-walk scale (m/s per second of sample period), applied
    /// uniformly to the three velocity error states. Like the position term, the
    /// predict step scales it by the IMU sample interval (`vel_noise = std * dt`).
    pub velocity_process_noise_std_mps: f64,
    /// Attitude random-walk scale (rad per second of sample period), applied
    /// uniformly to the three attitude error states and likewise scaled by the IMU
    /// sample interval in the predict step.
    pub attitude_process_noise_std_rad: f64,
    /// Additional linear states appended after velocity/attitude (e.g., map bias states).
    pub extra_state_dim: usize,
    /// Initial standard deviation for extra states (applied uniformly).
    pub extra_state_init_std: f64,
    /// Process noise standard deviation for extra states (random walk, applied uniformly).
    pub extra_state_process_noise_std: f64,
    /// Seed for the filter's random number generator, which draws the initial
    /// particle spread, the per-step process noise and the resampling indices. Runs
    /// with the same seed and the same inputs are reproducible.
    pub seed: u64,
    /// Recentre the error states on the nominal state after each weight update: the
    /// weighted-mean error is subtracted from every particle's position error and from
    /// its six base linear error states (velocity and attitude), so those states stay
    /// zero-mean. Any [`RbpfConfig::extra_state_dim`] states appended after them are
    /// left untouched, so an extra state such as a geophysical map bias keeps its
    /// absolute value across recentrings rather than being folded into the nominal
    /// state; issue #333 tracks that gap. The mean position error is always applied to
    /// the nominal state; the mean velocity/attitude error is applied only when a linear
    /// (Kalman) update has run since the previous recentring. Note that the subtraction
    /// from the particles is unconditional, so in the other case that mean is **discarded**
    /// rather than deferred -- it is removed from the cloud without ever reaching the
    /// nominal state.
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
    /// Position error relative to the nominal state, added to it to form this
    /// particle's position: latitude and longitude errors in radians, altitude error
    /// in metres (positive up, as with [`StrapdownState::altitude`]).
    pub position_error: Vector3<f64>,
    /// Linear error state carried by this particle's Kalman filter, added to the
    /// nominal state: three velocity errors in m/s (north, east, vertical -- vertical
    /// following the frame of the nominal state), three attitude errors in radians
    /// (roll, pitch, yaw), then [`RbpfConfig::extra_state_dim`] extra states.
    pub linear_state: DVector<f64>,
    /// Covariance of `linear_state`, square and in the same state ordering.
    pub linear_cov: DMatrix<f64>,
    /// Normalized importance weight; the weights of the cloud sum to one.
    pub weight: f64,
}

/// Rao-Blackwellized particle filter implementation.
#[derive(Debug)]
pub struct RaoBlackwellizedParticleFilter {
    config: RbpfConfig,
    particles: Vec<RbpfParticle>,
    nominal: StrapdownState,
    rng: StdRng,
    linear_update_applied: bool,
    /// Innovation gate applied by `update`; `None` accepts every measurement.
    innovation_gate: Option<InnovationGate>,
}

impl RaoBlackwellizedParticleFilter {
    const fn linear_state_dim(&self) -> usize {
        LINEAR_STATE_DIM_BASE + self.config.extra_state_dim
    }
    /// Create a new RBPF with particles initialized around the nominal state.
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] if any `position_init_std_m` component is
    /// not a usable standard deviation. Zero is a plausible thing for a user to write --
    /// "I know my start position exactly" -- and it used to panic here at construction.
    pub fn new(nominal: StrapdownState, config: RbpfConfig) -> Result<Self, StrapdownError> {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let linear_dim = LINEAR_STATE_DIM_BASE + config.extra_state_dim;

        let meters_to_rad = METERS_TO_DEGREES.to_radians();
        let pos_std = Vector3::new(
            config.position_init_std_m[0] * meters_to_rad,
            config.position_init_std_m[1] * meters_to_rad,
            config.position_init_std_m[2],
        );

        let position_normal = |axis: usize, name: &'static str| {
            Normal::new(0.0, pos_std[axis]).map_err(|e| StrapdownError::InvalidConfiguration {
                field: name,
                reason: format!("{} is not a usable standard deviation: {e}", pos_std[axis]),
            })
        };
        let normal_lat = position_normal(0, "position_init_std_m[0]")?;
        let normal_lon = position_normal(1, "position_init_std_m[1]")?;
        let normal_alt = position_normal(2, "position_init_std_m[2]")?;

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
            // Guarded by `extra_state_init_std > 0.0` on the line above.
            Some(crate::normal_with_std(config.extra_state_init_std))
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

        Ok(Self {
            config,
            particles,
            nominal,
            rng,
            linear_update_applied: false,
            innovation_gate: None,
        })
    }

    /// Access the nominal INS state.
    pub const fn nominal_state(&self) -> &StrapdownState {
        &self.nominal
    }

    /// Propagate the particle cloud through one inertial sample.
    ///
    /// The body of [`NavigationFilter::predict`]; kept as an inherent method taking a
    /// resolved [`ImuSample`] so the trait impl is only the input-resolution shim.
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `sample.dt` is not strictly positive (the rates the
    /// Jacobian needs are undefined then), and [`StrapdownError::NotSquare`] from the
    /// process-noise square root.
    fn predict_sample(&mut self, sample: &ImuSample) -> Result<(), StrapdownError> {
        // The state-transition Jacobian is derived in the rate domain, so it needs the
        // average rates over the interval rather than the increments themselves. The RBPF
        // carries no bias states -- its linear state is velocity, attitude and any extra
        // states -- so there is nothing to compensate the increments for first.
        let rates = sample.to_rates()?;
        let dt = sample.dt;
        let f = state_transition_jacobian(&self.nominal, &rates.accel, &rates.gyro, dt);
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
                q_l[(LINEAR_STATE_DIM_BASE + i, LINEAR_STATE_DIM_BASE + i)] = extra_noise.powi(2);
            }
        }

        // Propagate nominal state with strapdown mechanization.
        mechanize(&mut self.nominal, sample)?;

        let normal = crate::normal_with_std(1.0);

        // Conditional covariance recursion, computed once per step instead of
        // once per particle per step. It depends only on the shared
        // transition/noise matrices and the particle covariance, which stays
        // identical across particles (see
        // `rbpf_particle_covariances_stay_identical`, #268), so this is
        // bit-identical to the per-particle computation it replaces. Only the
        // per-particle noise draws and state propagation stay in the loop, in
        // the same order, keeping the RNG stream untouched.
        let Some(first) = self.particles.first() else {
            // No particles to propagate; nothing to do and nothing wrong.
            return Ok(());
        };
        let n = &f_nl_full * &first.linear_cov * f_nl_full.transpose() + &q_n;
        let n = symmetrize(&n);
        let n_inv = n
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(POSITION_STATE_DIM, POSITION_STATE_DIM));
        let l = &f_ll_full * &first.linear_cov * f_nl_full.transpose() * n_inv;
        let mut p_new =
            &f_ll_full * &first.linear_cov * f_ll_full.transpose() + &q_l - &l * &n * l.transpose();
        p_new = symmetrize(&p_new);
        for i in 0..linear_dim {
            p_new[(i, i)] += 1e-9;
        }
        let q_sqrt = matrix_square_root(&n)?;

        for particle in &mut self.particles {
            let x_n = particle.position_error;
            let x_l = particle.linear_state.clone();
            let x_n_vec = DVector::from_vec(vec![x_n[0], x_n[1], x_n[2]]);

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
            particle.linear_cov.clone_from(&p_new);
        }
        Ok(())
    }

    /// Apply a measurement to the particle cloud, the shared Kalman filter, or both.
    ///
    /// The body of [`NavigationFilter::update`]. Generic rather than taking `&dyn
    /// MeasurementModel` because the downcasts below are what select the specialised
    /// position/velocity/heading paths, and a monomorphised call site keeps them cheap.
    ///
    /// # Which branch a measurement belongs in
    ///
    /// This filter is Rao-Blackwellized: position error is carried by the particles,
    /// velocity and attitude error by a Kalman filter shared across them. A measurement
    /// has to be applied to whichever of the two actually carries the states it observes,
    /// and the answer is read off its Jacobian:
    ///
    /// * **Supported on position** (GNSS position, barometric altitude) -- reweight the
    ///   cloud. The particles differ in position, so their likelihoods differ, and the
    ///   weights are where the information lands.
    /// * **Supported on velocity or attitude** (GNSS velocity, magnetometer heading) --
    ///   run a Kalman update on the linear states. Reweighting cannot work here, and the
    ///   reason is structural rather than a matter of degree: every particle shares the
    ///   nominal attitude and carries a near-identical error state, so every particle
    ///   predicts a near-identical measurement and earns a near-identical weight. The
    ///   cloud has no spread along the axis the measurement constrains.
    /// * **Both** (GNSS position and velocity) -- do both, each on its own block.
    ///
    /// Sending a linear-state measurement to [`Self::update_weights_generic`] is
    /// therefore not an approximation but a silent no-op, and it is what #341 was: the
    /// magnetometer fell through to the generic path, 5,365 heading fixes moved the
    /// effective sample size from 500 to a median of 492.6, and the filter's yaw drifted
    /// unaided to 65.9 deg RMSE while the aid it was being handed was good to 17.1 deg.
    ///
    /// # Errors
    /// Propagates measurement failures — chiefly a geophysical model whose particle has
    /// drifted off the loaded map. Callers should consult
    /// [`StrapdownError::is_recoverable`] and skip the measurement rather than abort.
    fn update_with<M: MeasurementModel + ?Sized>(
        &mut self,
        measurement: &M,
    ) -> Result<(), StrapdownError> {
        if let Some(pos_meas) = measurement
            .as_any()
            .downcast_ref::<GPSPositionMeasurement>()
        {
            return self.update_position_only(pos_meas);
        }
        if let Some(vel_meas) = measurement
            .as_any()
            .downcast_ref::<GPSVelocityMeasurement>()
        {
            return self.update_velocity_only(vel_meas);
        }
        if let Some(pos_vel) = measurement
            .as_any()
            .downcast_ref::<GPSPositionAndVelocityMeasurement>()
        {
            return self.update_position_velocity(pos_vel);
        }
        if let Some(alt) = measurement
            .as_any()
            .downcast_ref::<RelativeAltitudeMeasurement>()
        {
            return self.update_position_only(alt);
        }
        if let Some(mag) = measurement
            .as_any()
            .downcast_ref::<MagnetometerYawMeasurement>()
        {
            return self.update_yaw_only(mag);
        }

        self.update_weights_generic(measurement)?;

        if self.config.zero_vertical_velocity {
            self.update_vertical_velocity_constraint();
        }
        Ok(())
    }

    /// Return weighted mean and covariance of the full 9-state estimate.
    ///
    /// # Attitude is averaged on the circle
    ///
    /// Position and velocity are ordinary linear quantities and take the plain weighted
    /// mean. The three attitude angles do not: they live on the circle, where a linear
    /// mean is not merely inaccurate but wrong in kind. Two particles at +179 deg and
    /// -179 deg are 2 deg apart and average to 180 deg, but their linear mean is 0 deg --
    /// the opposite heading -- and the unwrapped sum can leave `[-pi, pi]` entirely, which
    /// the reported solution must not do (#314): [`crate::sim::NavigationResult`] copies
    /// these three straight through without wrapping them. So the attitude channels take
    /// the mean *direction* of the cloud, which is invariant to wrapping and always lands
    /// on the principal branch, and the covariance differences the attitude residuals with
    /// [`crate::wrap_to_pi`] so a cloud near the cut reports its actual spread rather than
    /// a phantom variance of order `pi^2`.
    ///
    /// This is a correctness property of the estimator, not a tuning: on
    /// `core/tests/test_data.csv` the circular and linear means differ by at most 2e-4 deg,
    /// because the cloud there is tight (0.6 deg median spread, 5.2 deg at its widest).
    /// Nothing enforces that tightness in general -- it is set by
    /// [`RbpfConfig::attitude_init_std_rad`] and by how well the filter is tracking -- and
    /// the failure, when it comes, is silent.
    pub fn estimate(&self) -> (DVector<f64>, DMatrix<f64>) {
        let states: Vec<DVector<f64>> = self
            .particles
            .iter()
            .map(|particle| self.particle_state_vector(particle))
            .collect();

        let mut mean = DVector::<f64>::zeros(9);
        for (state, particle) in states.iter().zip(&self.particles) {
            mean += state * particle.weight;
        }
        for index in ATTITUDE_STATE_INDICES {
            mean[index] = circular_mean(
                states
                    .iter()
                    .zip(&self.particles)
                    .map(|(state, particle)| (&state[index], particle.weight)),
            );
        }

        let mut cov = DMatrix::<f64>::zeros(9, 9);
        for (state, particle) in states.iter().zip(&self.particles) {
            let mut diff = state - &mean;
            for index in ATTITUDE_STATE_INDICES {
                diff[index] = crate::wrap_to_pi(diff[index]);
            }
            cov += particle.weight * (&diff * diff.transpose());
        }
        cov = symmetrize(&cov);
        (mean, cov)
    }

    /// Score a measurement against the particle cloud summarised as a Gaussian.
    ///
    /// Always computes the NIS, gate or no gate, because
    /// [`sim::health::HealthMonitor`](crate::sim::health::HealthMonitor) consumes it
    /// to catch a filter that has diverged rather than merely been handed one bad
    /// fix. The `estimate()` call this costs is the same one `run_closed_loop` makes
    /// immediately afterwards for logging.
    ///
    /// # Errors
    /// Whatever the measurement model returns when evaluated at the ensemble mean, or
    /// a singular innovation covariance from
    /// [`normalized_innovation_squared`](crate::gating::normalized_innovation_squared).
    fn evaluate_ensemble_gate<M: MeasurementModel + ?Sized>(
        &self,
        measurement: &M,
    ) -> Result<UpdateOutcome, StrapdownError> {
        let (mean, covariance) = self.estimate();
        // Jacobian first: a geophysical model off the edge of its map reports that
        // here, whereas `get_expected_measurement` would quietly return NaN.
        let h = measurement.get_jacobian(&mean)?;
        let z_hat = measurement.get_expected_measurement(&mean);
        let mut innovation = measurement.get_measurement(&mean)? - z_hat;
        measurement.wrap_residual(&mut innovation);

        let s = &h * &covariance * h.transpose() + measurement.get_noise();
        let dof = innovation.len();
        let nis = normalized_innovation_squared(&innovation, &s)?;
        if let Some(gate) = self.innovation_gate
            && !gate.accepts(nis, dof)
        {
            // `debug`, not `warn`: `run_closed_loop` already warns once with the
            // total, and a run that gates a lot would otherwise bury every other
            // message under one line per rejected fix.
            log::debug!(
                "RBPF: measurement gated out, NIS = {nis:.3} > {:.3} (dof {dof})",
                gate.threshold(dof)
            );
            return Ok(UpdateOutcome::rejected(nis, dof));
        }
        Ok(UpdateOutcome::accepted(nis, dof))
    }

    /// Compute the effective sample size.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.particles.iter().map(|p| p.weight.powi(2)).sum();
        if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 }
    }

    fn update_position_only<M: MeasurementModel + ?Sized>(
        &mut self,
        measurement: &M,
    ) -> Result<(), StrapdownError> {
        self.update_weights_generic(measurement)?;
        if self.config.zero_vertical_velocity {
            self.update_vertical_velocity_constraint();
        }
        Ok(())
    }

    fn update_velocity_only(
        &mut self,
        measurement: &GPSVelocityMeasurement,
    ) -> Result<(), StrapdownError> {
        let measurement_vec = measurement.get_measurement(&DVector::zeros(9))?;
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
        Ok(())
    }

    /// Apply a magnetometer heading to the shared attitude states.
    ///
    /// Yaw is a linear state in this filter, so the heading goes through the Kalman branch
    /// exactly as GNSS velocity does -- see [`Self::update_with`] for why reweighting the
    /// cloud on it does nothing. The Jacobian is
    /// [`crate::linearize::magnetometer_yaw_jacobian`] restricted to the linear block: the
    /// expected measurement is the state yaw, so the only non-zero partial is the one
    /// selected here.
    ///
    /// That row's gain on the yaw error is exact at any tilt, not merely to first order:
    /// the correction is injected as `Rz(dyaw) Ry(dpitch) Rx(droll)` times the nominal, and
    /// `Rz` is outermost, so a pure-yaw error composes with the nominal yaw exactly. What
    /// the row omits is the cross-coupling -- at a tilted nominal the extracted Euler yaw
    /// also depends on the roll and pitch error states, by up to 0.24 per radian on this
    /// dataset. That omission is shared with the EKF, ESKF and UKF, which use the same
    /// Jacobian, and with [`Self::evaluate_ensemble_gate`], which forms this filter's own
    /// innovation covariance from it; correcting it here alone would put the update and the
    /// gate in different coordinates. Tracked crate-wide as #349.
    ///
    /// The residual is formed at the nominal state rather than at a zero state the way
    /// [`Self::update_velocity_only`] can, because a magnetometer heading is not
    /// state-independent: [`MagnetometerYawMeasurement::get_measurement`] levels the sensor
    /// with the state's roll and pitch and takes its declination from the state's position.
    /// It is then wrapped onto the circle, so a nominal yaw and a heading on opposite sides
    /// of the branch cut give the small correction they represent rather than a full turn.
    ///
    /// # Errors
    /// Propagated from [`MagnetometerYawMeasurement::get_measurement`].
    fn update_yaw_only(
        &mut self,
        measurement: &MagnetometerYawMeasurement,
    ) -> Result<(), StrapdownError> {
        let nominal = self.nominal_state_vector();
        let mut residual =
            measurement.get_measurement(&nominal)? - measurement.get_expected_measurement(&nominal);
        measurement.wrap_residual(&mut residual);

        let mut h = DMatrix::<f64>::zeros(1, self.linear_state_dim());
        h[(0, YAW_ERROR_STATE_INDEX)] = 1.0;
        self.update_linear_state(&residual, &h, &measurement.get_noise());

        if self.config.zero_vertical_velocity {
            self.update_vertical_velocity_constraint();
        }
        Ok(())
    }

    fn update_position_velocity(
        &mut self,
        measurement: &GPSPositionAndVelocityMeasurement,
    ) -> Result<(), StrapdownError> {
        self.update_weights_gps_position(measurement)?;

        let measurement_vec = measurement.get_measurement(&DVector::zeros(9))?;
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
        Ok(())
    }

    fn update_weights_gps_position(
        &mut self,
        measurement: &GPSPositionAndVelocityMeasurement,
    ) -> Result<(), StrapdownError> {
        let z_full = measurement.get_measurement(&DVector::zeros(9))?;
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
            self.recenter_errors()?;
        }

        self.maybe_resample();
        Ok(())
    }

    fn update_weights_generic<M: MeasurementModel + ?Sized>(
        &mut self,
        measurement: &M,
    ) -> Result<(), StrapdownError> {
        let mut log_weights = Vec::with_capacity(self.particles.len());
        let mut max_log = f64::NEG_INFINITY;

        for particle in &self.particles {
            let state = self.particle_state_vector_full(particle);
            let z = measurement.get_measurement(&state)?;
            let z_hat = measurement.get_expected_measurement(&state);
            let mut residual = z - z_hat;
            // Keep angular residuals on the circle; an unwrapped ±2π mag
            // residual would flatten the likelihood of every particle (#267).
            measurement.wrap_residual(&mut residual);

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
            self.recenter_errors()?;
        }

        self.maybe_resample();
        Ok(())
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

    /// The nominal trajectory as a 9-state vector, with every error state at zero.
    ///
    /// The one place the 9-state layout is written down; [`Self::particle_state_vector`]
    /// adds a particle's errors to this rather than repeating it.
    fn nominal_state_vector(&self) -> DVector<f64> {
        let (roll, pitch, yaw) = self.nominal.attitude.euler_angles();
        DVector::from_vec(vec![
            self.nominal.latitude,
            self.nominal.longitude,
            self.nominal.altitude,
            self.nominal.velocity_north,
            self.nominal.velocity_east,
            self.nominal.velocity_vertical,
            roll,
            pitch,
            yaw,
        ])
    }

    /// One particle's 9-state navigation solution: the nominal trajectory plus its errors.
    ///
    /// The three attitude entries are wrapped onto `[-pi, pi]`. `euler_angles` already
    /// returns that branch, but adding an error state to it does not stay on it: a nominal
    /// yaw of 179.9 deg plus a 0.3 deg error is 180.2 deg, off the branch every consumer
    /// assumes (#314). It is wrapped here, where the state is assembled, rather than at
    /// each of the four call sites that read it. This mirrors
    /// [`kalman::wrap_attitude_onto_principal_branch`](crate::kalman), including its
    /// deliberate choice not to clamp pitch to the `[-pi/2, pi/2]` the Euler decomposition
    /// produces: clamping would change the rotation rather than rename it.
    ///
    /// Adding the error state to the Euler angles is not the same map as composing it with
    /// the nominal rotation, which is how [`crate::linearize::apply_eskf_correction`]
    /// injects it. The two agree exactly on yaw and to first order elsewhere; the
    /// discrepancy is the crate-wide chart inconsistency tracked as #349, and wrapping
    /// neither causes nor cures it.
    fn particle_state_vector(&self, particle: &RbpfParticle) -> DVector<f64> {
        let mut state = self.nominal_state_vector();
        for i in 0..POSITION_STATE_DIM {
            state[i] += particle.position_error[i];
        }
        for i in 0..LINEAR_STATE_DIM_BASE {
            state[POSITION_STATE_DIM + i] += particle.linear_state[i];
        }
        for index in ATTITUDE_STATE_INDICES {
            state[index] = crate::wrap_to_pi(state[index]);
        }
        state
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

    /// # Errors
    /// Propagated from [`crate::linearize::apply_eskf_correction`].
    fn recenter_errors(&mut self) -> Result<(), StrapdownError> {
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
        crate::linearize::apply_eskf_correction(&mut self.nominal, &delta_x)?;

        for particle in &mut self.particles {
            particle.position_error -= mean_pos;
            for i in 0..LINEAR_STATE_DIM_BASE {
                particle.linear_state[i] -= mean_lin_base[i];
            }
        }

        self.linear_update_applied = false;
        Ok(())
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

impl NavigationFilter for RaoBlackwellizedParticleFilter {
    /// Predict step: propagate the nominal trajectory and the particle cloud.
    ///
    /// # Arguments
    ///
    /// * `control_input` - an [`ImuSample`] (integrated $\Delta v$ / $\Delta\theta$) or,
    ///   for callers still holding instantaneous rates, an
    ///   [`IMUData`](crate::IMUData).
    /// * `dt` - Time step in seconds. When `control_input` is an [`ImuSample`] this must
    ///   agree with the sample's own `dt`; see the Errors section.
    ///
    /// # Errors
    /// * [`StrapdownError::UnsupportedInput`] if `control_input` is neither inertial form.
    /// * [`StrapdownError::InconsistentTimestep`] if an [`ImuSample`]'s `dt` disagrees with
    ///   the `dt` argument.
    /// * [`StrapdownError::OutOfRange`] or [`StrapdownError::NonFinite`] propagated from
    ///   [`mechanize`], or [`StrapdownError::NotSquare`] from the process-noise square root.
    fn predict(&mut self, control_input: &dyn InputModel, dt: f64) -> Result<(), StrapdownError> {
        let sample = imu_sample_from_input(control_input, "RaoBlackwellizedParticleFilter", dt)?;
        self.predict_sample(&sample)
    }

    /// Update step: reweight the particle cloud against a measurement.
    ///
    /// # Innovation gating
    ///
    /// The NIS is evaluated against the *ensemble* mean and covariance, not against
    /// any single particle: `estimate()` already summarises the cloud as a Gaussian,
    /// and that summary is what the $\chi^2$ test assumes. This is an approximation
    /// the Kalman filters do not need to make -- a multi-modal cloud has no
    /// meaningful single innovation -- so treat a gated RBPF as a coarse outlier
    /// screen rather than the consistency test it is for the EKF/UKF/ESKF. Its real
    /// defence against a bad fix is that a fix no particle agrees with simply
    /// contributes a flat likelihood.
    ///
    /// # Errors
    /// Propagates measurement failures — chiefly a geophysical model whose particle has
    /// drifted off the loaded map. Callers should consult
    /// [`StrapdownError::is_recoverable`] and skip the measurement rather than abort.
    fn update(
        &mut self,
        measurement: &dyn MeasurementModel,
    ) -> Result<UpdateOutcome, StrapdownError> {
        let outcome = self.evaluate_ensemble_gate(measurement)?;
        if !outcome.accepted {
            return Ok(outcome);
        }
        self.update_with(measurement)?;
        Ok(outcome)
    }

    fn set_innovation_gate(&mut self, gate: Option<InnovationGate>) -> bool {
        self.innovation_gate = gate;
        true
    }

    /// The weighted mean of the 9-state navigation solution.
    ///
    /// Computed from the particle cloud on each call; [`Self::estimate`] returns the mean
    /// and covariance together and is cheaper when both are wanted.
    fn get_estimate(&self) -> DVector<f64> {
        self.estimate().0
    }

    /// The weighted covariance of the 9-state navigation solution.
    ///
    /// See [`Self::get_estimate`] on computing both in one pass.
    fn get_certainty(&self) -> DMatrix<f64> {
        self.estimate().1
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
    use crate::{IMUData, earth, generate_scenario_data};
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Rotation3;

    /// A heading measurement must actually reach the attitude states (#341).
    ///
    /// This is the structural regression guard, not an accuracy one. Yaw is a linear state
    /// here, so a magnetometer routed to the particle weights is a silent no-op: every
    /// particle shares the nominal attitude, so every particle scores the same likelihood
    /// and the estimate does not move. The filter then runs on dead-reckoned heading while
    /// appearing to be aided, which is what put its yaw RMSE at 65.9 deg on
    /// `core/tests/test_data.csv` against a magnetometer good to 17.1 deg.
    ///
    /// The assertion is that the estimate closes most of a known heading offset, which is
    /// what "the information arrived" means and what reweighting cannot produce. The
    /// residual-fraction bound is a loose one on purpose -- the point is no-op versus
    /// working, not a gain setting.
    #[test]
    fn rbpf_magnetometer_update_moves_yaw_toward_the_measurement() {
        // Level and pointing north, with the magnetometer seeing a field rotated 30 deg
        // away: in NED, yaw = atan2(-m_y, m_x), so this is a heading of -30 deg.
        let truth_yaw = -std::f64::consts::FRAC_PI_6;
        let nominal = StrapdownState {
            latitude: 0.7,
            longitude: -1.3,
            altitude: 100.0,
            attitude: Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            is_enu: false,
            ..StrapdownState::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            nominal,
            RbpfConfig {
                num_particles: 200,
                // Off, so the only thing moving yaw is the magnetometer.
                zero_vertical_velocity: false,
                ..RbpfConfig::default()
            },
        )
        .unwrap();

        let mag = MagnetometerYawMeasurement {
            mag_x: truth_yaw.cos(),
            mag_y: -truth_yaw.sin(),
            mag_z: 0.0,
            noise_std: 0.05,
            apply_declination: false,
            is_enu: false,
            ..MagnetometerYawMeasurement::default()
        };
        assert_approx_eq!(
            mag.get_measurement(&rbpf.nominal_state_vector()).unwrap()[0],
            truth_yaw,
            1e-12
        );

        let before = rbpf.estimate().0[8];
        assert_approx_eq!(before, 0.0, 1e-12);
        for _ in 0..20 {
            rbpf.update(&mag).unwrap();
        }
        let after = rbpf.estimate().0[8];

        let closed = (after - before) / (truth_yaw - before);
        assert!(
            closed > 0.9,
            "the magnetometer closed {:.1}% of a {:.1} deg heading offset; a heading that \
             reaches only the particle weights closes ~0% of it (#341). Yaw went {:.3} -> \
             {:.3} deg against a measured {:.3} deg",
            closed * 100.0,
            (truth_yaw - before).to_degrees(),
            before.to_degrees(),
            after.to_degrees(),
            truth_yaw.to_degrees()
        );
    }

    /// The reported attitude is a mean on the circle, on the principal branch (#341, #314).
    ///
    /// A cloud straddling the +/-pi cut is the case a linear weighted mean gets not merely
    /// imprecise but backwards, and `NavigationResult` copies these three channels straight
    /// through without wrapping, so `estimate` is the only place the invariant can hold.
    #[test]
    fn rbpf_attitude_estimate_is_a_circular_mean_on_the_principal_branch() {
        let nominal = StrapdownState {
            attitude: Rotation3::from_euler_angles(0.0, 0.0, std::f64::consts::PI),
            ..StrapdownState::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            nominal,
            RbpfConfig {
                num_particles: 2,
                ..RbpfConfig::default()
            },
        )
        .unwrap();

        // Straddle the cut: nominal yaw is +pi, so error states of -0.1 and +0.1 rad put the
        // two particles at +179.43 deg and -179.43 deg. Their mean heading is 180 deg; their
        // linear mean is pi, which is only right because the errors happen to be symmetric,
        // so skew it to make the two answers differ.
        rbpf.particles[0].linear_state[YAW_ERROR_STATE_INDEX] = -0.1;
        rbpf.particles[1].linear_state[YAW_ERROR_STATE_INDEX] = 0.3;
        let yaws: Vec<f64> = rbpf
            .particles
            .iter()
            .map(|p| rbpf.particle_state_vector(p)[8])
            .collect();
        assert!(
            yaws[0] > 0.0 && yaws[1] < 0.0,
            "test setup should straddle the branch cut, got {yaws:?}"
        );

        let (mean, cov) = rbpf.estimate();
        let yaw = mean[8];
        assert!(
            (-std::f64::consts::PI..=std::f64::consts::PI).contains(&yaw),
            "reported yaw {yaw} is off the principal branch (#314)"
        );
        // pi + (-0.1 + 0.3)/2 = pi + 0.1, wrapped.
        assert_approx_eq!(yaw, crate::wrap_to_pi(std::f64::consts::PI + 0.1), 1e-12);
        // The spread is 0.4 rad, so the variance is (0.2)^2 -- not the ~pi^2 an unwrapped
        // difference against the mean would report.
        assert_approx_eq!(cov[(8, 8)], 0.04, 1e-12);
    }

    /// A tight cloud must be unaffected by averaging on the circle rather than the line.
    ///
    /// The circular mean is the correct estimator, but it would not be worth having if it
    /// moved the answer for ordinary well-behaved clouds: it agrees with the linear mean to
    /// second order in the spread, and this pins that so a future change to
    /// [`circular_mean`] cannot quietly introduce a bias in the normal case.
    #[test]
    fn rbpf_circular_mean_agrees_with_the_linear_mean_for_a_tight_cloud() {
        let angles = [0.30, 0.31, 0.29, 0.305, 0.295];
        let weights = [0.1, 0.3, 0.2, 0.25, 0.15];
        let linear: f64 = angles.iter().zip(weights).map(|(a, w)| a * w).sum();
        let circular = circular_mean(angles.iter().zip(weights));
        assert_approx_eq!(circular, linear, 1e-6);
    }

    /// A cloud with no mean direction returns 0 rather than failing.
    ///
    /// Two antipodal angles of equal weight have a zero resultant, so no direction is more
    /// correct than any other. The documented behaviour is `atan2(0, 0)`; what matters is
    /// that it is finite and does not panic, since this runs in library code.
    #[test]
    fn rbpf_circular_mean_of_an_undirected_cloud_is_finite() {
        let angles = [0.0, std::f64::consts::PI];
        let mean = circular_mean(angles.iter().zip([0.5, 0.5]));
        assert!(
            mean.is_finite(),
            "circular mean should stay finite, got {mean}"
        );
    }

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
        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();

        for (imu, gps) in imu_data.iter().zip(gps_measurements.iter()) {
            rbpf.predict(imu, dt).unwrap();
            rbpf.update(gps).unwrap();
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
        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();

        let meas = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        rbpf.update(&meas).unwrap();

        let weight_sum: f64 = rbpf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);

        let (mean, _) = rbpf.estimate();
        assert!(mean[0].is_finite());
        assert!(mean[1].is_finite());
        assert!(mean[2].is_finite());
    }

    #[test]
    // Quarantined by #295, not tuned around. Correcting the `principal_radii`
    // units in #292 moved the stationary altitude error from 12.49 m to
    // 28.71 m against a 15 m bound that itself had only 1.2x margin over the
    // buggy baseline. The radii change by 0.4%, so the vertical channel is
    // compensating for the old units rather than responding to them; raising
    // the bound to 30 m would hide that. Re-enable when #295 is root-caused.
    //
    // Update from #297: correcting `transport_rate` to Groves 5.44 brought this
    // to 5.19 m, inside the original 15 m bound. #319 then moved it back out to
    // 23.52 m while improving both moving scenarios -- so the 5.19 m was a
    // cancellation between two bugs, not convergence. #321 brought it to 16.66 m
    // and the Jacobian corrections from its review to 31.51 m, all outside it.
    //
    // #325 and #317 -- the missing velocity and position columns of the Coriolis
    // and transport block -- brought it to **13.98 m**, inside the 15 m bound for
    // the first time since #292. Isolated by building the library with only the
    // altitude-row half-step disabled and re-running this test twice per
    // configuration:
    //
    //     pre-#325 library                    31.51 m   FAIL
    //     + #325 velocity + #317 position     13.98 m   PASS
    //     + the altitude-row half-step        35.92 m   FAIL
    //
    // The third line is what made this look like a regression on first reading.
    // That half-step is a real term but it is not what either issue asks for, it
    // is inconsistent on its own (rows 0 and 1 carry the same term and would be
    // left first-order), and it costs 22 m here -- so it is deferred to #338
    // rather than shipped. Recorded because a future reader will otherwise
    // rediscover only the 35.92 m.
    //
    // This test is still `#[ignore]`d. It passes, but against a bound #295 itself
    // calls fitted -- 15 m was chosen for 1.2x margin over a 12.49 m baseline that
    // was the product of two cancelling bugs -- and 13.98 m clears it by 7%.
    // Re-enabling on that number would be #288's mistake twice over. #295's
    // acceptance criterion is a *derived* bound; this measurement is a large step
    // toward it, not a substitute for it.
    //
    // #319 also established what the number actually measures: the truth here is
    // exactly stationary (altitude 1000.0000 m, all three velocities identically
    // zero, at every step), so the whole error is the filter's own altitude
    // climbing away from a fixed truth while it is fed 5 Hz GNSS altitude fixes.
    // That is a filter defect, not a mechanization one, and it is what #295 has
    // to explain before this test means anything. Four corrections to the
    // mechanization and its linearisation have now moved the number around
    // without closing the gap, which is the evidence for that reading.
    #[ignore = "RBPF vertical channel was tuned against the pre-#292 radii bug -- see #295"]
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

        let duration_seconds = 600;
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

        assert_solution_close_to_truth(&mean, truth, 25.0, 15.0, 0.5);
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

        let duration_seconds = 600;
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
        // Horizontal and velocity bounds are accuracy bounds and hold with three
        // orders of magnitude to spare (0.003 m, 0.003 m/s observed). The altitude
        // bound is not: it is an anti-divergence guard on the vertical channel
        // #295 has already flagged as untrustworthy. Final altitude error across
        // the three scenarios, as the mechanization was corrected:
        //
        //     scenario    pre-#297  post-#297  post-#319  post-#321  +Jacobian
        //     stationary    28.71 m     5.19 m    23.52 m    16.66 m    31.51 m  (quarantined)
        //     v north       21.58 m    25.89 m    18.00 m    18.02 m    27.15 m
        //     v east        13.74 m    10.36 m     5.74 m     0.51 m     7.86 m
        //
        // The last column is the two `transition_jacobian` corrections that came out
        // of PR review on #321 -- reflecting the Earth and transport rates into the
        // caller's frame, and the transport-to-attitude coupling sign. Splitting them
        // apart gives, for the eastward scenario alone, 0.51 m with neither, 21.39 m
        // with the frame reflection only, 24.06 m with the sign only and 7.86 m with
        // both. Four defensible covariance models, four unrelated answers, while the
        // horizontal and velocity errors never leave 0.003 m and 0.005 m/s. That
        // scatter is the argument for treating this bound as a guard rather than an
        // accuracy claim, and it is #295's whole point: the vertical channel is not
        // converging, so its 600 s endpoint is a sample. Correctness of those two
        // corrections rests on the finite-difference checks in `linearize.rs`, which
        // agree to 1e-12, not on the column above.
        //
        // 50 m is ~2x the worst of the three, matching the horizontal guard beside
        // it: a genuine divergence (1e8 m scale, cf. #266) still trips it, codegen
        // jitter cannot. Do not tighten to the observed value without fixing #295.
        assert_solution_close_to_truth(&mean, truth, 50.0, 50.0, 1.0);
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

        let duration_seconds = 600;
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

    /// #268: every particle's conditional covariance stays identical.
    ///
    /// The linear-state covariance recursion depends only on the shared
    /// transition/noise matrices and the particle's own covariance -- never on
    /// the particle state, weight, or measurement value -- and every update
    /// applies the same H/R to each particle. All particles start from the
    /// same clone, so they must remain bit-identical forever. This invariant
    /// is what allows hoisting the recursion out of the per-particle loop.
    #[test]
    fn rbpf_particle_covariances_stay_identical() {
        use crate::measurements::{
            GPSPositionAndVelocityMeasurement, MagnetometerYawMeasurement,
            RelativeAltitudeMeasurement,
        };

        let nominal = StrapdownState {
            latitude: 0.7,
            longitude: -1.3,
            altitude: 100.0,
            velocity_north: 5.0,
            velocity_east: 2.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };
        let config = RbpfConfig {
            num_particles: 20,
            seed: 42,
            ..RbpfConfig::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();
        let imu = IMUData {
            accel: Vector3::new(0.1, 0.05, 9.81),
            gyro: Vector3::new(0.001, -0.002, 0.0005),
        };
        let gps = GPSPositionAndVelocityMeasurement {
            latitude: 0.7,
            longitude: -1.3,
            altitude: 100.0,
            northward_velocity: 5.0,
            eastward_velocity: 2.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };
        let baro = RelativeAltitudeMeasurement {
            relative_altitude: 0.5,
            reference_altitude: 100.0,
        };
        let mag = MagnetometerYawMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: -45.0,
            noise_std: 0.2,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
            is_enu: false, // NED fixture
        };
        for _ in 0..50 {
            rbpf.predict(&imu, 0.1).unwrap();
            rbpf.update(&gps).unwrap();
            rbpf.update(&baro).unwrap();
            rbpf.update(&mag).unwrap();
        }
        let reference = rbpf.particles[0].linear_cov.clone();
        for (i, particle) in rbpf.particles.iter().enumerate().skip(1) {
            assert_eq!(
                particle.linear_cov, reference,
                "particle {i} covariance diverged from particle 0 (see #268)"
            );
        }
    }
}
