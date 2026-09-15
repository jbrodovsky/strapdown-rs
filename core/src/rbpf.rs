//! Rao-Blackwellized particle filter (RBPF) for inertial navigation.
//!
//! This filter represents position errors with particles and uses a shared
//! linear Kalman filter for velocity/attitude error states. It is intended
//! for map-matching and GNSS-aided navigation where measurements are highly
//! nonlinear in position but linear in the remaining states.

use crate::StrapdownError;
use crate::gating::{
    GatePolicy, GateRecovery, InnovationGate, UpdateOutcome, normalized_innovation_squared,
};
use crate::horizontal_meters_to_radians;
use crate::kalman::{expand_measurement_jacobian, imu_sample_from_input};
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

/// Width of the navigation estimate the filter reports: position, velocity and attitude.
///
/// The public spelling of the same number is
/// [`crate::sim::PARTICLE_FILTER_STATES`](crate::sim::PARTICLE_FILTER_STATES), which is where
/// the geophysical bias states are indexed from.
const NAVIGATION_STATE_DIM: usize = POSITION_STATE_DIM + LINEAR_STATE_DIM_BASE;

/// The two spellings of nine must stay the same nine: `sim` indexes this filter's geophysical
/// bias states from its own copy, so a change here that did not reach there would silently
/// read a navigation state as a map bias. A compile error is the cheapest place to catch that.
const _: () = assert!(NAVIGATION_STATE_DIM == crate::sim::PARTICLE_FILTER_STATES);

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

/// Convert a (north, east, up) standard deviation in metres to the units a particle's
/// [`RbpfParticle::position_error`] carries: radians of latitude, radians of longitude,
/// metres of altitude.
///
/// The two horizontal factors differ by `cos(latitude)` and are supplied by
/// [`crate::horizontal_meters_to_radians`], which reads them off the WGS84 radii of
/// curvature at the point given. Applying the latitude factor to both -- what this did
/// until #331 -- leaves the east extent short by that cosine, so a cloud asked for 10 m
/// got 7.7 m at 40 degrees of latitude and 1.7 m at 80. The altitude entry is already in
/// the state's units and passes through.
fn position_std_to_state_units(
    std_m: &Vector3<f64>,
    latitude_rad: f64,
    altitude_m: f64,
) -> Vector3<f64> {
    let (latitude_radians_per_meter, longitude_radians_per_meter) =
        horizontal_meters_to_radians(latitude_rad.to_degrees(), altitude_m);
    Vector3::new(
        std_m[0] * latitude_radians_per_meter,
        std_m[1] * longitude_radians_per_meter,
        std_m[2],
    )
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
    /// Initial position-error standard deviation in metres, as (north, east, up) extent
    /// on the ground. The horizontal entries are converted to the radian units the
    /// particles carry using the WGS84 radii of curvature at the nominal position:
    /// `1 / (R_N + h)` radians of latitude per metre of northing, and
    /// `1 / ((R_E + h) cos(latitude))` radians of longitude per metre of easting. The
    /// altitude entry is used in metres directly.
    ///
    /// Until #331 both horizontal entries took the latitude factor
    /// ([`crate::earth::METERS_TO_DEGREES`] in radians), which left the longitude sigma
    /// short by `cos(latitude)` and so spread the cloud only 7.7 m east-west of a
    /// requested 10 m at 40 degrees, and 1.7 m at 80 -- under-spread, increasingly so
    /// towards the poles, in the one direction a disagreeing fix then depletes.
    pub position_init_std_m: Vector3<f64>,
    /// Initial standard deviation of each of the three velocity error states, in m/s
    /// (applied uniformly to north, east and vertical).
    pub velocity_init_std_mps: f64,
    /// Initial standard deviation of each of the three attitude error states, in
    /// radians (applied uniformly to roll, pitch and yaw).
    pub attitude_init_std_rad: f64,
    /// Position random-walk rate, as (north, east, up) in **m/sqrt(s)** despite the
    /// `_m` in the name, which is kept for compatibility with existing configuration
    /// files. The predict step forms the per-step standard deviation as
    /// `std * sqrt(dt)` and converts the horizontal pair to radians exactly as
    /// [`RbpfConfig::position_init_std_m`] does, so the per-step variance is
    /// proportional to the elapsed time and the numeric value is unchanged at a 1 s
    /// step.
    ///
    /// Until #331 this was `std * dt`, which is a per-step variance proportional to
    /// `dt^2`: 100x too small at 100 Hz, and dependent on the log's sample rate rather
    /// than on elapsed time alone.
    ///
    /// It must cover unmodelled position wander between fixes: under the reference
    /// degraded profile (`Degraded { sigma_pos_m: 3.0 }`, 5 s fixes) the default 1 m
    /// starves the particle cloud (see #267) -- raise it explicitly in that
    /// configuration. Kept at 1 here because a wider default proposal measurably
    /// degrades clean stationary tracking.
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
    ///
    /// Like every other state in this filter they are carried as an error relative to a
    /// nominal, here [`RaoBlackwellizedParticleFilter::nominal_extra_state`]: a particle's
    /// entry is its deviation from that nominal, not the absolute bias. Read the absolute
    /// estimate back with [`RaoBlackwellizedParticleFilter::extra_state_estimate`].
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
    /// every one of its linear error states, so the cloud stays zero-mean in all of them.
    ///
    /// The mean position error is always applied to the nominal state; the mean
    /// velocity/attitude error is applied only when a linear (Kalman) update has run since
    /// the previous recentring. Note that the subtraction from the particles is
    /// unconditional, so in the other case that mean is **discarded** rather than deferred
    /// -- it is removed from the cloud without ever reaching the nominal state.
    ///
    /// The [`RbpfConfig::extra_state_dim`] states appended after velocity and attitude are
    /// recentred on the same terms, and their mean is never discarded. They have no home in
    /// the nine navigation states [`crate::linearize::apply_eskf_correction`] knows how to
    /// write to, which is why they used to be skipped here (#333), so they carry a nominal
    /// of their own -- [`RaoBlackwellizedParticleFilter::nominal_extra_state`] -- and the
    /// mean is moved into it. A geophysical map bias therefore keeps its absolute value,
    /// readable via [`RaoBlackwellizedParticleFilter::extra_state_estimate`], while the
    /// particles carry only its spread.
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
    /// (roll, pitch, yaw), then [`RbpfConfig::extra_state_dim`] extra states. The extra
    /// entries are errors relative to
    /// [`RaoBlackwellizedParticleFilter::nominal_extra_state`], not absolute values, in the
    /// same way the six before them are errors relative to the nominal trajectory.
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
    /// Nominal value of the [`RbpfConfig::extra_state_dim`] extra states, which the
    /// particles' extra entries are errors against. Empty when there are none.
    ///
    /// The navigation nominal is a [`StrapdownState`], which has nowhere to put a map bias,
    /// so the extra states get this second nominal rather than no nominal at all. Without
    /// it [`Self::recenter_errors`] could only leave them un-recentred or subtract their
    /// mean and lose it (#333).
    nominal_extra: DVector<f64>,
    rng: StdRng,
    linear_update_applied: bool,
    /// Innovation gate applied by `update` together with the recovery policy that keeps
    /// a rejection from being permanent; an empty gate accepts every measurement.
    gate_policy: GatePolicy,
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

        let pos_std = position_std_to_state_units(
            &config.position_init_std_m,
            nominal.latitude,
            nominal.altitude,
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

        let nominal_extra = DVector::zeros(config.extra_state_dim);

        Ok(Self {
            config,
            particles,
            nominal,
            nominal_extra,
            rng,
            linear_update_applied: false,
            gate_policy: GatePolicy::default(),
        })
    }

    /// Access the nominal INS state.
    pub const fn nominal_state(&self) -> &StrapdownState {
        &self.nominal
    }

    /// Access the nominal value of the extra states; empty when `extra_state_dim` is zero.
    ///
    /// This is the half of an extra state that recentring accumulates, not the estimate:
    /// the particles hold the rest. [`Self::extra_state_estimate`] adds the two.
    pub const fn nominal_extra_state(&self) -> &DVector<f64> {
        &self.nominal_extra
    }

    /// Weighted-mean estimate of the extra states -- their nominal plus the cloud's mean
    /// error. Empty when [`RbpfConfig::extra_state_dim`] is zero.
    ///
    /// With [`RbpfConfig::recenter_after_update`] on, the mean error term is ~0 immediately
    /// after an update and the estimate is essentially the nominal; with it off, the
    /// nominal stays at zero and the estimate is essentially the cloud mean. Both are the
    /// same quantity, which is the point of splitting it.
    pub fn extra_state_estimate(&self) -> DVector<f64> {
        let mut estimate = self.nominal_extra.clone();
        for particle in &self.particles {
            for i in 0..self.config.extra_state_dim {
                estimate[i] += particle.weight * particle.linear_state[LINEAR_STATE_DIM_BASE + i];
            }
        }
        estimate
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

        // Scale process noise with dt to approximate a continuous-time random walk. Such a
        // walk accumulates *variance* linearly in time -- `var(dt) = q dt` -- so the
        // standard deviation goes as `sqrt(dt)`, not as `dt`. Scaling the standard
        // deviation by `dt` made `q_n` proportional to `dt^2`: at the 5-100 Hz this crate
        // runs at, one to two orders of magnitude smaller than the configuration asked for,
        // and a function of the log's sample rate rather than of elapsed time alone -- a
        // 100 Hz log and a 50 Hz log of the same trajectory were given process noise
        // differing by 4x per step. Fixed in #331, whose scope is the position block: the
        // velocity, attitude and extra-state terms below still scale their standard
        // deviations by `dt` and have the same argument against them.
        let pos_noise = position_std_to_state_units(
            &self.config.position_process_noise_std_m,
            self.nominal.latitude,
            self.nominal.altitude,
        ) * dt.sqrt();
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
        self.weighted_moments(&states, NAVIGATION_STATE_DIM)
    }

    /// [`Self::estimate`] over the whole state the filter carries: the nine navigation states
    /// with the [`RbpfConfig::extra_state_dim`] extra states appended, and the covariance of
    /// all of them.
    ///
    /// Identical to [`Self::estimate`] when there are no extra states, which is every
    /// configuration but geophysical aiding.
    ///
    /// This exists because summarising the cloud as a 9-vector is not a harmless truncation
    /// once extra states are configured, and it has cost the two consumers below in the two
    /// different ways a dropped state can cost you.
    ///
    /// **Scoring a measurement.** A model that reads a state by index *from the end* gets the
    /// wrong entry rather than none: a geophysical map bias declared as "one from the end"
    /// resolves to the yaw angle in a 9-vector, so [`Self::evaluate_ensemble_gate`] scored
    /// every geophysical fix with an attitude angle substituted for the bias, while the weight
    /// update -- which goes through [`Self::particle_state_vector_full`] -- used the real one.
    ///
    /// **Reporting the estimate.** The dropped states are the ones geophysical aiding exists
    /// to produce. A gravity-aided run really does estimate a map bias and it really does
    /// move, but a nine-element summary has nowhere to put it, so
    /// [`crate::sim::NavigationResult::from_particle_filter`] wrote rows whose geophysical
    /// columns were empty for a run that had estimated them. Pair this with
    /// [`crate::sim::NavigationResult::from_particle_filter_with_geo`], which is told by a
    /// [`GeoStateLayout`](crate::sim::GeoStateLayout) which extra state is which; the vector
    /// itself cannot say, since a ten-element estimate is gravity-only or magnetic-only
    /// depending on the run's flags.
    ///
    /// [`Self::estimate`] is kept as the nine-state accessor rather than widened because the
    /// navigation solution is what nearly every caller wants, and its width should not depend
    /// on how the filter happens to be configured.
    pub fn estimate_with_extra_states(&self) -> (DVector<f64>, DMatrix<f64>) {
        if self.config.extra_state_dim == 0 {
            return self.estimate();
        }
        let states: Vec<DVector<f64>> = self
            .particles
            .iter()
            .map(|particle| self.particle_state_vector_full(particle))
            .collect();
        self.weighted_moments(&states, NAVIGATION_STATE_DIM + self.config.extra_state_dim)
    }

    /// Weighted mean and covariance of an assembled cloud, attitude handled on the circle.
    ///
    /// Shared by [`Self::estimate`] and [`Self::estimate_with_extra_states`], which differ
    /// only in how wide the assembled states are. [`ATTITUDE_STATE_INDICES`] addresses the
    /// same three channels in both, because the extra states are appended after them.
    ///
    /// `dim` is the caller's own width rather than `states[0].len()` so that a cloud with no
    /// particles still reports the shape its caller promised: the two accessors have
    /// different contracts, and inferring the width would collapse them both to whatever the
    /// fallback happened to be.
    fn weighted_moments(
        &self,
        states: &[DVector<f64>],
        dim: usize,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let mut mean = DVector::<f64>::zeros(dim);
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

        let mut cov = DMatrix::<f64>::zeros(dim, dim);
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
    /// Takes `&mut self` because a rejection is not inert: the cloud is spread out by
    /// [`GateRecovery::rejection_inflation`] on the way out, which is this filter's form
    /// of the covariance inflation the Kalman filters apply, and without it one rejection
    /// would make every subsequent fix disagree by more (#340).
    ///
    /// Always computes the NIS, gate or no gate, because
    /// [`sim::health::HealthMonitor`](crate::sim::health::HealthMonitor) consumes it
    /// to catch a filter that has diverged rather than merely been handed one bad
    /// fix. The summary this costs is the same one `run_closed_loop` computes
    /// immediately afterwards for logging.
    ///
    /// The cloud is summarised with [`Self::estimate_with_extra_states`] rather than
    /// [`Self::estimate`], so the measurement sees the same state layout here as it does in
    /// the weight update. A geophysical model reads its map bias by index from the end of
    /// the vector, and a 9-vector has no bias to read.
    ///
    /// # Errors
    /// Whatever the measurement model returns when evaluated at the ensemble mean, or
    /// a singular innovation covariance from
    /// [`normalized_innovation_squared`](crate::gating::normalized_innovation_squared).
    fn evaluate_ensemble_gate<M: MeasurementModel + ?Sized>(
        &mut self,
        measurement: &M,
    ) -> Result<UpdateOutcome, StrapdownError> {
        let (mean, covariance) = self.estimate_with_extra_states();
        // Jacobian first: a geophysical model off the edge of its map reports that
        // here -- and so does one handed a state vector with no room for the bias it
        // was told to read -- whereas `get_expected_measurement` would quietly return
        // NaN or silently drop the bias.
        let h = measurement.get_jacobian(&mean)?;
        let z_hat = measurement.get_expected_measurement(&mean);
        let mut innovation = measurement.get_measurement(&mean)? - z_hat;
        measurement.wrap_residual(&mut innovation);

        // Pad to the summary's width before multiplying. The navigation models return a
        // fixed nine columns -- `gps_position_jacobian` is 3x9 whatever it is handed -- so
        // against an augmented covariance this product was a 3x9 by an 11x11, which nalgebra
        // panics on. GNSS and geophysical fixes ride the same event stream, so that is every
        // aided run rather than a corner case. The Kalman filters have always widened here;
        // this is the same helper, and it rejects a Jacobian *wider* than the state rather
        // than truncating one.
        let h = expand_measurement_jacobian(h, covariance.ncols())?;

        let s = &h * &covariance * h.transpose() + measurement.get_noise();
        let dof = innovation.len();
        let nis = normalized_innovation_squared(&innovation, &s)?;
        let decision = self.gate_policy.decide(nis, dof, "RBPF");
        if !decision.outcome.accepted {
            // Inflate the cloud, not a matrix: see `inflate_particle_spread` for why
            // scaling every particle's deviation from the ensemble mean is the same
            // operation the Kalman filters perform on their covariance (#340).
            self.inflate_particle_spread(decision.covariance_inflation);
        }
        Ok(decision.outcome)
    }

    /// Multiply the ensemble covariance by `factor` by spreading the particles out.
    ///
    /// The Kalman filters recover from a gated-out measurement by scaling $P$; an
    /// ensemble has no $P$ to scale, so the equivalent operation is performed on the
    /// cloud itself. Replacing every particle's error state by
    /// $\bar{x} + \sqrt{f}(x_i - \bar{x})$ -- its deviation from the weighted ensemble
    /// mean, stretched -- multiplies the weighted sample covariance by exactly $f$ while
    /// leaving the mean, the weights and the particle identities alone. This is the
    /// multiplicative inflation of the ensemble-filter literature (Anderson & Anderson
    /// 1999), and it needs no draw from the RNG, so a seeded run stays reproducible.
    ///
    /// The stretch is what re-opens the gate, and it is enough on its own to do so:
    /// [`Self::evaluate_ensemble_gate`] scores against [`Self::weighted_moments`], which is
    /// the weighted *spread* of the particle states -- position, velocity, attitude and
    /// extra states alike -- so scaling every deviation by $\sqrt{f}$ multiplies the exact
    /// covariance the next NIS is computed from by $f$. `inflating_the_cloud_multiplies_its\
    /// _covariance_by_the_factor` asserts that on the same summary the gate uses.
    ///
    /// Each particle's own `linear_cov` is scaled by $f$ as well. That does not enter the
    /// gate's covariance, but it does enter each particle's own Kalman update, and leaving
    /// it behind would produce a wide cloud of individually over-confident particles -- the
    /// conditional half of the uncertainty contradicting the ensemble half.
    ///
    /// Unlike the Kalman filters, which inflate only the subspace the rejected measurement
    /// observed, this stretches the cloud in every direction: a weighted ensemble has no
    /// covariance matrix to project a Jacobian through, and re-weighting particles towards
    /// the observed subspace would change the distribution rather than its spread. The cost
    /// is the one [`GateDecision::inflate`](crate::gating::GateDecision::inflate) documents
    /// -- a rejection on one sensor widens every other sensor's gain -- so a gated RBPF
    /// aided by several sensors at different rates deserves more scepticism here than a
    /// gated ESKF does.
    ///
    /// A `factor` at or below 1, or one that is not finite, is a no-op.
    fn inflate_particle_spread(&mut self, factor: f64) {
        if !factor.is_finite() || factor <= 1.0 || self.particles.is_empty() {
            return;
        }
        let scale = factor.sqrt();
        let linear_dim = self.linear_state_dim();
        let mut mean_position = Vector3::zeros();
        let mut mean_linear = DVector::zeros(linear_dim);
        for particle in &self.particles {
            mean_position += particle.weight * particle.position_error;
            mean_linear += particle.weight * &particle.linear_state;
        }
        for particle in &mut self.particles {
            particle.position_error =
                mean_position + scale * (particle.position_error - mean_position);
            particle.linear_state = &mean_linear + scale * (&particle.linear_state - &mean_linear);
            particle.linear_cov *= factor;
        }
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

    /// The 9-state solution with the extra states appended, as a measurement model sees it.
    ///
    /// The appended entries are absolute: a particle's error plus [`Self::nominal_extra_state`].
    /// A consumer such as a geophysical bias reads them by index from the end of the vector
    /// and adds them to its predicted measurement, so it must see the whole bias and not
    /// just the part of it the cloud still carries.
    fn particle_state_vector_full(&self, particle: &RbpfParticle) -> DVector<f64> {
        let mut state = self.particle_state_vector(particle).as_slice().to_vec();
        for i in 0..self.config.extra_state_dim {
            state.push(self.nominal_extra[i] + particle.linear_state[LINEAR_STATE_DIM_BASE + i]);
        }
        DVector::from_vec(state)
    }

    /// Move the cloud's weighted-mean error onto the nominal states, leaving it zero-mean.
    ///
    /// Every error state is subtracted from every particle. Where that mean goes differs by
    /// block, because the two nominals accept it on different terms:
    ///
    /// * **Position** -- always folded into the navigation nominal.
    /// * **Velocity and attitude** -- folded in only when a linear (Kalman) update has run
    ///   since the last recentring, and otherwise discarded. See
    ///   [`RbpfConfig::recenter_after_update`].
    /// * **Extra states** -- always folded into [`Self::nominal_extra_state`]. They have no
    ///   row in the nine-state correction [`crate::linearize::apply_eskf_correction`]
    ///   applies, which is why recentring used to skip them outright (#333); a separate
    ///   nominal is what lets their mean be moved rather than either kept in the cloud or
    ///   lost.
    ///
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

        for i in 0..self.config.extra_state_dim {
            self.nominal_extra[i] += mean_lin[LINEAR_STATE_DIM_BASE + i];
        }

        // Every row of `mean_lin` is consumed: the base six by `delta_x` above, the rest by
        // `nominal_extra`. Subtracting the whole vector is what makes the cloud zero-mean in
        // the extra states as well as the navigation ones.
        let linear_dim = self.linear_state_dim();
        for particle in &mut self.particles {
            particle.position_error -= mean_pos;
            for i in 0..linear_dim {
                particle.linear_state[i] -= mean_lin[i];
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
        self.gate_policy.set_gate(gate);
        true
    }

    fn set_gate_recovery(&mut self, recovery: GateRecovery) -> bool {
        self.gate_policy.set_recovery(recovery);
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

    /// Latitude used by the #331 spread tests. Well away from the equator, where the
    /// `cos(latitude)` the longitude conversion needs is 0.5 -- so the defect the tests
    /// guard against halves the east spread rather than nudging it.
    const SPREAD_TEST_LATITUDE_DEG: f64 = 60.0;
    const SPREAD_TEST_LONGITUDE_DEG: f64 = -105.0;
    const SPREAD_TEST_ALTITUDE_M: f64 = 1000.0;

    /// Root-mean-square of a sample, which is its standard deviation when its mean is zero.
    fn root_mean_square(values: impl Iterator<Item = f64>) -> f64 {
        let (sum_of_squares, count) = values.fold((0.0, 0_usize), |(sum, n), value| {
            (value.mul_add(value, sum), n + 1)
        });
        (sum_of_squares / count as f64).sqrt()
    }

    /// A particle's position error as a ground displacement in metres, measured
    /// independently of the conversion under test.
    ///
    /// Great-circle distance on a sphere rather than the WGS84 radii the filter divides
    /// by, so it cannot agree with the code under test by construction. The two differ by
    /// a few tenths of a percent, which is why the assertions below carry a 3% tolerance
    /// and not a tighter one.
    fn position_error_ground_meters(
        nominal: &StrapdownState,
        position_error: &Vector3<f64>,
    ) -> Vector3<f64> {
        let north_m = earth::haversine_distance(
            nominal.latitude,
            nominal.longitude,
            nominal.latitude + position_error[0],
            nominal.longitude,
        );
        let east_m = earth::haversine_distance(
            nominal.latitude,
            nominal.longitude,
            nominal.latitude,
            nominal.longitude + position_error[1],
        );
        Vector3::new(north_m, east_m, position_error[2])
    }

    fn spread_test_nominal_state() -> StrapdownState {
        StrapdownState {
            latitude: SPREAD_TEST_LATITUDE_DEG.to_radians(),
            longitude: SPREAD_TEST_LONGITUDE_DEG.to_radians(),
            altitude: SPREAD_TEST_ALTITUDE_M,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        }
    }

    /// The initial cloud spreads the requested metres east as well as north (#331).
    ///
    /// `position_init_std_m` is a ground extent, so a requested 10 m must put 10 m of
    /// east-west spread on the ground at any latitude. Converting the east entry with the
    /// latitude factor -- [`earth::METERS_TO_DEGREES`] is degrees *of latitude* per metre --
    /// omits the `cos(latitude)` that belongs in the denominator, shrinking the east spread
    /// to `10 cos(latitude)`: 5 m at the 60 degrees used here. An under-spread prior is the
    /// condition that depletes a particle filter the first time a fix disagrees with it, so
    /// this is a correctness guard and not a tuning preference.
    #[test]
    fn rbpf_initial_cloud_spreads_the_requested_metres_in_both_directions() {
        const REQUESTED_HORIZONTAL_STD_M: f64 = 10.0;
        const REQUESTED_VERTICAL_STD_M: f64 = 5.0;

        let nominal = spread_test_nominal_state();
        let config = RbpfConfig {
            num_particles: 50_000,
            position_init_std_m: Vector3::new(
                REQUESTED_HORIZONTAL_STD_M,
                REQUESTED_HORIZONTAL_STD_M,
                REQUESTED_VERTICAL_STD_M,
            ),
            seed: 331,
            ..RbpfConfig::default()
        };
        let rbpf = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();

        let ground: Vec<Vector3<f64>> = rbpf
            .particles
            .iter()
            .map(|particle| position_error_ground_meters(&nominal, &particle.position_error))
            .collect();

        for (axis, name, requested) in [
            (0, "north", REQUESTED_HORIZONTAL_STD_M),
            (1, "east", REQUESTED_HORIZONTAL_STD_M),
            (2, "up", REQUESTED_VERTICAL_STD_M),
        ] {
            let observed = root_mean_square(ground.iter().map(|v| v[axis]));
            let relative_error = (observed - requested).abs() / requested;
            assert!(
                relative_error < 0.03,
                "{name} spread is {observed:.3} m for a requested {requested:.3} m at \
                 {SPREAD_TEST_LATITUDE_DEG} deg latitude; the east entry comes out \
                 cos(latitude) = {:.3} times the request when it is converted with the \
                 latitude factor (#331)",
                SPREAD_TEST_LATITUDE_DEG.to_radians().cos()
            );
        }
    }

    /// Inflating the cloud multiplies its covariance by the requested factor (#340).
    ///
    /// The Kalman filters recover from a gated-out measurement by scaling `P`; this is the
    /// same operation for an ensemble, and the claim worth checking is that it really is
    /// the same operation: the sample covariance scales by exactly the factor asked for,
    /// and the ensemble mean -- the navigation solution the run reports -- does not move.
    #[test]
    fn inflating_the_cloud_multiplies_its_covariance_by_the_factor() {
        const FACTOR: f64 = 4.0;

        let nominal = spread_test_nominal_state();
        let config = RbpfConfig {
            num_particles: 2_000,
            seed: 340,
            ..RbpfConfig::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();

        let (mean_before, covariance_before) = rbpf.estimate();
        rbpf.inflate_particle_spread(FACTOR);
        let (mean_after, covariance_after) = rbpf.estimate();

        for index in 0..mean_before.len() {
            assert_approx_eq!(
                mean_after[index],
                mean_before[index],
                1e-12 * mean_before[index].abs().max(1.0)
            );
        }
        for index in 0..covariance_before.nrows() {
            let expected = FACTOR * covariance_before[(index, index)];
            assert_approx_eq!(
                covariance_after[(index, index)],
                expected,
                1e-9 * expected.abs().max(1e-12)
            );
        }

        // A factor that would shrink the cloud, or one that is not a number, is a no-op
        // rather than a way to collapse the ensemble.
        for inert in [1.0, 0.5, f64::NAN] {
            let (mean, covariance) = rbpf.estimate();
            rbpf.inflate_particle_spread(inert);
            let (mean_now, covariance_now) = rbpf.estimate();
            assert_approx_eq!(mean_now[0], mean[0], 1e-15);
            assert_approx_eq!(covariance_now[(0, 0)], covariance[(0, 0)], 1e-18);
        }
    }

    /// Process noise accumulates with elapsed time, not with the log's sample rate (#331).
    ///
    /// `position_process_noise_std_m` names a continuous-time random walk, which accumulates
    /// *variance* linearly in time, so its standard deviation scales as `sqrt(dt)` and one
    /// second of propagation must produce the same spread however it is subdivided. Scaling
    /// the standard deviation by `dt` instead makes the per-step variance go as `dt^2`: the
    /// 100 Hz run below then ends a factor of ten tighter than the 10 Hz one, and both end
    /// one to two orders of magnitude tighter than the configuration asked for.
    ///
    /// The east channel is checked in metres alongside the others, so this also pins the
    /// `cos(latitude)` conversion on the process-noise path that
    /// `rbpf_initial_cloud_spreads_the_requested_metres_in_both_directions` pins on the
    /// initialization path.
    #[test]
    fn rbpf_process_noise_depends_on_elapsed_time_not_sample_rate() {
        const RATE_M_PER_SQRT_S: f64 = 1.0;
        const ELAPSED_S: f64 = 1.0;

        // One second of propagation, at two sample rates a decade apart.
        let spreads: Vec<Vector3<f64>> = [(100, 0.01), (10, 0.1)]
            .iter()
            .map(|&(steps, dt)| propagated_position_spread_m(steps, dt))
            .collect();

        let expected = RATE_M_PER_SQRT_S * ELAPSED_S.sqrt();
        for (axis, name) in [(0, "north"), (1, "east"), (2, "up")] {
            let fast = spreads[0][axis];
            let slow = spreads[1][axis];
            assert!(
                (fast - slow).abs() / expected < 0.05,
                "{name} spread after 1 s is {fast:.4} m at 100 Hz but {slow:.4} m at 10 Hz; \
                 process noise must depend on elapsed time alone (#331)"
            );
            for (observed, rate_hz) in [(fast, 100), (slow, 10)] {
                assert!(
                    (observed - expected).abs() / expected < 0.05,
                    "{name} spread after 1 s at {rate_hz} Hz is {observed:.4} m, not the \
                     {expected:.4} m a {RATE_M_PER_SQRT_S} m/sqrt(s) random walk accumulates \
                     (#331)"
                );
            }
        }
    }

    /// Spread in ground metres of a stationary cloud after `steps` predictions of `dt`.
    ///
    /// Everything but the position process noise is turned off -- a point-mass initial
    /// cloud, negligible velocity and attitude uncertainty, no velocity/attitude process
    /// noise -- so the only thing that reaches `position_error` is the term under test.
    fn propagated_position_spread_m(steps: usize, dt: f64) -> Vector3<f64> {
        let nominal = spread_test_nominal_state();
        let gravity = earth::gravity(&SPREAD_TEST_LATITUDE_DEG, &SPREAD_TEST_ALTITUDE_M);
        let config = RbpfConfig {
            num_particles: 50_000,
            // Not zero: these are standard deviations of a normal distribution, and the
            // point is to start the cloud from a point mass, not to exercise the
            // degenerate-distribution path.
            position_init_std_m: Vector3::new(1e-9, 1e-9, 1e-9),
            velocity_init_std_mps: 1e-9,
            attitude_init_std_rad: 1e-12,
            position_process_noise_std_m: Vector3::new(1.0, 1.0, 1.0),
            velocity_process_noise_std_mps: 0.0,
            attitude_process_noise_std_rad: 0.0,
            seed: 331,
            ..RbpfConfig::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();

        let imu = IMUData {
            accel: Vector3::new(0.0, 0.0, gravity),
            gyro: Vector3::zeros(),
        };
        for _ in 0..steps {
            rbpf.predict(&imu, dt).unwrap();
        }

        let ground: Vec<Vector3<f64>> = rbpf
            .particles
            .iter()
            .map(|particle| position_error_ground_meters(&nominal, &particle.position_error))
            .collect();
        Vector3::new(
            root_mean_square(ground.iter().map(|v| v[0])),
            root_mean_square(ground.iter().map(|v| v[1])),
            root_mean_square(ground.iter().map(|v| v[2])),
        )
    }

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

    /// The extra linear states, and their variances, come back through
    /// [`RaoBlackwellizedParticleFilter::estimate_with_extra_states`].
    ///
    /// The regression: `estimate` is a nine-state summary, so with extra states configured it
    /// silently drops exactly the quantity geophysical aiding exists to produce. Asserted
    /// against moments computed straight off the cloud, so the estimator has to agree with
    /// the particles rather than merely with itself.
    #[test]
    fn rbpf_estimate_with_extra_states_reports_the_extra_states_and_their_variance() {
        const EXTRA_INIT_STD: f64 = 3.0;

        let nominal = StrapdownState {
            latitude: 0.7,
            longitude: -1.3,
            altitude: 100.0,
            ..StrapdownState::default()
        };
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            nominal,
            RbpfConfig {
                num_particles: 400,
                extra_state_dim: 2,
                extra_state_init_std: EXTRA_INIT_STD,
                seed: 7,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        // Skew the weights so a weighted mean is not the same number as a plain one; an
        // unweighted implementation would otherwise pass this by coincidence.
        let count = rbpf.particles.len();
        let mut total = 0.0;
        for (i, particle) in rbpf.particles.iter_mut().enumerate() {
            particle.weight = 1.0 + (i % 5) as f64;
            total += particle.weight;
        }
        for particle in &mut rbpf.particles {
            particle.weight /= total;
        }

        let (nav_mean, nav_cov) = rbpf.estimate();
        let (full_mean, full_cov) = rbpf.estimate_with_extra_states();

        assert_eq!(
            nav_mean.len(),
            9,
            "`estimate` keeps its nine-state contract"
        );
        assert_eq!(nav_cov.shape(), (9, 9));
        assert_eq!(
            full_mean.len(),
            11,
            "the two extra states must be appended after the nine navigation states"
        );
        assert_eq!(full_cov.shape(), (11, 11));

        // The extra states are appended, so the navigation block must be the same answer.
        for i in 0..9 {
            assert_approx_eq!(full_mean[i], nav_mean[i], 1e-12);
            assert_approx_eq!(full_cov[(i, i)], nav_cov[(i, i)], 1e-12);
        }

        // Moments of the cloud, computed here rather than by the code under test.
        for extra in 0..2 {
            let row = LINEAR_STATE_DIM_BASE + extra;
            let expected_mean: f64 = rbpf
                .particles
                .iter()
                .map(|p| p.weight * p.linear_state[row])
                .sum();
            let expected_var: f64 = rbpf
                .particles
                .iter()
                .map(|p| p.weight * (p.linear_state[row] - expected_mean).powi(2))
                .sum();

            assert_approx_eq!(full_mean[9 + extra], expected_mean, 1e-12);
            assert_approx_eq!(full_cov[(9 + extra, 9 + extra)], expected_var, 1e-12);
            // A bias reported without an uncertainty is not usable, and a variance that came
            // out as zero would mean the draw never happened.
            assert!(
                full_cov[(9 + extra, 9 + extra)] > 0.0,
                "extra state {extra} reported a zero variance over {count} particles drawn \
                 with sigma {EXTRA_INIT_STD}"
            );
        }
    }

    /// With no extra states configured, the two accessors are the same estimate.
    ///
    /// Which is every configuration but geophysical aiding, so this is the case that must not
    /// change shape underneath existing callers.
    #[test]
    fn rbpf_estimate_with_extra_states_matches_estimate_when_there_are_none() {
        let rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState::default(),
            RbpfConfig {
                num_particles: 50,
                seed: 11,
                ..RbpfConfig::default()
            },
        )
        .unwrap();

        let (mean, cov) = rbpf.estimate();
        let (full_mean, full_cov) = rbpf.estimate_with_extra_states();
        assert_eq!(full_mean.len(), 9);
        assert_eq!(full_cov.shape(), (9, 9));
        for i in 0..9 {
            assert_approx_eq!(full_mean[i], mean[i], 1e-15);
            assert_approx_eq!(full_cov[(i, i)], cov[(i, i)], 1e-15);
        }
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

    /// The filter configuration every `rbpf_runs_on_scenario_*` test uses.
    ///
    /// Hoisted out of the runner because [`assert_solution_consistent_with_posterior`] reads
    /// the tuning back off it to anchor the reported posterior: an anchor computed from the
    /// config cannot drift away from the filter it is checking the way a literal can.
    fn scenario_config() -> RbpfConfig {
        RbpfConfig {
            num_particles: 10000,
            position_init_std_m: Vector3::new(10.0, 10.0, 5.0),
            seed: 123,
            zero_vertical_velocity: true,
            zero_vertical_velocity_std_mps: 0.05,
            ..RbpfConfig::default()
        }
    }

    /// Steady-state posterior standard deviation of the **altitude** channel, in metres.
    ///
    /// The altitude channel of this filter is, once it has settled, a scalar random walk of
    /// per-step standard deviation `position_process_noise_std_m[2] * sqrt(dt)` observed
    /// every step by a fix of standard deviation `vertical_noise_std`. Its posterior
    /// variance is the fixed point of the scalar Riccati recursion
    ///
    /// $$ P = \frac{(P + Q) R}{P + Q + R}, \qquad Q = \sigma_w^2, \quad R = \sigma_v^2 $$
    ///
    /// Writing $x = P + Q$ for the predicted variance, that is $x^2 - Qx - QR = 0$, whose
    /// positive root is $x = \tfrac{1}{2}(Q + \sqrt{Q^2 + 4QR})$ and so $P = x - Q$.
    ///
    /// # Altitude only, and why
    ///
    /// This is *not* a general model of the filter's position block, and it is used here
    /// only as the anchor in [`assert_solution_consistent_with_posterior`] -- never as an
    /// error bound. [`RaoBlackwellizedParticleFilter::predict_sample`] draws its position
    /// noise from `f_nl P_l f_nl^T + q_n`, so the velocity and attitude covariance
    /// contributes to every position innovation and the true per-step process noise exceeds
    /// `q_n` alone. Whether that matters is a question about magnitudes, and the two
    /// horizontal channels answer it differently from the vertical one:
    ///
    /// * **Altitude.** The coupling enters through `f[(2, 5)] = ±dt` against a vertical
    ///   velocity variance the zero-vertical-velocity pseudo-measurement holds near
    ///   1.0e-5 m^2/s^2, so `f_nl P_l f_nl^T` contributes ~4e-7 m^2 against a `q_n` of
    ///   0.2 m^2 -- two parts in a million. Measured over the twelve runs tabulated on
    ///   [`rbpf_runs_on_scenario_stationary`], this expression predicts 0.8944 m and the
    ///   filter reports 0.888-0.909 m: agreement to 1.6%.
    /// * **Horizontal.** Nothing aids the horizontal velocity states in these scenarios --
    ///   the fixes are position-only -- so their error variance stays large and the
    ///   coupling dominates. The same expression predicts 1.4623 m per axis where the
    ///   filter reports 1.84-1.88 m north and 1.52-1.57 m east, low by ~26%. A bound built
    ///   on it would be tighter than 3 sigma while claiming to be 3 sigma, and could reject
    ///   a healthy run. The horizontal channels are therefore bounded by the reported
    ///   posterior and are not anchored to any closed form.
    ///
    /// Deriving the horizontal steady state honestly means solving the coupled
    /// position/linear recursion, which is reimplementing the filter inside its own test.
    /// Bounding against the reported posterior instead costs nothing and is exact by
    /// construction; the altitude anchor below is what stops that being circular.
    fn steady_state_altitude_std(config: &RbpfConfig, dt: f64, vertical_noise_std_m: f64) -> f64 {
        let q = (config.position_process_noise_std_m[2] * dt.sqrt()).powi(2);
        let r = vertical_noise_std_m.powi(2);
        let predicted = 0.5 * (q + q.mul_add(q, 4.0 * q * r).sqrt());
        (predicted - q).sqrt()
    }

    /// The reported position covariance as (north, east, up) standard deviations in ground
    /// metres.
    ///
    /// [`RaoBlackwellizedParticleFilter::estimate`] reports latitude and longitude variance
    /// in rad^2, which are not comparable to each other or to the great-circle distance
    /// [`assert_solution_close_to_truth`] measures. Converting with the WGS84 radii of
    /// curvature at the estimate -- and the `cos(latitude)` a radian of longitude carries --
    /// puts all three on the metric the assertion uses.
    fn posterior_position_std_m(mean: &DVector<f64>, cov: &DMatrix<f64>) -> Vector3<f64> {
        let latitude_rad = mean[0];
        let altitude_m = mean[2];
        let (meridian_radius, transverse_radius, _) =
            earth::principal_radii(&latitude_rad.to_degrees(), &altitude_m);
        Vector3::new(
            cov[(0, 0)].sqrt() * (meridian_radius + altitude_m),
            cov[(1, 1)].sqrt() * (transverse_radius + altitude_m) * latitude_rad.cos(),
            cov[(2, 2)].sqrt(),
        )
    }

    /// How many standard deviations of its own posterior the filter's final error may be.
    ///
    /// Three: the ordinary Gaussian consistency threshold. An estimate further than this
    /// from truth is inconsistent with the uncertainty the filter is reporting, which is the
    /// defect worth failing on -- not any particular metre count.
    const CONSISTENCY_SIGMAS: f64 = 3.0;

    /// Factor by which the reported altitude sigma may differ from the derived one.
    ///
    /// The two agree to 1.6% across twelve runs, so 1.25 is loose by an order of magnitude
    /// against the observed scatter. It is deliberately not tightened to that scatter: the
    /// assertion exists to catch a *regime* change -- a collapsed cloud, or a posterior
    /// inflated until the consistency bound admits anything -- not to pin a fourth decimal.
    /// The #295 cloud collapse missed it by a factor of 1e10.
    const ALTITUDE_POSTERIOR_TOLERANCE: f64 = 1.25;

    /// Assert a scenario run is consistent with the filter's own posterior, and that the
    /// posterior is itself the one the configuration implies.
    ///
    /// Two assertions, because either alone is defeatable:
    ///
    /// 1. **The estimate lies within [`CONSISTENCY_SIGMAS`] of the reported posterior.**
    ///    This is the textbook consistency test and needs no model of the filter -- the
    ///    reported covariance already carries the position/linear coupling that makes a
    ///    closed form intractable here. On its own, though, a filter that inflated its
    ///    covariance would pass it trivially.
    /// 2. **The reported altitude sigma matches [`steady_state_altitude_std`].** An
    ///    independent, configuration-derived anchor on the one channel where the scalar
    ///    model is exact. This is the assertion that fails hardest on #295: with the cloud
    ///    collapsed the filter reported ~1e-10 m against a derived 0.894 m.
    ///
    /// Together they say the filter's uncertainty is the one its tuning implies *and* its
    /// estimate is consistent with that uncertainty, which is what the fitted metre counts
    /// these replaced were reaching for.
    fn assert_solution_consistent_with_posterior(
        mean: &DVector<f64>,
        cov: &DMatrix<f64>,
        truth: &StrapdownState,
        gps: &GPSPositionMeasurement,
        sample_rate_hz: usize,
        max_velocity_error_mps: f64,
    ) {
        let dt = 1.0 / sample_rate_hz as f64;
        let reported = posterior_position_std_m(mean, cov);
        let derived_altitude_std =
            steady_state_altitude_std(&scenario_config(), dt, gps.vertical_noise_std);

        let ratio = reported[2] / derived_altitude_std;
        assert!(
            (1.0 / ALTITUDE_POSTERIOR_TOLERANCE..=ALTITUDE_POSTERIOR_TOLERANCE).contains(&ratio),
            "reported altitude sigma {:.4e} m is {ratio:.3}x the {derived_altitude_std:.4} m the \
             configuration implies. Outside {ALTITUDE_POSTERIOR_TOLERANCE}x the vertical channel \
             is in a different regime from the one the bound below assumes -- a collapsed cloud \
             reads far below (this was ~1e-10 m under #295), an inflated one far above",
            reported[2]
        );

        // The great-circle distance combines the two horizontal axes, so its bound is three
        // sigma of their quadrature sum.
        let max_horizontal_error_m = CONSISTENCY_SIGMAS * reported[0].hypot(reported[1]);
        let max_altitude_error_m = CONSISTENCY_SIGMAS * reported[2];
        assert_solution_close_to_truth(
            mean,
            truth,
            max_horizontal_error_m,
            max_altitude_error_m,
            max_velocity_error_mps,
        );
    }

    /// A measurement that reads a bias state off the end of the full state vector, the way
    /// the `geonav` map models do. Nothing in `strapdown-core` carries an extra state, so
    /// without this the [`RbpfConfig::extra_state_dim`] path has no exercise here.
    ///
    /// `bias_from_end` counts back from the end of whatever vector it is handed, exactly as
    /// `GravityMeasurement::bias_from_end` does -- `geonav` depends on this crate, not the
    /// other way round, so the contract is mirrored here rather than imported. That
    /// counting-from-the-end is why the state vector's *width* matters and not just its
    /// leading nine entries.
    #[derive(Debug)]
    struct BiasedAltitudeMeasurement {
        observed: f64,
        noise_std: f64,
        bias_from_end: Option<usize>,
        /// Widths this model has been handed, for asserting what a caller passed it.
        seen_state_len: std::cell::RefCell<Vec<usize>>,
    }

    impl BiasedAltitudeMeasurement {
        fn new(observed: f64, bias_from_end: Option<usize>) -> Self {
            Self {
                observed,
                noise_std: 1.0,
                bias_from_end,
                seen_state_len: std::cell::RefCell::new(Vec::new()),
            }
        }

        /// The bias entry, or `None` when the vector is too narrow to carry one.
        ///
        /// The index must land *after* the nine navigation states; a resolved index inside
        /// them means the caller passed a vector with no bias in it, which is the failure
        /// this guards (see `geonav`'s `resolve_bias_index`).
        fn bias(&self, state: &DVector<f64>) -> Option<f64> {
            let offset = self.bias_from_end?;
            let index = state.len().checked_sub(offset)?;
            (offset > 0 && index >= 9).then(|| state[index])
        }
    }

    impl MeasurementModel for BiasedAltitudeMeasurement {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn get_dimension(&self) -> usize {
            1
        }
        fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
            Ok(DVector::from_vec(vec![self.observed]))
        }
        fn get_noise(&self) -> DMatrix<f64> {
            DMatrix::from_element(1, 1, self.noise_std.powi(2))
        }
        fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
            self.seen_state_len.borrow_mut().push(state.len());
            DVector::from_vec(vec![state[2] + self.bias(state).unwrap_or(0.0)])
        }
        fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
            let mut h = DMatrix::<f64>::zeros(1, state.len());
            h[(0, 2)] = 1.0;
            if let Some(offset) = self.bias_from_end {
                let index = state.len().checked_sub(offset).filter(|i| *i >= 9).ok_or(
                    StrapdownError::DimensionMismatch {
                        what: "geophysical bias state",
                        expected: 9 + offset,
                        got: state.len(),
                    },
                )?;
                h[(0, index)] = 1.0;
            }
            Ok(h)
        }
    }

    fn rbpf_with_extra_states(extra_state_dim: usize, seed: u64) -> RaoBlackwellizedParticleFilter {
        RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 256,
                extra_state_dim,
                extra_state_init_std: 0.5,
                seed,
                // Off: the vertical-velocity pseudo-measurement is a linear update and would
                // only add noise to what these tests are watching.
                zero_vertical_velocity: false,
                ..RbpfConfig::default()
            },
        )
        .unwrap()
    }

    /// The weighted mean of a particle's extra states must end up on their nominal (#333).
    ///
    /// `recenter_after_update` promises a zero-mean cloud; it used to deliver that for six of
    /// `6 + extra_state_dim` linear states, skipping exactly the geophysical bias states --
    /// the ones most likely to carry a persistent non-zero mean, since absorbing a map bias
    /// is what they are for. The fix is not to subtract the mean and lose it but to give the
    /// extra states a nominal of their own, so this asserts both halves: the cloud comes out
    /// zero-mean, *and* every particle's absolute bias -- the quantity a measurement model
    /// reads -- is untouched.
    #[test]
    fn rbpf_recentring_moves_the_extra_state_mean_onto_its_own_nominal() {
        let mut rbpf = rbpf_with_extra_states(2, 7);

        // A large common offset on top of the initial spread: this is the map bias.
        for particle in &mut rbpf.particles {
            particle.linear_state[LINEAR_STATE_DIM_BASE] += 25.0;
            particle.linear_state[LINEAR_STATE_DIM_BASE + 1] -= 4.0;
        }
        let estimate_before = rbpf.extra_state_estimate();
        let full_before: Vec<DVector<f64>> = rbpf
            .particles
            .iter()
            .map(|p| rbpf.particle_state_vector_full(p))
            .collect();
        assert_eq!(full_before[0].len(), 11, "9 nav states plus two extras");
        assert_approx_eq!(rbpf.nominal_extra_state()[0], 0.0, 1e-12);

        rbpf.recenter_errors().unwrap();

        for i in 0..2 {
            let cloud_mean: f64 = rbpf
                .particles
                .iter()
                .map(|p| p.weight * p.linear_state[LINEAR_STATE_DIM_BASE + i])
                .sum();
            assert_approx_eq!(cloud_mean, 0.0, 1e-12);
            // The mean went somewhere rather than being dropped.
            assert_approx_eq!(rbpf.nominal_extra_state()[i], estimate_before[i], 1e-12);
            assert_approx_eq!(rbpf.extra_state_estimate()[i], estimate_before[i], 1e-12);
        }
        assert!(
            rbpf.nominal_extra_state()[0] > 20.0 && rbpf.nominal_extra_state()[1] < -1.0,
            "the nominal should have absorbed the +25 / -4 offsets, got {:?}",
            rbpf.nominal_extra_state()
        );

        // What a measurement model sees is the sum of the two, so it must not have moved.
        for (particle, before) in rbpf.particles.iter().zip(&full_before) {
            let after = rbpf.particle_state_vector_full(particle);
            for i in 9..11 {
                assert_approx_eq!(after[i], before[i], 1e-12);
            }
        }
    }

    /// The same, reached through the public update path rather than by calling the private
    /// recentring directly -- the geophysical aiding in `strapdown-sim` gets here via
    /// `update_weights_generic`, and this is the configuration that made #333 reachable.
    #[test]
    fn rbpf_extra_states_are_recentred_through_a_measurement_update() {
        let mut rbpf = rbpf_with_extra_states(1, 11);
        for particle in &mut rbpf.particles {
            particle.linear_state[LINEAR_STATE_DIM_BASE] += 12.0;
        }

        // Consistent with the cloud: nominal altitude plus the bias the states carry.
        let measurement = BiasedAltitudeMeasurement::new(100.0 + 12.0, Some(1));
        assert!(rbpf.update(&measurement).unwrap().accepted);

        let cloud_mean: f64 = rbpf
            .particles
            .iter()
            .map(|p| p.weight * p.linear_state[LINEAR_STATE_DIM_BASE])
            .sum();
        // Not exactly zero: resampling runs after recentring and redraws the cloud, which
        // reintroduces a sampling mean of order `extra_state_init_std / sqrt(N)` ~ 0.03.
        // The bound is generous against that and still two orders below the 12.0 the
        // un-recentred cloud used to carry.
        assert!(
            cloud_mean.abs() < 0.25,
            "the extra-state cloud should be left ~zero-mean, got {cloud_mean}"
        );
        assert!(
            (rbpf.extra_state_estimate()[0] - 12.0).abs() < 1.0,
            "the bias estimate should survive recentring near its 12.0 truth, got {}",
            rbpf.extra_state_estimate()[0]
        );
    }

    /// With recentring off, the extra nominal stays at zero and the cloud keeps the bias.
    ///
    /// Both halves are the same estimate; this pins the other end of the split so a future
    /// change cannot start writing the nominal when the config says not to.
    #[test]
    fn rbpf_extra_state_nominal_is_untouched_when_recentring_is_off() {
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState::default(),
            RbpfConfig {
                num_particles: 64,
                extra_state_dim: 1,
                extra_state_init_std: 0.5,
                recenter_after_update: false,
                zero_vertical_velocity: false,
                seed: 3,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        for particle in &mut rbpf.particles {
            particle.linear_state[LINEAR_STATE_DIM_BASE] += 6.0;
        }

        let measurement = BiasedAltitudeMeasurement::new(6.0, Some(1));
        rbpf.update(&measurement).unwrap();

        assert_approx_eq!(rbpf.nominal_extra_state()[0], 0.0, 1e-12);
        assert!(
            (rbpf.extra_state_estimate()[0] - 6.0).abs() < 1.0,
            "the cloud should still carry the whole bias, got {}",
            rbpf.extra_state_estimate()[0]
        );
    }

    /// The innovation gate must score a measurement against the states it declares.
    ///
    /// `evaluate_ensemble_gate` summarised the cloud with `estimate()`, which returns the
    /// nine navigation states and nothing else. A geophysical model reads its map bias by
    /// index from the *end* of the vector, so in a 9-vector `bias_from_end: Some(1)`
    /// resolved to `state[8]` -- the yaw angle -- and every geophysical fix was gated on an
    /// attitude angle standing in for the bias. The weight update was unaffected, since it
    /// goes through `particle_state_vector_full`, so the two halves of one update disagreed
    /// about what the state vector meant, and the NIS the health monitor reads came from
    /// the wrong one.
    ///
    /// The setup separates the two readings by construction: yaw is 0.9 rad and the bias is
    /// 12.0, and the measurement is consistent with the bias. A gate reading the bias sees
    /// an innovation near zero; a gate reading yaw sees one of about 11.1.
    #[test]
    fn rbpf_gate_scores_the_extra_state_and_not_the_yaw_angle() {
        let mut rbpf = RaoBlackwellizedParticleFilter::new(
            StrapdownState {
                latitude: 0.7,
                longitude: -1.3,
                altitude: 100.0,
                attitude: Rotation3::from_euler_angles(0.0, 0.0, 0.9),
                is_enu: false,
                ..StrapdownState::default()
            },
            RbpfConfig {
                num_particles: 128,
                extra_state_dim: 1,
                extra_state_init_std: 0.0,
                // A tight cloud, so the innovation below is the bias and not the spread.
                position_init_std_m: Vector3::new(0.01, 0.01, 0.01),
                attitude_init_std_rad: 0.0,
                velocity_init_std_mps: 0.0,
                zero_vertical_velocity: false,
                seed: 5,
                ..RbpfConfig::default()
            },
        )
        .unwrap();
        for particle in &mut rbpf.particles {
            particle.linear_state[LINEAR_STATE_DIM_BASE] = 12.0;
        }
        assert_approx_eq!(rbpf.estimate().0[8], 0.9, 1e-9);

        let measurement = BiasedAltitudeMeasurement::new(100.0 + 12.0, Some(1));
        let outcome = rbpf.evaluate_ensemble_gate(&measurement).unwrap();

        assert_eq!(
            rbpf.estimate_with_extra_states().0.len(),
            10,
            "the gate's summary must carry the extra state"
        );
        assert_eq!(
            measurement.seen_state_len.borrow().as_slice(),
            &[10],
            "the gate handed the model a {:?}-wide state; a 9-wide one has no bias to read \
             and resolves `bias_from_end: Some(1)` to the yaw angle",
            measurement.seen_state_len.borrow()
        );
        // Reading yaw instead of the bias would put the innovation at 100.9 - 112 = -11.1
        // and the NIS at ~123 against a noise variance of 1.
        assert!(
            outcome.nis < 1e-6,
            "a measurement consistent with the bias should gate at ~zero NIS, got {}",
            outcome.nis
        );
        assert!(outcome.accepted);
    }

    /// The augmented summary must agree with the plain one on the states they share, and
    /// must be the plain one exactly when there are no extra states -- which is every
    /// configuration but geophysical aiding, so this is the path that must not move.
    #[test]
    fn rbpf_augmented_estimate_matches_the_nine_state_estimate() {
        let mut rbpf = rbpf_with_extra_states(2, 19);
        for particle in &mut rbpf.particles {
            particle.linear_state[LINEAR_STATE_DIM_BASE] += 3.0;
        }
        let (mean, cov) = rbpf.estimate();
        let (full_mean, full_cov) = rbpf.estimate_with_extra_states();
        assert_eq!(full_mean.len(), 11);
        for i in 0..9 {
            assert_approx_eq!(full_mean[i], mean[i], 1e-12);
            for j in 0..9 {
                assert_approx_eq!(full_cov[(i, j)], cov[(i, j)], 1e-12);
            }
        }
        assert_approx_eq!(full_mean[9], rbpf.extra_state_estimate()[0], 1e-12);
        assert_approx_eq!(full_mean[10], rbpf.extra_state_estimate()[1], 1e-12);

        let plain = rbpf_with_extra_states(0, 19);
        let (a, _) = plain.estimate();
        let (b, _) = plain.estimate_with_extra_states();
        assert_eq!(a, b);
    }

    /// An ordinary GNSS fix must survive a filter that carries extra states.
    ///
    /// The gate summarises the cloud at the augmented width, but the navigation models
    /// return a fixed nine-column Jacobian -- `gps_position_jacobian` is 3x9 -- so
    /// `h * covariance` was a 3x9 against an 11x11 and nalgebra panicked before the update
    /// ran. GNSS fixes and geophysical fixes ride the same event stream, so this is every
    /// geophysically-aided run rather than a corner case.
    #[test]
    fn rbpf_gps_update_survives_a_filter_carrying_extra_states() {
        let mut rbpf = rbpf_with_extra_states(2, 23);
        let gps = GPSPositionMeasurement {
            latitude: 0.7_f64.to_degrees(),
            longitude: (-1.3_f64).to_degrees(),
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
        };
        let outcome = rbpf.update(&gps).unwrap();
        assert!(
            outcome.accepted,
            "a fix at the nominal position should pass the gate, NIS was {}",
            outcome.nis
        );
        // The barometer is the other fixed-width model on this path.
        let baro = RelativeAltitudeMeasurement {
            relative_altitude: 0.0,
            reference_altitude: 100.0,
        };
        assert!(rbpf.update(&baro).unwrap().accepted);
    }

    fn run_rbpf_on_scenario(
        nominal: StrapdownState,
        imu_data: &[IMUData],
        gps_measurements: &[GPSPositionMeasurement],
        sample_rate_hz: usize,
    ) -> (DVector<f64>, DMatrix<f64>) {
        assert_eq!(imu_data.len(), gps_measurements.len());
        let dt = 1.0 / sample_rate_hz as f64;

        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, scenario_config()).unwrap();

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

    /// The vertical channel must be aided, not merely present (#295).
    ///
    /// The truth here is exactly stationary -- altitude 1000.0000 m and all three
    /// velocities identically zero at every step -- and the filter is handed a noiseless
    /// 1000 m altitude fix 3000 times, so anything but a centimetre-scale answer means the
    /// altitude channel is not receiving those fixes at all. For most of this test's life
    /// it was not, and the run ended tens of metres away.
    ///
    /// # What was wrong
    ///
    /// `generate_scenario_data` built its fixes with `horizontal_noise_std: 5.0 *
    /// METERS_TO_DEGREES`, but that field is metres and `GPSPositionMeasurement::get_noise`
    /// converts it itself. The squared conversion left a horizontal sigma of 7.1e-12 rad --
    /// 45 micrometres.
    ///
    /// A Kalman filter survives that: R is diagonal, so an absurd horizontal entry leaves
    /// the vertical gain alone. A particle weight does not, because it is a single scalar
    /// over all three channels. At 45 um the horizontal term dominated the likelihood
    /// completely, the cloud resampled onto whichever particle fit horizontally without
    /// regard to its altitude, and after the first update every particle was a copy of one
    /// survivor: the reported altitude variance was ~1e-20 m^2 and the weighted mean
    /// altitude error was identically zero, so `recenter_errors` never moved the nominal
    /// altitude either. The vertical channel then ran open-loop for 600 s, seeded with that
    /// one survivor's arbitrary N(0, 5 m) initial altitude error.
    ///
    /// That is the whole of #295, including the part that made it look like tuning. An
    /// unaided INS vertical channel is exponentially unstable, so the 600 s endpoint was a
    /// draw rather than a convergent value, and any perturbation resampled it -- which is
    /// why a 0.4% change to the ellipsoid radii in #292 moved it by 16 m, and why four
    /// successive mechanization corrections moved it around without closing the gap.
    ///
    /// # The measurement
    ///
    /// #331 established the sharpest form of the symptom before the cause was known: a 5x
    /// increase in the vertical process noise bought ~2.4x the altitude error, so the error
    /// went as `sqrt(Q)` -- the channel was dominated by its own process noise rather than
    /// held by the 5 Hz fixes it was being given, and a healthy channel would barely notice
    /// Q. Re-running that sweep across the same four seeds, before and after the fix units
    /// are corrected, final altitude error in metres:
    ///
    ///     seed        stationary        v north          v east
    ///                 pre     post      pre     post     pre     post
    ///     123        72.67   0.0122    53.50   0.0045   24.73   0.0007
    ///     7          17.59   0.0014    11.32   0.0188   49.83   0.0046
    ///     20260915   13.91   0.0073     9.62   0.0016   21.75   0.0045
    ///     991        28.23   0.0136    10.66   0.0163    5.96   0.0037
    ///
    /// Three orders of magnitude, and the seed scatter is gone with it: the endpoint is no
    /// longer a draw from a wide distribution but the same centimetre answer every time,
    /// which is what "converged" means and what no amount of mechanization work had
    /// produced. The cloud's altitude spread goes from ~1e-10 m to the 0.888-0.909 m its
    /// own noise model calls for.
    ///
    /// The two moving columns also carry the fix/truth timestamp alignment described on
    /// `generate_scenario_data`: its fixes used to be built before the propagation and were
    /// therefore one sample stale, a 2 m along-track pull at 10 m/s. The stationary column
    /// is bit-identical with and without that second correction, which is the check that it
    /// does what it claims -- a stationary platform has no along-track lag to remove.
    ///
    /// The horizontal error moving the *other* way -- 0.003 m before, 0.016-0.051 m after --
    /// is the same defect seen from the front: 3 mm against a nominally 5 m fix was the
    /// filter reporting how tight it had actually been told the fix was.
    ///
    /// # Why the EKF/UKF/ESKF were never affected
    ///
    /// They never consumed this generator: `generate_scenario_data`'s fixes reach only the
    /// RBPF tests in this module, and the Kalman filters are compared on
    /// `core/tests/filter_comparison.rs`, whose builder already passes metres and says why.
    /// All four filters clear that suite's 3 m altitude bound on a shared scenario. The
    /// coupling that caused this is structural to the particle weight and has no analogue
    /// in a Kalman update, so there is no corresponding compensation to remove from the
    /// vertical channel they share.
    ///
    /// # The bound
    ///
    /// [`assert_solution_consistent_with_posterior`], not a metre count: the estimate must
    /// lie within three sigma of the covariance the filter itself reports, and that
    /// covariance must be the one the configuration implies. The 15 m this replaces was
    /// fitted -- 1.2x margin over a 12.49 m baseline that was itself the product of
    /// cancelling bugs -- and is what #295 and #288 asked to be rid of. Both halves catch
    /// the defect decisively: the collapsed cloud reported ~1e-10 m against a derived
    /// 0.894 m, and its 13.98 m error was 22 sigma of even the healthy posterior.
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

        let (mean, cov) =
            run_rbpf_on_scenario(nominal, &imu_data, &gps_measurements, sample_rate_hz);
        let truth = true_states.last().unwrap();

        // The velocity bound stays a literal. The fixes here are position-only, so nothing
        // aids the horizontal velocity states directly and there is no posterior of theirs
        // in the 9-state covariance worth bounding against; 0.5 m/s is an anti-divergence
        // guard on a channel that measures 0.004-0.017 m/s across the twelve runs above.
        assert_solution_consistent_with_posterior(
            &mean,
            &cov,
            truth,
            &gps_measurements[0],
            sample_rate_hz,
            0.5,
        );
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

        let (mean, cov) =
            run_rbpf_on_scenario(nominal, &imu_data, &gps_measurements, sample_rate_hz);
        let truth = true_states.last().unwrap();

        // Expect northward motion; RBPF estimate should reflect it.
        assert!(mean[0] > initial_state.latitude);
        // Tightened from the 50 m / 150 m guards that stood here until #295. They were
        // loose because the vertical channel was unaided and its 600 s endpoint was a draw
        // rather than a converged value -- the four-seed table this comment used to carry
        // measured 9.62 m to 53.50 m on this scenario alone, and said not to tighten until
        // #295 was understood. It was the fix units, not the mechanization: the same four
        // seeds now give 0.0016 m to 0.0188 m here. See
        // `rbpf_runs_on_scenario_stationary` for the full sweep and the root cause.
        assert_solution_consistent_with_posterior(
            &mean,
            &cov,
            truth,
            &gps_measurements[0],
            sample_rate_hz,
            1.0,
        );
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

        let (mean, cov) =
            run_rbpf_on_scenario(nominal, &imu_data, &gps_measurements, sample_rate_hz);
        let truth = true_states.last().unwrap();

        // Expect eastward motion; RBPF estimate should reflect it.
        assert!(mean[1] > initial_state.longitude);
        // Bounded against the reported posterior, as in the two scenarios above, in place of
        // the 50 m / 150 m guards #295 removed the reason for. Across the four seeds swept
        // in `rbpf_runs_on_scenario_stationary` this scenario went from 5.96-49.83 m of
        // altitude error to 0.0007-0.0046 m.
        assert_solution_consistent_with_posterior(
            &mean,
            &cov,
            truth,
            &gps_measurements[0],
            sample_rate_hz,
            1.0,
        );
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
