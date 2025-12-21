//! Particle filter implementation for nonlinear Bayesian state estimation.
//!
//! This module provides a particle filter (also known as Sequential Monte Carlo) implementation
//! for strapdown inertial navigation. Unlike Kalman filters which assume Gaussian distributions,
//! particle filters can represent arbitrary probability distributions through weighted samples.
//!
//! # Key Components
//!
//! - [`Particle`]: Individual state hypothesis with navigation state, biases, and weight
//! - [`ParticleAveragingStrategy`]: Methods for extracting state estimates from particles
//! - [`ParticleResamplingStrategy`]: Algorithms for combating particle degeneracy
//!
//! # 2.5D Navigation
//!
//! This module implements a 2.5D navigation approach where the vertical channel (altitude and
//! vertical velocity) is treated with simplified dynamics compared to full 6-DOF strapdown.
//! The key modification is in the [`forward_2_5d`] function, which propagates horizontal states
//! with full strapdown mechanization while treating vertical velocity as a random walk.

use crate::earth;
use crate::kalman::{InitialState, NavigationFilter};
use crate::linalg::{robust_spd_solve, symmetrize};
use crate::measurements::MeasurementModel;
use crate::{wrap_to_2pi, IMUData, StrapdownState, attitude_update, position_update};

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector2, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::fmt::{self, Debug, Display};

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
}

impl Particle {
    /// Convert particle to state vector representation.
    ///
    /// Returns a DVector containing all states in order:
    /// [lat, lon, alt, v_n, v_e, v_v, roll, pitch, yaw, b_a, b_g, other_states...]
    pub fn to_state_vector(&self) -> DVector<f64> {
        let (roll, pitch, yaw) = self.nav_state.attitude.euler_angles();
        let mut state = vec![
            self.nav_state.latitude,
            self.nav_state.longitude,
            self.nav_state.altitude,
            self.nav_state.velocity_north,
            self.nav_state.velocity_east,
            self.nav_state.velocity_vertical,
            roll,
            pitch,
            yaw,
        ];

        // Add biases
        state.extend(self.accel_bias.iter());
        state.extend(self.gyro_bias.iter());

        // Add optional states
        if let Some(ref other) = self.other_states {
            state.extend(other.iter());
        }

        DVector::from_vec(state)
    }

    /// Create particle from state vector.
    ///
    /// # Arguments
    /// * `state_vector` - State vector [lat, lon, alt, v_n, v_e, v_v, φ, θ, ψ, b_a, b_g, ...]
    /// * `is_enu` - Coordinate frame flag
    pub fn from_state_vector(state_vector: &DVector<f64>, is_enu: bool) -> Self {
        let nav_state = StrapdownState {
            latitude: state_vector[0],
            longitude: state_vector[1],
            altitude: state_vector[2],
            velocity_north: state_vector[3],
            velocity_east: state_vector[4],
            velocity_vertical: state_vector[5],
            attitude: Rotation3::from_euler_angles(state_vector[6], state_vector[7], state_vector[8]),
            is_enu,
        };

        let accel_bias = if state_vector.len() >= 15 {
            DVector::from_vec(vec![state_vector[9], state_vector[10], state_vector[11]])
        } else {
            DVector::zeros(3)
        };

        let gyro_bias = if state_vector.len() >= 15 {
            DVector::from_vec(vec![state_vector[12], state_vector[13], state_vector[14]])
        } else {
            DVector::zeros(3)
        };

        let other_states = if state_vector.len() > 15 {
            Some(DVector::from_vec(state_vector.rows(15, state_vector.len() - 15).iter().copied().collect()))
        } else {
            None
        };

        Particle {
            nav_state,
            accel_bias,
            gyro_bias,
            other_states,
            state_size: state_vector.len(),
            weight: 1.0,
        }
    }
}

// ============================================================================
// Particle Filter Configuration Structs and Enums
// ============================================================================

/// Process noise parameters for particle filter propagation.
///
/// Defines the standard deviations of process noise for each state component.
/// Process noise represents uncertainty in the system dynamics and is injected
/// during the predict step to maintain particle diversity.
///
/// For 2.5D navigation, vertical components (altitude, vertical velocity) should
/// have significantly larger noise than horizontal components to allow particles
/// to explore vertical uncertainty.
#[derive(Clone, Debug)]
pub struct ProcessNoise {
    /// Position noise standard deviations [σ_lat, σ_lon, σ_h] (rad, rad, m)
    pub position_std: Vector3<f64>,

    /// Velocity noise standard deviations [σ_vn, σ_ve, σ_vv] (m/s)
    pub velocity_std: Vector3<f64>,

    /// Attitude noise standard deviations [σ_φ, σ_θ, σ_ψ] (rad)
    pub attitude_std: Vector3<f64>,

    /// Accelerometer bias noise standard deviations [σ_ba_x, σ_ba_y, σ_ba_z] (m/s²)
    pub accel_bias_std: Vector3<f64>,

    /// Gyroscope bias noise standard deviations [σ_bg_x, σ_bg_y, σ_bg_z] (rad/s)
    pub gyro_bias_std: Vector3<f64>,

    /// Damping states noise (for third-order vertical channel, if used)
    pub damping_states_std: Option<Vector2<f64>>,
}

impl Default for ProcessNoise {
    /// Default process noise for 2.5D navigation with smartphone-grade IMU.
    ///
    /// Key characteristics:
    /// - Tight horizontal position/velocity noise
    /// - **Large vertical position/velocity noise** (10-100× horizontal) for 2.5D
    /// - Moderate attitude noise
    /// - Small bias drift
    fn default() -> Self {
        ProcessNoise {
            // Position: tight horizontal, loose vertical
            position_std: Vector3::new(1e-3, 1e-3, 5e-2), // [rad, rad, m]

            // Velocity: moderate horizontal, LARGE vertical (2.5D key)
            velocity_std: Vector3::new(1e-2, 1e-2, 1e-1), // [m/s, m/s, m/s]

            // Attitude: moderate uncertainty
            attitude_std: Vector3::new(1e-3, 1e-3, 1e-3), // [rad]

            // Biases: small random walk
            accel_bias_std: Vector3::new(1e-3, 1e-3, 1e-3), // [m/s²]
            gyro_bias_std: Vector3::new(1e-4, 1e-4, 1e-4),  // [rad/s]

            // No damping states by default
            damping_states_std: None,
        }
    }
}

// ============================================================================
// Rao-Blackwellized Particle Filter Structs
// ============================================================================

/// Process noise parameters for Rao-Blackwellized Particle Filter.
///
/// Partitions process noise into two components:
/// - Position noise (for particle diffusion)
/// - Linear state noise (for per-particle UKF)
#[derive(Clone, Debug)]
pub struct RBProcessNoise {
    /// Position noise standard deviations [σ_lat, σ_lon, σ_h] (rad, rad, m)
    pub position_std: Vector3<f64>,

    /// Linear state noise covariance matrix (12×12) for per-particle UKF
    /// Includes: velocity (3), attitude (3), accel biases (3), gyro biases (3)
    pub linear_states_covariance: DMatrix<f64>,
}

impl Default for RBProcessNoise {
    /// Default process noise for RBPF with 2.5D navigation.
    fn default() -> Self {
        // Position noise (for particle diffusion)
        let position_std = Vector3::new(1e-3, 1e-3, 5e-2);

        // Linear state noise (for UKF predict)
        let linear_noise_diag = vec![
            1e-2, 1e-2, 1e-1, // velocity (2.5D: large v_v)
            1e-3, 1e-3, 1e-3, // attitude
            1e-3, 1e-3, 1e-3, // accel biases
            1e-4, 1e-4, 1e-4, // gyro biases
        ];
        let linear_states_covariance =
            DMatrix::from_diagonal(&DVector::from_vec(linear_noise_diag));

        RBProcessNoise {
            position_std,
            linear_states_covariance,
        }
    }
}

/// Per-particle Extended Kalman Filter for linear states.
///
/// Implements EKF for the 12-state linear/conditionally-Gaussian subspace:
/// velocity (3), attitude (3), accelerometer biases (3), gyroscope biases (3).
/// 
/// The EKF uses linearization (Jacobians) instead of sigma points, making it
/// computationally more efficient than the UKF at the cost of some accuracy
/// for highly nonlinear systems.
#[derive(Clone, Debug)]
pub struct PerParticleEKF {
    /// Mean state: [v_n, v_e, v_v, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    pub mean_state: DVector<f64>,

    /// Covariance matrix (12×12)
    pub covariance: DMatrix<f64>,
}

/// Per-particle Unscented Kalman Filter for linear states.
///
/// Implements UKF for the 12-state linear/conditionally-Gaussian subspace:
/// velocity (3), attitude (3), accelerometer biases (3), gyroscope biases (3).
#[derive(Clone, Debug)]
pub struct PerParticleUKF {
    /// Mean state: [v_n, v_e, v_v, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    pub mean_state: DVector<f64>,

    /// Covariance matrix (12×12)
    pub covariance: DMatrix<f64>,

    /// UKF parameters
    alpha: f64,
    #[allow(dead_code)]
    beta: f64,
    kappa: f64,

    /// Cached UKF weights (computed once in constructor)
    weights_mean: DVector<f64>,
    weights_cov: DVector<f64>,
}

impl PerParticleEKF {
    /// Create a new per-particle EKF.
    ///
    /// # Arguments
    /// * `initial_state` - Initial 12-state vector
    /// * `initial_cov` - Initial 12×12 covariance matrix
    pub fn new(
        initial_state: DVector<f64>,
        initial_cov: DMatrix<f64>,
    ) -> Self {
        PerParticleEKF {
            mean_state: initial_state,
            covariance: initial_cov,
        }
    }

    /// Get current state estimate.
    pub fn get_estimate(&self) -> DVector<f64> {
        self.mean_state.clone()
    }

    /// Get current covariance.
    pub fn get_covariance(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }

    /// EKF predict step conditioned on particle position.
    ///
    /// Propagates the 12-state linear system: velocity, attitude, biases.
    /// Position-dependent terms (gravity, Earth rate, transport rate) are
    /// evaluated at the given particle position.
    ///
    /// Uses linearization (Jacobian computation) instead of sigma points.
    ///
    /// # Arguments
    /// * `position` - Particle position [lat, lon, alt]
    /// * `imu_data` - IMU measurements (bias-corrected)
    /// * `process_noise` - Process noise covariance (12×12)
    /// * `dt` - Time step
    /// * `is_enu` - Coordinate frame flag
    pub fn predict(
        &mut self,
        position: &Vector3<f64>,
        imu_data: IMUData,
        process_noise: &DMatrix<f64>,
        dt: f64,
        is_enu: bool,
    ) {
        // Extract current state
        let velocity = Vector3::new(self.mean_state[0], self.mean_state[1], self.mean_state[2]);
        let attitude_euler = Vector3::new(self.mean_state[3], self.mean_state[4], self.mean_state[5]);
        let accel_bias = Vector3::new(self.mean_state[6], self.mean_state[7], self.mean_state[8]);
        let gyro_bias = Vector3::new(self.mean_state[9], self.mean_state[10], self.mean_state[11]);

        // Bias-corrected IMU
        let corrected_accel = imu_data.accel - accel_bias;
        let corrected_gyro = imu_data.gyro - gyro_bias;

        // Propagate state through nonlinear dynamics
        let (roll, pitch, yaw) = (attitude_euler[0], attitude_euler[1], attitude_euler[2]);
        let c_bn = Rotation3::from_euler_angles(roll, pitch, yaw);
        let specific_force = c_bn.matrix() * corrected_accel;

        let new_velocity = propagate_velocity_2_5d(&velocity, &specific_force, position, dt, is_enu);
        let new_attitude = propagate_attitude(&attitude_euler, &corrected_gyro, position, dt);

        // Biases stay constant (random walk handled by process noise)
        let new_accel_bias = accel_bias;
        let new_gyro_bias = gyro_bias;

        // Update mean state
        self.mean_state[0] = new_velocity[0];
        self.mean_state[1] = new_velocity[1];
        self.mean_state[2] = new_velocity[2];
        self.mean_state[3] = new_attitude[0];
        self.mean_state[4] = new_attitude[1];
        self.mean_state[5] = new_attitude[2];
        self.mean_state[6] = new_accel_bias[0];
        self.mean_state[7] = new_accel_bias[1];
        self.mean_state[8] = new_accel_bias[2];
        self.mean_state[9] = new_gyro_bias[0];
        self.mean_state[10] = new_gyro_bias[1];
        self.mean_state[11] = new_gyro_bias[2];

        // Compute state transition matrix F (Jacobian of dynamics)
        let f = self.compute_state_transition_jacobian(
            position,
            &velocity,
            &attitude_euler,
            &corrected_accel,
            &corrected_gyro,
            dt,
            is_enu,
        );

        // Propagate covariance: P(k+1|k) = F * P(k|k) * F^T + Q
        let cov_predicted = &f * &self.covariance * f.transpose() + process_noise;
        self.covariance = symmetrize(&cov_predicted);

        // Add small regularization to ensure positive definiteness
        for i in 0..self.covariance.nrows() {
            self.covariance[(i, i)] = self.covariance[(i, i)].max(1e-12);
        }
    }

    /// Compute the state transition Jacobian matrix F.
    ///
    /// The state is [v_n, v_e, v_v, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz].
    /// Returns the 12×12 Jacobian ∂f/∂x evaluated at the current state.
    fn compute_state_transition_jacobian(
        &self,
        _position: &Vector3<f64>,
        _velocity: &Vector3<f64>,
        attitude_euler: &Vector3<f64>,
        corrected_accel: &Vector3<f64>,
        _corrected_gyro: &Vector3<f64>,
        dt: f64,
        _is_enu: bool,
    ) -> DMatrix<f64> {
        // For simplicity, use a first-order approximation: F ≈ I + dt * A
        // where A is the continuous-time dynamics Jacobian
        let mut f = DMatrix::identity(12, 12);

        let (roll, pitch, yaw) = (attitude_euler[0], attitude_euler[1], attitude_euler[2]);
        let c_bn = Rotation3::from_euler_angles(roll, pitch, yaw);

        // Velocity derivatives w.r.t. attitude (how rotation affects specific force)
        // ∂v/∂attitude: specific force transformation depends on attitude
        let dcbn_droll = compute_dcbn_droll(roll, pitch, yaw);
        let dcbn_dpitch = compute_dcbn_dpitch(roll, pitch, yaw);
        let dcbn_dyaw = compute_dcbn_dyaw(roll, pitch, yaw);

        // ∂(C_bn * f_b) / ∂roll
        let df_droll = dcbn_droll * corrected_accel;
        let df_dpitch = dcbn_dpitch * corrected_accel;
        let df_dyaw = dcbn_dyaw * corrected_accel;

        // Velocity w.r.t. attitude (rows 0-2, cols 3-5)
        f[(0, 3)] += dt * df_droll[0];
        f[(0, 4)] += dt * df_dpitch[0];
        f[(0, 5)] += dt * df_dyaw[0];
        f[(1, 3)] += dt * df_droll[1];
        f[(1, 4)] += dt * df_dpitch[1];
        f[(1, 5)] += dt * df_dyaw[1];
        // Vertical velocity is decoupled in 2.5D (stays at identity)

        // Velocity w.r.t. accelerometer bias (rows 0-2, cols 6-8)
        // ∂v/∂b_a = -C_bn * dt (negative because we subtract bias)
        let dvn_dba = -c_bn.matrix() * dt;
        for i in 0..3 {
            for j in 0..3 {
                f[(i, 6 + j)] += dvn_dba[(i, j)];
            }
        }

        // Attitude w.r.t. gyro bias (rows 3-5, cols 9-11)
        // ∂attitude/∂b_g ≈ -dt (simplified, actual is more complex)
        f[(3, 9)] -= dt;
        f[(4, 10)] -= dt;
        f[(5, 11)] -= dt;

        // Biases are random walk: ∂b/∂b = I (already in identity matrix)

        f
    }

    /// EKF update step, returns marginal log-likelihood for particle weighting.
    ///
    /// Performs standard EKF measurement update on the linear states and
    /// computes the marginal likelihood p(z | position) which is used to
    /// update the particle weight.
    ///
    /// # Arguments
    /// * `position` - Particle position
    /// * `measurement` - Measurement model
    ///
    /// # Returns
    /// Log-likelihood for particle weight update
    pub fn update<M: MeasurementModel + ?Sized>(
        &mut self,
        position: &Vector3<f64>,
        measurement: &M,
    ) -> f64 {
        // Construct full 15-state vector from position + linear states
        let mut full_state = DVector::zeros(15);
        full_state[0] = position[0]; // lat
        full_state[1] = position[1]; // lon
        full_state[2] = position[2]; // alt
        full_state.rows_mut(3, 12).copy_from(&self.mean_state);

        // Get predicted measurement
        let predicted_meas = measurement.get_expected_measurement(&full_state);
        let innovation = measurement.get_vector() - &predicted_meas;

        // Get measurement noise covariance
        let r = measurement.get_noise();
        let meas_dim = innovation.len();

        // Compute measurement Jacobian H (∂h/∂x)
        let h = self.compute_measurement_jacobian(position, measurement);

        // Compute innovation covariance: S = H * P * H^T + R
        let s = &h * &self.covariance * h.transpose() + r;
        let s_sym = symmetrize(&s);

        // Compute Kalman gain: K = P * H^T * S^-1
        let s_inv = robust_spd_solve(&s_sym, &DMatrix::identity(meas_dim, meas_dim));
        let kalman_gain = &self.covariance * h.transpose() * &s_inv;

        // Update state: x = x + K * innovation
        self.mean_state += &kalman_gain * &innovation;

        // Update covariance (Joseph form for numerical stability)
        // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        let i_kh = DMatrix::identity(12, 12) - &kalman_gain * &h;
        let updated_cov = &i_kh * &self.covariance * i_kh.transpose() 
            + &kalman_gain * measurement.get_noise() * kalman_gain.transpose();
        self.covariance = symmetrize(&updated_cov);

        // Add small regularization to ensure positive definiteness
        for i in 0..self.covariance.nrows() {
            self.covariance[(i, i)] = self.covariance[(i, i)].max(1e-12);
        }

        // Compute log-likelihood for particle weighting
        let det = s_sym.determinant().max(1e-10);
        let mahalanobis = (&innovation.transpose() * &s_inv * &innovation)[(0, 0)];
        let log_likelihood = -0.5
            * (mahalanobis + det.ln() + meas_dim as f64 * (2.0 * std::f64::consts::PI).ln());

        log_likelihood
    }

    /// Compute measurement Jacobian H (∂h/∂x).
    ///
    /// Uses numerical differentiation for generality.
    fn compute_measurement_jacobian<M: MeasurementModel + ?Sized>(
        &self,
        position: &Vector3<f64>,
        measurement: &M,
    ) -> DMatrix<f64> {
        let meas_dim = measurement.get_vector().len();
        let mut h = DMatrix::zeros(meas_dim, 12);

        // Perturbation size for numerical differentiation
        let eps = 1e-7;

        // Construct nominal full state
        let mut nominal_state = DVector::zeros(15);
        nominal_state[0] = position[0];
        nominal_state[1] = position[1];
        nominal_state[2] = position[2];
        nominal_state.rows_mut(3, 12).copy_from(&self.mean_state);

        let nominal_meas = measurement.get_expected_measurement(&nominal_state);

        // Perturb each linear state and compute derivative
        for i in 0..12 {
            let mut perturbed_state = nominal_state.clone();
            perturbed_state[3 + i] += eps;

            let perturbed_meas = measurement.get_expected_measurement(&perturbed_state);
            let diff = (perturbed_meas - &nominal_meas) / eps;

            h.set_column(i, &diff);
        }

        h
    }
}

/// Compute derivative of rotation matrix C_bn with respect to roll.
fn compute_dcbn_droll(roll: f64, pitch: f64, yaw: f64) -> Matrix3<f64> {
    let cr = roll.cos();
    let sr = roll.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let cy = yaw.cos();
    let sy = yaw.sin();

    Matrix3::new(
        0.0, 0.0, 0.0,
        (sr * sp * cy - cr * sy) - (-sr * sp * cy + cr * sy), (sr * sp * sy + cr * cy) - (-sr * sp * sy - cr * cy), sr * cp - (-sr * cp),
        (sr * sp * cy + cr * sy) - (cr * sy - sr * sp * cy), (sr * sp * sy - cr * cy) - (-cr * cy - sr * sp * sy), sr * cp - sr * cp,
    )
}

/// Compute derivative of rotation matrix C_bn with respect to pitch.
fn compute_dcbn_dpitch(roll: f64, pitch: f64, yaw: f64) -> Matrix3<f64> {
    let cr = roll.cos();
    let sr = roll.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let cy = yaw.cos();
    let sy = yaw.sin();

    Matrix3::new(
        -sp * cy, -sp * sy, -cp,
        cr * cp * cy - (-cr * sp * cy), cr * cp * sy - (-cr * sp * sy), -cr * sp - cr * cp,
        sr * cp * cy - (-sr * sp * cy), sr * cp * sy - (-sr * sp * sy), -sr * sp - sr * cp,
    )
}

/// Compute derivative of rotation matrix C_bn with respect to yaw.
fn compute_dcbn_dyaw(roll: f64, pitch: f64, yaw: f64) -> Matrix3<f64> {
    let cr = roll.cos();
    let sr = roll.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let cy = yaw.cos();
    let sy = yaw.sin();

    Matrix3::new(
        -cp * sy, cp * cy, 0.0,
        -(cr * sp * sy + sr * cy) + (cr * sp * cy - sr * sy), (cr * sp * cy - sr * sy) - (cr * sp * sy + sr * cy), 0.0,
        -(sr * sp * sy - cr * cy) + (sr * sp * cy + cr * sy), (sr * sp * cy + cr * sy) - (sr * sp * sy - cr * cy), 0.0,
    )
}

impl PerParticleUKF {
    /// Create a new per-particle UKF.
    ///
    /// # Arguments
    /// * `initial_state` - Initial 12-state vector
    /// * `initial_cov` - Initial 12×12 covariance matrix
    /// * `alpha` - UKF spread parameter (typically 1e-3)
    /// * `beta` - UKF distribution parameter (2.0 for Gaussian)
    /// * `kappa` - UKF secondary scaling (typically 0.0)
    pub fn new(
        initial_state: DVector<f64>,
        initial_cov: DMatrix<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> Self {
        let n = initial_state.len();
        let lambda = alpha * alpha * (n as f64 + kappa) - n as f64;

        // Compute UKF weights
        let num_sigma = 2 * n + 1;
        let mut weights_mean = DVector::zeros(num_sigma);
        let mut weights_cov = DVector::zeros(num_sigma);

        weights_mean[0] = lambda / (n as f64 + lambda);
        weights_cov[0] = lambda / (n as f64 + lambda) + (1.0 - alpha * alpha + beta);

        for i in 1..num_sigma {
            weights_mean[i] = 0.5 / (n as f64 + lambda);
            weights_cov[i] = 0.5 / (n as f64 + lambda);
        }

        PerParticleUKF {
            mean_state: initial_state,
            covariance: initial_cov,
            alpha,
            beta,
            kappa,
            weights_mean,
            weights_cov,
        }
    }

    /// Get sigma points for UKF propagation.
    ///
    /// Returns a matrix where each column is a sigma point.
    fn get_sigma_points(&self) -> DMatrix<f64> {
        let n = self.mean_state.len();
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;
        let num_sigma = 2 * n + 1;

        let mut sigma_points = DMatrix::zeros(n, num_sigma);

        // Center point
        sigma_points.set_column(0, &self.mean_state);

        // Compute matrix square root: P = L * L^T
        let cholesky = self.covariance.clone().cholesky();
        if cholesky.is_none() {
            // Fallback: use symmetric eigenvalue decomposition
            let eigen = self.covariance.clone().symmetric_eigen();
            let sqrt_p = &eigen.eigenvectors
                * DMatrix::from_diagonal(&eigen.eigenvalues.map(|x| x.max(0.0).sqrt()))
                * eigen.eigenvectors.transpose();
            let scale = ((n as f64 + lambda).sqrt()) * &sqrt_p;

            for i in 0..n {
                sigma_points.set_column(1 + i, &(&self.mean_state + scale.column(i)));
                sigma_points.set_column(1 + n + i, &(&self.mean_state - scale.column(i)));
            }
        } else {
            let l = cholesky.unwrap().l();
            let scale = (n as f64 + lambda).sqrt() * l;

            for i in 0..n {
                sigma_points.set_column(1 + i, &(&self.mean_state + scale.column(i)));
                sigma_points.set_column(1 + n + i, &(&self.mean_state - scale.column(i)));
            }
        }

        sigma_points
    }

    /// Get current state estimate.
    pub fn get_estimate(&self) -> DVector<f64> {
        self.mean_state.clone()
    }

    /// Get current covariance.
    pub fn get_covariance(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }

    /// UKF predict step conditioned on particle position.
    ///
    /// Propagates the 12-state linear system: velocity, attitude, biases.
    /// Position-dependent terms (gravity, Earth rate, transport rate) are
    /// evaluated at the given particle position.
    ///
    /// # Arguments
    /// * `position` - Particle position [lat, lon, alt]
    /// * `imu_data` - IMU measurements (bias-corrected)
    /// * `process_noise` - Process noise covariance (12×12)
    /// * `dt` - Time step
    /// * `is_enu` - Coordinate frame flag
    pub fn predict(
        &mut self,
        position: &Vector3<f64>,
        imu_data: IMUData,
        process_noise: &DMatrix<f64>,
        dt: f64,
        is_enu: bool,
    ) {
        // Get sigma points
        let sigma_points = self.get_sigma_points();

        // Propagate each sigma point
        let propagated = self.propagate_sigma_points(&sigma_points, position, imu_data, dt, is_enu);

        // Compute predicted mean
        let mut mean_predicted = DVector::zeros(12);
        for (i, col) in propagated.column_iter().enumerate() {
            mean_predicted += self.weights_mean[i] * col;
        }

        // Compute predicted covariance
        let mut cov_predicted = DMatrix::zeros(12, 12);
        for (i, col) in propagated.column_iter().enumerate() {
            let diff = col - &mean_predicted;
            cov_predicted += self.weights_cov[i] * (&diff * diff.transpose());
        }
        cov_predicted += process_noise;

        self.mean_state = mean_predicted;
        self.covariance = symmetrize(&cov_predicted);

        // Add small regularization to ensure positive definiteness
        for i in 0..self.covariance.nrows() {
            self.covariance[(i, i)] = self.covariance[(i, i)].max(1e-12);
        }
    }

    /// Propagate sigma points through the dynamics.
    fn propagate_sigma_points(
        &self,
        sigma_points: &DMatrix<f64>,
        position: &Vector3<f64>,
        imu_data: IMUData,
        dt: f64,
        is_enu: bool,
    ) -> DMatrix<f64> {
        let num_sigma = sigma_points.ncols();
        let mut propagated = DMatrix::zeros(12, num_sigma);

        for (i, sigma_col) in sigma_points.column_iter().enumerate() {
            // Extract states from sigma point
            let velocity = Vector3::new(sigma_col[0], sigma_col[1], sigma_col[2]);
            let attitude_euler = Vector3::new(sigma_col[3], sigma_col[4], sigma_col[5]);
            let accel_bias = Vector3::new(sigma_col[6], sigma_col[7], sigma_col[8]);
            let gyro_bias = Vector3::new(sigma_col[9], sigma_col[10], sigma_col[11]);

            // Bias-corrected IMU
            let corrected_accel = imu_data.accel - accel_bias;
            let corrected_gyro = imu_data.gyro - gyro_bias;

            // Propagate velocity (horizontal components, vertical simplified)
            let (roll, pitch, yaw) = (attitude_euler[0], attitude_euler[1], attitude_euler[2]);
            let c_bn = Rotation3::from_euler_angles(roll, pitch, yaw);
            let specific_force = c_bn.matrix() * corrected_accel;

            // Compute velocity update (2.5D: horizontal full, vertical simple)
            let new_velocity = propagate_velocity_2_5d(&velocity, &specific_force, position, dt, is_enu);

            // Propagate attitude
            let new_attitude = propagate_attitude(&attitude_euler, &corrected_gyro, position, dt);

            // Biases: random walk (stay constant in predict, noise added separately)
            let new_accel_bias = accel_bias;
            let new_gyro_bias = gyro_bias;

            // Store propagated sigma point
            propagated[(0, i)] = new_velocity[0];
            propagated[(1, i)] = new_velocity[1];
            propagated[(2, i)] = new_velocity[2];
            propagated[(3, i)] = new_attitude[0];
            propagated[(4, i)] = new_attitude[1];
            propagated[(5, i)] = new_attitude[2];
            propagated[(6, i)] = new_accel_bias[0];
            propagated[(7, i)] = new_accel_bias[1];
            propagated[(8, i)] = new_accel_bias[2];
            propagated[(9, i)] = new_gyro_bias[0];
            propagated[(10, i)] = new_gyro_bias[1];
            propagated[(11, i)] = new_gyro_bias[2];
        }

        propagated
    }

    /// UKF update step, returns marginal log-likelihood for particle weighting.
    ///
    /// Performs standard UKF measurement update on the linear states and
    /// computes the marginal likelihood p(z | position) which is used to
    /// update the particle weight.
    ///
    /// # Arguments
    /// * `position` - Particle position
    /// * `measurement` - Measurement model
    ///
    /// # Returns
    /// Log-likelihood for particle weight update
    pub fn update<M: MeasurementModel + ?Sized>(
        &mut self,
        position: &Vector3<f64>,
        measurement: &M,
    ) -> f64 {
        // Get sigma points
        let sigma_points = self.get_sigma_points();

        // Transform sigma points through measurement model
        let num_sigma = sigma_points.ncols();
        let meas_dim = measurement.get_vector().len();
        let mut predicted_measurements = DMatrix::zeros(meas_dim, num_sigma);

        for (i, sigma_col) in sigma_points.column_iter().enumerate() {
            // Reconstruct full 15-state vector from position + linear states
            let mut full_state = DVector::zeros(15);
            full_state[0] = position[0]; // lat
            full_state[1] = position[1]; // lon
            full_state[2] = position[2]; // alt
            full_state.rows_mut(3, 12).copy_from(&sigma_col);

            let pred_meas = measurement.get_expected_measurement(&full_state);
            predicted_measurements.set_column(i, &pred_meas);
        }

        // Compute predicted measurement mean
        let mut mean_meas = DVector::zeros(meas_dim);
        for (i, col) in predicted_measurements.column_iter().enumerate() {
            mean_meas += self.weights_mean[i] * col;
        }

        // Compute innovation covariance
        let mut innovation_cov = DMatrix::zeros(meas_dim, meas_dim);
        for (i, col) in predicted_measurements.column_iter().enumerate() {
            let diff = col - &mean_meas;
            innovation_cov += self.weights_cov[i] * (&diff * diff.transpose());
        }
        innovation_cov += measurement.get_noise();

        // Compute cross-covariance
        let mut cross_cov = DMatrix::zeros(12, meas_dim);
        for (i, sigma_col) in sigma_points.column_iter().enumerate() {
            let state_diff = sigma_col - &self.mean_state;
            let meas_diff = predicted_measurements.column(i) - &mean_meas;
            cross_cov += self.weights_cov[i] * (&state_diff * meas_diff.transpose());
        }

        // Compute Kalman gain
        let s_inv = robust_spd_solve(&symmetrize(&innovation_cov), &DMatrix::identity(meas_dim, meas_dim));
        let kalman_gain = &cross_cov * &s_inv;

        // Update state and covariance
        let innovation = measurement.get_vector() - mean_meas;
        self.mean_state += &kalman_gain * &innovation;

        // Compute determinant before moving innovation_cov
        let det = innovation_cov.determinant().max(1e-10);

        // Use Joseph form for numerical stability: P = P - K*S*K'
        // But compute as P - Pxy * inv(S) * Pxy' for better stability
        let updated_cov = &self.covariance - &cross_cov * &s_inv * cross_cov.transpose();
        self.covariance = symmetrize(&updated_cov);

        // Add small regularization to ensure positive definiteness
        for i in 0..self.covariance.nrows() {
            self.covariance[(i, i)] = self.covariance[(i, i)].max(1e-12);
        }

        // Compute log-likelihood for particle weighting
        let mahalanobis = (&innovation.transpose() * &s_inv * &innovation)[(0, 0)];
        let log_likelihood = -0.5
            * (mahalanobis + det.ln() + meas_dim as f64 * (2.0 * std::f64::consts::PI).ln());

        log_likelihood
    }
}

/// Rao-Blackwellized particle for RBPF.
///
/// Contains nonlinear position states (as particle) and conditionally-linear
/// states (as per-particle EKF).
#[derive(Clone, Debug)]
pub struct RBParticle {
    /// Position states: [latitude (rad), longitude (rad), altitude (m)]
    pub position: Vector3<f64>,

    /// Per-particle EKF for linear states (12-state: velocity, attitude, biases)
    pub kalman_filter: PerParticleEKF,

    /// Importance weight (unnormalized likelihood)
    pub weight: f64,
}

impl RBParticle {
    /// Convert RB particle to full 15-state vector for compatibility.
    ///
    /// Returns: [lat, lon, alt, v_n, v_e, v_v, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    pub fn to_state_vector(&self) -> DVector<f64> {
        let mut state = DVector::zeros(15);
        state[0] = self.position[0]; // lat
        state[1] = self.position[1]; // lon
        state[2] = self.position[2]; // alt

        let linear_states = self.kalman_filter.get_estimate();
        state.rows_mut(3, 12).copy_from(&linear_states);

        state
    }

    /// Create RB particle from full 15-state vector.
    pub fn from_state_vector(
        state: &DVector<f64>,
        initial_linear_cov: &DMatrix<f64>,
    ) -> Self {
        let position = Vector3::new(state[0], state[1], state[2]);
        let linear_states = state.rows(3, 12).into_owned();

        let kalman_filter = PerParticleEKF::new(
            linear_states,
            initial_linear_cov.clone(),
        );

        RBParticle {
            position,
            kalman_filter,
            weight: 1.0,
        }
    }

    /// Get position as Vector3.
    pub fn get_position(&self) -> Vector3<f64> {
        self.position
    }

    /// Get linear states from UKF.
    pub fn get_linear_states(&self) -> DVector<f64> {
        self.kalman_filter.get_estimate()
    }
}

/// Vertical channel treatment mode for particle filter.
#[derive(Clone, Debug, PartialEq)]
pub enum VerticalChannelMode {
    /// Third-order damping with feedback control.
    ///
    /// Adds two extra states (altitude error, altitude error rate) with
    /// damping coefficients k1 and k2. More complex but theoretically rigorous.
    ThirdOrderDamping { k1: f64, k2: f64 },

    /// Simplified 2.5D navigation (recommended).
    ///
    /// Treats vertical velocity as a random walk with large process noise.
    /// Simpler and effective for intermittent GPS scenarios.
    Simplified,
}

/// Resampling strategy for combating particle degeneracy.
#[derive(Clone, Debug, PartialEq)]
pub enum ResamplingStrategy {
    /// Systematic resampling (recommended).
    ///
    /// Single random draw determines all resampled indices. Most commonly used.
    Systematic,

    /// Stratified resampling.
    ///
    /// Random draw within each stratum. Slightly better variance than systematic.
    Stratified,

    /// Residual resampling.
    ///
    /// Deterministic part + random part. Good for avoiding sample impoverishment.
    Residual,

    /// Multinomial resampling.
    ///
    /// Independent draws with replacement. Simple but high variance.
    Multinomial,
}

/// Strategy for extracting state estimate from particle ensemble.
#[derive(Clone, Debug, PartialEq)]
pub enum AveragingStrategy {
    /// Weighted mean of all particles (recommended for Gaussian-like distributions).
    WeightedMean,

    /// Use the particle with maximum weight (good for multimodal distributions).
    MaximumWeight,

    /// Mean after removing outliers (robust to particle degeneracy).
    MeanWithTrimming { trim_fraction: f64 },
}

// ============================================================================
// Particle Filter Implementation
// ============================================================================

/// Particle filter for strapdown inertial navigation.
///
/// Implements Sequential Monte Carlo estimation for INS using a particle
/// representation of the posterior distribution. Supports 2.5D navigation
/// mode for simplified vertical channel treatment.
///
/// # Example
/// ```no_run
/// use strapdown::particle::{ParticleFilter, ProcessNoise, VerticalChannelMode, ResamplingStrategy, AveragingStrategy};
/// use strapdown::kalman::InitialState;
///
/// let initial_state = InitialState::default();
/// let process_noise = ProcessNoise::default();
///
/// let pf = ParticleFilter::new(
///     initial_state,
///     vec![0.0; 6], // IMU biases
///     None, // No additional states
///     vec![1e-3; 15], // Initial covariance diagonal
///     process_noise,
///     1000, // Number of particles
///     VerticalChannelMode::Simplified,
///     ResamplingStrategy::Systematic,
///     AveragingStrategy::WeightedMean,
///     Some(42), // Random seed
/// );
/// ```
pub struct ParticleFilter {
    /// Ensemble of particles representing the posterior distribution
    particles: Vec<Particle>,

    /// Number of particles in the ensemble
    num_particles: usize,

    /// Process noise parameters
    process_noise: ProcessNoise,

    /// Vertical channel treatment mode
    vertical_channel_mode: VerticalChannelMode,

    /// Resampling strategy
    resampling_strategy: ResamplingStrategy,

    /// Averaging strategy for state estimation
    averaging_strategy: AveragingStrategy,

    /// Effective particle count threshold for triggering resampling
    effective_particle_threshold: f64,

    /// Random number generator (seeded for reproducibility)
    rng: StdRng,

    /// Coordinate frame flag (ENU or NED)
    #[allow(dead_code)]
    is_enu: bool,

    /// State dimension
    state_size: usize,
}

impl Debug for ParticleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParticleFilter")
            .field("num_particles", &self.num_particles)
            .field("state_size", &self.state_size)
            .field("vertical_channel_mode", &self.vertical_channel_mode)
            .field("resampling_strategy", &self.resampling_strategy)
            .field("averaging_strategy", &self.averaging_strategy)
            .finish()
    }
}

impl Display for ParticleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParticleFilter(N={}, states={}, mode={:?})",
            self.num_particles, self.state_size, self.vertical_channel_mode
        )
    }
}

impl ParticleFilter {
    /// Create a new particle filter.
    ///
    /// # Arguments
    /// * `initial_state` - Initial navigation state (mean)
    /// * `imu_biases` - Initial IMU biases [b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    /// * `other_states` - Optional additional states (e.g., for damping)
    /// * `covariance_diagonal` - Initial covariance diagonal for particle initialization
    /// * `process_noise` - Process noise parameters
    /// * `num_particles` - Number of particles in ensemble
    /// * `vertical_mode` - Vertical channel treatment mode
    /// * `resampling_strategy` - Resampling algorithm
    /// * `averaging_strategy` - State estimation strategy
    /// * `seed` - Optional random seed for reproducibility
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_state: InitialState,
        imu_biases: Vec<f64>,
        other_states: Option<Vec<f64>>,
        covariance_diagonal: Vec<f64>,
        process_noise: ProcessNoise,
        num_particles: usize,
        vertical_mode: VerticalChannelMode,
        resampling_strategy: ResamplingStrategy,
        averaging_strategy: AveragingStrategy,
        seed: Option<u64>,
    ) -> Self {
        // Initialize RNG
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            // Use a default seed for reproducibility if no seed provided
            StdRng::seed_from_u64(0)
        };

        // Build mean state vector
        let mut mean = if initial_state.in_degrees {
            vec![
                initial_state.latitude.to_radians(),
                initial_state.longitude.to_radians(),
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        } else {
            vec![
                initial_state.latitude,
                initial_state.longitude,
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        };
        mean.extend(&imu_biases);
        if let Some(ref other) = other_states {
            mean.extend(other);
        }

        let state_size = mean.len();
        let mean_state = DVector::from_vec(mean);

        // Initialize particles by sampling from N(μ₀, P₀)
        let particles: Vec<Particle> = (0..num_particles)
            .map(|_| {
                let mut sampled_state = mean_state.clone();

                // Sample each state component from Gaussian
                for i in 0..state_size {
                    let std_dev = covariance_diagonal[i].sqrt();
                    let noise = Normal::new(0.0, std_dev).unwrap().sample(&mut rng);
                    sampled_state[i] += noise;
                }

                // Wrap angles to [-π, π]
                sampled_state[6] = wrap_to_2pi(sampled_state[6]);
                sampled_state[7] = wrap_to_2pi(sampled_state[7]);
                sampled_state[8] = wrap_to_2pi(sampled_state[8]);

                let mut particle = Particle::from_state_vector(&sampled_state, initial_state.is_enu);
                particle.weight = 1.0 / num_particles as f64;
                particle
            })
            .collect();

        ParticleFilter {
            particles,
            num_particles,
            process_noise,
            vertical_channel_mode: vertical_mode,
            resampling_strategy,
            averaging_strategy,
            effective_particle_threshold: 0.5 * num_particles as f64,
            rng,
            is_enu: initial_state.is_enu,
            state_size,
        }
    }

    /// Compute effective particle count (ESS).
    ///
    /// ESS = 1 / Σ(w_i²)
    ///
    /// Returns a value between 1 (all weight on one particle) and N (uniform weights).
    fn effective_particle_count(&self) -> f64 {
        let sum_squared_weights: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        1.0 / sum_squared_weights
    }

    /// Normalize particle weights to sum to 1.
    fn normalize_weights(&mut self) {
        let sum_weights: f64 = self.particles.iter().map(|p| p.weight).sum();
        if sum_weights > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= sum_weights;
            }
        } else {
            // All weights are zero - reset to uniform
            for particle in &mut self.particles {
                particle.weight = 1.0 / self.num_particles as f64;
            }
        }
    }

    /// Add process noise to a particle's state.
    ///
    /// This is critical for 2.5D navigation: vertical velocity and altitude
    /// receive large process noise to maintain particle diversity.
    fn add_process_noise(
        rng: &mut StdRng,
        process_noise: &ProcessNoise,
        particle: &mut Particle,
        dt: f64,
    ) {
        let sqrt_dt = dt.sqrt();

        // Position noise (lat, lon, alt)
        let lat_noise = Normal::new(0.0, process_noise.position_std[0]).unwrap().sample(rng);
        let lon_noise = Normal::new(0.0, process_noise.position_std[1]).unwrap().sample(rng);
        let alt_noise = Normal::new(0.0, process_noise.position_std[2]).unwrap().sample(rng);

        particle.nav_state.latitude += lat_noise * sqrt_dt;
        particle.nav_state.longitude += lon_noise * sqrt_dt;
        particle.nav_state.altitude += alt_noise * sqrt_dt;

        // Velocity noise (v_n, v_e, v_v) - NOTE: large v_v noise for 2.5D
        let vn_noise = Normal::new(0.0, process_noise.velocity_std[0]).unwrap().sample(rng);
        let ve_noise = Normal::new(0.0, process_noise.velocity_std[1]).unwrap().sample(rng);
        let vv_noise = Normal::new(0.0, process_noise.velocity_std[2]).unwrap().sample(rng);

        particle.nav_state.velocity_north += vn_noise * sqrt_dt;
        particle.nav_state.velocity_east += ve_noise * sqrt_dt;
        particle.nav_state.velocity_vertical += vv_noise * sqrt_dt; // Key for 2.5D!

        // Attitude noise (roll, pitch, yaw)
        let (roll, pitch, yaw) = particle.nav_state.attitude.euler_angles();
        let roll_noise = Normal::new(0.0, process_noise.attitude_std[0]).unwrap().sample(rng);
        let pitch_noise = Normal::new(0.0, process_noise.attitude_std[1]).unwrap().sample(rng);
        let yaw_noise = Normal::new(0.0, process_noise.attitude_std[2]).unwrap().sample(rng);

        particle.nav_state.attitude = Rotation3::from_euler_angles(
            wrap_to_2pi(roll + roll_noise * sqrt_dt),
            wrap_to_2pi(pitch + pitch_noise * sqrt_dt),
            wrap_to_2pi(yaw + yaw_noise * sqrt_dt),
        );

        // Bias noise (random walk)
        for i in 0..3 {
            let accel_noise = Normal::new(0.0, process_noise.accel_bias_std[i]).unwrap().sample(rng);
            particle.accel_bias[i] += accel_noise * sqrt_dt;

            let gyro_noise = Normal::new(0.0, process_noise.gyro_bias_std[i]).unwrap().sample(rng);
            particle.gyro_bias[i] += gyro_noise * sqrt_dt;
        }

        // Damping states noise (if applicable)
        if let Some(ref damping_std) = process_noise.damping_states_std {
            if let Some(ref mut other) = particle.other_states {
                for i in 0..other.len().min(2) {
                    let noise = Normal::new(0.0, damping_std[i]).unwrap().sample(rng);
                    other[i] += noise * sqrt_dt;
                }
            }
        }
    }

    /// Compute measurement likelihood for a single particle.
    ///
    /// Returns p(z|x) ∝ exp(-0.5 * (z - h(x))ᵀ R⁻¹ (z - h(x)))
    fn compute_likelihood(innovation: &DVector<f64>, meas_cov: &DMatrix<f64>) -> f64 {
        // Robust inverse using SVD
        let inv_cov = robust_spd_solve(&symmetrize(meas_cov), &DMatrix::identity(meas_cov.nrows(), meas_cov.ncols()));
        let mahalanobis = (innovation.transpose() * inv_cov * innovation)[(0, 0)];

        // Multivariate Gaussian likelihood (unnormalized)
        (-0.5 * mahalanobis).exp()
    }

    /// Systematic resampling algorithm.
    ///
    /// Uses a single random draw and deterministic spacing to select particles.
    /// This is the most commonly used resampling algorithm.
    fn systematic_resample(&mut self) {
        let n = self.num_particles;
        let mut new_particles = Vec::with_capacity(n);

        // Compute cumulative sum of weights
        let mut cumsum = vec![0.0; n];
        cumsum[0] = self.particles[0].weight;
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + self.particles[i].weight;
        }

        // Systematic sampling
        let step = 1.0 / n as f64;
        let start: f64 = self.rng.random_range(0.0..step);

        for i in 0..n {
            let u = start + i as f64 * step;
            let idx = cumsum.iter().position(|&cs| cs >= u).unwrap_or(n - 1);
            let mut new_particle = self.particles[idx].clone();
            new_particle.weight = 1.0 / n as f64; // Reset to uniform
            new_particles.push(new_particle);
        }

        self.particles = new_particles;
    }

    /// Resample particles based on configured strategy.
    fn resample(&mut self) {
        match self.resampling_strategy {
            ResamplingStrategy::Systematic => self.systematic_resample(),
            _ => {
                // For now, default to systematic
                // TODO: Implement other strategies
                self.systematic_resample();
            }
        }
    }

    /// Compute weighted mean estimate of state.
    fn weighted_mean_estimate(&self) -> DVector<f64> {
        let mut mean_state = DVector::zeros(self.state_size);

        for particle in &self.particles {
            let state_vec = particle.to_state_vector();
            mean_state += particle.weight * state_vec;
        }

        // Wrap angles to [-π, π]
        mean_state[6] = wrap_to_2pi(mean_state[6]);
        mean_state[7] = wrap_to_2pi(mean_state[7]);
        mean_state[8] = wrap_to_2pi(mean_state[8]);

        mean_state
    }

    /// Compute empirical covariance from particle ensemble.
    fn empirical_covariance(&self) -> DMatrix<f64> {
        let mean = self.weighted_mean_estimate();
        let mut cov = DMatrix::zeros(self.state_size, self.state_size);

        for particle in &self.particles {
            let state_vec = particle.to_state_vector();
            let diff = &state_vec - &mean;
            cov += particle.weight * &diff * diff.transpose();
        }

        symmetrize(&cov)
    }
}

// ============================================================================
// Rao-Blackwellized Particle Filter Implementation
// ============================================================================

/// Rao-Blackwellized Particle Filter for strapdown inertial navigation.
///
/// Partitions the 15-state INS into:
/// - Nonlinear states (position: 3) estimated via particles
/// - Conditionally-linear states (velocity, attitude, biases: 12) estimated via per-particle UKF
///
/// This reduces computational burden while improving bias estimation quality.
///
/// # Example
/// ```no_run
/// use strapdown::particle::{RaoBlackwellizedParticleFilter, RBProcessNoise, VerticalChannelMode, ResamplingStrategy};
/// use strapdown::kalman::InitialState;
///
/// let initial_state = InitialState::default();
/// let process_noise = RBProcessNoise::default();
///
/// let rbpf = RaoBlackwellizedParticleFilter::new(
///     initial_state,
///     vec![0.0; 6], // IMU biases
///     vec![1e-3; 15], // Initial covariance diagonal
///     process_noise,
///     100, // Fewer particles than standard PF
///     VerticalChannelMode::Simplified,
///     ResamplingStrategy::Systematic,
///     Some(42), // Random seed
/// );
/// ```
pub struct RaoBlackwellizedParticleFilter {
    /// Ensemble of RB particles
    particles: Vec<RBParticle>,

    /// Number of particles
    num_particles: usize,

    /// Process noise parameters
    process_noise: RBProcessNoise,

    /// Vertical channel mode (2.5D simplified)
    vertical_channel_mode: VerticalChannelMode,

    /// Resampling strategy
    resampling_strategy: ResamplingStrategy,

    /// Effective particle threshold for resampling
    effective_particle_threshold: f64,

    /// Random number generator (seeded)
    rng: StdRng,

    /// Coordinate frame flag
    is_enu: bool,
}

impl Debug for RaoBlackwellizedParticleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RaoBlackwellizedParticleFilter")
            .field("num_particles", &self.num_particles)
            .field("vertical_channel_mode", &self.vertical_channel_mode)
            .field("resampling_strategy", &self.resampling_strategy)
            .finish()
    }
}

impl Display for RaoBlackwellizedParticleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RaoBlackwellizedParticleFilter(N={}, mode={:?})",
            self.num_particles, self.vertical_channel_mode
        )
    }
}

impl RaoBlackwellizedParticleFilter {
    /// Create a new Rao-Blackwellized particle filter with EKF.
    ///
    /// # Arguments
    /// * `initial_state` - Initial navigation state (mean)
    /// * `imu_biases` - Initial IMU biases [b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    /// * `covariance_diagonal` - Initial covariance diagonal (15-element: 3 position + 12 linear)
    /// * `process_noise` - Process noise parameters for RBPF
    /// * `num_particles` - Number of particles in ensemble (typically 50-200)
    /// * `vertical_mode` - Vertical channel treatment mode
    /// * `resampling_strategy` - Resampling algorithm
    /// * `seed` - Optional random seed for reproducibility
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_state: InitialState,
        imu_biases: Vec<f64>,
        covariance_diagonal: Vec<f64>,
        process_noise: RBProcessNoise,
        num_particles: usize,
        vertical_mode: VerticalChannelMode,
        resampling_strategy: ResamplingStrategy,
        seed: Option<u64>,
    ) -> Self {
        // Initialize RNG
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::seed_from_u64(0)
        };

        // Build mean state vector
        let mut mean = if initial_state.in_degrees {
            vec![
                initial_state.latitude.to_radians(),
                initial_state.longitude.to_radians(),
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        } else {
            vec![
                initial_state.latitude,
                initial_state.longitude,
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        };
        mean.extend(&imu_biases);

        let mean_state = DVector::from_vec(mean);

        // Extract position and linear covariances
        let position_cov_diag = Vector3::new(
            covariance_diagonal[0],
            covariance_diagonal[1],
            covariance_diagonal[2],
        );

        let linear_cov_diag = DVector::from_vec(covariance_diagonal[3..15].to_vec());
        let linear_cov = DMatrix::from_diagonal(&linear_cov_diag);

        // Initialize particles by sampling positions and initializing EKFs
        let particles: Vec<RBParticle> = (0..num_particles)
            .map(|_| {
                // Sample position from N(μ_pos, P_pos)
                let mut position = Vector3::new(mean_state[0], mean_state[1], mean_state[2]);
                for i in 0..3 {
                    let std_dev = position_cov_diag[i].sqrt();
                    let noise = Normal::new(0.0, std_dev).unwrap().sample(&mut rng);
                    position[i] += noise;
                }

                // Initialize per-particle EKF with mean linear states
                let linear_states = mean_state.rows(3, 12).into_owned();
                let kalman_filter =
                    PerParticleEKF::new(linear_states, linear_cov.clone());

                RBParticle {
                    position,
                    kalman_filter,
                    weight: 1.0 / num_particles as f64,
                }
            })
            .collect();

        RaoBlackwellizedParticleFilter {
            particles,
            num_particles,
            process_noise,
            vertical_channel_mode: vertical_mode,
            resampling_strategy,
            effective_particle_threshold: 0.5 * num_particles as f64,
            rng,
            is_enu: initial_state.is_enu,
        }
    }

    /// Compute effective particle count (ESS).
    fn effective_particle_count(&self) -> f64 {
        let sum_squared_weights: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        1.0 / sum_squared_weights
    }

    /// Normalize particle weights to sum to 1.
    fn normalize_weights(&mut self) {
        let sum_weights: f64 = self.particles.iter().map(|p| p.weight).sum();
        if sum_weights > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= sum_weights;
            }
        } else {
            // Reset to uniform
            for particle in &mut self.particles {
                particle.weight = 1.0 / self.num_particles as f64;
            }
        }
    }

    /// Weighted mean of position states.
    fn weighted_position_mean(&self) -> Vector3<f64> {
        let mut mean_pos = Vector3::zeros();
        for particle in &self.particles {
            mean_pos += particle.weight * particle.position;
        }
        mean_pos
    }

    /// Weighted mean of linear states from per-particle UKFs.
    fn weighted_linear_states_mean(&self) -> DVector<f64> {
        let mut mean_linear = DVector::zeros(12);
        for particle in &self.particles {
            let linear_states = particle.kalman_filter.get_estimate();
            mean_linear += particle.weight * linear_states;
        }

        // Wrap angles to [-π, π]
        mean_linear[3] = wrap_to_2pi(mean_linear[3]); // roll
        mean_linear[4] = wrap_to_2pi(mean_linear[4]); // pitch
        mean_linear[5] = wrap_to_2pi(mean_linear[5]); // yaw

        mean_linear
    }

    /// Compute combined covariance from particle positions and UKF covariances.
    fn compute_combined_covariance(&self) -> DMatrix<f64> {
        let mut cov = DMatrix::zeros(15, 15);

        // 1. Position covariance (from particle ensemble)
        let mean_pos = self.weighted_position_mean();
        for particle in &self.particles {
            let diff_pos = particle.position - mean_pos;
            for i in 0..3 {
                for j in 0..3 {
                    cov[(i, j)] += particle.weight * diff_pos[i] * diff_pos[j];
                }
            }
        }

        // 2. Weighted average of per-particle UKF covariances (linear states)
        for particle in &self.particles {
            let ukf_cov = particle.kalman_filter.get_covariance();
            for i in 0..12 {
                for j in 0..12 {
                    cov[(3 + i, 3 + j)] += particle.weight * ukf_cov[(i, j)];
                }
            }
        }

        // 3. Cross terms (position-linear correlation) - simplified as zero for now
        // In full implementation, would compute cross-covariance from particles

        symmetrize(&cov)
    }

    /// Systematic resampling for RBPF.
    fn systematic_resample(&mut self) {
        let n = self.num_particles;
        let mut new_particles = Vec::with_capacity(n);

        // Compute cumulative sum of weights
        let mut cumsum = vec![0.0; n];
        cumsum[0] = self.particles[0].weight;
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + self.particles[i].weight;
        }

        // Systematic sampling
        let step = 1.0 / n as f64;
        let start: f64 = self.rng.random_range(0.0..step);

        for i in 0..n {
            let u = start + i as f64 * step;
            let idx = cumsum.iter().position(|&cs| cs >= u).unwrap_or(n - 1);
            let mut new_particle = self.particles[idx].clone();
            new_particle.weight = 1.0 / n as f64; // Reset to uniform
            new_particles.push(new_particle);
        }

        self.particles = new_particles;
    }

    /// Resample particles based on configured strategy.
    fn resample(&mut self) {
        match self.resampling_strategy {
            ResamplingStrategy::Systematic => self.systematic_resample(),
            _ => {
                // Default to systematic for other strategies (not yet implemented)
                self.systematic_resample();
            }
        }
    }
}

// ============================================================================
// NavigationFilter Trait Implementation for RBPF
// ============================================================================

impl NavigationFilter for RaoBlackwellizedParticleFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        let process_noise = self.process_noise.clone();
        let is_enu = self.is_enu;

        for particle in &mut self.particles {
            // 1. Get current velocity and attitude from UKF
            let linear_states = particle.kalman_filter.get_estimate();
            let velocity = Vector3::new(linear_states[0], linear_states[1], linear_states[2]);

            // 2. Propagate position using forward_2_5d_rbpf (attitude, velocity given)
            forward_2_5d_rbpf(&mut particle.position, &velocity, dt);

            // 3. Add process noise to position
            add_position_noise(
                &mut particle.position,
                &process_noise.position_std,
                dt,
                &mut self.rng,
            );

            // 4. Extract biases from UKF state
            let accel_bias = Vector3::new(linear_states[6], linear_states[7], linear_states[8]);
            let gyro_bias = Vector3::new(linear_states[9], linear_states[10], linear_states[11]);

            // 5. Bias-corrected IMU
            let corrected_imu = IMUData {
                accel: imu_data.accel - accel_bias,
                gyro: imu_data.gyro - gyro_bias,
            };

            // 6. UKF predict for linear states (conditioned on new position)
            particle.kalman_filter.predict(
                &particle.position,
                corrected_imu,
                &process_noise.linear_states_covariance,
                dt,
                is_enu,
            );
        }
    }

    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // 1. Compute marginal likelihood for each particle
        for particle in &mut self.particles {
            // UKF update returns marginal likelihood p(z|position)
            let log_likelihood = particle.kalman_filter.update(&particle.position, measurement);
            particle.weight *= log_likelihood.exp();
        }

        // 2. Normalize weights
        self.normalize_weights();

        // 3. Check ESS and resample if needed
        let ess = self.effective_particle_count();
        if ess < self.effective_particle_threshold {
            self.resample();
        }
    }

    fn get_estimate(&self) -> DVector<f64> {
        // 1. Weighted average of positions
        let mean_position = self.weighted_position_mean();

        // 2. Weighted average of per-particle UKF means
        let mean_linear_states = self.weighted_linear_states_mean();

        // 3. Concatenate into 15-state vector
        let mut state = DVector::zeros(15);
        state.fixed_rows_mut::<3>(0).copy_from(&mean_position);
        state.fixed_rows_mut::<12>(3).copy_from(&mean_linear_states);
        state
    }

    fn get_certainty(&self) -> DMatrix<f64> {
        // Combined covariance from:
        // 1. Particle position covariance (empirical from weighted samples)
        // 2. Weighted average of per-particle UKF covariances
        // 3. Cross terms (position-linear states correlation)
        self.compute_combined_covariance()
    }
}

// ============================================================================
// NavigationFilter Trait Implementation
// ============================================================================

impl NavigationFilter for ParticleFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        let process_noise = self.process_noise.clone();
        let vertical_mode = self.vertical_channel_mode.clone();

        for particle in &mut self.particles {
            // 1. Bias-corrected IMU data
            let corrected_imu = IMUData {
                accel: imu_data.accel - &particle.accel_bias,
                gyro: imu_data.gyro - &particle.gyro_bias,
            };

            // 2. Propagate based on vertical channel mode
            match vertical_mode {
                VerticalChannelMode::Simplified => {
                    // Use 2.5D forward propagation
                    forward_2_5d(&mut particle.nav_state, corrected_imu, dt);
                }
                VerticalChannelMode::ThirdOrderDamping { .. } => {
                    // TODO: Implement third-order damping
                    // For now, use 2.5D as fallback
                    forward_2_5d(&mut particle.nav_state, corrected_imu, dt);
                }
            }

            // 3. Add process noise (maintains particle diversity)
            Self::add_process_noise(&mut self.rng, &process_noise, particle, dt);
        }
    }

    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // 1. Compute weights based on measurement likelihood
        for particle in &mut self.particles {
            let predicted_meas = measurement.get_expected_measurement(&particle.to_state_vector());
            let innovation = measurement.get_vector() - predicted_meas;
            let meas_cov = measurement.get_noise();

            let likelihood = Self::compute_likelihood(&innovation, &meas_cov);
            particle.weight *= likelihood;
        }

        // 2. Normalize weights
        self.normalize_weights();

        // 3. Check effective particle count
        let n_eff = self.effective_particle_count();

        // 4. Resample if needed
        if n_eff < self.effective_particle_threshold {
            self.resample();
        }
    }

    fn get_estimate(&self) -> DVector<f64> {
        match self.averaging_strategy {
            AveragingStrategy::WeightedMean => self.weighted_mean_estimate(),
            AveragingStrategy::MaximumWeight => {
                // Find particle with maximum weight
                let max_particle = self
                    .particles
                    .iter()
                    .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
                    .unwrap();
                max_particle.to_state_vector()
            }
            AveragingStrategy::MeanWithTrimming { .. } => {
                // TODO: Implement trimmed mean
                // For now, use weighted mean as fallback
                self.weighted_mean_estimate()
            }
        }
    }

    fn get_certainty(&self) -> DMatrix<f64> {
        self.empirical_covariance()
    }
}

// ============================================================================
// 2.5D Navigation Functions
// ============================================================================

/// Modified forward propagation for 2.5D navigation.
///
/// Implements full strapdown mechanization for horizontal navigation (attitude,
/// horizontal velocities, horizontal position) while treating the vertical channel
/// with simplified dynamics. Vertical acceleration is NOT integrated into vertical
/// velocity; instead, vertical velocity follows a random walk model constrained by
/// measurements and process noise.
///
/// # Arguments
/// * `state` - Mutable reference to the navigation state
/// * `imu_data` - Bias-corrected IMU measurements
/// * `dt` - Time step in seconds
///
/// # Mathematical Basis
/// - **Horizontal states**: Full Groves Equations 5.46, 5.54, 5.56
/// - **Vertical velocity**: Random walk (process noise driven)
/// - **Altitude**: Simple integration of vertical velocity
///
/// # Key Difference from Standard `forward()`
/// The standard strapdown mechanization integrates vertical acceleration to update
/// vertical velocity: `v_v(t+dt) = v_v(t) + ∫[a_z + g - Coriolis] dt`
///
/// In 2.5D navigation, we decouple this: `v_v(t+dt) = v_v(t)` during propagation,
/// and vertical velocity changes come from process noise and measurement updates only.
///
/// # Example
/// ```
/// use strapdown::particle::forward_2_5d;
/// use strapdown::{StrapdownState, IMUData};
/// use nalgebra::Vector3;
///
/// let mut state = StrapdownState::default();
/// let imu = IMUData {
///     accel: Vector3::new(0.0, 0.0, 9.81),
///     gyro: Vector3::zeros(),
/// };
/// forward_2_5d(&mut state, imu, 0.1);
/// ```
pub fn forward_2_5d(state: &mut StrapdownState, imu_data: IMUData, dt: f64) {
    // 1. ATTITUDE UPDATE (unchanged from standard forward)
    //    Equation 5.46 - uses gyros, accounts for Earth rate and transport rate
    let c_0: Rotation3<f64> = state.attitude;
    let c_1: Matrix3<f64> = attitude_update(state, imu_data.gyro, dt);

    // 2. SPECIFIC FORCE TRANSFORMATION (unchanged)
    //    Equation 5.47 - transform body frame accel to nav frame
    let f: Vector3<f64> = 0.5 * (c_0.matrix() + c_1) * imu_data.accel;

    // 3. HORIZONTAL VELOCITY UPDATE (modified - only horizontal components)
    //    Full mechanization for v_n and v_e, IGNORE v_v integration
    let velocity_horizontal = velocity_update_horizontal(state, f, dt);

    // 4. VERTICAL VELOCITY (simplified - no deterministic integration)
    //    Keep current vertical velocity unchanged (process noise will be added later)
    let velocity_vertical = state.velocity_vertical;

    // 5. POSITION UPDATE
    //    - Horizontal: Full mechanization using updated horizontal velocities
    //    - Vertical: Simple integration of vertical velocity
    let velocity = Vector3::new(velocity_horizontal[0], velocity_horizontal[1], velocity_vertical);
    let (lat_1, lon_1, alt_1) = position_update(state, velocity, dt);

    // 6. UPDATE STATE
    state.attitude = Rotation3::from_matrix(&c_1);
    state.velocity_north = velocity_horizontal[0];
    state.velocity_east = velocity_horizontal[1];
    state.velocity_vertical = velocity_vertical; // Unchanged from predict
    state.latitude = lat_1;
    state.longitude = lon_1;
    state.altitude = alt_1;
}

/// Horizontal velocity update for 2.5D navigation.
///
/// Implements full strapdown velocity update (Groves Equation 5.54) but only for
/// horizontal components (north and east). The vertical component is ignored.
///
/// This function computes the full velocity update equation including:
/// - Specific force (transformed to nav frame)
/// - Gravity
/// - Coriolis effect (2 * Earth rotation rate)
/// - Transport rate effect
///
/// But only returns the horizontal (north, east) components.
///
/// # Arguments
/// * `state` - Current navigation state
/// * `specific_force` - Specific force vector in navigation frame (m/s²)
/// * `dt` - Time step in seconds
///
/// # Returns
/// Vector2 containing [v_north, v_east] in m/s
fn velocity_update_horizontal(
    state: &StrapdownState,
    specific_force: Vector3<f64>,
    dt: f64,
) -> Vector2<f64> {
    // Compute transport rate (effect of velocity on local-level frame rotation)
    let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
        &state.latitude.to_degrees(),
        &state.altitude,
        &Vector3::from_vec(vec![
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
        ]),
    ));

    // Compute Earth rotation rate in local-level frame
    let rotation_rate: Matrix3<f64> =
        earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&state.latitude.to_degrees()));

    // Get ECEF position vector (for Coriolis calculation)
    let r = earth::ecef_to_lla(&state.latitude.to_degrees(), &state.longitude.to_degrees());

    let velocity: Vector3<f64> = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_vertical,
    );

    // Gravity vector (only vertical component non-zero)
    let gravity = Vector3::new(
        0.0,
        0.0,
        earth::gravity(&state.latitude.to_degrees(), &state.altitude),
    );
    // Adjust sign based on coordinate frame convention
    let gravity = if state.is_enu { -gravity } else { gravity };

    // Full velocity update equation (Equation 5.54)
    // v(t+dt) = v(t) + [f + g - r * (Ω_el + 2*Ω_ie) * v] * dt
    let velocity_full = velocity
        + (specific_force + gravity - r * (transport_rate + 2.0 * rotation_rate) * velocity) * dt;

    // RETURN ONLY HORIZONTAL COMPONENTS
    // This is the key difference from standard velocity_update()
    Vector2::new(velocity_full[0], velocity_full[1])
}

/// Propagate velocity states for RBPF (2.5D navigation).
///
/// Updates velocity given specific force and position.Used in per-particle UKF.
///
/// # Arguments
/// * `velocity` - Current velocity [v_n, v_e, v_v]
/// * `specific_force` - Specific force in navigation frame
/// * `position` - Position [lat, lon, alt]
/// * `dt` - Time step
/// * `is_enu` - Coordinate frame flag
///
/// # Returns
/// Updated velocity vector
fn propagate_velocity_2_5d(
    velocity: &Vector3<f64>,
    specific_force: &Vector3<f64>,
    position: &Vector3<f64>,
    dt: f64,
    is_enu: bool,
) -> Vector3<f64> {
    let lat_deg = position[0].to_degrees();
    let alt = position[2];

    // Compute transport rate
    let transport_rate = earth::vector_to_skew_symmetric(&earth::transport_rate(&lat_deg, &alt, velocity));

    // Compute Earth rotation rate
    let rotation_rate = earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&lat_deg));

    // Get ECEF position for Coriolis
    let r = earth::ecef_to_lla(&lat_deg, &position[1].to_degrees());

    // Gravity
    let mut gravity = Vector3::new(0.0, 0.0, earth::gravity(&lat_deg, &alt));
    gravity = if is_enu { -gravity } else { gravity };

    // Full velocity update (Equation 5.54)
    let velocity_full =
        velocity + (specific_force + gravity - r * (transport_rate + 2.0 * rotation_rate) * velocity) * dt;

    // For 2.5D: use full horizontal velocities, but keep vertical velocity unchanged
    // (process noise will perturb it)
    Vector3::new(velocity_full[0], velocity_full[1], velocity[2])
}

/// Propagate attitude states for RBPF.
///
/// Updates Euler angles given angular rates and position. Used in per-particle UKF.
///
/// # Arguments
/// * `attitude_euler` - Current Euler angles [roll, pitch, yaw]
/// * `gyro` - Bias-corrected gyro measurements (body frame)
/// * `position` - Position [lat, lon, alt]
/// * `dt` - Time step
///
/// # Returns
/// Updated Euler angles
fn propagate_attitude(
    attitude_euler: &Vector3<f64>,
    gyro: &Vector3<f64>,
    position: &Vector3<f64>,
    dt: f64,
) -> Vector3<f64> {
    let lat_deg = position[0].to_degrees();
    let _alt = position[2];

    // Create rotation matrix from Euler angles
    let (roll, pitch, yaw) = (attitude_euler[0], attitude_euler[1], attitude_euler[2]);
    let c_bn = Rotation3::from_euler_angles(roll, pitch, yaw);

    // Compute Earth rate and transport rate in body frame
    let earth_rate_nav = earth::earth_rate_lla(&lat_deg);
    let earth_rate_body = c_bn.matrix().transpose() * earth_rate_nav;

    // For transport rate, we need velocity - use zero velocity approximation for UKF
    // (actual velocity is in the sigma point but coupling is weak)
    let transport_rate_body = Vector3::zeros();

    // Attitude update: integrate angular rate
    let omega_ib_b = gyro;
    let omega_nb_b = omega_ib_b - earth_rate_body - transport_rate_body;

    // Small angle approximation for Euler angle rates
    let omega_norm = omega_nb_b.norm();
    if omega_norm < 1e-8 {
        // No rotation
        *attitude_euler
    } else {
        // Update rotation matrix
        let skew_omega = earth::vector_to_skew_symmetric(&omega_nb_b);
        let delta_c = Matrix3::identity() + skew_omega * dt;
        let c_new = delta_c * c_bn.matrix();

        // Convert back to Euler angles
        let rot_new = Rotation3::from_matrix(&c_new);
        let (roll_new, pitch_new, yaw_new) = rot_new.euler_angles();

        Vector3::new(
            wrap_to_2pi(roll_new),
            wrap_to_2pi(pitch_new),
            wrap_to_2pi(yaw_new),
        )
    }
}

/// Forward propagation for RBPF position states (2.5D navigation).
///
/// Propagates position given velocity from per-particle UKF.
///
/// # Arguments
/// * `position` - Mutable reference to position [lat, lon, alt]
/// * `velocity` - Velocity from UKF [v_n, v_e, v_v]
/// * `dt` - Time step
pub fn forward_2_5d_rbpf(position: &mut Vector3<f64>, velocity: &Vector3<f64>, dt: f64) {
    let lat = position[0];
    let lon = position[1];
    let alt = position[2];

    // Get Earth parameters
    let lat_deg = lat.to_degrees();
    let (r_n, r_e, _r_d) = earth::principal_radii(&lat_deg, &alt);

    // Update position using geodetic equations
    let d_lat = velocity[0] / (r_n + alt) * dt;
    let d_lon = velocity[1] / ((r_e + alt) * lat.cos()) * dt;
    let d_alt = velocity[2] * dt;

    position[0] = lat + d_lat;
    position[1] = lon + d_lon;
    position[2] = alt + d_alt;
}

/// Add process noise to position states (for RBPF particle diffusion).
fn add_position_noise(
    position: &mut Vector3<f64>,
    noise_std: &Vector3<f64>,
    dt: f64,
    rng: &mut StdRng,
) {
    let sqrt_dt = dt.sqrt();

    for i in 0..3 {
        let noise = Normal::new(0.0, noise_std[i]).unwrap().sample(rng);
        position[i] += noise * sqrt_dt;
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{earth, forward};
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_forward_2_5d_vs_forward_horizontal() {
        // Test that horizontal states match between forward() and forward_2_5d()
        let mut state_standard = StrapdownState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            velocity_north: 10.0,
            velocity_east: 5.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let mut state_2_5d = state_standard;

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &100.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        let dt = 0.1;

        // Propagate both
        forward(&mut state_standard, imu_data, dt);
        forward_2_5d(&mut state_2_5d, imu_data, dt);

        // Horizontal states should match closely
        assert_approx_eq!(state_standard.latitude, state_2_5d.latitude, 1e-9);
        assert_approx_eq!(state_standard.longitude, state_2_5d.longitude, 1e-9);
        assert_approx_eq!(
            state_standard.velocity_north,
            state_2_5d.velocity_north,
            1e-6
        );
        assert_approx_eq!(state_standard.velocity_east, state_2_5d.velocity_east, 1e-6);

        // Attitude should match
        let (r1, p1, y1) = state_standard.attitude.euler_angles();
        let (r2, p2, y2) = state_2_5d.attitude.euler_angles();
        assert_approx_eq!(r1, r2, 1e-9);
        assert_approx_eq!(p1, p2, 1e-9);
        assert_approx_eq!(y1, y2, 1e-9);

        // Vertical velocity should be DIFFERENT
        // Standard: integrates acceleration -> should be approximately unchanged (hovering)
        // 2.5D: no integration -> should be exactly unchanged
        assert_approx_eq!(state_2_5d.velocity_vertical, 0.0, 1e-12);
    }

    #[test]
    fn test_forward_2_5d_vertical_decoupling() {
        // Test that vertical acceleration does NOT affect vertical velocity in 2.5D
        let mut state = StrapdownState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 5.0, // Initial vertical velocity
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        // Strong vertical acceleration (in free fall)
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 0.0), // Free fall - no measured accel
            gyro: Vector3::zeros(),
        };

        let dt = 0.1;
        let initial_vv = state.velocity_vertical;

        forward_2_5d(&mut state, imu_data, dt);

        // In 2.5D, vertical velocity should be UNCHANGED
        assert_approx_eq!(state.velocity_vertical, initial_vv, 1e-12);

        // But altitude should still integrate from velocity
        let expected_alt = 100.0 + initial_vv * dt;
        assert_approx_eq!(state.altitude, expected_alt, 1e-6);
    }

    #[test]
    fn test_velocity_update_horizontal_only() {
        // Test that velocity_update_horizontal returns only 2 components
        let state = StrapdownState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            velocity_north: 10.0,
            velocity_east: 5.0,
            velocity_vertical: 2.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        let specific_force = Vector3::new(1.0, 0.5, -9.81);
        let dt = 0.1;

        let v_horizontal = velocity_update_horizontal(&state, specific_force, dt);

        // Should return 2-element vector
        assert_eq!(v_horizontal.len(), 2);

        // Should be close to initial horizontal velocities plus small changes
        assert!((v_horizontal[0] - 10.0).abs() < 1.0); // North velocity
        assert!((v_horizontal[1] - 5.0).abs() < 1.0); // East velocity
    }

    #[test]
    fn test_forward_2_5d_with_rotation() {
        // Test 2.5D propagation with body rotation
        let mut state = StrapdownState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::identity(),
            is_enu: true,
        };

        // Rotate around Z-axis (yaw)
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 9.81),
            gyro: Vector3::new(0.0, 0.0, 0.1), // 0.1 rad/s yaw rate
        };

        let dt = 0.1;

        forward_2_5d(&mut state, imu_data, dt);

        // Attitude should have changed
        let (_, _, yaw) = state.attitude.euler_angles();
        assert!(yaw.abs() > 0.0);

        // Position and velocity should still be reasonable
        assert!(state.latitude.abs() < 1e-3);
        assert!(state.longitude.abs() < 1e-3);
        assert_approx_eq!(state.altitude, 100.0, 0.1);
    }

    // ========================================================================
    // Particle struct tests
    // ========================================================================

    #[test]
    fn test_particle_to_state_vector() {
        let particle = Particle {
            nav_state: StrapdownState {
                latitude: 0.1,
                longitude: 0.2,
                altitude: 100.0,
                velocity_north: 10.0,
                velocity_east: 5.0,
                velocity_vertical: 2.0,
                attitude: Rotation3::from_euler_angles(0.01, 0.02, 0.03),
                is_enu: true,
            },
            accel_bias: DVector::from_vec(vec![0.1, 0.2, 0.3]),
            gyro_bias: DVector::from_vec(vec![0.01, 0.02, 0.03]),
            other_states: None,
            state_size: 15,
            weight: 0.5,
        };

        let state_vec = particle.to_state_vector();

        assert_eq!(state_vec.len(), 15);
        assert_approx_eq!(state_vec[0], 0.1, 1e-12); // latitude
        assert_approx_eq!(state_vec[1], 0.2, 1e-12); // longitude
        assert_approx_eq!(state_vec[2], 100.0, 1e-12); // altitude
        assert_approx_eq!(state_vec[3], 10.0, 1e-12); // v_n
        assert_approx_eq!(state_vec[4], 5.0, 1e-12); // v_e
        assert_approx_eq!(state_vec[5], 2.0, 1e-12); // v_v
        // Euler angles [6, 7, 8]
        assert_approx_eq!(state_vec[9], 0.1, 1e-12); // accel bias x
        assert_approx_eq!(state_vec[14], 0.03, 1e-12); // gyro bias z
    }

    #[test]
    fn test_particle_from_state_vector() {
        let state_vec = DVector::from_vec(vec![
            0.1, 0.2, 100.0, // position
            10.0, 5.0, 2.0, // velocity
            0.01, 0.02, 0.03, // attitude
            0.1, 0.2, 0.3, // accel bias
            0.01, 0.02, 0.03, // gyro bias
        ]);

        let particle = Particle::from_state_vector(&state_vec, true);

        assert_approx_eq!(particle.nav_state.latitude, 0.1, 1e-12);
        assert_approx_eq!(particle.nav_state.altitude, 100.0, 1e-12);
        assert_approx_eq!(particle.nav_state.velocity_north, 10.0, 1e-12);
        assert_eq!(particle.accel_bias.len(), 3);
        assert_approx_eq!(particle.accel_bias[0], 0.1, 1e-12);
        assert_eq!(particle.gyro_bias.len(), 3);
        assert_approx_eq!(particle.gyro_bias[2], 0.03, 1e-12);
        assert_eq!(particle.state_size, 15);
    }

    #[test]
    fn test_particle_roundtrip() {
        let original = Particle {
            nav_state: StrapdownState {
                latitude: 0.1,
                longitude: 0.2,
                altitude: 100.0,
                velocity_north: 10.0,
                velocity_east: 5.0,
                velocity_vertical: 2.0,
                attitude: Rotation3::from_euler_angles(0.01, 0.02, 0.03),
                is_enu: true,
            },
            accel_bias: DVector::from_vec(vec![0.1, 0.2, 0.3]),
            gyro_bias: DVector::from_vec(vec![0.01, 0.02, 0.03]),
            other_states: None,
            state_size: 15,
            weight: 0.5,
        };

        let state_vec = original.to_state_vector();
        let recovered = Particle::from_state_vector(&state_vec, true);

        assert_approx_eq!(original.nav_state.latitude, recovered.nav_state.latitude, 1e-12);
        assert_approx_eq!(original.nav_state.altitude, recovered.nav_state.altitude, 1e-12);
        assert_approx_eq!(
            original.nav_state.velocity_north,
            recovered.nav_state.velocity_north,
            1e-12
        );
    }

    // ========================================================================
    // ProcessNoise tests
    // ========================================================================

    #[test]
    fn test_process_noise_default() {
        let pn = ProcessNoise::default();

        // Check 2.5D characteristics: large vertical noise
        assert!(pn.position_std[2] > pn.position_std[0]); // alt > lat
        assert!(pn.velocity_std[2] > pn.velocity_std[0]); // v_v > v_n

        // Typical values
        assert!(pn.position_std[2] > 1e-3); // Altitude noise > 1mm
        assert!(pn.velocity_std[2] > 1e-2); // Vertical velocity noise > 1cm/s
        assert_eq!(pn.damping_states_std, None);
    }

    // ========================================================================
    // ParticleFilter initialization tests
    // ========================================================================

    #[test]
    fn test_particle_filter_initialization() {
        use crate::kalman::InitialState;

        let initial_state = InitialState {
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: true,
            is_enu: true,
        };

        let pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-3; 15],
            ProcessNoise::default(),
            100,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        assert_eq!(pf.num_particles, 100);
        assert_eq!(pf.particles.len(), 100);
        assert_eq!(pf.state_size, 15);

        // Check weights sum to 1
        let weight_sum: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);

        // Check particles are different (sampled from distribution)
        let lat0 = pf.particles[0].nav_state.latitude;
        let lat1 = pf.particles[1].nav_state.latitude;
        assert!((lat0 - lat1).abs() > 0.0); // Should be different due to sampling
    }

    #[test]
    fn test_particle_filter_effective_particle_count() {
        use crate::kalman::InitialState;

        let initial_state = InitialState::default();
        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-6; 15],
            ProcessNoise::default(),
            100,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Uniform weights -> ESS = N
        let ess = pf.effective_particle_count();
        assert_approx_eq!(ess, 100.0, 1.0);

        // Set one particle to have all weight
        for particle in &mut pf.particles {
            particle.weight = 0.0;
        }
        pf.particles[0].weight = 1.0;

        let ess_degenerate = pf.effective_particle_count();
        assert_approx_eq!(ess_degenerate, 1.0, 1e-9);
    }

    #[test]
    fn test_particle_filter_normalize_weights() {
        use crate::kalman::InitialState;

        let initial_state = InitialState::default();
        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-6; 15],
            ProcessNoise::default(),
            10,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Set arbitrary weights
        for (i, particle) in pf.particles.iter_mut().enumerate() {
            particle.weight = (i + 1) as f64;
        }

        pf.normalize_weights();

        // Check sum is 1
        let weight_sum: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);

        // Check relative ordering maintained
        assert!(pf.particles[9].weight > pf.particles[0].weight);
    }

    #[test]
    fn test_particle_filter_normalize_zero_weights() {
        use crate::kalman::InitialState;

        let initial_state = InitialState::default();
        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-6; 15],
            ProcessNoise::default(),
            10,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Set all weights to zero
        for particle in &mut pf.particles {
            particle.weight = 0.0;
        }

        pf.normalize_weights();

        // Should reset to uniform
        let weight_sum: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-9);
        assert_approx_eq!(pf.particles[0].weight, 0.1, 1e-9);
    }

    // ========================================================================
    // Predict tests
    // ========================================================================

    #[test]
    fn test_particle_filter_predict() {
        use crate::kalman::InitialState;

        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-6; 15],
            ProcessNoise::default(),
            50,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        let initial_lat = pf.particles[0].nav_state.latitude;

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 9.81),
            gyro: Vector3::zeros(),
        };

        pf.predict(imu_data, 0.1);

        // Particles should have moved (due to velocity and process noise)
        let new_lat = pf.particles[0].nav_state.latitude;
        assert!((new_lat - initial_lat).abs() > 0.0);

        // Particles should be diverse (process noise added)
        let lats: Vec<f64> = pf.particles.iter().map(|p| p.nav_state.latitude).collect();
        let lat_variance = lats.iter().map(|&x| (x - lats[0]).powi(2)).sum::<f64>() / lats.len() as f64;
        assert!(lat_variance > 0.0);
    }

    // ========================================================================
    // Update and resampling tests
    // ========================================================================

    #[test]
    fn test_particle_filter_update() {
        use crate::kalman::InitialState;
        use crate::measurements::GPSPositionMeasurement;

        let initial_state = InitialState {
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: true,
            is_enu: true,
        };

        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-2; 15], // Larger initial uncertainty
            ProcessNoise::default(),
            100,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Measurement near initial state
        let measurement = GPSPositionMeasurement {
            latitude: 37.001,
            longitude: -122.001,
            altitude: 101.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
        };

        pf.update(&measurement);

        // Weights should sum to 1 after update
        let weight_sum: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-6);

        // Estimate should be reasonable (close to measurement)
        let estimate = pf.get_estimate();
        let lat_deg = estimate[0].to_degrees();
        let lon_deg = estimate[1].to_degrees();
        assert!((lat_deg - 37.0).abs() < 1.0); // Within 1 degree
        assert!((lon_deg - (-122.0)).abs() < 1.0); // Within 1 degree
    }

    #[test]
    fn test_particle_filter_resampling_trigger() {
        use crate::kalman::InitialState;
        use crate::measurements::GPSPositionMeasurement;

        let initial_state = InitialState::default();
        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-2; 15],
            ProcessNoise::default(),
            100,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Manually set weights to cause degeneracy
        for (i, particle) in pf.particles.iter_mut().enumerate() {
            if i == 0 {
                particle.weight = 0.9;
            } else {
                particle.weight = 0.1 / 99.0;
            }
        }

        let ess_before = pf.effective_particle_count();
        assert!(ess_before < 50.0); // Should be degenerate

        // Update with measurement (should trigger resampling)
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 10.0,
        };

        pf.update(&measurement);

        // After resampling, weights should be more uniform
        let ess_after = pf.effective_particle_count();
        assert!(ess_after > ess_before); // ESS should improve
    }

    #[test]
    fn test_systematic_resampling() {
        use crate::kalman::InitialState;

        let initial_state = InitialState::default();
        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-6; 15],
            ProcessNoise::default(),
            10,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Set non-uniform weights
        for (i, particle) in pf.particles.iter_mut().enumerate() {
            particle.weight = if i < 5 { 0.1 } else { 0.1 };
        }
        pf.particles[0].weight = 0.5;
        pf.normalize_weights();

        let pre_resample_lat = pf.particles[0].nav_state.latitude;

        pf.systematic_resample();

        // Weights should be uniform after resampling
        assert_approx_eq!(pf.particles[0].weight, 0.1, 1e-9);

        // Number of particles should be unchanged
        assert_eq!(pf.particles.len(), 10);

        // Particle with high weight should be duplicated
        let count_similar = pf
            .particles
            .iter()
            .filter(|p| (p.nav_state.latitude - pre_resample_lat).abs() < 1e-12)
            .count();
        assert!(count_similar > 1); // Should have multiple copies
    }

    // ========================================================================
    // State estimation tests
    // ========================================================================

    #[test]
    fn test_weighted_mean_estimate() {
        use crate::kalman::InitialState;

        let initial_state = InitialState::default();
        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-6; 15],
            ProcessNoise::default(),
            3,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Set known particle states and weights
        pf.particles[0].nav_state.altitude = 100.0;
        pf.particles[0].weight = 0.5;

        pf.particles[1].nav_state.altitude = 200.0;
        pf.particles[1].weight = 0.3;

        pf.particles[2].nav_state.altitude = 300.0;
        pf.particles[2].weight = 0.2;

        let estimate = pf.get_estimate();

        // Weighted mean: 100*0.5 + 200*0.3 + 300*0.2 = 50 + 60 + 60 = 170
        assert_approx_eq!(estimate[2], 170.0, 1e-9);
    }

    #[test]
    fn test_maximum_weight_estimate() {
        use crate::kalman::InitialState;

        let initial_state = InitialState::default();
        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-6; 15],
            ProcessNoise::default(),
            3,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::MaximumWeight,
            Some(42),
        );

        // Set known particle states and weights
        pf.particles[0].nav_state.altitude = 100.0;
        pf.particles[0].weight = 0.3;

        pf.particles[1].nav_state.altitude = 200.0;
        pf.particles[1].weight = 0.6; // Maximum

        pf.particles[2].nav_state.altitude = 300.0;
        pf.particles[2].weight = 0.1;

        let estimate = pf.get_estimate();

        // Should return particle with max weight (altitude = 200)
        assert_approx_eq!(estimate[2], 200.0, 1e-9);
    }

    #[test]
    fn test_empirical_covariance() {
        use crate::kalman::InitialState;

        let initial_state = InitialState::default();
        let pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-3; 15],
            ProcessNoise::default(),
            100,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        let cov = pf.get_certainty();

        // Covariance should be positive semi-definite
        assert_eq!(cov.nrows(), 15);
        assert_eq!(cov.ncols(), 15);

        // Diagonal elements should be positive
        for i in 0..15 {
            assert!(cov[(i, i)] >= 0.0);
        }

        // Should be symmetric
        for i in 0..15 {
            for j in 0..15 {
                assert_approx_eq!(cov[(i, j)], cov[(j, i)], 1e-9);
            }
        }
    }

    // ========================================================================
    // Likelihood computation tests
    // ========================================================================

    #[test]
    fn test_compute_likelihood_zero_innovation() {
        let innovation = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let meas_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));

        let likelihood = ParticleFilter::compute_likelihood(&innovation, &meas_cov);

        // Zero innovation -> likelihood = exp(0) = 1.0
        assert_approx_eq!(likelihood, 1.0, 1e-9);
    }

    #[test]
    fn test_compute_likelihood_large_innovation() {
        let innovation = DVector::from_vec(vec![10.0, 10.0, 10.0]);
        let meas_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));

        let likelihood = ParticleFilter::compute_likelihood(&innovation, &meas_cov);

        // Large innovation -> small likelihood
        assert!(likelihood < 0.01);
        assert!(likelihood > 0.0);
    }

    // ========================================================================
    // Integration tests
    // ========================================================================

    #[test]
    fn test_particle_filter_full_cycle() {
        use crate::kalman::InitialState;
        use crate::measurements::GPSPositionMeasurement;

        let initial_state = InitialState {
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: true,
            is_enu: true,
        };

        let mut pf = ParticleFilter::new(
            initial_state,
            vec![0.0; 6],
            None,
            vec![1e-3; 15],
            ProcessNoise::default(),
            100,
            VerticalChannelMode::Simplified,
            ResamplingStrategy::Systematic,
            AveragingStrategy::WeightedMean,
            Some(42),
        );

        // Simulate several predict-update cycles
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, 9.81),
            gyro: Vector3::zeros(),
        };

        for i in 0..10 {
            // Predict
            pf.predict(imu_data, 0.1);

            // Update every 5 steps
            if i % 5 == 0 {
                let measurement = GPSPositionMeasurement {
                    latitude: 37.0 + (i as f64) * 0.0001,
                    longitude: -122.0,
                    altitude: 100.0,
                    horizontal_noise_std: 5.0,
                    vertical_noise_std: 10.0,
                };
                pf.update(&measurement);
            }
        }

        // Filter should still be functional
        let estimate = pf.get_estimate();
        assert_eq!(estimate.len(), 15);

        // State should be reasonable (particles haven't diverged too much)
        let lat_deg = estimate[0].to_degrees();
        let lon_deg = estimate[1].to_degrees();
        assert!(lat_deg > 36.0 && lat_deg < 38.0); // Should stay near initial
        assert!(lon_deg > -123.0 && lon_deg < -121.0);

        // Weights should sum to 1
        let weight_sum: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert_approx_eq!(weight_sum, 1.0, 1e-6);

        // Covariance should be reasonable
        let cov = pf.get_certainty();
        assert!(cov[(0, 0)] > 0.0); // Should have some uncertainty
    }
}
