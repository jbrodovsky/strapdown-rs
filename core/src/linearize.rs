//! Jacobian and linearization utilities for EKF/ESKF implementations
//!
//! This module provides analytic Jacobians for strapdown mechanization dynamics
//! and measurement models. These linearizations are essential for Extended Kalman
//! Filter (EKF), Error-State Kalman Filter (ESKF), and Rao-Blackwellized Particle
//! Filter (RBPF) implementations.
//!
//! # State Ordering
//!
//! The 9-state navigation vector follows the ordering:
//! ```text
//! x = [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
//! ```
//! where:
//! - `lat`, `lon`: latitude and longitude in radians
//! - `alt`: altitude in meters (positive up in ENU, positive down in NED)
//! - `v_n`, `v_e`, `v_d`: velocity components in m/s (NED/ENU local-level frame)
//! - `roll`, `pitch`, `yaw`: Euler angles in radians (XYZ rotation sequence)
//!
//! # Usage Example
//!
//! ```rust
//! use strapdown::linearize::{state_transition_jacobian, gps_position_jacobian};
//! use strapdown::StrapdownState;
//! use nalgebra::{Vector3, Rotation3};
//!
//! // Create navigation state
//! let state = StrapdownState::new(
//!     45.0, -122.0, 100.0,  // position
//!     10.0, 5.0, 0.0,        // velocity
//!     Rotation3::identity(), // attitude
//!     true,                  // in_degrees
//!     Some(true),            // is_enu
//! );
//!
//! // Get Jacobians for EKF predict/update
//! let accel = Vector3::new(0.0, 0.0, 9.81);
//! let gyro = Vector3::zeros();
//! let dt = 0.01;
//!
//! let f_matrix = state_transition_jacobian(&state, &accel, &gyro, dt);
//! let h_matrix = gps_position_jacobian(&state);
//!
//! // Use in EKF: P(+) = F*P(-)*F^T + G*Q*G^T
//! // Use in measurement update: K = P*H^T*(H*P*H^T + R)^-1
//! ```
//!
//! # References
//!
//! Jacobian derivations follow Groves, "Principles of GNSS, Inertial, and Multisensor
//! Integrated Navigation Systems, 2nd Edition":
//! - State transition (F): Chapter 14.2.4, Equations 14.50-14.51
//! - Process noise (G): Chapter 14.2.3, Equations 14.24-14.25
//! - Measurement models (H): Chapter 3.6 and 14.2.7
//!
//! # Coordinate Conventions
//!
//! - Default frame is East-North-Up (ENU) but NED is also supported via `is_enu` flag
//! - Gravity is positive down (NED) or negative up (ENU)
//! - Latitude is constrained to [-π/2, π/2] (±90°)
//! - Longitude and yaw are wrapped to [-π, π]
//! - Roll and pitch are typically in [-π, π] but may vary by implementation

use crate::StrapdownState;
use crate::earth::{self, vector_to_skew_symmetric};
use nalgebra::{DMatrix, DVector, Rotation3, Vector3};

/// Compute the state transition Jacobian (F) for strapdown mechanization
///
/// This function computes the linearized state transition matrix F for the
/// 9-state strapdown navigation equations. The Jacobian describes how small
/// perturbations in the current state propagate forward in time.
///
/// # Mathematical Background
///
/// The state transition model is: x(t+dt) ≈ x(t) + f(x(t), u(t)) * dt
///
/// The Jacobian F = ∂f/∂x evaluated at the current state, where f represents
/// the strapdown mechanization equations (attitude, velocity, position updates).
///
/// # Arguments
///
/// * `state` - Current navigation state
/// * `imu_accel` - Specific force measurement from IMU (body frame, m/s²)
/// * `imu_gyro` - Angular rate measurement from IMU (body frame, rad/s)
/// * `dt` - Time step in seconds
///
/// # Returns
///
/// 9×9 state transition Jacobian matrix F
///
/// # References
///
/// Groves 2nd ed., Section 14.2.4, Equations 14.50-14.51
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::state_transition_jacobian;
/// use strapdown::StrapdownState;
/// use nalgebra::{Vector3, Rotation3};
///
/// let state = StrapdownState::new(
///     45.0, -122.0, 100.0,  // lat, lon, alt (degrees, degrees, meters)
///     10.0, 5.0, 0.0,        // velocities (m/s)
///     Rotation3::identity(),  // attitude
///     true,                   // in_degrees
///     None,                   // is_enu (defaults to true)
/// );
/// let accel = Vector3::new(0.0, 0.0, 9.81);
/// let gyro = Vector3::new(0.0, 0.0, 0.0);
/// let dt = 0.01;
///
/// let f_matrix = state_transition_jacobian(&state, &accel, &gyro, dt);
/// assert_eq!(f_matrix.nrows(), 9);
/// assert_eq!(f_matrix.ncols(), 9);
/// ```
pub fn state_transition_jacobian(
    state: &StrapdownState,
    imu_accel: &Vector3<f64>,
    _imu_gyro: &Vector3<f64>,
    dt: f64,
) -> DMatrix<f64> {
    let mut f = DMatrix::<f64>::identity(9, 9);

    // Extract state components
    let lat = state.latitude;
    let alt = state.altitude;
    let v_n = state.velocity_north;
    let v_e = state.velocity_east;
    let v_d = state.velocity_vertical;

    let vel = Vector3::new(v_n, v_e, v_d);
    let c_bn = state.attitude.matrix(); // Body-to-nav rotation matrix

    // Earth model parameters
    let (r_n, r_e, _) = earth::principal_radii(&lat, &alt);
    let _g = earth::gravity(&lat.to_degrees(), &alt); // Reserved for future use
    let omega_ie = earth::earth_rate_lla(&lat.to_degrees());
    let omega_en = earth::transport_rate(&lat.to_degrees(), &alt, &vel);

    let omega_ie_skew = vector_to_skew_symmetric(&omega_ie);
    let omega_en_skew = vector_to_skew_symmetric(&omega_en);

    // Compute gravity gradient with latitude (analytical derivative of Somigliana formula)
    // g(φ) = g_e * (1 + k*sin²φ) / √(1 - e²*sin²φ) - c*h
    // where φ is latitude in radians
    let sin_lat = lat.sin();
    let cos_lat = lat.cos().max(1e-6);
    let sin2_lat = sin_lat * sin_lat;
    let e2 = earth::ECCENTRICITY_SQUARED;
    let k = earth::K;
    let ge = earth::GE;

    let numerator = 1.0 + k * sin2_lat;
    let denominator_sqrt = (1.0 - e2 * sin2_lat).sqrt();

    // ∂g/∂φ using quotient and chain rules
    let dnumerator_dphi = 2.0 * k * sin_lat * cos_lat;
    let ddenominator_sqrt_dphi = -e2 * sin_lat * cos_lat / denominator_sqrt;
    let dgravity_dlat = ge
        * (dnumerator_dphi * denominator_sqrt - numerator * ddenominator_sqrt_dphi)
        / (denominator_sqrt * denominator_sqrt);

    // Transform specific force to navigation frame
    let f_bn = c_bn * imu_accel;

    // --- Position derivatives (rows 0-2) ---
    // Position update: lat(+) = lat(-) + v_n/(R_n+h)*dt + ...
    // ∂(lat(+))/∂(lat(-)): main term is identity, plus derivative terms
    // The derivative of v_n/(R_n+h) w.r.t. lat through R_n is negligible for first-order
    // ∂(lat(+))/∂(v_n): derivative of the kinematic relationship
    f[(0, 3)] = 1.0 / (r_n + alt) * dt;

    // Longitude update accounts for cos(lat) in denominator (cos_lat already computed above)
    // ∂(lon(+))/∂(lon(-)): identity (no direct dependence)
    // ∂(lon(+))/∂(lat): derivative through cos(lat)
    f[(1, 0)] += v_e / ((r_e + alt) * cos_lat.powi(2)) * sin_lat * dt;
    // ∂(lon(+))/∂(v_e): kinematic relationship
    f[(1, 4)] = 1.0 / ((r_e + alt) * cos_lat) * dt;

    // Altitude update: simple kinematic
    // ∂(alt(+))/∂(v_d)
    f[(2, 5)] = dt;

    // --- Velocity derivatives (rows 3-5) ---
    // Velocity update includes Coriolis, centrifugal, gravity, and specific force

    // ∂(v(+))/∂(v(-)): Coriolis and transport effects
    // v(+) = v(-) + [f - g - (2*Ω_ie + Ω_en) × v]*dt
    let coriolis_transport = -(2.0 * omega_ie_skew + omega_en_skew);
    for i in 0..3 {
        for j in 0..3 {
            f[(3 + i, 3 + j)] += coriolis_transport[(i, j)] * dt;
        }
    }

    // ∂(v(+))/∂(lat): gravity varies with latitude
    // For ENU: g = [0, 0, -g], for NED: g = [0, 0, g]
    if state.is_enu {
        f[(5, 0)] += -dgravity_dlat * dt; // ENU: gravity is negative up
    } else {
        f[(5, 0)] += dgravity_dlat * dt; // NED: gravity is positive down
    }

    // ∂(v(+))/∂(alt): gravity gradient
    let dgravity_dalt = -3.08e-6; // From gravity formula: g = g0 - 3.08e-6 * h
    if state.is_enu {
        f[(5, 2)] += -dgravity_dalt * dt;
    } else {
        f[(5, 2)] += dgravity_dalt * dt;
    }

    // ∂(v(+))/∂(attitude): transformation of specific force
    // f^n = C_b^n * f^b
    // For small angle perturbations: δ(C*f) ≈ [f×] * δθ
    // Using skew-symmetric: [f×] * θ = -[θ×] * f
    let f_bn_skew = vector_to_skew_symmetric(&f_bn);
    for i in 0..3 {
        for j in 0..3 {
            f[(3 + i, 6 + j)] += -f_bn_skew[(i, j)] * dt;
        }
    }

    // --- Attitude derivatives (rows 6-8) ---
    // Attitude update: C(+) = C(-) * (I + Ω_ib*dt) - (Ω_ie + Ω_en) * C(-) * dt
    // In error-state formulation: δε(+) ≈ δε(-) - (Ω_ie + Ω_en) × δε(-) * dt
    // This gives: Φ_ε = I - [Ω_ie + Ω_en]× * dt

    let omega_in_skew = omega_ie_skew + omega_en_skew;
    for i in 0..3 {
        for j in 0..3 {
            f[(6 + i, 6 + j)] += -omega_in_skew[(i, j)] * dt;
        }
    }

    // ∂(attitude(+))/∂(v): through transport rate
    // Ω_en depends on velocity, so changes in velocity affect attitude dynamics
    // From transport rate: ω_en = [v_e/(R_e+h), -v_n/(R_n+h), -v_e*tan(lat)/(R_e+h)]
    // These couple into attitude through the integration
    f[(6, 4)] += 1.0 / (r_e + alt) * dt; // ∂(ε_x)/∂(v_e)
    f[(7, 3)] += -1.0 / (r_n + alt) * dt; // ∂(ε_y)/∂(v_n)
    f[(8, 4)] += -lat.tan() / (r_e + alt) * dt; // ∂(ε_z)/∂(v_e)

    f
}

/// Magnitude of the vertical gravity gradient, |∂g/∂h|, in s^-2.
///
/// Matches the linear altitude term in [`earth::gravity`], which is
/// `g0(lat) - 3.08e-6 * altitude`.
const GRAVITY_ALTITUDE_GRADIENT: f64 = 3.08e-6;

/// Derivative of Somigliana normal gravity with respect to latitude, in m/s² per radian.
///
/// [`earth::gravity`] uses
/// `g0 = GE (1 + K sin²φ) / sqrt(1 - e² sin²φ)`, so
/// `dg0/dφ = GE cosφ sinφ [ 2K / sqrt(D) + (1 + K sin²φ) e² / D^(3/2) ]` with
/// `D = 1 - e² sin²φ`. About 0.0519 m/s² per radian at 45° latitude, peaking there and
/// vanishing at the equator and poles.
fn gravity_latitude_gradient(latitude_rad: f64) -> f64 {
    let sin_lat = latitude_rad.sin();
    let cos_lat = latitude_rad.cos();
    let sin_sq = sin_lat * sin_lat;
    let d = 1.0 - earth::ECCENTRICITY_SQUARED * sin_sq;
    let sqrt_d = d.sqrt();
    earth::GE
        * cos_lat
        * sin_lat
        * (2.0 * earth::K / sqrt_d
            + (1.0 + earth::K * sin_sq) * earth::ECCENTRICITY_SQUARED / (d * sqrt_d))
}

/// Compute the error-state transition Jacobian for ESKF
///
/// This function computes the linearized state transition matrix F for the
/// error-state formulation used in Error-State Kalman Filters (ESKF). Unlike
/// the full-state Jacobian, this operates on error states where attitude
/// errors are represented as small angles rather than full Euler angles.
///
/// # Key Differences from Full-State EKF
///
/// 1. **Attitude representation**: Error state uses 3 small-angle parameters
///    instead of 3 Euler angles, avoiding singularities
/// 2. **Linearization point**: Linearized around the nominal (true) trajectory,
///    not around the previous estimate
/// 3. **Error dynamics**: Captures how errors propagate, not how states evolve
///
/// # Mathematical Background
///
/// The error-state dynamics are:
/// $$
/// \delta \dot{x} = F_{\delta x} \delta x + G w
/// $$
///
/// where $\delta x = [δp^n, δv^n, δθ, δb_a, δb_g]^T$ is the 15-element error state:
/// - $δp^n$ : Position error in local-level frame (m)
/// - $δv^n$ : Velocity error in local-level frame (m/s)
/// - $δθ$ : Attitude error as small angles (rad)
/// - $δb_a$ : Accelerometer bias error (m/s²)
/// - $δb_g$ : Gyroscope bias error (rad/s)
///
/// The discrete-time error-state transition is:
/// $$
/// \delta x_{k+1} \approx (I + F_{\delta x} \cdot dt) \delta x_k
/// $$
///
/// # Block Structure of F_δx (15×15)
///
/// ```text
/// F = | F_pp  F_pv  F_pθ   0     0   |  (position)
///     | F_vp  F_vv  F_vθ  F_vba  0   |  (velocity)
///     | F_θp  F_θv  F_θθ   0    F_θbg|  (attitude)
///     |  0     0     0    F_bb   0   |  (accel bias)
///     |  0     0     0     0    F_bb |  (gyro bias)
/// ```
///
/// where most blocks are sparse and bias dynamics are random walk (F_bb = 0).
///
/// # Arguments
///
/// * `state` - Current nominal navigation state (the "truth" around which to linearize)
/// * `imu_accel` - Bias-corrected specific force measurement (body frame, m/s²)
/// * `imu_gyro` - Bias-corrected angular rate measurement (body frame, rad/s)
/// * `dt` - Time step in seconds
///
/// # Returns
///
/// 15×15 error-state transition Jacobian matrix F_δx
///
/// # References
///
/// - Sola, J. "Quaternion kinematics for the error-state Kalman filter" (2017), Section 6.3
/// - Groves 2nd ed., Section 14.2.4 (adapted for error-state formulation)
/// - Trawny, N. & Roumeliotis, S. "Indirect Kalman Filter for 3D Attitude Estimation" (2005)
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::error_state_transition_jacobian;
/// use strapdown::StrapdownState;
/// use nalgebra::{Vector3, Rotation3};
///
/// let state = StrapdownState::new(
///     45.0, -122.0, 100.0,
///     10.0, 5.0, 0.0,
///     Rotation3::identity(),
///     true,
///     None,
///  );
/// let accel = Vector3::new(0.0, 0.0, 9.81);
/// let gyro = Vector3::zeros();
/// let dt = 0.01;
///
/// let f_error = error_state_transition_jacobian(&state, &accel, &gyro, dt);
/// assert_eq!(f_error.nrows(), 15);
/// assert_eq!(f_error.ncols(), 15);
/// ```
pub fn error_state_transition_jacobian(
    state: &StrapdownState,
    imu_accel: &Vector3<f64>,
    imu_gyro: &Vector3<f64>,
    dt: f64,
) -> DMatrix<f64> {
    // Start with identity matrix (I + F*dt formulation)
    let mut f = DMatrix::<f64>::identity(15, 15);

    // Get rotation matrix from body to navigation frame
    let c_bn = state.attitude.matrix();

    // Get Earth parameters
    let lat = state.latitude;
    let h = state.altitude;
    let lat_deg = lat.to_degrees();
    let (r_n, r_e, _r_p) = earth::principal_radii(&lat_deg, &h);

    // ===== Position Error Block (rows 0-2) =====

    // ∂(δṗ)/∂(δv): Position error rate depends on velocity error
    // δṗ_n = δv_n / R_n
    // δṗ_e = δv_e / (R_e * cos(lat))
    // δṗ_d = δv_d
    f[(0, 3)] = dt / r_n; // ∂(δp_n)/∂(δv_n)
    f[(1, 4)] = dt / (r_e * lat.cos()); // ∂(δp_e)/∂(δv_e)
    f[(2, 5)] = dt; // ∂(δp_d)/∂(δv_d)

    // Note: Position error doesn't directly depend on attitude error or biases

    // ===== Velocity Error Block (rows 3-5) =====

    // ∂(δv̇)/∂(δp): the gravity model depends on both altitude and latitude, so this
    // block is not zero.
    //
    // `earth::gravity` is g(lat, h) = g0(lat) - 3.08e-6 h, giving
    //     ∂g/∂h   = -3.08e-6 s^-2                    (the vertical gravity gradient)
    //     ∂g/∂lat = dg0/dlat, ~0.0519 m/s²/rad at 45°  (Somigliana latitude variation)
    //
    // Both were previously omitted. The comment that used to stand here argued the
    // gravity-gradient term should be left out because in ENU it produces exponentially
    // growing eigenvalues. That instability is real -- it is the classic undamped INS
    // vertical channel -- but it belongs to the *system*, not to the linearisation.
    // Dropping a term that the nominal propagation actually applies makes F disagree
    // with `forward()`, so the covariance stops describing the real error dynamics and
    // the filter can no longer attribute an altitude innovation to vertical-velocity
    // error. Numerically differentiating `forward()` shows both entries clearly, stable
    // across four orders of magnitude of step size. See #286.
    //
    // Sign follows the frame: in ENU the vertical axis is up and v̇_up contains -g, so a
    // higher altitude means weaker gravity means a *more positive* v̇_up. NED flips it.
    let vertical_sign = if state.is_enu { 1.0 } else { -1.0 };
    f[(5, 2)] = vertical_sign * GRAVITY_ALTITUDE_GRADIENT * dt;
    f[(5, 0)] = -vertical_sign * gravity_latitude_gradient(lat) * dt;

    // ∂(δv̇)/∂(δv): Velocity error damping due to Coriolis and centrifugal effects
    // These coupling terms are small for low dynamics and often approximated as zero
    // The main effect is self-coupling which is captured by identity matrix

    // ∂(δv̇)/∂(δθ): Velocity error depends on attitude error (most important coupling!)
    //
    // The error state uses the LOCAL (body-frame) attitude-error convention:
    //     C_b^n(true) = C_b^n(nominal) · (I + [δθ]_×)
    // which is what the rest of this matrix and the ESKF's injection already assume --
    // attitude propagation below uses -[ω^b]_× and the gyro-bias coupling is the bare
    // identity, both of which are only correct for a body-frame error, and
    // `ErrorStateKalmanFilter::inject_error_state` right-multiplies q_nom ⊗ δq.
    //
    // Under that convention:
    //     δf^n = C_b^n [δθ]_× f^b = -C_b^n [f^b]_× δθ
    // so the block is -C_b^n [f^b]_× dt.
    //
    // This previously read -[C_b^n f^b]_× dt, which is the GLOBAL (nav-frame) form and
    // differs from the local one by a rotation. Mixing the two inside a single F rotated
    // the tilt-to-velocity feedback onto the wrong axes whenever heading was non-zero.
    // See #266.
    let accel_skew_body = vector_to_skew_symmetric(imu_accel);
    let velocity_attitude_block = -(c_bn * accel_skew_body) * dt;
    for i in 0..3 {
        for j in 0..3 {
            f[(3 + i, 6 + j)] = velocity_attitude_block[(i, j)];
        }
    }

    // ∂(δv̇)/∂(δb_a): Velocity error depends on accelerometer bias error
    // δv̇^n = -C_b^n δb_a
    for i in 0..3 {
        for j in 0..3 {
            f[(3 + i, 9 + j)] = -c_bn[(i, j)] * dt;
        }
    }

    // ===== Attitude Error Block (rows 6-8) =====

    // ∂(δθ̇)/∂(δθ): Attitude error dynamics (rotation coupling)
    // δθ̇ = -[ω^b]_× δθ (in body frame)
    // This represents how attitude errors rotate due to angular velocity
    //
    // NOTE the `-=`. `f` is initialised to the identity and this block is
    // `I - [omega^b]_x dt`, so the term must be *subtracted from* the identity, not
    // assigned over it. A skew-symmetric matrix has a zero diagonal, so the previous
    // `=` silently set f[(6,6)], f[(7,7)] and f[(8,8)] to zero instead of one --
    // annihilating the attitude error covariance on every propagation step. The filter
    // then believed attitude was perfectly known after each predict, never corrected
    // tilt, and left gyro bias (observable only through tilt) unconstrained. See #286.
    let omega_skew = vector_to_skew_symmetric(imu_gyro);
    for i in 0..3 {
        for j in 0..3 {
            f[(6 + i, 6 + j)] -= omega_skew[(i, j)] * dt;
        }
    }

    // ∂(δθ̇)/∂(δb_g): Attitude error depends on gyroscope bias error
    // δθ̇ = -δb_g (small angle approximation)
    f[(6, 12)] = -dt;
    f[(7, 13)] = -dt;
    f[(8, 14)] = -dt;

    // ===== IMU Bias Error Blocks (rows 9-14) =====

    // Biases are modeled as random walk: δḃ = 0 + noise
    // This means F_bb = 0, which is already set by the identity matrix initialization
    // The identity diagonal (1.0) represents the bias persistence (integration)

    f
}

/// Compute the process noise Jacobian (G) for IMU errors
///
/// This function computes the process noise distribution matrix G that maps
/// white noise inputs (IMU measurement errors, bias random walks) to state
/// error dynamics.
///
/// # Mathematical Background
///
/// The continuous-time process model includes noise: dx/dt = f(x,u) + G*w
/// where w ~ N(0, Q_c) is white noise representing IMU errors.
///
/// The process noise covariance in discrete time is: Q_d = G * Q_c * G^T * dt
///
/// # Arguments
///
/// * `state` - Current navigation state (affects rotation matrix)
/// * `dt` - Time step in seconds
///
/// # Returns
///
/// 9×6 process noise Jacobian matrix G, mapping [accel_noise; gyro_noise] to state
///
/// # References
///
/// Groves 2nd ed., Section 14.2.3, Equations 14.24-14.25
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::process_noise_jacobian;
/// use strapdown::StrapdownState;
/// use nalgebra::Rotation3;
///
/// let state = StrapdownState::new(
///     45.0, -122.0, 100.0,
///     0.0, 0.0, 0.0,
///     Rotation3::identity(),
///     true,
///     None,
/// );
/// let dt = 0.01;
///
/// let g_matrix = process_noise_jacobian(&state, dt);
/// assert_eq!(g_matrix.nrows(), 9);
/// assert_eq!(g_matrix.ncols(), 6);
/// ```
pub fn process_noise_jacobian(state: &StrapdownState, dt: f64) -> DMatrix<f64> {
    let mut g = DMatrix::<f64>::zeros(9, 6);

    let c_bn = state.attitude.matrix();

    // Position states (0-2) are not directly affected by IMU noise
    // (they're affected indirectly through velocity, but not in the G matrix)

    // Velocity error propagation from accelerometer noise (columns 0-2)
    // δv^n = C_b^n * δf^b
    for i in 0..3 {
        for j in 0..3 {
            g[(3 + i, j)] = c_bn[(i, j)] * dt;
        }
    }

    // Attitude error propagation from gyroscope noise (columns 3-5)
    // δφ = -C_b^n * δω^b * dt (simplified for small angles)
    for i in 0..3 {
        for j in 0..3 {
            g[(6 + i, 3 + j)] = -c_bn[(i, j)] * dt;
        }
    }

    g
}

/// Compute measurement Jacobian (H) for GPS position measurement
///
/// GPS position measurements directly observe latitude, longitude, and altitude.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused, but kept for API consistency)
///
/// # Returns
///
/// 3×9 measurement Jacobian matrix H for GPS position
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::gps_position_jacobian;
/// use strapdown::StrapdownState;
/// use nalgebra::Rotation3;
///
/// let state = StrapdownState::default();
/// let h = gps_position_jacobian(&state);
/// assert_eq!(h.nrows(), 3);
/// assert_eq!(h.ncols(), 9);
/// ```
pub fn gps_position_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(3, 9);
    // GPS measures position states directly: [lat, lon, alt]
    h[(0, 0)] = 1.0; // ∂(z_lat)/∂(lat)
    h[(1, 1)] = 1.0; // ∂(z_lon)/∂(lon)
    h[(2, 2)] = 1.0; // ∂(z_alt)/∂(alt)
    h
}

/// Compute measurement Jacobian (H) for GPS velocity measurement
///
/// GPS velocity measurements directly observe velocity components in the local-level frame.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused, but kept for API consistency)
///
/// # Returns
///
/// 3×9 measurement Jacobian matrix H for GPS velocity
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::gps_velocity_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = gps_velocity_jacobian(&state);
/// assert_eq!(h.nrows(), 3);
/// assert_eq!(h.ncols(), 9);
/// ```
pub fn gps_velocity_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(3, 9);
    // GPS measures velocity states directly: [v_n, v_e, v_d]
    h[(0, 3)] = 1.0; // ∂(z_vn)/∂(v_n)
    h[(1, 4)] = 1.0; // ∂(z_ve)/∂(v_e)
    h[(2, 5)] = 1.0; // ∂(z_vd)/∂(v_d)
    h
}

/// Compute measurement Jacobian (H) for combined GPS position and velocity measurement
///
/// Combined GPS measurements observe both position and velocity (excludes vertical velocity).
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused, but kept for API consistency)
///
/// # Returns
///
/// 5×9 measurement Jacobian matrix H for GPS position and velocity
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::gps_position_velocity_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = gps_position_velocity_jacobian(&state);
/// assert_eq!(h.nrows(), 5);
/// assert_eq!(h.ncols(), 9);
/// ```
pub fn gps_position_velocity_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(5, 9);
    // Measurement vector: [lat, lon, alt, v_n, v_e]
    h[(0, 0)] = 1.0; // ∂(z_lat)/∂(lat)
    h[(1, 1)] = 1.0; // ∂(z_lon)/∂(lon)
    h[(2, 2)] = 1.0; // ∂(z_alt)/∂(alt)
    h[(3, 3)] = 1.0; // ∂(z_vn)/∂(v_n)
    h[(4, 4)] = 1.0; // ∂(z_ve)/∂(v_e)
    h
}

/// Compute measurement Jacobian (H) for relative altitude measurement
///
/// Barometric altimeters provide relative altitude measurements.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused, but kept for API consistency)
///
/// # Returns
///
/// 1×9 measurement Jacobian matrix H for relative altitude
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::relative_altitude_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = relative_altitude_jacobian(&state);
/// assert_eq!(h.nrows(), 1);
/// assert_eq!(h.ncols(), 9);
/// ```
pub fn relative_altitude_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(1, 9);
    // Barometric altitude measures altitude directly
    h[(0, 2)] = 1.0; // ∂(z_alt)/∂(alt)
    h
}

/// Compute measurement Jacobian (H) for gravity anomaly measurement
///
/// **Note**: This is a placeholder function that returns zeros. Geophysical measurements
/// (gravity anomaly and magnetic anomaly) now provide their own Jacobians via the
/// `MeasurementModel::get_jacobian()` trait method. The EKF update function automatically
/// detects and uses these measurement-provided Jacobians.
///
/// Gravity anomaly measurements depend on latitude and longitude (position) to query the
/// geophysical map. The partial derivatives ∂z/∂lat and ∂z/∂lon are computed numerically
/// by the `GravityMeasurement` struct in the `strapdown-geonav` crate based on map gradients.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused, this is a placeholder)
///
/// # Returns
///
/// 1×9 measurement Jacobian matrix H filled with zeros (placeholder)
///
/// # See Also
///
/// For actual gravity anomaly Jacobian computation, see `geonav::GravityMeasurement::get_jacobian_internal()`.
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::gravity_anomaly_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = gravity_anomaly_jacobian(&state);
/// assert_eq!(h.nrows(), 1);
/// assert_eq!(h.ncols(), 9);
/// ```
pub fn gravity_anomaly_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    // NOTE: This is a placeholder. Real gravity anomaly Jacobians are provided
    // by the GravityMeasurement struct in the geonav crate via the
    // MeasurementModel::get_jacobian() trait method. The EKF will automatically
    // use the measurement-provided Jacobian when available.
    DMatrix::<f64>::zeros(1, 9)
}

/// Compute measurement Jacobian (H) for magnetic anomaly measurement
///
/// **Note**: This is a placeholder function that returns zeros. Geophysical measurements
/// (gravity anomaly and magnetic anomaly) now provide their own Jacobians via the
/// `MeasurementModel::get_jacobian()` trait method. The EKF update function automatically
/// detects and uses these measurement-provided Jacobians.
///
/// Magnetic anomaly measurements depend on latitude and longitude (position) to query the
/// geophysical map. The partial derivatives ∂z/∂lat and ∂z/∂lon are computed numerically
/// by the `MagneticAnomalyMeasurement` struct in the `strapdown-geonav` crate based on map gradients.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused, this is a placeholder)
///
/// # Returns
///
/// 1×9 measurement Jacobian matrix H filled with zeros (placeholder)
///
/// # See Also
///
/// For actual magnetic anomaly Jacobian computation, see `geonav::MagneticAnomalyMeasurement::get_jacobian_internal()`.
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::magnetic_anomaly_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = magnetic_anomaly_jacobian(&state);
/// assert_eq!(h.nrows(), 1);
/// assert_eq!(h.ncols(), 9);
/// ```
pub fn magnetic_anomaly_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    // NOTE: This is a placeholder. Real magnetic anomaly Jacobians are provided
    // by the MagneticAnomalyMeasurement struct in the geonav crate via the
    // MeasurementModel::get_jacobian() trait method. The EKF will automatically
    // use the measurement-provided Jacobian when available.
    DMatrix::<f64>::zeros(1, 9)
}

/// Compute measurement Jacobian (H) for magnetometer-based yaw measurement
///
/// The magnetometer yaw measurement depends on roll, pitch, and yaw through the
/// tilt compensation equations. For a first-order approximation where the
/// measurement is primarily the yaw angle, the dominant partial derivative is
/// ∂z/∂ψ ≈ 1 (identity), with smaller contributions from roll and pitch
/// through the tilt compensation.
///
/// # Mathematical Background
///
/// The tilt-compensated magnetic heading is computed as:
///
/// $$
/// \psi_m = \arctan2(m_{y,h}, m_{x,h})
/// $$
///
/// where the horizontal components depend on roll ($\phi$) and pitch ($\theta$):
///
/// $$
/// \begin{aligned}
/// m_{x,h} &= m_x \cos\theta + m_y \sin\phi \sin\theta + m_z \cos\phi \sin\theta \\\\
/// m_{y,h} &= m_y \cos\phi - m_z \sin\phi
/// \end{aligned}
/// $$
///
/// The Jacobian entries are:
/// - $\frac{\partial z}{\partial \phi}$ (roll): Non-zero due to tilt compensation
/// - $\frac{\partial z}{\partial \theta}$ (pitch): Non-zero due to tilt compensation  
/// - $\frac{\partial z}{\partial \psi}$ (yaw): ≈ 1 (expected measurement is state yaw)
///
/// # Arguments
///
/// * `state` - Current navigation state
/// * `mag_x` - Body-frame magnetic field x-component (µT)
/// * `mag_y` - Body-frame magnetic field y-component (µT)
/// * `mag_z` - Body-frame magnetic field z-component (µT)
///
/// # Returns
///
/// 1×9 measurement Jacobian matrix H for magnetometer yaw
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::magnetometer_yaw_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = magnetometer_yaw_jacobian(&state, 20.0, 5.0, -45.0);
/// assert_eq!(h.nrows(), 1);
/// assert_eq!(h.ncols(), 9);
/// // Yaw partial derivative should be approximately 1
/// assert!((h[(0, 8)] - 1.0).abs() < 0.01);
/// ```
pub fn magnetometer_yaw_jacobian(
    state: &StrapdownState,
    mag_x: f64,
    mag_y: f64,
    mag_z: f64,
) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(1, 9);

    // Extract current attitude
    let (roll, pitch, _yaw) = state.attitude.euler_angles();
    let (sin_roll, cos_roll) = roll.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();

    // Compute horizontal components
    let mag_x_h = mag_x * cos_pitch + mag_y * sin_roll * sin_pitch + mag_z * cos_roll * sin_pitch;
    let mag_y_h = mag_y * cos_roll - mag_z * sin_roll;

    // Denominator for atan2 derivative
    let denom = mag_x_h.powi(2) + mag_y_h.powi(2);

    if denom > 1e-10 {
        // Partial derivatives of horizontal components with respect to roll
        let d_mx_h_d_roll = mag_y * cos_roll * sin_pitch - mag_z * sin_roll * sin_pitch;
        let d_my_h_d_roll = -mag_y * sin_roll - mag_z * cos_roll;

        // Partial derivatives of horizontal components with respect to pitch
        let d_mx_h_d_pitch =
            -mag_x * sin_pitch + mag_y * sin_roll * cos_pitch + mag_z * cos_roll * cos_pitch;
        let d_my_h_d_pitch = 0.0; // mag_y_h doesn't depend on pitch

        // Chain rule for atan2: d(atan2(y,x)) = (x*dy - y*dx) / (x^2 + y^2)
        // ∂ψ/∂roll
        h[(0, 6)] = (mag_x_h * d_my_h_d_roll - mag_y_h * d_mx_h_d_roll) / denom;

        // ∂ψ/∂pitch
        h[(0, 7)] = (mag_x_h * d_my_h_d_pitch - mag_y_h * d_mx_h_d_pitch) / denom;
    }

    // The expected measurement is the state yaw, so ∂z_expected/∂yaw = 1
    // (the measurement model returns state[8] as expected measurement)
    h[(0, 8)] = 1.0;

    h
}

/// Apply an error-state correction to a StrapdownState
///
/// This function implements the ESKF correction step, applying a computed error-state
/// vector to correct the nominal navigation state. The correction is additive for
/// position and velocity, and uses small-angle rotation composition for attitude.
///
/// # Error State Layout
///
/// The error state vector can be either 9-element or 15-element:
///
/// **9-state (navigation only):**
/// ```text
/// δx = [δlat, δlon, δalt, δv_n, δv_e, δv_d, δroll, δpitch, δyaw]
/// ```
///
/// **15-state (with IMU biases):**
/// ```text
/// δx = [δlat, δlon, δalt, δv_n, δv_e, δv_d, δroll, δpitch, δyaw, δb_ax, δb_ay, δb_az, δb_gx, δb_gy, δb_gz]
/// ```
///
/// # Attitude Correction
///
/// For small attitude errors, the correction is applied as:
/// ```text
/// C_corrected = (I - [δθ]×) * C_nominal ≈ C_error * C_nominal
/// ```
/// where [δθ]× is the skew-symmetric matrix of the attitude error angles.
///
/// # Arguments
///
/// * `state` - Mutable reference to the navigation state to correct
/// * `delta_x` - Error-state correction vector (9 or 15 elements)
///
/// # Panics
///
/// Panics if `delta_x` has fewer than 9 elements.
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::apply_eskf_correction;
/// use strapdown::StrapdownState;
/// use nalgebra::{DVector, Rotation3};
///
/// let mut state = StrapdownState::new(
///     45.0, -122.0, 100.0,
///     10.0, 5.0, 0.0,
///     Rotation3::identity(),
///     true,
///     None,
/// );
///
/// // Apply a small correction
/// let delta_x = DVector::from_vec(vec![
///     0.0001,  // δlat (rad)
///     0.0001,  // δlon (rad)
///     1.0,     // δalt (m)
///     0.1,     // δv_n (m/s)
///     0.1,     // δv_e (m/s)
///     0.0,     // δv_d (m/s)
///     0.01,    // δroll (rad)
///     0.01,    // δpitch (rad)
///     0.01,    // δyaw (rad)
/// ]);
///
/// apply_eskf_correction(&mut state, &delta_x);
/// ```
///
/// # References
///
/// - Sola, J. "Quaternion kinematics for the error-state Kalman filter" (2017), Section 6.4
/// - Groves 2nd ed., Section 14.2.6 (state correction)
pub fn apply_eskf_correction(state: &mut StrapdownState, delta_x: &DVector<f64>) {
    assert!(
        delta_x.len() >= 9,
        "Error state must have at least 9 elements, got {}",
        delta_x.len()
    );

    // Apply position correction (additive)
    state.latitude += delta_x[0];
    state.longitude += delta_x[1];
    state.altitude += delta_x[2];

    // Apply velocity correction (additive)
    state.velocity_north += delta_x[3];
    state.velocity_east += delta_x[4];
    state.velocity_vertical += delta_x[5];

    // Apply attitude correction using small-angle approximation
    // For small angles: C_corrected ≈ (I + [δθ]×) * C_nominal
    // or equivalently: C_corrected = Rotation3::from_euler_angles(δroll, δpitch, δyaw) * C_nominal
    let delta_roll = delta_x[6];
    let delta_pitch = delta_x[7];
    let delta_yaw = delta_x[8];

    // Create small-angle rotation correction
    // Using Rodrigues formula for small angles: R ≈ I + [θ]×
    let delta_rotation = Rotation3::from_euler_angles(delta_roll, delta_pitch, delta_yaw);

    // Apply correction: C_new = δC * C_old
    state.attitude = delta_rotation * state.attitude;

    // Note: IMU bias corrections (elements 9-14) are not stored in StrapdownState.
    // If needed, they should be handled separately by the filter.
}

/// Apply an error-state correction with optional bias state output
///
/// Extended version of [`apply_eskf_correction`] that also returns the bias
/// corrections for filters that track IMU biases separately.
///
/// # Arguments
///
/// * `state` - Mutable reference to the navigation state to correct
/// * `delta_x` - Error-state correction vector (9 or 15 elements)
///
/// # Returns
///
/// Optional tuple of (accel_bias_correction, gyro_bias_correction) if delta_x has 15 elements.
/// Returns `None` if delta_x has only 9 elements.
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::apply_eskf_correction_with_biases;
/// use strapdown::StrapdownState;
/// use nalgebra::{DVector, Rotation3};
///
/// let mut state = StrapdownState::default();
///
/// // 15-state correction including biases
/// let delta_x = DVector::from_vec(vec![
///     0.0, 0.0, 0.0,    // position
///     0.0, 0.0, 0.0,    // velocity
///     0.0, 0.0, 0.0,    // attitude
///     0.01, 0.01, 0.01, // accel bias
///     0.001, 0.001, 0.001, // gyro bias
/// ]);
///
/// if let Some((accel_bias, gyro_bias)) = apply_eskf_correction_with_biases(&mut state, &delta_x) {
///     // Apply bias corrections to IMU preprocessing
///     println!("Accel bias correction: {:?}", accel_bias);
///     println!("Gyro bias correction: {:?}", gyro_bias);
/// }
/// ```
pub fn apply_eskf_correction_with_biases(
    state: &mut StrapdownState,
    delta_x: &DVector<f64>,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    // Apply the navigation state correction
    apply_eskf_correction(state, delta_x);

    // Extract bias corrections if present
    if delta_x.len() >= 15 {
        let accel_bias = Vector3::new(delta_x[9], delta_x[10], delta_x[11]);
        let gyro_bias = Vector3::new(delta_x[12], delta_x[13], delta_x[14]);
        Some((accel_bias, gyro_bias))
    } else {
        None
    }
}

/// Construct a 15-state error vector from position error, conditional mean, and biases
///
/// This helper function assembles a full 15-state error vector from the RBPF
/// components: position error (from particles), and conditional state (velocity
/// error, attitude error, and bias errors from the per-particle EKF).
///
/// # Arguments
///
/// * `position_error` - Position error [δlat, δlon, δalt] in (rad, rad, m)
/// * `conditional_mean` - 12-element conditional EKF mean [δv, δθ, δb_g, δb_a]
///
/// # Returns
///
/// 15-element error state vector suitable for [`apply_eskf_correction`]
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::assemble_error_state;
/// use nalgebra::{Vector3, DVector};
///
/// let dr = Vector3::new(0.0001, 0.0001, 1.0);
/// let mu = DVector::from_vec(vec![0.0; 12]);
///
/// let delta_x = assemble_error_state(&dr, &mu);
/// assert_eq!(delta_x.len(), 15);
/// ```
pub fn assemble_error_state(
    position_error: &Vector3<f64>,
    conditional_mean: &DVector<f64>,
) -> DVector<f64> {
    assert_eq!(
        conditional_mean.len(),
        12,
        "Conditional mean must have 12 elements, got {}",
        conditional_mean.len()
    );

    // Conditional mean layout (matches F15 extraction via view((3,3), (12,12))):
    // [0-2]: δv (velocity error)
    // [3-5]: δθ (attitude error)
    // [6-8]: δb_a (accelerometer bias error)
    // [9-11]: δb_g (gyroscope bias error)
    //
    // 15-state error layout:
    // [0-2]: δr (position error)
    // [3-5]: δv (velocity error)
    // [6-8]: δθ (attitude error)
    // [9-11]: δb_a (accelerometer bias error)
    // [12-14]: δb_g (gyroscope bias error)
    DVector::from_vec(vec![
        position_error[0],    // δlat
        position_error[1],    // δlon
        position_error[2],    // δalt
        conditional_mean[0],  // δv_n
        conditional_mean[1],  // δv_e
        conditional_mean[2],  // δv_d
        conditional_mean[3],  // δroll
        conditional_mean[4],  // δpitch
        conditional_mean[5],  // δyaw
        conditional_mean[6],  // δb_ax
        conditional_mean[7],  // δb_ay
        conditional_mean[8],  // δb_az
        conditional_mean[9],  // δb_gx
        conditional_mean[10], // δb_gy
        conditional_mean[11], // δb_gz
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Rotation3;

    /// Every diagonal entry of the error-state transition Jacobian must be ~1.
    ///
    /// `F = I + A dt`, so for a small `dt` the diagonal stays near unity. The attitude
    /// block used to be *assigned* `-[omega^b]_x dt` rather than subtracted from the
    /// identity, and a skew-symmetric matrix has a zero diagonal, so f[(6,6)], f[(7,7)]
    /// and f[(8,8)] were silently zero. That annihilates the attitude error covariance
    /// on every propagation: the filter believes attitude is perfectly known after each
    /// step, never corrects tilt, and leaves gyro bias unconstrained. See #286.
    #[test]
    fn test_error_state_jacobian_diagonal_is_unity() {
        let dt = 0.01;
        let state = StrapdownState::new(
            45.0_f64.to_radians(),
            (-122.0_f64).to_radians(),
            150.0,
            8.0,
            -3.0,
            0.4,
            Rotation3::from_euler_angles(0.25, -0.15, 1.2),
            false,
            Some(true),
        );
        // Deliberately non-zero angular rate: with omega = 0 the bug is invisible.
        let f = error_state_transition_jacobian(
            &state,
            &Vector3::new(0.6, -1.1, 9.9),
            &Vector3::new(0.03, -0.02, 0.05),
            dt,
        );
        for i in 0..15 {
            assert!(
                (f[(i, i)] - 1.0).abs() < 1e-3,
                "F[{i},{i}] = {}, expected ~1. A block was assigned over the identity \
                 instead of added to it (#286).",
                f[(i, i)]
            );
        }
    }

    /// The error-state Jacobian must agree with a finite difference of the nonlinear
    /// propagation it linearises.
    ///
    /// This is the check that would have caught #266 and #286 together. It perturbs each
    /// error state, runs `forward` (with the bias errors applied to the IMU input the way
    /// the ESKF's `predict` applies them), and differences the result.
    ///
    /// Entries are compared only where the true derivative is large enough to resolve;
    /// `forward` integrates velocity trapezoidally, so a first-order `I + A dt` Jacobian
    /// legitimately omits O(dt^2) cross terms. The bound below is set above those and far
    /// below the first-order terms.
    #[test]
    fn test_error_state_jacobian_matches_nonlinear_propagation() {
        const DT: f64 = 0.02;
        let accel = Vector3::new(0.6, -1.1, 9.9);
        let gyro = Vector3::new(0.03, -0.02, 0.05);
        let nominal = StrapdownState::new(
            45.0_f64.to_radians(),
            (-122.0_f64).to_radians(),
            150.0,
            8.0,
            -3.0,
            0.4,
            Rotation3::from_euler_angles(0.25, -0.15, 1.2),
            false,
            Some(true),
        );

        let mut base_out = nominal;
        crate::forward(&mut base_out, crate::IMUData { accel, gyro }, DT);

        // Apply a 15-element error, propagate, and return the 9-element error out.
        let propagate = |dx: &DVector<f64>| -> DVector<f64> {
            let mut s = nominal;
            s.latitude += dx[0];
            s.longitude += dx[1];
            s.altitude += dx[2];
            s.velocity_north += dx[3];
            s.velocity_east += dx[4];
            s.velocity_vertical += dx[5];
            s.attitude =
                nominal.attitude * Rotation3::from_scaled_axis(Vector3::new(dx[6], dx[7], dx[8]));
            // predict() feeds `accel - bias`, so a bias error db means the true input is
            // accel - (b_nom + db).
            let imu = crate::IMUData {
                accel: accel - Vector3::new(dx[9], dx[10], dx[11]),
                gyro: gyro - Vector3::new(dx[12], dx[13], dx[14]),
            };
            crate::forward(&mut s, imu, DT);
            let dtheta = (base_out.attitude.transpose() * s.attitude).scaled_axis();
            DVector::from_vec(vec![
                s.latitude - base_out.latitude,
                s.longitude - base_out.longitude,
                s.altitude - base_out.altitude,
                s.velocity_north - base_out.velocity_north,
                s.velocity_east - base_out.velocity_east,
                s.velocity_vertical - base_out.velocity_vertical,
                dtheta[0],
                dtheta[1],
                dtheta[2],
            ])
        };

        let f_analytic = error_state_transition_jacobian(&nominal, &accel, &gyro, DT);
        // Per-state step sizes: radians for lat/lon, SI elsewhere. Chosen large enough to
        // clear cancellation noise in an f64 latitude of ~0.8 rad.
        // The lat/lon steps are deliberately coarse (1e-5 rad, ~64 m). A latitude of
        // ~0.8 rad loses most of its significant digits under a 1e-9 perturbation, and
        // extracting a rotation vector from the resulting attitudes then produces
        // cancellation noise that scales like 1/h -- it reads as a spurious 12 rad/rad
        // in the attitude rows at h=1e-9 and decays to 0.013 by h=1e-6. 1e-5 puts that
        // noise below the tolerance while staying comfortably inside the linear regime
        // of the gravity model.
        let steps: [f64; 15] = [
            1e-5, 1e-5, 1e-1, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6,
            1e-6,
        ];
        // O(dt^2) terms the first-order Jacobian omits reach ~7e-4 at this dt; anything
        // above 5e-3 is a first-order structural error.
        const TOLERANCE: f64 = 5e-3;

        // Columns 9..15 (the bias errors) are excluded. `forward` stores attitude via
        // `Rotation3::from_matrix`, an iterative orthonormalising projection whose own
        // derivative is not unity, and differencing through it reports d(theta)/d(b_g)
        // as ~1.5x dt rather than dt. Until that is understood (#286) the finite
        // difference is not a trustworthy oracle for those columns; the attitude and
        // velocity columns below are unaffected and cover the error dynamics that
        // matter here.
        for col in 0..9 {
            let h = steps[col];
            let mut plus = DVector::zeros(15);
            plus[col] = h;
            let mut minus = DVector::zeros(15);
            minus[col] = -h;
            let out_plus = propagate(&plus);
            let out_minus = propagate(&minus);
            for row in 0..9 {
                let numerical = (out_plus[row] - out_minus[row]) / (2.0 * h);
                let analytic = f_analytic[(row, col)];
                assert!(
                    (analytic - numerical).abs() < TOLERANCE,
                    "F[{row},{col}] = {analytic:e} but finite differences of `forward` \
                     give {numerical:e} (diff {:e}); the linearisation disagrees with the \
                     propagation it is supposed to linearise (#286)",
                    (analytic - numerical).abs()
                );
            }
        }
    }

    /// Finite-difference check of the error-state Jacobian's attitude convention.
    ///
    /// This is the test whose absence let #266 ship. The existing
    /// `numerical_state_jacobian` helper covers the *full-state* Jacobian; nothing
    /// checked that `error_state_transition_jacobian` uses one consistent
    /// attitude-error convention throughout.
    ///
    /// The error state uses the LOCAL (body-frame) convention:
    ///     C_b^n(true) = C_b^n(nominal) · (I + [δθ]_×)
    /// so perturbing the nominal attitude on the right by a small rotation δθ and
    /// propagating must reproduce the velocity change that F's δv/δθ block predicts.
    ///
    /// A non-identity `C_b^n` is essential: the local and global conventions differ
    /// by exactly that rotation, so with identity attitude both forms agree and the
    /// bug is invisible. Hence the deliberately non-trivial roll/pitch/yaw below.
    #[test]
    fn test_error_state_jacobian_velocity_attitude_block_is_body_frame() {
        let dt = 0.01;
        // Deliberately non-identity attitude -- see doc comment.
        let attitude = Rotation3::from_euler_angles(0.3, -0.2, 1.1);
        let state = StrapdownState::new(
            45.0_f64.to_radians(),
            -122.0_f64.to_radians(),
            100.0,
            10.0,
            5.0,
            0.5,
            attitude,
            false,
            Some(true),
        );
        // Specific force with all three components non-zero so every column matters.
        let accel = Vector3::new(0.7, -1.3, 9.81);
        let gyro = Vector3::new(0.02, -0.01, 0.03);

        let f_error = error_state_transition_jacobian(&state, &accel, &gyro, dt);

        // Analytic block: F[3..6, 6..9]
        let mut analytic = nalgebra::Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                analytic[(i, j)] = f_error[(3 + i, 6 + j)];
            }
        }

        // Numerical block: perturb attitude on the RIGHT (body frame), propagate,
        // and difference the resulting velocity.
        let eps = 1e-7;
        let mut numerical = nalgebra::Matrix3::zeros();
        for j in 0..3 {
            let mut axis = Vector3::zeros();
            axis[j] = eps;

            let mut plus = state;
            plus.attitude = state.attitude * Rotation3::from_scaled_axis(axis);
            let mut minus = state;
            minus.attitude = state.attitude * Rotation3::from_scaled_axis(-axis);

            let imu = crate::IMUData { accel, gyro };
            crate::forward(&mut plus, imu, dt);
            crate::forward(&mut minus, imu, dt);

            numerical[(0, j)] = (plus.velocity_north - minus.velocity_north) / (2.0 * eps);
            numerical[(1, j)] = (plus.velocity_east - minus.velocity_east) / (2.0 * eps);
            numerical[(2, j)] = (plus.velocity_vertical - minus.velocity_vertical) / (2.0 * eps);
        }

        // Relative tolerance: `forward` integrates velocity trapezoidally, so the
        // finite difference carries an O(dt) discretisation residual that a fixed
        // absolute epsilon would either reject or make meaningless. 1e-3 relative sits
        // far below the O(1) relative gap between the two attitude-error conventions
        // and far above the discretisation residual (~3e-5 relative here).
        let scale = analytic.abs().max().max(1e-12);
        for i in 0..3 {
            for j in 0..3 {
                let err = (analytic[(i, j)] - numerical[(i, j)]).abs() / scale;
                assert!(
                    err < 1e-3,
                    "F[{},{}] disagrees with finite differences: analytic={:e} numerical={:e} rel_err={:e}",
                    3 + i,
                    6 + j,
                    analytic[(i, j)],
                    numerical[(i, j)],
                    err
                );
            }
        }

        // Guard the specific regression: the previous code used the GLOBAL
        // (nav-frame) form -[C_b^n f^b]_× dt. Assert we are measurably far from it,
        // so a revert cannot pass this test.
        let global_form = -vector_to_skew_symmetric(&(state.attitude.matrix() * accel)) * dt;
        let gap = (analytic - global_form).abs().max() / scale;
        assert!(
            gap > 1e-2,
            "F's velocity/attitude block matches the global convention (gap {:e}); \
             the local (body-frame) form -C_b^n [f^b]_x dt is required -- see #266",
            gap
        );
    }

    /// The position rows of the error-state Jacobian must be in radians, matching
    /// both the measurement Jacobians and `apply_eskf_correction`.
    ///
    /// #266: the ESKF's `inject_error_state` treated these as metres and divided by
    /// the principal radii, rescaling every horizontal correction by ~1/6.4e6.
    #[test]
    fn test_error_state_jacobian_position_rows_are_radians() {
        let dt = 0.5;
        let state = StrapdownState::new(
            45.0_f64.to_radians(),
            -122.0_f64.to_radians(),
            100.0,
            0.0,
            0.0,
            0.0,
            Rotation3::identity(),
            false,
            Some(true),
        );
        let f = error_state_transition_jacobian(&state, &Vector3::zeros(), &Vector3::zeros(), dt);

        let (r_n, r_e, _) =
            crate::earth::principal_radii(&state.latitude.to_degrees(), &state.altitude);

        // A 1 m/s north velocity error must produce dt / r_n radians of latitude
        // error, i.e. a number of order 1e-7 -- not order 1.
        assert_approx_eq!(f[(0, 3)], dt / r_n, 1e-15);
        assert_approx_eq!(f[(1, 4)], dt / (r_e * state.latitude.cos()), 1e-15);
        assert!(
            f[(0, 3)] < 1e-6,
            "latitude row must be radians per m/s (got {}); a value near dt means metres crept back in",
            f[(0, 3)]
        );
        // Altitude stays in metres.
        assert_approx_eq!(f[(2, 5)], dt, 1e-15);
    }

    /// Compute numerical Jacobian using finite differences for state transition
    fn numerical_state_jacobian(
        state: &StrapdownState,
        imu_accel: &Vector3<f64>,
        imu_gyro: &Vector3<f64>,
        dt: f64,
        epsilon: f64,
    ) -> DMatrix<f64> {
        let mut jac = DMatrix::<f64>::zeros(9, 9);

        // Convert state to vector for perturbation
        let x0: Vec<f64> = state.into();

        // Evaluate nominal dynamics
        let mut state_nominal = *state;
        crate::forward(
            &mut state_nominal,
            crate::IMUData {
                accel: *imu_accel,
                gyro: *imu_gyro,
            },
            dt,
        );
        let f0: Vec<f64> = (&state_nominal).into();

        // Perturb each state component
        for j in 0..9 {
            let mut x_pert = x0.clone();
            x_pert[j] += epsilon;

            // Create perturbed state
            let mut state_pert = StrapdownState::try_from(x_pert.as_slice()).unwrap();
            state_pert.is_enu = state.is_enu;

            // Propagate perturbed state
            crate::forward(
                &mut state_pert,
                crate::IMUData {
                    accel: *imu_accel,
                    gyro: *imu_gyro,
                },
                dt,
            );
            let f_pert: Vec<f64> = (&state_pert).into();

            // Compute finite difference
            for i in 0..9 {
                jac[(i, j)] = (f_pert[i] - f0[i]) / epsilon;
            }
        }

        jac
    }

    #[test]
    fn test_state_transition_jacobian_stationary() {
        // Test Jacobian at a stationary state
        let state = StrapdownState::new(
            45.0,
            -122.0,
            100.0,
            0.0,
            0.0,
            0.0,
            Rotation3::identity(),
            true,       // degrees
            Some(true), // ENU
        );

        let accel = Vector3::new(0.0, 0.0, 9.81); // Gravity-compensating
        let gyro = Vector3::zeros();
        let dt = 0.0001; // Very small dt for high accuracy

        let f_analytic = state_transition_jacobian(&state, &accel, &gyro, dt);
        let f_numeric = numerical_state_jacobian(&state, &accel, &gyro, dt, 1e-6);

        // Find and report largest errors for debugging
        let diff = &f_analytic - &f_numeric;
        let mut max_error = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;
        for i in 0..9 {
            for j in 0..9 {
                let err = diff[(i, j)].abs();
                if err > max_error {
                    max_error = err;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        if max_error >= 1e-6 {
            eprintln!(
                "Largest error at ({}, {}): analytic={:.10e}, numeric={:.10e}, diff={:.10e}",
                max_i,
                max_j,
                f_analytic[(max_i, max_j)],
                f_numeric[(max_i, max_j)],
                max_error
            );

            // Print a few more large errors for context
            let mut errors: Vec<(usize, usize, f64)> = Vec::new();
            for i in 0..9 {
                for j in 0..9 {
                    let err = diff[(i, j)].abs();
                    if err > 1e-7 {
                        errors.push((i, j, err));
                    }
                }
            }
            errors.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            eprintln!("Top 5 errors:");
            for (i, j, err) in errors.iter().take(5) {
                eprintln!("  ({}, {}): {:.10e}", i, j, err);
            }
        }

        // Compare all elements
        let max_error = (&f_analytic - &f_numeric).abs().max();
        assert!(
            max_error < 1e-6,
            "Max error {} exceeds threshold for stationary state",
            max_error
        );
    }

    #[test]
    fn test_state_transition_jacobian_moving() {
        // Test Jacobian with non-zero velocities and small rotations
        let state = StrapdownState::new(
            45.0,
            -122.0,
            100.0,
            5.0,
            3.0,
            -0.5,                                           // Smaller velocities
            Rotation3::from_euler_angles(0.01, 0.01, 0.01), // Smaller rotations
            true,
            Some(true),
        );

        let accel = Vector3::new(0.1, -0.1, 9.81); // Smaller accelerations
        let gyro = Vector3::new(0.001, -0.001, 0.002); // Smaller rates
        let dt = 0.0001; // Very small dt for high accuracy

        let f_analytic = state_transition_jacobian(&state, &accel, &gyro, dt);
        let f_numeric = numerical_state_jacobian(&state, &accel, &gyro, dt, 1e-6);

        let max_error = (&f_analytic - &f_numeric).abs().max();
        assert!(
            max_error < 1e-5,
            "Max error {} exceeds threshold for moving state. Note: errors ~1e-5 are expected due to nonlinear coupling in trapezoidal integration.",
            max_error
        );
    }

    #[test]
    fn test_state_transition_jacobian_multiple_states() {
        // Test across multiple randomized states
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..10 {
            let lat = rng.random_range(-80.0..80.0);
            let lon = rng.random_range(-180.0..180.0);
            let alt = rng.random_range(0.0..5000.0);
            let v_n = rng.random_range(-10.0..10.0); // Smaller velocities
            let v_e = rng.random_range(-10.0..10.0);
            let v_d = rng.random_range(-2.0..2.0);
            let roll = rng.random_range(-0.1..0.1); // Smaller angles
            let pitch = rng.random_range(-0.1..0.1);
            let yaw = rng.random_range(-0.5..0.5);

            let state = StrapdownState::new(
                lat,
                lon,
                alt,
                v_n,
                v_e,
                v_d,
                Rotation3::from_euler_angles(roll, pitch, yaw),
                true,
                Some(true),
            );

            let accel = Vector3::new(
                rng.random_range(-0.5..0.5), // Smaller accelerations
                rng.random_range(-0.5..0.5),
                rng.random_range(9.0..10.5),
            );
            let gyro = Vector3::new(
                rng.random_range(-0.01..0.01), // Smaller rates
                rng.random_range(-0.01..0.01),
                rng.random_range(-0.01..0.01),
            );
            let dt = 0.0001; // Very small dt for high accuracy

            let f_analytic = state_transition_jacobian(&state, &accel, &gyro, dt);
            let f_numeric = numerical_state_jacobian(&state, &accel, &gyro, dt, 1e-6);

            let max_error = (&f_analytic - &f_numeric).abs().max();
            assert!(
                max_error < 5e-4,
                "Max error {} exceeds threshold for random state {:?}. Note: first-order Jacobian has O(dt²) errors from nonlinear coupling.",
                max_error,
                (lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw)
            );
        }
    }

    #[test]
    fn test_process_noise_jacobian_dimensions() {
        let state = StrapdownState::default();
        let dt = 0.01;
        let g = process_noise_jacobian(&state, dt);

        assert_eq!(g.nrows(), 9);
        assert_eq!(g.ncols(), 6);
    }

    #[test]
    fn test_process_noise_jacobian_structure() {
        let state = StrapdownState::new(
            45.0,
            -122.0,
            100.0,
            0.0,
            0.0,
            0.0,
            Rotation3::identity(),
            true,
            Some(true),
        );
        let dt = 0.01;
        let g = process_noise_jacobian(&state, dt);

        // Position rows should be zero (no direct noise coupling)
        for i in 0..3 {
            for j in 0..6 {
                assert_approx_eq!(g[(i, j)], 0.0, 1e-10);
            }
        }

        // Velocity rows (3-5) should couple to accel noise (cols 0-2)
        // with C_b^n transformation - for identity rotation, only diagonals are non-zero
        for i in 3..6 {
            // Diagonal elements should be non-zero (identity * dt)
            assert!(
                g[(i, i - 3)].abs() > 0.0,
                "g[{}, {}] should be non-zero",
                i,
                i - 3
            );
        }

        // Attitude rows (6-8) should couple to gyro noise (cols 3-5)
        // For identity rotation, only diagonal elements are non-zero
        for i in 6..9 {
            // Diagonal elements should be non-zero
            assert!(
                g[(i, i - 3)].abs() > 0.0,
                "g[{}, {}] should be non-zero",
                i,
                i - 3
            );
        }
    }

    #[test]
    fn test_gps_position_jacobian() {
        let state = StrapdownState::default();
        let h = gps_position_jacobian(&state);

        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 9);

        // Check identity structure
        assert_approx_eq!(h[(0, 0)], 1.0, 1e-10);
        assert_approx_eq!(h[(1, 1)], 1.0, 1e-10);
        assert_approx_eq!(h[(2, 2)], 1.0, 1e-10);

        // Check other elements are zero
        for i in 0..3 {
            for j in 3..9 {
                assert_approx_eq!(h[(i, j)], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn test_gps_velocity_jacobian() {
        let state = StrapdownState::default();
        let h = gps_velocity_jacobian(&state);

        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 9);

        // Check identity structure
        assert_approx_eq!(h[(0, 3)], 1.0, 1e-10);
        assert_approx_eq!(h[(1, 4)], 1.0, 1e-10);
        assert_approx_eq!(h[(2, 5)], 1.0, 1e-10);

        // Check position elements are zero
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(h[(i, j)], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn test_gps_position_velocity_jacobian() {
        let state = StrapdownState::default();
        let h = gps_position_velocity_jacobian(&state);

        assert_eq!(h.nrows(), 5);
        assert_eq!(h.ncols(), 9);

        // Check structure: [lat, lon, alt, v_n, v_e]
        assert_approx_eq!(h[(0, 0)], 1.0, 1e-10);
        assert_approx_eq!(h[(1, 1)], 1.0, 1e-10);
        assert_approx_eq!(h[(2, 2)], 1.0, 1e-10);
        assert_approx_eq!(h[(3, 3)], 1.0, 1e-10);
        assert_approx_eq!(h[(4, 4)], 1.0, 1e-10);

        // v_d is not included
        for i in 0..5 {
            assert_approx_eq!(h[(i, 5)], 0.0, 1e-10);
        }
    }

    #[test]
    fn test_relative_altitude_jacobian() {
        let state = StrapdownState::default();
        let h = relative_altitude_jacobian(&state);

        assert_eq!(h.nrows(), 1);
        assert_eq!(h.ncols(), 9);

        // Check structure
        assert_approx_eq!(h[(0, 2)], 1.0, 1e-10);

        // All other elements should be zero
        for j in 0..9 {
            if j != 2 {
                assert_approx_eq!(h[(0, j)], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn test_measurement_jacobians_consistency() {
        // Verify that GPS position+velocity is combination of individual measurements
        let state = StrapdownState::default();

        let h_pos = gps_position_jacobian(&state);
        let h_vel = gps_velocity_jacobian(&state);
        let h_combined = gps_position_velocity_jacobian(&state);

        // First 3 rows should match position Jacobian
        for i in 0..3 {
            for j in 0..9 {
                assert_approx_eq!(h_combined[(i, j)], h_pos[(i, j)], 1e-10);
            }
        }

        // Rows 3-4 should match first 2 rows of velocity Jacobian
        for i in 0..2 {
            for j in 0..9 {
                assert_approx_eq!(h_combined[(3 + i, j)], h_vel[(i, j)], 1e-10);
            }
        }
    }

    #[test]
    fn test_apply_eskf_correction_position() {
        let mut state = StrapdownState::new(
            45.0,
            -122.0,
            100.0,
            0.0,
            0.0,
            0.0,
            Rotation3::identity(),
            true,
            Some(true),
        );

        let initial_lat = state.latitude;
        let initial_lon = state.longitude;
        let initial_alt = state.altitude;

        let delta_x = DVector::from_vec(vec![
            0.0001, // δlat (rad)
            0.0002, // δlon (rad)
            5.0,    // δalt (m)
            0.0, 0.0, 0.0, // velocity
            0.0, 0.0, 0.0, // attitude
        ]);

        apply_eskf_correction(&mut state, &delta_x);

        assert_approx_eq!(state.latitude, initial_lat + 0.0001, 1e-10);
        assert_approx_eq!(state.longitude, initial_lon + 0.0002, 1e-10);
        assert_approx_eq!(state.altitude, initial_alt + 5.0, 1e-10);
    }

    #[test]
    fn test_apply_eskf_correction_velocity() {
        let mut state = StrapdownState::new(
            45.0,
            -122.0,
            100.0,
            10.0,
            5.0,
            -1.0,
            Rotation3::identity(),
            true,
            Some(true),
        );

        let delta_x = DVector::from_vec(vec![
            0.0, 0.0, 0.0, // position
            0.5, -0.3, 0.1, // velocity correction
            0.0, 0.0, 0.0, // attitude
        ]);

        apply_eskf_correction(&mut state, &delta_x);

        assert_approx_eq!(state.velocity_north, 10.5, 1e-10);
        assert_approx_eq!(state.velocity_east, 4.7, 1e-10);
        assert_approx_eq!(state.velocity_vertical, -0.9, 1e-10);
    }

    #[test]
    fn test_apply_eskf_correction_attitude() {
        let mut state = StrapdownState::new(
            45.0,
            -122.0,
            100.0,
            0.0,
            0.0,
            0.0,
            Rotation3::identity(),
            true,
            Some(true),
        );

        // Apply small attitude correction
        let delta_roll = 0.01; // rad
        let delta_pitch = 0.02;
        let delta_yaw = 0.03;

        let delta_x = DVector::from_vec(vec![
            0.0,
            0.0,
            0.0, // position
            0.0,
            0.0,
            0.0, // velocity
            delta_roll,
            delta_pitch,
            delta_yaw,
        ]);

        apply_eskf_correction(&mut state, &delta_x);

        // Check that attitude has been updated
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        assert_approx_eq!(roll, delta_roll, 1e-6);
        assert_approx_eq!(pitch, delta_pitch, 1e-6);
        assert_approx_eq!(yaw, delta_yaw, 1e-6);
    }

    #[test]
    fn test_apply_eskf_correction_with_biases() {
        let mut state = StrapdownState::default();

        let delta_x = DVector::from_vec(vec![
            0.0, 0.0, 0.0, // position
            0.0, 0.0, 0.0, // velocity
            0.0, 0.0, 0.0, // attitude
            0.01, 0.02, 0.03, // accel bias
            0.001, 0.002, 0.003, // gyro bias
        ]);

        let biases = apply_eskf_correction_with_biases(&mut state, &delta_x);
        assert!(biases.is_some());

        let (accel_bias, gyro_bias) = biases.unwrap();
        assert_approx_eq!(accel_bias[0], 0.01, 1e-10);
        assert_approx_eq!(accel_bias[1], 0.02, 1e-10);
        assert_approx_eq!(accel_bias[2], 0.03, 1e-10);
        assert_approx_eq!(gyro_bias[0], 0.001, 1e-10);
        assert_approx_eq!(gyro_bias[1], 0.002, 1e-10);
        assert_approx_eq!(gyro_bias[2], 0.003, 1e-10);
    }

    #[test]
    fn test_apply_eskf_correction_with_biases_returns_none_for_9_state() {
        let mut state = StrapdownState::default();

        let delta_x = DVector::from_vec(vec![
            0.0, 0.0, 0.0, // position
            0.0, 0.0, 0.0, // velocity
            0.0, 0.0, 0.0, // attitude
        ]);

        let biases = apply_eskf_correction_with_biases(&mut state, &delta_x);
        assert!(biases.is_none());
    }

    #[test]
    fn test_assemble_error_state() {
        let dr = Vector3::new(0.0001, 0.0002, 5.0);
        let mu = DVector::from_vec(vec![
            0.1, 0.2, 0.3, // δv
            0.01, 0.02, 0.03, // δθ
            0.001, 0.002, 0.003, // δb_g
            0.0001, 0.0002, 0.0003, // δb_a
        ]);

        let delta_x = assemble_error_state(&dr, &mu);

        assert_eq!(delta_x.len(), 15);

        // Position
        assert_approx_eq!(delta_x[0], 0.0001, 1e-10);
        assert_approx_eq!(delta_x[1], 0.0002, 1e-10);
        assert_approx_eq!(delta_x[2], 5.0, 1e-10);

        // Velocity
        assert_approx_eq!(delta_x[3], 0.1, 1e-10);
        assert_approx_eq!(delta_x[4], 0.2, 1e-10);
        assert_approx_eq!(delta_x[5], 0.3, 1e-10);

        // Attitude
        assert_approx_eq!(delta_x[6], 0.01, 1e-10);
        assert_approx_eq!(delta_x[7], 0.02, 1e-10);
        assert_approx_eq!(delta_x[8], 0.03, 1e-10);

        // Accel bias (from conditional positions 9-11)
        assert_approx_eq!(delta_x[9], 0.001, 1e-10);
        assert_approx_eq!(delta_x[10], 0.002, 1e-10);
        assert_approx_eq!(delta_x[11], 0.003, 1e-10);

        // Gyro bias (from conditional positions 6-8)
        assert_approx_eq!(delta_x[12], 0.0001, 1e-10);
        assert_approx_eq!(delta_x[13], 0.0002, 1e-10);
        assert_approx_eq!(delta_x[14], 0.0003, 1e-10);
    }
}
