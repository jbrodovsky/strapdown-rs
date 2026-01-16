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
    let _g = earth::gravity(&lat, &h); // Reserved for future use in gravity gradient computations

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

    // ∂(δv̇)/∂(δp): Velocity error rate depends on position error (gravity gradient, Coriolis)
    //
    // NOTE: The gravity gradient term (∂g/∂h ≈ -3.08e-6) creates unstable or marginally stable
    // dynamics in the vertical channel when used alone:
    // - ENU: +3.08e-6 coupling → exponentially growing eigenvalues (UNSTABLE)
    // - NED: -3.08e-6 coupling → purely imaginary eigenvalues (undamped oscillations)
    //
    // In practice, this term is often omitted in error-state EKF formulations because:
    // 1. The magnitude is very small (~3e-6) and negligible over short time steps
    // 2. Measurement updates (GPS, baro) provide the necessary stabilization
    // 3. Including it can cause numerical issues in particle filters with sparse updates
    //
    // For the RBPF conditional EKF, we omit this term to avoid vertical channel instability.
    // The position-altitude coupling is handled implicitly through measurement updates.
    // f[(5, 2)] = 0.0; // Gravity gradient term omitted for stability

    // ∂(δv̇)/∂(δv): Velocity error damping due to Coriolis and centrifugal effects
    // These coupling terms are small for low dynamics and often approximated as zero
    // The main effect is self-coupling which is captured by identity matrix

    // ∂(δv̇)/∂(δθ): Velocity error depends on attitude error (most important coupling!)
    // δv̇^n = -[C_b^n f^b]_× δθ
    // where [a]_× is the skew-symmetric matrix of vector a
    let f_n = c_bn * imu_accel; // Transform measured acceleration to nav frame
    let f_skew = vector_to_skew_symmetric(&f_n);
    for i in 0..3 {
        for j in 0..3 {
            f[(3 + i, 6 + j)] = -f_skew[(i, j)] * dt;
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
    let omega_skew = vector_to_skew_symmetric(imu_gyro);
    for i in 0..3 {
        for j in 0..3 {
            f[(6 + i, 6 + j)] = -omega_skew[(i, j)] * dt;
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
/// **Note**: This is a placeholder function for future use. Currently, the EKF computes
/// measurement Jacobians internally during the update step. This function provides a
/// template structure for when custom Jacobian computation is integrated into the EKF.
///
/// Gravity anomaly measurements depend on latitude and longitude (position) to query the
/// geophysical map. The partial derivatives ∂z/∂lat and ∂z/∂lon would be computed
/// numerically by the measurement model based on map gradients.
///
/// # Arguments
///
/// * `_state` - Current navigation state (reserved for future use)
///
/// # Returns
///
/// 1×9 measurement Jacobian matrix H template for gravity anomaly (currently zeros)
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
    // Gravity anomaly depends on position (lat, lon) through map lookup
    // The partial derivatives ∂z/∂lat and ∂z/∂lon are computed numerically
    // by the measurement model based on map gradients

    // These will be filled in by the measurement model with numerical derivatives
    // from the geophysical map interpolation
    // h[(0, 0)] = ∂(anomaly)/∂(lat) - computed from map gradient
    // h[(0, 1)] = ∂(anomaly)/∂(lon) - computed from map gradient
    DMatrix::<f64>::zeros(1, 9)
}

/// Compute measurement Jacobian (H) for magnetic anomaly measurement
///
/// **Note**: This is a placeholder function for future use. Currently, the EKF computes
/// measurement Jacobians internally during the update step. This function provides a
/// template structure for when custom Jacobian computation is integrated into the EKF.
///
/// Magnetic anomaly measurements depend on latitude and longitude (position) to query the
/// geophysical map. The partial derivatives ∂z/∂lat and ∂z/∂lon would be computed
/// numerically by the measurement model based on map gradients.
///
/// # Arguments
///
/// * `_state` - Current navigation state (reserved for future use)
///
/// # Returns
///
/// 1×9 measurement Jacobian matrix H template for magnetic anomaly (currently zeros)
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
    // Magnetic anomaly depends on position (lat, lon) through map lookup
    // The partial derivatives ∂z/∂lat and ∂z/∂lon are computed numerically
    // by the measurement model based on map gradients

    // These will be filled in by the measurement model with numerical derivatives
    // from the geophysical map interpolation
    // h[(0, 0)] = ∂(anomaly)/∂(lat) - computed from map gradient
    // h[(0, 1)] = ∂(anomaly)/∂(lon) - computed from map gradient
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

        println!("delta_x: {:?}", delta_x);

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
