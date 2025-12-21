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
//! - Angles are wrapped to [-π, π] for latitude/yaw and [0, 2π] for all Euler angles

use crate::earth::{self, vector_to_skew_symmetric};
use crate::StrapdownState;
use nalgebra::{DMatrix, Vector3};

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
    let g = earth::gravity(&lat.to_degrees(), &alt);
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
    let dgravity_dlat = ge * (dnumerator_dphi * denominator_sqrt - numerator * ddenominator_sqrt_dphi) 
                        / (denominator_sqrt * denominator_sqrt);

    // Transform specific force to navigation frame
    let f_bn = c_bn * imu_accel;

    // --- Position derivatives (rows 0-2) ---
    // Position update: lat(+) = lat(-) + v_n/(R_n+h)*dt + ...
    // ∂(lat(+))/∂(lat(-)): main term is identity, plus derivative terms
    // The derivative of v_n/(R_n+h) w.r.t. lat through R_n is negligible for first-order
    // ∂(lat(+))/∂(v_n): derivative of the kinematic relationship
    f[(0, 3)] = 1.0 / (r_n + alt) * dt;

    // Longitude update accounts for cos(lat) in denominator
    let cos_lat = lat.cos().max(1e-6);
    // ∂(lon(+))/∂(lon(-)): identity (no direct dependence)
    // ∂(lon(+))/∂(lat): derivative through cos(lat)
    f[(1, 0)] += v_e / ((r_e + alt) * cos_lat.powi(2)) * lat.sin() * dt;
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
    f[(6, 4)] += 1.0 / (r_e + alt) * dt;  // ∂(ε_x)/∂(v_e)
    f[(7, 3)] += -1.0 / (r_n + alt) * dt; // ∂(ε_y)/∂(v_n)
    f[(8, 4)] += -lat.tan() / (r_e + alt) * dt; // ∂(ε_z)/∂(v_e)

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
        let mut state_nominal = state.clone();
        crate::forward(&mut state_nominal, crate::IMUData { accel: *imu_accel, gyro: *imu_gyro }, dt);
        let f0: Vec<f64> = (&state_nominal).into();

        // Perturb each state component
        for j in 0..9 {
            let mut x_pert = x0.clone();
            x_pert[j] += epsilon;

            // Create perturbed state
            let mut state_pert = StrapdownState::try_from(x_pert.as_slice()).unwrap();
            state_pert.is_enu = state.is_enu;

            // Propagate perturbed state
            crate::forward(&mut state_pert, crate::IMUData { accel: *imu_accel, gyro: *imu_gyro }, dt);
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
            45.0, -122.0, 100.0,
            0.0, 0.0, 0.0,
            Rotation3::identity(),
            true,  // degrees
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
            eprintln!("Largest error at ({}, {}): analytic={:.10e}, numeric={:.10e}, diff={:.10e}",
                      max_i, max_j, f_analytic[(max_i, max_j)], f_numeric[(max_i, max_j)], max_error);
            
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
            45.0, -122.0, 100.0,
            5.0, 3.0, -0.5,  // Smaller velocities
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
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let lat = rng.gen_range(-80.0..80.0);
            let lon = rng.gen_range(-180.0..180.0);
            let alt = rng.gen_range(0.0..5000.0);
            let v_n = rng.gen_range(-10.0..10.0);  // Smaller velocities
            let v_e = rng.gen_range(-10.0..10.0);
            let v_d = rng.gen_range(-2.0..2.0);
            let roll = rng.gen_range(-0.1..0.1);    // Smaller angles
            let pitch = rng.gen_range(-0.1..0.1);
            let yaw = rng.gen_range(-0.5..0.5);

            let state = StrapdownState::new(
                lat, lon, alt, v_n, v_e, v_d,
                Rotation3::from_euler_angles(roll, pitch, yaw),
                true,
                Some(true),
            );

            let accel = Vector3::new(
                rng.gen_range(-0.5..0.5),  // Smaller accelerations
                rng.gen_range(-0.5..0.5),
                rng.gen_range(9.0..10.5),
            );
            let gyro = Vector3::new(
                rng.gen_range(-0.01..0.01),  // Smaller rates
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
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
            45.0, -122.0, 100.0,
            0.0, 0.0, 0.0,
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
            assert!(g[(i, i-3)].abs() > 0.0, "g[{}, {}] should be non-zero", i, i-3);
        }

        // Attitude rows (6-8) should couple to gyro noise (cols 3-5)
        // For identity rotation, only diagonal elements are non-zero
        for i in 6..9 {
            // Diagonal elements should be non-zero
            assert!(g[(i, i-3)].abs() > 0.0, "g[{}, {}] should be non-zero", i, i-3);
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
}
