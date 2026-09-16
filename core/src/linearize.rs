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
//!     None,                  // is_enu (None = the crate default, NED)
//! ).unwrap();
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
//! - Default frame is North-East-Down (NED); ENU is supported via the `is_enu` flag
//! - Gravity is positive down (NED) or negative up (ENU)
//! - Latitude is constrained to [-π/2, π/2] (±90°)
//! - Longitude and yaw are wrapped to [-π, π]
//! - Roll and pitch are typically in [-π, π] but may vary by implementation

use crate::StrapdownError;

/// Accelerometer and gyroscope bias corrections extracted from a 15-element error state.
pub type ImuBiasCorrection = (Vector3<f64>, Vector3<f64>);
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
///     None,                   // is_enu (defaults to NED)
/// ).unwrap();
/// let accel = Vector3::new(0.0, 0.0, 9.81);
/// let gyro = Vector3::new(0.0, 0.0, 0.0);
/// let dt = 0.01;
///
/// let f_matrix = state_transition_jacobian(&state, &accel, &gyro, dt);
/// assert_eq!(f_matrix.nrows(), 9);
/// assert_eq!(f_matrix.ncols(), 9);
/// ```
/// Map Euler-angle rates onto a navigation-frame rotation vector, $E(\Phi)$.
///
/// [`StrapdownState`] stores attitude as nalgebra's intrinsic XYZ Euler angles, so
/// $C_b^n = R_z(\psi) R_y(\theta) R_x(\phi)$ and the nav-frame angular velocity is
///
/// $$ \omega = \dot\psi \hat z + \dot\theta (R_z \hat y) + \dot\phi (R_z R_y \hat x) $$
///
/// whose columns are exactly this matrix. It is what converts a Jacobian written in
/// rotation-vector form into one with respect to the Euler angles the state actually holds.
///
/// Singular at $\theta = \pm 90°$ (gimbal lock), where the Euler parametrisation itself
/// stops being a chart; callers fall back to the rotation-vector form there.
fn euler_rate_matrix(roll: f64, pitch: f64, yaw: f64) -> nalgebra::Matrix3<f64> {
    let rz = Rotation3::from_axis_angle(&Vector3::z_axis(), yaw);
    let ry = Rotation3::from_axis_angle(&Vector3::y_axis(), pitch);
    let _ = roll; // the roll axis is the body x-axis carried through R_z R_y
    nalgebra::Matrix3::from_columns(&[(rz * ry) * Vector3::x(), rz * Vector3::y(), Vector3::z()])
}

/// How the IMU bias states enter the navigation states' transition Jacobian, $\partial x^+ /
/// \partial b$.
///
/// Returns `(velocity_block, attitude_block)`: the 3x3 blocks belonging at rows 3..6 against
/// columns 9..12, and rows 6..9 against columns 12..15, of a 15-state $F$.
///
/// # Why this exists
///
/// A 15-state filter that leaves these blocks zero does not have 15 states. With a
/// block-diagonal $P_0$ and measurement models that observe navigation states only, a zero
/// coupling block means `P[0..9, 9..15]` starts at zero and **can never become nonzero**, so
/// the Kalman gain over the bias rows is identically zero and the bias estimate never leaves
/// its seed. That was [`crate::kalman::ExtendedKalmanFilter`]'s state for the whole of its
/// history (#394): measured over 299 predict/update steps, its `max |P[nav, bias]|` was
/// **exactly 0.0** against the UKF's 4.3e-3, and its bias estimate was still all zeros.
///
/// The UKF needs none of this -- it gets the coupling for free, because a sigma point
/// perturbed in a bias state mechanizes to a different navigation state. Only a filter that
/// linearises has to supply it, and of the two that do, only the ESKF did.
///
/// # The two blocks
///
/// **Velocity.** Mechanization applies $C_b^n$ to the bias-corrected specific force, so
/// $\dot v^n = C_b^n (f^b - b_a) + \ldots$ and
///
/// $$ \frac{\partial v^+}{\partial b_a} = -C_b^n \, \Delta t $$
///
/// Velocity is a navigation-frame vector under either parametrisation, so this block is the
/// same matrix for both and needs no conversion.
///
/// **Attitude.** This is where the two filters genuinely differ, and it is not a sign.
/// [`AttitudeParametrization::RotationVector`] here means the **navigation-frame**
/// perturbation $\tilde C = (I + [\delta\theta\times]) C$ that the rest of
/// [`transition_jacobian`] is written in. A body-frame rate perturbation reaches it through
/// the attitude, so
///
/// $$ \frac{\partial \theta^+_n}{\partial b_g} = -C_b^n \, \Delta t $$
///
/// and for a state holding Euler angles the row converts out of rotation-vector space the
/// same way [`transition_jacobian`]'s attitude block does, through $E(\Phi^+)^{-1}$:
///
/// $$ \frac{\partial \Phi^+}{\partial b_g} = -E(\Phi^+)^{-1} C_b^n \, \Delta t $$
///
/// **This is not the ESKF's `-I`.** `error_state_transition_jacobian` writes
/// `f[(6,12)] = -dt` and so on, which is correct *there* because that matrix uses the
/// **body-frame** error convention $\tilde C = C (I + [\delta\theta\times])$ -- documented at
/// its velocity-attitude block -- under which $\delta\theta$ and $b_g$ live in the same frame
/// and the coupling is the bare identity. Two different conventions, two different matrices,
/// both right for their own filter. Transcribing one into the other would rotate the
/// gyro-bias observability onto the wrong axes at any non-zero heading, which is #266's
/// failure one block over.
///
/// Near gimbal lock $E$ is singular, and the rotation-vector form is kept -- wrong but
/// bounded -- rather than inverting a near-singular matrix. Same policy, and same reason, as
/// [`transition_jacobian`].
#[must_use]
pub fn bias_coupling_blocks(
    state: &StrapdownState,
    imu_gyro: &Vector3<f64>,
    dt: f64,
    attitude: AttitudeParametrization,
) -> (nalgebra::Matrix3<f64>, nalgebra::Matrix3<f64>) {
    let c_bn = *state.attitude.matrix();
    let velocity_block = -c_bn * dt;

    let attitude_block = if attitude == AttitudeParametrization::Euler {
        let next_attitude = crate::attitude_update(state, *imu_gyro * dt, dt);
        let next_rotation = Rotation3::from_matrix_unchecked(next_attitude);
        let (next_roll, next_pitch, next_yaw) = next_rotation.euler_angles();
        euler_rate_matrix_inverse(next_roll, next_pitch, next_yaw)
            .map_or(velocity_block, |next_inverse| next_inverse * velocity_block)
    } else {
        velocity_block
    };

    (velocity_block, attitude_block)
}

/// Largest factor the Euler conversion may amplify a Jacobian block by before it is refused.
///
/// $E(\Phi)^{-1}$ grows as $1/\cos\theta$, so it is unbounded at gimbal lock. Measured on
/// `euler_rate_matrix`, the largest entry of the inverse is very nearly $0.955/\cos\theta$:
///
/// | pitch | 0 deg | 45 | 80 | 89 | 89.9 | 89.99 | 90 |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | max abs entry | 1.00 | 1.35 | 5.50 | 54.7 | 547 | 5,474 | **1.6e16** |
///
/// 100 admits everything up to about 89.45 degrees of pitch, which is past anything a vehicle
/// trajectory reaches and well short of where the conversion stops meaning anything.
const MAX_EULER_RATE_AMPLIFICATION: f64 = 100.0;

/// $E(\Phi)^{-1}$, or `None` when the Euler parametrisation is too close to gimbal lock for it
/// to mean anything.
///
/// # Why `try_inverse` alone is not the guard
///
/// Because it never fires. `try_inverse` returns `None` only for a matrix that is *exactly*
/// singular, and `euler_rate_matrix` is not exactly singular even at a pitch of exactly 90
/// degrees -- it returns an inverse whose largest entry is **1.6e16**. Both callers documented
/// a fallback to the bounded rotation-vector form "rather than inverting a near-singular
/// matrix", and both then inverted the near-singular matrix, because the condition they
/// branched on was never true. This checks the quantity that actually matters -- how big the
/// conversion's entries are -- rather than a singularity test that a near-singular matrix
/// passes.
fn euler_rate_matrix_inverse(roll: f64, pitch: f64, yaw: f64) -> Option<nalgebra::Matrix3<f64>> {
    let inverse = euler_rate_matrix(roll, pitch, yaw).try_inverse()?;
    (inverse.abs().max() <= MAX_EULER_RATE_AMPLIFICATION).then_some(inverse)
}

/// How a Jacobian's attitude columns are parametrised.
///
/// The two consumers of the transition Jacobian hold attitude differently, and the blocks
/// that touch attitude are genuinely different matrices as a result. Naming the choice keeps
/// each filter's convention visible at the call site instead of implied by which function it
/// happens to call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttitudeParametrization {
    /// Attitude perturbations are a nav-frame rotation vector.
    ///
    /// What an error-state filter carries, and the form `δ(C f) = -[f^n×] δθ` is written in.
    RotationVector,
    /// Attitude perturbations are increments of the stored XYZ Euler angles.
    ///
    /// What a full-state filter whose state vector holds roll, pitch and yaw carries. Differs
    /// from [`Self::RotationVector`] by the Euler-rate matrix `E(Φ)`, and the difference is
    /// the same order as the terms themselves -- not a refinement (#307).
    Euler,
}

/// Full-state transition Jacobian with rotation-vector attitude perturbations.
///
/// For error-state filters, whose attitude correction is a rotation vector. A full-state
/// filter storing Euler angles wants [`euler_state_transition_jacobian`] instead.
#[must_use]
pub fn state_transition_jacobian(
    state: &StrapdownState,
    imu_accel: &Vector3<f64>,
    imu_gyro: &Vector3<f64>,
    dt: f64,
) -> DMatrix<f64> {
    transition_jacobian(
        state,
        imu_accel,
        imu_gyro,
        dt,
        AttitudeParametrization::RotationVector,
    )
}

/// Full-state transition Jacobian with Euler-angle attitude perturbations.
///
/// For a filter whose state vector holds roll, pitch and yaw directly -- the EKF. Using the
/// rotation-vector form here is what drove #307: on a typical state a roll perturbation moves
/// north velocity by 7.7e-2 where the rotation-vector form predicts exactly zero, so the gain
/// was computed against the wrong sensitivity on every step and the filter diverged to
/// ~14,707 km on `core/tests/test_data.csv`.
#[must_use]
pub fn euler_state_transition_jacobian(
    state: &StrapdownState,
    imu_accel: &Vector3<f64>,
    imu_gyro: &Vector3<f64>,
    dt: f64,
) -> DMatrix<f64> {
    transition_jacobian(
        state,
        imu_accel,
        imu_gyro,
        dt,
        AttitudeParametrization::Euler,
    )
}

fn transition_jacobian(
    state: &StrapdownState,
    imu_accel: &Vector3<f64>,
    imu_gyro: &Vector3<f64>,
    dt: f64,
    attitude: AttitudeParametrization,
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
    // Degrees, like the `to_degrees()` calls immediately below -- `lat` is radians (#292).
    let (r_n, r_e, _) = earth::principal_radii(&lat.to_degrees(), &alt);
    let _g = earth::gravity(&lat.to_degrees(), &alt); // Reserved for future use
    let omega_ie = earth::earth_rate_lla(&lat.to_degrees());
    let omega_en = earth::transport_rate(&lat.to_degrees(), &alt, &vel);
    // Both are NED, and the blocks below multiply them against *this state's* velocity and
    // attitude errors. Reflect them into the caller's convention first -- as pseudovectors,
    // see `flip_vertical_rate` -- or the Coriolis and attitude blocks describe a different
    // frame from the mechanization they are supposed to linearise. `mechanize` canonicalises
    // to NED as of #321; without this the covariance stopped tracking it for ENU states.
    //
    // `transport_rate` reads only the horizontal components of `vel`, and those are the same
    // in both conventions, so it needs no conversion on the way in.
    let (omega_ie, omega_en) = if state.is_enu {
        (
            crate::flip_vertical_rate(&omega_ie),
            crate::flip_vertical_rate(&omega_en),
        )
    } else {
        (omega_ie, omega_en)
    };

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

    // The Euler parametrisation at this state, and its inverse at the propagated state.
    //
    // Both attitude-bearing blocks below are naturally written for a rotation-vector
    // perturbation. When the caller's state holds Euler angles they have to be converted;
    // when it holds a rotation vector they are already right, and `E` is the identity here.
    // Near gimbal lock `E` is singular and no conversion exists, so the rotation-vector form
    // is kept -- wrong but bounded -- rather than inverting a near-singular matrix.
    let use_euler = attitude == AttitudeParametrization::Euler;
    let euler_matrix = if use_euler {
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        euler_rate_matrix(roll, pitch, yaw)
    } else {
        nalgebra::Matrix3::identity()
    };
    let euler_matrix_next_inverse = if use_euler {
        let next_attitude = crate::attitude_update(state, *imu_gyro * dt, dt);
        let next_rotation = Rotation3::from_matrix_unchecked(next_attitude);
        let (next_roll, next_pitch, next_yaw) = next_rotation.euler_angles();
        euler_rate_matrix_inverse(next_roll, next_pitch, next_yaw)
    } else {
        Some(nalgebra::Matrix3::identity())
    };

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

    // Altitude update: simple kinematic.
    //
    // `altitude` is height above the ellipsoid -- positive *up* -- in both frames, but
    // `velocity_vertical` is positive *down* in NED. `position_update` integrates it with
    // exactly this sign; the Jacobian has to agree or the filter believes climbing and
    // descending are swapped. This was unconditionally `+dt`, which is right in ENU and
    // backwards in NED -- the default frame since queue 3 (#307).
    f[(2, 5)] = if state.is_enu { dt } else { -dt };

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

    // ...and the *other* half of the same derivative, because ω_en is itself a function of v.
    //
    // 5.54's Coriolis and transport term is quadratic in velocity -- 5.44 makes ω_en linear
    // in v -- so
    //
    //     ∂/∂v_j [ -(2ω_ie + ω_en(v)) × v ] = -(2ω_ie + ω_en) × e_j - (∂ω_en/∂v_j) × v
    //
    // and the skew matrix above is only the first half: column j of `coriolis_transport` is
    // exactly -(2ω_ie + ω_en) × e_j. This is not an O(dt²) truncation that a shorter step
    // would shrink relative to what is kept -- it scales with dt exactly as the kept half
    // does, so no choice of dt separates them. At the 120 m/s state in this file's tests the
    // two halves are the same order, which left f[(5,3)] -- vertical-velocity response to
    // north-velocity error -- wrong by a factor of two. See #325.
    //
    // Frame handling is the pseudovector reflection `omega_en` itself receives above, applied
    // to each column. With F = diag(1, 1, -1) the stored-ENU block is F B F, and since
    // F(a × b) = -(Fa) × (Fb) while `flip_vertical_rate(w) = -F w`, column j reduces to
    // -flip_vertical_rate(∂ω_en/∂v_j) × v_stored: the naive substitution, for the same reason
    // the transport-coupling comment further down says the gradient "picks up -F on the left".
    // The columns it reads -- north and east velocity -- are identical in both conventions.
    // `transition_jacobian_agrees_across_vertical_conventions` pins this to 1e-18 and is the
    // oracle for it; skipping the reflection, or using `flip_vertical`, fails there first.
    for (column, gradient) in transport_rate_velocity_gradients(lat, alt)
        .iter()
        .enumerate()
    {
        let gradient = if state.is_enu {
            crate::flip_vertical_rate(gradient)
        } else {
            *gradient
        };
        let contribution = -gradient.cross(&vel) * dt;
        for (row, value) in contribution.iter().enumerate() {
            f[(3 + row, 3 + column)] += *value;
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

    // ∂(v(+))/∂(position) through the Coriolis and transport rates as well.
    //
    // Gravity is not the only position dependence the velocity rows have: 5.54 applies
    // -(2ω_ie + ω_en) × v, ω_ie = Ω[cos φ, 0, -sin φ] is a function of latitude outright, and
    // ω_en carries latitude through tan φ and the principal radii and altitude weakly through
    // the radii. Both were modelled as exactly zero, so f[(3,0)] and f[(4,0)] were never
    // assigned and the Coriolis share of f[(5,0)] -- 0.7 % of the gravity term there, and of
    // the opposite sign -- was missing too. See #317.
    //
    // Small: ~1e-5 per radian at 12 m/s, an order more at 120 m/s, against blocks of ~1e-1
    // this file already gets right. They are carried anyway for the reason #286, #303 and
    // #307 each settled the same way -- a term the nominal propagation actually applies
    // belongs in F, or the covariance stops describing the real error dynamics -- and because
    // leaving them out is what kept `core/tests/jacobian_agreement.rs` a smoke test.
    //
    // Longitude is genuinely absent rather than omitted, and so is ∂v/∂alt for the horizontal
    // rows in practice: WGS84 gravity and the principal radii are axisymmetric, and the
    // altitude column comes out at ~1e-14 against velocities of 12 m/s. It falls out of the
    // same cross product for free, so it is written rather than special-cased, but it does
    // nothing.
    let (d_velocity_dlat, d_velocity_dalt) = velocity_position_coupling(state, dt);
    for row in 0..3 {
        f[(3 + row, 0)] += d_velocity_dlat[row];
        f[(3 + row, 2)] += d_velocity_dalt[row];
    }

    // ∂(v(+))/∂(attitude): transformation of specific force.
    //
    // `δ(C f) = -[f^n×] δθ` holds for `δθ` a **rotation vector**, but this filter's state
    // holds Euler angles, and the two are not interchangeable: at this state a roll
    // perturbation moves `v_n` by 7.7e-2 where the rotation-vector form predicts exactly
    // zero. Composing with `E(Φ)` converts the rotation-vector Jacobian into one with
    // respect to the angles actually stored, which is what the EKF's covariance needs.
    //
    // This was the dominant error behind #307: the mismatch is the same order as the terms
    // themselves, so the EKF's gain was computed against the wrong sensitivity on every step.
    let f_bn_skew = vector_to_skew_symmetric(&f_bn);
    let d_velocity_d_euler = -f_bn_skew * euler_matrix;
    for i in 0..3 {
        for j in 0..3 {
            f[(3 + i, 6 + j)] += d_velocity_d_euler[(i, j)] * dt;
        }
    }

    // --- Attitude derivatives (rows 6-8) ---
    // Attitude update: C(+) = C(-) * (I + Ω_ib*dt) - (Ω_ie + Ω_en) * C(-) * dt
    // In error-state formulation: δε(+) ≈ δε(-) - (Ω_ie + Ω_en) × δε(-) * dt
    // This gives: Φ_ε = I - [Ω_ie + Ω_en]× * dt

    // Same conversion on the attitude block itself. In rotation-vector form the transition is
    // `I - [ω_in×] dt`; in Euler form it is that, sandwiched between the parametrisation at
    // the output and input points: `E(Φ⁺)⁻¹ (I - [ω_in×] dt) E(Φ⁻)`. The sandwich is not a
    // refinement -- the body rotation dominates it, so the off-diagonal terms it produces are
    // ~2.9e-4 here against the ~7e-7 the rotation-vector form alone gives.
    let omega_in_skew = omega_ie_skew + omega_en_skew;
    let rotation_vector_transition = nalgebra::Matrix3::identity() - omega_in_skew * dt;
    let attitude_transition = euler_matrix_next_inverse.map_or_else(
        || rotation_vector_transition,
        |next_inverse| next_inverse * rotation_vector_transition * euler_matrix,
    );
    for i in 0..3 {
        for j in 0..3 {
            f[(6 + i, 6 + j)] = attitude_transition[(i, j)];
        }
    }

    // --- ∂(attitude(+))/∂(position) and ∂(attitude(+))/∂(v): through ω_in ---
    //
    // 5.46 propagates `C+ = C + C[ω_ib x] dt - [ω_in x] C dt`, where ω_in = ω_ie + ω_en.
    // Perturbing any state element that ω_in is a function of therefore gives
    // `δC+ = -[δω_in x] C dt`, and against this parametrisation's nav-frame perturbation
    // `C~ = (I + [δθ x]) C` that reads
    //
    //     δθ+ = -δω_in dt
    //
    // so every column here is minus the corresponding gradient of ω_in, times dt -- no cross
    // product, unlike the velocity rows, because 5.46 applies the rate to `C` rather than to
    // `v`. 5.44 makes ω_en linear in the horizontal velocity and both rates carry latitude
    // (ω_en also altitude, weakly, through the principal radii), so the block spans columns
    // 0, 2, 3 and 4. Longitude and the vertical-velocity column are genuinely absent: WGS84's
    // principal radii are axisymmetric and `transport_rate` never reads `v_D`.
    //
    // The **velocity** columns are minus dω_en/dv dt, and used to be plus. Finite-differencing
    // `mechanize` returns exactly the negation of the old entries, at every dt and in both
    // frames -- see
    // `transition_jacobian_velocity_columns_match_finite_differences_in_both_frames`.
    //
    // The **position** columns were modelled as exactly zero until #339: the same latitude
    // dependence #317 gave the velocity rows out of 5.54, applied to 5.46 and left undone.
    // The only difference between the two is the coefficient on ω_ie -- one here against
    // 5.54's two -- which is why `rate_position_gradients` hands back the two gradients
    // separately rather than pre-summed. Small (~5.4e-7 per radian on
    // `core/tests/jacobian_agreement.rs`'s state, two orders under that file's bound) but
    // first order in dt, so no shorter step shrinks it relative to what is kept, and it is
    // carried for the reason #286, #303, #307 and #317 each settled the same way: a term the
    // nominal propagation actually applies belongs in F.
    //
    // Frame handling is the pseudovector reflection `omega_ie` and `omega_en` receive above,
    // applied to each gradient -- `flip_vertical_rate`, not `flip_vertical`, since δθ and ω
    // are both pseudovectors while latitude, altitude and the horizontal velocities are
    // frame-independent scalars. For the velocity columns that reproduces the hand-rolled
    // sign this block used to carry: `-F` on the left flips rows 6 and 7 and leaves row 8,
    // the vertical component, alone. `transition_jacobian_agrees_across_vertical_conventions`
    // pins all of it to 1e-18.
    let (attitude_latitude, attitude_altitude) = attitude_position_coupling(state, dt);
    let mut attitude_coupling = nalgebra::Matrix3x6::zeros();
    attitude_coupling.set_column(0, &attitude_latitude);
    attitude_coupling.set_column(2, &attitude_altitude);
    for (column, gradient) in transport_rate_velocity_gradients(lat, alt)
        .iter()
        .enumerate()
    {
        let gradient = if state.is_enu {
            crate::flip_vertical_rate(gradient)
        } else {
            *gradient
        };
        attitude_coupling.set_column(3 + column, &(-gradient * dt));
    }

    // The whole block is written in rotation-vector form above, and the Euler caller's
    // attitude rows are increments of the stored angles, so it needs `E(Φ⁺)⁻¹` on the left
    // -- the same conversion the attitude/attitude block gets, without the right-hand factor,
    // because the *inputs* here are position and velocity and carry no parametrisation of
    // their own. The velocity columns used to be written straight into `f` and so were the
    // one part of the attitude rows that never saw this: in the Euler form they came out
    // sign-flipped and of the same order (f[(7,3)] +1.568e-9 against a numeric -1.156e-9),
    // which is the signature of a missing left factor. Near gimbal lock, where no conversion
    // exists, the rotation-vector form is kept -- wrong but bounded -- exactly as above.
    let attitude_coupling = euler_matrix_next_inverse.map_or(attitude_coupling, |next_inverse| {
        next_inverse * attitude_coupling
    });
    for row in 0..3 {
        for column in 0..6 {
            f[(6 + row, column)] += attitude_coupling[(row, column)];
        }
    }

    // --- Not done here: the position rows' half-step over the *updated* velocity ---
    //
    // `position_update` integrates all three position rows trapezoidally over the propagated
    // velocity (Groves 5.56), so each inherits every dependence that velocity has at half a
    // step -- `f[(row, c)] += <rate> * 0.5 * dt * f[(3 + row, c)]`. That is an identity
    // rather than a model choice, and it is the largest remaining disagreement in the matrix:
    // ~3e-5 for altitude-vs-attitude on the `jacobian_agreement` state, against an analytic
    // zero.
    //
    // It is deliberately NOT applied, because doing only the altitude row -- which is where
    // it is largest, and all #317 asks for -- is the same inconsistency this file declines in
    // `error_state_transition_jacobian` below: rows 0 and 1 carry the identical term
    // (measured ~9.2e-11 at `frame_check_state`, 3.7% of the largest non-identity entry in
    // those rows) and would be left first-order while row 2 became second-order.
    //
    // That used to be the *second* reason. The first was a measurement -- adding the
    // altitude row alone took `rbpf::tests::rbpf_runs_on_scenario_stationary` from 13.98 m
    // of stationary altitude error to 35.92 m -- and it is void. Both numbers came from a
    // filter whose vertical channel was receiving no aiding at all, because
    // `generate_scenario_data` handed it a 45 um GPS fix and the particle weights collapsed
    // on the horizontal channel alone; see that test for the whole of #295. Re-measured with
    // the fix units corrected, on the three scenarios (stationary, v north, v east), final
    // altitude error in metres:
    //
    //     without the half-step    0.0122   0.0106   0.0301
    //     with it                  0.0031   0.0166   0.0026
    //
    // Two better, one worse, all four hundredths of a metre against a posterior sigma of
    // 0.894 m -- which is to say it is now below the noise rather than worth 22 m either
    // way. It is a real term and nothing here argues against it any more except consistency
    // across the three rows.
    //
    // Tracked in #338, with the measurements, as one change across all three rows.

    f
}

/// Magnitude of the vertical gravity gradient, |∂g/∂h|, in s^-2.
///
/// Matches the linear altitude term in [`earth::gravity`], which is
/// `g0(lat) - 3.08e-6 * altitude`.
const GRAVITY_ALTITUDE_GRADIENT: f64 = 3.08e-6;

/// Velocity derivatives of the transport rate, resolved in NED.
///
/// Groves 5.44 resolves the transport rate as
///
/// $$ \omega_{en}^n = \left[ \frac{v_E}{R_E + h}, \; \frac{-v_N}{R_N + h}, \;
/// \frac{-v_E \tan\varphi}{R_E + h} \right]^T $$
///
/// which makes 5.54's Coriolis and transport acceleration $-(2\omega_{ie} + \omega_{en})
/// \times v$ *quadratic* in velocity. Its velocity derivative therefore has two halves,
///
/// $$ \frac{\partial}{\partial v_j} \left[ -(2\omega_{ie} + \omega_{en}(v)) \times v \right]
/// = -(2\omega_{ie} + \omega_{en}) \times e_j \; - \; \frac{\partial \omega_{en}}{\partial
/// v_j} \times v $$
///
/// and this function supplies the gradient in the second, one [`Vector3`] per column $j$ of
/// the velocity block. The vertical column is identically zero -- [`earth::transport_rate`]
/// reads only the horizontal velocity -- and is returned anyway so callers can index all
/// three uniformly.
///
/// Resolved in NED, like [`earth::transport_rate`] itself. Callers reflect each column with
/// `flip_vertical_rate` for an ENU state, exactly as they reflect the rate.
fn transport_rate_velocity_gradients(latitude_rad: f64, altitude: f64) -> [Vector3<f64>; 3] {
    let (meridian_radius, transverse_radius, _) =
        earth::principal_radii(&latitude_rad.to_degrees(), &altitude);
    let north_radius = meridian_radius + altitude;
    let east_radius = transverse_radius + altitude;
    [
        // ∂ω_en/∂v_N: only the second component carries v_N.
        Vector3::new(0.0, -1.0 / north_radius, 0.0),
        // ∂ω_en/∂v_E: the first and third.
        Vector3::new(1.0 / east_radius, 0.0, -latitude_rad.tan() / east_radius),
        // ∂ω_en/∂v_D: the transport rate does not see the vertical channel at all.
        Vector3::zeros(),
    ]
}

/// Position derivatives of the Earth and transport rates, resolved in NED.
///
/// Groves 5.54 applies $-(2\omega_{ie} + \omega_{en}) \times v$ and 5.46 applies
/// $-(\omega_{ie} + \omega_{en}) \times C$. Both rates depend on latitude -- $\omega_{ie}$
/// directly through
///
/// $$ \omega_{ie}^n = \Omega_E \left[ \cos\varphi, \; 0, \; -\sin\varphi \right]^T $$
///
/// and $\omega_{en}$ through $\tan\varphi$ and the principal radii -- and $\omega_{en}$ weakly
/// on altitude, through those same radii. Gravity is therefore not the only position
/// dependence the velocity rows have, which is what #317 records, and the attitude rows have
/// one for the same reason, which is what #339 records.
///
/// Both consumers take the two latitude gradients from here rather than a pre-summed one,
/// because that is the *only* thing that differs between the two equations: 5.54 weights
/// $\omega_{ie}$ by two and 5.46 by one. See [`velocity_position_coupling`] and
/// [`attitude_position_coupling`].
///
/// Returns $(\partial \omega_{ie} / \partial \varphi, \; \partial \omega_{en} / \partial
/// \varphi, \; \partial \omega_{en} / \partial h)$, all NED and all per radian or per metre.
/// Callers reflect them with `flip_vertical_rate` for an ENU state, exactly as they reflect
/// the rates themselves: latitude and altitude are frame-independent scalars, so the
/// derivative of a pseudovector reflects the way the pseudovector does.
///
/// The radius derivatives come from $R_E = a D^{-1/2}$ and $R_N = a(1 - e^2) D^{-3/2}$ with
/// $D = 1 - e^2 \sin^2\varphi$, so both are the radius itself times $e^2 \sin\varphi
/// \cos\varphi / D$, and $R_N$ carries a factor of three from the steeper exponent.
fn rate_position_gradients(
    latitude_rad: f64,
    altitude: f64,
    velocity_north: f64,
    velocity_east: f64,
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let sin_lat = latitude_rad.sin();
    // The latitude derivative of the vertical transport component carries sec²φ, so it
    // diverges faster at the pole than the tan φ already in the attitude block. Guarded the
    // same way `transition_jacobian` and `position_update` guard theirs: a large number in
    // the covariance rather than an infinity.
    let cos_lat = latitude_rad.cos().max(1e-6);
    let (meridian_radius, transverse_radius, _) =
        earth::principal_radii(&latitude_rad.to_degrees(), &altitude);
    let denominator = 1.0 - earth::ECCENTRICITY_SQUARED * sin_lat * sin_lat;
    let radius_scale = earth::ECCENTRICITY_SQUARED * sin_lat * cos_lat / denominator;
    let transverse_gradient = transverse_radius * radius_scale;
    let meridian_gradient = 3.0 * meridian_radius * radius_scale;
    let north_radius = meridian_radius + altitude;
    let east_radius = transverse_radius + altitude;
    // Every entry below differentiates a `1/(R + h)`, so every entry carries `(R + h)^-2`.
    // Named rather than written inline because `clippy::suspicious_operation_groupings`
    // reads `a * b / (c * c)` as a likely typo for `a * c / (b * c)`.
    let north_radius_squared = north_radius * north_radius;
    let east_radius_squared = east_radius * east_radius;
    let tan_lat = sin_lat / cos_lat;

    let earth_rate_latitude = Vector3::new(-earth::RATE * sin_lat, 0.0, -earth::RATE * cos_lat);
    let transport_latitude = Vector3::new(
        -velocity_east * transverse_gradient / east_radius_squared,
        velocity_north * meridian_gradient / north_radius_squared,
        -velocity_east / (cos_lat * cos_lat * east_radius)
            + velocity_east * tan_lat * transverse_gradient / east_radius_squared,
    );
    let transport_altitude = Vector3::new(
        -velocity_east / east_radius_squared,
        velocity_north / north_radius_squared,
        velocity_east * tan_lat / east_radius_squared,
    );
    (earth_rate_latitude, transport_latitude, transport_altitude)
}

/// The Coriolis and transport contribution to the velocity rows' position columns.
///
/// Shared by [`transition_jacobian`] and [`error_state_transition_jacobian`] so the two
/// cannot drift apart: both linearise the same line of Groves 5.54, and both hold position as
/// (latitude, longitude, altitude) in radians and metres, so the latitude derivative goes into
/// column 0 unscaled in either. Longitude is exactly absent -- WGS84 gravity and the principal
/// radii are axisymmetric, so nothing in the mechanization depends on it.
///
/// Returns `(∂v/∂latitude, ∂v/∂altitude)` for one step of length `dt`, already reflected into
/// `state`'s vertical convention.
fn velocity_position_coupling(state: &StrapdownState, dt: f64) -> (Vector3<f64>, Vector3<f64>) {
    let velocity = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_vertical,
    );
    let (earth_rate_latitude, transport_latitude, transport_altitude) = rate_position_gradients(
        state.latitude,
        state.altitude,
        state.velocity_north,
        state.velocity_east,
    );
    let rate_latitude = 2.0 * earth_rate_latitude + transport_latitude;
    let (rate_latitude, rate_altitude) = if state.is_enu {
        (
            crate::flip_vertical_rate(&rate_latitude),
            crate::flip_vertical_rate(&transport_altitude),
        )
    } else {
        (rate_latitude, transport_altitude)
    };
    (
        -rate_latitude.cross(&velocity) * dt,
        -rate_altitude.cross(&velocity) * dt,
    )
}

/// The Earth- and transport-rate contribution to the *attitude* rows' position columns.
///
/// Companion to [`velocity_position_coupling`], built from the same two gradients. Groves
/// 5.46 applies $-(\omega_{ie} + \omega_{en}) \times C$ where 5.54 applies
/// $-(2\omega_{ie} + \omega_{en}) \times v$, so this differs from that function in exactly
/// two ways: the coefficient on the Earth rate is one rather than two, and the summed
/// gradient *is* the answer instead of being crossed into velocity. Against this
/// parametrisation's nav-frame perturbation $\tilde C = (I + [\delta\theta \times]) C$,
/// 5.46 reads
///
/// $$ \delta\theta^+ = -\delta\omega_{in} \, \Delta t $$
///
/// directly, so one step's sensitivity to a position error is minus the rate's position
/// gradient, times $\Delta t$.
///
/// Returns `(∂θ/∂latitude, ∂θ/∂altitude)` for one step of length `dt`, as a nav-frame
/// rotation vector already reflected into `state`'s vertical convention -- $\delta\theta$
/// and $\omega$ are both pseudovectors and latitude and altitude are frame-independent
/// scalars, so the derivative reflects the way the rate itself does. Longitude is absent for
/// the reason it is absent from the velocity rows: WGS84's principal radii are axisymmetric,
/// so nothing in either rate depends on it.
///
/// This is the rotation-vector form, like [`rate_position_gradients`] itself. A caller
/// holding Euler angles must compose it with $E(\Phi^+)^{-1}$, the same way the attitude
/// block proper is composed.
fn attitude_position_coupling(state: &StrapdownState, dt: f64) -> (Vector3<f64>, Vector3<f64>) {
    let (earth_rate_latitude, transport_latitude, transport_altitude) = rate_position_gradients(
        state.latitude,
        state.altitude,
        state.velocity_north,
        state.velocity_east,
    );
    let rate_latitude = earth_rate_latitude + transport_latitude;
    let (rate_latitude, rate_altitude) = if state.is_enu {
        (
            crate::flip_vertical_rate(&rate_latitude),
            crate::flip_vertical_rate(&transport_altitude),
        )
    } else {
        (rate_latitude, transport_altitude)
    };
    (-rate_latitude * dt, -rate_altitude * dt)
}

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
/// # Block Structure of `F_δx` (15×15)
///
/// ```text
/// F = | F_pp  F_pv  F_pθ   0     0   |  (position)
///     | F_vp  F_vv  F_vθ  F_vba  0   |  (velocity)
///     | F_θp  F_θv  F_θθ   0    F_θbg|  (attitude)
///     |  0     0     0    F_bb   0   |  (accel bias)
///     |  0     0     0     0    F_bb |  (gyro bias)
/// ```
///
/// where most blocks are sparse and bias dynamics are random walk (`F_bb` = 0).
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
/// 15×15 error-state transition Jacobian matrix `F_δx`
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
///  ).unwrap();
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
    // ∂(δaltitude)/∂(δv_vertical), and it changes sign with the frame.
    //
    // The third position error is an **altitude** error: `ErrorStateKalmanFilter::
    // inject_error_state` adds it straight onto `nominal_altitude`, which is height above the
    // ellipsoid -- positive *up* -- in both frames. `velocity_vertical` is positive *down* in
    // NED, and `position_update` integrates it with a matching sign flip.
    //
    // This was unconditionally `+dt`. In NED that tells the filter a positive vertical
    // velocity *raises* altitude when it lowers it, which turns the altitude/vertical-velocity
    // pair into positive feedback: the vertical channel then grows without bound from any
    // non-zero seed error, while a run seeded exactly on truth stays stable because nothing
    // ever excites it. That is precisely the signature reported in #303. NED has been the
    // default frame since queue 3.
    f[(2, 5)] = if state.is_enu { dt } else { -dt };

    // Note: Position error doesn't directly depend on attitude error or biases.
    //
    // The full-state `transition_jacobian` does carry ∂alt/∂(everything row 5 depends on),
    // because `position_update` integrates the *updated* vertical velocity trapezoidally and
    // that half-step is an identity, not a model choice. It is deliberately not ported here.
    // These rows are the first-order rate form -- f[(0,3)] = dt/R_N, with no trapezoid
    // either, pinned to 1e-15 by `test_error_state_jacobian_position_rows_are_radians` --
    // so adding the half-step to row 2 alone would make one of the three position rows
    // second-order and leave the other two first-order. Both omissions are the same O(dt²),
    // and correcting them is one change to make together or not at all.

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

    // ...and the rest of ∂(δv̇)/∂(δp), through the two rates rather than through gravity.
    //
    // Same terms the full-state Jacobian carries; shared with it so the two linearisations of
    // 5.54 cannot drift apart. The position error here is (δlatitude, δlongitude, δaltitude)
    // in radians and metres -- see `ErrorStateKalmanFilter::inject_error_state` and
    // `test_error_state_jacobian_position_rows_are_radians` -- so the latitude derivative goes
    // into column 0 unscaled, exactly as it does there. Note the `+=`: the two gravity entries
    // above are assignments and this adds the Coriolis share on top of f[(5,0)] and f[(5,2)].
    let (d_velocity_dlat, d_velocity_dalt) = velocity_position_coupling(state, dt);
    for row in 0..3 {
        f[(3 + row, 0)] += d_velocity_dlat[row];
        f[(3 + row, 2)] += d_velocity_dalt[row];
    }

    // ∂(δv̇)/∂(δv): the Coriolis and transport block, both halves of it.
    //
    // This block used to be absent, with the note that the coupling is "small for low
    // dynamics and often approximated as zero". The decision #325 asks for is to carry it,
    // for three reasons:
    //
    // 1. It is in the reference model. Groves section 14.2.4 -- already cited by this
    //    function's doc comment -- carries -(Ω_en + 2Ω_ie) as the velocity/velocity block of
    //    the local-navigation-frame error dynamics, beside the velocity/position and
    //    velocity/attitude blocks this function already has. There is no reading under which
    //    the rule that kept the gravity gradient (#286) drops the Coriolis term sitting in
    //    the same line of 5.54.
    // 2. It is not a small-dynamics approximation so much as a small-*duration* one. The
    //    entries are ~1.5e-6 at dt = 0.01, but they are systematic rather than noise: this is
    //    the Schuler and Foucault coupling, and over a thousand steps it rotates the velocity
    //    error covariance by ~1.5e-3 rad in a direction the ESKF otherwise never models.
    // 3. Leaving it out made the ESKF and the EKF disagree about the same physics, which is
    //    the condition #266, #286 and #307 each arose from.
    //
    // Both halves are included, for the reason spelled out at the corresponding block of
    // `transition_jacobian`: ω_en is a function of v, so 5.54's term is quadratic in it and
    // the skew matrix alone is only ∂/∂v_j of the second factor.
    //
    // Cost: the ESKF's covariance moves, by ~1.5e-6 per step on the velocity block. Measured
    // end to end on `core/tests/test_data.csv` by `test_rmse_benchmark_across_filters`, the
    // ESKF's horizontal RMSE is unchanged at 23.49 m and its vertical at 2.40 m; pitch moves
    // by 0.001 deg and yaw by 0.001 deg. `core/tests/filter_comparison.rs` and
    // `core/tests/aiding.rs` pass unchanged. The payoff is not accuracy -- it is that F now
    // describes the dynamics `mechanize` actually applies.
    let velocity = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_vertical,
    );
    let omega_ie = earth::earth_rate_lla(&lat_deg);
    let omega_en = earth::transport_rate(&lat_deg, &h, &velocity);
    // Both rates come out of `earth` in NED and are applied here against *this* state's
    // velocity error, so they are reflected for an ENU state exactly as `transition_jacobian`
    // reflects them -- as pseudovectors, see `flip_vertical_rate` (#321).
    let (omega_ie, omega_en) = if state.is_enu {
        (
            crate::flip_vertical_rate(&omega_ie),
            crate::flip_vertical_rate(&omega_en),
        )
    } else {
        (omega_ie, omega_en)
    };
    let coriolis_transport =
        -(2.0 * vector_to_skew_symmetric(&omega_ie) + vector_to_skew_symmetric(&omega_en));
    for i in 0..3 {
        for j in 0..3 {
            f[(3 + i, 3 + j)] += coriolis_transport[(i, j)] * dt;
        }
    }
    for (column, gradient) in transport_rate_velocity_gradients(lat, h).iter().enumerate() {
        let gradient = if state.is_enu {
            crate::flip_vertical_rate(gradient)
        } else {
            *gradient
        };
        let contribution = -gradient.cross(&velocity) * dt;
        for (row, value) in contribution.iter().enumerate() {
            f[(3 + row, 3 + column)] += *value;
        }
    }

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
/// where w ~ N(0, `Q_c`) is white noise representing IMU errors.
///
/// The process noise covariance in discrete time is: `Q_d` = G * `Q_c` * G^T * dt
///
/// # Arguments
///
/// * `state` - Current navigation state (affects rotation matrix)
/// * `dt` - Time step in seconds
///
/// # Returns
///
/// 9×6 process noise Jacobian matrix G, mapping [`accel_noise`; `gyro_noise`] to state
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
/// ).unwrap();
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

/// Measurement Jacobian for a barometer whose bias the filter estimates, full width.
///
/// $h(\mathbf{x}) = alt + b$, so the row is a 1 in the altitude column and a 1 in the bias
/// column, and everything else is zero.
///
/// # Why this returns the filter's full width
///
/// The same reason [`zaru_jacobian`] does, and it is load-bearing rather than stylistic. The
/// other Jacobians in this module return nine columns and let
/// `expand_measurement_jacobian` pad them on the right. Padding a nine-column row out to the
/// filter's width puts a **zero** in the bias column, which is not an error and not detected:
/// the bias would be unobservable, its gain identically zero and its covariance growing on
/// process noise alone. That is #394's failure mode -- six states a filter carried, seeded and
/// never estimated -- reached by a different route.
///
/// # Panics
///
/// Never: `bias_index` is validated against the state width by
/// [`RelativeAltitudeMeasurement::require_bias_state`](crate::measurements::RelativeAltitudeMeasurement)
/// before this is called, and `state_dim` comes from the state itself.
#[must_use]
pub fn relative_altitude_bias_jacobian(state_dim: usize, bias_index: usize) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(1, state_dim.max(9));
    h[(0, 2)] = 1.0; // d(z_alt)/d(alt)
    if bias_index < h.ncols() {
        h[(0, bias_index)] = 1.0; // d(z_alt)/d(baro bias)
    }
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
/// [`MagnetometerYawMeasurement`](crate::measurements::MagnetometerYawMeasurement)'s expected
/// measurement is the state's yaw and nothing else, so
///
/// $$
/// h(\mathbf{x}) = \psi = x_8, \qquad H = \frac{\partial h}{\partial \mathbf{x}}
///                                     = \begin{bmatrix} 0 & \cdots & 0 & 1 \end{bmatrix},
/// $$
///
/// a single 1 in the yaw column.
///
/// # Why the tilt columns are zero
///
/// This is worth stating explicitly, because it is not obvious and this function used to do
/// the other thing. The magnetometer model is unusual among this crate's measurements: its
/// `z` is not a raw observation but a *pseudo*-measurement, derived by levelling the raw
/// field with roll and pitch taken from the current estimate. So `z` genuinely does vary with
/// roll and pitch, and this function used to return that variation --
/// $\partial z/\partial\phi$ and $\partial z/\partial\theta$ -- in columns 6 and 7.
///
/// That is the wrong quantity for the contract every filter here relies on: the update forms
/// the residual as $z - h(\mathbf{x})$ while taking $H = \partial h/\partial\mathbf{x}$, so a
/// column holding $\partial z/\partial x_i$ enters the gain with the opposite sign to the one
/// it describes. The [`ErrorStateKalmanFilter`](crate::kalman::ErrorStateKalmanFilter) already
/// refused these columns for exactly that reason, overwriting the attitude block with finite
/// differences of the expected measurement (#286); the EKF, which uses the analytic form
/// as-is, did not, and the two filters disagreed about what $H$ meant for this one model.
/// They now agree.
///
/// The neglected term is real but small and is deliberately left to the measurement noise:
/// the tilt sensitivity of a levelled heading is bounded well below the 0.2 rad
/// [`MAG_YAW_NOISE`](crate::measurements::MAG_YAW_NOISE) the sim path assigns, which is itself
/// tighter than the 0.293 rad scatter the sensor actually shows on
/// `core/tests/test_data.csv`. Folding $-\partial z/\partial\mathbf{x}$ in properly would make
/// the innovation's linearization exact, but it is a change to the
/// [`MeasurementModel`](crate::measurements::MeasurementModel) contract -- what `get_jacobian`
/// promises, for every model -- rather than to this function, so it is not made here.
///
/// The result is frame-independent, unlike the measurement itself: the NED and ENU heading
/// formulas differ, but both are functions of the levelled field alone, and $h$ is the state's
/// yaw in whichever frame that state is expressed in. There is no `is_enu` to thread here.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused; retained for signature parity with the other
///   Jacobians in this module, and because a future exact form would need it)
/// * `_mag_x` - Body-frame magnetic field x-component (µT), unused for the same reason
/// * `_mag_y` - Body-frame magnetic field y-component (µT), unused for the same reason
/// * `_mag_z` - Body-frame magnetic field z-component (µT), unused for the same reason
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
/// // The expected measurement is the state yaw, so the yaw column is exactly 1 ...
/// assert!((h[(0, 8)] - 1.0).abs() < 1e-12);
/// // ... and every other column, the tilt columns included, is exactly 0.
/// assert!((0..8).all(|i| h[(0, i)] == 0.0));
/// ```
///
/// # Compatibility
///
/// All four parameters are now ignored; the result is a constant. The signature is kept so
/// existing callers still compile, but a caller that was relying on the roll and pitch columns
/// gets zeros with no signal. Measured cost of the change on `core/tests/test_data.csv`: EKF
/// yaw RMSE 22.647 deg with the old analytic tilt columns against 22.671 deg with them zeroed
/// -- 0.1%. They were never carrying much; they were carrying it with the wrong sign (#305).
pub fn magnetometer_yaw_jacobian(
    _state: &StrapdownState,
    _mag_x: f64,
    _mag_y: f64,
    _mag_z: f64,
) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(1, 9);

    // The expected measurement is the state yaw, so ∂h/∂ψ = 1 and every other partial is
    // identically zero. See the "Why the tilt columns are zero" section above before
    // reinstating a tilt term here: the levelling's dependence on roll and pitch belongs to
    // `z`, not to `h`, and putting it in this matrix feeds it back with the wrong sign.
    h[(0, 8)] = 1.0;

    h
}

/// Compute measurement Jacobian (H) for a zero-velocity update (ZUPT)
///
/// A ZUPT asserts that the local-level-frame velocity is zero while the platform is
/// stationary, so the pseudo-measurement is $h(x) = [v_n, v_e, v_d]^\top$ and the
/// Jacobian is identical to [`gps_velocity_jacobian`]: the identity on the velocity
/// block. It is given its own name because the two differ in everything except this
/// matrix -- source, noise model, and the conditions under which they may be applied
/// -- and a reader following the ZUPT path should not have to discover that it
/// borrows the GNSS velocity Jacobian.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused; the model is linear in velocity)
///
/// # Returns
///
/// 3×9 measurement Jacobian matrix H for the zero-velocity pseudo-measurement
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::zupt_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = zupt_jacobian(&state);
/// assert_eq!((h.nrows(), h.ncols()), (3, 9));
/// assert_eq!(h[(0, 3)], 1.0);
/// assert_eq!(h[(2, 5)], 1.0);
/// ```
///
/// # References
///
/// - Groves 2nd ed., Section 15.2.1 (zero-velocity updates)
pub fn zupt_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(3, 9);
    // The pseudo-measurement observes the velocity states directly: [v_n, v_e, v_d]
    h[(0, 3)] = 1.0; // ∂(z_vn)/∂(v_n)
    h[(1, 4)] = 1.0; // ∂(z_ve)/∂(v_e)
    h[(2, 5)] = 1.0; // ∂(z_vd)/∂(v_d)
    h
}

/// Compute measurement Jacobian (H) for a zero-angular-rate update (ZARU)
///
/// # Why this one is 15 columns
///
/// A stationary platform's gyroscopes read the Earth rate plus their own bias and
/// nothing else. Subtracting the Earth rate leaves a direct observation of the gyro
/// bias, so ZARU is a measurement on states 12..15 of the 15-state vector:
///
/// ```text
/// z = omega_measured - C_n^b * omega_ie^n,    h(x) = [b_gx, b_gy, b_gz]
/// ```
///
/// Angular rate is not a navigation state, so there is no 9-state form of this
/// Jacobian -- a 9-state filter has nothing for ZARU to correct, and asking for one
/// is a configuration error rather than something to paper over with a zero block.
/// The other Jacobians in this module are 9 columns and are zero-padded by the
/// filter; this one arrives at full width and is used as-is.
///
/// # Arguments
///
/// * `_state` - Current navigation state (unused; the model is linear in the bias)
///
/// # Returns
///
/// 3×15 measurement Jacobian matrix H for the zero-angular-rate pseudo-measurement
///
/// # Example
///
/// ```rust
/// use strapdown::linearize::zaru_jacobian;
/// use strapdown::StrapdownState;
///
/// let state = StrapdownState::default();
/// let h = zaru_jacobian(&state);
/// assert_eq!((h.nrows(), h.ncols()), (3, 15));
/// assert_eq!(h[(0, 12)], 1.0);
/// assert_eq!(h[(2, 14)], 1.0);
/// // Nothing outside the gyro-bias block is observed.
/// assert_eq!(h.view((0, 0), (3, 12)).iter().copied().sum::<f64>(), 0.0);
/// ```
///
/// # References
///
/// - Groves 2nd ed., Section 15.2.2 (zero-angular-rate updates)
pub fn zaru_jacobian(_state: &StrapdownState) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(3, 15);
    // The pseudo-measurement observes the gyro bias states directly.
    h[(0, 12)] = 1.0; // ∂(z_gx)/∂(b_gx)
    h[(1, 13)] = 1.0; // ∂(z_gy)/∂(b_gy)
    h[(2, 14)] = 1.0; // ∂(z_gz)/∂(b_gz)
    h
}

/// Apply an error-state correction to a `StrapdownState`
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
/// where `[δθ]×` is the skew-symmetric matrix of the attitude error angles.
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
/// ).unwrap();
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
/// # Errors
/// [`StrapdownError::DimensionMismatch`] if `delta_x` has fewer than 9 elements. This is a
/// public function taking a caller-supplied vector, so the precondition is reported rather
/// than asserted (#254).
pub fn apply_eskf_correction(
    state: &mut StrapdownState,
    delta_x: &DVector<f64>,
) -> Result<(), StrapdownError> {
    if delta_x.len() < 9 {
        return Err(StrapdownError::DimensionMismatch {
            what: "ESKF error state",
            expected: 9,
            got: delta_x.len(),
        });
    }

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
    Ok(())
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
/// Optional tuple of (`accel_bias_correction`, `gyro_bias_correction`) if `delta_x` has 15 elements.
/// Returns `None` if `delta_x` has only 9 elements.
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
/// let biases = apply_eskf_correction_with_biases(&mut state, &delta_x).unwrap();
/// if let Some((accel_bias, gyro_bias)) = biases {
///     // Apply bias corrections to IMU preprocessing
///     println!("Accel bias correction: {:?}", accel_bias);
///     println!("Gyro bias correction: {:?}", gyro_bias);
/// }
/// ```
/// # Errors
/// Propagated from [`apply_eskf_correction`]: the error state must have at least 9 elements.
pub fn apply_eskf_correction_with_biases(
    state: &mut StrapdownState,
    delta_x: &DVector<f64>,
) -> Result<Option<ImuBiasCorrection>, StrapdownError> {
    // Apply the navigation state correction
    apply_eskf_correction(state, delta_x)?;

    // Extract bias corrections if present
    if delta_x.len() >= 15 {
        let accel_bias = Vector3::new(delta_x[9], delta_x[10], delta_x[11]);
        let gyro_bias = Vector3::new(delta_x[12], delta_x[13], delta_x[14]);
        Ok(Some((accel_bias, gyro_bias)))
    } else {
        Ok(None)
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
/// * `conditional_mean` - 12-element conditional EKF mean [δv, δθ, `δb_g`, `δb_a`]
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
/// let delta_x = assemble_error_state(&dr, &mu).unwrap();
/// assert_eq!(delta_x.len(), 15);
/// ```
///
/// # Errors
/// [`StrapdownError::DimensionMismatch`] if `conditional_mean` is not 12 elements.
pub fn assemble_error_state(
    position_error: &Vector3<f64>,
    conditional_mean: &DVector<f64>,
) -> Result<DVector<f64>, StrapdownError> {
    if conditional_mean.len() != 12 {
        return Err(StrapdownError::DimensionMismatch {
            what: "RBPF conditional mean",
            expected: 12,
            got: conditional_mean.len(),
        });
    }

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
    Ok(DVector::from_vec(vec![
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
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::{Rotation3, UnitQuaternion};

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
        )
        .unwrap();
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
        // O(dt^2) terms the first-order Jacobian omits reach ~7e-4 at this dt; anything
        // above 5e-3 is a first-order structural error.
        const TOLERANCE: f64 = 5e-3;

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
        )
        .unwrap();

        let mut base_out = nominal;
        crate::mechanize(
            &mut base_out,
            &crate::ImuSample::from_rates(&crate::IMUData { accel, gyro }, DT),
        )
        .unwrap();

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
            crate::mechanize(&mut s, &crate::ImuSample::from_rates(&imu, DT)).unwrap();
            // Via the quaternion, not `Rotation3::scaled_axis`. That method is unusable as a
            // finite-difference oracle at this scale: it collapses rotations below its
            // internal epsilon to exactly zero, and on a matrix whose trace rounds a whisker
            // above 3 -- which a product of two independently orthonormalised attitudes
            // routinely does -- it returns NaN from `acos` of a value just over 1. The
            // quaternion path takes the angle from `atan2` and has neither failure mode. It
            // resolves a genuine 1e-11 rotation that `scaled_axis` reports as 0.
            let dtheta =
                UnitQuaternion::from_rotation_matrix(&(base_out.attitude.transpose() * s.attitude))
                    .scaled_axis();
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
    ///     `C_b^n(true)` = `C_b^n(nominal)` · (I + [δθ]_×)
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
        )
        .unwrap();
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
            crate::mechanize(&mut plus, &crate::ImuSample::from_rates(&imu, dt)).unwrap();
            crate::mechanize(&mut minus, &crate::ImuSample::from_rates(&imu, dt)).unwrap();

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
            "F's velocity/attitude block matches the global convention (gap {gap:e}); \
             the local (body-frame) form -C_b^n [f^b]_x dt is required -- see #266"
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
        )
        .unwrap();
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

    /// A non-degenerate state for the frame checks below: both hemispheres' worth of
    /// latitude structure, all three velocity channels turning, and a tilted attitude.
    fn frame_check_state() -> StrapdownState {
        StrapdownState {
            latitude: 51.5_f64.to_radians(),
            longitude: (-0.12_f64).to_radians(),
            altitude: 2400.0,
            velocity_north: 120.0,
            velocity_east: -45.0,
            velocity_vertical: 6.0, // descending, NED
            attitude: Rotation3::from_euler_angles(0.18, -0.27, 2.4),
            is_enu: false,
        }
    }

    /// The reflection carrying a NED 9-state error vector to the ENU convention.
    ///
    /// Position is untouched, the vertical velocity flips, and the attitude error is a
    /// nav-frame rotation *vector* -- a pseudovector, so it reflects as `-F`, not `F`.
    fn error_state_reflection() -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            1.0, 1.0, 1.0, // latitude, longitude, altitude
            1.0, 1.0, -1.0, // north, east, vertical velocity
            -1.0, -1.0, 1.0, // nav-frame rotation vector
        ]))
    }

    /// The transition Jacobian has to describe the same dynamics in either convention.
    ///
    /// `mechanize` canonicalises to NED (#321), so the Jacobian that linearises it must too.
    /// Before this, `transition_jacobian` built `omega_ie` and `omega_en` from NED vectors and
    /// applied them to ENU velocity and attitude errors, leaving the RBPF's and EKF's
    /// covariance propagating dynamics the mechanization no longer had.
    ///
    /// The check is exact rather than approximate: the two conventions are related by a
    /// signature matrix `T`, so `F_enu = T F_ned T` entry for entry.
    #[test]
    fn transition_jacobian_agrees_across_vertical_conventions() {
        let ned = frame_check_state();
        let enu = ned.to_enu();
        // Body-frame IMU: specific force is an ordinary vector, angular rate is not.
        let accel_ned = Vector3::new(0.42, -0.31, -9.72);
        let gyro_ned = Vector3::new(0.011, -0.007, 0.023);
        let accel_enu = Vector3::new(0.42, -0.31, 9.72);
        let gyro_enu = Vector3::new(-0.011, 0.007, 0.023);
        let dt = 0.02;

        let f_ned = state_transition_jacobian(&ned, &accel_ned, &gyro_ned, dt);
        let f_enu = state_transition_jacobian(&enu, &accel_enu, &gyro_enu, dt);
        let t = error_state_reflection();
        let expected = &t * &f_ned * &t;

        let max_error = (&f_enu - &expected).abs().max();
        assert!(
            max_error < 1e-18,
            "ENU and NED Jacobians disagree by {max_error:e}; they describe the same dynamics \
             and are related by a signature matrix"
        );
        // Non-degenerate: the entries the reflection actually acts on have to be present, or
        // the agreement above is an agreement about zeros.
        assert!(f_ned[(3, 4)].abs() > 1e-7, "Coriolis block is empty");
        assert!(f_ned[(6, 4)].abs() > 1e-11, "transport coupling is empty");
        assert!(f_ned[(5, 0)].abs() > 1e-5, "gravity latitude term is empty");
    }

    /// Finite-difference the velocity columns in both conventions, at a `dt` where the
    /// Coriolis and transport terms are actually resolvable.
    ///
    /// The existing `test_state_transition_jacobian_*` checks run at `dt = 1e-4`, where every
    /// term touched here is ~1e-9 -- three orders below their 1e-6 tolerance. They would pass
    /// with the Coriolis block deleted outright, in either frame.
    ///
    /// Only the velocity columns are differenced, so no attitude parametrisation is involved
    /// on the input side; the attitude rows come back out as a nav-frame rotation vector,
    /// which is what `AttitudeParametrization::RotationVector` means.
    ///
    /// # Why every entry is now checked, and why not all of them at 1e-14
    ///
    /// Five of the nine velocity-row entries used to be excluded outright. 5.54's Coriolis
    /// term is *quadratic* in velocity -- `ω_en(v) × v` -- and the Jacobian differentiated
    /// only the second factor, omitting `(∂ω_en/∂v_j) × v`. That omission was first-order,
    /// not O(dt²): it scaled with `dt` exactly as the terms that were kept, so no choice of
    /// `dt` separated them. Measured at `dt = 0.01` on this state before #325:
    ///
    /// ```text
    ///     entry     analytic        numeric         shortfall
    ///     (3,3)      1.000000e0      1.000000e0      9.3e-9
    ///     (5,3)     -1.881761e-7    -3.763275e-7     1.9e-7   (a factor of two)
    ///     (3,4)     -1.052891e-6    -9.644178e-7     8.8e-8
    ///     (4,4)      1.000000e0      1.000000e0      2.5e-7
    ///     (5,4)     -8.375075e-7    -7.671326e-7     7.0e-8
    /// ```
    ///
    /// The exclusion list is gone: all nine entries are checked. #325 asked for a flat 1e-14
    /// as well, and that is not attainable -- not because anything is still missing from the
    /// first-order Jacobian, but because the mechanization has a second-order term these five
    /// entries alone are exposed to. 5.47 rotates the sensed increment with the *averaged*
    /// attitude `0.5 (C0 + C1) Δv`, and `C1` carries velocity through 5.46's transport rate,
    /// so the true derivative contains
    ///
    /// ```text
    ///     0.5 (∂C1/∂v_j) Δv = -0.5 dt^2 (∂ω_en/∂v_j) × f^n
    /// ```
    ///
    /// which a first-order `I + A dt` Jacobian does not carry and should not. Hence the
    /// per-column bound below, `0.5 dt² |∂ω_en/∂v_j| |f^n|`, floored at the 1e-14 rounding
    /// limit. It reproduces the old split exactly and explains it: the four entries that met
    /// 1e-14 are precisely the ones where that cross product vanishes -- all of column 2,
    /// where `∂ω_en/∂v_D` is identically zero, plus `(4,3)`, where `∂ω_en/∂v_N` is parallel
    /// to the axis being read. The five that did not are the five the term reaches, and they
    /// now agree to 7.6e-11 where they used to be out by up to 2.5e-7: a factor of 3000, at a
    /// bound that shrinks as `dt²` rather than staying put.
    ///
    /// **This bound is structurally near-saturated, by design, and that is worth knowing
    /// before diagnosing a failure.** `|a × b| <= |a| |b|` is tight exactly when the two are
    /// perpendicular, and `∂ω_en/∂v_j` is close to perpendicular to `f^n` for any near-level
    /// attitude -- so `(3,3)` consumes 94% of its bound and `(4,4)` 84%. The slack is the
    /// cosine of that angle and nothing else. A failure here is therefore as likely to mean
    /// "the sample attitude or IMU moved" as "the Jacobian is wrong"; check
    /// `frame_check_state` and `frame_check_imu` first. Using `|(∂ω_en/∂v_j) × f^n|` itself
    /// rather than the norm product would remove the saturation, at the cost of duplicating
    /// the very expression under test inside its own oracle.
    #[test]
    fn transition_jacobian_velocity_columns_match_finite_differences_in_both_frames() {
        // Floating-point floor on the numeric side: the difference of two propagated
        // velocities of ~120 m/s over a 1 m/s step, so ~eps * |v| / step ~ 1e-14.
        const ROUNDING_FLOOR: f64 = 1e-14;
        let dt = 0.01;
        let step = 1.0; // m/s
        for base in [frame_check_state(), frame_check_state().to_enu()] {
            let (accel, gyro) = frame_check_imu(base.is_enu);
            let analytic = state_transition_jacobian(&base, &accel, &gyro, dt);

            let propagate = |offset: Vector3<f64>| {
                let mut s = base;
                s.velocity_north += offset[0];
                s.velocity_east += offset[1];
                s.velocity_vertical += offset[2];
                crate::mechanize(
                    &mut s,
                    &crate::ImuSample::from_rates(&crate::IMUData { accel, gyro }, dt),
                )
                .unwrap();
                s
            };
            let nominal = propagate(Vector3::zeros());

            // The second-order term derived above, per column: `0.5 dt² |∂ω_en/∂v_j| |f^n|`.
            // `|a × b| <= |a||b|`, so this is an upper bound rather than a fit, and the
            // reflection an ENU state applies to the gradient preserves its norm.
            let gradients = transport_rate_velocity_gradients(base.latitude, base.altitude);
            let specific_force_norm = (base.attitude.matrix() * accel).norm();

            for column in 0..3 {
                let mut offset = Vector3::zeros();
                offset[column] = step;
                let plus = propagate(offset);
                let minus = propagate(-offset);
                let averaging_bound =
                    0.5 * dt * dt * gradients[column].norm() * specific_force_norm;
                let tolerance = averaging_bound.max(ROUNDING_FLOOR);

                // Velocity rows.
                let numeric = [
                    (plus.velocity_north - minus.velocity_north) / (2.0 * step),
                    (plus.velocity_east - minus.velocity_east) / (2.0 * step),
                    (plus.velocity_vertical - minus.velocity_vertical) / (2.0 * step),
                ];
                for (row, value) in numeric.iter().enumerate() {
                    assert_approx_eq!(analytic[(3 + row, 3 + column)], *value, tolerance);
                }

                // Attitude rows, as a nav-frame rotation vector: the left perturbation
                // `C_pert = (I + [dtheta x]) C_nom`. Via the quaternion, for the reasons
                // given in `test_error_state_jacobian_matches_nonlinear_propagation`.
                let delta = |propagated: &StrapdownState| {
                    UnitQuaternion::from_rotation_matrix(
                        &(propagated.attitude * nominal.attitude.transpose()),
                    )
                    .scaled_axis()
                };
                let numeric_attitude = (delta(&plus) - delta(&minus)) / (2.0 * step);
                for row in 0..3 {
                    assert_approx_eq!(
                        analytic[(6 + row, 3 + column)],
                        numeric_attitude[row],
                        1e-12
                    );
                }
            }

            // Non-degenerate: the entries actually checked are orders of magnitude above the
            // tolerances they are checked against.
            assert!(
                analytic[(4, 3)].abs() > 1e-7,
                "Coriolis off-diagonal is empty"
            );
            assert!(
                analytic[(6, 4)].abs() > 1e-10,
                "transport coupling is empty"
            );
        }
    }

    /// [`bias_coupling_blocks`] agrees with finite differences of the mechanization it
    /// linearises, at three attitudes including one well away from level.
    ///
    /// The oracle is `mechanize` itself, perturbed in the bias and differenced -- not a second
    /// analytic expression, which would risk putting the thing under test inside its own
    /// oracle.
    ///
    /// # Reading the tolerance
    ///
    /// The residual is **flat across four decades of step size** -- 6.011e-6 at `h = 1e-4` and
    /// the same at `h = 1e-7` -- which is what says it is not truncation. It is this Jacobian's
    /// own first-order-in-`dt` approximation, and it comes to about `3e-4` relative, of the
    /// order of `|omega| dt`. Every other block in this matrix makes the same approximation.
    ///
    /// The bound is therefore relative to the block's own scale rather than absolute, and the
    /// thing it would catch is a wrong *frame* or a wrong sign -- either of which lands at
    /// order 1, not 3e-4. That is the failure mode worth guarding: the attitude block here is
    /// `-E(Phi+)^-1 C_b^n` where the ESKF's is a bare `-I`, and transcribing one into the
    /// other is a rotation, not a rounding error.
    #[test]
    fn bias_coupling_blocks_match_finite_differences_of_the_mechanization() {
        /// Step size for the central difference. Any value in `[1e-7, 1e-4]` gives the same
        /// answer; see the doc comment.
        const STEP: f64 = 1e-6;
        /// The first-order-in-`dt` residual measured at 3e-4 relative; this leaves an order of
        /// headroom without admitting a sign or frame error, which would be order 1.
        const MAX_RELATIVE_ERROR: f64 = 3e-3;

        /// Mechanize one step against a given pair of biases, and return the 9-state result.
        fn step(
            base: &StrapdownState,
            accel: Vector3<f64>,
            gyro: Vector3<f64>,
            accel_bias: Vector3<f64>,
            gyro_bias: Vector3<f64>,
            dt: f64,
        ) -> [f64; 9] {
            let mut state = *base;
            let sample = crate::ImuSample {
                delta_v: (accel - accel_bias) * dt,
                delta_theta: (gyro - gyro_bias) * dt,
                dt,
            };
            crate::mechanize(&mut state, &sample).expect("mechanization must succeed");
            let (roll, pitch, yaw) = state.attitude.euler_angles();
            [
                state.latitude,
                state.longitude,
                state.altitude,
                state.velocity_north,
                state.velocity_east,
                state.velocity_vertical,
                roll,
                pitch,
                yaw,
            ]
        }

        let dt = 0.02;
        let accel = Vector3::new(0.3, -0.2, -9.7);
        let gyro = Vector3::new(0.01, -0.02, 0.03);

        for (label, roll, pitch, yaw, is_enu) in [
            ("level north NED", 0.0, 0.0, 0.0, false),
            ("banked NED", 0.3, -0.2, 1.1, false),
            ("steep NED", -0.5, 0.6, -2.0, false),
            // ENU is not a relabelling here: `attitude_update` takes a separate reflected
            // path, so a sign or conjugation error in the gyro-bias block can pass every NED
            // case and still break ENU bias observability. The filter-level regression is NED
            // only, so this is the one place that difference is exercised.
            ("level north ENU", 0.0, 0.0, 0.0, true),
            ("banked ENU", 0.3, -0.2, 1.1, true),
            ("steep ENU", -0.5, 0.6, -2.0, true),
        ] {
            let base = StrapdownState {
                latitude: 40.0_f64.to_radians(),
                longitude: (-75.0_f64).to_radians(),
                altitude: 120.0,
                velocity_north: 12.0,
                velocity_east: -4.0,
                velocity_vertical: 0.5,
                attitude: Rotation3::from_euler_angles(roll, pitch, yaw),
                is_enu,
            };
            let (velocity_block, attitude_block) =
                bias_coupling_blocks(&base, &gyro, dt, AttitudeParametrization::Euler);
            let zero = Vector3::zeros();

            for (block_name, block, first_row, perturb_gyro) in [
                ("velocity/accel-bias", velocity_block, 3, false),
                ("attitude/gyro-bias", attitude_block, 6, true),
            ] {
                let scale = block.abs().max();
                for column in 0..3 {
                    let mut up_bias = Vector3::zeros();
                    up_bias[column] = STEP;
                    let mut down_bias = Vector3::zeros();
                    down_bias[column] = -STEP;
                    let (up, down) = if perturb_gyro {
                        (
                            step(&base, accel, gyro, zero, up_bias, dt),
                            step(&base, accel, gyro, zero, down_bias, dt),
                        )
                    } else {
                        (
                            step(&base, accel, gyro, up_bias, zero, dt),
                            step(&base, accel, gyro, down_bias, zero, dt),
                        )
                    };
                    for row in 0..3 {
                        let numeric = (up[first_row + row] - down[first_row + row]) / (2.0 * STEP);
                        let analytic = block[(row, column)];
                        assert!(
                            (numeric - analytic).abs() <= MAX_RELATIVE_ERROR * scale,
                            "{label}, {block_name} block ({row}, {column}): analytic \
                             {analytic}, finite difference {numeric}. A residual this size is \
                             a wrong frame or a wrong sign, not the first-order-in-dt \
                             approximation every block here makes."
                        );
                    }
                }
            }
        }
    }

    /// The IMU for [`frame_check_state`], reflected into whichever convention it carries.
    ///
    /// Specific force is an ordinary vector and angular rate is a pseudovector, so the two
    /// reflect differently -- the same distinction `flip_vertical` and `flip_vertical_rate`
    /// draw. Handing both frames the *same* body triple would compare two different physical
    /// motions and make any frame check meaningless.
    fn frame_check_imu(is_enu: bool) -> (Vector3<f64>, Vector3<f64>) {
        if is_enu {
            (
                Vector3::new(0.42, -0.31, 9.72),
                Vector3::new(-0.011, 0.007, 0.023),
            )
        } else {
            (
                Vector3::new(0.42, -0.31, -9.72),
                Vector3::new(0.011, -0.007, 0.023),
            )
        }
    }

    /// Central-difference one column of the Jacobian in the *increment* domain.
    ///
    /// Differences `mechanize(x) - x` rather than `mechanize(x)`, which is what makes a
    /// latitude column measurable at all. Latitude here is ~0.9 rad and the propagated
    /// velocity ~120 m/s; differencing the absolute velocity across a 1e-6 rad step leaves
    /// the answer sitting on `ulp(120) / 2e-6` ~ 7e-9 of quantisation noise, against terms of
    /// 3e-5. Subtracting the unperturbed value first removes the large constant before the
    /// cancellation happens, and since the perturbation touches only one element the two
    /// forms are identical in exact arithmetic.
    ///
    /// Returns `∂(mechanize(x) - x)/∂x_column`, i.e. the analytic Jacobian *minus the
    /// identity*, as a nine-element state vector.
    fn increment_derivative(base: &StrapdownState, column: usize, step: f64, dt: f64) -> Vec<f64> {
        let (accel, gyro) = frame_check_imu(base.is_enu);
        let increment = |delta: f64| -> Vec<f64> {
            let mut vector = Vec::<f64>::from(base);
            vector[column] += delta;
            let mut state = StrapdownState::try_from(vector.as_slice()).unwrap();
            state.is_enu = base.is_enu;
            let before = Vec::<f64>::from(&state);
            crate::mechanize(
                &mut state,
                &crate::ImuSample::from_rates(&crate::IMUData { accel, gyro }, dt),
            )
            .unwrap();
            let after = Vec::<f64>::from(&state);
            after.iter().zip(&before).map(|(a, b)| a - b).collect()
        };
        let plus = increment(step);
        let minus = increment(-step);
        plus.iter()
            .zip(&minus)
            .map(|(a, b)| (a - b) / (2.0 * step))
            .collect()
    }

    /// The velocity rows' *position* columns, finite-differenced in both conventions.
    ///
    /// Companion to the velocity-column test above: 5.54 applies `-(2ω_ie + ω_en) × v` and
    /// both rates are functions of latitude, so these entries are not zero. Before #317 and
    /// #325 they were: `f[(3,0)]` and `f[(4,0)]` were never assigned, and `f[(5,0)]` carried
    /// only the gravity gradient, overshooting the true value by the missing Coriolis share.
    /// On the state below, analytic against finite difference, before:
    ///
    /// ```text
    ///     entry     analytic        numeric         shortfall
    ///     (3,0)      0.0             3.268497e-5     3.3e-5
    ///     (4,0)      0.0             8.035883e-5     8.0e-5
    ///     (5,0)      5.058048e-4     4.546776e-4     5.1e-5
    /// ```
    ///
    /// The altitude column is checked too. It is analytically non-zero and numerically inert
    /// -- ~1e-12 against velocities of 120 m/s, because the principal radii change by a part
    /// in 1e6 per kilometre -- so it is here to pin the sign of a term that falls out of the
    /// same cross product for free, not because it does anything.
    #[test]
    fn transition_jacobian_position_columns_match_finite_differences_in_both_frames() {
        // Derived, not fitted. The only thing the analytic form omits here is the
        // mechanization's interval-averaged attitude: 5.47 rotates specific force with
        // 0.5*(C0 + C1) where the Jacobian uses C0, and C1 carries latitude through 5.46's
        // -[ω_in×] C dt. The shortfall is therefore bounded by
        //
        //     0.5 * dt^2 * |∂ω_in/∂φ| * |f^n|
        //
        // and with |∂ω_in/∂φ| <= Ω + |v_E| sec^2(φ) / R_E ~ 9e-5 rad/s per radian here and
        // |f^n| <= 10 m/s^2, that is 0.5 * 1e-4 * 9e-5 * 10 = 4.5e-8 at dt = 0.01. The
        // tolerance clears it by a factor of two and still sits ~300x under the smallest
        // entry being checked. Measured worst is 2.3e-8.
        const TOLERANCE: f64 = 1e-7;
        // The altitude column is three orders below the latitude column, so it needs its own
        // tolerance -- but it has to stay *below* the entries it checks or it pins nothing.
        // Measured here: f[(3,2)] = 4.46e-13, f[(4,2)] = 1.73e-12, f[(5,2)] = -3.08e-8,
        // against residuals of 1.9e-14, 1.4e-14 and 2.2e-15. 1e-13 is five times the worst
        // residual and below the smallest entry, so an analytic zero or a sign flip in any of
        // the three fails. At the 1e-12 this first carried, f[(3,2)] was less than half the
        // tolerance and the assertion was vacuous for that entry.
        const ALTITUDE_TOLERANCE: f64 = 1e-13;
        let dt = 0.01;

        for base in [frame_check_state(), frame_check_state().to_enu()] {
            let frame = if base.is_enu { "ENU" } else { "NED" };
            let (accel, gyro) = frame_check_imu(base.is_enu);
            let analytic = state_transition_jacobian(&base, &accel, &gyro, dt);

            // Latitude, in radians: 1e-6 rad is ~6 m, small enough that the second
            // derivative of gravity is invisible and large enough to clear the floor.
            let latitude = increment_derivative(&base, 0, 1e-6, dt);
            for row in 3..6 {
                assert_approx_eq!(analytic[(row, 0)], latitude[row], TOLERANCE);
            }
            // Longitude is not merely omitted, it is genuinely absent: WGS84 gravity and the
            // principal radii are axisymmetric, so perturbing longitude moves nothing but
            // longitude, bit for bit.
            let longitude = increment_derivative(&base, 1, 1e-6, dt);
            for row in 3..6 {
                assert_approx_eq!(analytic[(row, 1)], longitude[row], 1e-18);
            }
            // Altitude, in metres.
            let altitude = increment_derivative(&base, 2, 0.1, dt);
            for row in 3..6 {
                assert_approx_eq!(analytic[(row, 2)], altitude[row], ALTITUDE_TOLERANCE);
            }

            // Non-degenerate: the two entries that were exactly zero before #317 must now
            // carry something far above the tolerance they are checked against, or the
            // agreement above is an agreement about nothing.
            assert!(
                analytic[(3, 0)].abs() > 1e-5 && analytic[(4, 0)].abs() > 1e-5,
                "in {frame} the Coriolis latitude terms are empty: f[(3,0)] = {:e}, \
                 f[(4,0)] = {:e}",
                analytic[(3, 0)],
                analytic[(4, 0)]
            );
        }
    }

    /// One step of the mechanization from a singly-perturbed copy of `base`.
    ///
    /// Shared by the two attitude-row checks below, which both need the propagated *state*
    /// rather than the Euler increment `increment_derivative` returns: the attitude rows of
    /// the rotation-vector Jacobian are a nav-frame rotation vector, and reading one off a
    /// pair of Euler triples is exactly the confusion #307 was.
    fn propagate_perturbed(
        base: &StrapdownState,
        element: usize,
        delta: f64,
        dt: f64,
    ) -> StrapdownState {
        let (accel, gyro) = frame_check_imu(base.is_enu);
        let mut vector = Vec::<f64>::from(base);
        vector[element] += delta;
        let mut state = StrapdownState::try_from(vector.as_slice()).unwrap();
        state.is_enu = base.is_enu;
        crate::mechanize(
            &mut state,
            &crate::ImuSample::from_rates(&crate::IMUData { accel, gyro }, dt),
        )
        .unwrap();
        state
    }

    /// How far one step's nav-frame rotation vector moves per unit of `element`.
    ///
    /// The left perturbation `C_pert = (I + [dtheta x]) C_nom`, via the quaternion, for the
    /// reasons given in `test_error_state_jacobian_matches_nonlinear_propagation`.
    fn rotation_vector_derivative(
        base: &StrapdownState,
        element: usize,
        step: f64,
        dt: f64,
    ) -> Vector3<f64> {
        let nominal = propagate_perturbed(base, element, 0.0, dt);
        let axis = |state: &StrapdownState| {
            UnitQuaternion::from_rotation_matrix(&(state.attitude * nominal.attitude.transpose()))
                .scaled_axis()
        };
        (axis(&propagate_perturbed(base, element, step, dt))
            - axis(&propagate_perturbed(base, element, -step, dt)))
            / (2.0 * step)
    }

    /// The second-order term the attitude rows' non-attitude columns are exposed to, per
    /// column, and nothing else.
    ///
    /// `mechanize` applies the one-step rotation `delta = C0 dtheta_b - omega_in dt` and the
    /// Jacobian is its derivative, which for these columns is exact -- 5.46 is linear in
    /// `omega_in`. What is *not* exact is the comparison: the numeric side reads the rotation
    /// vector of `exp([delta_+ x]) exp(-[delta_- x])`, and Baker-Campbell-Hausdorff leaves a
    /// commutator behind, so a central difference over step `h` returns
    ///
    /// ```text
    ///     g - 0.5 g x delta_0,    g = d(delta)/d(element)
    /// ```
    ///
    /// against the analytic `g`. Hence `0.5 |g| |delta_0|`, with `|a x b| <= |a| |b|` and
    /// `|delta_0| <= (|omega_ib| + |omega_in|) dt`. Every column's `g` is `dt` times that
    /// column's gradient of `omega_in`, so the whole bound is
    ///
    /// ```text
    ///     0.5 dt^2 |d(omega_in)/d(element)| (|omega_ib| + |omega_in|)
    /// ```
    ///
    /// Derived rather than fitted (#288), and near-saturated: against the rotation-vector
    /// form the latitude column consumes 65% of it and `v_N` 89%, because `|a x b| <= |a| |b|`
    /// is tight exactly when the two are perpendicular and a near-level attitude puts them
    /// close to it. A failure here is therefore as likely to mean "`frame_check_state` or
    /// `frame_check_imu` moved" as "the Jacobian is wrong"; check those first.
    ///
    /// Returns one bound per state element, zero for the elements `omega_in` does not depend
    /// on -- longitude and the vertical velocity -- where the caller falls back to its
    /// rounding floor.
    fn attitude_row_second_order_bounds(base: &StrapdownState, dt: f64) -> [f64; 6] {
        let (_, gyro) = frame_check_imu(base.is_enu);
        let velocity = Vector3::new(
            base.velocity_north,
            base.velocity_east,
            base.velocity_vertical,
        );
        let latitude_degrees = base.latitude.to_degrees();
        let rate_magnitude = gyro.norm()
            + earth::earth_rate_lla(&latitude_degrees).norm()
            + earth::transport_rate(&latitude_degrees, &base.altitude, &velocity).norm();

        let (earth_latitude, transport_latitude, transport_altitude) = rate_position_gradients(
            base.latitude,
            base.altitude,
            base.velocity_north,
            base.velocity_east,
        );
        let velocity_gradients = transport_rate_velocity_gradients(base.latitude, base.altitude);
        // Norms, so the pseudovector reflection an ENU state applies is irrelevant here.
        let gradients = [
            (earth_latitude + transport_latitude).norm(),
            0.0, // longitude: WGS84 is axisymmetric, so neither rate sees it
            transport_altitude.norm(),
            velocity_gradients[0].norm(),
            velocity_gradients[1].norm(),
            0.0, // vertical velocity: `transport_rate` never reads it
        ];
        gradients.map(|gradient| 0.5 * dt * dt * gradient * rate_magnitude)
    }

    /// The attitude rows' *position* columns, finite-differenced in both conventions.
    ///
    /// The half of #339 that was missing outright. Groves 5.46 applies `-(omega_ie +
    /// omega_en) x C` and both rates are functions of latitude -- exactly the dependence #317
    /// gave the velocity rows out of 5.54 -- so these entries are not zero. Before this they
    /// were, and the gap was *first* order in dt, which is why `core/tests/
    /// jacobian_agreement.rs`'s derived bound could not account for it. On the state below,
    /// analytic against finite difference, before:
    ///
    /// ```text
    ///     entry     analytic        numeric         shortfall
    ///     (6,0)      0.0             5.704638e-7     5.7e-7
    ///     (7,0)      0.0            -1.790235e-9     1.8e-9
    ///     (8,0)      0.0             2.725918e-7     2.7e-7
    /// ```
    ///
    /// and `f[(8,0)]` reproduces in closed form as `Omega cos(51.5 deg) dt = 4.54e-7` less
    /// the `sec^2` transport share of 1.82e-7, so it is derivable rather than merely observed.
    ///
    /// The altitude column is checked for the same reason it is checked on the velocity rows:
    /// it is analytically non-zero and numerically inert -- ~1e-14, because the principal
    /// radii move by a part in 1e6 per kilometre -- so it pins the sign of a term that comes
    /// out of the same gradient for free rather than doing any work. Longitude is genuinely
    /// absent, bit for bit.
    #[test]
    fn transition_jacobian_attitude_rows_position_columns_match_finite_differences_in_both_frames()
    {
        let dt = 0.01;
        // 1e-5 rad is ~64 m of latitude, small enough that the second derivative of either
        // rate is invisible and large enough that the rounding floor below stays under the
        // derived second-order bound. At the 1e-6 the velocity rows use, the floor would be
        // 1.1e-10 and would set the tolerance instead.
        let steps = [1e-5, 1e-5, 1.0];
        for base in [frame_check_state(), frame_check_state().to_enu()] {
            let frame = if base.is_enu { "ENU" } else { "NED" };
            let (accel, gyro) = frame_check_imu(base.is_enu);
            let analytic = state_transition_jacobian(&base, &accel, &gyro, dt);
            let bounds = attitude_row_second_order_bounds(&base, dt);

            for (column, step) in steps.iter().enumerate() {
                let numeric = rotation_vector_derivative(&base, column, *step, dt);
                // The numeric side differences `C_pert C_nom^T`, whose entries are 1 +/- the
                // rotation, so it carries an absolute error of ~eps before the division. The
                // altitude column is checked entirely against this floor, its entries being
                // ~1e-14; the latitude column is dominated by the second-order term above.
                let tolerance = bounds[column] + f64::EPSILON / (2.0 * step);
                for row in 0..3 {
                    assert_approx_eq!(analytic[(6 + row, column)], numeric[row], tolerance);
                }
            }

            // Non-degenerate: the two entries that were exactly zero before #339 must now
            // carry something far above the tolerance they are checked against, or the
            // agreement above is an agreement about nothing.
            assert!(
                analytic[(6, 0)].abs() > 1e-8 && analytic[(8, 0)].abs() > 1e-8,
                "in {frame} the attitude rows' latitude terms are empty: f[(6,0)] = {:e}, \
                 f[(8,0)] = {:e}",
                analytic[(6, 0)],
                analytic[(8, 0)]
            );
        }
    }

    /// The attitude rows' non-attitude columns must be converted to the Euler parametrisation
    /// too, not just the attitude/attitude block.
    ///
    /// The other half of #339. These columns are written in rotation-vector form -- `delta
    /// theta+ = -delta omega_in dt` -- and `euler_state_transition_jacobian`'s rows are
    /// increments of the stored angles, so the block needs `E(Phi+)^-1` on the left. The
    /// velocity columns used to be written straight into `f` from the NED formula and were
    /// the one part of the attitude rows that never saw it, leaving them in the wrong
    /// parametrisation in the Euler form. Measured on the state below, before:
    ///
    /// ```text
    ///     entry     analytic        numeric         shortfall
    ///     (6,4)     -1.564051e-9     1.196714e-9     2.8e-9
    ///     (7,3)      1.568134e-9    -1.156443e-9     2.7e-9
    /// ```
    ///
    /// -- sign-flipped and of the same order, which is the signature of a missing left
    /// factor, and not a small correction to the entries but a replacement of them.
    ///
    /// Checked against Euler-angle finite differences rather than against the
    /// rotation-vector matrix times `E^-1`, which would restate the implementation inside its
    /// own oracle.
    #[test]
    fn euler_jacobian_converts_the_attitude_rows_non_attitude_columns() {
        let dt = 0.01;
        let steps = [1e-5, 1e-5, 1.0, 1.0, 1.0, 1.0];
        for base in [frame_check_state(), frame_check_state().to_enu()] {
            let frame = if base.is_enu { "ENU" } else { "NED" };
            let (accel, gyro) = frame_check_imu(base.is_enu);
            let analytic = euler_state_transition_jacobian(&base, &accel, &gyro, dt);
            let bounds = attitude_row_second_order_bounds(&base, dt);

            // The conversion is a linear map, so it carries the rotation-vector bound over
            // with it at its own gain. Frobenius dominates the operator norm, so this stays
            // an upper bound; it is 1.776 here -- the geometry of `E` itself, nothing near
            // gimbal lock, where it would diverge as `sec(pitch)`.
            let nominal = propagate_perturbed(&base, 0, 0.0, dt);
            let (roll, pitch, yaw) = nominal.attitude.euler_angles();
            let conversion_gain = euler_rate_matrix(roll, pitch, yaw)
                .try_inverse()
                .expect("the sample attitude is far from gimbal lock")
                .norm();

            let angles = |state: &StrapdownState| {
                let (roll, pitch, yaw) = state.attitude.euler_angles();
                Vector3::new(roll, pitch, yaw)
            };
            for (column, step) in steps.iter().enumerate() {
                let numeric = (angles(&propagate_perturbed(&base, column, *step, dt))
                    - angles(&propagate_perturbed(&base, column, -*step, dt)))
                    / (2.0 * step);
                // Two contributions, and they arise in different places: the second-order
                // term comes through the conversion and so picks up its gain, while the
                // rounding floor is on the numeric side, which is already in Euler units.
                // Angles are bounded by pi, so differencing two of them costs `eps * pi`
                // before the division -- the altitude column is checked entirely against
                // this, its entries being ~1e-14.
                let tolerance = conversion_gain * bounds[column]
                    + f64::EPSILON * std::f64::consts::PI / (2.0 * step);
                for row in 0..3 {
                    assert_approx_eq!(analytic[(6 + row, column)], numeric[row], tolerance);
                }
            }

            // Non-degenerate: the conversion must actually have moved these columns, or the
            // agreement above would also hold for the unconverted rotation-vector block.
            let rotation_vector = state_transition_jacobian(&base, &accel, &gyro, dt);
            let moved = (6..9)
                .flat_map(|row| (0..6).map(move |column| (row, column)))
                .map(|(row, column)| {
                    (analytic[(row, column)] - rotation_vector[(row, column)]).abs()
                })
                .fold(0.0, f64::max);
            assert!(
                moved > 1e-9,
                "in {frame} the Euler and rotation-vector attitude rows differ by only \
                 {moved:e} outside the attitude columns; if the conversion has been dropped, \
                 these columns have regressed to #339"
            );
        }
    }

    /// The ESKF must linearise the same Coriolis and transport terms the full-state
    /// Jacobian does.
    ///
    /// Before #325 `error_state_transition_jacobian` had no velocity block at all -- the
    /// comment read that the coupling is "small for low dynamics and often approximated as
    /// zero" -- and no Coriolis contribution to its position columns either, so the two
    /// linearisations of Groves 5.54 disagreed about the same physics. That divergence
    /// between the EKF's and the ESKF's idea of the dynamics is the condition #266, #286 and
    /// #307 each arose from.
    ///
    /// The comparison is against the full-state Jacobian rather than a fresh finite
    /// difference on purpose: those blocks are already pinned to the mechanization by
    /// `transition_jacobian_velocity_columns_match_finite_differences_in_both_frames` and
    /// `transition_jacobian_position_columns_match_finite_differences_in_both_frames`, and
    /// what needs guarding here is that the two functions cannot drift apart again.
    #[test]
    fn error_state_jacobian_carries_the_same_coriolis_terms_as_the_full_state_form() {
        let dt = 0.01;

        for base in [frame_check_state(), frame_check_state().to_enu()] {
            let frame = if base.is_enu { "ENU" } else { "NED" };
            let (accel, gyro) = frame_check_imu(base.is_enu);
            let full = state_transition_jacobian(&base, &accel, &gyro, dt);
            let error = error_state_transition_jacobian(&base, &accel, &gyro, dt);

            // The velocity block is built from the same helpers in both, so it agrees to the
            // last bit; 1e-18 says so rather than leaving room for a near-miss.
            for row in 3..6 {
                for column in 3..6 {
                    assert_approx_eq!(full[(row, column)], error[(row, column)], 1e-18);
                }
            }
            // The position columns agree to rounding rather than exactly: both carry the same
            // Coriolis contribution, but the gravity gradient beside it is computed from the
            // Somigliana formula in one and from `gravity_latitude_gradient`'s algebraically
            // equivalent rearrangement in the other. That duplication is real and is left for
            // a separate change; 1e-12 is far under the ~5e-4 entries and far over the
            // difference between two spellings of the same quotient rule.
            for row in 3..6 {
                assert_approx_eq!(full[(row, 0)], error[(row, 0)], 1e-12);
                assert_approx_eq!(full[(row, 2)], error[(row, 2)], 1e-12);
            }

            // Non-degenerate: the off-diagonal Coriolis entries and the latitude column have
            // to actually be there. At 120 m/s the quadratic half is the same order as the
            // linear half, which is the whole of #325.
            assert!(
                error[(4, 3)].abs() > 1e-7,
                "in {frame} the ESKF Coriolis block is empty"
            );
            assert!(
                error[(3, 0)].abs() > 1e-5,
                "in {frame} the ESKF Coriolis latitude column is empty"
            );
        }
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
        crate::mechanize(
            &mut state_nominal,
            &crate::ImuSample::from_rates(
                &crate::IMUData {
                    accel: *imu_accel,
                    gyro: *imu_gyro,
                },
                dt,
            ),
        )
        .unwrap();
        let f0: Vec<f64> = (&state_nominal).into();

        // Perturb each state component
        for j in 0..9 {
            let mut x_pert = x0.clone();
            x_pert[j] += epsilon;

            // Create perturbed state
            let mut state_pert = StrapdownState::try_from(x_pert.as_slice()).unwrap();
            state_pert.is_enu = state.is_enu;

            // Propagate perturbed state
            crate::mechanize(
                &mut state_pert,
                &crate::ImuSample::from_rates(
                    &crate::IMUData {
                        accel: *imu_accel,
                        gyro: *imu_gyro,
                    },
                    dt,
                ),
            )
            .unwrap();
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
        )
        .unwrap();

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
                eprintln!("  ({i}, {j}): {err:.10e}");
            }
        }

        // Compare all elements
        let max_error = (&f_analytic - &f_numeric).abs().max();
        assert!(
            max_error < 1e-6,
            "Max error {max_error} exceeds threshold for stationary state"
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
        )
        .unwrap();

        let accel = Vector3::new(0.1, -0.1, 9.81); // Smaller accelerations
        let gyro = Vector3::new(0.001, -0.001, 0.002); // Smaller rates
        let dt = 0.0001; // Very small dt for high accuracy

        let f_analytic = state_transition_jacobian(&state, &accel, &gyro, dt);
        let f_numeric = numerical_state_jacobian(&state, &accel, &gyro, dt, 1e-6);

        let max_error = (&f_analytic - &f_numeric).abs().max();
        assert!(
            max_error < 1e-5,
            "Max error {max_error} exceeds threshold for moving state. Note: errors ~1e-5 are expected due to nonlinear coupling in trapezoidal integration."
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
            )
            .unwrap();

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
        )
        .unwrap();
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
        )
        .unwrap();

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

        apply_eskf_correction(&mut state, &delta_x).unwrap();

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
        )
        .unwrap();

        let delta_x = DVector::from_vec(vec![
            0.0, 0.0, 0.0, // position
            0.5, -0.3, 0.1, // velocity correction
            0.0, 0.0, 0.0, // attitude
        ]);

        apply_eskf_correction(&mut state, &delta_x).unwrap();

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
        )
        .unwrap();

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

        apply_eskf_correction(&mut state, &delta_x).unwrap();

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

        let biases = apply_eskf_correction_with_biases(&mut state, &delta_x).unwrap();
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

        let biases = apply_eskf_correction_with_biases(&mut state, &delta_x).unwrap();
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

        let delta_x = assemble_error_state(&dr, &mu).unwrap();

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
