#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//! The analytic transition Jacobians must agree with the mechanization they linearise.
//!
//! This is the test that would have caught #307 years earlier than a divergence on real data
//! did. A Jacobian is a derivative of a specific function, so it can be checked directly
//! against a finite difference of that function rather than inferred from whether a filter
//! built on it happens to converge. The failure mode it guards against is quiet: a wrong
//! block does not crash, it just makes the Kalman gain wrong, and the filter degrades in a
//! way that looks like bad tuning.
//!
//! Two things this test pins down that the filter-level tests could not:
//!
//! 1. **`∂alt/∂v_vertical` is frame-dependent.** `altitude` is positive up in both frames but
//!    `velocity_vertical` is positive *down* in NED, so the entry is `-dt` there and `+dt` in
//!    ENU. It was unconditionally `+dt`.
//! 2. **The attitude columns depend on how attitude is parametrised.** `δ(C f) = -[f^n×] δθ`
//!    holds for a rotation vector. A filter storing Euler angles needs that composed with the
//!    Euler-rate matrix, and the difference is the same size as the terms themselves -- on the
//!    state below a roll perturbation moves north velocity by 7.7e-2 where the
//!    rotation-vector form says exactly zero.

use nalgebra::{Rotation3, Vector3};
use strapdown::linearize::{euler_state_transition_jacobian, state_transition_jacobian};
use strapdown::{IMUData, ImuSample, StrapdownState, mechanize};

const LABELS: [&str; 9] = [
    "lat", "lon", "alt", "v_n", "v_e", "v_v", "roll", "pitch", "yaw",
];

/// Central-difference steps per state element, in that element's own units.
///
/// Latitude and longitude are radians, so a metre-scale step would be enormous; attitude
/// needs a step small enough to stay linear but large enough to clear the `1e-16` floor of
/// the mechanization's own arithmetic.
const STEPS: [f64; 9] = [1e-9, 1e-9, 1e-3, 1e-4, 1e-4, 1e-4, 1e-7, 1e-7, 1e-7];

/// A state with nothing zero or symmetric, so no block can agree by accident.
fn sample_state(is_enu: bool) -> StrapdownState {
    StrapdownState {
        latitude: 40.0_f64.to_radians(),
        longitude: -75.0_f64.to_radians(),
        altitude: 150.0,
        velocity_north: 12.0,
        velocity_east: -4.0,
        velocity_vertical: 0.7,
        attitude: Rotation3::from_euler_angles(0.05, -0.03, 0.9),
        is_enu,
    }
}

const fn sample_imu(is_enu: bool) -> IMUData {
    IMUData {
        // Level-ish, so the vertical axis carries the reaction to gravity with the sign the
        // frame calls for, plus enough horizontal force and rotation to excite every block.
        accel: Vector3::new(0.3, -0.2, if is_enu { 9.81 } else { -9.81 }),
        gyro: Vector3::new(0.01, -0.02, 0.03),
    }
}

fn propagate(mut state: StrapdownState, imu: &IMUData, dt: f64) -> Vec<f64> {
    mechanize(&mut state, &ImuSample::from_rates(imu, dt)).expect("mechanize should succeed");
    Vec::<f64>::from(&state)
}

fn perturbed(base: &StrapdownState, index: usize, delta: f64) -> StrapdownState {
    let mut vector = Vec::<f64>::from(base);
    vector[index] += delta;
    let mut state = StrapdownState::try_from(vector).expect("perturbed state should be valid");
    state.is_enu = base.is_enu;
    state
}

/// Largest absolute disagreement between an analytic Jacobian and the finite-difference one.
fn worst_disagreement(analytic: &nalgebra::DMatrix<f64>, state: &StrapdownState) -> (f64, String) {
    let dt = 0.01;
    let imu = sample_imu(state.is_enu);
    let mut worst = 0.0_f64;
    let mut where_ = String::from("(none)");

    for column in 0..9 {
        let step = STEPS[column];
        let plus = propagate(perturbed(state, column, step), &imu, dt);
        let minus = propagate(perturbed(state, column, -step), &imu, dt);

        for row in 0..9 {
            let numeric = (plus[row] - minus[row]) / (2.0 * step);
            let difference = (analytic[(row, column)] - numeric).abs();
            if difference > worst {
                worst = difference;
                where_ = format!(
                    "d({})/d({}): analytic {:.6e}, numeric {:.6e}",
                    LABELS[row],
                    LABELS[column],
                    analytic[(row, column)],
                    numeric
                );
            }
        }
    }
    (worst, where_)
}

/// Tolerance on the absolute disagreement, in mixed units.
///
/// The residual is dominated by two things that are not modelling errors: the central
/// difference's own truncation, and the second-order attitude averaging in `mechanize` that a
/// first-order Jacobian does not carry. Both leave entries around 3e-5 here. The defects this
/// test exists to catch were 2e-2 (the altitude sign, a full `dt`) and 7.7e-2 (the attitude
/// columns), so this threshold separates them by three orders of magnitude.
///
/// It is deliberately *not* tightened to the observed residual: doing that would make the
/// test fail on an unrelated change to the mechanization's integration order, which is not
/// what it is for.
const MAX_DISAGREEMENT: f64 = 1e-4;

#[test]
fn euler_jacobian_matches_the_mechanization_in_both_frames() {
    for is_enu in [false, true] {
        let state = sample_state(is_enu);
        let imu = sample_imu(is_enu);
        let analytic = euler_state_transition_jacobian(&state, &imu.accel, &imu.gyro, 0.01);
        let (worst, where_) = worst_disagreement(&analytic, &state);
        assert!(
            worst < MAX_DISAGREEMENT,
            "Euler transition Jacobian disagrees with the mechanization in {} by {worst:.3e}, \
             over the {MAX_DISAGREEMENT:.0e} tolerance. Worst entry -- {where_}",
            if is_enu { "ENU" } else { "NED" }
        );
    }
}

/// The altitude row is frame-dependent, and getting it wrong is a pure sign flip.
///
/// Called out separately from the sweep above because it is the one entry whose *sign* is
/// decided by the frame, and a sign flip in a position/velocity coupling is what makes a
/// vertical channel run away rather than merely mistune.
#[test]
fn altitude_row_follows_the_frame() {
    let dt = 0.01;
    for is_enu in [false, true] {
        let state = sample_state(is_enu);
        let imu = sample_imu(is_enu);
        let expected = if is_enu { dt } else { -dt };

        for analytic in [
            euler_state_transition_jacobian(&state, &imu.accel, &imu.gyro, dt),
            state_transition_jacobian(&state, &imu.accel, &imu.gyro, dt),
        ] {
            let actual = analytic[(2, 5)];
            assert!(
                (actual - expected).abs() < 1e-12,
                "in {} d(alt)/d(v_vertical) should be {expected:.3e}, got {actual:.3e}. \
                 `altitude` is positive up in both frames but `velocity_vertical` is positive \
                 down in NED, so this entry changes sign with the frame",
                if is_enu { "ENU" } else { "NED" }
            );
        }
    }
}

/// The two parametrisations must actually differ, and only in the attitude columns.
///
/// If a refactor ever collapses them back into one matrix, the EKF silently regresses to the
/// #307 behaviour. This asserts the distinction is real rather than decorative.
#[test]
fn the_two_parametrisations_differ_only_in_the_attitude_columns() {
    let dt = 0.01;
    let state = sample_state(false);
    let imu = sample_imu(false);
    let euler = euler_state_transition_jacobian(&state, &imu.accel, &imu.gyro, dt);
    let rotation_vector = state_transition_jacobian(&state, &imu.accel, &imu.gyro, dt);

    for row in 0..9 {
        for column in 0..6 {
            let difference = (euler[(row, column)] - rotation_vector[(row, column)]).abs();
            assert!(
                difference < 1e-12,
                "the parametrisations should agree outside the attitude columns, but \
                 d({})/d({}) differs by {difference:.3e}",
                LABELS[row],
                LABELS[column]
            );
        }
    }

    let attitude_difference: f64 = (3..9)
        .flat_map(|row| (6..9).map(move |column| (row, column)))
        .map(|(row, column)| (euler[(row, column)] - rotation_vector[(row, column)]).abs())
        .fold(0.0, f64::max);
    assert!(
        attitude_difference > 1e-3,
        "the Euler and rotation-vector forms should differ substantially in the attitude \
         columns, but the largest difference is {attitude_difference:.3e}. If these have \
         become the same matrix, the EKF has regressed to #307"
    );
}
