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
//! 3. **Whole blocks can be missing without any filter noticing.** The velocity rows carried
//!    no position dependence beyond gravity (#317) and differentiated only one factor of a
//!    quadratic Coriolis term (#325); the attitude rows then had the same latitude dependence
//!    missing from Groves 5.46 that #317 had filled in for 5.54 (#339). Each was ~5e-7 to
//!    3e-5 against an analytic zero, well inside the 1e-4 this file used to allow. Filling
//!    them in is what let the tolerance move to a *derived* 6e-5 -- see `MAX_DISAGREEMENT` --
//!    where every contribution is named and computed rather than being an unexamined budget:
//!    one second-order averaging term and one deliberately-omitted half-step (#338).

use nalgebra::{Rotation3, Vector3};
use strapdown::linearize::{
    error_state_transition_jacobian, euler_state_transition_jacobian, state_transition_jacobian,
};
use strapdown::{IMUData, ImuSample, StrapdownState, mechanize};

const LABELS: [&str; 9] = [
    "lat", "lon", "alt", "v_n", "v_e", "v_v", "roll", "pitch", "yaw",
];

/// Central-difference steps per state element, in that element's own units.
///
/// Latitude and longitude are radians, so a metre-scale step would be enormous; attitude
/// needs a step small enough to stay linear but large enough to clear the `1e-16` floor of
/// the mechanization's own arithmetic.
///
/// These are three to four orders larger than they were, and the reason is the switch to
/// increment-domain differencing below: with the large constant gone, truncation is the only
/// thing left to trade against, so the steps move to where the *analytic* terms are best
/// resolved rather than to where the cancellation is least bad. Verified by sweeping each
/// column an order either side -- the latitude column converges to 2e-8 here, against 3e-8 at
/// the old 1e-9.
const STEPS: [f64; 9] = [1e-6, 1e-6, 1e-1, 1e-2, 1e-2, 1e-2, 1e-5, 1e-5, 1e-5];

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

/// One step of the mechanization, as the *increment* it applies rather than the state it
/// lands on.
///
/// Differencing `mechanize(x) - x` instead of `mechanize(x)` is what makes the position rows
/// measurable. Altitude here is 150 m, so `ulp(150)` is 2.8e-14 and a central difference over
/// a 1e-9 rad latitude step cannot resolve anything below 2.8e-14 / 2e-9 = 1.4e-5 -- which is
/// exactly the residual this sweep used to report on the altitude row, and which #317 read as
/// a longitude artefact. Subtracting the unperturbed value before the cancellation removes
/// the constant that sets that floor; the two forms are identical in exact arithmetic.
fn propagate(state: StrapdownState, imu: &IMUData, dt: f64) -> Vec<f64> {
    let before = Vec::<f64>::from(&state);
    let mut after = state;
    mechanize(&mut after, &ImuSample::from_rates(imu, dt)).expect("mechanize should succeed");
    Vec::<f64>::from(&after)
        .iter()
        .zip(&before)
        .map(|(a, b)| a - b)
        .collect()
}

fn perturbed(base: &StrapdownState, index: usize, delta: f64) -> StrapdownState {
    let mut vector = Vec::<f64>::from(base);
    vector[index] += delta;
    let mut state = StrapdownState::try_from(vector).expect("perturbed state should be valid");
    state.is_enu = base.is_enu;
    state
}

/// Largest absolute disagreement between an analytic Jacobian and the finite-difference one.
///
/// `propagate` returns the increment, so the finite difference is `∂(f(x) - x)/∂x` and the
/// analytic side has to shed its identity to match: hence the `- 1` on the diagonal.
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
            let identity = if row == column { 1.0 } else { 0.0 };
            let difference = (analytic[(row, column)] - identity - numeric).abs();
            if difference > worst {
                worst = difference;
                where_ = format!(
                    "d({})/d({}): analytic {:.6e}, numeric {:.6e} (both less the identity)",
                    LABELS[row],
                    LABELS[column],
                    analytic[(row, column)] - identity,
                    numeric
                );
            }
        }
    }
    (worst, where_)
}

/// Tolerance on the absolute disagreement, in mixed units.
///
/// Derived from the one second-order term `mechanize` carries and a first-order Jacobian does
/// not. Equation 5.47 rotates the sensed increment with the interval-averaged attitude,
/// `0.5 * (C0 + C1) Δv`, where the Jacobian uses `C0` alone. The gap between them is
/// `0.5 (C1 - C0) Δv ≈ 0.5 C0 [δθ×] Δv` with `|δθ| = |ω_ib| dt` and `|Δv| = |f^b| dt`, so the
/// residual it leaves in any column is bounded by
///
/// ```text
///     0.5 * |ω_ib| * |f^b| * dt^2 = 0.5 * 0.03742 * 9.816 * 1e-4 = 1.84e-5
/// ```
///
/// for the sample IMU below.
///
/// **Second contribution: the position rows' half-step, deliberately omitted (#338).**
/// `position_update` integrates each position row trapezoidally over the *propagated*
/// velocity, so the true `f[(row, c)]` contains `0.5 * dt * f[(3 + row, c)]` which a
/// first-order Jacobian does not carry. That is computable from the analytic matrix's own
/// row 5 rather than measured from the residual:
///
/// ```text
///     max_c 0.5 * dt * |f[(5, c)]| = 0.5 * 0.01 * 6.897e-3 = 3.45e-5
/// ```
///
/// the maximum being the roll column in ENU. Summing the two, since the worst entry may take
/// either: `1.84e-5 + 3.45e-5 = 5.3e-5`, and 6e-5 clears it.
///
/// This is a *derived* bound and not the observed residual rounded up (#288): both terms come
/// from the sample IMU and from the Jacobian's own entries, and the half-step prediction is
/// exact -- 2.970e-5 predicted against 2.975e-5 measured for `∂alt/∂pitch` in NED, and
/// -3.449e-5 against -3.451e-5 for `∂alt/∂roll` in ENU. It was 1e-4, loose enough to have
/// accepted the missing Coriolis position terms (3.0e-5) indefinitely. If the mechanization's
/// integration order changes, or #338 lands, recompute from the expressions above rather than
/// fitting the number to whatever comes out.
///
/// **Both contributions are now second order, and #339 is why that is worth stating.** The
/// attitude rows' position columns used to be exactly zero against a genuine *first-order*
/// term of ~5.4e-7 -- the same latitude dependence #317 gave the velocity rows out of Groves
/// 5.54, applied to 5.46. Two orders below the bound, so it never set it, but it scaled with
/// `dt` where everything else here scales with `dt^2`, and fixing #338 alone would have
/// driven the residual down toward something the formula above does not predict. It is fixed,
/// and the two expressions are now the whole budget: the worst entry is `∂alt/∂roll` in ENU
/// at 3.45e-5, exactly the half-step, and the attitude rows' position columns agree to 8.7e-11
/// in this sweep -- see `linearize`'s
/// `transition_jacobian_attitude_rows_position_columns_match_finite_differences_in_both_frames`
/// and `euler_jacobian_converts_the_attitude_rows_non_attitude_columns`, which pin them at
/// their own derived bounds rather than against this file's much looser one.
const MAX_DISAGREEMENT: f64 = 6e-5;

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
/// decided by the frame, and because a sign flip in a position/velocity coupling does not
/// merely mistune a filter -- it inverts the altitude/vertical-velocity feedback loop, so the
/// vertical channel grows without bound from any seed error while a run seeded exactly on
/// truth stays perfectly stable. That combination is what made it survive so long: #307 in
/// `state_transition_jacobian` and #303 in `error_state_transition_jacobian` were the same
/// mistake in two functions.
///
/// All three Jacobians are checked, because all three carry the entry and all three had it
/// wrong at some point.
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
            error_state_transition_jacobian(&state, &imu.accel, &imu.gyro, dt),
        ] {
            let actual = analytic[(2, 5)];
            assert!(
                (actual - expected).abs() < 1e-12,
                "in {} d(alt)/d(v_vertical) should be {expected:.3e}, got {actual:.3e}. \
                 `altitude` is positive up in both frames but `velocity_vertical` is positive \
                 down in NED, so this entry changes sign with the frame (#303, #307)",
                if is_enu { "ENU" } else { "NED" }
            );
        }
    }
}

/// The two parametrisations must actually differ, and in every block that touches attitude.
///
/// If a refactor ever collapses them back into one matrix, the EKF silently regresses to the
/// #307 behaviour. This asserts the distinction is real rather than decorative.
///
/// "Every block that touches attitude" is two things, and #339 is why the distinction has to
/// be drawn: the attitude *columns* are an input-side perturbation, so they differ in the
/// rows that read attitude (velocity and attitude), while the attitude *rows* are an
/// output-side one and differ in **every** column. This test used to claim the two agreed
/// outside the attitude columns, which was true only because the attitude rows' position
/// columns were zero in both forms and their velocity columns were written straight into `f`
/// without the conversion -- the two halves of #339. What is genuinely parametrisation-free
/// is the position and velocity rows' position and velocity columns, and that is what is
/// asserted below.
#[test]
fn the_two_parametrisations_differ_in_every_block_that_touches_attitude() {
    let dt = 0.01;
    let state = sample_state(false);
    let imu = sample_imu(false);
    let euler = euler_state_transition_jacobian(&state, &imu.accel, &imu.gyro, dt);
    let rotation_vector = state_transition_jacobian(&state, &imu.accel, &imu.gyro, dt);

    for row in 0..6 {
        for column in 0..6 {
            let difference = (euler[(row, column)] - rotation_vector[(row, column)]).abs();
            assert!(
                difference < 1e-12,
                "neither the position nor the velocity rows read or produce an attitude \
                 perturbation in their position and velocity columns, so the parametrisations \
                 should agree there -- but d({})/d({}) differs by {difference:.3e}",
                LABELS[row],
                LABELS[column]
            );
        }
    }

    // The attitude columns, in the rows that read them.
    let attitude_column_difference: f64 = (3..9)
        .flat_map(|row| (6..9).map(move |column| (row, column)))
        .map(|(row, column)| (euler[(row, column)] - rotation_vector[(row, column)]).abs())
        .fold(0.0, f64::max);
    assert!(
        attitude_column_difference > 1e-3,
        "the Euler and rotation-vector forms should differ substantially in the attitude \
         columns, but the largest difference is {attitude_column_difference:.3e}. If these \
         have become the same matrix, the EKF has regressed to #307"
    );

    // The attitude rows, in the columns that are *not* attitude -- position and velocity.
    // Far smaller, because `ω_in` is small, but the conversion is the same one: on this state
    // the largest of them is ~1.8e-7, against entries of ~5e-7.
    let attitude_row_difference: f64 = (6..9)
        .flat_map(|row| (0..6).map(move |column| (row, column)))
        .map(|(row, column)| (euler[(row, column)] - rotation_vector[(row, column)]).abs())
        .fold(0.0, f64::max);
    assert!(
        attitude_row_difference > 1e-9,
        "the attitude rows are an output-side perturbation, so they need `E(Φ⁺)⁻¹` in their \
         position and velocity columns too -- but the largest difference there is \
         {attitude_row_difference:.3e}. If the conversion has been dropped, those columns \
         have regressed to #339"
    );
}
