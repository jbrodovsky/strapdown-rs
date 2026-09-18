//! #399: a one-ulp change to the filter's input must not move its answer by a metre.
//!
//! The accuracy baseline was not reproducible across small source changes, and the recorded
//! hypothesis was the innovation gate -- a continuous statistic turned into a discrete
//! accept/reject, one flipped decision separating two trajectories. Instrumenting
//! `GatePolicy::decide` refuted it outright: gating is opt-in and `perf_baseline.rs` never
//! installs one, so across the whole suite all **137,876** calls took the "no gate" branch
//! and not one accept/reject decision existed to flip. Every other discrete branch in the
//! filter path was saturated too -- 98,786 of 98,786 sigma-point factorizations took plain
//! Cholesky with no jitter and no eigenvalue fallback, no IMU-bias clamp bound in 394,740
//! opportunities, and the Euler-rate inverse was admitted on all 106,518.
//!
//! The amplifier is arithmetic, not a branch: the scaled unscented transform's own weights.
//! At the `alpha = 1e-3` this crate shipped, a 16-state filter forms its mean as
//! `-999,999 * x_0 + sum 31,250 * x_i`, so six of `f64`'s sixteen significant digits go to
//! cancellation on every propagation step. That is why the UKF was the *only* estimator
//! affected: perturbing one WGS84 constant by a single ulp moved a gated UKF metric by
//! **9.77%** and moved every EKF, ESKF and RBPF metric by less than `1e-6`%, on the same
//! scenarios including an identically-coasting `syn_outage_60s__eskf`.
//!
//! This file is the end-to-end guard, in metres rather than percentages. See
//! `ClosedLoopConfig::ukf_alpha` for the full table.

use strapdown::IMUQuality;
use strapdown::error::StrapdownError;
use strapdown::messages::{AidingConfig, GnssFaultModel, MeasurementScheduler, build_event_stream};
use strapdown::sim::{
    NAVIGATION_STATES, SyntheticConfig, SyntheticInitialState, TestDataRecord, UkfConfig,
    generate_synthetic, initialize_ukf, run_closed_loop,
};

const SEED: u64 = 20_250_914;

fn synthetic_config(duration_s: f64) -> SyntheticConfig {
    let mut built = SyntheticConfig::default();
    built.output = String::new();
    built.initial_state = SyntheticInitialState {
        latitude_deg: 40.0,
        longitude_deg: -75.0,
        altitude_m: 200.0,
        velocity_north_mps: 40.0,
        velocity_east_mps: 30.0,
        velocity_down_mps: 0.0,
        roll_deg: 0.0,
        pitch_deg: 0.0,
        yaw_deg: 36.869_897_645_844_02,
        angular_velocity_x_dps: 0.0,
        angular_velocity_y_dps: 0.0,
        angular_velocity_z_dps: 0.0,
        is_enu: false,
    };
    built.duration_s = duration_s;
    built.sample_rate_hz = 50.0;
    built.imu_quality = IMUQuality::Consumer;
    built.seed = SEED;
    built.no_noise = false;
    built.gnss_horizontal_noise_m = 3.0;
    built.gnss_vertical_noise_m = 5.0;
    built.baro_noise_std_pa = 30.0;
    built.mag_noise_std_ut = 0.5;
    built.mag_hard_iron_std_ut = 0.0;
    built
}

fn aiding() -> AidingConfig {
    let mut built = AidingConfig::default();
    built.scheduler = MeasurementScheduler::DutyCycle {
        on_s: 60.0,
        off_s: 60.0,
        start_phase_s: 0.0,
    };
    built.fault = GnssFaultModel::None;
    built.seed = SEED;
    built.baro_bias_index = Some(NAVIGATION_STATES);
    built
}

/// Run the UKF over `records` and return the final latitude and longitude, in degrees.
///
/// Fallible rather than `expect`-ing internally because the zero-panic lint applies here:
/// `clippy.toml`'s `allow-panic-in-tests` covers code clippy recognises as a test, and a
/// helper a test calls is not that. The callers below are `#[test]` functions and may unwrap.
fn solve(records: &[TestDataRecord]) -> Result<(f64, f64), StrapdownError> {
    let stream = build_event_stream(records, &aiding(), false)?;
    let mut filter = initialize_ukf(&records[0], {
        let mut built = UkfConfig::default();
        built.is_enu = false;
        built.estimate_baro_bias = true;
        built
    })?;
    let out = run_closed_loop(&mut filter, stream, None, None)?;
    let last = out
        .last()
        .ok_or_else(|| StrapdownError::InvalidConfiguration {
            field: "ukf_conditioning::solve",
            reason: "the closed loop returned no rows".to_string(),
        })?;
    Ok((last.latitude, last.longitude))
}

/// The bound, in metres of final position.
///
/// Measured over 180 s at 50 Hz with 60 s of GNSS then 60 s of free inertial -- the shape of
/// `syn_outage_60s`, which is the row #399 reported moving:
///
/// | `ukf_alpha` | final position moved by a one-ulp input change |
/// |---|---:|
/// | `1e-3` (the old default) | **1.76 m** |
/// | `0.1` (shipped) | **0.000178 m** |
///
/// One centimetre sits 55x above what the filter now does and 99x below what it used to,
/// so this fails on the defect and on nothing else. It is a bound on *numerical* sensitivity
/// and says nothing about accuracy: the two runs are the same filter on the same data,
/// differing by the smallest change `f64` can represent.
const MAX_ONE_ULP_POSITION_DIVERGENCE_M: f64 = 0.01;

/// A one-ulp change to the initial latitude must not move the answer by a centimetre.
#[test]
fn one_ulp_of_input_cannot_move_the_solution_by_a_centimetre() {
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
    let (_, records) = generate_synthetic(&synthetic_config(180.0), &mut rng).expect("synthetic");

    let mut nudged = records.clone();
    nudged[0].latitude = f64::from_bits(nudged[0].latitude.to_bits() + 1);
    assert_ne!(
        nudged[0].latitude, records[0].latitude,
        "the perturbation must actually perturb, or this test measures nothing"
    );

    let (lat_a, lon_a) = solve(&records).expect("the baseline run must succeed");
    let (lat_b, lon_b) = solve(&nudged).expect("the perturbed run must succeed");
    let north_m = (lat_b - lat_a).to_radians() * 6_378_137.0;
    let east_m = (lon_b - lon_a).to_radians() * 6_378_137.0 * 40.0_f64.to_radians().cos();
    let divergence_m = north_m.hypot(east_m);

    assert!(
        divergence_m < MAX_ONE_ULP_POSITION_DIVERGENCE_M,
        "one ulp of initial latitude moved the final position by {divergence_m:.6e} m, over \
         the {MAX_ONE_ULP_POSITION_DIVERGENCE_M} m bound. The usual cause is `ukf_alpha` \
         having been lowered: the sigma-point weights go as 1/alpha^2, and at 1e-3 this \
         measures 1.76 m. See #399."
    );
}

/// The cheap early warning, naming the cause rather than the symptom.
///
/// The test above runs a filter for 180 s to find out; this one reads it off the shipped
/// default. `w_0 = 1 - alpha^-2` and `w_i = 1/(2 n alpha^2)`, and every term of the mean is
/// `|w|` times the answer it sums to, so `log10|w_0|` is a direct count of the significant
/// digits cancellation costs on every step.
#[test]
fn the_shipped_sigma_point_weights_do_not_cancel_away_the_mantissa() {
    /// Digits of `f64`'s sixteen that may go to cancellation in the sigma-point mean.
    const MAX_DIGITS_LOST: f64 = 2.5;

    let alpha = strapdown::sim::ClosedLoopConfig::default().ukf_alpha;
    assert!(alpha > 0.0, "alpha must be positive, got {alpha}");

    let weight_zero = 1.0 - 1.0 / (alpha * alpha);
    let digits_lost = weight_zero.abs().log10();
    assert!(
        digits_lost <= MAX_DIGITS_LOST,
        "the default ukf_alpha of {alpha} gives w_0 = {weight_zero:.6}, spending \
         {digits_lost:.2} of f64's ~16 significant digits on cancellation in every \
         sigma-point mean. The textbook 1e-3 spends 6, which is #399. Raise alpha, or \
         re-derive this bound with the end-to-end measurement in this file."
    );

    // And the weights must still be a partition of unity, or the transform is broken rather
    // than merely ill-conditioned.
    let n = 16.0_f64;
    let weight_i = 1.0 / (2.0 * n * alpha * alpha);
    let total = weight_zero + 2.0 * n * weight_i;
    assert!(
        (total - 1.0).abs() < 1e-12,
        "sigma-point weights must sum to 1, got {total}"
    );
}
