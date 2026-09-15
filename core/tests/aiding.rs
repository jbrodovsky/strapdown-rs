//! Innovation gating and ZUPT/ZARU aiding, end to end through the filters.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//!
//! These are the acceptance criteria of #260 and #261 that no unit test can carry, because
//! both are claims about what a filter *does over a run* rather than about what a function
//! returns:
//!
//! - #260: gating rejects artificial outliers while accepting valid fixes under noise.
//! - #261: ZUPT/ZARU prevent position and attitude drift during prolonged stops.
//!
//! Both are stated as comparisons against the same filter without the feature, on the same
//! seeded stream. An absolute bound on the gated run alone would pass just as happily on a
//! filter that had stopped updating at all; the paired form cannot.
//!
//! # Tolerances
//!
//! The bounds below are empirical, matching the convention in `filter_comparison.rs` and
//! `integration_tests.rs`: measured on a run of this scenario and given headroom. They are
//! regression checks, not design targets. Each carries the measured value it was derived
//! from so a future change can tell "this got worse" from "this was always marginal".

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use strapdown::earth::haversine_distance;
use strapdown::gating::InnovationGate;
use strapdown::kalman::{
    ErrorStateKalmanFilter, ExtendedKalmanFilter, InitialState, UnscentedKalmanFilter,
};
use strapdown::measurements::{
    GPSPositionAndVelocityMeasurement, ZaruMeasurement, ZuptMeasurement,
};
use strapdown::sim::DEFAULT_PROCESS_NOISE;
use strapdown::stationary::{StationaryConfig, StationaryDetector};
use strapdown::{IMUData, ImuSample, NavigationFilter, StrapdownState, mechanize};

/// Scenario latitude, degrees. Mid-latitude, so Earth rate has both a north and a down
/// component and neither drops out of the ZARU geometry.
const LATITUDE_DEG: f64 = 40.0;
/// Scenario longitude, degrees.
const LONGITUDE_DEG: f64 = -105.0;
/// Scenario altitude, meters above the ellipsoid.
const ALTITUDE_M: f64 = 1000.0;

/// Shared process noise, 15-state. Matches `filter_comparison.rs` so a divergence between
/// the two suites is attributable to the feature under test rather than to the tuning.
/// Per-step position process noise, expressed in meters and converted below.
const POSITION_PROCESS_NOISE_M: f64 = 0.1;

fn process_noise() -> [f64; 15] {
    let horizontal_rad = meters_to_radians(POSITION_PROCESS_NOISE_M).powi(2);
    [
        horizontal_rad,
        horizontal_rad,
        POSITION_PROCESS_NOISE_M.powi(2), // altitude, meters
        1e-3,
        1e-3,
        1e-3, // velocity
        1e-5,
        1e-5,
        1e-5, // attitude
        1e-6,
        1e-6,
        1e-6, // accelerometer bias
        1e-8,
        1e-8,
        1e-8, // gyroscope bias
    ]
}

/// Initial horizontal position uncertainty, meters.
const INITIAL_POSITION_STD_M: f64 = 10.0;

/// Convert a horizontal distance in meters to the equivalent angle in radians.
///
/// Every position quantity in this file is written in meters and converted here exactly
/// once. Writing them directly in rad^2 is what makes `1e-6` look reasonable next to an
/// altitude term in m^2 when it is in fact a 6.4 km standard deviation.
fn meters_to_radians(meters: f64) -> f64 {
    (meters * strapdown::earth::METERS_TO_DEGREES).to_radians()
}

fn initial_covariance() -> [f64; 15] {
    let horizontal = meters_to_radians(INITIAL_POSITION_STD_M).powi(2);
    [
        horizontal,
        horizontal,
        INITIAL_POSITION_STD_M.powi(2), // altitude, meters
        0.1,
        0.1,
        0.1, // velocity
        0.01,
        0.01,
        0.01, // attitude
        0.01,
        0.01,
        0.01, // accelerometer bias
        0.001,
        0.001,
        0.001, // gyroscope bias
    ]
}

/// UKF sigma-point tuning, matching `sim::default_ukf_*`.
const UKF_ALPHA: f64 = 1e-3;
const UKF_BETA: f64 = 2.0;
const UKF_KAPPA: f64 = 0.0;

fn process_noise_matrix() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_row_slice(&process_noise()))
}

/// The seed state in the form the Kalman-family constructors take.
///
/// Built as a struct literal rather than through [`InitialState::new`], matching the other
/// test seeds in `filter_comparison.rs` and `integration_tests.rs`. The constructor is
/// equally correct -- `engine.rs` and the `core/examples` binaries use it -- and this is
/// only a convention among the test fixtures.
fn initial_state(state: &StrapdownState) -> InitialState {
    let (roll, pitch, yaw) = state.attitude.euler_angles();
    InitialState {
        latitude: state.latitude,
        longitude: state.longitude,
        altitude: state.altitude,
        northward_velocity: state.velocity_north,
        eastward_velocity: state.velocity_east,
        vertical_velocity: state.velocity_vertical,
        roll,
        pitch,
        yaw,
        in_degrees: false,
        is_enu: false,
    }
}

/// Inertial stream a perfect, perfectly level IMU would report for `state`.
///
/// Specific force opposing gravity, and the angular rate that holds the platform level
/// against Earth rate and transport rate. Same construction as `filter_comparison.rs`: the
/// stream is the specification and [`mechanize`] defines what it means, so truth and the
/// generator cannot disagree.
fn level_imu(state: &StrapdownState) -> IMUData {
    let latitude_deg = state.latitude.to_degrees();
    let velocity = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_vertical,
    );
    let gravity = strapdown::earth::gravity(&latitude_deg, &state.altitude);
    let nav_rate = strapdown::earth::earth_rate_lla(&latitude_deg)
        + strapdown::earth::transport_rate(&latitude_deg, &state.altitude, &velocity);
    IMUData {
        accel: state.attitude.inverse() * Vector3::new(0.0, 0.0, -gravity),
        gyro: state.attitude.inverse() * nav_rate,
    }
}

/// Inverse of [`meters_to_radians`], for reading a covariance entry back in meters.
fn radians_to_meters(radians: f64) -> f64 {
    radians.to_degrees() / strapdown::earth::METERS_TO_DEGREES
}

/// Horizontal great-circle distance between an estimate and a truth state, meters.
fn horizontal_error_m(estimate: &DVector<f64>, truth: &StrapdownState) -> f64 {
    haversine_distance(estimate[0], estimate[1], truth.latitude, truth.longitude)
}

/// Wrap an angle into `[-pi, pi]` so an error either side of the branch cut is comparable.
fn wrap_to_pi(angle_rad: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let wrapped = angle_rad.rem_euclid(two_pi);
    if wrapped > std::f64::consts::PI {
        wrapped - two_pi
    } else {
        wrapped
    }
}

/// Largest per-axis attitude error between an estimate and a truth state, radians.
fn attitude_error_rad(estimate: &DVector<f64>, truth: &StrapdownState) -> f64 {
    let (roll, pitch, yaw) = truth.attitude.euler_angles();
    [
        wrap_to_pi(estimate[6] - roll),
        wrap_to_pi(estimate[7] - pitch),
        wrap_to_pi(estimate[8] - yaw),
    ]
    .into_iter()
    .map(f64::abs)
    .fold(0.0, f64::max)
}

// ============================================================ #260: innovation gating

/// Duration of the gating scenario, seconds.
const GATING_DURATION_S: usize = 100;
/// Inertial sample rate for the gating scenario, Hz.
const GATING_SAMPLE_RATE_HZ: usize = 5;
/// One GNSS fix per second at 5 Hz.
const GPS_DECIMATION: usize = 5;
/// Northward ground speed held for the gating run, m/s.
const GATING_VELOCITY_NORTH_MPS: f64 = 10.0;
/// Reported horizontal accuracy of a fix, meters.
const GPS_HORIZONTAL_NOISE_M: f64 = 5.0;
/// Reported vertical accuracy of a fix, meters.
const GPS_VERTICAL_NOISE_M: f64 = 2.0;
/// Reported velocity accuracy of a fix, m/s.
const GPS_VELOCITY_NOISE_MPS: f64 = 0.2;
/// Repeating pattern of fix offsets, in units of the declared horizontal accuracy.
///
/// A fixed bounded pattern rather than an RNG draw: the tests assert that *no* honest fix
/// is gated out, which against random noise would be a statement about the tail of a
/// particular seed. Spans +/-1.2 sigma, which is where honest fixes live.
const NOISE_PATTERN: [f64; 7] = [0.4, -1.1, 0.7, -0.3, 1.2, -0.8, 0.1];

/// Length of [`NOISE_PATTERN`], as the width [`fix_noise_rms_m`] averages over.
const NOISE_PATTERN_LEN: u8 = 7;

/// Fix-noise amplitudes [`a_note_on_filter_consistency`] sweeps, as multiples of
/// [`NOISE_PATTERN`].
///
/// Three decades, because the point of the sweep is the *shape* of the response rather than
/// any single row. A vertical leak that is second-order in the disturbance is a
/// linearisation dropping curvature; one that is independent of the disturbance is an
/// unstable mode being excited. Those two are indistinguishable at a single amplitude, which
/// is the whole reason #303 was hard to characterise.
const NOISE_SCALES: [f64; 5] = [0.0, 0.1, 1.0, 10.0, 100.0];

/// Index in [`NOISE_SCALES`] of the scenario every other test in this file runs.
///
/// The bound is asserted at this amplitude and below. The two rows above it are diagnostics:
/// 38 m and 380 m rms are not GNSS, they are there to expose the scaling law.
const NOMINAL_NOISE_SCALE: usize = 2;

/// A northbound run with realistically noisy GNSS fixes and no outliers.
struct GatingScenario {
    samples: Vec<ImuSample>,
    gps: Vec<GPSPositionAndVelocityMeasurement>,
    truth: Vec<StrapdownState>,
    initial: StrapdownState,
}

/// Build the gating scenario: level northbound at 10 m/s, seeded on truth.
///
/// Outliers are *not* baked in here. They are applied individually by the tests that need
/// them, sized against the filter's own reported covariance -- see
/// [`gating_rejects_a_fix_inconsistent_with_the_filters_own_uncertainty`] for why a fixed
/// displacement in metres would test the tuning rather than the gate. Fix noise comes from
/// [`NOISE_PATTERN`].
fn build_gating_scenario() -> GatingScenario {
    build_gating_scenario_scaled(1.0)
}

/// [`build_gating_scenario`], with [`NOISE_PATTERN`] scaled by `noise_scale`.
///
/// Split out for [`a_note_on_filter_consistency`], which is a statement about how the
/// filters respond to the *amplitude* of the fix noise and so cannot use a single fixed one.
fn build_gating_scenario_scaled(noise_scale: f64) -> GatingScenario {
    let truth_initial = StrapdownState {
        latitude: LATITUDE_DEG.to_radians(),
        longitude: LONGITUDE_DEG.to_radians(),
        altitude: ALTITUDE_M,
        velocity_north: GATING_VELOCITY_NORTH_MPS,
        velocity_east: 0.0,
        velocity_vertical: 0.0,
        attitude: Rotation3::identity(),
        is_enu: false,
    };

    let dt = 1.0 / GATING_SAMPLE_RATE_HZ as f64;
    let num_samples = GATING_DURATION_S * GATING_SAMPLE_RATE_HZ;
    let mut samples = Vec::with_capacity(num_samples);
    let mut gps = Vec::with_capacity(num_samples);
    let mut truth = Vec::with_capacity(num_samples + 1);

    let mut current = truth_initial;
    for index in 0..num_samples {
        truth.push(current);
        let imu = level_imu(&current);
        let sample = ImuSample::from_rates(&imu, dt);
        samples.push(sample);
        mechanize(&mut current, &sample).unwrap();

        // Recorded *after* the propagation, so `gps[i]` describes the state a filter holds
        // once it has consumed `samples[i]` -- see the note in `filter_comparison.rs` on
        // why a one-sample-stale fix diverges the vertical channel.
        let jitter = NOISE_PATTERN[index % NOISE_PATTERN.len()] * noise_scale;
        let mut latitude_deg = current.latitude.to_degrees();
        let meters_to_degrees = strapdown::earth::METERS_TO_DEGREES;
        latitude_deg += jitter * GPS_HORIZONTAL_NOISE_M * meters_to_degrees;

        gps.push(GPSPositionAndVelocityMeasurement {
            latitude: latitude_deg,
            longitude: current.longitude.to_degrees(),
            altitude: current.altitude,
            northward_velocity: current.velocity_north,
            eastward_velocity: current.velocity_east,
            horizontal_noise_std: GPS_HORIZONTAL_NOISE_M,
            vertical_noise_std: GPS_VERTICAL_NOISE_M,
            velocity_noise_std: GPS_VELOCITY_NOISE_MPS,
        });
    }
    truth.push(current);

    GatingScenario {
        samples,
        gps,
        truth,
        initial: truth_initial,
    }
}

/// Drive one filter through the whole gating scenario, returning the fixes the gate
/// rejected.
fn rejected_fixes(
    filter: &mut dyn NavigationFilter,
    scenario: &GatingScenario,
    gate: Option<InnovationGate>,
) -> Vec<usize> {
    filter.set_innovation_gate(gate);
    let mut rejected = Vec::new();
    for (index, sample) in scenario.samples.iter().enumerate() {
        filter.predict(sample, sample.dt).unwrap();
        if index % GPS_DECIMATION == 0 && !filter.update(&scenario.gps[index]).unwrap().accepted {
            rejected.push(index);
        }
    }
    rejected
}

/// Build the three Kalman-family filters seeded on the scenario's initial state.
///
/// Returned as trait objects in a fixed order so a filter that stopped implementing
/// [`NavigationFilter`] fails to build here rather than quietly dropping out.
fn gating_filters(initial: &StrapdownState) -> Vec<(&'static str, Box<dyn NavigationFilter>)> {
    filters_with_process_noise(initial, &process_noise_matrix())
}

/// [`gating_filters`], with the process noise supplied by the caller.
///
/// Split out for [`the_shipped_default_process_noise_lets_the_eskf_filter`], which is a
/// test *of* a process-noise diagonal and so cannot use this file's own.
fn filters_with_process_noise(
    initial: &StrapdownState,
    process_noise: &DMatrix<f64>,
) -> Vec<(&'static str, Box<dyn NavigationFilter>)> {
    let init = initial_state(initial);
    let biases = [0.0_f64; 6];
    vec![
        (
            "ESKF",
            Box::new(ErrorStateKalmanFilter::new(
                &init,
                &biases,
                initial_covariance().to_vec(),
                process_noise.clone(),
            )) as Box<dyn NavigationFilter>,
        ),
        (
            "EKF",
            Box::new(ExtendedKalmanFilter::new(
                &init,
                &biases,
                initial_covariance().to_vec(),
                process_noise.clone(),
                true,
            )) as Box<dyn NavigationFilter>,
        ),
        (
            "UKF",
            Box::new(UnscentedKalmanFilter::new(
                &init,
                &biases,
                None,
                initial_covariance().to_vec(),
                process_noise.clone(),
                UKF_ALPHA,
                UKF_BETA,
                UKF_KAPPA,
            )) as Box<dyn NavigationFilter>,
        ),
    ]
}

#[test]
fn gating_accepts_every_honest_fix() {
    // The half of #260 that is easy to get wrong in the safe direction: a gate tight
    // enough to reject real noise silently degrades the run to dead reckoning while
    // reporting nothing. Every fix here is within 1.2 sigma.
    let scenario = build_gating_scenario();
    for (name, mut filter) in gating_filters(&scenario.initial) {
        let rejected = rejected_fixes(
            filter.as_mut(),
            &scenario,
            Some(InnovationGate::chi_squared(0.999).unwrap()),
        );
        assert!(
            rejected.is_empty(),
            "{name} gated out {} honest fixes (first at index {:?})",
            rejected.len(),
            rejected.first()
        );
    }
}

#[test]
fn gating_rejects_a_fix_inconsistent_with_the_filters_own_uncertainty() {
    // A gate is specified against the filter's *own* innovation covariance, not against
    // metres: it rejects what the filter's model says should not happen. So the outlier
    // here is sized in sigmas read back from the filter after it has settled, which
    // makes the test independent of how well any particular filter is tuned.
    //
    // That independence is not academic. See `a_note_on_filter_consistency` below, which
    // measures it: the EKF and UKF report horizontal position uncertainties of 450 m and
    // 201 m while sitting within 6 m of truth, so a fixed 200 m displacement is *within*
    // what either believes possible and is correctly not gated. Sizing the outlier in
    // sigmas tests the gate; sizing it in metres would test the tuning.
    const OUTLIER_SIGMAS: f64 = 50.0;

    let scenario = build_gating_scenario();
    for (name, mut filter) in gating_filters(&scenario.initial) {
        let gate = InnovationGate::chi_squared(0.999).unwrap();
        filter.set_innovation_gate(Some(gate));

        // Settle the filter on clean data first, so its covariance reflects a converged
        // run rather than the initial guess.
        for (index, sample) in scenario.samples.iter().enumerate() {
            filter.predict(sample, sample.dt).unwrap();
            if index % GPS_DECIMATION == 0 {
                let outcome = filter.update(&scenario.gps[index]).unwrap();
                assert!(
                    outcome.accepted,
                    "{name} gated an honest fix while settling"
                );
            }
        }

        let before_state = filter.get_estimate();
        let before_covariance = filter.get_certainty();
        let position_sigma_rad = before_covariance[(0, 0)].sqrt();

        let last = scenario.gps.last().unwrap();
        let outlier = GPSPositionAndVelocityMeasurement {
            latitude: last.latitude + (OUTLIER_SIGMAS * position_sigma_rad).to_degrees(),
            ..last.clone()
        };

        let outcome = filter.update(&outlier).unwrap();
        assert!(
            !outcome.accepted,
            "{name} accepted a {OUTLIER_SIGMAS} sigma fix (NIS {:.3}, threshold {:.3})",
            outcome.nis,
            gate.threshold(outcome.dof)
        );
        assert!(
            outcome.nis > gate.threshold(outcome.dof),
            "{name} reported a NIS below the threshold it was rejected on"
        );

        // The property the whole design rests on: a rejected update leaves the state and
        // the covariance exactly as they were. For the ESKF this is load-bearing rather
        // than merely tidy -- `inject_error_state` mutates the nominal state and zeroes
        // the error state, so there is no way to undo a correction once it has started.
        assert_eq!(
            filter.get_estimate(),
            before_state,
            "{name} modified the state on a rejected update"
        );
        assert_eq!(
            filter.get_certainty(),
            before_covariance,
            "{name} modified the covariance on a rejected update"
        );
    }
}

#[test]
fn an_ungated_filter_still_reports_its_nis() {
    // `HealthMonitor` consumes the NIS of every update, gate or no gate: that is how a
    // run that is merely unlucky is told apart from one that has diverged. A filter with
    // no gate must therefore still report a real statistic, not a placeholder.
    let scenario = build_gating_scenario();
    for (name, mut filter) in gating_filters(&scenario.initial) {
        filter.set_innovation_gate(None);
        for (index, sample) in scenario.samples.iter().enumerate() {
            filter.predict(sample, sample.dt).unwrap();
            if index % GPS_DECIMATION == 0 {
                let outcome = filter.update(&scenario.gps[index]).unwrap();
                assert!(outcome.accepted, "{name} rejected with no gate installed");
                assert_eq!(outcome.dof, 5, "{name} reported the wrong DOF");
                assert!(
                    outcome.nis.is_finite() && outcome.nis >= 0.0,
                    "{name} reported an unusable NIS: {}",
                    outcome.nis
                );
            }
        }
    }
}

#[test]
fn every_filter_honours_the_gate_setting() {
    // `set_innovation_gate` returns whether the filter will honour the gate. A filter
    // that quietly ignored it would make every gating assertion above vacuous.
    let scenario = build_gating_scenario();
    for (name, mut filter) in gating_filters(&scenario.initial) {
        assert!(
            filter.set_innovation_gate(Some(InnovationGate::default())),
            "{name} does not honour innovation gating"
        );
    }
}

#[test]
fn gating_keeps_a_rejected_outlier_out_of_the_solution() {
    // The point of rejecting them, stated as a pair: the same filter, the same stream,
    // the same single bad fix, with and without the gate. An absolute bound on the gated
    // run alone would pass just as happily on a filter that had stopped updating.
    const OUTLIER_SIGMAS: f64 = 50.0;
    let scenario = build_gating_scenario();

    for ((name, mut ungated), (_, mut gated)) in gating_filters(&scenario.initial)
        .into_iter()
        .zip(gating_filters(&scenario.initial))
    {
        let mut errors = Vec::new();
        for (filter, gate) in [
            (ungated.as_mut(), None),
            (
                gated.as_mut(),
                Some(InnovationGate::chi_squared(0.999).unwrap()),
            ),
        ] {
            filter.set_innovation_gate(gate);
            for (index, sample) in scenario.samples.iter().enumerate() {
                filter.predict(sample, sample.dt).unwrap();
                if index % GPS_DECIMATION == 0 {
                    filter.update(&scenario.gps[index]).unwrap();
                }
            }
            let position_sigma_rad = filter.get_certainty()[(0, 0)].sqrt();
            let last = scenario.gps.last().unwrap();
            filter
                .update(&GPSPositionAndVelocityMeasurement {
                    latitude: last.latitude + (OUTLIER_SIGMAS * position_sigma_rad).to_degrees(),
                    ..last.clone()
                })
                .unwrap();
            errors.push(horizontal_error_m(
                &filter.get_estimate(),
                scenario.truth.last().unwrap(),
            ));
        }

        let (without, with) = (errors[0], errors[1]);
        println!("{name}: ungated {without:.3} m, gated {with:.3} m after one outlier");
        assert!(
            with < without,
            "{name}: gating did not reduce the error the outlier caused \
             ({with:.3} m gated vs {without:.3} m ungated)"
        );
    }
}

/// Root-mean-square of the fix perturbation [`build_gating_scenario_scaled`] applies, meters.
fn fix_noise_rms_m(noise_scale: f64) -> f64 {
    let mean_square =
        NOISE_PATTERN.iter().map(|v| v * v).sum::<f64>() / f64::from(NOISE_PATTERN_LEN);
    mean_square.sqrt() * GPS_HORIZONTAL_NOISE_M * noise_scale
}

/// |altitude error| after the step just taken, checked finite before it is returned.
///
/// The check is explicit rather than left to the comparison in [`peak_altitude_error_m`]:
/// `f64::max` returns its *finite* operand when the other is NaN, so a filter that had
/// diverged all the way to a non-finite state would leave the running peak untouched and
/// sail through the bound. A NaN is exactly what the divergence this test guards against
/// ends in, so the one state that must never pass would have been the one that always did.
fn checked_altitude_error_m(
    name: &str,
    filter: &dyn NavigationFilter,
    scenario: &GatingScenario,
    index: usize,
    stage: &str,
) -> f64 {
    let altitude = filter.get_estimate()[2];
    assert!(
        altitude.is_finite(),
        "{name} altitude is {altitude} after the {stage} at sample {index}"
    );
    (altitude - scenario.truth[index + 1].altitude).abs()
}

/// Largest |altitude error| over a whole run, sampled after every step the filter takes.
///
/// After each `predict` as well as each `update`. Sampling only on fix epochs would see one
/// step in [`GPS_DECIMATION`], and a channel that ran away between fixes and was hauled back
/// by each one would never appear in the peak at all.
fn peak_altitude_error_m(
    name: &str,
    filter: &mut dyn NavigationFilter,
    scenario: &GatingScenario,
) -> f64 {
    filter.set_innovation_gate(None);
    let mut peak_m: f64 = 0.0;
    for (index, sample) in scenario.samples.iter().enumerate() {
        filter.predict(sample, sample.dt).unwrap();
        peak_m = peak_m.max(checked_altitude_error_m(
            name, &*filter, scenario, index, "predict",
        ));
        if index % GPS_DECIMATION == 0 {
            filter.update(&scenario.gps[index]).unwrap();
            peak_m = peak_m.max(checked_altitude_error_m(
                name, &*filter, scenario, index, "update",
            ));
        }
    }
    peak_m
}

#[test]
fn a_note_on_filter_consistency() {
    // Not a test of anything in this PR. It is the reason the gating tests above size their
    // outlier in sigmas rather than in metres, recorded as a runnable fact rather than a
    // comment that can quietly stop being true.
    //
    // # The vertical channel
    //
    // #303 reported that the ESKF and EKF vertical channels run away from any non-zero seed
    // error, with the UKF unaffected, and fingered the analytic Jacobians in `linearize.rs`.
    // This scenario reaches the same coupling without any seed error at all: every filter
    // starts exactly on truth, and `NOISE_PATTERN` perturbs the fixes in *latitude only*.
    // The fix altitude is truth to the bit, so every metre of altitude error measured below
    // is horizontal-to-vertical leakage through the transition Jacobian and nothing else.
    //
    // That leakage is now bounded, and -- more to the point -- it scales the way a correct
    // linearisation says it must. Peak |altitude error| against the amplitude of the
    // horizontal disturbance driving it, over `NOISE_SCALES`. This test runs that sweep and
    // prints the table, so these rows are output rather than recollection:
    //
    //     fix noise (rms) | ESKF      | EKF      | UKF
    //     ----------------|-----------|----------|--------
    //     0 m             | 2.3e-13 m | 0.0 m    | 0.279 m
    //     0.38 m          | 1.7e-7 m  | 1.4e-8 m | 0.279 m
    //     3.80 m (this)   | 1.5e-5 m  | 1.4e-7 m | 0.279 m
    //     38.0 m          | 1.5e-3 m  | 1.4e-6 m | 0.279 m
    //     380 m           | 1.5e-1 m  | 1.4e-5 m | 0.279 m
    //
    // The ESKF's leak is *quadratic* in the disturbance -- ten times the excitation for a
    // hundred times the error -- which is the signature of a first-order term that cancels
    // exactly, leaving only the curvature the linearisation legitimately drops. The EKF's is
    // linear and seven orders of magnitude smaller than the horizontal disturbance producing
    // it. The UKF's is flat across all three decades of excitation because it is not a
    // response to the fixes at all: it is the sigma-point transient out of
    // `INITIAL_POSITION_STD_M`, and it is there in full with noise-free fixes.
    //
    // None of that is the "unstable mode being excited" #303 describes, and the issue is not
    // reproducible anywhere in this history. This test passes at every commit since the file
    // landed in `546675d`: at each one that touched `linearize.rs` (`d5af42d`, `ce15e71`,
    // `ffc0a1d`), and at each of the ten merged by #342 -- including `bdffed9`, the
    // Coriolis/transport differentiation of #325/#317 that was the obvious candidate for
    // having fixed it, which it cannot have been, because the test is green at its parent
    // too. The `#[ignore]` it used to carry, and the 9.8e7 m / 2.0e4 m divergences recorded
    // with it, describe a state of the aiding branch that predates anything reachable from
    // here.
    //
    // # The horizontal channel, which is still inconsistent
    //
    // The vertical half being healthy does not make the filters consistent, and the reason
    // `gating_rejects_a_fix_inconsistent_with_the_filters_own_uncertainty` sizes its outlier
    // in sigmas survives intact. Reported horizontal position sigma against the error
    // actually achieved, printed by this test alongside the sweep above:
    //
    //     filter | reported lat sigma | peak horizontal error
    //     -------|--------------------|----------------------
    //     ESKF   | 1.24 m             | 2.14 m
    //     EKF    | 450.22 m           | 6.00 m
    //     UKF    | 201.40 m           | 6.04 m
    //
    // Measured with this file's `process_noise()`; the `#[ignore]`d companion
    // `the_shipped_default_process_noise_lets_every_filter_filter` quotes ~493 m for the EKF
    // because it drives `DEFAULT_PROCESS_NOISE` instead. Both say the same thing.
    //
    // The EKF and UKF believe their horizontal position is uncertain to hundreds of metres
    // while sitting within six of truth -- and that 6 m is exactly the largest perturbation
    // in `NOISE_PATTERN` (1.2 * `GPS_HORIZONTAL_NOISE_M`), so both are landing on each fix
    // rather than averaging across them. A filter whose reported uncertainty
    // exceeds its actual error by two orders of magnitude cannot gate: a genuine 200 m
    // multipath fix is well *within* what it believes possible and is correctly accepted.
    // Sizing the outlier in sigmas tests the gate; sizing it in metres would test the tuning.
    // Gating will only do what #260 asks of it once that is fixed.
    //
    // # The bound
    //
    // Set by the UKF at 0.279 m, four orders of magnitude above the ESKF's peak and six
    // above the EKF's. It is a seed transient rather than a response to the fixes, so it
    // moves if `INITIAL_POSITION_STD_M` or the `UKF_*` tuning is changed and will not move
    // if the Jacobians regress. Half a metre gives that 1.8x, and sits under
    // both independent ceilings that carry meaning here -- the 1.12-1.19 m settled altitude
    // sigma all three filters report, and `GPS_VERTICAL_NOISE_M`, the accuracy a single fix
    // is declared to have. A run that exceeds it has put the vertical error outside the
    // uncertainty the filter itself advertises. The scenario is fully deterministic (fixed
    // `NOISE_PATTERN`, no RNG), so the margin is headroom for retuning, not for variance.
    const MAX_ALTITUDE_ERROR_M: f64 = 0.5;

    // The vertical table: peak |altitude error| against the amplitude driving it.
    for (scale_index, noise_scale) in NOISE_SCALES.into_iter().enumerate() {
        let scenario = build_gating_scenario_scaled(noise_scale);
        let fix_rms_m = fix_noise_rms_m(noise_scale);
        for (name, mut filter) in gating_filters(&scenario.initial) {
            let peak_m = peak_altitude_error_m(name, filter.as_mut(), &scenario);
            println!("{name}: {fix_rms_m:.3} m rms fixes -> peak |altitude error| {peak_m:.3e} m");
            if scale_index <= NOMINAL_NOISE_SCALE {
                assert!(
                    peak_m < MAX_ALTITUDE_ERROR_M,
                    "{name} vertical channel reached {peak_m:.3e} m under {fix_rms_m:.3} m rms \
                     fix noise"
                );
            }
        }
    }

    // The horizontal table: what each filter believes about its position against what it
    // achieved. Reported, not asserted -- this half is the open defect, and an assertion
    // here would be a test written to fail.
    let scenario = build_gating_scenario();
    for (name, mut filter) in gating_filters(&scenario.initial) {
        filter.set_innovation_gate(None);
        let mut peak_horizontal_m: f64 = 0.0;
        for (index, sample) in scenario.samples.iter().enumerate() {
            filter.predict(sample, sample.dt).unwrap();
            if index % GPS_DECIMATION == 0 {
                filter.update(&scenario.gps[index]).unwrap();
            }
            peak_horizontal_m = peak_horizontal_m.max(horizontal_error_m(
                &filter.get_estimate(),
                &scenario.truth[index + 1],
            ));
        }
        println!(
            "{name}: reports {:.2} m latitude sigma against a {peak_horizontal_m:.2} m peak \
             horizontal error",
            radians_to_meters(filter.get_certainty()[(0, 0)].sqrt())
        );
    }
}

// ================================================== #308: the shipped process noise

/// Number of GNSS fixes discarded before the statistics below are collected.
///
/// Both assertions are statements about the *steady state*, so the initial transient from
/// [`INITIAL_POSITION_STD_M`] down to the settled prior has to be excluded or it dominates
/// the averages. Half the run is chosen against the scalar Riccati recursion this scenario
/// reduces to. With $R = (5\text{ m})^2$, a per-fix-interval process variance of
/// $5 \times (0.1\text{ m})^2$ (five 5 Hz predicts between 1 Hz fixes) and
/// $P_0 = (10\text{ m})^2$, the information form $1/P_{k} = 1/P_0 + k/R$ reaches the
/// recursion's 1.14 m^2 fixed point in about twenty fixes. Fifty is a little over twice
/// that, and leaves fifty fixes of statistics.
const PROCESS_NOISE_SETTLING_FIXES: usize = GATING_DURATION_S / 2;

/// Ceiling on the settled prior's share of the innovation covariance, $HPH^T / R$.
///
/// The mechanism #308 describes, stated as an inequality: $S = HPH^T + R$ dominated by the
/// prior rather than by the measurement. One is the break-even point, and it is a property
/// of the filter rather than of this scenario -- $HPH^T = R$ is a Kalman gain of exactly
/// $1/2$, the point either side of which the update trusts the prediction more or the fix
/// more. A filter whose prior is *worse* than a single fix is not filtering: it can only
/// ever discard what it knows and copy the measurement, which is precisely what the defect
/// made all three filters do.
///
/// The same scalar recursion predicts the healthy value at $1.14 / 25 = 0.046$, and the
/// ESKF measures 0.063 -- the excess is the fifteen-state coupling the scalar model leaves
/// out. So this is not a close-run bound; it is set where it is because that is where the
/// meaning is, not for the margin. Against the pre-#308 diagonal the ratio is 8.1e6.
const MAX_PRIOR_SHARE_OF_INNOVATION_COVARIANCE: f64 = 1.0;

/// Ceiling on the fraction of the GNSS fix noise allowed through into the solution.
///
/// The other half of the issue's description, and the observable one: a filter told its own
/// prediction is worthless lands on each fix, so its horizontal error *is* that fix's error
/// and nothing is averaged down.
///
/// Derivation. Over a settled run the position error obeys $e_k = (1-K)e_{k-1} + K n_k$ for
/// fix error $n_k$ and steady-state gain $K$, a one-pole filter whose output standard
/// deviation is $\sqrt{K/(2-K)}$ times its input's. The ceiling of one half therefore says
/// exactly $K \le 0.4$: the update must give the prediction at least 60% of the weight.
/// That is the same "is it filtering at all" statement as
/// [`MAX_PRIOR_SHARE_OF_INNOVATION_COVARIANCE`] made about the output rather than the
/// covariance, and deliberately weaker than the $K = 0.043$ this tuning actually implies
/// ($\sqrt{K/(2-K)} = 0.15$, against 0.072 measured -- the fix noise here is a bounded
/// repeating pattern rather than white, and a one-pole filter rejects a period-7 sequence
/// of near-zero mean better than it rejects white noise). The two bounds are independent
/// observations of one defect, not a tightened pair. Against the pre-#308 diagonal
/// $K = 0.9997$ and the ratio is 1.000: the solution *is* the fix.
const MAX_FIX_NOISE_PASSED_THROUGH: f64 = 0.5;

/// Root-mean-square of a sample of errors, or `None` if the sample is empty.
fn root_mean_square(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum_of_squares: f64 = values.iter().map(|v| v * v).sum();
    Some((sum_of_squares / values.len() as f64).sqrt())
}

/// What a run of [`measure_filtering`] observed, once the transient has been dropped.
struct FilteringOutcome {
    /// Largest $HPH^T / R$ seen on a settled fix, unit-free.
    worst_prior_share: f64,
    /// Root-mean-square horizontal error of the solution against truth, meters.
    solution_rms_m: f64,
    /// Root-mean-square horizontal error of the fixes the run consumed, meters.
    fix_rms_m: f64,
}

impl FilteringOutcome {
    /// Fraction of the fix noise that survived into the solution.
    fn passed_through(&self) -> f64 {
        self.solution_rms_m / self.fix_rms_m
    }
}

/// Drive one filter through the gating scenario and measure whether it filtered.
///
/// Ungated on purpose: gating is a separate question, and a gate that rejected the honest
/// fixes would hide the behaviour being measured behind an empty sample.
fn measure_filtering(
    name: &str,
    filter: &mut dyn NavigationFilter,
    scenario: &GatingScenario,
) -> FilteringOutcome {
    // Latitude's entry in the fix's own noise covariance, rad^2 -- built the same way
    // `GPSPositionAndVelocityMeasurement::get_noise` builds it, so the ratio below is
    // against the R the update actually used.
    let measurement_variance_rad2 = meters_to_radians(GPS_HORIZONTAL_NOISE_M).powi(2);
    filter.set_innovation_gate(None);

    let mut prior_shares = Vec::new();
    let mut solution_errors_m = Vec::new();
    let mut fix_errors_m = Vec::new();
    let mut fixes_seen = 0_usize;

    for (index, sample) in scenario.samples.iter().enumerate() {
        filter.predict(sample, sample.dt).unwrap();
        if index % GPS_DECIMATION != 0 {
            continue;
        }
        // Read the prior *before* the update consumes it: this is the $HPH^T$ the
        // innovation covariance is formed from, and the Jacobian of this measurement is
        // the identity on the position block.
        let prior_variance_rad2 = filter.get_certainty()[(0, 0)];
        let outcome = filter.update(&scenario.gps[index]).unwrap();
        assert!(
            outcome.accepted,
            "{name} rejected a fix with no gate installed"
        );

        fixes_seen += 1;
        if fixes_seen <= PROCESS_NOISE_SETTLING_FIXES {
            continue;
        }
        prior_shares.push(prior_variance_rad2 / measurement_variance_rad2);

        // `truth[index + 1]` is the state the fix was recorded against: the scenario
        // samples truth *after* propagating `samples[index]`.
        let truth = &scenario.truth[index + 1];
        let fix = &scenario.gps[index];
        solution_errors_m.push(horizontal_error_m(&filter.get_estimate(), truth));
        fix_errors_m.push(haversine_distance(
            fix.latitude.to_radians(),
            fix.longitude.to_radians(),
            truth.latitude,
            truth.longitude,
        ));
    }

    let outcome = FilteringOutcome {
        worst_prior_share: prior_shares
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        solution_rms_m: root_mean_square(&solution_errors_m).unwrap(),
        fix_rms_m: root_mean_square(&fix_errors_m).unwrap(),
    };
    println!(
        "{name}: worst HPH'/R {:.4}, solution {:.3} m rms against {:.3} m rms of fix noise \
         ({:.3} passed through)",
        outcome.worst_prior_share,
        outcome.solution_rms_m,
        outcome.fix_rms_m,
        outcome.passed_through()
    );
    outcome
}

/// Assert both halves of "this filter filtered" on a measured run.
fn assert_filtered(name: &str, outcome: &FilteringOutcome) {
    assert!(
        outcome.worst_prior_share <= MAX_PRIOR_SHARE_OF_INNOVATION_COVARIANCE,
        "{name}: the settled prior contributes {:.3e} times the measurement's own variance \
         to S = HPH' + R, so the update discards the prediction and copies the fix -- the \
         #308 failure mode",
        outcome.worst_prior_share
    );
    assert!(
        outcome.passed_through() <= MAX_FIX_NOISE_PASSED_THROUGH,
        "{name}: {:.3} of the fix noise reached the solution ({:.3} m rms against {:.3} m \
         rms of fix error); a filtering solution attenuates it, a solution that lands on \
         each fix does not",
        outcome.passed_through(),
        outcome.solution_rms_m,
        outcome.fix_rms_m
    );
}

#[test]
fn the_shipped_default_process_noise_lets_the_eskf_filter() {
    // The regression test #308 did not have. Every other assertion in this workspace that
    // touches the default diagonal reads its entries back, or builds its own copy -- which
    // is why a horizontal process noise of 6.4 km per step and one of 0.1 m per step were
    // indistinguishable to the entire suite. This one never mentions a number from the
    // array. It drives a filter the crate ships, with the diagonal the crate ships, on
    // honest fixes, and asserts the two things an over-inflated Q destroys.
    //
    // Deliberately `strapdown::sim::DEFAULT_PROCESS_NOISE` and not this file's own
    // `process_noise()`: the local copy was already built through `meters_to_radians`, so
    // using it would test the fixture instead of the library.
    //
    // The ESKF, because it is the filter `sim::initialize_eskf` and `engine::InsEngine`
    // build by default, and because it is the one of the three whose reported uncertainty
    // is currently believable -- see the `#[ignore]`d companion below for the other two.
    let scenario = build_gating_scenario();
    let shipped = DMatrix::from_diagonal(&DVector::from_row_slice(&DEFAULT_PROCESS_NOISE));
    let (name, mut filter) = filters_with_process_noise(&scenario.initial, &shipped)
        .into_iter()
        .find(|(name, _)| *name == "ESKF")
        .expect("the filter list no longer contains an ESKF");
    let outcome = measure_filtering(name, filter.as_mut(), &scenario);
    assert_filtered(name, &outcome);
}

#[test]
#[ignore = "EKF and UKF report horizontal uncertainties of ~493 m and ~201 m on a run \
            whose actual error is metres -- pre-existing, #303 (closed as completed, \
            still reproducible)"]
fn the_shipped_default_process_noise_lets_every_filter_filter() {
    // The same two assertions across all three filters, which is what #308 is really a
    // statement about -- Q is shared, so the claim "the filter can filter" should not be
    // filter-specific.
    //
    // It is, for reasons that are not #308's. Measured here with the fixed diagonal:
    //
    //     filter | worst HPH'/R | implied sigma | fix noise passed through
    //     -------|--------------|---------------|------------------------
    //     ESKF   | 0.063        | 1.3 m         | 0.072
    //     UKF    | 1622         | 201 m         | 1.000
    //     EKF    | 9729         | 493 m         | 1.000
    //
    // Those two numbers are the same ones `a_note_on_filter_consistency` above records
    // from the other direction ("the EKF reports a position sigma of roughly 500 m after
    // converging to metres"), reached here without any seed error: #303's covariance
    // divergence, not an over-inflated Q. Q cannot be the cause -- the ESKF consumes the
    // identical diagonal on the identical stream and settles at 0.063.
    //
    // Left `#[ignore]`d and asserting the healthy behaviour rather than relaxed to a bound
    // 9729 would clear, which would import #303's numbers into #308's test and leave
    // nothing watching either (#288). It turns green when #303 does.
    let scenario = build_gating_scenario();
    let shipped = DMatrix::from_diagonal(&DVector::from_row_slice(&DEFAULT_PROCESS_NOISE));
    for (name, mut filter) in filters_with_process_noise(&scenario.initial, &shipped) {
        let outcome = measure_filtering(name, filter.as_mut(), &scenario);
        assert_filtered(name, &outcome);
    }
}

// ============================================================== #261: ZUPT and ZARU

/// Duration of the stationary scenario, seconds.
const STOP_DURATION_S: usize = 600;
/// Inertial sample rate for the stationary scenario, Hz.
const STOP_SAMPLE_RATE_HZ: usize = 10;
/// Aiding interval in inertial samples -- one ZUPT/ZARU per second at 10 Hz.
const AIDING_DECIMATION: usize = 10;
/// Accelerometer bias the filter is not told about, m/s^2 per axis.
///
/// Consumer-MEMS scale. Left uncorrected it integrates to ~3.6 km of position error over
/// the 600 s stop, which is the drift ZUPT exists to arrest.
const ACCEL_BIAS_MPS2: f64 = 0.02;
/// Gyroscope bias the filter is not told about, rad/s per axis.
///
/// ~0.06 deg/s. Left uncorrected it integrates to ~0.6 rad of attitude error over the
/// stop, which is the drift ZARU exists to arrest.
const GYRO_BIAS_RPS: f64 = 1.0e-3;

/// A platform that is stationary for the whole run, with a biased IMU.
struct StopScenario {
    /// Inertial increments as the (biased) sensor reports them.
    samples: Vec<ImuSample>,
    /// What the sensor reports, in rate form, for the stationary detector and ZARU.
    rates: Vec<IMUData>,
    /// The single truth state, held for the whole run.
    truth: StrapdownState,
}

/// Build the stationary scenario.
///
/// Truth is a platform at rest: the ideal IMU output is the level stream from [`level_imu`],
/// and the sensor adds a constant bias on every axis. Nothing else is corrupted, so every
/// metre of drift the filter accumulates is attributable to those two biases.
fn build_stop_scenario() -> StopScenario {
    let truth = StrapdownState {
        latitude: LATITUDE_DEG.to_radians(),
        longitude: LONGITUDE_DEG.to_radians(),
        altitude: ALTITUDE_M,
        velocity_north: 0.0,
        velocity_east: 0.0,
        velocity_vertical: 0.0,
        attitude: Rotation3::identity(),
        is_enu: false,
    };

    let dt = 1.0 / STOP_SAMPLE_RATE_HZ as f64;
    let num_samples = STOP_DURATION_S * STOP_SAMPLE_RATE_HZ;
    let ideal = level_imu(&truth);

    let mut samples = Vec::with_capacity(num_samples);
    let mut rates = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let measured = IMUData {
            accel: ideal.accel + Vector3::repeat(ACCEL_BIAS_MPS2),
            gyro: ideal.gyro + Vector3::repeat(GYRO_BIAS_RPS),
        };
        samples.push(ImuSample::from_rates(&measured, dt));
        rates.push(measured);
    }

    StopScenario {
        samples,
        rates,
        truth,
    }
}

/// What a stationary run produced.
struct StopOutcome {
    horizontal_error_m: f64,
    attitude_error_rad: f64,
    /// Inertial samples at which the detector considered the platform stationary.
    stationary_samples: usize,
    /// Aiding updates actually applied.
    updates_applied: usize,
}

/// Drive an ESKF through a prolonged stop, optionally applying ZUPT and ZARU.
///
/// The detector runs on every sample regardless, so `stationary_samples` is comparable
/// between the aided and unaided runs -- it is a property of the stream, not of the aiding.
fn run_stop(apply_zupt: bool, apply_zaru: bool) -> StopOutcome {
    let scenario = build_stop_scenario();
    let mut filter = ErrorStateKalmanFilter::new(
        &initial_state(&scenario.truth),
        &[0.0; 6],
        initial_covariance().to_vec(),
        process_noise_matrix(),
    );
    // A window sized for this scenario's 10 Hz stream: 2 s of history, 2 s of dwell.
    let mut detector = StationaryDetector::new(StationaryConfig {
        window: 20,
        min_stationary_samples: 20,
        ..StationaryConfig::default()
    });

    let mut stationary_samples = 0;
    let mut updates_applied = 0;
    for (index, sample) in scenario.samples.iter().enumerate() {
        filter.predict(sample, sample.dt).unwrap();

        let is_stationary = detector.push(&scenario.rates[index]);
        if is_stationary {
            stationary_samples += 1;
        }
        if !is_stationary || index % AIDING_DECIMATION != 0 {
            continue;
        }
        if apply_zupt {
            filter.update(&ZuptMeasurement::default()).unwrap();
            updates_applied += 1;
        }
        if apply_zaru {
            let gyro = scenario.rates[index].gyro;
            filter
                .update(&ZaruMeasurement::new(gyro, 1.0e-3).unwrap())
                .unwrap();
            updates_applied += 1;
        }
    }

    let estimate = filter.get_estimate();
    StopOutcome {
        horizontal_error_m: horizontal_error_m(&estimate, &scenario.truth),
        attitude_error_rad: attitude_error_rad(&estimate, &scenario.truth),
        stationary_samples,
        updates_applied,
    }
}

#[test]
fn the_detector_recognises_a_prolonged_stop() {
    // Precondition for everything below: if the detector never fires, the aided runs are
    // just the unaided run and the comparisons are vacuous.
    let outcome = run_stop(false, false);
    let total = STOP_DURATION_S * STOP_SAMPLE_RATE_HZ;
    assert!(
        outcome.stationary_samples > total * 9 / 10,
        "detector fired on only {} of {total} stationary samples",
        outcome.stationary_samples
    );
}

#[test]
fn zupt_and_zaru_arrest_drift_over_a_prolonged_stop() {
    let unaided = run_stop(false, false);
    let aided = run_stop(true, true);

    println!(
        "unaided: {:.1} m, {:.4} rad | aided: {:.3} m, {:.6} rad ({} updates)",
        unaided.horizontal_error_m,
        unaided.attitude_error_rad,
        aided.horizontal_error_m,
        aided.attitude_error_rad,
        aided.updates_applied
    );
    assert!(aided.updates_applied > 0, "no aiding was applied");
    assert!(
        aided.horizontal_error_m < unaided.horizontal_error_m,
        "aiding did not reduce position drift"
    );
    assert!(
        aided.attitude_error_rad < unaided.attitude_error_rad,
        "aiding did not reduce attitude drift"
    );
}

#[test]
fn zupt_alone_arrests_position_drift() {
    let unaided = run_stop(false, false);
    let zupt_only = run_stop(true, false);
    println!(
        "zupt only: {:.3} m, {:.6} rad (unaided {:.1} m)",
        zupt_only.horizontal_error_m, zupt_only.attitude_error_rad, unaided.horizontal_error_m
    );
    assert!(zupt_only.horizontal_error_m < unaided.horizontal_error_m);
}

#[test]
fn zaru_alone_arrests_attitude_drift() {
    let unaided = run_stop(false, false);
    let zaru_only = run_stop(false, true);
    println!(
        "zaru only: {:.3} m, {:.6} rad (unaided {:.4} rad)",
        zaru_only.horizontal_error_m, zaru_only.attitude_error_rad, unaided.attitude_error_rad
    );
    assert!(zaru_only.attitude_error_rad < unaided.attitude_error_rad);
}
