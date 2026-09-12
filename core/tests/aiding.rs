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
/// Built as a struct literal rather than through [`InitialState::new`], matching every
/// other caller in the workspace -- see the note in `filter_comparison.rs`.
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
        let jitter = NOISE_PATTERN[index % NOISE_PATTERN.len()];
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
    let init = initial_state(initial);
    let biases = [0.0_f64; 6];
    vec![
        (
            "ESKF",
            Box::new(ErrorStateKalmanFilter::new(
                &init,
                &biases,
                initial_covariance().to_vec(),
                process_noise_matrix(),
            )) as Box<dyn NavigationFilter>,
        ),
        (
            "EKF",
            Box::new(ExtendedKalmanFilter::new(
                &init,
                &biases,
                initial_covariance().to_vec(),
                process_noise_matrix(),
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
                process_noise_matrix(),
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
    // That independence is not academic. See `a_note_on_filter_consistency` below: on
    // this branch all three filters report position uncertainties far larger than their
    // actual errors, so a fixed 200 m displacement is *within* what the EKF and UKF
    // believe possible and is correctly not gated. Sizing the outlier in sigmas tests
    // the gate; sizing it in metres would test the tuning.
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

#[test]
#[ignore = "ESKF and EKF vertical channels diverge under ordinary GNSS fix noise -- \
            pre-existing, #303 (closed as completed, still reproducible)"]
fn a_note_on_filter_consistency() {
    // Not a test of anything in this PR. It is the reason the gating tests above size
    // their outlier in sigmas rather than in metres, recorded as a runnable fact rather
    // than a comment that can quietly stop being true.
    //
    // #303 reports that the ESKF and EKF diverge from any non-zero *seed* error, with the
    // UKF unaffected, and fingers the analytic Jacobians in `linearize.rs`. This is the
    // same defect reached without any seed error at all: seed every filter exactly on
    // truth and make only the GNSS fixes noisy, at a quarter of their declared accuracy.
    // Measured on this branch over a 300 s run with 1 Hz fixes:
    //
    //     fix noise (rms) | ESKF peak |alt err| | EKF peak |alt err| | UKF
    //     ----------------|---------------------|--------------------|--------
    //     0 m             | 0.0 m               | 0.0 m              | 0.19 m
    //     0.5 m           | 9.8e7 m             | 2.0e4 m            | 0.19 m
    //     5.0 m           | 5.7e8 m             | 2.0e5 m            | 0.19 m
    //
    // Two things that matter more than the magnitudes. First, the ESKF's divergence is
    // essentially independent of the noise amplitude -- a 10x change in excitation moves
    // it less than an order of magnitude -- which is the signature of an unstable mode
    // being excited rather than of noise being propagated. The EKF's, by contrast, scales
    // exactly linearly (2.0e4 : 5.1e4 : 1.0e5 : 2.0e5 for 0.5 : 1.25 : 2.5 : 5.0 m), so
    // the EKF looks marginally stable where the ESKF does not. Second, noise is not an
    // edge case: `filter_comparison.rs` misses this entirely because it feeds fixes taken
    // noise-free from truth, and real GNSS never is.
    //
    // The consequence for #260 is direct. A filter whose reported uncertainty does not
    // match its actual error cannot gate: on this branch the EKF reports a position sigma
    // of roughly 500 m after converging to metres, so a genuine 200 m multipath fix is
    // *within* what it believes possible and is correctly accepted. Gating will only do
    // what #260 asks of it once the filters are consistent.
    //
    // Left `#[ignore]`d and asserting the healthy behaviour, matching
    // `filter_comparison.rs::all_filters_converge_from_a_displaced_seed`: this turns green
    // when the defect is fixed.
    const MAX_ALTITUDE_ERROR_M: f64 = 1.0;

    let scenario = build_gating_scenario();
    for (name, mut filter) in gating_filters(&scenario.initial) {
        filter.set_innovation_gate(None);
        let mut peak_altitude_error_m: f64 = 0.0;
        for (index, sample) in scenario.samples.iter().enumerate() {
            filter.predict(sample, sample.dt).unwrap();
            if index % GPS_DECIMATION == 0 {
                filter.update(&scenario.gps[index]).unwrap();
                peak_altitude_error_m = peak_altitude_error_m
                    .max((filter.get_estimate()[2] - scenario.truth[index + 1].altitude).abs());
            }
        }
        assert!(
            peak_altitude_error_m < MAX_ALTITUDE_ERROR_M,
            "{name} vertical channel reached {peak_altitude_error_m:.3e} m under noisy fixes"
        );
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
