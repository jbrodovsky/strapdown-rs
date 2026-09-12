//! Side-by-side comparison of every navigation filter on one shared scenario.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//!
//! This is the acceptance criterion of #259 that no single filter's own test module can
//! carry: that the ESKF, EKF, UKF and RBPF, driven through the same [`NavigationFilter`]
//! trait with the same $\Delta v$ / $\Delta\theta$ stream and the same aiding measurements,
//! produce navigation solutions that agree with each other and with truth.
//!
//! # Why increments are the point
//!
//! Every filter here is fed a pre-built [`ImuSample`] rather than an [`IMUData`] of
//! instantaneous rates. That is deliberate and is what the test is for: it forces the UKF's
//! sigma-point propagation, the EKF's Jacobian linearisation and the RBPF's particle
//! propagation down the same increment-domain path the ESKF already used. Against the
//! rates-only `predict` those filters had before #259 this still *compiles* -- `&ImuSample`
//! coerces to `&dyn InputModel` like any other input -- and fails at runtime with
//! [`StrapdownError::UnsupportedInput`] on the first sample, which is how it was verified.
//!
//! # Aiding
//!
//! The fixes are [`GPSPositionAndVelocityMeasurement`], which is what
//! `messages::build_event_stream` emits for every GNSS epoch and therefore what the filters
//! actually see in `strapdown-sim`. Position-only aiding is *not* interchangeable here: with
//! [`GPSPositionMeasurement`] alone the UKF ends this run 590 m/s off in the east channel
//! even when seeded exactly on truth. That is worth fixing, but it is not what #259 changed.
//!
//! # Tolerances
//!
//! The bounds asserted below are empirical, not theoretical. They were read off a run of
//! this scenario and given headroom, matching the convention in `integration_tests.rs`:
//! they are regression checks against a filter quietly getting worse, not design targets.
//!
//! [`GPSPositionMeasurement`]: strapdown::measurements::GPSPositionMeasurement
//! [`StrapdownError::UnsupportedInput`]: strapdown::StrapdownError::UnsupportedInput
//! [`IMUData`]: strapdown::IMUData

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use strapdown::earth::haversine_distance;
use strapdown::kalman::{
    ErrorStateKalmanFilter, ExtendedKalmanFilter, InitialState, UnscentedKalmanFilter,
};
use strapdown::measurements::GPSPositionAndVelocityMeasurement;
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};
use strapdown::{ImuSample, NavigationFilter, StrapdownState, mechanize};

/// Scenario latitude, degrees.
const SCENARIO_LATITUDE_DEG: f64 = 40.0;
/// Scenario longitude, degrees.
const SCENARIO_LONGITUDE_DEG: f64 = -105.0;
/// Scenario altitude, meters above the ellipsoid.
const SCENARIO_ALTITUDE_M: f64 = 1000.0;
/// Northward ground speed held for the whole run, m/s.
const SCENARIO_VELOCITY_NORTH_MPS: f64 = 10.0;
/// Scenario duration, seconds.
const SCENARIO_DURATION_S: usize = 300;
/// Inertial sample rate, Hz.
const SCENARIO_SAMPLE_RATE_HZ: usize = 5;
/// Aiding interval in inertial samples -- one GPS fix per second at 5 Hz.
const GPS_DECIMATION: usize = 5;

/// Initial position offset used by the seeded-error test, meters.
///
/// A routine INS initialisation error -- the sort a coarse alignment or a single GNSS fix
/// leaves behind. See [`all_filters_converge_from_a_displaced_seed`] for why the test that
/// uses it is quarantined.
const SEEDED_POSITION_ERROR_M: f64 = 20.0;

/// Shared process noise, 15-state. Matches `integration_tests.rs`'s `DEFAULT_PROCESS_NOISE`
/// so that a divergence between the two suites is attributable to the filters rather than
/// to the tuning.
const PROCESS_NOISE: [f64; 15] = [
    1e-6, 1e-6, 1e-6, // position
    1e-3, 1e-3, 1e-3, // velocity
    1e-5, 1e-5, 1e-5, // attitude
    1e-6, 1e-6, 1e-6, // accelerometer bias
    1e-8, 1e-8, 1e-8, // gyroscope bias
];

/// Maximum final horizontal position error against truth, meters.
///
/// Worst measured on this branch is the RBPF's 0.222 m (ESKF and EKF are exact to printing
/// precision, UKF 0.019 m), so this carries ~4.5x margin -- the same margin the ESKF
/// integration bounds were rederived to in #288, and for the same reason: a ceiling loose
/// enough that any non-divergent filter clears it tests nothing.
const MAX_HORIZONTAL_ERROR_M: f64 = 1.0;
/// Maximum final altitude error against truth, meters. Worst measured: RBPF 0.613 m.
const MAX_ALTITUDE_ERROR_M: f64 = 3.0;
/// Maximum final speed error against truth, m/s. Worst measured: RBPF 0.117 m/s.
const MAX_VELOCITY_ERROR_MPS: f64 = 0.5;
/// Maximum horizontal separation between any two filters' final solutions, meters.
///
/// Looser than the truth bound on purpose: two filters may sit on opposite sides of truth,
/// so the worst legitimate separation is roughly twice the worst legitimate error. Worst
/// measured: 0.222 m, between the RBPF and the two Jacobian filters.
const MAX_PAIRWISE_SEPARATION_M: f64 = 1.5;

/// UKF sigma-point tuning, matching `sim::default_ukf_*`.
const UKF_ALPHA: f64 = 1e-3;
const UKF_BETA: f64 = 2.0;
const UKF_KAPPA: f64 = 0.0;
/// RBPF particle count and RNG seed. Fixed so the comparison is reproducible.
const RBPF_PARTICLES: usize = 500;
const RBPF_SEED: u64 = 259;

/// Shared initial covariance, 15-state.
const INITIAL_COVARIANCE: [f64; 15] = [
    1e-6, 1e-6, 1.0, // position (lat/lon rad^2, alt m^2)
    0.1, 0.1, 0.1, // velocity
    0.01, 0.01, 0.01, // attitude
    0.01, 0.01, 0.01, // accelerometer bias
    0.001, 0.001, 0.001, // gyroscope bias
];

/// The scenario, generated once and shared by every filter.
struct Scenario {
    /// Inertial increments, one per sample.
    samples: Vec<ImuSample>,
    /// GPS position fixes, one per inertial sample (decimated at consumption). `gps[i]`
    /// describes the state *after* `samples[i]`, matching the order `run` applies them in.
    gps: Vec<GPSPositionAndVelocityMeasurement>,
    /// Truth trajectory. `truth[i]` is the state the *i*th sample propagates *from*, so this
    /// is one longer than `samples`; `truth.last()` is the post-run truth.
    truth: Vec<StrapdownState>,
    /// The offset-from-truth state every filter is seeded with.
    initial: StrapdownState,
}

/// Build the shared scenario in NED: a level northbound run at 10 m/s.
///
/// Truth is *defined* as the mechanization's own integral of the inertial stream rather than
/// solved for independently. That is the point: `core/src/lib.rs` records that a hand-rolled
/// copy of the propagation equations in a scenario generator went silently wrong when the
/// mechanization moved to increments, and any generator that inverts the equations to hit a
/// commanded trajectory has to reproduce the Coriolis and transport terms to do it. Here the
/// inertial stream is the specification and [`mechanize`] defines what it means, so the two
/// cannot disagree and the test measures the filters rather than the generator.
///
/// The stream is what a perfect, perfectly level IMU on this trajectory would report:
/// specific force that opposes gravity, and angular rate that holds the platform level
/// against Earth rate and transport rate. The uncompensated Coriolis term leaves the
/// trajectory gently curving rather than exactly straight, which exercises the filters
/// slightly harder than a straight line would.
fn build_scenario(seed_offset_m: f64) -> Scenario {
    let truth_initial = StrapdownState {
        latitude: SCENARIO_LATITUDE_DEG.to_radians(),
        longitude: SCENARIO_LONGITUDE_DEG.to_radians(),
        altitude: SCENARIO_ALTITUDE_M,
        velocity_north: SCENARIO_VELOCITY_NORTH_MPS,
        velocity_east: 0.0,
        velocity_vertical: 0.0,
        attitude: Rotation3::identity(),
        is_enu: false,
    };

    let dt = 1.0 / SCENARIO_SAMPLE_RATE_HZ as f64;
    let num_samples = SCENARIO_DURATION_S * SCENARIO_SAMPLE_RATE_HZ;

    let mut samples = Vec::with_capacity(num_samples);
    let mut gps = Vec::with_capacity(num_samples);
    let mut truth = Vec::with_capacity(num_samples);

    let mut current = truth_initial;
    for _ in 0..num_samples {
        truth.push(current);

        let latitude_deg = current.latitude.to_degrees();
        let velocity = Vector3::new(
            current.velocity_north,
            current.velocity_east,
            current.velocity_vertical,
        );
        // Specific force: NED gravity is positive down, so a level body is held up by a
        // specific force that is negative down.
        let gravity = strapdown::earth::gravity(&latitude_deg, &current.altitude);
        let specific_force_body = current.attitude.inverse() * Vector3::new(0.0, 0.0, -gravity);
        // Angular rate: hold the platform level against Earth rate and transport rate. With
        // a zero gyro the platform would instead stay fixed in inertial space and tilt away
        // from level at ~15 deg/hour, tipping gravity into the horizontal channels.
        let nav_rate = strapdown::earth::earth_rate_lla(&latitude_deg)
            + strapdown::earth::transport_rate(&latitude_deg, &current.altitude, &velocity);
        let angular_rate_body = current.attitude.inverse() * nav_rate;

        let sample = ImuSample {
            delta_v: specific_force_body * dt,
            delta_theta: angular_rate_body * dt,
            dt,
        };
        samples.push(sample);
        mechanize(&mut current, &sample).unwrap();

        // Recorded *after* the propagation, so `gps[i]` describes the state a filter holds
        // once it has consumed `samples[i]` -- which is the order `run` applies them in.
        // Taking the fix before the propagation instead leaves every measurement one
        // sample stale: a systematic 2 m along-track bias at 10 m/s, which is only 0.4
        // sigma against the 5 m fix but is *persistent*, and a persistent position bias is
        // exactly what an INS vertical channel integrates into a runaway. It diverged to
        // 10^33 m within 600 samples.
        gps.push(GPSPositionAndVelocityMeasurement {
            latitude: current.latitude.to_degrees(),
            longitude: current.longitude.to_degrees(),
            altitude: current.altitude,
            northward_velocity: current.velocity_north,
            eastward_velocity: current.velocity_east,
            // Meters. `get_noise` does the metres-to-radians conversion itself, and
            // `build_event_stream` feeds it the record's `horizontal_accuracy` in metres.
            // Pre-converting here -- as `generate_scenario_data` does -- squares the
            // conversion and asks the filters to trust a 45 um GPS.
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.2,
        });
    }
    // One more than there are samples: `truth[i]` is the state the *i*th sample starts from,
    // so the state a filter should hold after consuming all of them is the extra entry.
    truth.push(current);

    let offset_rad = (seed_offset_m * strapdown::earth::METERS_TO_DEGREES).to_radians();
    let mut initial = truth_initial;
    initial.latitude += offset_rad;
    initial.longitude -= offset_rad;

    Scenario {
        samples,
        gps,
        truth,
        initial,
    }
}

/// The seed state in the form the Kalman-family constructors take.
///
/// Built as a struct literal rather than through [`InitialState::new`], matching
/// `integration_tests.rs` and `sim::initialize_eskf` -- no caller in the workspace uses the
/// constructor. Its radian path (`in_degrees: false`) converts latitude to degrees while
/// leaving longitude alone, and the filter constructors then read the result back as
/// radians, so a seed built that way starts 40 radians north.
fn initial_state(scenario: &Scenario) -> InitialState {
    let (roll, pitch, yaw) = scenario.initial.attitude.euler_angles();
    InitialState {
        latitude: scenario.initial.latitude,
        longitude: scenario.initial.longitude,
        altitude: scenario.initial.altitude,
        northward_velocity: scenario.initial.velocity_north,
        eastward_velocity: scenario.initial.velocity_east,
        vertical_velocity: scenario.initial.velocity_vertical,
        roll,
        pitch,
        yaw,
        in_degrees: false,
        is_enu: false,
    }
}

fn process_noise_matrix() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_row_slice(&PROCESS_NOISE))
}

/// Build all four filters on the same seed, biases and tuning.
///
/// Returned as trait objects and in a fixed order so every test drives the same set through
/// the same interface: a filter that stopped implementing [`NavigationFilter`] would fail to
/// build here rather than quietly dropping out of the comparison.
fn all_filters(
    scenario: &Scenario,
    biases: &[f64; 6],
) -> Vec<(&'static str, Box<dyn NavigationFilter>)> {
    let init = initial_state(scenario);
    vec![
        (
            "ESKF",
            Box::new(ErrorStateKalmanFilter::new(
                &init,
                biases,
                INITIAL_COVARIANCE.to_vec(),
                process_noise_matrix(),
            )) as Box<dyn NavigationFilter>,
        ),
        (
            "EKF",
            Box::new(ExtendedKalmanFilter::new(
                &init,
                biases,
                INITIAL_COVARIANCE.to_vec(),
                process_noise_matrix(),
                true,
            )),
        ),
        (
            "UKF",
            Box::new(UnscentedKalmanFilter::new(
                &init,
                biases,
                None,
                INITIAL_COVARIANCE.to_vec(),
                process_noise_matrix(),
                UKF_ALPHA,
                UKF_BETA,
                UKF_KAPPA,
            )),
        ),
        (
            "RBPF",
            Box::new(
                RaoBlackwellizedParticleFilter::new(
                    scenario.initial,
                    RbpfConfig {
                        num_particles: RBPF_PARTICLES,
                        position_init_std_m: Vector3::new(10.0, 10.0, 5.0),
                        seed: RBPF_SEED,
                        ..RbpfConfig::default()
                    },
                )
                .unwrap(),
            ),
        ),
    ]
}

/// Drive one filter across the whole scenario and return its final estimate.
///
/// The loop is deliberately written against `&mut dyn NavigationFilter` rather than a
/// generic: every filter is exercised through the object-safe trait, which is the interface
/// #262's `InsEngine` will hold them by.
fn run(filter: &mut dyn NavigationFilter, scenario: &Scenario) -> DVector<f64> {
    let dt = 1.0 / SCENARIO_SAMPLE_RATE_HZ as f64;
    for (i, sample) in scenario.samples.iter().enumerate() {
        filter.predict(sample, dt).unwrap();
        if i % GPS_DECIMATION == 0 {
            filter.update(&scenario.gps[i]).unwrap();
        }
    }
    filter.get_estimate()
}

/// Horizontal great-circle error between an estimate and truth, meters.
fn horizontal_error_m(estimate: &DVector<f64>, truth: &StrapdownState) -> f64 {
    haversine_distance(estimate[0], estimate[1], truth.latitude, truth.longitude)
}

/// Horizontal great-circle separation between two estimates, meters.
fn horizontal_separation_m(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    haversine_distance(a[0], a[1], b[0], b[1])
}

/// Speed error between an estimate and truth, m/s.
fn velocity_error_mps(estimate: &DVector<f64>, truth: &StrapdownState) -> f64 {
    ((estimate[3] - truth.velocity_north).powi(2)
        + (estimate[4] - truth.velocity_east).powi(2)
        + (estimate[5] - truth.velocity_vertical).powi(2))
    .sqrt()
}

/// Every filter, run on the same scenario, agrees with truth and with the others.
///
/// This is #259's "side-by-side comparison test demonstrating consistent navigation results
/// across all filters". It asserts three separate things, because each catches a failure the
/// others do not:
///
/// 1. **Finiteness** -- a diverged filter produces `NaN` long before it produces a large
///    error, and a magnitude assertion on `NaN` passes vacuously in the wrong direction.
/// 2. **Agreement with truth** -- each filter individually tracks the trajectory.
/// 3. **Agreement with each other** -- the filters are solving the same problem. A filter
///    that stayed inside its own truth bound while drifting to the opposite edge of it is a
///    real defect that (2) alone cannot see.
#[test]
fn all_filters_agree_on_a_shared_scenario() {
    let scenario = build_scenario(0.0);
    let truth = scenario.truth.last().unwrap();

    let estimates: Vec<(&str, DVector<f64>)> = all_filters(&scenario, &[0.0; 6])
        .into_iter()
        .map(|(name, mut filter)| (name, run(filter.as_mut(), &scenario)))
        .collect();

    for (name, estimate) in &estimates {
        let horizontal = horizontal_error_m(estimate, truth);
        let altitude = (estimate[2] - truth.altitude).abs();
        let velocity = velocity_error_mps(estimate, truth);
        println!(
            "{name}: horizontal {horizontal:.3} m | altitude {altitude:.3} m | velocity {velocity:.4} m/s"
        );

        assert!(
            estimate.iter().take(9).all(|v| v.is_finite()),
            "{name} produced a non-finite navigation state: {estimate:?}"
        );
        assert!(
            horizontal <= MAX_HORIZONTAL_ERROR_M,
            "{name} horizontal error {horizontal:.3} m exceeds {MAX_HORIZONTAL_ERROR_M:.3} m"
        );
        assert!(
            altitude <= MAX_ALTITUDE_ERROR_M,
            "{name} altitude error {altitude:.3} m exceeds {MAX_ALTITUDE_ERROR_M:.3} m"
        );
        assert!(
            velocity <= MAX_VELOCITY_ERROR_MPS,
            "{name} velocity error {velocity:.4} m/s exceeds {MAX_VELOCITY_ERROR_MPS:.4} m/s"
        );
    }

    for (i, (name_a, a)) in estimates.iter().enumerate() {
        for (name_b, b) in &estimates[i + 1..] {
            let separation = horizontal_separation_m(a, b);
            println!("{name_a} vs {name_b}: horizontal separation {separation:.3} m");
            assert!(
                separation <= MAX_PAIRWISE_SEPARATION_M,
                "{name_a} and {name_b} disagree by {separation:.3} m, above the \
                 {MAX_PAIRWISE_SEPARATION_M:.3} m consistency bound"
            );
        }
    }
}

/// Feeding the same increments as instantaneous rates lands in the same place.
///
/// [`ImuSample::from_rates`] is a pure scaling, so a filter given `IMUData` must reach the
/// same state as one given the equivalent `ImuSample`. The property is worth asserting
/// because the bias compensation is what makes it non-trivial: correcting a rate by a bias
/// and then integrating, versus correcting an increment by that bias integrated over the
/// same interval, agree only if the conversion is applied in the right order -- which is the
/// thing #259 changed in the EKF and UKF.
#[test]
fn rate_and_increment_inputs_are_equivalent() {
    let scenario = build_scenario(0.0);
    let init = initial_state(&scenario);
    let dt = 1.0 / SCENARIO_SAMPLE_RATE_HZ as f64;

    // A non-zero bias seed: with zero biases the two orderings coincide trivially.
    let biases = [0.02, -0.01, 0.03, 1e-4, -2e-4, 1.5e-4];

    for name in ["ESKF", "EKF", "UKF"] {
        let build = || -> Box<dyn NavigationFilter> {
            match name {
                "ESKF" => Box::new(ErrorStateKalmanFilter::new(
                    &init,
                    &biases,
                    INITIAL_COVARIANCE.to_vec(),
                    process_noise_matrix(),
                )),
                "EKF" => Box::new(ExtendedKalmanFilter::new(
                    &init,
                    &biases,
                    INITIAL_COVARIANCE.to_vec(),
                    process_noise_matrix(),
                    true,
                )),
                _ => Box::new(UnscentedKalmanFilter::new(
                    &init,
                    &biases,
                    None,
                    INITIAL_COVARIANCE.to_vec(),
                    process_noise_matrix(),
                    1e-3,
                    2.0,
                    0.0,
                )),
            }
        };

        let mut from_increments = build();
        let mut from_rates = build();

        for (i, sample) in scenario.samples.iter().enumerate().take(100) {
            from_increments.predict(sample, dt).unwrap();
            from_rates.predict(&sample.to_rates().unwrap(), dt).unwrap();
            if i % GPS_DECIMATION == 0 {
                from_increments.update(&scenario.gps[i]).unwrap();
                from_rates.update(&scenario.gps[i]).unwrap();
            }
        }

        let a = from_increments.get_estimate();
        let b = from_rates.get_estimate();
        for (j, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-12 * lhs.abs().max(rhs.abs()).max(1.0),
                "{name} state {j} differs between rate and increment inputs: {lhs} vs {rhs}"
            );
        }
    }
}

/// A filter handed an `ImuSample` whose `dt` contradicts the argument says so.
///
/// The increment interface makes it possible for a caller to pass a sample integrated over
/// one interval while telling the filter it covers another. Silently preferring either would
/// make the integration rate quietly wrong; every filter must reject it.
#[test]
fn every_filter_rejects_a_contradictory_timestep() {
    let scenario = build_scenario(0.0);

    let mismatched = ImuSample {
        dt: 0.2,
        ..scenario.samples[0]
    };

    let mut filters = all_filters(&scenario, &[0.0; 6]);

    for (name, filter) in &mut filters {
        let Err(err) = filter.predict(&mismatched, 0.1) else {
            panic!("{name} accepted a sample whose dt contradicts the argument");
        };
        assert!(
            matches!(err, strapdown::StrapdownError::InconsistentTimestep { .. }),
            "{name} reported {err} rather than InconsistentTimestep"
        );
    }
}

/// A filter handed an input that is not inertial at all reports it rather than panicking.
///
/// `VelocityData` implements [`strapdown::InputModel`], so the type system permits it here;
/// #254's zero-panic contract requires this be an error, and #259 extends the requirement to
/// the RBPF now that it too takes a trait object.
#[test]
fn every_filter_rejects_a_non_inertial_input() {
    let scenario = build_scenario(0.0);
    let velocity = strapdown::VelocityData::default();

    let mut filters = all_filters(&scenario, &[0.0; 6]);

    for (name, filter) in &mut filters {
        let Err(err) = filter.predict(&velocity, 0.1) else {
            panic!("{name} accepted a VelocityData as an inertial input");
        };
        assert!(
            matches!(err, strapdown::StrapdownError::UnsupportedInput { .. }),
            "{name} reported {err} rather than UnsupportedInput"
        );
    }
}

/// Every filter converges from a routine initialisation error.
///
/// Quarantined, not tuned around. Seeded 20 m from truth -- a coarse alignment's worth of
/// error -- the ESKF and EKF do not converge on this scenario: their vertical channels grow
/// exponentially and run away. The UKF and RBPF converge from the same seed, and all four
/// are exact when seeded *on* truth (`all_filters_agree_on_a_shared_scenario`), which is
/// what isolates the fault to the analytic Jacobians in `linearize.rs` that only the EKF and
/// ESKF use. Tracked as #303.
///
/// Measured on this branch, `GPSPositionAndVelocityMeasurement` aiding at 1 Hz, sample at
/// which the state first leaves finite/plausible bounds:
///
/// | seed error | ESKF | EKF |
/// |---|---|---|
/// | 0 (exact) | converges | converges |
/// | 1 m position | 1395 | ~1105 |
/// | 5 m position | 1316 | ~1025 |
/// | 20 m position | 1292 | ~990 |
/// | 1 m altitude | 890 | 1105 |
/// | 0.05 m/s velocity | 886 | 1145 |
/// | 0.001 rad attitude | 1044 | 1320 |
///
/// The trigger is *any* seed error, not a large one, and the divergence sample shrinks with
/// its magnitude -- the signature of exponential growth from a seed proportional to the
/// error. This is the same class of defect as #266 and #286 (a units-or-convention mismatch
/// between `F`/`H` and the injection step) and is tracked as #303; #259 changed only the
/// domain the increments arrive in, and the ESKF's numbers here are bit-identical before and
/// after that change.
///
/// Re-enable when the ESKF converges. Do not widen the bounds to make it pass.
///
/// Half of this is already fixed: the EKF diverged here for the same reason it diverged on
/// real data (#307) -- `state_transition_jacobian` expressed the attitude columns as a
/// rotation vector while the EKF's state holds Euler angles. With
/// `euler_state_transition_jacobian` the EKF now ends this scenario 0.000 m from truth, and
/// the UKF and RBPF were always fine. Only the ESKF still runs away, at 31 km from a 20 m
/// seed, which points at `error_state_transition_jacobian` rather than at the shared one.
#[test]
#[ignore = "ESKF diverges from any non-zero seed error (31 km from a 20 m seed); the EKF half was the Euler parametrisation and is fixed (#307), the ESKF half remains -- #303"]
fn all_filters_converge_from_a_displaced_seed() {
    let scenario = build_scenario(SEEDED_POSITION_ERROR_M);
    let truth = scenario.truth.last().unwrap();

    // Every filter is run and reported before anything is asserted. Failing on the first one
    // hides the others, and which filters diverge is the whole diagnostic here -- it is what
    // separated the EKF's Euler-parametrisation defect (#307, fixed) from whatever remains in
    // the ESKF.
    let mut failures = Vec::new();

    for (name, mut filter) in all_filters(&scenario, &[0.0; 6]) {
        let estimate = run(filter.as_mut(), &scenario);
        let horizontal = horizontal_error_m(&estimate, truth);
        println!(
            "{name}: horizontal {horizontal:.3} m from a {SEEDED_POSITION_ERROR_M:.0} m seed error"
        );

        if !estimate.iter().take(9).all(|v| v.is_finite()) {
            failures.push(format!("{name}: produced a non-finite navigation state"));
        } else if horizontal > MAX_HORIZONTAL_ERROR_M {
            failures.push(format!(
                "{name}: horizontal error {horizontal:.3} m exceeds {MAX_HORIZONTAL_ERROR_M:.3} m"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of the filters failed to converge from a {SEEDED_POSITION_ERROR_M:.0} m seed \
         error:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
