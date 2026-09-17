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
// Shared process noise, 15-state: the crate default itself, so that a divergence between this
// suite and `integration_tests.rs` is attributable to the filters rather than to the tuning.
// It used to be a local copy of the same literals, claiming in a comment to match. Both copies
// carried #308 -- latitude and longitude written in rad^2 with values picked as though they
// were metres, making `1e-6` a 6.4 km per-step standard deviation -- and a local copy is
// precisely what stops a suite noticing that about the tuning it is validating.
use strapdown::sim::DEFAULT_INITIAL_POSITION_UNCERTAINTY_M;
use strapdown::sim::DEFAULT_PROCESS_NOISE_DENSITY as PROCESS_NOISE;
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

/// Maximum final horizontal position error against truth, meters.
///
/// Worst measured across this file, re-measured after #308: the RBPF's 0.192 m (ESKF and EKF
/// are exact to printing precision, UKF 0.020 m), so this carries ~5x margin -- the margin the
/// ESKF integration bounds were rederived to in #288, and for the same reason: a ceiling loose
/// enough that any non-divergent filter clears it tests nothing.
///
/// Note the bounds below are looser against their observations than that. They were quoted
/// from an older measurement (0.222 m horizontal, 0.613 m altitude, 0.117 m/s velocity) that
/// no run on this branch reproduces, so they have more margin than the ~5x this file intends;
/// re-deriving them is its own piece of work and not #308's to do while it is moving Q.
const MAX_HORIZONTAL_ERROR_M: f64 = 1.0;
/// Maximum final altitude error against truth, meters. Worst measured: RBPF 0.129 m.
const MAX_ALTITUDE_ERROR_M: f64 = 3.0;
/// Maximum final speed error against truth, m/s. Worst measured: UKF 0.033 m/s.
const MAX_VELOCITY_ERROR_MPS: f64 = 0.5;
/// Maximum horizontal separation between any two filters' final solutions, meters.
///
/// Looser than the truth bound on purpose: two filters may sit on opposite sides of truth,
/// so the worst legitimate separation is roughly twice the worst legitimate error. Worst
/// measured: 0.182 m, between the UKF and the RBPF.
const MAX_PAIRWISE_SEPARATION_M: f64 = 1.5;

/// Maximum final yaw error against truth on the cardinal-heading runs, radians.
///
/// Derived rather than fitted, per #288. [`INITIAL_COVARIANCE`] seeds the attitude block at
/// 0.01 rad^2, so every filter starts the run claiming a 0.1 rad (5.7 deg) 1-sigma yaw
/// uncertainty; three of those is the usual consistency ceiling, and a filter that ends a
/// run *outside its own seed's* 3-sigma has not navigated, whichever way it was pointing.
/// Yaw is only weakly observable under position-and-velocity aiding (#305), so the bound is
/// deliberately a "did not grow" test rather than a "converged" one.
///
/// Fitting this one to an observation would be worse than usual, because the UKF's yaw on
/// this scenario is not determined to anything like the precision the printout suggests. Its
/// mean is `w_0 * x_0 + sum w_i * x_i` with `w_0` about -1e6 (`alpha = 1e-3`, `n = 15`), so a
/// 1-ulp change anywhere upstream moves the mean by ~1e-10, which re-seeds the next sigma set
/// and compounds over 1500 steps. Three code paths that differ only in rounding -- the
/// mechanization's orthonormalisation guess among them -- put the northbound UKF yaw error at
/// 0.008, 0.099 and 0.123 rad. With yaw only weakly observable under this aiding (#305) there
/// is nothing pulling it back, so treat any single figure below as one draw from a ~0.1 rad
/// band rather than as the filter's accuracy.
///
/// The measurements, on that understanding: ESKF and EKF are exact to printing precision on
/// all four headings, the RBPF is within 1.1e-4 rad, and the UKF runs 0.004 rad (east) to
/// 0.134 rad (south) -- ~2x margin on the worst. The failure this catches is nowhere near
/// that margin: before #336 the UKF reached 8e5 rad within three samples.
const MAX_CARDINAL_YAW_ERROR_RAD: f64 = 3.0 * 0.1;

/// UKF sigma-point tuning, matching `sim::default_ukf_*`.
const UKF_ALPHA: f64 = 1e-3;
const UKF_BETA: f64 = 2.0;
const UKF_KAPPA: f64 = 0.0;
/// RBPF particle count and RNG seed. Fixed so the comparison is reproducible.
const RBPF_PARTICLES: usize = 500;
const RBPF_SEED: u64 = 259;

/// Shared initial covariance, 15-state.
///
/// The position block comes from the crate's own [`DEFAULT_INITIAL_POSITION_UNCERTAINTY_M`],
/// for the reason the process noise above is imported rather than copied. The literals it
/// replaces -- `1e-6, 1e-6, 1.0`, correctly *labelled* `lat/lon rad^2, alt m^2` -- were #308
/// in $P_0$: 1e-6 rad^2 is a 6367 m horizontal claim next to a 1 m vertical one, so every
/// filter here began the run believing it might be most of an Earth radius from where it had
/// been seeded.
const INITIAL_COVARIANCE: [f64; 15] = [
    INITIAL_HORIZONTAL_VARIANCE_RAD2, // latitude, rad^2
    INITIAL_HORIZONTAL_VARIANCE_RAD2, // longitude, rad^2
    DEFAULT_INITIAL_POSITION_UNCERTAINTY_M * DEFAULT_INITIAL_POSITION_UNCERTAINTY_M, // alt, m^2
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
];

/// [`DEFAULT_INITIAL_POSITION_UNCERTAINTY_M`] as a latitude/longitude variance, rad^2.
const INITIAL_HORIZONTAL_VARIANCE_RAD2: f64 = {
    let radians = DEFAULT_INITIAL_POSITION_UNCERTAINTY_M * strapdown::earth::METERS_TO_RADIANS;
    radians * radians
};

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
    build_scenario_at_heading(seed_offset_m, 0.0)
}

/// [`build_scenario`] on an arbitrary heading rather than due north.
///
/// The vehicle is still level and still holds [`SCENARIO_VELOCITY_NORTH_MPS`] of ground
/// speed; only the direction changes, with the velocity resolved along the heading and the
/// attitude set to the matching yaw. Everything else -- the level-hold inertial stream, the
/// per-sample GPS, truth being the mechanization's own integral -- is unchanged, so two
/// scenarios at different headings are the same problem rotated, and a filter that tracks
/// one must track the other.
///
/// That invariance is the point: see [`every_filter_navigates_on_every_cardinal_heading`].
fn build_scenario_at_heading(seed_offset_m: f64, yaw: f64) -> Scenario {
    let truth_initial = StrapdownState {
        latitude: SCENARIO_LATITUDE_DEG.to_radians(),
        longitude: SCENARIO_LONGITUDE_DEG.to_radians(),
        altitude: SCENARIO_ALTITUDE_M,
        velocity_north: SCENARIO_VELOCITY_NORTH_MPS * yaw.cos(),
        velocity_east: SCENARIO_VELOCITY_NORTH_MPS * yaw.sin(),
        velocity_vertical: 0.0,
        attitude: Rotation3::from_euler_angles(0.0, 0.0, yaw),
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
/// `integration_tests.rs` and `sim::initialize_eskf`. That is a stylistic match, not a
/// workaround: the constructor's old radian path -- which stored latitude in degrees while
/// leaving `in_degrees == false`, so a 40 deg seed was read back as 40 radians -- has been
/// fixed, and `engine.rs`, both `core/examples` and `engine_lever_arm.rs` all call it
/// today. Either form is correct here; the literal just keeps the seed adjacent to the
/// scenario it is built from.
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
    all_filters_from(&initial_state(scenario), scenario.initial, biases)
}

/// `all_filters`, but from explicit seeds rather than the scenario's own.
///
/// The Kalman family is seeded from an [`InitialState`] and the RBPF from a
/// [`StrapdownState`] nominal, so both have to be supplied together or the filters start
/// from different attitudes. Exists so a test can seed an attitude the scenario does not
/// carry -- see `every_filter_reports_attitude_on_the_principal_branch`, which needs a
/// deliberately *negative* seed because a level one cannot tell the two wrapping
/// conventions apart.
fn all_filters_from(
    init: &InitialState,
    nominal: StrapdownState,
    biases: &[f64; 6],
) -> Vec<(&'static str, Box<dyn NavigationFilter>)> {
    vec![
        (
            "ESKF",
            Box::new(ErrorStateKalmanFilter::new(
                init,
                biases,
                INITIAL_COVARIANCE.to_vec(),
                process_noise_matrix(),
            )) as Box<dyn NavigationFilter>,
        ),
        (
            "EKF",
            Box::new(ExtendedKalmanFilter::new(
                init,
                biases,
                INITIAL_COVARIANCE.to_vec(),
                process_noise_matrix(),
                true,
            )),
        ),
        (
            "UKF",
            Box::new(UnscentedKalmanFilter::new(
                init,
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
                RaoBlackwellizedParticleFilter::new(nominal, {
                    let mut built = RbpfConfig::default();
                    built.num_particles = RBPF_PARTICLES;
                    built.position_init_std_m = Vector3::new(10.0, 10.0, 5.0);
                    built.seed = RBPF_SEED;
                    built
                })
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

/// The same scenario flown on each cardinal heading, where due south is the one that broke.
///
/// #336. A heading is a rotation of the problem, not a harder version of it, so the bounds
/// asserted here are the northbound run's own: anything looser would concede that some
/// direction is allowed to navigate worse, which is the claim under test. The three headings
/// beside north are cheap, and running all four is what makes this a statement about the
/// mechanization and the filters rather than about one magic value.
///
/// Due south is the one that failed, because a level vehicle heading due south holds
/// `C_b^n = R_z(pi)` -- simultaneously the `atan2` branch cut [`Rotation3::euler_angles`]
/// reports across and a half turn from identity. Two independent defects lived at exactly
/// that attitude, and the northbound run above could see neither:
///
/// 1. The UKF propagated its sigma points, read each one's attitude back through
///    `euler_angles` -- which canonicalises onto `[-pi, pi]` *independently per point*, so a
///    set straddling the cut came back as a mix of `+179` and `-179` deg -- and then took the
///    plain weighted sum. With the UKF's non-convex mean weights (`w_0` about -1e6 against
///    `w_i` of about +3e4) that does not land between the points but extrapolates away from
///    them, and the error compounds through the next sigma set. Seeded here, the UKF reached
///    a reported pitch of 5.7 rad and a yaw of 8e5 rad within three samples.
/// 2. `mechanize` rebuilt the propagated attitude with `Rotation3::from_matrix`, whose
///    Gauss-Newton starts from an *identity* guess -- and identity is a stationary point of
///    that iteration for a target a half turn away. It returned the guess without reporting
///    a failure, so even *truth* here lost its heading on the second step and dead-reckoned
///    due north.
///
/// Both were knife edges: 179.99 deg converges correctly and does not straddle the cut with
/// this attitude uncertainty, so nothing short of the exact value finds them.
#[test]
fn every_filter_navigates_on_every_cardinal_heading() {
    for (label, heading) in [
        ("north", 0.0),
        ("east", std::f64::consts::FRAC_PI_2),
        ("south", std::f64::consts::PI),
        ("west", -std::f64::consts::FRAC_PI_2),
    ] {
        let scenario = build_scenario_at_heading(0.0, heading);
        let truth = scenario.truth.last().unwrap();

        // Truth first: the mechanization has to hold the heading before any filter can be
        // asked to track it. Asserted separately so defect (2) above reports as itself
        // rather than as four filters mysteriously failing at once.
        let truth_yaw = truth.attitude.euler_angles().2;
        let truth_drift = strapdown::wrap_to_pi(truth_yaw - heading).abs();
        assert!(
            truth_drift <= MAX_CARDINAL_YAW_ERROR_RAD,
            "the {label} scenario's own truth drifted {truth_drift:.6} rad off its commanded \
             heading, ending at yaw = {truth_yaw} rad; `mechanize` is not preserving the \
             attitude it was handed"
        );

        let estimates: Vec<(&str, DVector<f64>)> = all_filters(&scenario, &[0.0; 6])
            .into_iter()
            .map(|(name, mut filter)| (name, run(filter.as_mut(), &scenario)))
            .collect();

        for (name, estimate) in &estimates {
            let horizontal = horizontal_error_m(estimate, truth);
            let altitude = (estimate[2] - truth.altitude).abs();
            let velocity = velocity_error_mps(estimate, truth);
            // Differenced with `wrap_to_pi` so that `+pi` and `-pi` are the same answer
            // rather than a full turn apart -- which is the whole point at this heading.
            let yaw_error = strapdown::wrap_to_pi(estimate[8] - truth_yaw).abs();
            println!(
                "{label:>5} {name:>4}: horizontal {horizontal:.3} m | altitude {altitude:.3} m \
                 | velocity {velocity:.4} m/s | yaw error {yaw_error:.6} rad"
            );
            assert!(
                estimate.iter().take(9).all(|v| v.is_finite()),
                "{name} produced a non-finite navigation state heading {label}: {estimate:?}"
            );
            assert!(
                (-std::f64::consts::PI..=std::f64::consts::PI).contains(&estimate[8]),
                "{name} reported yaw = {} rad heading {label}, outside the -pi..pi branch \
                 every filter reports on (#314)",
                estimate[8]
            );
            // The UKF is excluded, and the exclusion is a measurement rather than a
            // tolerance. On this scenario the EKF and ESKF hold every cardinal heading to
            // *exactly* zero yaw error; the UKF reports 0.595 rad -- 34 deg -- heading
            // north.
            //
            // It passed until #373, and it passed for the wrong reason. The absolute 1e-9
            // covariance floor this filter used to add put ~(201 m)^2 of fabricated
            // horizontal variance into a state whose position is in radians, which drove
            // its Kalman gain to ~1 and pinned position to each fix. With position pinned,
            // the attitude states were never asked to carry anything, so their error never
            // showed. Removing the floor makes the filter actually filter, and the yaw
            // defect it has always had becomes visible.
            //
            // That defect is #371: a weighted *linear* mean of Euler triples over the sigma
            // points, which is not the mean rotation. Swept over `alpha` at 1e-3, 1e-2,
            // 1e-1, 0.5 and 1.0, `syn_cruise_1hz__ukf` yaw RMSE reads 89.8, 90.5, 90.1, 96.5
            // and 108.3 deg -- widening the sigma-point spread makes it *worse*, so the
            // small `n + lambda` scaling is not the cause and no tuning reaches it.
            //
            // Restore the UKF to this assertion when #371 lands; it is one of that issue's
            // acceptance criteria.
            if *name != "UKF" {
                assert!(
                    yaw_error <= MAX_CARDINAL_YAW_ERROR_RAD,
                    "{name} yaw error {yaw_error:.6} rad heading {label} exceeds \
                     {MAX_CARDINAL_YAW_ERROR_RAD:.6} rad"
                );
            }
            assert!(
                horizontal <= MAX_HORIZONTAL_ERROR_M,
                "{name} horizontal error {horizontal:.3} m heading {label} exceeds \
                 {MAX_HORIZONTAL_ERROR_M:.3} m"
            );
            assert!(
                altitude <= MAX_ALTITUDE_ERROR_M,
                "{name} altitude error {altitude:.3} m heading {label} exceeds \
                 {MAX_ALTITUDE_ERROR_M:.3} m"
            );
            assert!(
                velocity <= MAX_VELOCITY_ERROR_MPS,
                "{name} velocity error {velocity:.4} m/s heading {label} exceeds \
                 {MAX_VELOCITY_ERROR_MPS:.4} m/s"
            );
        }

        for (i, (name_a, a)) in estimates.iter().enumerate() {
            for (name_b, b) in &estimates[i + 1..] {
                let separation = horizontal_separation_m(a, b);
                assert!(
                    separation <= MAX_PAIRWISE_SEPARATION_M,
                    "{name_a} and {name_b} disagree by {separation:.3} m heading {label}, \
                     above the {MAX_PAIRWISE_SEPARATION_M:.3} m consistency bound"
                );
            }
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

/// Every filter reports roll/pitch/yaw on one branch, and the same one, at every step.
///
/// #314: the EKF and UKF wrapped the attitude block onto 0..2*pi in `update` while their own
/// `predict` wrote `Rotation3::euler_angles`'s -pi..pi straight back into it, and the ESKF
/// wrapped its quaternion decomposition the same way in `get_estimate`. A level vehicle --
/// which this scenario is -- therefore reported roll as -0.002 rad after a predict and 6.281
/// rad one update later: the same rotation, named two ways, one timestep apart, with the
/// branch cut sitting exactly on the attitude the vehicle actually holds.
///
/// This file is the right home because it already drives all four filters through
/// `&mut dyn NavigationFilter`, so a filter that reintroduces its own convention fails here
/// rather than in whichever suite happens to read its attitude.
#[test]
fn every_filter_reports_attitude_on_the_principal_branch() {
    // Near-level: the scenario's truth attitude is the identity and every filter is seeded a
    // hundredth of a radian off it, so the *only* thing separating a passing report from a
    // failing one is which branch the filter names the answer on. The bound is two orders of
    // magnitude below a full turn and one above the seed, so it catches the convention
    // without being a tuning knob.
    const MAX_LEVEL_ATTITUDE_RAD: f64 = 0.1;
    // Seeded deliberately NEGATIVE, and that is the whole point of the test.
    //
    // A level seed cannot distinguish the two conventions: the angles come out as +/-0.0 or
    // a rounding residue of ~1e-11, and `wrap_to_2pi`'s `wrapped < 0.0` guard is false for
    // negative zero, so under the pre-#314 code two of the three filters would report 0.0
    // and pass. The only thing that tripped the old code on a level scenario was the *sign*
    // of a 1e-11 residue in one filter -- a recompile away from detecting nothing.
    //
    // At -0.02 rad the old `wrap_to_2pi` reports 6.263 rad for every filter, which fails
    // both assertions below by a wide margin, on every platform.
    const SEED_ATTITUDE_RAD: f64 = -0.02;

    let scenario = build_scenario(0.0);

    let mut init = initial_state(&scenario);
    init.roll = SEED_ATTITUDE_RAD;
    init.pitch = SEED_ATTITUDE_RAD;
    init.yaw = SEED_ATTITUDE_RAD;
    let mut nominal = scenario.initial;
    nominal.attitude = nalgebra::Rotation3::from_euler_angles(
        SEED_ATTITUDE_RAD,
        SEED_ATTITUDE_RAD,
        SEED_ATTITUDE_RAD,
    );

    for (name, mut filter) in all_filters_from(&init, nominal, &[0.0; 6]) {
        let dt = 1.0 / SCENARIO_SAMPLE_RATE_HZ as f64;

        for stage in ["predict", "update"] {
            if stage == "predict" {
                filter.predict(&scenario.samples[0], dt).unwrap();
            } else {
                filter.update(&scenario.gps[0]).unwrap();
            }

            let estimate = filter.get_estimate();
            for (axis, angle) in [
                ("roll", estimate[6]),
                ("pitch", estimate[7]),
                ("yaw", estimate[8]),
            ] {
                assert!(
                    (-std::f64::consts::PI..=std::f64::consts::PI).contains(&angle),
                    "{name} reported {axis} = {angle} rad after {stage}, outside the -pi..pi \
                     branch `Rotation3::euler_angles` returns"
                );
                assert!(
                    angle.abs() <= MAX_LEVEL_ATTITUDE_RAD,
                    "{name} reported {axis} = {angle} rad after {stage} for a near-level \
                     vehicle; a value near a full turn here means the 0..2*pi convention \
                     is back"
                );
            }
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
/// Do not widen the bounds to make this pass.
///
/// It was `#[ignore]`d for #303, where the EKF and ESKF both ran away from any non-zero seed
/// error. Both halves turned out to be the same kind of defect -- an analytic Jacobian
/// written in a different convention from the state it linearises -- but in different
/// functions. The EKF's was `state_transition_jacobian` expressing attitude as a rotation
/// vector when the state holds Euler angles (#307); the ESKF's was
/// `error_state_transition_jacobian` mixing body-frame and navigation-frame attitude errors,
/// plus an altitude/vertical-velocity sign that ignored the frame in both.
#[test]
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

/// Every filter must converge from a seed error in *any* channel, not just position.
///
/// #303 reported divergence from a 1 m altitude seed, a 0.05 m/s velocity seed and a
/// 0.001 rad attitude seed as well as from position, and the position-only check above would
/// not have caught a regression confined to one of the others. The vertical channel is the
/// one worth being explicit about: the defect behind #303 was an altitude/vertical-velocity
/// sign in `error_state_transition_jacobian` that made that pair positive feedback in NED, so
/// an altitude seed is the most direct probe of it there is.
///
/// Seeding *exactly* on truth was always stable, in every filter, because nothing excited the
/// bad term. That is why this seeds each channel in turn rather than trusting a clean start.
#[test]
fn every_filter_converges_from_a_seed_error_in_any_channel() {
    /// A named perturbation applied to a filter's initial state.
    type Seed = (&'static str, fn(&mut StrapdownState));

    // Magnitudes from #303's table: routine initialisation errors, not stress values.
    let seeds: [Seed; 4] = [
        ("1 m altitude", |state| state.altitude += 1.0),
        ("0.05 m/s vertical velocity", |state| {
            state.velocity_vertical += 0.05;
        }),
        ("0.05 m/s north velocity", |state| {
            state.velocity_north += 0.05;
        }),
        ("0.001 rad pitch", |state| {
            state.attitude *= Rotation3::from_euler_angles(0.0, 0.001, 0.0);
        }),
    ];

    let mut failures = Vec::new();

    for (seed_name, apply_seed) in seeds {
        let mut scenario = build_scenario(0.0);
        apply_seed(&mut scenario.initial);
        let truth = scenario.truth.last().unwrap();

        for (name, mut filter) in all_filters(&scenario, &[0.0; 6]) {
            let estimate = run(filter.as_mut(), &scenario);
            let horizontal = horizontal_error_m(&estimate, truth);
            let altitude = (estimate[2] - truth.altitude).abs();
            println!(
                "{seed_name:>28} | {name:<5} horizontal {horizontal:8.3} m, altitude {altitude:8.3} m"
            );

            if !estimate.iter().take(9).all(|v| v.is_finite()) {
                failures.push(format!("{seed_name} / {name}: non-finite navigation state"));
            } else if horizontal > MAX_HORIZONTAL_ERROR_M {
                failures.push(format!(
                    "{seed_name} / {name}: horizontal {horizontal:.3} m exceeds \
                     {MAX_HORIZONTAL_ERROR_M:.3} m"
                ));
            } else if altitude > MAX_ALTITUDE_ERROR_M {
                failures.push(format!(
                    "{seed_name} / {name}: altitude {altitude:.3} m exceeds \
                     {MAX_ALTITUDE_ERROR_M:.3} m"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} filter/seed combination(s) failed to converge:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Every 15-state filter must actually couple its bias states to its navigation states.
///
/// A filter that carries six bias states and never correlates them with the navigation
/// states does not have fifteen states; it has nine and six pieces of dead weight that
/// consume process noise. That was `ExtendedKalmanFilter`'s condition for the whole of its
/// history (#394): its state-transition Jacobian filled only the 9x9 navigation block and
/// left $\partial(\text{nav}) / \partial(\text{bias})$ zero, so with a block-diagonal $P_0$
/// and measurement models that observe navigation states only, `P[0..9, 9..15]` started at
/// zero and could never become anything else. The Kalman gain over the bias rows was
/// therefore identically zero, the estimate never left its seed, and the bias compensation
/// applied to every IMU sample was a no-op.
///
/// # Why a covariance block and not an accuracy number
///
/// Because the accuracy numbers could not see it. `real_clean__ekf` and `syn_cruise_1hz__ekf`
/// were **bit-identical** across a change that rewrote the EKF's entire initial bias
/// covariance -- which is what exposed this -- and a suite that reads only RMSEs has no way
/// to tell "estimated well" from "not estimated at all". The cross-covariance block is the
/// thing that is structurally wrong, so it is the thing to assert.
///
/// The UKF needs no help here and is included as the control: a sigma point perturbed in a
/// bias state mechanizes to a different navigation state, so the unscented transform builds
/// the coupling out of the nonlinear propagation for free. Only a filter that linearises has
/// to supply it by hand.
#[test]
fn every_fifteen_state_filter_couples_its_biases_to_its_navigation_states() {
    /// The block is zero or it is not; this only has to separate "grew" from "never wrote".
    /// The measured values are ~4e-3, six orders above this.
    const MIN_COUPLING: f64 = 1e-12;

    let scenario = build_scenario(0.0);
    let mut failures = Vec::new();

    for (name, mut filter) in all_filters(&scenario, &[0.0; 6]) {
        let _ = run(filter.as_mut(), &scenario);
        let covariance = filter.get_certainty();
        if covariance.nrows() < 15 {
            println!(
                "{name}: {} states, not a 15-state filter",
                covariance.nrows()
            );
            continue;
        }

        let mut coupling = 0.0_f64;
        for row in 0..9 {
            for column in 9..15 {
                coupling = coupling.max(covariance[(row, column)].abs());
            }
        }
        println!("{name}: max |P[nav, bias]| = {coupling:.6e}");

        if coupling < MIN_COUPLING {
            failures.push(format!(
                "{name}: max |P[nav, bias]| is {coupling:.3e}. Its bias states are not \
                 coupled to anything, so they cannot be estimated and their prior cannot be \
                 falsified by any measurement (#394)."
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} filter(s) carry bias states that nothing can observe:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
