//! The RBPF's position cloud must not collapse to a point when the weights degenerate.
//!
//! Resampling clones particles exactly: a survivor is a bit-identical copy of its ancestor.
//! When the likelihood is far more peaked than the cloud is wide, one particle takes
//! essentially all the weight, and the resample then replaces the whole cloud with copies of
//! that single point. The reported position covariance -- which for an RBPF *is* the cloud's
//! spread, since position is the particle partition -- becomes the float noise around zero.
//!
//! Measured on the reference recording before the fix: 42 of 3597 update epochs reported a
//! horizontal sigma below a millimetre, the smallest **4.95 nanometres**, against a GNSS fix
//! specified at 3.81 m. At those epochs the effective sample size immediately before the
//! resample was **1.0** and the resample produced **one** distinct ancestor out of 500.
//!
//! #385. The repair is roughening ([`RbpfConfig::roughening_factor`]).
//!
//! Since the filter was restructured after Canciani & Raquet -- two sampled states rather than
//! three, altitude in the Kalman partition -- the reference recording no longer degenerates
//! that way even without roughening: its smallest reported sigma unroughened is 3 cm to 33 cm
//! across the configurations measured. So the recording is kept as a regression guard, and the
//! demonstration that roughening is what prevents a collapse when one does happen uses a fix
//! built to cause one.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests does not reach them; unwrapping is how these assert"
)]

use nalgebra::Rotation3;
use strapdown::measurements::GPSPositionMeasurement;
use strapdown::messages::{AidingConfig, Event, build_event_stream};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};
use strapdown::sim::TestDataRecord;
use strapdown::{IMUData, NavigationFilter, StrapdownState};

/// The slice the gated `real_rbpf_slice__rbpf` scenario uses, so the two agree.
const SLICE_SAMPLES: usize = 1200;
const PARTICLES: usize = 500;

/// The horizontal random walk the `conf/` recipes run, m/sqrt(s).
///
/// Not the filter's default, which is Canciani & Raquet's zero (eq. 19). With zero, the cloud's
/// extent shrinks every epoch, so roughening -- which scales with that extent -- cannot hold it
/// open; that collapse is the reason the recipes set this, and is documented on
/// `RbpfConfig::horizontal_process_noise_std_m`. Roughening is tested here as it runs.
const RECIPE_HORIZONTAL_PROCESS_NOISE: nalgebra::Vector2<f64> = nalgebra::Vector2::new(1.0, 1.0);

/// Metres per radian of latitude, near enough for turning a variance into a legible sigma.
const M_PER_RAD: f64 = 6_371_000.0;

/// Every reported horizontal sigma, in metres, over the reference slice.
fn reported_sigmas(roughening_factor: f64) -> Vec<f64> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/test_data.csv");
    let records: Vec<TestDataRecord> = TestDataRecord::from_csv(path).unwrap();
    let slice: Vec<TestDataRecord> = records.into_iter().take(SLICE_SAMPLES).collect();

    let first = &slice[0];
    let (roll, pitch, yaw) = first.attitude().euler_angles();
    let (velocity_north, velocity_east) = first.ground_track_velocity();
    let nominal = StrapdownState {
        latitude: first.latitude.to_radians(),
        longitude: first.longitude.to_radians(),
        altitude: first.altitude,
        velocity_north,
        velocity_east,
        velocity_vertical: 0.0,
        attitude: Rotation3::from_euler_angles(roll, pitch, yaw),
        // The recording is ENU, as `integration_tests.rs`'s `TEST_DATA_IS_ENU` records.
        is_enu: true,
    };

    let mut config = RbpfConfig::default();
    config.num_particles = PARTICLES;
    config.seed = 42;
    config.roughening_factor = roughening_factor;
    config.horizontal_process_noise_std_m = RECIPE_HORIZONTAL_PROCESS_NOISE;
    let mut filter = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();

    let stream = build_event_stream(&slice, &AidingConfig::default(), true).unwrap();

    let mut sigmas = Vec::new();
    for event in stream.events {
        match event {
            Event::Imu { dt_s, imu, .. } => filter.predict(&imu, dt_s).unwrap(),
            Event::Measurement { meas, .. } => {
                filter.update(meas.as_ref()).unwrap();
                let (_, covariance) = filter.estimate();
                sigmas.push(covariance[(0, 0)].sqrt() * M_PER_RAD);
            }
        }
    }
    sigmas
}

/// No epoch may report a position uncertainty that is physically meaningless.
///
/// The bound is a millimetre, which is four orders of magnitude below the smallest sigma the
/// roughened filter actually reports (0.12 m) and five above the collapse it is catching
/// (4.95e-9 m). It is deliberately nowhere near either, so it fails only on a real collapse
/// and not on ordinary movement of the estimate.
///
/// It is **not** set at the GNSS fix noise. A filter fusing a 1 Hz fix with continuous
/// inertial data is entitled to report a posterior tighter than one fix -- that is what
/// filtering is for -- so a bound at 3.81 m would forbid the correct behaviour along with the
/// incorrect.
#[test]
fn the_position_cloud_does_not_collapse_to_a_point() {
    const FLOOR_M: f64 = 1e-3;

    let sigmas = reported_sigmas(RbpfConfig::default().roughening_factor);
    let collapsed: Vec<(usize, f64)> = sigmas
        .iter()
        .enumerate()
        .filter(|(_, s)| **s < FLOOR_M)
        .map(|(i, s)| (i, *s))
        .collect();

    assert!(
        collapsed.is_empty(),
        "{} of {} epochs report a horizontal sigma below {FLOOR_M} m, the smallest {:e} m. \
         The cloud has collapsed onto a single resampled ancestor; see `roughening_factor`.",
        collapsed.len(),
        sigmas.len(),
        collapsed
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::INFINITY, f64::min)
    );
}

/// The sigma reported right after each of a few centimetre-accurate fixes against a 10 m
/// cloud, with the given roughening.
///
/// A likelihood three orders of magnitude narrower than the cloud puts essentially all the
/// weight on one particle, which is the condition #385 recorded on the reference recording
/// before the restructure made it rare there.
fn sigmas_after_degenerate_fixes(roughening_factor: f64) -> Vec<f64> {
    let nominal = StrapdownState {
        latitude: 40.0_f64.to_radians(),
        longitude: (-105.0_f64).to_radians(),
        altitude: 1600.0,
        attitude: Rotation3::identity(),
        is_enu: true,
        ..StrapdownState::default()
    };
    let mut config = RbpfConfig::default();
    config.num_particles = PARTICLES;
    config.seed = 385;
    config.roughening_factor = roughening_factor;
    config.horizontal_process_noise_std_m = RECIPE_HORIZONTAL_PROCESS_NOISE;
    let mut filter = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();
    let fix = GPSPositionMeasurement {
        latitude: 40.0,
        longitude: -105.0,
        altitude: 1600.0,
        horizontal_noise_std: 0.01,
        vertical_noise_std: 1.0,
    };
    let imu = IMUData {
        accel: nalgebra::Vector3::new(0.0, 0.0, strapdown::earth::gravity(&40.0, &1600.0)),
        gyro: nalgebra::Vector3::zeros(),
    };
    let mut sigmas = Vec::new();
    for _ in 0..5 {
        filter.predict(&imu, 0.1).unwrap();
        filter.update(&fix).unwrap();
        let (_, covariance) = filter.estimate();
        sigmas.push(covariance[(0, 0)].sqrt() * M_PER_RAD);
    }
    sigmas
}

/// The guard above must be able to fail: a fix that degenerates the weights collapses the
/// cloud without roughening, and does not with it.
///
/// Without this, a change that quietly stopped the cloud from ever being resampled would leave
/// the test above green while removing the thing it checks.
#[test]
fn without_roughening_a_degenerate_fix_collapses_the_cloud() {
    let unroughened = sigmas_after_degenerate_fixes(0.0);
    assert!(
        unroughened.iter().any(|sigma| *sigma < 1e-3),
        "with roughening disabled a centimetre fix against a 10 m cloud no longer collapses it \
         (sigmas {unroughened:?}), so `the_position_cloud_does_not_collapse_to_a_point` is no \
         longer testing anything. Either the resampling path changed or the weights no longer \
         degenerate."
    );
    let roughened = sigmas_after_degenerate_fixes(RbpfConfig::default().roughening_factor);
    assert!(
        roughened.iter().all(|sigma| *sigma >= 1e-3),
        "with roughening on, the same fixes still collapsed the cloud: {roughened:?}"
    );
}
