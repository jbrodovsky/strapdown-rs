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

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests does not reach them; unwrapping is how these assert"
)]

use nalgebra::Rotation3;
use strapdown::NavigationFilter;
use strapdown::StrapdownState;
use strapdown::messages::{AidingConfig, Event, build_event_stream};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};
use strapdown::sim::TestDataRecord;

/// The slice the gated `real_rbpf_slice__rbpf` scenario uses, so the two agree.
const SLICE_SAMPLES: usize = 1200;
const PARTICLES: usize = 500;

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
        is_enu: false,
    };

    let mut config = RbpfConfig::default();
    config.num_particles = PARTICLES;
    config.seed = 42;
    config.roughening_factor = roughening_factor;
    let mut filter = RaoBlackwellizedParticleFilter::new(nominal, config).unwrap();

    let stream = build_event_stream(&slice, &AidingConfig::default(), false).unwrap();

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

/// The guard above must be able to fail: with roughening off, the collapse returns.
///
/// Without this, a change that quietly stopped the cloud from ever being resampled would leave
/// the test above green while removing the thing it checks.
#[test]
fn without_roughening_the_collapse_is_reproducible() {
    let sigmas = reported_sigmas(0.0);
    let collapsed = sigmas.iter().filter(|s| **s < 1e-3).count();

    assert!(
        collapsed > 0,
        "with roughening disabled the cloud no longer collapses, so \
         `the_position_cloud_does_not_collapse_to_a_point` is no longer testing anything. \
         Either the resampling path changed or the scenario no longer degenerates."
    );
}
