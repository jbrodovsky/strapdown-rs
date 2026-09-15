//! End-to-end cover for the geophysical closed loop.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//!
//! Until this existed, nothing ran the geophysical closed loop from a map on disk through to a
//! finished solution. `run_geo_closed_loop_cli` carried a comment saying it had no test at all,
//! and the consequence was that the feature could not complete a single run: the filters carry
//! one bias state per active map, `NavigationResult`'s conversion asserted a flat fifteen, and
//! every run died on its first result regardless of filter or map.
//!
//! The assertion was right to exist -- a wrong state shape really is a crate invariant
//! violation -- so the fix was to tell the conversion what the extra states are rather than to
//! loosen it. These tests hold both halves of that: a geophysical run completes and labels its
//! bias column, and a run that declares no geophysical states still rejects a longer state.
//!
//! The particle filter reached the same solution from the other direction. Its conversion is
//! nine states rather than fifteen and never asserted its way out of the problem: it accepted
//! the run and wrote rows whose geophysical columns were blank, for a filter that had
//! estimated the bias all along. Its innovation gate read the same short summary and scored
//! every geophysical fix on an attitude angle standing in for the bias.
//! `gravity_aided_particle_filter_labels_its_bias_state` covers both against a real map.

use std::rc::Rc;

use chrono::{TimeZone, Utc};
use geonav::{
    GeoBiasLayout, GeoMap, GeophysicalMeasurementType, GravityMeasurement, GravityResolution,
    MagneticResolution, NAVIGATION_AND_IMU_BIAS_STATE_DIM, NAVIGATION_STATE_DIM,
    build_event_stream,
};
use nalgebra::{DMatrix, DVector};
use strapdown::kalman::ExtendedKalmanFilter;
use strapdown::messages::{Event, GnssDegradationConfig, GnssFaultModel, GnssScheduler};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};
use strapdown::sim::{
    DEFAULT_PROCESS_NOISE, GeoStateLayout, NavigationResult, TestDataRecord, UkfConfig,
    initialize_ukf, run_closed_loop, run_closed_loop_with_geo,
};
use strapdown::{NavigationFilter, StrapdownState};

/// A small gravity-anomaly map written to a temporary NetCDF file.
///
/// Generated rather than vendored: the grid only has to cover the synthetic track below and be
/// smooth enough to have a usable gradient, and a checked-in `.nc` would be a binary fixture no
/// one can review. The field is two sinusoids so the anomaly actually varies along the track --
/// a constant map is one the filter can learn nothing from, which would let the bias assertion
/// below pass for the wrong reason.
fn write_gravity_map(path: &std::path::Path) {
    write_anomaly_map(path, 0.0);
}

/// The same grid as a magnetic-anomaly map, offset so the track's constant observed anomaly
/// sits inside its range rather than far outside it.
///
/// `MagneticAnomalyMeasurement` differences the observed field against the World Magnetic
/// Model, so the map this is matched against has to be in the same units and neighbourhood as
/// that difference; `magnetometer_reading` below is what puts it there.
fn write_magnetic_map(path: &std::path::Path) {
    write_anomaly_map(path, MAGNETIC_MAP_OFFSET_NT);
}

fn write_anomaly_map(path: &std::path::Path, offset: f64) {
    let lats: Vec<f64> = (0..=40).map(|i| 40.0 + f64::from(i) * 0.005).collect();
    let lons: Vec<f64> = (0..=40).map(|i| -76.0 + f64::from(i) * 0.005).collect();
    let mut z = Vec::with_capacity(lats.len() * lons.len());
    for lat in &lats {
        for lon in &lons {
            z.push(
                offset + 25.0 * ((lat - 40.0) * 600.0).sin() + 15.0 * ((lon + 76.0) * 400.0).cos(),
            );
        }
    }

    let mut file = netcdf::create(path).expect("a temporary NetCDF file must be creatable");
    file.add_dimension("lat", lats.len()).unwrap();
    file.add_dimension("lon", lons.len()).unwrap();
    file.add_variable::<f64>("lat", &["lat"])
        .unwrap()
        .put_values(&lats, ..)
        .unwrap();
    file.add_variable::<f64>("lon", &["lon"])
        .unwrap()
        .put_values(&lons, ..)
        .unwrap();
    file.add_variable::<f64>("z", &["lat", "lon"])
        .unwrap()
        .put_values(&z, ..)
        .unwrap();
}

/// A short, level, due-north track inside the generated map, at 1 Hz.
///
/// Deliberately synthetic rather than `core/tests/test_data.csv`: this test is about the state
/// plumbing, and a hand-built track keeps it fast and keeps its assertions independent of that
/// recording's own quirks.
fn synthetic_track(samples: usize) -> Vec<TestDataRecord> {
    let start = Utc.with_ymd_and_hms(2025, 3, 1, 0, 0, 0).unwrap();
    (0..samples)
        .map(|i| TestDataRecord {
            time: start + chrono::Duration::seconds(i64::try_from(i).unwrap_or(i64::MAX)),
            latitude: 40.05 + f64::from(u32::try_from(i).unwrap_or(u32::MAX)) * 1e-5,
            longitude: -75.95,
            altitude: 100.0,
            speed: 1.0,
            bearing: 0.0,
            horizontal_accuracy: 5.0,
            vertical_accuracy: 3.0,
            speed_accuracy: 0.5,
            bearing_accuracy: 1.0,
            // Level and at rest: +g on the device up-axis, which is the ENU convention this
            // format uses and what `is_enu: true` declares below.
            acc_z: 9.81,
            qw: 1.0,
            ..Default::default()
        })
        .collect()
}

/// Where the magnetic map's anomalies sit, and what the track's magnetometer reads.
///
/// Both in the units `MagneticAnomalyMeasurement` works in. The map is the same field as the
/// gravity one shifted here, and the observed reading is a constant inside that band, so the
/// innovation is the map's own variation along the track plus whatever the bias is carrying --
/// small enough to be a well-posed update, varying enough that the bias is actually observable.
const MAGNETIC_MAP_OFFSET_NT: f64 = 300.0;
const MAGNETOMETER_READING_NT: f64 = 300.0;

/// The same track with a magnetometer that reads a constant total field.
fn with_magnetometer(records: Vec<TestDataRecord>) -> Vec<TestDataRecord> {
    records
        .into_iter()
        .map(|record| TestDataRecord {
            mag_z: MAGNETOMETER_READING_NT,
            ..record
        })
        .collect()
}

fn passthrough_config() -> GnssDegradationConfig {
    GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    }
}

/// A geophysically aided UKF carrying one bias state, tuned as the CLI tunes it.
fn aided_ukf(first: &TestDataRecord) -> strapdown::kalman::UnscentedKalmanFilter {
    let mut process_noise: Vec<f64> = DEFAULT_PROCESS_NOISE.into();
    process_noise.push(1e-9);
    initialize_ukf(
        first,
        UkfConfig {
            other_states: Some(vec![0.0]),
            other_states_covariance: Some(vec![100.0]),
            process_noise_diagonal: Some(process_noise),
            is_enu: true,
            ..Default::default()
        },
    )
    .expect("a geophysically aided UKF must initialise")
}

/// A geophysically aided EKF carrying one bias state, tuned as the CLI's `FilterType::Ekf` arm
/// tunes it.
///
/// The CLI builds this by hand -- `initialize_ekf` has no `other_states`, so the geophysical
/// covariance and process noise are extended at the call site -- which is why it is worth
/// mirroring here rather than reaching for a constructor.
fn aided_ekf(first: &TestDataRecord) -> ExtendedKalmanFilter {
    let mut covariance_diagonal = vec![
        1e-10, 1e-10, 1.0, // position
        0.1, 0.1, 0.1, // velocity
        1e-4, 1e-4, 1e-4, // attitude
        1e-6, 1e-6, 1e-6, // accel bias
        1e-8, 1e-8, 1e-8, // gyro bias
    ];
    covariance_diagonal.push(1.0);
    let mut process_noise_vec = vec![
        1e-12, 1e-12, 1e-6, // position
        1e-6, 1e-6, 1e-6, // velocity
        1e-9, 1e-9, 1e-9, // attitude
        1e-9, 1e-9, 1e-9, // accel bias
        1e-9, 1e-9, 1e-9, // gyro bias
    ];
    process_noise_vec.push(1e-9);

    ExtendedKalmanFilter::new(
        &first.initial_state(true),
        &[0.0; 6],
        covariance_diagonal,
        DMatrix::from_diagonal(&DVector::from_vec(process_noise_vec)),
        true,
    )
}

/// The two assertions that separate an estimated bias from a carried-along constant.
///
/// The movement check alone is not enough: a bias row that only ever accumulates process noise
/// is the signature of the frozen state, and it would pass as soon as anything at all nudged the
/// bias. Requiring the variance to fall below its seed is what says the measurements are
/// actually informing it.
fn assert_bias_is_estimated(biases: &[f64], covariances: &[f64], name: &str) {
    assert!(
        biases.iter().all(|b| b.is_finite()),
        "every estimated {name} bias must be finite"
    );
    assert!(
        biases.iter().any(|b| (b - biases[0]).abs() > 1e-9),
        "the {name} bias never moved from its seed, so the aiding is not reaching the state"
    );

    let seed = covariances[0];
    assert!(
        covariances.iter().any(|c| *c < seed),
        "the {name} bias variance never fell below its seed of {seed}, so the measurements are \
         not informing it"
    );
}

/// The bias layout for a gravity-only run, in both the forms a geophysical run needs.
///
/// `GeoBiasLayout` tells the measurement models where the bias lives; `GeoStateLayout` tells
/// `NavigationResult` the same thing on the `core` side of the dependency edge. Derived from
/// the first rather than declared twice, exactly as `run_geo_closed_loop_cli` does it, so the
/// placement has one source of truth.
fn gravity_only_layouts() -> (GeoBiasLayout, GeoStateLayout) {
    let bias = GeoBiasLayout::appended(NAVIGATION_AND_IMU_BIAS_STATE_DIM, true, false)
        .expect("a gravity-only layout over the 15-state Kalman vector must be valid")
        .expect("asking for a gravity bias must yield a layout");
    let state = GeoStateLayout::new(
        bias.state_dim(),
        bias.gravity_bias().map(|b| b.index),
        bias.magnetic_bias().map(|b| b.index),
    );
    (bias, state)
}

/// The same pair for a magnetic-only run.
fn magnetic_only_layouts() -> (GeoBiasLayout, GeoStateLayout) {
    let bias = GeoBiasLayout::appended(NAVIGATION_AND_IMU_BIAS_STATE_DIM, false, true)
        .expect("a magnetic-only layout over the 15-state Kalman vector must be valid")
        .expect("asking for a magnetic bias must yield a layout");
    let state = GeoStateLayout::new(
        bias.state_dim(),
        bias.gravity_bias().map(|b| b.index),
        bias.magnetic_bias().map(|b| b.index),
    );
    (bias, state)
}

/// A gravity-aided run completes and carries its bias state into the solution.
///
/// This is the regression: before the layout reached the conversion, this run panicked on
/// its very first result with "State vector must have 15 elements; got 16".
#[test]
fn gravity_aided_closed_loop_completes_and_labels_its_bias_state() {
    let dir = std::env::temp_dir().join(format!("geonav-closed-loop-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let map_path = dir.join("gravity.nc");
    write_gravity_map(&map_path);

    let map = Rc::new(
        GeoMap::load_geomap(
            &map_path,
            GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute),
        )
        .expect("the generated map must load"),
    );

    let records = synthetic_track(60);
    // One extra state for the one map, exactly as `run_geo_closed_loop_cli` builds it.
    let (bias_layout, layout) = gravity_only_layouts();
    assert_eq!(layout.len(), 1);
    assert_eq!(layout.state_dim(), 16);

    let events = build_event_stream(
        &records,
        &passthrough_config(),
        Some(Rc::clone(&map)),
        Some(1.0),
        None,
        None,
        Some(1.0),
        Some(bias_layout),
    )
    .expect("the geophysical event stream must build");

    let mut ukf = aided_ukf(&records[0]);

    // That the filter really carries sixteen states is asserted by the conversion itself: it
    // requires `NAVIGATION_STATES + layout.len()` exactly, so a run that completes with this
    // layout could not have had any other shape. `NavigationFilter` is crate-private, so this
    // is also the only way to state it from outside the crate.
    let results = run_closed_loop_with_geo(&mut ukf, events, None, None, layout)
        .expect("the geophysical closed loop must complete a run");

    assert!(!results.is_empty(), "the run must produce solutions");
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.gravity_bias.is_some(),
            "row {i} carried a gravity map, so its gravity bias must be populated"
        );
        assert!(
            result.gravity_bias_cov.is_some(),
            "row {i} must carry the gravity bias covariance alongside the bias"
        );
        // The distinction the `Option` exists for: no magnetic map means no magnetic column,
        // which is not the same as a magnetic bias estimated at zero.
        assert!(
            result.magnetic_bias.is_none(),
            "row {i} carried no magnetic map, so its magnetic bias must be absent, not zero"
        );
        assert!(result.magnetic_bias_cov.is_none());
    }

    // The bias is a state the filter estimates, not a constant it carries along: with gradient
    // under the track, the measurements have to move it off its seed.
    let biases: Vec<f64> = results.iter().filter_map(|r| r.gravity_bias).collect();
    assert!(
        biases.iter().all(|b| b.is_finite()),
        "every estimated gravity bias must be finite"
    );
    assert!(
        biases.iter().any(|b| (b - biases[0]).abs() > 1e-9),
        "the gravity bias never moved from its seed, so the aiding is not reaching the state"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A run that declares no geophysical states still rejects a filter that has them.
///
/// The fix told the conversion about the extra states; it did not loosen the invariant.
/// A 16-state filter handed to the plain entry point is a caller that forgot its layout, and
/// silently writing a solution that drops the bias state would be worse than the panic.
#[test]
#[should_panic(expected = "State vector must have 15 elements")]
fn plain_closed_loop_still_rejects_a_geophysical_filter() {
    let records = synthetic_track(5);
    let events = strapdown::messages::build_event_stream(&records, &passthrough_config(), true)
        .expect("the plain event stream must build");
    let mut ukf = aided_ukf(&records[0]);
    let _ = run_closed_loop(&mut ukf, events, None, None);
}

/// The EKF branch of the geophysical CLI carries its bias state too, and estimates it.
///
/// The CLI builds the EKF by hand -- `initialize_ekf` has no `other_states`, so the geophysical
/// covariance and process noise are extended at the call site -- which makes it a different
/// construction path from the UKF above and worth covering separately. This mirrors what
/// `run_geo_closed_loop_cli`'s `FilterType::Ekf` arm assembles.
///
/// The movement assertion at the end is a second regression on this path, and the reason the
/// comment that used to stand here -- saying the bias does not move on this path and that
/// asserting it would fail -- is gone. The EKF's bias used to sit at exactly its seed for a whole
/// run while its variance grew on process noise alone, because the anomaly models' Jacobian was
/// a fixed 1x9: the expected measurement added the bias but the linearization claimed no
/// dependence on it, so the gain's bias row was zero, and because the Joseph update leaves
/// `P[bias, :]` untouched when that row is zero, no cross-covariance with position ever
/// developed to make it non-zero either. The UKF never showed this -- its sigma points propagate
/// the augmented state whether or not anything declares a derivative for it, and it never calls
/// `get_jacobian` at all -- which is why the two filters disagreed on the same run and same map.
/// Giving the Jacobian its bias column is what closed the gap; this pins it shut.
#[test]
fn ekf_branch_completes_and_labels_its_bias_state() {
    let dir = std::env::temp_dir().join(format!("geonav-ekf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let map_path = dir.join("gravity.nc");
    write_gravity_map(&map_path);

    let map = Rc::new(
        GeoMap::load_geomap(
            &map_path,
            GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute),
        )
        .expect("the generated map must load"),
    );

    let records = synthetic_track(60);
    let (bias_layout, layout) = gravity_only_layouts();
    let events = build_event_stream(
        &records,
        &passthrough_config(),
        Some(Rc::clone(&map)),
        Some(1.0),
        None,
        None,
        Some(1.0),
        Some(bias_layout),
    )
    .expect("the geophysical event stream must build");

    let mut ekf = aided_ekf(&records[0]);

    let results = run_closed_loop_with_geo(&mut ekf, events, None, None, layout)
        .expect("the geophysical EKF must complete a run");

    assert!(!results.is_empty());
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.gravity_bias.is_some(),
            "row {i} carried a gravity map, so its gravity bias must be populated"
        );
        assert!(result.gravity_bias_cov.is_some());
        assert!(
            result.magnetic_bias.is_none(),
            "row {i} carried no magnetic map, so its magnetic bias must be absent"
        );
    }

    // The same assertion the UKF above makes: the bias is a state the filter estimates, not a
    // constant it carries along.
    let biases: Vec<f64> = results.iter().filter_map(|r| r.gravity_bias).collect();
    let covariances: Vec<f64> = results.iter().filter_map(|r| r.gravity_bias_cov).collect();
    assert_bias_is_estimated(&biases, &covariances, "gravity");

    std::fs::remove_dir_all(&dir).ok();
}

/// The magnetic half of the same path: a magnetic-only EKF estimates its bias too.
///
/// `MagneticAnomalyMeasurement` carries its own copy of the resolve-and-fill logic that the
/// gravity test above covers, and the CLI reaches it for magnetic-only and combined-map runs, so
/// a regression in that column would leave every gravity assertion green while magnetic runs
/// quietly went back to carrying a frozen bias. This is the end-to-end half of that cover;
/// `test_magnetic_jacobian_carries_a_column_for_the_declared_bias` in the crate's own tests is
/// the direct one.
#[test]
fn magnetic_only_ekf_estimates_its_bias_state() {
    let dir = std::env::temp_dir().join(format!("geonav-ekf-mag-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let map_path = dir.join("magnetic.nc");
    write_magnetic_map(&map_path);

    let map = Rc::new(
        GeoMap::load_geomap(
            &map_path,
            GeophysicalMeasurementType::Magnetic(MagneticResolution::TwoMinutes),
        )
        .expect("the generated map must load"),
    );

    let records = with_magnetometer(synthetic_track(60));
    let (bias_layout, layout) = magnetic_only_layouts();
    let events = build_event_stream(
        &records,
        &passthrough_config(),
        None,
        None,
        Some(Rc::clone(&map)),
        Some(10.0),
        Some(1.0),
        Some(bias_layout),
    )
    .expect("the geophysical event stream must build");

    let mut ekf = aided_ekf(&records[0]);
    let results = run_closed_loop_with_geo(&mut ekf, events, None, None, layout)
        .expect("the magnetic-only EKF must complete a run");

    assert!(!results.is_empty());
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.magnetic_bias.is_some(),
            "row {i} carried a magnetic map, so its magnetic bias must be populated"
        );
        assert!(result.magnetic_bias_cov.is_some());
        // The mirror of the gravity run's check: no gravity map means no gravity column, which
        // is not the same as a gravity bias estimated at zero.
        assert!(
            result.gravity_bias.is_none(),
            "row {i} carried no gravity map, so its gravity bias must be absent, not zero"
        );
        assert!(result.gravity_bias_cov.is_none());
    }

    let biases: Vec<f64> = results.iter().filter_map(|r| r.magnetic_bias).collect();
    let covariances: Vec<f64> = results.iter().filter_map(|r| r.magnetic_bias_cov).collect();
    assert_bias_is_estimated(&biases, &covariances, "magnetic");

    std::fs::remove_dir_all(&dir).ok();
}

/// The same gravity-only pair for a filter with no IMU-bias block.
///
/// `gravity_only_layouts` builds the fifteen-state Kalman version; this one passes
/// [`NAVIGATION_STATE_DIM`] as the base, which is what `sim` passes for an RBPF run, so the
/// bias lands at index 9 rather than 15. Everything downstream -- where the measurement reads
/// its bias from, which column the conversion files it in -- follows from that one number.
fn gravity_only_particle_layouts() -> (GeoBiasLayout, GeoStateLayout) {
    let bias = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, false)
        .expect("a gravity-only layout over the 9-state particle vector must be valid")
        .expect("asking for a gravity bias must yield a layout");
    let state = GeoStateLayout::new(
        bias.state_dim(),
        bias.gravity_bias().map(|b| b.index),
        bias.magnetic_bias().map(|b| b.index),
    );
    (bias, state)
}

/// A gravity-aided particle-filter run scores and reports its bias state.
///
/// Both halves of what a nine-state summary of the cloud costs, against a real map.
///
/// *Reporting.* The particle filter was the one aided path still writing rows with the
/// geophysical columns blank. It configures `extra_state_dim` for each active map and the
/// particles really do carry a bias -- the weight update reads it -- but the solution was
/// assembled from `estimate()`, which has nowhere to put it, so every row came out looking
/// complete and missing the quantity the aiding exists to produce.
///
/// *Scoring.* The gate had the same summary and a worse failure mode: a model that reads its
/// bias by index got the wrong entry rather than none, so every geophysical fix was scored on
/// an attitude angle standing in for the bias (#354). `core`'s
/// `rbpf_gate_scores_the_extra_state_and_not_the_yaw_angle` covers that with a synthetic
/// measurement; the NIS assertions below are the same property with a `GravityMeasurement`
/// reading a NetCDF map, which is the configuration it was reported against.
///
/// This runs the loop `strapdown-sim`'s `run_rbpf_event_loop` runs, which lives in that
/// binary and so cannot be called from here. Under test is the trio it now uses:
/// `estimate_with_extra_states`, the gate inside `update`, and
/// `from_particle_filter_with_geo`.
#[test]
fn gravity_aided_particle_filter_labels_its_bias_state() {
    let dir = std::env::temp_dir().join(format!("geonav-rbpf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let map_path = dir.join("gravity.nc");
    write_gravity_map(&map_path);

    let map = Rc::new(
        GeoMap::load_geomap(
            &map_path,
            GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute),
        )
        .expect("the generated map must load"),
    );

    let records = synthetic_track(60);
    // One map, so one extra linear state, appended after the nine navigation states. `sim`
    // derives the filter's `extra_state_dim`, the measurement's declared bias index and this
    // layout from the one `GeoBiasLayout`, which is what keeps all three in step.
    let (bias_layout, layout) = gravity_only_particle_layouts();
    assert_eq!(layout.len(), 1);
    assert_eq!(
        layout.state_dim(),
        10,
        "nine navigation states and the one map bias -- not the Kalman sixteen"
    );
    assert_eq!(layout.gravity_index(), Some(9));

    let events = build_event_stream(
        &records,
        &passthrough_config(),
        Some(Rc::clone(&map)),
        Some(1.0),
        None,
        None,
        Some(1.0),
        Some(bias_layout),
    )
    .expect("the geophysical event stream must build");

    let first = &records[0];
    let (velocity_north, velocity_east) = first.ground_track_velocity();
    let nominal = StrapdownState {
        latitude: first.latitude.to_radians(),
        longitude: first.longitude.to_radians(),
        altitude: first.altitude,
        velocity_north,
        velocity_east,
        velocity_vertical: 0.0,
        attitude: first.attitude(),
        is_enu: true,
    };
    let mut rbpf = RaoBlackwellizedParticleFilter::new(
        nominal,
        RbpfConfig {
            num_particles: 200,
            extra_state_dim: layout.len(),
            extra_state_init_std: 10.0,
            extra_state_process_noise_std: 0.1,
            seed: 42,
            ..RbpfConfig::default()
        },
    )
    .expect("the aided RBPF must initialise");

    let start_time = events.start_time;
    let mut results = Vec::new();
    let (mean, cov) = rbpf.estimate_with_extra_states();
    assert_eq!(
        mean.len(),
        10,
        "nine navigation states plus the one map bias"
    );
    results.push(NavigationResult::from_particle_filter_with_geo(
        &start_time,
        &mean,
        &cov,
        layout,
    ));

    let mut gravity_nis: Vec<f64> = Vec::new();
    for event in events.events {
        let elapsed_s = match &event {
            Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
        };
        let ts = start_time + chrono::Duration::milliseconds((elapsed_s * 1000.0).round() as i64);
        match event {
            Event::Imu { dt_s, imu, .. } => rbpf.predict(&imu, dt_s).unwrap(),
            Event::Measurement { meas, .. } => {
                let is_gravity = meas.as_any().downcast_ref::<GravityMeasurement>().is_some();
                // This `unwrap` is itself the gate assertion. `evaluate_ensemble_gate`
                // summarises the cloud before handing it to the model, and the model checks
                // the width against the `BiasState` it was declared with: a nine-state
                // summary fails here with
                // `DimensionMismatch { what: "geophysical bias state: filter state width",
                // expected: 10, got: 9 }` rather than quietly scoring the wrong entry.
                let outcome = rbpf.update(meas.as_ref()).unwrap();
                if is_gravity {
                    gravity_nis.push(outcome.nis);
                }
            }
        }
        let (mean, cov) = rbpf.estimate_with_extra_states();
        results.push(NavigationResult::from_particle_filter_with_geo(
            &ts, &mean, &cov, layout,
        ));
    }

    assert!(!results.is_empty(), "the run must produce solutions");
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.gravity_bias.is_some(),
            "row {i} carried a gravity map, so its gravity bias must be populated"
        );
        assert!(
            result.gravity_bias_cov.is_some(),
            "row {i} must carry the gravity bias covariance alongside the bias"
        );
        // The distinction the `Option` exists for, as on the Kalman paths.
        assert!(
            result.magnetic_bias.is_none(),
            "row {i} carried no magnetic map, so its magnetic bias must be absent, not zero"
        );
        assert!(result.magnetic_bias_cov.is_none());
    }

    let biases: Vec<f64> = results.iter().filter_map(|r| r.gravity_bias).collect();
    let variances: Vec<f64> = results.iter().filter_map(|r| r.gravity_bias_cov).collect();
    assert!(
        biases.iter().all(|b| b.is_finite()),
        "every estimated gravity bias must be finite"
    );
    assert!(
        variances.iter().all(|v| v.is_finite() && *v > 0.0),
        "a bias reported with no uncertainty on it is not one a reader can use"
    );

    // The bias is a state the cloud estimates, not a seed carried along. Unlike the Kalman
    // paths the geophysical fix never enters a linear update here -- it reweights and
    // resamples the particles -- so this is the assertion that the reweighting is reaching the
    // bias dimension at all. Over this track it walks from 0.70 to 3.93 mGal while the
    // reported variance falls from 103 to 0.077, so the bound below is far looser than the
    // movement it is guarding.
    assert!(
        biases.iter().any(|b| (b - biases[0]).abs() > 1e-9),
        "the gravity bias never moved from its seed, so the aiding is not reaching the state"
    );
    assert!(
        variances.iter().any(|v| (v - variances[0]).abs() > 1e-9),
        "the reported bias variance never changed, so it is a constant rather than this run's \
         uncertainty"
    );
    // The gate, end to end against a real map. #354 fixed `evaluate_ensemble_gate` to
    // summarise the cloud with `estimate_with_extra_states`, and covered it with a synthetic
    // measurement in `core`; this is the same property with a `GravityMeasurement` reading a
    // NetCDF map, which is the configuration the defect was reported against.
    //
    // A correctly scored one-degree-of-freedom fix has a NIS of order 1, and these do: 59
    // fixes, median 0.76, largest 1.01. Scoring them on the yaw angle instead -- ~0 rad on
    // this due-north track -- in place of a bias that converges near 15 mGal, against a
    // 1 mGal noise standard deviation, would put the NIS two orders of magnitude higher. The
    // bound below sits between the two rather than fitting the observed numbers.
    assert!(
        gravity_nis.len() >= 50,
        "the run scored only {} geophysical fixes; with too few this asserts nothing",
        gravity_nis.len()
    );
    assert!(
        gravity_nis
            .iter()
            .all(|nis| nis.is_finite() && *nis >= 0.0 && *nis < 10.0),
        "every geophysical fix must gate at a NIS of order 1; got a largest of {}",
        gravity_nis
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    std::fs::remove_dir_all(&dir).ok();
}
