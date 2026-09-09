//! Comprehensive integration tests for INS filters using real data
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//!
//! This module contains end-to-end integration tests for the strapdown inertial navigation
//! filters using real data recorded from a MEMS-grade IMU. See [mems-nav-dataset](www.github.com/jbrodovsky/mems-nav-dataset).
//! These tests ensure that the entire navigation system works as expected in realistic scenarios, not just
//! at the API level but with actual IMU and GNSS data.
//!
//! ## Error Metrics
//!
//! The tests use the following error metrics to validate filter performance:
//! - **Horizontal position error**: Haversine distance between estimated and GNSS positions (meters)
//! - **Altitude error**: Simple difference between estimated and GNSS altitude (meters)
//! - **Velocity error**: Component-wise differences for north, east, and down velocities (m/s)
//! - **Orientation error**: Component-wise differences for roll, pitch, and yaw (radians)
//!
//! The specific performance numbers given in the assertions in the test are not theoretical
//! or design goals, but rather empirically derived from running the filters on the dataset and observing
//! performance on the test data set. They serve as regression checks to ensure that future changes
//! do not degrade performance.
//!
//! ## Test Structure
//!
//! Tests load real data from CSV files, run the filters, and compute error metrics against
//! GNSS measurements. The tests verify that:
//! 1. Filters complete without errors
//! 2. Position errors remain within reasonable bounds
//! 3. Velocity and orientation estimates are stable
//! 4. The closed-loop filter outperforms dead reckoning
use std::path::Path;

use strapdown::NavigationFilter;
use strapdown::StrapdownState;
use strapdown::earth::haversine_distance;
use strapdown::kalman::{
    ErrorStateKalmanFilter, ExtendedKalmanFilter, InitialState, UnscentedKalmanFilter,
};
use strapdown::messages::{
    Event, GnssDegradationConfig, GnssFaultModel, GnssScheduler, build_event_stream,
};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};
use strapdown::sim::{
    NavigationResult, TestDataRecord, dead_reckoning, initialize_eskf, run_closed_loop,
};

use nalgebra::{DMatrix, DVector, Quaternion, Rotation3, UnitQuaternion, Vector3};

/// Default process noise covariance for testing (15-state)
const DEFAULT_PROCESS_NOISE: [f64; 15] = [
    1e-6, // latitude noise
    1e-6, // longitude noise
    1e-6, // altitude noise
    1e-3, // velocity north noise
    1e-3, // velocity east noise
    1e-3, // velocity down noise
    1e-5, // roll noise
    1e-5, // pitch noise
    1e-5, // yaw noise
    1e-6, // acc bias x noise
    1e-6, // acc bias y noise
    1e-6, // acc bias z noise
    1e-8, // gyro bias x noise
    1e-8, // gyro bias y noise
    1e-8, // gyro bias z noise
];

/// Default initial covariance for testing (15-state)
const DEFAULT_INITIAL_COVARIANCE: [f64; 15] = [
    1e-6, 1e-6, 1.0, // position covariance (lat, lon, alt in meters)
    0.1, 0.1, 0.1, // velocity covariance (m/s)
    0.01, 0.01, 0.01, // attitude covariance (radians)
    0.01, 0.01, 0.01, // accelerometer bias covariance (m/s²)
    0.001, 0.001, 0.001, // gyroscope bias covariance (rad/s)
];

/// Minimum meaningful drift for dead reckoning comparison (meters)
/// Below this threshold, the comparison is not meaningful as the vehicle may be stationary
const MIN_DRIFT_FOR_COMPARISON: f64 = 5.0;

/// The ESKF tuning under test is the shipped one, not a test-local copy.
///
/// These used to be independent constants -- an initial covariance whose bias entries were
/// five orders of magnitude looser than `initialize_eskf`'s and a process noise 8x the
/// library default -- with nothing recording which of the two tunings was intended. The
/// answer is that there is only one, so these now alias the library's, and a change to the
/// tuning a `strapdown-sim` user gets is a change to what this suite measures.
///
/// What the old test-local tuning cost, measured on `test_data.csv` with full GNSS aiding:
/// its 8x bias-state process noise carried the gyro-bias estimate into the anti-windup
/// clamp on 76 of the 5,366 samples, including the last, where gyro bias y sat at exactly
/// the 0.05 rad/s cap. Attribution is unambiguous -- holding the initial covariance and
/// scaling only the bias entries of Q reproduces the saturation (37 clamped samples at 8x,
/// none at 4x), while scaling only the nine navigation-state entries does not (none) -- so
/// what those runs demonstrated about the bias estimates was the clamp, not the estimator.
/// `strapdown::sim::ESKF_INITIAL_ERROR_COVARIANCE` documents the priors themselves.
const ESKF_INITIAL_COVARIANCE: [f64; 15] = strapdown::sim::ESKF_INITIAL_ERROR_COVARIANCE;
const ESKF_PROCESS_NOISE: [f64; 15] = strapdown::sim::DEFAULT_PROCESS_NOISE;
/// Anti-windup caps the ESKF clamps its bias estimates to (`kalman.rs`, #286).
///
/// Orders of magnitude above legitimate consumer-MEMS turn-on biases (~0.1 m/s^2,
/// ~0.01 rad/s) and far below the runaway values a persistently faulty aiding sensor
/// otherwise produces -- 9.2 m/s^2 and 6.5 rad/s on this very dataset before #286.
const MAX_ACCEL_BIAS_MPS2: f64 = 2.0;
const MAX_GYRO_BIAS_RPS: f64 = 0.05;

/// Assert the bias estimates stayed bounded at *every* sample, not just the last one.
///
/// #258 asks for boundedness across the whole run, and the distinction matters: checking
/// only the final estimate cannot tell a filter whose biases never moved from one that
/// wound up to a runaway value mid-run and was dragged back by a later fix. The clamp in
/// `inject_error_state` guarantees the final value regardless, so a final-sample assertion
/// tests the clamp rather than the filter.
fn assert_bias_estimates_bounded(results: &[NavigationResult], context: &str) {
    let accel_peak = |pick: fn(&NavigationResult) -> f64| {
        results
            .iter()
            .map(|r| pick(r).abs())
            .fold(0.0_f64, f64::max)
    };
    println!(
        "{context}: peak |bias| accel=[{:.4}, {:.4}, {:.4}] m/s^2, gyro=[{:.5}, {:.5}, {:.5}] rad/s",
        accel_peak(|r| r.acc_bias_x),
        accel_peak(|r| r.acc_bias_y),
        accel_peak(|r| r.acc_bias_z),
        accel_peak(|r| r.gyro_bias_x),
        accel_peak(|r| r.gyro_bias_y),
        accel_peak(|r| r.gyro_bias_z),
    );

    for (i, result) in results.iter().enumerate() {
        for (axis, bias) in [
            ("x", result.acc_bias_x),
            ("y", result.acc_bias_y),
            ("z", result.acc_bias_z),
        ] {
            assert!(
                bias.is_finite() && bias.abs() <= MAX_ACCEL_BIAS_MPS2,
                "{context}: accel bias {axis} left its physical bound at sample {i} of {}: \
                 {bias:.3} m/s^2 exceeds {MAX_ACCEL_BIAS_MPS2} (see #258, #286)",
                results.len()
            );
        }
        for (axis, bias) in [
            ("x", result.gyro_bias_x),
            ("y", result.gyro_bias_y),
            ("z", result.gyro_bias_z),
        ] {
            assert!(
                bias.is_finite() && bias.abs() <= MAX_GYRO_BIAS_RPS,
                "{context}: gyro bias {axis} left its physical bound at sample {i} of {}: \
                 {bias:.4} rad/s exceeds {MAX_GYRO_BIAS_RPS} (see #258, #286)",
                results.len()
            );
        }
    }
}

/// Error statistics for a navigation solution
#[allow(
    clippy::struct_field_names,
    reason = "every field is an error metric; dropping the `_error` suffix would make `mean_horizontal` and `rms_altitude` ambiguous against the non-error quantities in scope"
)]
#[derive(Debug, Clone)]
struct ErrorStats {
    /// Mean horizontal position error (meters)
    mean_horizontal_error: f64,
    /// Minimum horizontal position error (meters)
    min_horizontal_error: f64,
    /// Median horizontal position error (meters)
    median_horizontal_error: f64,
    /// Maximum horizontal position error (meters)
    max_horizontal_error: f64,
    /// Root mean square horizontal position error (meters)
    rms_horizontal_error: f64,
    /// Mean altitude error (meters)
    mean_altitude_error: f64,
    /// Minimum altitude error (meters)
    min_altitude_error: f64,
    /// Median altitude error (meters)
    median_altitude_error: f64,
    /// Maximum altitude error (meters)
    max_altitude_error: f64,
    /// Root mean square altitude error (meters)
    rms_altitude_error: f64,
    /// Mean velocity north error (m/s)
    mean_velocity_north_error: f64,
    /// Mean velocity east error (m/s)
    mean_velocity_east_error: f64,
    /// Mean velocity down error (m/s)
    mean_velocity_vertical_error: f64,
}

impl ErrorStats {
    /// Create a new ErrorStats with all zeros
    const fn new() -> Self {
        Self {
            mean_horizontal_error: 0.0,
            min_horizontal_error: 0.0,
            median_horizontal_error: 0.0,
            max_horizontal_error: 0.0,
            rms_horizontal_error: 0.0,
            mean_altitude_error: 0.0,
            min_altitude_error: 0.0,
            median_altitude_error: 0.0,
            max_altitude_error: 0.0,
            rms_altitude_error: 0.0,
            mean_velocity_north_error: 0.0,
            mean_velocity_east_error: 0.0,
            mean_velocity_vertical_error: 0.0,
        }
    }
}

/// Compute error metrics between navigation results and GNSS truth data
///
/// This function calculates various error metrics by comparing the filter's navigation
/// solution against GNSS measurements treated as ground truth. It computes:
/// - Horizontal position error using haversine distance
/// - Altitude error as simple difference
/// - Velocity component errors
///
/// # Arguments
/// - `results` - Navigation results from filter (estimated state)
/// - `records` - Test data records containing GNSS measurements (truth)
///
/// # Returns
/// ErrorStats containing mean, max, and RMS errors for various quantities
fn compute_error_metrics(results: &[NavigationResult], records: &[TestDataRecord]) -> ErrorStats {
    let mut horizontal_errors = Vec::new();
    let mut altitude_errors = Vec::new();
    let mut velocity_north_errors = Vec::new();
    let mut velocity_east_errors = Vec::new();
    let mut velocity_vertical_errors = Vec::new();

    // Match navigation results to GNSS measurements by timestamp
    for (i, result) in results.iter().enumerate() {
        // Find matching record by timestamp
        if let Some(record) = records.iter().find(|r| r.time == result.timestamp) {
            // Skip if GNSS data is invalid (NaN)
            if record.latitude.is_nan()
                || record.longitude.is_nan()
                || record.altitude.is_nan()
                || record.horizontal_accuracy.is_nan()
            {
                continue;
            }

            // Debug first few values
            if i < 3 {
                println!(
                    "Record {}: result.lat={:.6}, result.lon={:.6}, result.alt={:.2}",
                    i, result.latitude, result.longitude, result.altitude
                );
                println!(
                    "Record {}: record.lat={:.6}, record.lon={:.6}, record.alt={:.2}",
                    i, record.latitude, record.longitude, record.altitude
                );
            }

            // Compute horizontal position error using haversine distance
            // NavigationResult stores lat/lon in degrees, TestDataRecord also in degrees
            let horizontal_error = haversine_distance(
                result.latitude.to_radians(),
                result.longitude.to_radians(),
                record.latitude.to_radians(),
                record.longitude.to_radians(),
            );

            if i < 3 {
                println!("Record {i}: horizontal_error={horizontal_error:.2}m");
            }

            // Skip invalid errors (NaN or Inf)
            if !horizontal_error.is_finite() {
                if i < 10 || horizontal_errors.len() < 10 {
                    println!("WARNING: Skipping non-finite horizontal_error at index {i}");
                }
                continue;
            }

            horizontal_errors.push(horizontal_error);

            // Compute altitude error
            let altitude_error = (result.altitude - record.altitude).abs();
            if altitude_error.is_finite() {
                altitude_errors.push(altitude_error);
            }

            // Compute velocity errors
            // Note: GNSS provides speed and bearing, need to convert to N-E components
            let gnss_vel_north = record.speed * record.bearing.to_radians().cos();
            let gnss_vel_east = record.speed * record.bearing.to_radians().sin();

            let vn_err = (result.velocity_north - gnss_vel_north).abs();
            let ve_err = (result.velocity_east - gnss_vel_east).abs();
            let vd_err = result.velocity_vertical.abs();

            if vn_err.is_finite() {
                velocity_north_errors.push(vn_err);
            }
            if ve_err.is_finite() {
                velocity_east_errors.push(ve_err);
            }
            if vd_err.is_finite() {
                velocity_vertical_errors.push(vd_err);
            }
        }
    }

    // Compute statistics
    let mut stats = ErrorStats::new();

    println!("Collected {} horizontal errors", horizontal_errors.len());
    if horizontal_errors.len() > 10 {
        println!(
            "Last 10 horizontal errors: {:?}",
            &horizontal_errors[horizontal_errors.len() - 10..]
        );
    }

    if !horizontal_errors.is_empty() {
        stats.mean_horizontal_error =
            horizontal_errors.iter().sum::<f64>() / horizontal_errors.len() as f64;
        stats.min_horizontal_error = horizontal_errors
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        stats.max_horizontal_error = horizontal_errors
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        stats.rms_horizontal_error = (horizontal_errors.iter().map(|e| e.powi(2)).sum::<f64>()
            / horizontal_errors.len() as f64)
            .sqrt();

        // Compute median
        let mut sorted_horizontal = horizontal_errors.clone();
        sorted_horizontal.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted_horizontal.len() / 2;
        stats.median_horizontal_error = if sorted_horizontal.len() % 2 == 0 {
            f64::midpoint(sorted_horizontal[mid - 1], sorted_horizontal[mid])
        } else {
            sorted_horizontal[mid]
        };
    }

    if !altitude_errors.is_empty() {
        stats.mean_altitude_error =
            altitude_errors.iter().sum::<f64>() / altitude_errors.len() as f64;
        stats.min_altitude_error = altitude_errors
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        stats.max_altitude_error = altitude_errors
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        stats.rms_altitude_error = (altitude_errors.iter().map(|e| e.powi(2)).sum::<f64>()
            / altitude_errors.len() as f64)
            .sqrt();

        // Compute median
        let mut sorted_altitude = altitude_errors.clone();
        sorted_altitude.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted_altitude.len() / 2;
        stats.median_altitude_error = if sorted_altitude.len() % 2 == 0 {
            f64::midpoint(sorted_altitude[mid - 1], sorted_altitude[mid])
        } else {
            sorted_altitude[mid]
        };
    }

    if !velocity_north_errors.is_empty() {
        stats.mean_velocity_north_error =
            velocity_north_errors.iter().sum::<f64>() / velocity_north_errors.len() as f64;
        stats.mean_velocity_east_error =
            velocity_east_errors.iter().sum::<f64>() / velocity_east_errors.len() as f64;
        stats.mean_velocity_vertical_error =
            velocity_vertical_errors.iter().sum::<f64>() / velocity_vertical_errors.len() as f64;
    }

    stats
}

/// Load test data from the provided CSV file
///
/// # Arguments
/// - `path` - Path to the CSV file containing test data
///
/// # Returns
/// Vector of TestDataRecord instances
fn load_test_data(path: &Path) -> Vec<TestDataRecord> {
    TestDataRecord::from_csv(path)
        .unwrap_or_else(|_| panic!("Failed to load test data from CSV: {}", path.display()))
}

/// Create an initial state from the first test data record
///
/// # Arguments
/// - `first_record` - The first test data record
///
/// # Returns
/// InitialState for filter initialization
fn create_initial_state(first_record: &TestDataRecord) -> InitialState {
    // NOTE: Test data from Sensor Logger has:
    //   - latitude/longitude in degrees
    //   - roll/pitch/yaw in a different Euler convention than nalgebra's XYZ
    //   - quaternion (qw, qx, qy, qz) is the most reliable attitude representation
    //
    // We use the quaternion to extract XYZ Euler angles that nalgebra expects.
    use nalgebra::{Quaternion, Rotation3, UnitQuaternion};

    // Convert quaternion to rotation matrix, then extract XYZ Euler angles
    let quat = UnitQuaternion::from_quaternion(Quaternion::new(
        first_record.qw,
        first_record.qx,
        first_record.qy,
        first_record.qz,
    ));
    let rot: Rotation3<f64> = quat.into();
    let (roll, pitch, yaw) = rot.euler_angles();

    InitialState {
        latitude: first_record.latitude.to_radians(),
        longitude: first_record.longitude.to_radians(),
        altitude: first_record.altitude,
        northward_velocity: first_record.speed * first_record.bearing.to_radians().cos(),
        eastward_velocity: first_record.speed * first_record.bearing.to_radians().sin(),
        vertical_velocity: 0.0,
        roll,
        pitch,
        yaw,
        in_degrees: false, // All angles now in radians
        // ENU on purpose: `test_data.csv` is a Sensor Logger export, whose accelerometer reads
        // +g along the device's up-axis at rest. Pinning it explicitly -- rather than leaning on
        // the crate default, which is now NED -- is what keeps this suite's numbers unchanged
        // across the frame flip, so any movement here is attributable to something else.
        is_enu: true,
    }
}

/// Create a nominal StrapdownState from the first test data record
fn create_nominal_state(first_record: &TestDataRecord) -> StrapdownState {
    let quat = UnitQuaternion::from_quaternion(Quaternion::new(
        first_record.qw,
        first_record.qx,
        first_record.qy,
        first_record.qz,
    ));
    let rot: Rotation3<f64> = quat.into();
    let (roll, pitch, yaw) = rot.euler_angles();

    StrapdownState {
        latitude: first_record.latitude.to_radians(),
        longitude: first_record.longitude.to_radians(),
        altitude: first_record.altitude,
        velocity_north: first_record.speed * first_record.bearing.to_radians().cos(),
        velocity_east: first_record.speed * first_record.bearing.to_radians().sin(),
        velocity_vertical: 0.0,
        attitude: Rotation3::from_euler_angles(roll, pitch, yaw),
        // ENU on purpose: `test_data.csv` is a Sensor Logger export, whose accelerometer reads
        // +g along the device's up-axis at rest. Pinning it explicitly -- rather than leaning on
        // the crate default, which is now NED -- is what keeps this suite's numbers unchanged
        // across the frame flip, so any movement here is attributable to something else.
        is_enu: true,
    }
}

/// Particle count for RBPF tests running against undegraded GNSS.
///
/// This matches `RbpfConfig::default()`. A sweep over `core/tests/test_data.csv`
/// (seed 42) shows accuracy here is flat in particle count, so the previous value
/// of 5000 bought nothing but runtime:
///
/// | particles | median horiz | rms horiz | wall  |
/// |-----------|--------------|-----------|-------|
/// | 250       | 23.86 m      | 24.41 m   | 12 s  |
/// | 500       | 23.67 m      | 24.19 m   | 23 s  |
/// | 5000      | 23.50 m      | 23.98 m   | 226 s |
///
/// The assertions below clear their thresholds by roughly 9x at this count.
const RBPF_PARTICLES: usize = 500;

/// Particle count for the degraded-GNSS RBPF test.
///
/// Kept at 5000 as the reference configuration: with the #267 fixes (wrapped
/// angular likelihoods, proposal matched to the fault scale) the error
/// decreases with particle count and then plateaus (250→2000: 150→125→109→111 m
/// median; the 2000-vs-1000 wiggle is within seed noise, measured at 15% spread
/// across seeds) and is robust across seeds (117-136 m at 500 particles), so
/// this asserts a bound rather than the old 5000-or-bust coincidence.
const RBPF_DEGRADED_PARTICLES: usize = 5000;

fn run_rbpf_with_cfg(
    records: &[TestDataRecord],
    cfg: &GnssDegradationConfig,
    rbpf_config: RbpfConfig,
) -> Vec<NavigationResult> {
    let stream = build_event_stream(records, cfg);

    let nominal = create_nominal_state(&records[0]);
    let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, rbpf_config).unwrap();

    let start_time = stream.start_time;
    let mut results: Vec<NavigationResult> = Vec::with_capacity(stream.events.len());
    let mut last_ts: Option<chrono::DateTime<chrono::Utc>> = None;

    let stream_events_len = stream.events.len();
    for (i, event) in stream.events.into_iter().enumerate() {
        let elapsed_s = match &event {
            Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
        };
        let ts = start_time + chrono::Duration::milliseconds((elapsed_s * 1000.0).round() as i64);

        match event {
            Event::Imu { dt_s, imu, .. } => rbpf.predict(&imu, dt_s).unwrap(),
            Event::Measurement { meas, .. } => rbpf.update(meas.as_ref()).unwrap(),
        }

        if Some(ts) != last_ts {
            if let Some(prev_ts) = last_ts {
                let (mean, cov) = rbpf.estimate();
                results.push(NavigationResult::from_particle_filter(
                    &prev_ts, &mean, &cov,
                ));
            }
            last_ts = Some(ts);
        }

        if i + 1 == stream_events_len {
            let (mean, cov) = rbpf.estimate();
            results.push(NavigationResult::from_particle_filter(&ts, &mean, &cov));
        }
    }

    results
}

fn run_rbpf(records: &[TestDataRecord]) -> Vec<NavigationResult> {
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };
    run_rbpf_with_cfg(
        records,
        &cfg,
        RbpfConfig {
            num_particles: RBPF_PARTICLES,
            seed: 42,
            ..RbpfConfig::default()
        },
    )
}

/// Test dead reckoning on real data to establish baseline
///
/// This test runs pure INS dead reckoning (no GNSS corrections) on real data and
/// verifies that the filter completes without errors. It also computes error metrics
/// to establish a baseline for comparison with closed-loop filtering.
#[test]
fn test_dead_reckoning_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning
    let results = dead_reckoning(&records).unwrap();

    // Verify results
    assert_eq!(
        results.len(),
        records.len(),
        "Dead reckoning should produce one result per input record"
    );

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics for reference
    println!("\n=== Dead Reckoning Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Dead reckoning will drift over time, but should not produce NaN or infinite values
    for result in &results {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite: {}",
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite: {}",
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite: {}",
            result.altitude
        );
    }
}

/// Test UKF closed-loop filter on real data
///
/// This test runs a closed-loop UKF with GNSS measurements on real data and verifies that:
/// 1. The filter completes without errors
/// 2. Position errors remain bounded
/// 3. The filter performs better than dead reckoning
#[test]
fn test_ukf_closed_loop_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize UKF
    let imu_biases = vec![0.0; 6]; // Zero initial bias estimates
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None, // No measurement bias
        initial_covariance,
        process_noise,
        1e-3, // alpha
        2.0,  // beta
        0.0,  // kappa
    );

    // Create event stream with passthrough scheduler (all GNSS measurements used)
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results =
        run_closed_loop(&mut ukf, stream, None, None).expect("Closed-loop filter should complete");

    // Verify results
    assert!(
        !results.is_empty(),
        "Closed-loop filter should produce results"
    );

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== UKF Closed-Loop Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Assert error bounds: they should hold for a healthy aided filter on this
    // data. The three filters agree at ~24 m horizontal rms, an order of
    // magnitude above the ~4.7 m fix noise floor because of dynamics, so the
    // bounds below are ~1.6x the observed healthy value: tight enough that any
    // divergence trips them instantly (dead reckoning is at 5e6 m), loose
    // enough that floating-point codegen differences between platforms cannot
    // cross them (see #288: the old 39.0 m bound carried only 2% margin).
    let rms_horizontal_limit = 40.0;
    let max_horizontal_limit = 60.0;
    let rms_altitude_limit = 50.0;
    let max_altitude_limit = 250.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "UKF RMS horizontal error should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "UKF maximum horizontal error should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "UKF RMS altitude error should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "UKF maximum altitude error should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Verify no NaN or infinite values in results
    for result in &results {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite: {}",
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite: {}",
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite: {}",
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite: {}",
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite: {}",
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity down should be finite: {}",
            result.velocity_vertical
        );
    }
}

/// Test UKF with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate and verifies
/// that the filter still performs reasonably well, though with higher errors than full-rate GNSS.
#[test]
fn test_ukf_with_degraded_gnss() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize UKF
    let imu_biases = vec![0.0; 6];
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None,
        initial_covariance,
        process_noise,
        1e-3, // alpha
        2.0,  // beta
        0.0,  // kappa
    );

    // Create event stream with periodic scheduler (e.g., every 5 seconds)
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::FixedInterval {
            interval_s: 5.0,
            phase_s: 0.0,
        },
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results = run_closed_loop(&mut ukf, stream, None, None)
        .expect("Closed-loop filter with degraded GNSS should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== UKF with Degraded GNSS (5s updates) Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Error bounds should be looser than full-rate GNSS but still reasonable
    assert!(
        stats.rms_horizontal_error < 50.0,
        "RMS horizontal error with degraded GNSS should be less than 50m, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < 400.0,
        "Maximum horizontal error with degraded GNSS should be less than 400m, got {:.2}m",
        stats.max_horizontal_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test that closed-loop UKF outperforms dead reckoning
///
/// This test runs both dead reckoning and UKF on the same data and verifies that
/// the UKF produces lower errors than dead reckoning, demonstrating the benefit
/// of GNSS-aided navigation.
#[test]
fn test_ukf_outperforms_dead_reckoning() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning
    let dr_results = dead_reckoning(&records).unwrap();
    let dr_stats = compute_error_metrics(&dr_results, &records);

    // Run UKF
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6];
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001,
    ];
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None,
        initial_covariance,
        process_noise,
        1e-3,
        2.0,
        0.0,
    );

    let scheduler = GnssScheduler::PassThrough;
    let fault_model = GnssFaultModel::None;
    let cfg = GnssDegradationConfig {
        scheduler,
        fault: fault_model,
        ..Default::default()
    };
    let stream = build_event_stream(&records, &cfg);

    let ukf_results = run_closed_loop(&mut ukf, stream, None, None).expect("UKF should complete");
    let ukf_stats = compute_error_metrics(&ukf_results, &records);

    // Print comparison
    println!("\n=== Performance Comparison ===");
    println!(
        "Dead Reckoning RMS Horizontal Error: {:.2}m",
        dr_stats.rms_horizontal_error
    );
    println!(
        "UKF RMS Horizontal Error: {:.2}m",
        ukf_stats.rms_horizontal_error
    );
    println!(
        "Improvement: {:.1}%",
        (1.0 - ukf_stats.rms_horizontal_error / dr_stats.rms_horizontal_error) * 100.0
    );

    // UKF should significantly outperform dead reckoning
    // Allow for some tolerance in case of very short datasets or near-stationary conditions
    if dr_stats.rms_horizontal_error > MIN_DRIFT_FOR_COMPARISON {
        // Only compare if DR has meaningful drift
        assert!(
            ukf_stats.rms_horizontal_error < dr_stats.rms_horizontal_error,
            "UKF should have lower RMS horizontal error than dead reckoning. UKF: {:.2}m, DR: {:.2}m",
            ukf_stats.rms_horizontal_error,
            dr_stats.rms_horizontal_error
        );
    }
}

// ==================== Extended Kalman Filter Integration Tests ====================

/// Test EKF closed-loop filter on real data
///
/// This test runs a closed-loop EKF with GNSS measurements on real data and verifies that:
/// 1. The filter completes without errors
/// 2. Position errors remain bounded
/// 3. The filter performs comparably to UKF
#[test]
fn test_ekf_closed_loop_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize EKF with 15-state configuration (with biases)
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    // Initialize EKF (note: EKF constructor differs from UKF - no measurement bias parameter,
    // uses use_biases flag instead of optional measurement_bias)
    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_covariance,
        process_noise,
        true, // use_biases (15-state configuration)
    );

    // Create event stream with passthrough scheduler (all GNSS measurements used)
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results = run_closed_loop(&mut ekf, stream, None, None)
        .expect("Closed-loop EKF filter should complete");

    // Verify results
    assert!(
        !results.is_empty(),
        "Closed-loop EKF filter should produce results"
    );

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== EKF Closed-Loop Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Assert error bounds - these should be reasonable for a working filter with GNSS
    // With good GNSS, horizontal error should be within a few meters RMS
    // EKF may have slightly higher errors than UKF due to linearization

    // Same healthy-filter rationale as the UKF test (see #288): the three
    // filters agree at ~24-27 m horizontal rms, so hold the EKF to ~1.7x that.
    // Altitude bounds stay wide deliberately: the EKF vertical channel can
    // excursion under sparse aiding (#290), and with 1 s fixes that stays
    // reined in (max 173.5 m observed) but is not bit-stable across platforms.
    let rms_horizontal_limit = 45.0;
    let max_horizontal_limit = 175.0;
    let rms_altitude_limit = 150.0;
    let max_altitude_limit = 1230.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "EKF RMS horizontal error should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "EKF maximum horizontal error should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "EKF RMS altitude error should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "EKF maximum altitude error should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Verify no NaN or infinite values in results
    for result in &results {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite: {}",
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite: {}",
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite: {}",
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite: {}",
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite: {}",
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity vertical should be finite: {}",
            result.velocity_vertical
        );
    }
}

/// Test EKF with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate and verifies
/// that the filter still performs reasonably well, though with higher errors than full-rate GNSS.
///
/// Degradation profile: `FixedInterval { interval_s: 5.0 }` with `fault: None`
/// (uncorrupted fixes, dataset accuracies: horizontal sigma ~4.7 m, vertical
/// sigma ~1.4 m), plus the per-sample baro/mag aiding present in every stream.
#[test]
fn test_ekf_with_degraded_gnss() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize EKF
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_covariance,
        process_noise,
        true, // 15-state with biases
    );

    // Create event stream with periodic scheduler (e.g., every 5 seconds)
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::FixedInterval {
            interval_s: 5.0,
            phase_s: 0.0,
        },
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results = run_closed_loop(&mut ekf, stream, None, None)
        .expect("Closed-loop EKF filter with degraded GNSS should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== EKF with Degraded GNSS (5s updates) Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Error bounds should be looser than full-rate GNSS but still reasonable.
    // EKF may have slightly higher errors than UKF due to linearization.
    //
    // Two kinds of bound are used here, and they must not be confused (see #288).
    // Typical (median) accuracy is governed by dead-reckoning drift between the
    // 5 s fixes: rate error x 5 s plus the fix noise floor (~4.7 m horizontal,
    // ~1.4 m vertical). With a 10 m/s credible horizontal rate error and a
    // 5 m/s credible vertical rate error that gives 50 m / 25 m; observed
    // medians on Linux are 29.5 m / 8.8 m, so both carry real margin.
    // The median is used (rather than the mean) because it is insensitive to
    // the excursion tail and hence stable across floating-point codegen.
    //
    // The rms/max asserts are anti-divergence guards, not accuracy bounds: the
    // EKF vertical channel suffers a large excursion at ~29 m/s with 5 s fixes
    // (rms 81 m Linux / 186 m macOS, max ~945 m; UKF holds 5.9 m on the same
    // stream), tracked by #290. They are set at ~1.6-2x the worst observed
    // cross-platform value so a genuine divergence (1e8 m scale, cf. #266)
    // still trips them while codegen jitter cannot. Do not tighten these to
    // observed values without fixing #290 first.
    let median_horizontal_limit = 50.0;
    let median_altitude_limit = 25.0;
    let rms_horizontal_limit = 150.0;
    let max_horizontal_limit = 1700.0;
    let rms_altitude_limit = 300.0;
    let max_altitude_limit = 2000.0;

    assert!(
        stats.median_horizontal_error < median_horizontal_limit,
        "EKF median horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        median_horizontal_limit,
        stats.median_horizontal_error
    );
    assert!(
        stats.median_altitude_error < median_altitude_limit,
        "EKF median altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        median_altitude_limit,
        stats.median_altitude_error
    );

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "EKF RMS horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "EKF maximum horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "EKF RMS altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "EKF maximum altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test that closed-loop EKF outperforms dead reckoning
///
/// This test runs both dead reckoning and EKF on the same data and verifies that
/// the EKF produces lower errors than dead reckoning, demonstrating the benefit
/// of GNSS-aided navigation.
#[test]
fn test_ekf_outperforms_dead_reckoning() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning
    let dr_results = dead_reckoning(&records).unwrap();
    let dr_stats = compute_error_metrics(&dr_results, &records);

    // Run EKF
    let initial_state = create_initial_state(&records[0]);
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_covariance,
        process_noise,
        true, // 15-state
    );

    let scheduler = GnssScheduler::PassThrough;
    let fault_model = GnssFaultModel::None;
    let cfg = GnssDegradationConfig {
        scheduler,
        fault: fault_model,
        ..Default::default()
    };
    let stream = build_event_stream(&records, &cfg);

    let ekf_results = run_closed_loop(&mut ekf, stream, None, None).expect("EKF should complete");
    let ekf_stats = compute_error_metrics(&ekf_results, &records);

    // Print comparison
    println!("\n=== Performance Comparison (EKF vs Dead Reckoning) ===");
    println!(
        "Dead Reckoning RMS Horizontal Error: {:.2}m",
        dr_stats.rms_horizontal_error
    );
    println!(
        "EKF RMS Horizontal Error: {:.2}m",
        ekf_stats.rms_horizontal_error
    );
    println!(
        "Improvement: {:.1}%",
        (1.0 - ekf_stats.rms_horizontal_error / dr_stats.rms_horizontal_error) * 100.0
    );

    // EKF should significantly outperform dead reckoning
    // Allow for some tolerance in case of very short datasets or near-stationary conditions
    if dr_stats.rms_horizontal_error > MIN_DRIFT_FOR_COMPARISON {
        // Only compare if DR has meaningful drift
        assert!(
            ekf_stats.rms_horizontal_error < dr_stats.rms_horizontal_error,
            "EKF should have lower RMS horizontal error than dead reckoning. EKF: {:.2}m, DR: {:.2}m",
            ekf_stats.rms_horizontal_error,
            dr_stats.rms_horizontal_error
        );
    }
}

// ==================== Error-State Kalman Filter Integration Tests ====================

/// Test ESKF closed-loop filter on real data
///
/// This test runs a closed-loop ESKF with GNSS measurements on real data and verifies that:
/// 1. The filter completes without errors
/// 2. Position errors remain bounded
/// 3. The filter performs comparably to UKF/EKF
/// 4. Quaternion normalization is maintained
#[test]
fn test_eskf_closed_loop_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize ESKF with 15-state configuration (error-state representation)
    // Use ESKF-specific covariance and process noise to prevent divergence
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    // Initialize ESKF
    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    // Create event stream with passthrough scheduler (all GNSS measurements used)
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results = run_closed_loop(&mut eskf, stream, None, None)
        .expect("Closed-loop ESKF filter should complete");

    // Verify results
    assert!(
        !results.is_empty(),
        "Closed-loop ESKF filter should produce results"
    );

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== ESKF Closed-Loop Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Assert error bounds on the shipped ESKF tuning (see ESKF_INITIAL_COVARIANCE above).

    // Horizontal bounds are physical, not fitted. With continuous GNSS aiding at a
    // few metres of position noise, a correctly closed loosely-coupled filter must
    // stay in the tens of metres; the UKF and EKF sit at 23.6 m and 26.6 m rms on this
    // dataset and the ESKF is now at 23.5 m. The limits below match the UKF test's
    // (~1.7x observed) so all three filters are held to the same standard: they
    // still fail loudly if the horizontal loop opens again (before #266 this run
    // produced 1734 m rms).
    let rms_horizontal_limit = 40.0;
    let max_horizontal_limit = 60.0;

    // Vertical bounds, tightened when #286 landed. The UKF achieves 2.8 m rms /
    // 12.1 m peak on this data and the ESKF now sits alongside it at 2.8 m / 12.3 m.
    // It used to read 2.4 m / 9.2 m: dropping the test-local 8x process noise for the
    // shipped tuning gave up 0.4 m of vertical rms, which is what the 8x bias-state
    // random walk was buying by letting the bias estimates run into the anti-windup
    // clamp. Limits keep ~3.5x margin, so any return of the vertical-channel
    // divergence (previously 119 m rms / 385 m peak) trips them immediately while
    // healthy-filter codegen jitter across platforms cannot.
    let rms_altitude_limit = 10.0;
    let max_altitude_limit = 40.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "ESKF RMS horizontal error should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "ESKF maximum horizontal error should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "ESKF RMS altitude error should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "ESKF maximum altitude error should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Bias plausibility (#286, #258): estimates must stay within the anti-windup caps
    // that bound them at every sample of the run, not merely at the end. Before the
    // #286 fixes these reached 9.2 m/s² and 6.5 rad/s on this same data.
    assert_bias_estimates_bounded(&results, "ESKF closed-loop");
    let final_est = eskf.get_estimate();
    println!(
        "Final biases: accel=[{:.4}, {:.4}, {:.4}] m/s², gyro=[{:.5}, {:.5}, {:.5}] rad/s",
        final_est[9], final_est[10], final_est[11], final_est[12], final_est[13], final_est[14]
    );

    // Verify no NaN or infinite values in results
    for result in &results {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite: {}",
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite: {}",
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite: {}",
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite: {}",
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite: {}",
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity vertical should be finite: {}",
            result.velocity_vertical
        );
    }
}

/// Test ESKF with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate (2s intervals).
#[test]
fn test_eskf_with_degraded_gnss() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize ESKF with ESKF-specific covariance and process noise
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_error_covariance,
        process_noise,
    );

    // Create event stream with periodic scheduler (e.g., every 5 seconds)
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::FixedInterval {
            interval_s: 2.0,
            phase_s: 0.0,
        },
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results = run_closed_loop(&mut eskf, stream, None, None)
        .expect("Closed-loop ESKF filter with degraded GNSS should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== ESKF with Degraded GNSS (2s updates) Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Error bounds for degraded GNSS (2s update intervals).
    //
    // Re-enabled and tightened when #286 landed: with 2 s fixes the healthy
    // ESKF sits at 23.7 m horizontal rms / 40.0 m peak and 3.9 m altitude rms /
    // 14.5 m peak -- barely above the full-rate numbers (23.5 / 2.8 m), since
    // 2 s of MEMS dead-reckoning drift is small next to the fix noise floor.
    // Limits carry ~2.5-4.5x margin: the previous 1000/3500/400/3000 m ceilings
    // were vacuous (any non-divergent filter passed) and are replaced with
    // bounds that actually fail if the vertical channel regresses.
    assert!(
        stats.rms_horizontal_error < 60.0,
        "RMS horizontal error with degraded GNSS should be less than 60m, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < 150.0,
        "Maximum horizontal error with degraded GNSS should be less than 150m, got {:.2}m",
        stats.max_horizontal_error
    );

    assert!(
        stats.rms_altitude_error < 15.0,
        "RMS altitude error with degraded GNSS should be less than 15m, got {:.2}m",
        stats.rms_altitude_error
    );

    assert!(
        stats.max_altitude_error < 60.0,
        "Maximum altitude error with degraded GNSS should be less than 60m, got {:.2}m",
        stats.max_altitude_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }

    // Degraded aiding is the condition bias windup showed up under (#286): halving the
    // fix rate doubles the interval a mis-scaled correction has to accumulate over before
    // the next measurement pulls it back.
    assert_bias_estimates_bounded(&results, "ESKF degraded GNSS");
}

/// Test that closed-loop ESKF outperforms dead reckoning
///
/// This test runs both dead reckoning and ESKF on the same data and verifies that
/// the ESKF produces lower errors than dead reckoning, demonstrating the benefit
/// of GNSS-aided navigation with error-state formulation.
#[test]
fn test_eskf_outperforms_dead_reckoning() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning
    let dr_results = dead_reckoning(&records).unwrap();
    let dr_stats = compute_error_metrics(&dr_results, &records);

    // Run ESKF
    let initial_state = create_initial_state(&records[0]);
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };
    let stream = build_event_stream(&records, &cfg);

    let eskf_results =
        run_closed_loop(&mut eskf, stream, None, None).expect("ESKF should complete");
    let eskf_stats = compute_error_metrics(&eskf_results, &records);

    // Print comparison
    println!("\n=== Performance Comparison (ESKF vs Dead Reckoning) ===");
    println!(
        "Dead Reckoning RMS Horizontal Error: {:.2}m",
        dr_stats.rms_horizontal_error
    );
    println!(
        "ESKF RMS Horizontal Error: {:.2}m",
        eskf_stats.rms_horizontal_error
    );
    println!(
        "Improvement: {:.1}%",
        (1.0 - eskf_stats.rms_horizontal_error / dr_stats.rms_horizontal_error) * 100.0
    );

    // ESKF should significantly outperform dead reckoning
    // Allow for some tolerance in case of very short datasets or near-stationary conditions
    if dr_stats.rms_horizontal_error > MIN_DRIFT_FOR_COMPARISON {
        // Only compare if DR has meaningful drift
        assert!(
            eskf_stats.rms_horizontal_error < dr_stats.rms_horizontal_error,
            "ESKF should have lower RMS horizontal error than dead reckoning. ESKF: {:.2}m, DR: {:.2}m",
            eskf_stats.rms_horizontal_error,
            dr_stats.rms_horizontal_error
        );
    }
}

/// Test ESKF stability with high dynamics
///
/// This test verifies that ESKF maintains stable estimates and proper quaternion
/// normalization even with high dynamics (rapid maneuvers, large accelerations).
/// This is a key advantage of the error-state formulation.
#[test]
fn test_eskf_stability_high_dynamics() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // This test does not apply a distinct high-dynamics tuning, and did not before: a
    // 5x covariance and a 10x process noise sat here `_`-prefixed and unused. They are
    // removed rather than left to imply a tuning the test never applied. What it
    // actually exercises is that the shipped tuning stays finite and keeps the
    // quaternion normalised across the run; a real high-dynamics variant would need a
    // dataset segment with the dynamics to match and is separate work.
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    // Use passthrough GNSS to help constrain the solution
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results = run_closed_loop(&mut eskf, stream, None, None)
        .expect("ESKF with high dynamics should complete");

    // Verify all results are valid (no NaN or Inf)
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite at step {}: {}",
            i,
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite at step {}: {}",
            i,
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite at step {}: {}",
            i,
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite at step {}: {}",
            i,
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite at step {}: {}",
            i,
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity vertical should be finite at step {}: {}",
            i,
            result.velocity_vertical
        );
    }

    // Compute error metrics to verify reasonable performance
    let stats = compute_error_metrics(&results, &records);

    println!("\n=== ESKF High Dynamics Stability Test ===");
    println!(
        "RMS Horizontal Error: {:.2}m, Max: {:.2}m",
        stats.rms_horizontal_error, stats.max_horizontal_error
    );
    println!(
        "RMS Altitude Error: {:.2}m, Max: {:.2}m",
        stats.rms_altitude_error, stats.max_altitude_error
    );

    // Physical bounds, not fitted. Before #266 the ESKF's horizontal loop was open --
    // corrections were rescaled by ~1/6.4e6 -- and this run produced ~1900 m rms, which
    // is what the previous 1905.0 / 2494.0 limits were pinned to. Those numbers were
    // tight enough to the observed value that macOS and Windows failed them at 2121.97 m
    // purely on floating-point code generation. With the loop closed the run sits at
    // 24 m rms, so bound it where a GNSS-aided filter physically belongs.
    let rms_horizontal_limit = 100.0;
    let max_horizontal_limit = 250.0;
    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "RMS horizontal error should remain bounded with high dynamics, expected and error less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "Maximum horizontal error should remain bounded with high dynamics, expected less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
}

/// The construction path a user of the default filter actually takes (#258).
///
/// Every other ESKF test in this file hands `ErrorStateKalmanFilter::new` a covariance and
/// an `InitialState` the suite assembled itself. Now that `FilterType` defaults to `Eskf`,
/// what a `strapdown-sim closed-loop` run actually executes is `initialize_eskf`, and until
/// this test existed nothing exercised it end to end. The two now share a tuning, so what
/// is left to cover here is the entry point itself, called exactly as `strapdown-sim` calls
/// it: every optional argument `None`, and an `InitialState` built from the record's own
/// roll/pitch/yaw in degrees rather than from Euler angles derived from the record
/// quaternion the way `create_initial_state` does. That difference is visible below.
#[test]
fn test_eskf_default_initialization_on_real_data() {
    /// Samples the vertical channel is allowed to settle over: 30 s at this recording's 1 Hz.
    const SETTLING_SAMPLES: usize = 30;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    assert!(!records.is_empty(), "test data should not be empty");

    // Every optional argument `None`: exactly what `strapdown-sim` passes.
    let mut eskf = initialize_eskf(&records[0], None, None, None, None)
        .expect("the default ESKF initialisation must succeed on real data");

    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };
    let results = run_closed_loop(&mut eskf, build_event_stream(&records, &cfg), None, None)
        .expect("the default ESKF must complete the full run");
    assert_eq!(
        results.len(),
        records.len(),
        "the filter must emit one solution per input record"
    );

    let stats = compute_error_metrics(&results, &records);
    println!("\n=== ESKF Default Initialization ===");
    println!(
        "Horizontal Error: rms={:.2}m, max={:.2}m",
        stats.rms_horizontal_error, stats.max_horizontal_error
    );
    println!(
        "Altitude Error: rms={:.2}m, max={:.2}m",
        stats.rms_altitude_error, stats.max_altitude_error
    );

    // Held to the same standard as `test_eskf_closed_loop_on_real_data`. Horizontally the
    // two runs are indistinguishable -- 23.5 m rms / 37.9 m peak, to the centimetre -- which
    // is what continuous GNSS aiding should produce regardless of how the initial attitude
    // was assembled. The vertical channel is where the two differ; see below.
    assert!(
        stats.rms_horizontal_error < 40.0,
        "default-initialised ESKF RMS horizontal error should be under 40m, got {:.2}m",
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < 60.0,
        "default-initialised ESKF max horizontal error should be under 60m, got {:.2}m",
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < 10.0,
        "default-initialised ESKF RMS altitude error should be under 10m, got {:.2}m",
        stats.rms_altitude_error
    );

    // The vertical channel is unobservable at t=0: the filter starts with zero vertical
    // velocity and no knowledge of the accelerometer bias, and needs a few GNSS fixes
    // before it can separate the two. That settling transient peaks at 19.2 m on sample 3
    // of this 1 Hz recording -- it was 42.6 m while the initial altitude error covariance
    // was 1e-4 m², a 1 cm standard deviation that left the filter refusing the very
    // correction it needed. It is bounded separately from the steady state, which is the
    // quantity a vertical-channel regression would move. Excluding it wholesale would hide
    // a divergence, so it gets its own, looser ceiling rather than no ceiling.
    let settled_max_altitude_error = results
        .iter()
        .zip(records.iter())
        .skip(SETTLING_SAMPLES)
        .map(|(result, record)| (result.altitude - record.altitude).abs())
        .fold(0.0_f64, f64::max);
    println!("Altitude Error after settling: max={settled_max_altitude_error:.2}m");
    assert!(
        stats.max_altitude_error < 100.0,
        "default-initialised ESKF altitude settling transient should be under 100m, got {:.2}m",
        stats.max_altitude_error
    );
    // ~3x margin over the 12.3 m observed across the remaining 5,336 samples, and far
    // below the 385 m peak the pre-#286 vertical divergence produced.
    assert!(
        settled_max_altitude_error < 40.0,
        "default-initialised ESKF max altitude error after settling should be under 40m, got {settled_max_altitude_error:.2}m"
    );

    // The anti-windup clamp never engages on this tuning: peak |gyro bias| is 0.0284 rad/s
    // against the 0.05 rad/s cap. So the bound below is a statement about the estimator
    // rather than about the clamp -- which is the whole reason the tuning is what it is,
    // and which the test-local constants could not say while their 8x bias-state process
    // noise was firing the clamp on 76 of the 5,366 samples.
    assert_bias_estimates_bounded(&results, "ESKF default initialization");
    for (i, result) in results.iter().enumerate() {
        for (axis, bias) in [
            ("x", result.gyro_bias_x),
            ("y", result.gyro_bias_y),
            ("z", result.gyro_bias_z),
        ] {
            assert!(
                bias.abs() < MAX_GYRO_BIAS_RPS,
                "gyro bias {axis} reached the anti-windup clamp at sample {i} \
                 ({bias:.5} rad/s): the default tuning is no longer estimating the bias, \
                 it is being held by the clamp (#258)"
            );
        }
    }
}

/// Test comparison of all three filter types (UKF, EKF, ESKF)
///
/// This test runs all three filter types on the same data and compares their performance.
/// It verifies that all filters produce reasonable results and helps understand their
/// relative strengths.
#[test]
// #[ignore = "ESKF diverges on extended real-world datasets - requires further tuning"]
fn test_filter_comparison() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let initial_state = create_initial_state(&records[0]);
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };

    // Run UKF
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        None,
        initial_covariance.clone(),
        process_noise.clone(),
        1e-3,
        2.0,
        0.0,
    );
    let stream_ukf = build_event_stream(&records, &cfg);
    let ukf_results =
        run_closed_loop(&mut ukf, stream_ukf, None, None).expect("UKF should complete");
    let ukf_stats = compute_error_metrics(&ukf_results, &records);

    // Run EKF
    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_covariance,
        process_noise,
        true,
    );
    let stream_ekf = build_event_stream(&records, &cfg);
    let ekf_results =
        run_closed_loop(&mut ekf, stream_ekf, None, None).expect("EKF should complete");
    let ekf_stats = compute_error_metrics(&ekf_results, &records);

    // Run ESKF
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    let stream_eskf = build_event_stream(&records, &cfg);
    let eskf_results =
        run_closed_loop(&mut eskf, stream_eskf, None, None).expect("ESKF should complete");
    let eskf_stats = compute_error_metrics(&eskf_results, &records);

    // Print comparison
    println!("\n=== Filter Performance Comparison ===");
    println!(
        "UKF  - RMS Horizontal: {:.2}m, RMS Altitude: {:.2}m, Max Horizontal: {:.2}m",
        ukf_stats.rms_horizontal_error,
        ukf_stats.rms_altitude_error,
        ukf_stats.max_horizontal_error
    );
    println!(
        "EKF  - RMS Horizontal: {:.2}m, RMS Altitude: {:.2}m, Max Horizontal: {:.2}m",
        ekf_stats.rms_horizontal_error,
        ekf_stats.rms_altitude_error,
        ekf_stats.max_horizontal_error
    );
    println!(
        "ESKF - RMS Horizontal: {:.2}m, RMS Altitude: {:.2}m, Max Horizontal: {:.2}m",
        eskf_stats.rms_horizontal_error,
        eskf_stats.rms_altitude_error,
        eskf_stats.max_horizontal_error
    );

    // All filters should produce reasonable results. Bounds are ~1.6x the
    // observed healthy-filter rms (~24-27 m for all three; dead reckoning is
    // at 5e6 m), not fitted to observed values -- see #288.
    assert!(
        ukf_stats.rms_horizontal_error < 40.0,
        "UKF RMS horizontal error should be reasonable"
    );
    assert!(
        ekf_stats.rms_horizontal_error < 45.0,
        "EKF RMS horizontal error should be reasonable"
    );
    // The 1905.0 m tolerance this used to carry was the signature of #266: the ESKF's
    // horizontal corrections were divided by the principal radii, leaving that channel
    // open loop. With the units fixed the ESKF tracks the other two filters, so hold it
    // to the same standard as the EKF.
    assert!(
        eskf_stats.rms_horizontal_error < 45.0,
        "ESKF RMS horizontal error should be comparable to UKF/EKF, got {:.2}m",
        eskf_stats.rms_horizontal_error
    );

    // Verify all filters completed without producing invalid values
    assert_eq!(ukf_results.len(), ekf_results.len());
    assert_eq!(ekf_results.len(), eskf_results.len());
}

/// Test RBPF on real data with GNSS measurements
#[test]
fn test_rbpf_closed_loop_on_real_data() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let results = run_rbpf(&records);
    assert!(!results.is_empty(), "RBPF should produce results");

    let stats = compute_error_metrics(&results, &records);

    println!("\n=== RBPF Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    assert!(
        stats.rms_horizontal_error < 2200.0,
        "RBPF RMS horizontal error should be less than 2200m, got {:.2}m",
        stats.rms_horizontal_error
    );
    assert!(
        stats.median_horizontal_error < 210.0,
        "RBPF median horizontal error should be less than 210m, got {:.2}m",
        stats.median_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < 100.0,
        "RBPF RMS altitude error should be less than 100m, got {:.2}m",
        stats.rms_altitude_error
    );

    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test RBPF with degraded GNSS measurements.
///
/// Re-enabled when #267 landed. Note the ~200 s runtime: this is the suite's
/// long pole by design (5000 particles over 10.7k events), kept because it is
/// the only test exercising the particle filter under faulted, sparse aiding.
#[test]
fn test_rbpf_with_degraded_gnss() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::FixedInterval {
            interval_s: 5.0,
            phase_s: 0.0,
        },
        fault: GnssFaultModel::Degraded {
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
        },
        ..Default::default()
    };

    let results = run_rbpf_with_cfg(
        &records,
        &cfg,
        RbpfConfig {
            num_particles: RBPF_DEGRADED_PARTICLES,
            seed: 42,
            // Proposal matched to the fault scale: the AR(1) wander
            // (sigma_pos_m 3.0, quasi-bias ±20 m) over 5 s fixes starves the
            // default 1 m proposal cloud (see #267). Explicit here rather
            // than in the default: a wider default proposal measurably
            // degrades clean stationary tracking.
            position_process_noise_std_m: Vector3::new(3.0, 3.0, 3.0),
            ..RbpfConfig::default()
        },
    );
    assert!(!results.is_empty(), "RBPF should produce results");

    let stats = compute_error_metrics(&results, &records);

    println!("\n=== RBPF Degraded GNSS Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Bounds carry ~3-3.5x margin over the observed healthy values
    // (rms_h 204 m, median_h 70 m, rms_alt 8.4 m at seed 42; medians 117-136 m
    // across seeds 1,2,3,7,123 at 500 particles). The old 2200 m rms ceiling
    // was vacuous -- any non-divergent filter passed -- and is replaced with a
    // guard that still trips on the pre-#267 behaviour (km-scale medians).
    assert!(
        stats.rms_horizontal_error < 600.0,
        "RBPF RMS horizontal error with degraded GNSS should be less than 600m, got {:.2}m",
        stats.rms_horizontal_error
    );
    assert!(
        stats.median_horizontal_error < 250.0,
        "RBPF median horizontal error with degraded GNSS should be less than 250m, got {:.2}m",
        stats.median_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < 30.0,
        "RBPF RMS altitude error with degraded GNSS should be less than 30m, got {:.2}m",
        stats.rms_altitude_error
    );

    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test that filter output length matches input data length
///
/// This test verifies that all filter implementations (UKF, EKF, ESKF, dead reckoning)
/// produce output with the same number of records as the input data. This is critical for
/// downstream analysis tools that expect aligned data streams.
#[test]
fn test_filter_output_length_matches_input() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let input_length = records.len();
    println!("Testing with {input_length} input records");

    // Test dead reckoning
    let dr_results = dead_reckoning(&records).unwrap();
    assert_eq!(
        dr_results.len(),
        input_length,
        "Dead reckoning output length {} should match input length {}",
        dr_results.len(),
        input_length
    );
    println!(
        "✓ Dead reckoning: {} outputs for {} inputs",
        dr_results.len(),
        input_length
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6]; // Zero initial bias estimates
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));
    let degradation = GnssDegradationConfig::default();

    // Test UKF
    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None, // No measurement bias
        initial_covariance.clone(),
        process_noise.clone(),
        1e-3, // alpha
        2.0,  // beta
        0.0,  // kappa
    );

    let event_stream = build_event_stream(&records, &degradation);
    let ukf_results = run_closed_loop(&mut ukf, event_stream, None, None)
        .expect("UKF closed loop should complete successfully");

    assert_eq!(
        ukf_results.len(),
        input_length,
        "UKF output length {} should match input length {}",
        ukf_results.len(),
        input_length
    );
    println!(
        "✓ UKF: {} outputs for {} inputs",
        ukf_results.len(),
        input_length
    );

    // Test EKF
    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        initial_covariance,
        process_noise,
        true,
    );

    let event_stream = build_event_stream(&records, &degradation);
    let ekf_results = run_closed_loop(&mut ekf, event_stream, None, None)
        .expect("EKF closed loop should complete successfully");

    assert_eq!(
        ekf_results.len(),
        input_length,
        "EKF output length {} should match input length {}",
        ekf_results.len(),
        input_length
    );
    println!(
        "✓ EKF: {} outputs for {} inputs",
        ekf_results.len(),
        input_length
    );

    // ESKF length coverage lives in test_eskf_output_length_matches_input,
    // re-enabled when the vertical-channel divergence fix (#286) landed.

    println!("\n✅ All filters produce output length matching input length: {input_length}");
}
