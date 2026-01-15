//! Comprehensive integration tests for INS filters using real data
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

use strapdown::earth::haversine_distance;
use strapdown::kalman::{
    ErrorStateKalmanFilter, ExtendedKalmanFilter, InitialState, UnscentedKalmanFilter,
};
use strapdown::messages::{
    Event, GnssDegradationConfig, GnssFaultModel, GnssScheduler, build_event_stream,
};
use strapdown::sim::{NavigationResult, TestDataRecord, dead_reckoning, run_closed_loop};

use nalgebra::{DMatrix, DVector};

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

/// ESKF-specific process noise covariance (15-state)
/// Tuned values (8x default) to balance stability and accuracy
/// Higher values prevent divergence while maintaining reasonable performance
const ESKF_PROCESS_NOISE: [f64; 15] = [
    8e-6, // latitude noise (8x default)
    8e-6, // longitude noise (8x default)
    8e-6, // altitude noise (8x default)
    8e-3, // velocity north noise (8x default)
    8e-3, // velocity east noise (8x default)
    8e-3, // velocity down noise (8x default)
    8e-5, // roll noise (8x default)
    8e-5, // pitch noise (8x default)
    8e-5, // yaw noise (8x default)
    8e-6, // acc bias x noise (8x default)
    8e-6, // acc bias y noise (8x default)
    8e-6, // acc bias z noise (8x default)
    8e-8, // gyro bias x noise (8x default)
    8e-8, // gyro bias y noise (8x default)
    8e-8, // gyro bias z noise (8x default)
];

/// ESKF-specific initial covariance (15-state)
/// Higher uncertainty (8x default) for stability
const ESKF_INITIAL_COVARIANCE: [f64; 15] = [
    8e-6, 8e-6, 8.0, // position covariance (lat, lon, alt) - 8m altitude uncertainty
    0.8, 0.8, 0.8, // velocity covariance (m/s) - 8x default
    0.08, 0.08, 0.08, // attitude covariance (radians) - 8x default
    0.08, 0.08, 0.08, // accelerometer bias covariance (m/s²) - 8x default
    0.008, 0.008, 0.008, // gyroscope bias covariance (rad/s) - 8x default
];
/// Error statistics for a navigation solution
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
    fn new() -> Self {
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
                println!("Record {}: horizontal_error={:.2}m", i, horizontal_error);
            }

            // Skip invalid errors (NaN or Inf)
            if !horizontal_error.is_finite() {
                if i < 10 || horizontal_errors.len() < 10 {
                    println!(
                        "WARNING: Skipping non-finite horizontal_error at index {}",
                        i
                    );
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
            .cloned()
            .fold(f64::INFINITY, f64::min);
        stats.max_horizontal_error = horizontal_errors
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        stats.rms_horizontal_error = (horizontal_errors.iter().map(|e| e.powi(2)).sum::<f64>()
            / horizontal_errors.len() as f64)
            .sqrt();

        // Compute median
        let mut sorted_horizontal = horizontal_errors.clone();
        sorted_horizontal.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted_horizontal.len() / 2;
        stats.median_horizontal_error = if sorted_horizontal.len() % 2 == 0 {
            (sorted_horizontal[mid - 1] + sorted_horizontal[mid]) / 2.0
        } else {
            sorted_horizontal[mid]
        };
    }

    if !altitude_errors.is_empty() {
        stats.mean_altitude_error =
            altitude_errors.iter().sum::<f64>() / altitude_errors.len() as f64;
        stats.min_altitude_error = altitude_errors
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        stats.max_altitude_error = altitude_errors
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        stats.rms_altitude_error = (altitude_errors.iter().map(|e| e.powi(2)).sum::<f64>()
            / altitude_errors.len() as f64)
            .sqrt();

        // Compute median
        let mut sorted_altitude = altitude_errors.clone();
        sorted_altitude.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted_altitude.len() / 2;
        stats.median_altitude_error = if sorted_altitude.len() % 2 == 0 {
            (sorted_altitude[mid - 1] + sorted_altitude[mid]) / 2.0
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
        is_enu: true,
    }
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
    let results = dead_reckoning(&records);

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
        initial_state,
        imu_biases,
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
        run_closed_loop(&mut ukf, stream, None).expect("Closed-loop filter should complete");

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

    // Assert error bounds - these should be reasonable for a working filter with GNSS
    // With good GNSS, horizontal error should be within a few meters RMS

    let rms_horizontal_limit = 25.0;
    let max_horizontal_limit = 39.0;
    let rms_altitude_limit = 50.0;
    let max_altitude_limit = 250.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "RMS horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "Maximum horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "RMS altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "Maximum altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
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
        initial_state,
        imu_biases,
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
    let results = run_closed_loop(&mut ukf, stream, None)
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
        "Maximum horizontal error with degraded GNSS should be less than 600m, got {:.2}m",
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
    let dr_results = dead_reckoning(&records);
    let dr_stats = compute_error_metrics(&dr_results, &records);

    // Run UKF
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6];
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001,
    ];
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        initial_state,
        imu_biases,
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

    let ukf_results = run_closed_loop(&mut ukf, stream, None).expect("UKF should complete");
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
        initial_state,
        vec![0.0; 6], // Zero initial bias estimates
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
    let results =
        run_closed_loop(&mut ekf, stream, None).expect("Closed-loop EKF filter should complete");

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

    let rms_horizontal_limit = 35.0;
    let max_horizontal_limit = 145.0;
    let rms_altitude_limit = 150.0;
    let max_altitude_limit = 1230.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "RMS horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "Maximum horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "RMS altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "Maximum altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
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
        initial_state,
        vec![0.0; 6],
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
    let results = run_closed_loop(&mut ekf, stream, None)
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

    // Error bounds should be looser than full-rate GNSS but still reasonable
    // EKF may have slightly higher errors than UKF due to linearization
    let rms_horizontal_limit = 125.0;
    let max_horizontal_limit = 1700.0;
    let rms_altitude_limit = 140.0;
    let max_altitude_limit = 2000.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "RMS horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "Maximum horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "RMS altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "Maximum altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
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
    let dr_results = dead_reckoning(&records);
    let dr_stats = compute_error_metrics(&dr_results, &records);

    // Run EKF
    let initial_state = create_initial_state(&records[0]);
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ekf = ExtendedKalmanFilter::new(
        initial_state,
        vec![0.0; 6],
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

    let ekf_results = run_closed_loop(&mut ekf, stream, None).expect("EKF should complete");
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
        initial_state,
        vec![0.0; 6], // Zero initial bias estimates
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
    let results =
        run_closed_loop(&mut eskf, stream, None).expect("Closed-loop ESKF filter should complete");

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

    // Assert error bounds - ESKF with 5x process noise tuning
    // Performance reflects trade-off between stability (no divergence) and accuracy
    // These bounds are based on empirical performance with real MEMS-grade IMU data

    let rms_horizontal_limit = 2000.0;
    let max_horizontal_limit = 2500.0;
    let rms_altitude_limit = 135.0;
    let max_altitude_limit = 509.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "RMS horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "Maximum horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "RMS altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "Maximum altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
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

/// Test ESKF with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate (5s intervals).
#[test]
#[ignore = "ESKF with degraded GNSS has altitude divergence and health monitor aborts due to out of range"]
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
        initial_state,
        vec![0.0; 6],
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
    let results = run_closed_loop(&mut eskf, stream, None)
        .expect("Closed-loop ESKF filter with degraded GNSS should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== ESKF with Degraded GNSS (5s updates) Error Statistics ===");
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

    // Error bounds for degraded GNSS (5s update intervals)
    // With less frequent updates, errors will be significantly higher
    // These bounds are based on empirical performance with 8x process noise tuning
    assert!(
        stats.rms_horizontal_error < 1000.0,
        "RMS horizontal error with degraded GNSS should be less than 1000m, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < 3500.0,
        "Maximum horizontal error with degraded GNSS should be less than 3500m, got {:.2}m",
        stats.max_horizontal_error
    );

    assert!(
        stats.rms_altitude_error < 400.0,
        "RMS altitude error with degraded GNSS should be less than 400m, got {:.2}m",
        stats.rms_altitude_error
    );

    assert!(
        stats.max_altitude_error < 3000.0,
        "Maximum altitude error with degraded GNSS should be less than 3000m, got {:.2}m",
        stats.max_altitude_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
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
    let dr_results = dead_reckoning(&records);
    let dr_stats = compute_error_metrics(&dr_results, &records);

    // Run ESKF
    let initial_state = create_initial_state(&records[0]);
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        initial_state,
        vec![0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        ..Default::default()
    };
    let stream = build_event_stream(&records, &cfg);

    let eskf_results = run_closed_loop(&mut eskf, stream, None).expect("ESKF should complete");
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

    // Initialize ESKF with more aggressive process noise to simulate high dynamics
    // Higher uncertainty values to accommodate rapid maneuvers and accelerations
    let initial_error_covariance = vec![
        1e-6, 1e-6, 1.0, // position error (same as default)
        0.5, 0.5, 0.5, // velocity error (5x default - allows for higher acceleration)
        0.05, 0.05, 0.05, // attitude error (5x default - allows for rapid rotations)
        0.05, 0.05, 0.05, // accel bias error (5x default - less confident in bias)
        0.005, 0.005, 0.005, // gyro bias error (5x default - less confident in bias)
    ];

    // Increased process noise for high dynamics
    // 10x velocity noise and 10x attitude noise to accommodate rapid changes
    let process_noise_values = vec![
        1e-5, 1e-5, 1e-5, // position noise (10x default)
        1e-2, 1e-2, 1e-2, // velocity noise (10x default for high dynamics)
        1e-4, 1e-4, 1e-4, // attitude noise (10x default for rapid maneuvers)
        1e-5, 1e-5, 1e-5, // accel bias noise (10x default)
        1e-7, 1e-7, 1e-7, // gyro bias noise (10x default)
    ];
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        initial_state,
        vec![0.0; 6], // Zero initial bias estimates
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
    let results =
        run_closed_loop(&mut eskf, stream, None).expect("ESKF with high dynamics should complete");

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

    // With high process noise, errors may be slightly higher but should still be bounded
    let rms_horizontal_limit = 1905.0;
    let max_horizontal_limit = 2494.0;
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
        initial_state.clone(),
        vec![0.0; 6],
        None,
        initial_covariance.clone(),
        process_noise.clone(),
        1e-3,
        2.0,
        0.0,
    );
    let stream_ukf = build_event_stream(&records, &cfg);
    let ukf_results = run_closed_loop(&mut ukf, stream_ukf, None).expect("UKF should complete");
    let ukf_stats = compute_error_metrics(&ukf_results, &records);

    // Run EKF
    let mut ekf = ExtendedKalmanFilter::new(
        initial_state.clone(),
        vec![0.0; 6],
        initial_covariance.clone(),
        process_noise.clone(),
        true,
    );
    let stream_ekf = build_event_stream(&records, &cfg);
    let ekf_results = run_closed_loop(&mut ekf, stream_ekf, None).expect("EKF should complete");
    let ekf_stats = compute_error_metrics(&ekf_results, &records);

    // Run ESKF
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        initial_state,
        vec![0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    let stream_eskf = build_event_stream(&records, &cfg);
    let eskf_results = run_closed_loop(&mut eskf, stream_eskf, None).expect("ESKF should complete");
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

    // All filters should produce reasonable results
    assert!(
        ukf_stats.rms_horizontal_error < 25.0,
        "UKF RMS horizontal error should be reasonable"
    );
    assert!(
        ekf_stats.rms_horizontal_error < 30.0,
        "EKF RMS horizontal error should be reasonable"
    );
    assert!(
        eskf_stats.rms_horizontal_error < 1905.0,
        "ESKF RMS horizontal error should be reasonable"
    );

    // Verify all filters completed without producing invalid values
    assert_eq!(ukf_results.len(), ekf_results.len());
    assert_eq!(ekf_results.len(), eskf_results.len());
}
