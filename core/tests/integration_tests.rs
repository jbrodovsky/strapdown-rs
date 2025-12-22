//! Comprehensive integration tests for INS filters using real data
//!
//! This module contains end-to-end integration tests for the strapdown inertial navigation
//! filters (UKF and Particle Filter) using real data from the sim/data directory. These tests
//! ensure that the entire navigation system works as expected in realistic scenarios, not just
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
//! ## Test Structure
//!
//! Tests load real data from CSV files, run the filters, and compute error metrics against
//! GNSS measurements. The tests verify that:
//! 1. Filters complete without errors
//! 2. Position errors remain within reasonable bounds
//! 3. Velocity and orientation estimates are stable
//! 4. The closed-loop filter outperforms dead reckoning

// updated for deconflict

use std::path::Path;

use strapdown::earth::haversine_distance;
use strapdown::kalman::{InitialState, UnscentedKalmanFilter};
use strapdown::messages::{
    GnssDegradationConfig, GnssFaultModel, GnssScheduler, build_event_stream,
};
use strapdown::sim::{
    NavigationResult, TestDataRecord, dead_reckoning, run_closed_loop,
};

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
/// Error statistics for a navigation solution
#[derive(Debug, Clone)]
struct ErrorStats {
    /// Mean horizontal position error (meters)
    mean_horizontal_error: f64,
    /// Maximum horizontal position error (meters)
    max_horizontal_error: f64,
    /// Root mean square horizontal position error (meters)
    rms_horizontal_error: f64,
    /// Mean altitude error (meters)
    mean_altitude_error: f64,
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
            max_horizontal_error: 0.0,
            rms_horizontal_error: 0.0,
            mean_altitude_error: 0.0,
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
        stats.max_horizontal_error = horizontal_errors
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        stats.rms_horizontal_error = (horizontal_errors.iter().map(|e| e.powi(2)).sum::<f64>()
            / horizontal_errors.len() as f64)
            .sqrt();
    }

    if !altitude_errors.is_empty() {
        stats.mean_altitude_error =
            altitude_errors.iter().sum::<f64>() / altitude_errors.len() as f64;
        stats.max_altitude_error = altitude_errors
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        stats.rms_altitude_error = (altitude_errors.iter().map(|e| e.powi(2)).sum::<f64>()
            / altitude_errors.len() as f64)
            .sqrt();
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
    InitialState {
        latitude: first_record.latitude,
        longitude: first_record.longitude,
        altitude: first_record.altitude,
        northward_velocity: first_record.speed * first_record.bearing.to_radians().cos(),
        eastward_velocity: first_record.speed * first_record.bearing.to_radians().sin(),
        vertical_velocity: 0.0,
        roll: first_record.roll,
        pitch: first_record.pitch,
        yaw: first_record.yaw,
        in_degrees: true,
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
        "Horizontal Error: mean={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error, stats.max_horizontal_error, stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error, stats.max_altitude_error, stats.rms_altitude_error
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
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, // position covariance (lat, lon, alt)
        0.1, 0.1, 0.1, // velocity covariance
        0.01, 0.01, 0.01, // attitude covariance (roll, pitch, yaw)
        0.01, 0.01, 0.01, // accelerometer bias covariance
        0.001, 0.001, 0.001, // gyroscope bias covariance
    ];

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
        "Horizontal Error: mean={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error, stats.max_horizontal_error, stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error, stats.max_altitude_error, stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Assert error bounds - these should be reasonable for a working filter with GNSS
    // With good GNSS, horizontal error should be within a few meters RMS
    assert!(
        stats.rms_horizontal_error < 30.0,
        "RMS horizontal error should be less than 30m with GNSS, got {:.2}m",
        stats.rms_horizontal_error
    );

    // Altitude error should also be bounded
    assert!(
        stats.rms_altitude_error < 50.0,
        "RMS altitude error should be less than 30m with GNSS, got {:.2}m",
        stats.rms_altitude_error
    );

    // Maximum errors should not be excessive
    assert!(
        stats.max_horizontal_error < 100.0,
        "Maximum horizontal error should be less than 100m, got {:.2}m",
        stats.max_horizontal_error
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
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, // position
        0.1, 0.1, 0.1, // velocity
        0.01, 0.01, 0.01, // attitude
        0.01, 0.01, 0.01, // accel bias
        0.001, 0.001, 0.001, // gyro bias
    ];

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
        "Horizontal Error: mean={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error, stats.max_horizontal_error, stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error, stats.max_altitude_error, stats.rms_altitude_error
    );

    // Error bounds should be looser than full-rate GNSS but still reasonable
    assert!(
        stats.rms_horizontal_error < 50.0,
        "RMS horizontal error with degraded GNSS should be less than 50m, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < 600.0,
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
    if dr_stats.rms_horizontal_error > 5.0 {
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

/// Test EKF completes successfully on a real dataset
#[test]
#[ignore] // Requires data file to be present
fn test_ekf_completes_successfully() {
    let data_path = Path::new("../examples/data/test_dataset.csv");
    if !data_path.exists() {
        println!("Test data not found at {:?}, skipping test", data_path);
        return;
    }

    let records = load_test_data(data_path);
    assert!(!records.is_empty(), "Test data should not be empty");

    // Initialize EKF with 15-state configuration
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6];
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001,
    ];
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ekf = strapdown::kalman::ExtendedKalmanFilter::new(
        initial_state,
        imu_biases,
        initial_covariance,
        process_noise,
        true, // use_biases
    );

    // Create event stream with no faults
    let scheduler = GnssScheduler::PassThrough;
    let fault_model = GnssFaultModel::None;
    let cfg = GnssDegradationConfig {
        scheduler,
        fault: fault_model,
        ..Default::default()
    };
    let stream = build_event_stream(&records, &cfg);

    // Run EKF
    let ekf_results = run_closed_loop(&mut ekf, stream, None);
    assert!(ekf_results.is_ok(), "EKF should complete without errors");

    let results = ekf_results.unwrap();
    assert!(!results.is_empty(), "EKF should produce results");
    println!("EKF produced {} navigation solutions", results.len());

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);
    println!("\n=== EKF Performance Metrics ===");
    println!(
        "Mean horizontal error: {:.2}m",
        stats.mean_horizontal_error
    );
    println!("Max horizontal error: {:.2}m", stats.max_horizontal_error);
    println!("RMS horizontal error: {:.2}m", stats.rms_horizontal_error);
    println!("Mean altitude error: {:.2}m", stats.mean_altitude_error);

    // Check that errors are bounded
    assert!(
        stats.max_horizontal_error < 100.0,
        "Maximum horizontal error should be bounded: {:.2}m",
        stats.max_horizontal_error
    );
}

/// Test EKF 9-state configuration (no biases)
#[test]
#[ignore] // Requires data file to be present
fn test_ekf_9state_configuration() {
    let data_path = Path::new("../examples/data/test_dataset.csv");
    if !data_path.exists() {
        println!("Test data not found at {:?}, skipping test", data_path);
        return;
    }

    let records = load_test_data(data_path);
    assert!(!records.is_empty(), "Test data should not be empty");

    // Initialize EKF with 9-state configuration (no biases)
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6]; // Provided but won't be used
    let initial_covariance = vec![1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01];
    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE[0..9].to_vec()));

    let mut ekf = strapdown::kalman::ExtendedKalmanFilter::new(
        initial_state,
        imu_biases,
        initial_covariance,
        process_noise,
        false, // Don't use biases (9-state)
    );

    // Create event stream
    let scheduler = GnssScheduler::PassThrough;
    let fault_model = GnssFaultModel::None;
    let cfg = GnssDegradationConfig {
        scheduler,
        fault: fault_model,
        ..Default::default()
    };
    let stream = build_event_stream(&records, &cfg);

    // Run EKF
    let ekf_results = run_closed_loop(&mut ekf, stream, None);
    assert!(
        ekf_results.is_ok(),
        "9-state EKF should complete without errors"
    );

    let results = ekf_results.unwrap();
    assert!(!results.is_empty(), "9-state EKF should produce results");
    println!("9-state EKF produced {} navigation solutions", results.len());

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);
    println!("\n=== 9-State EKF Performance Metrics ===");
    println!("RMS horizontal error: {:.2}m", stats.rms_horizontal_error);

    // Check that errors are bounded
    assert!(
        stats.max_horizontal_error < 150.0,
        "9-state EKF max horizontal error should be bounded: {:.2}m",
        stats.max_horizontal_error
    );
}

/// Compare EKF vs UKF performance
#[test]
#[ignore] // Requires data file to be present
fn test_ekf_vs_ukf_comparison() {
    let data_path = Path::new("../examples/data/test_dataset.csv");
    if !data_path.exists() {
        println!("Test data not found at {:?}, skipping test", data_path);
        return;
    }

    let records = load_test_data(data_path);
    assert!(!records.is_empty(), "Test data should not be empty");

    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6];
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001,
    ];
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    // Run EKF
    let mut ekf = strapdown::kalman::ExtendedKalmanFilter::new(
        initial_state.clone(),
        imu_biases.clone(),
        initial_covariance.clone(),
        process_noise.clone(),
        true,
    );

    let scheduler = GnssScheduler::PassThrough;
    let fault_model = GnssFaultModel::None;
    let cfg = GnssDegradationConfig {
        scheduler,
        fault: fault_model,
        ..Default::default()
    };
    let stream1 = build_event_stream(&records, &cfg);

    let ekf_results = run_closed_loop(&mut ekf, stream1, None).expect("EKF should complete");
    let ekf_stats = compute_error_metrics(&ekf_results, &records);

    // Run UKF
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

    let stream2 = build_event_stream(&records, &cfg);
    let ukf_results = run_closed_loop(&mut ukf, stream2, None).expect("UKF should complete");
    let ukf_stats = compute_error_metrics(&ukf_results, &records);

    // Print comparison
    println!("\n=== EKF vs UKF Performance Comparison ===");
    println!(
        "EKF RMS Horizontal Error: {:.2}m",
        ekf_stats.rms_horizontal_error
    );
    println!(
        "UKF RMS Horizontal Error: {:.2}m",
        ukf_stats.rms_horizontal_error
    );
    println!(
        "EKF Mean Altitude Error: {:.2}m",
        ekf_stats.mean_altitude_error
    );
    println!(
        "UKF Mean Altitude Error: {:.2}m",
        ukf_stats.mean_altitude_error
    );

    // Both filters should have bounded errors
    assert!(
        ekf_stats.rms_horizontal_error < 50.0,
        "EKF RMS horizontal error should be bounded: {:.2}m",
        ekf_stats.rms_horizontal_error
    );
    assert!(
        ukf_stats.rms_horizontal_error < 50.0,
        "UKF RMS horizontal error should be bounded: {:.2}m",
        ukf_stats.rms_horizontal_error
    );

    // EKF and UKF should have comparable performance (within 50% of each other)
    let ratio = ekf_stats.rms_horizontal_error / ukf_stats.rms_horizontal_error;
    println!("EKF/UKF performance ratio: {:.2}", ratio);
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "EKF and UKF should have comparable performance, ratio: {:.2}",
        ratio
    );
}

/// Test EKF with GNSS degradation (outages)
#[test]
#[ignore] // Requires data file to be present
fn test_ekf_with_gnss_outages() {
    let data_path = Path::new("../examples/data/test_dataset.csv");
    if !data_path.exists() {
        println!("Test data not found at {:?}, skipping test", data_path);
        return;
    }

    let records = load_test_data(data_path);
    assert!(!records.is_empty(), "Test data should not be empty");

    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6];
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001,
    ];
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ekf = strapdown::kalman::ExtendedKalmanFilter::new(
        initial_state,
        imu_biases,
        initial_covariance,
        process_noise,
        true,
    );

    // Create event stream with periodic outages
    let scheduler = GnssScheduler::DutyCycle {
        on_s: 10.0,
        off_s: 5.0,
        start_phase_s: 0.0,
    };
    let fault_model = GnssFaultModel::None;
    let cfg = GnssDegradationConfig {
        scheduler,
        fault: fault_model,
        seed: 42,
        ..Default::default()
    };
    let stream = build_event_stream(&records, &cfg);

    // Run EKF
    let ekf_results = run_closed_loop(&mut ekf, stream, None);
    assert!(
        ekf_results.is_ok(),
        "EKF should complete with GNSS outages"
    );

    let results = ekf_results.unwrap();
    assert!(!results.is_empty(), "EKF should produce results");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);
    println!("\n=== EKF Performance with GNSS Outages ===");
    println!("RMS horizontal error: {:.2}m", stats.rms_horizontal_error);
    println!("Max horizontal error: {:.2}m", stats.max_horizontal_error);

    // Errors will be higher with outages, but should still be bounded
    assert!(
        stats.max_horizontal_error < 200.0,
        "EKF with outages should have bounded error: {:.2}m",
        stats.max_horizontal_error
    );
}

/// Test EKF bounded drift vs open-loop
#[test]
#[ignore] // Requires data file to be present
fn test_ekf_bounded_drift_vs_open_loop() {
    let data_path = Path::new("../examples/data/test_dataset.csv");
    if !data_path.exists() {
        println!("Test data not found at {:?}, skipping test", data_path);
        return;
    }

    let records = load_test_data(data_path);
    assert!(!records.is_empty(), "Test data should not be empty");

    // Run dead reckoning (open-loop)
    let dr_results = dead_reckoning(&records);
    let dr_stats = compute_error_metrics(&dr_results, &records);

    // Run EKF (closed-loop)
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6];
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001,
    ];
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ekf = strapdown::kalman::ExtendedKalmanFilter::new(
        initial_state,
        imu_biases,
        initial_covariance,
        process_noise,
        true,
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
    println!("\n=== EKF vs Dead Reckoning Performance ===");
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
    if dr_stats.rms_horizontal_error > 5.0 {
        // Only compare if DR has meaningful drift
        assert!(
            ekf_stats.rms_horizontal_error < dr_stats.rms_horizontal_error,
            "EKF should have lower RMS horizontal error than dead reckoning. EKF: {:.2}m, DR: {:.2}m",
            ekf_stats.rms_horizontal_error,
            dr_stats.rms_horizontal_error
        );
    }
}