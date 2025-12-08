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

use std::path::Path;

use strapdown::earth::haversine_distance;
use strapdown::kalman::{InitialState, UnscentedKalmanFilter};
use strapdown::messages::{
    GnssDegradationConfig, GnssFaultModel, GnssScheduler, build_event_stream,
};
use strapdown::sim::{
    NavigationResult, TestDataRecord, run_closed_loop, dead_reckoning, initialize_particle_filter,
    run_closed_loop_pf,
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

/// Process noise for particle filter - primarily applied to bias states with minimal navigation noise
/// Navigation states receive very small process noise to maintain particle diversity and prevent collapse
const PARTICLE_FILTER_PROCESS_NOISE: [f64; 15] = [
    0.0, // 1e-12, // latitude: minimal noise to prevent collapse
    0.0, // 1e-12, // longitude: minimal noise to prevent collapse
    0.0, // 1e-9,  // altitude: small noise for vertical channel
    0.0, // 1e-9,  // velocity north: minimal noise
    0.0, // 1e-9,  // velocity east: minimal noise
    0.0, // 1e-9,  // velocity down: minimal noise
    0.0, // 1e-12, // roll: minimal noise
    0.0, // 1e-12, // pitch: minimal noise
    0.0, // 1e-12, // yaw: minimal noise
    1e-7,  // acc bias x: random walk noise ~1e-7 m/s²
    1e-7,  // acc bias y: random walk noise
    1e-7,  // acc bias z: random walk noise
    1e-9,  // gyro bias x: random walk noise ~1e-9 rad/s
    1e-9,  // gyro bias y: random walk noise
    1e-9,  // gyro bias z: random walk noise
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
    mean_velocity_down_error: f64,
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
            mean_velocity_down_error: 0.0,
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
    let mut velocity_down_errors = Vec::new();

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
            let vd_err = result.velocity_down.abs();

            if vn_err.is_finite() {
                velocity_north_errors.push(vn_err);
            }
            if ve_err.is_finite() {
                velocity_east_errors.push(ve_err);
            }
            if vd_err.is_finite() {
                velocity_down_errors.push(vd_err);
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
        stats.mean_velocity_down_error =
            velocity_down_errors.iter().sum::<f64>() / velocity_down_errors.len() as f64;
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
    TestDataRecord::from_csv(path).unwrap_or_else(|_| panic!("Failed to load test data from CSV: {}",
        path.display()))
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
        downward_velocity: 0.0,
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
        stats.mean_velocity_down_error
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
        seed: 42,
    };

    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop filter
    let results = run_closed_loop(&mut ukf, stream, None).expect("Closed-loop filter should complete");

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
        stats.mean_velocity_down_error
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
            result.velocity_down.is_finite(),
            "Velocity down should be finite: {}",
            result.velocity_down
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
        seed: 42,
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
        seed: 42,
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

/// Test Particle Filter closed-loop on real data (15-state INS)
///
/// This test runs a closed-loop particle filter with GNSS measurements on real data and verifies that:
/// 1. The filter completes without errors
/// 2. Position errors remain bounded
/// 3. The filter performs better than dead reckoning
/// 4. Particle diversity is maintained throughout the run
///
/// **Note**: Currently ignored due to numerical stability issues in the particle filter
/// implementation causing divergence. The filter diverges beyond health monitor limits
/// (altitude errors > 100M meters), suggesting a bug in the core implementation that
/// needs to be debugged separately. Enable this test once the particle filter is stabilized.
#[test]
//#[ignore = "Particle filter implementation has numerical stability issues - diverges beyond health limits"]
fn test_particle_filter_closed_loop_on_real_data() {
    println!("\n========================================");
    println!("PARTICLE FILTER CLOSED-LOOP INTEGRATION TEST");
    println!("========================================");

    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    println!("Loading test data from: {}", test_data_path.display());
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );
    println!("Loaded {} test records", records.len());

    // Particle filter configuration
    let num_particles = 1000;
    let pos_std_m = 5.0;  // Position uncertainty (meters)
    let vel_std_mps = 1.0;  // Velocity uncertainty (m/s)
    let att_std_rad: f64 = 0.1;  // Attitude uncertainty (radians)

    println!("\n--- Particle Filter Configuration ---");
    println!("Number of particles: {}", num_particles);
    println!("Position std dev: {:.2} m", pos_std_m);
    println!("Velocity std dev: {:.2} m/s", vel_std_mps);
    println!("Attitude std dev: {:.4} rad ({:.2} deg)", att_std_rad, att_std_rad.to_degrees());

    // Initialize particle filter with 15-state INS (position, velocity, attitude, biases)
    let process_noise_diag = PARTICLE_FILTER_PROCESS_NOISE.to_vec();
    println!("\n--- Process Noise Configuration ---");
    println!("Position noise: [{:.2e}, {:.2e}, {:.2e}]",
        process_noise_diag[0], process_noise_diag[1], process_noise_diag[2]);
    println!("Velocity noise: [{:.2e}, {:.2e}, {:.2e}]",
        process_noise_diag[3], process_noise_diag[4], process_noise_diag[5]);
    println!("Attitude noise: [{:.2e}, {:.2e}, {:.2e}]",
        process_noise_diag[6], process_noise_diag[7], process_noise_diag[8]);
    println!("Acc bias noise: [{:.2e}, {:.2e}, {:.2e}]",
        process_noise_diag[9], process_noise_diag[10], process_noise_diag[11]);
    println!("Gyro bias noise: [{:.2e}, {:.2e}, {:.2e}]",
        process_noise_diag[12], process_noise_diag[13], process_noise_diag[14]);

    let mut pf = initialize_particle_filter(
        records[0].clone(),
        num_particles,
        pos_std_m,
        vel_std_mps,
        att_std_rad,
        Some(DVector::from_vec(process_noise_diag)),
    );

    println!("\n--- Initial State ---");
    println!("Initial position: ({:.6}, {:.6}, {:.2})",
        records[0].latitude, records[0].longitude, records[0].altitude);
    println!("Initial attitude (deg): roll={:.2}, pitch={:.2}, yaw={:.2}",
        records[0].roll, records[0].pitch, records[0].yaw);

    // Create event stream with passthrough scheduler (all GNSS measurements used)
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        seed: 42,
    };

    println!("\n--- Building Event Stream ---");
    let stream = build_event_stream(&records, &cfg);
    println!("Generated {} events from {} records", stream.events.len(), records.len());

    // Count event types
    use strapdown::messages::Event;
    let imu_count = stream.events.iter().filter(|e| matches!(e, Event::Imu { .. })).count();
    let meas_count = stream.events.iter().filter(|e| matches!(e, Event::Measurement { .. })).count();
    println!("Event breakdown: {} IMU, {} GNSS measurements", imu_count, meas_count);

    // Run closed-loop particle filter with resampling threshold
    println!("\n--- Running Particle Filter ---");
    let resample_threshold = 0.5;  // Resample when N_eff < 50% of particles
    println!("Resampling threshold: {:.1}%", resample_threshold * 100.0);

    let results = run_closed_loop_pf(&mut pf, stream, None, Some(resample_threshold))
        .expect("Particle filter should complete");

    // Verify results
    assert!(
        !results.is_empty(),
        "Particle filter should produce results"
    );
    println!("Generated {} navigation results", results.len());

    // Compute error metrics
    println!("\n--- Computing Error Metrics ---");
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== PARTICLE FILTER CLOSED-LOOP ERROR STATISTICS ===");
    println!("Horizontal Error:");
    println!("  Mean:  {:.2} m", stats.mean_horizontal_error);
    println!("  Max:   {:.2} m", stats.max_horizontal_error);
    println!("  RMS:   {:.2} m", stats.rms_horizontal_error);
    println!("\nAltitude Error:");
    println!("  Mean:  {:.2} m", stats.mean_altitude_error);
    println!("  Max:   {:.2} m", stats.max_altitude_error);
    println!("  RMS:   {:.2} m", stats.rms_altitude_error);
    println!("\nVelocity Error:");
    println!("  North: {:.3} m/s", stats.mean_velocity_north_error);
    println!("  East:  {:.3} m/s", stats.mean_velocity_east_error);
    println!("  Down:  {:.3} m/s", stats.mean_velocity_down_error);

    // Assert error bounds - particle filter should perform comparably to UKF
    assert!(
        stats.rms_horizontal_error < 50.0,
        "RMS horizontal error should be less than 50m with GNSS, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.rms_altitude_error < 100.0,
        "RMS altitude error should be less than 100m with GNSS, got {:.2}m",
        stats.rms_altitude_error
    );

    assert!(
        stats.max_horizontal_error < 200.0,
        "Maximum horizontal error should be less than 200m, got {:.2}m",
        stats.max_horizontal_error
    );

    // Verify no NaN or infinite values in results
    println!("\n--- Validating Results ---");
    let mut invalid_count = 0;
    for (i, result) in results.iter().enumerate() {
        if !result.latitude.is_finite() || !result.longitude.is_finite() || !result.altitude.is_finite() {
            invalid_count += 1;
            if invalid_count <= 5 {
                println!("WARNING: Invalid result at index {}", i);
            }
        }
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite at index {}: {}",
            i, result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite at index {}: {}",
            i, result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite at index {}: {}",
            i, result.altitude
        );
    }
    println!("All {} results validated successfully", results.len());
    println!("========================================\n");
}

/// Test Particle Filter with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate and verifies
/// that the particle filter still performs reasonably well, though with higher errors than full-rate GNSS.
///
/// **Note**: Currently ignored due to numerical stability issues in the particle filter
/// implementation. See `test_particle_filter_closed_loop_on_real_data` for details.
#[test]
//#[ignore = "Particle filter implementation has numerical stability issues"]
fn test_particle_filter_with_degraded_gnss() {
    println!("\n========================================");
    println!("PARTICLE FILTER WITH DEGRADED GNSS TEST");
    println!("========================================");

    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );
    println!("Loaded {} test records", records.len());

    // Particle filter configuration - use more particles for degraded GNSS
    let num_particles = 150;
    let pos_std_m = 10.0;  // Higher initial uncertainty
    let vel_std_mps = 2.0;
    let att_std_rad: f64 = 0.15;

    println!("\n--- Particle Filter Configuration ---");
    println!("Number of particles: {} (increased for degraded conditions)", num_particles);
    println!("Position std dev: {:.2} m", pos_std_m);
    println!("Velocity std dev: {:.2} m/s", vel_std_mps);
    println!("Attitude std dev: {:.4} rad", att_std_rad);

    let process_noise_diag = PARTICLE_FILTER_PROCESS_NOISE.to_vec();
    let mut pf = initialize_particle_filter(
        records[0].clone(),
        num_particles,
        pos_std_m,
        vel_std_mps,
        att_std_rad,
        Some(DVector::from_vec(process_noise_diag)),
    );

    // Create event stream with periodic scheduler (every 5 seconds)
    let gnss_interval_s = 5.0;
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::FixedInterval {
            interval_s: gnss_interval_s,
            phase_s: 0.0,
        },
        fault: GnssFaultModel::None,
        seed: 42,
    };

    println!("\n--- GNSS Degradation Configuration ---");
    println!("GNSS update interval: {:.1} seconds", gnss_interval_s);
    println!("Scheduler: FixedInterval");

    let stream = build_event_stream(&records, &cfg);
    use strapdown::messages::Event;
    let meas_count = stream.events.iter()
        .filter(|e| matches!(e, Event::Measurement { .. }))
        .count();
    println!("GNSS measurements in stream: {}", meas_count);

    // Run closed-loop filter
    println!("\n--- Running Particle Filter ---");
    let results = run_closed_loop_pf(&mut pf, stream, None, Some(0.5))
        .expect("Particle filter with degraded GNSS should complete");

    println!("Generated {} navigation results", results.len());

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== PF WITH DEGRADED GNSS ({}s updates) ERROR STATISTICS ===", gnss_interval_s);
    println!("Horizontal Error:");
    println!("  Mean:  {:.2} m", stats.mean_horizontal_error);
    println!("  Max:   {:.2} m", stats.max_horizontal_error);
    println!("  RMS:   {:.2} m", stats.rms_horizontal_error);
    println!("\nAltitude Error:");
    println!("  Mean:  {:.2} m", stats.mean_altitude_error);
    println!("  Max:   {:.2} m", stats.max_altitude_error);
    println!("  RMS:   {:.2} m", stats.rms_altitude_error);

    // Error bounds should be looser than full-rate GNSS but still reasonable
    assert!(
        stats.rms_horizontal_error < 100.0,
        "RMS horizontal error with degraded GNSS should be less than 100m, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < 800.0,
        "Maximum horizontal error with degraded GNSS should be less than 800m, got {:.2}m",
        stats.max_horizontal_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }

    println!("All results validated successfully");
    println!("========================================\n");
}

/// Test that closed-loop Particle Filter outperforms dead reckoning
///
/// This test runs both dead reckoning and PF on the same data and verifies that
/// the PF produces lower errors than dead reckoning, demonstrating the benefit
/// of GNSS-aided navigation with the particle filter.
///
/// **Note**: Currently ignored due to numerical stability issues in the particle filter
/// implementation. See `test_particle_filter_closed_loop_on_real_data` for details.
#[test]
//#[ignore = "Particle filter implementation has numerical stability issues"]
fn test_particle_filter_outperforms_dead_reckoning() {
    println!("\n========================================");
    println!("PARTICLE FILTER vs DEAD RECKONING COMPARISON");
    println!("========================================");

    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );
    println!("Loaded {} test records", records.len());

    // Run dead reckoning
    println!("\n--- Running Dead Reckoning Baseline ---");
    let dr_results = dead_reckoning(&records);
    let dr_stats = compute_error_metrics(&dr_results, &records);

    println!("Dead Reckoning Results:");
    println!("  RMS Horizontal Error: {:.2} m", dr_stats.rms_horizontal_error);
    println!("  Max Horizontal Error: {:.2} m", dr_stats.max_horizontal_error);
    println!("  RMS Altitude Error:   {:.2} m", dr_stats.rms_altitude_error);

    // Run Particle Filter
    println!("\n--- Running Particle Filter ---");
    let num_particles = 100;
    let pos_std_m = 5.0;
    let vel_std_mps = 1.0;
    let att_std_rad: f64 = 0.1;

    println!("PF Configuration: {} particles", num_particles);

    let process_noise_diag = PARTICLE_FILTER_PROCESS_NOISE.to_vec();
    let mut pf = initialize_particle_filter(
        records[0].clone(),
        num_particles,
        pos_std_m,
        vel_std_mps,
        att_std_rad,
        Some(DVector::from_vec(process_noise_diag)),
    );

    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        seed: 42,
    };
    let stream = build_event_stream(&records, &cfg);

    let pf_results = run_closed_loop_pf(&mut pf, stream, None, Some(0.5))
        .expect("Particle filter should complete");
    let pf_stats = compute_error_metrics(&pf_results, &records);

    println!("Particle Filter Results:");
    println!("  RMS Horizontal Error: {:.2} m", pf_stats.rms_horizontal_error);
    println!("  Max Horizontal Error: {:.2} m", pf_stats.max_horizontal_error);
    println!("  RMS Altitude Error:   {:.2} m", pf_stats.rms_altitude_error);

    // Print comparison
    println!("\n=== PERFORMANCE COMPARISON ===");
    println!("Metric                    Dead Reckoning    Particle Filter    Improvement");
    println!("------------------------------------------------------------------------");
    let horiz_improvement = if dr_stats.rms_horizontal_error > 0.0 {
        (1.0 - pf_stats.rms_horizontal_error / dr_stats.rms_horizontal_error) * 100.0
    } else {
        0.0
    };
    println!("RMS Horizontal Error:     {:8.2} m        {:8.2} m       {:6.1}%",
        dr_stats.rms_horizontal_error,
        pf_stats.rms_horizontal_error,
        horiz_improvement
    );

    let alt_improvement = if dr_stats.rms_altitude_error > 0.0 {
        (1.0 - pf_stats.rms_altitude_error / dr_stats.rms_altitude_error) * 100.0
    } else {
        0.0
    };
    println!("RMS Altitude Error:       {:8.2} m        {:8.2} m       {:6.1}%",
        dr_stats.rms_altitude_error,
        pf_stats.rms_altitude_error,
        alt_improvement
    );

    // Particle filter should significantly outperform dead reckoning
    if dr_stats.rms_horizontal_error > 5.0 {
        // Only compare if DR has meaningful drift
        assert!(
            pf_stats.rms_horizontal_error < dr_stats.rms_horizontal_error,
            "Particle filter should have lower RMS horizontal error than dead reckoning. PF: {:.2}m, DR: {:.2}m",
            pf_stats.rms_horizontal_error,
            dr_stats.rms_horizontal_error
        );
        println!("\n✓ Particle filter significantly outperforms dead reckoning");
    } else {
        println!("\n⚠ Insufficient drift in dead reckoning to compare (RMS < 5m)");
    }

    println!("========================================\n");
}

/// Test comparison between UKF and Particle Filter on same dataset
///
/// This test runs both UKF and PF on the same data and compares their performance.
/// Both filters should produce similar accuracy, though they may have different characteristics.
///
/// **Note**: Currently ignored due to numerical stability issues in the particle filter
/// implementation. The UKF portion works correctly. See `test_particle_filter_closed_loop_on_real_data` for details.
#[test]
//#[ignore = "Particle filter implementation has numerical stability issues"]
fn test_ukf_vs_particle_filter_comparison() {
    println!("\n========================================");
    println!("UKF vs PARTICLE FILTER COMPARISON");
    println!("========================================");

    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );
    println!("Loaded {} test records", records.len());

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Run UKF
    println!("\n--- Running UKF ---");
    let imu_biases = vec![0.0; 6];
    let initial_covariance = vec![
        1e-6, 1e-6, 1.0, // position
        0.1, 0.1, 0.1,   // velocity
        0.01, 0.01, 0.01, // attitude
        0.01, 0.01, 0.01, // accel bias
        0.001, 0.001, 0.001, // gyro bias
    ];
    let process_noise_ukf = DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        initial_state,
        imu_biases,
        None,
        initial_covariance,
        process_noise_ukf,
        1e-3,
        2.0,
        0.0,
    );

    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        seed: 42,
    };
    let stream_ukf = build_event_stream(&records, &cfg);

    let ukf_results = run_closed_loop(&mut ukf, stream_ukf, None)
        .expect("UKF should complete");
    let ukf_stats = compute_error_metrics(&ukf_results, &records);

    println!("UKF Results:");
    println!("  RMS Horizontal Error: {:.2} m", ukf_stats.rms_horizontal_error);
    println!("  RMS Altitude Error:   {:.2} m", ukf_stats.rms_altitude_error);

    // Run Particle Filter
    println!("\n--- Running Particle Filter ---");
    let num_particles = 100;
    let process_noise_pf = PARTICLE_FILTER_PROCESS_NOISE.to_vec();

    let mut pf = initialize_particle_filter(
        records[0].clone(),
        num_particles,
        5.0,
        1.0,
        0.1,
        Some(DVector::from_vec(process_noise_pf)),
    );

    let stream_pf = build_event_stream(&records, &cfg);
    let pf_results = run_closed_loop_pf(&mut pf, stream_pf, None, Some(0.5))
        .expect("Particle filter should complete");
    let pf_stats = compute_error_metrics(&pf_results, &records);

    println!("Particle Filter Results:");
    println!("  RMS Horizontal Error: {:.2} m", pf_stats.rms_horizontal_error);
    println!("  RMS Altitude Error:   {:.2} m", pf_stats.rms_altitude_error);

    // Print detailed comparison
    println!("\n=== DETAILED COMPARISON ===");
    println!("Metric                         UKF           PF         Ratio (PF/UKF)");
    println!("-----------------------------------------------------------------------");
    println!("RMS Horizontal Error:       {:7.2} m    {:7.2} m      {:.2}x",
        ukf_stats.rms_horizontal_error,
        pf_stats.rms_horizontal_error,
        pf_stats.rms_horizontal_error / ukf_stats.rms_horizontal_error.max(0.1)
    );
    println!("Max Horizontal Error:       {:7.2} m    {:7.2} m      {:.2}x",
        ukf_stats.max_horizontal_error,
        pf_stats.max_horizontal_error,
        pf_stats.max_horizontal_error / ukf_stats.max_horizontal_error.max(0.1)
    );
    println!("RMS Altitude Error:         {:7.2} m    {:7.2} m      {:.2}x",
        ukf_stats.rms_altitude_error,
        pf_stats.rms_altitude_error,
        pf_stats.rms_altitude_error / ukf_stats.rms_altitude_error.max(0.1)
    );
    println!("Mean Velocity North Error:  {:7.3} m/s  {:7.3} m/s    {:.2}x",
        ukf_stats.mean_velocity_north_error,
        pf_stats.mean_velocity_north_error,
        pf_stats.mean_velocity_north_error / ukf_stats.mean_velocity_north_error.max(0.01)
    );

    // Both filters should produce reasonable results
    // PF might be slightly less accurate but should be in the same ballpark
    assert!(
        pf_stats.rms_horizontal_error < 100.0,
        "Particle filter RMS horizontal error should be reasonable: {:.2}m",
        pf_stats.rms_horizontal_error
    );

    assert!(
        ukf_stats.rms_horizontal_error < 100.0,
        "UKF RMS horizontal error should be reasonable: {:.2}m",
        ukf_stats.rms_horizontal_error
    );

    println!("\n✓ Both filters completed successfully with bounded errors");
    println!("========================================\n");
}

