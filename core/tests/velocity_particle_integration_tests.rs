//! Integration tests for velocity-based particle filter
//!
//! Tests the particle filter with simulated constant velocity scenarios
//! where velocities are supplied as inputs to the predict step.

use nalgebra::DVector;
use strapdown::NavigationFilter;
use strapdown::measurements::GPSPositionMeasurement;
use strapdown::particle::{VelocityInput, VelocityParticleFilter};

/// Helper function to create simulated GPS measurements from ground truth state
fn create_gps_measurement(lat_deg: f64, lon_deg: f64, alt: f64) -> GPSPositionMeasurement {
    GPSPositionMeasurement {
        latitude: lat_deg,
        longitude: lon_deg,
        altitude: alt,
        horizontal_noise_std: 5.0, // 5 meter horizontal accuracy
        vertical_noise_std: 2.0,   // 2 meter vertical accuracy
    }
}

/// Simulate constant velocity motion with velocity-based particle filter
///
/// # Arguments
/// * `v_n` - Northward velocity (m/s)
/// * `v_e` - Eastward velocity (m/s)
/// * `v_d` - Vertical velocity (m/s, positive down in NED)
/// * `duration` - Total simulation duration (seconds)
/// * `dt` - Time step (seconds)
/// * `gps_interval` - GPS measurement interval (seconds)
/// * `num_particles` - Number of particles
/// * `process_noise_std` - Process noise standard deviation (meters)
///
/// # Returns
/// Tuple of (final position error (m), mean position error (m))
fn simulate_constant_velocity(
    v_n: f64,
    v_e: f64,
    v_d: f64,
    duration: f64,
    dt: f64,
    gps_interval: f64,
    num_particles: usize,
    process_noise_std: f64,
) -> (f64, f64) {
    // Initial state at equator, zero altitude
    let initial_lat: f64 = 0.0; // degrees
    let initial_lon: f64 = 0.0; // degrees
    let initial_alt: f64 = 0.0; // meters

    let initial_state = DVector::from_vec(vec![
        initial_lat.to_radians(),
        initial_lon.to_radians(),
        initial_alt,
    ]);

    // Initialize particle filter
    let mut pf = VelocityParticleFilter::new_with_seed(
        initial_state.clone(),
        num_particles,
        process_noise_std,
        42, // Fixed seed for reproducibility
    );

    // Ground truth state for comparison
    let mut true_lat = initial_lat.to_radians();
    let mut true_lon = initial_lon.to_radians();
    let mut true_alt = initial_alt;

    // Earth radius approximation at equator
    let r_earth = 6378137.0; // meters

    let mut time = 0.0;
    let mut last_gps_time = 0.0;
    let mut position_errors = Vec::new();

    // Create velocity input
    let vel_input = VelocityInput {
        v_north: v_n,
        v_east: v_e,
        v_vertical: v_d,
    };

    while time < duration {
        // Propagate particle filter with velocity input
        pf.predict(&vel_input, dt);

        // Update ground truth position using constant velocity
        let delta_lat = (v_n * dt) / r_earth;
        let delta_lon = (v_e * dt) / (r_earth * true_lat.cos().max(1e-8));
        let delta_alt = -v_d * dt; // negative because down is positive

        true_lat += delta_lat;
        true_lon += delta_lon;
        true_alt += delta_alt;

        // GPS measurement update at specified interval
        if time - last_gps_time >= gps_interval {
            let gps_meas =
                create_gps_measurement(true_lat.to_degrees(), true_lon.to_degrees(), true_alt);

            pf.update(&gps_meas);

            last_gps_time = time;
        }

        // Compute position error
        let estimate = pf.get_estimate();
        let est_lat = estimate[0];
        let est_lon = estimate[1];
        let est_alt = estimate[2];

        // Haversine distance for horizontal error
        let dlat = est_lat - true_lat;
        let dlon = est_lon - true_lon;
        let a = (dlat / 2.0).sin().powi(2)
            + true_lat.cos() * est_lat.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        let horizontal_error = r_earth * c;

        let vertical_error = (est_alt - true_alt).abs();
        let position_error = (horizontal_error.powi(2) + vertical_error.powi(2)).sqrt();

        position_errors.push(position_error);

        time += dt;
    }

    // Compute final errors
    let estimate = pf.get_estimate();
    let est_lat = estimate[0];
    let est_lon = estimate[1];
    let est_alt = estimate[2];

    // Final position error
    let dlat = est_lat - true_lat;
    let dlon = est_lon - true_lon;
    let a =
        (dlat / 2.0).sin().powi(2) + true_lat.cos() * est_lat.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let final_horizontal_error = r_earth * c;
    let final_vertical_error = (est_alt - true_alt).abs();
    let final_position_error =
        (final_horizontal_error.powi(2) + final_vertical_error.powi(2)).sqrt();

    // Mean position error
    let mean_position_error = position_errors.iter().sum::<f64>() / position_errors.len() as f64;

    (final_position_error, mean_position_error)
}

#[test]
fn test_constant_velocity_north() {
    println!("\n=== Test: Constant Velocity North (Velocity-Based PF) ===");

    // Simulate 10 m/s northward velocity for 60 seconds
    let v_n = 10.0; // m/s
    let v_e = 0.0;
    let v_d = 0.0;
    let duration = 60.0; // seconds
    let dt = 0.1; // 10 Hz update rate
    let gps_interval = 1.0; // 1 Hz GPS
    let num_particles = 1000;
    let process_noise_std = 0.5; // 0.5 meters

    let (final_pos_error, mean_pos_error) = simulate_constant_velocity(
        v_n,
        v_e,
        v_d,
        duration,
        dt,
        gps_interval,
        num_particles,
        process_noise_std,
    );

    println!("Final position error: {:.2} m", final_pos_error);
    println!("Mean position error: {:.2} m", mean_pos_error);

    // Expected distance traveled: 10 m/s * 60 s = 600 meters
    // With 1 Hz GPS updates (5m accuracy), errors should be small
    assert!(
        final_pos_error < 50.0,
        "Final position error should be less than 50m with GPS, got {:.2}m",
        final_pos_error
    );

    assert!(
        mean_pos_error < 30.0,
        "Mean position error should be less than 30m, got {:.2}m",
        mean_pos_error
    );
}

#[test]
fn test_constant_velocity_east() {
    println!("\n=== Test: Constant Velocity East (Velocity-Based PF) ===");

    // Simulate 10 m/s eastward velocity for 60 seconds
    let v_n = 0.0;
    let v_e = 10.0; // m/s
    let v_d = 0.0;
    let duration = 60.0;
    let dt = 0.1;
    let gps_interval = 1.0;
    let num_particles = 1000;
    let process_noise_std = 0.5;

    let (final_pos_error, mean_pos_error) = simulate_constant_velocity(
        v_n,
        v_e,
        v_d,
        duration,
        dt,
        gps_interval,
        num_particles,
        process_noise_std,
    );

    println!("Final position error: {:.2} m", final_pos_error);
    println!("Mean position error: {:.2} m", mean_pos_error);

    assert!(
        final_pos_error < 50.0,
        "Final position error should be less than 50m with GPS, got {:.2}m",
        final_pos_error
    );

    assert!(
        mean_pos_error < 30.0,
        "Mean position error should be less than 30m, got {:.2}m",
        mean_pos_error
    );
}

#[test]
fn test_constant_velocity_northeast() {
    println!("\n=== Test: Constant Velocity Northeast (Velocity-Based PF) ===");

    // Simulate northeast motion: 10 m/s north, 10 m/s east
    let v_n = 10.0; // m/s
    let v_e = 10.0; // m/s
    let v_d = 0.0;
    let duration = 60.0;
    let dt = 0.1;
    let gps_interval = 1.0;
    let num_particles = 1000;
    let process_noise_std = 0.5;

    let (final_pos_error, mean_pos_error) = simulate_constant_velocity(
        v_n,
        v_e,
        v_d,
        duration,
        dt,
        gps_interval,
        num_particles,
        process_noise_std,
    );

    println!("Final position error: {:.2} m", final_pos_error);
    println!("Mean position error: {:.2} m", mean_pos_error);

    assert!(
        final_pos_error < 60.0,
        "Final position error should be less than 60m with GPS, got {:.2}m",
        final_pos_error
    );

    assert!(
        mean_pos_error < 35.0,
        "Mean position error should be less than 35m, got {:.2}m",
        mean_pos_error
    );
}

#[test]
fn test_open_loop_drift() {
    println!("\n=== Test: Open-Loop Drift (No GPS) ===");

    // Test drift rate without GPS (open-loop)
    let v_n = 10.0;
    let v_e = 5.0;
    let v_d = 0.0;
    let duration = 300.0; // 5 minutes
    let dt = 0.1;
    let num_particles = 1000;
    let process_noise_std = 1.0; // 1 meter process noise

    let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let mut pf =
        VelocityParticleFilter::new_with_seed(initial_state, num_particles, process_noise_std, 42);

    // Ground truth
    let mut true_lat = 0.0_f64;
    let mut true_lon = 0.0_f64;
    let r_earth = 6378137.0;

    let vel_input = VelocityInput {
        v_north: v_n,
        v_east: v_e,
        v_vertical: v_d,
    };

    let mut time = 0.0;
    let mut position_errors = Vec::new();

    while time < duration {
        pf.predict(&vel_input, dt);

        // Update ground truth
        true_lat += (v_n * dt) / r_earth;
        true_lon += (v_e * dt) / (r_earth * true_lat.cos().max(1e-8));

        // Compute error
        let estimate = pf.get_estimate();
        let dlat = estimate[0] - true_lat;
        let dlon = estimate[1] - true_lon;
        let a = (dlat / 2.0).sin().powi(2)
            + true_lat.cos() * estimate[0].cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        let error = r_earth * c;

        position_errors.push(error);

        time += dt;
    }

    let mean_error = position_errors.iter().sum::<f64>() / position_errors.len() as f64;
    let drift_rate_m_per_s = mean_error / duration;
    let drift_rate_km_per_h = drift_rate_m_per_s * 3.6;

    println!("Mean drift: {:.2} m", mean_error);
    println!("Drift rate: {:.3} km/h", drift_rate_km_per_h);
    println!("Drift rate: {:.3} m/s", drift_rate_m_per_s);

    // With 1m process noise, drift should be in a reasonable range
    // This is open-loop so drift accumulates over time
    assert!(
        drift_rate_km_per_h < 5.0,
        "Drift rate should be < 5 km/h, got {:.3} km/h",
        drift_rate_km_per_h
    );
}
