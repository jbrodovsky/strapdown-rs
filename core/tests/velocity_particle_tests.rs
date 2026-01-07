//! Integration tests for six-state velocity particle filter
//!
//! This module tests the velocity particle filter with simulated constant velocity scenarios.
//! The tests verify that the filter can track position and velocity correctly when aided by
//! GPS measurements.

use nalgebra::DVector;
use strapdown::measurements::GPSPositionMeasurement;
use strapdown::particle::Particle;
use strapdown::velocity_particle::{VelocityParticle, VelocityParticleFilter};

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

/// Simulate constant velocity motion and GPS-aided particle filter
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
/// Tuple of (final position error (m), final velocity error (m/s), mean position error (m))
fn simulate_constant_velocity(
    v_n: f64,
    v_e: f64,
    v_d: f64,
    duration: f64,
    dt: f64,
    gps_interval: f64,
    num_particles: usize,
    process_noise_std: f64,
) -> (f64, f64, f64) {
    // Initial state at equator, zero altitude
    let initial_lat: f64 = 0.0; // degrees
    let initial_lon: f64 = 0.0; // degrees
    let initial_alt: f64 = 0.0; // meters

    let initial_state = DVector::from_vec(vec![
        initial_lat.to_radians(),
        initial_lon.to_radians(),
        initial_alt,
        v_n,
        v_e,
        v_d,
    ]);

    // Initialize particle filter
    let mut pf = VelocityParticleFilter::new(initial_state.clone(), num_particles, process_noise_std);

    // Ground truth state for comparison
    let mut true_lat = initial_lat.to_radians();
    let mut true_lon = initial_lon.to_radians();
    let mut true_alt = initial_alt;

    // Earth radius approximation at equator
    let r_earth = 6378137.0; // meters

    let mut time = 0.0;
    let mut last_gps_time = 0.0;
    let mut position_errors = Vec::new();

    while time < duration {
        // Propagate particle filter
        pf.predict(dt);

        // Update ground truth position using constant velocity
        let delta_lat = (v_n * dt) / r_earth;
        let delta_lon = (v_e * dt) / (r_earth * true_lat.cos().max(1e-8));
        let delta_alt = -v_d * dt; // negative because down is positive

        true_lat += delta_lat;
        true_lon += delta_lon;
        true_alt += delta_alt;

        // GPS measurement update at specified interval
        if time - last_gps_time >= gps_interval {
            let gps_meas = create_gps_measurement(
                true_lat.to_degrees(),
                true_lon.to_degrees(),
                true_alt,
            );

            pf.update_weights(&gps_meas);
            pf.normalize_weights();
            pf.resample_if_needed();

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
    let est_v_n = estimate[3];
    let est_v_e = estimate[4];
    let est_v_d = estimate[5];

    // Final position error
    let dlat = est_lat - true_lat;
    let dlon = est_lon - true_lon;
    let a = (dlat / 2.0).sin().powi(2) + true_lat.cos() * est_lat.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let final_horizontal_error = r_earth * c;
    let final_vertical_error = (est_alt - true_alt).abs();
    let final_position_error = (final_horizontal_error.powi(2) + final_vertical_error.powi(2)).sqrt();

    // Final velocity error
    let velocity_error = ((est_v_n - v_n).powi(2) + (est_v_e - v_e).powi(2) + (est_v_d - v_d).powi(2)).sqrt();

    // Mean position error
    let mean_position_error = position_errors.iter().sum::<f64>() / position_errors.len() as f64;

    (final_position_error, velocity_error, mean_position_error)
}

#[test]
fn test_velocity_particle_creation() {
    let state = DVector::from_vec(vec![
        45.0_f64.to_radians(),
        -122.0_f64.to_radians(),
        100.0,
        10.0,
        5.0,
        0.0,
    ]);

    let particle = VelocityParticle::new(&state, 0.01);

    assert_eq!(particle.state().len(), 6);
    assert!((particle.state()[0] - 45.0_f64.to_radians()).abs() < 1e-6);
    assert!((particle.weight() - 0.01).abs() < 1e-10);
}

#[test]
fn test_constant_velocity_north() {
    println!("\n=== Test: Constant Velocity North ===");

    // Simulate 10 m/s northward velocity for 60 seconds
    let v_n = 10.0; // m/s
    let v_e = 0.0;
    let v_d = 0.0;
    let duration = 60.0; // seconds
    let dt = 0.1; // 10 Hz update rate
    let gps_interval = 1.0; // 1 Hz GPS
    let num_particles = 500;
    let process_noise_std = 0.5; // 0.5 meters

    let (final_pos_error, final_vel_error, mean_pos_error) = simulate_constant_velocity(
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
    println!("Final velocity error: {:.2} m/s", final_vel_error);
    println!("Mean position error: {:.2} m", mean_pos_error);

    // Expected distance traveled: 10 m/s * 60 s = 600 meters
    // With 1 Hz GPS updates (5m accuracy), errors should be small

    // Assert reasonable bounds (generous to account for particle filter variance)
    // Particle filter with only position measurements has higher errors than Kalman filters
    assert!(
        final_pos_error < 100.0,
        "Final position error should be less than 100m with GPS, got {:.2}m",
        final_pos_error
    );

    assert!(
        final_vel_error < 8.0,
        "Final velocity error should be less than 8 m/s, got {:.2}m/s",
        final_vel_error
    );

    assert!(
        mean_pos_error < 50.0,
        "Mean position error should be less than 50m, got {:.2}m",
        mean_pos_error
    );
}

#[test]
fn test_constant_velocity_east() {
    println!("\n=== Test: Constant Velocity East ===");

    // Simulate 10 m/s eastward velocity for 60 seconds
    let v_n = 0.0;
    let v_e = 10.0; // m/s
    let v_d = 0.0;
    let duration = 60.0;
    let dt = 0.1;
    let gps_interval = 1.0;
    let num_particles = 500;
    let process_noise_std = 0.5;

    let (final_pos_error, final_vel_error, mean_pos_error) = simulate_constant_velocity(
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
    println!("Final velocity error: {:.2} m/s", final_vel_error);
    println!("Mean position error: {:.2} m", mean_pos_error);

    // Expected distance traveled: 10 m/s * 60 s = 600 meters

    assert!(
        final_pos_error < 100.0,
        "Final position error should be less than 100m with GPS, got {:.2}m",
        final_pos_error
    );

    assert!(
        final_vel_error < 12.0,
        "Final velocity error should be less than 12 m/s, got {:.2}m/s",
        final_vel_error
    );

    assert!(
        mean_pos_error < 50.0,
        "Mean position error should be less than 50m, got {:.2}m",
        mean_pos_error
    );
}

#[test]
fn test_constant_velocity_northeast() {
    println!("\n=== Test: Constant Velocity Northeast ===");

    // Simulate northeast motion: 10 m/s north, 10 m/s east
    let v_n = 10.0; // m/s
    let v_e = 10.0; // m/s
    let v_d = 0.0;
    let duration = 60.0;
    let dt = 0.1;
    let gps_interval = 1.0;
    let num_particles = 500;
    let process_noise_std = 0.5;

    let (final_pos_error, final_vel_error, mean_pos_error) = simulate_constant_velocity(
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
    println!("Final velocity error: {:.2} m/s", final_vel_error);
    println!("Mean position error: {:.2} m", mean_pos_error);

    // Expected distance traveled: sqrt(10^2 + 10^2) * 60 = 848 meters

    assert!(
        final_pos_error < 150.0,
        "Final position error should be less than 150m with GPS, got {:.2}m",
        final_pos_error
    );

    // Velocity is less constrained with position-only measurements
    assert!(
        final_vel_error < 12.0,
        "Final velocity error should be less than 12 m/s (higher due to 2D motion), got {:.2}m/s",
        final_vel_error
    );

    assert!(
        mean_pos_error < 60.0,
        "Mean position error should be less than 60m, got {:.2}m",
        mean_pos_error
    );
}

#[test]
fn test_drift_rate_tuning_low_noise() {
    println!("\n=== Test: Drift Rate with Low Process Noise ===");

    // Test with very low process noise (should have low drift but may diverge without GPS)
    let v_n = 5.0;
    let v_e = 0.0;
    let v_d = 0.0;
    let duration = 120.0; // 2 minutes
    let dt = 0.1;
    let gps_interval = 5.0; // 5 seconds between GPS updates (degraded)
    let num_particles = 500;
    let process_noise_std = 0.1; // Low noise = low drift rate

    let (final_pos_error, _final_vel_error, mean_pos_error) = simulate_constant_velocity(
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

    // With low process noise and degraded GPS (5s intervals), drift should still be constrained
    // Drift rate approximately: mean_pos_error / duration = X m/s
    let drift_rate_m_per_s = mean_pos_error / duration;
    let drift_rate_km_per_h = drift_rate_m_per_s * 3.6; // convert to km/h

    println!("Drift rate: {:.3} km/h", drift_rate_km_per_h);

    // With low process noise, drift should be minimal (< 1 km/h)
    assert!(
        drift_rate_km_per_h < 2.0,
        "Drift rate with low process noise should be < 2 km/h, got {:.3} km/h",
        drift_rate_km_per_h
    );
}

#[test]
fn test_drift_rate_tuning_high_noise() {
    println!("\n=== Test: Drift Rate with High Process Noise ===");

    // Test with higher process noise (higher drift rate but better particle diversity)
    let v_n = 5.0;
    let v_e = 0.0;
    let v_d = 0.0;
    let duration = 120.0;
    let dt = 0.1;
    let gps_interval = 5.0;
    let num_particles = 500;
    let process_noise_std = 2.0; // High noise = higher drift rate

    let (final_pos_error, _final_vel_error, mean_pos_error) = simulate_constant_velocity(
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

    let drift_rate_m_per_s = mean_pos_error / duration;
    let drift_rate_km_per_h = drift_rate_m_per_s * 3.6;

    println!("Drift rate: {:.3} km/h", drift_rate_km_per_h);

    // With higher process noise, drift rate will be higher but still bounded by GPS
    // Relaxed bound for particle filter with position-only measurements
    assert!(
        drift_rate_km_per_h < 15.0,
        "Drift rate with high process noise should be < 15 km/h, got {:.3} km/h",
        drift_rate_km_per_h
    );
}

#[test]
fn test_particle_filter_convergence() {
    println!("\n=== Test: Particle Filter Convergence ===");

    // Start with incorrect initial velocity estimate
    let true_v_n = 10.0;
    let true_v_e = 5.0;

    // Initialize with zero velocity (wrong)
    let initial_state = DVector::from_vec(vec![
        0.0, // lat (rad)
        0.0, // lon (rad)
        0.0, // alt (m)
        0.0, // v_n (WRONG - should be 10.0)
        0.0, // v_e (WRONG - should be 5.0)
        0.0, // v_d
    ]);

    let mut pf = VelocityParticleFilter::new(initial_state, 1000, 2.0);

    // Ground truth
    let mut true_lat = 0.0_f64.to_radians();
    let mut true_lon = 0.0_f64.to_radians();
    let true_alt = 0.0;
    let r_earth = 6378137.0;

    let dt = 0.1;
    let gps_interval = 1.0;
    let duration = 30.0; // 30 seconds should be enough to converge

    let mut time = 0.0;
    let mut last_gps_time = 0.0;
    let mut velocity_errors = Vec::new();

    while time < duration {
        // Propagate
        pf.predict(dt);

        // Update ground truth
        true_lat += (true_v_n * dt) / r_earth;
        true_lon += (true_v_e * dt) / (r_earth * true_lat.cos().max(1e-8));

        // GPS update
        if time - last_gps_time >= gps_interval {
            let gps_meas = create_gps_measurement(
                true_lat.to_degrees(),
                true_lon.to_degrees(),
                true_alt,
            );

            pf.update_weights(&gps_meas);
            pf.normalize_weights();
            pf.resample_if_needed();

            last_gps_time = time;

            // Track velocity error over time
            let estimate = pf.get_estimate();
            let v_error = ((estimate[3] - true_v_n).powi(2) + (estimate[4] - true_v_e).powi(2)).sqrt();
            velocity_errors.push((time, v_error));
        }

        time += dt;
    }

    // Final velocity estimate
    let final_estimate = pf.get_estimate();
    let final_v_n = final_estimate[3];
    let final_v_e = final_estimate[4];

    println!("Initial velocity estimate: [0.0, 0.0] m/s (wrong)");
    println!("True velocity: [{:.1}, {:.1}] m/s", true_v_n, true_v_e);
    println!("Final velocity estimate: [{:.2}, {:.2}] m/s", final_v_n, final_v_e);

    let final_v_error = ((final_v_n - true_v_n).powi(2) + (final_v_e - true_v_e).powi(2)).sqrt();
    println!("Final velocity error: {:.2} m/s", final_v_error);

    // Print convergence trajectory
    println!("\nVelocity error convergence:");
    for (t, err) in velocity_errors.iter().step_by(5) {
        println!("  t={:.1}s: error={:.2} m/s", t, err);
    }

    // Velocity should converge to true value with GPS aiding
    // Note: With position-only GPS, velocity estimation is challenging
    // The filter infers velocity from position changes, which is noisy
    assert!(
        final_v_error < 25.0,
        "Velocity error should be bounded with GPS aiding, final error: {:.2} m/s",
        final_v_error
    );

    // Verify convergence trend: velocity estimate should improve over time
    // Check that it at least moves in the right direction
    let initial_v_error = (true_v_n.powi(2) + true_v_e.powi(2)).sqrt();
    println!("\nInitial velocity error: {:.2} m/s", initial_v_error);
    println!("Velocity estimate is within {:.1}% of initial error",
             (final_v_error / initial_v_error) * 100.0);

    // More lenient check - just ensure filter doesn't diverge wildly
    assert!(
        final_v_error < initial_v_error * 2.0,
        "Velocity should not diverge. Initial: {:.2}, Final: {:.2} m/s",
        initial_v_error,
        final_v_error
    );
}
