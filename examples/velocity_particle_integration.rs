//! Velocity-informed particle filter example
//!
//! This example demonstrates how to use the velocity-informed particle filter
//! alongside an INS/GNSS filter (UKF) for geophysical navigation. The particle
//! filter uses velocity estimates from the UKF to propagate position-only particles,
//! while incorporating geophysical anomaly measurements for enhanced position estimation.
//!
//! # Architecture
//!
//! ```text
//!           IMU + GNSS
//!                |
//!                v
//!         +-------------+
//!         |     UKF     |  <- Full state filter (position, velocity, attitude)
//!         +-------------+
//!            |        |
//!            v        v
//!        Position  Velocity
//!            |        |
//!            |        v
//!            |  +------------------+
//!            |  | Particle Filter  |  <- Position-only tracking
//!            |  |  + Geo Anomaly   |     with geophysical measurements
//!            |  +------------------+
//!            |        |
//!            v        v
//!       UKF Position  PF Position
//! ```
//!
//! # Usage
//!
//! This is a conceptual example showing the architecture. In practice, you would
//! need actual IMU/GNSS data and geophysical maps.
//!
//! ```bash
//! cargo run --example velocity_particle_integration
//! ```

use nalgebra::DVector;
use std::rc::Rc;

// Note: This is a conceptual example. In a real application, you would need:
// 1. Actual IMU/GNSS data
// 2. Geophysical maps (gravity or magnetic)
// 3. HDF5/NetCDF library dependencies for loading maps

fn main() {
    println!("=== Velocity-Informed Particle Filter Example ===\n");
    
    // This example shows the conceptual architecture and API usage
    // For a complete working example, see the geonav-sim examples directory
    
    println!("Architecture:");
    println!("  1. UKF provides full state estimation (position, velocity, attitude)");
    println!("  2. Particle filter tracks position using UKF velocity estimates");
    println!("  3. Geophysical measurements (gravity/magnetic) enhance PF position");
    println!("  4. Open-loop: PF does not feed back to UKF\n");
    
    println!("Key Parameters:");
    println!("  - num_particles: Controls accuracy vs computational cost");
    println!("    Typical range: 500-5000 particles");
    println!("    More particles = better approximation but slower");
    println!("");
    println!("  - process_noise_std: Controls particle diversity (meters)");
    println!("    Typical range: 0.1-10.0 meters");
    println!("    Higher values = more exploration but less precision");
    println!("    Lower values = more precision but risk particle degeneracy");
    println!("");
    
    println!("Usage Pattern:");
    println!("  ```rust");
    println!("  // Initialize UKF for INS/GNSS");
    println!("  let mut ukf = UnscentedKalmanFilter::new(initial_state, ...);");
    println!("  ");
    println!("  // Initialize particle filter for position tracking");
    println!("  let initial_position = DVector::from_vec(vec![");
    println!("      initial_state.latitude.to_radians(),");
    println!("      initial_state.longitude.to_radians(),");
    println!("      initial_state.altitude");
    println!("  ]);");
    println!("  let mut pf = VelocityInformedParticleFilter::new(");
    println!("      initial_position,");
    println!("      1000,  // num_particles");
    println!("      1.0,   // process_noise_std (meters)");
    println!("  );");
    println!("  ");
    println!("  // Main loop");
    println!("  for event in event_stream {{");
    println!("      match event {{");
    println!("          Event::Imu {{ dt_s, imu, .. }} => {{");
    println!("              // Propagate UKF");
    println!("              ukf.predict(&imu, dt_s);");
    println!("              ");
    println!("              // Extract velocity from UKF");
    println!("              let velocity = ukf.get_estimate().rows(3, 3).into_owned();");
    println!("              ");
    println!("              // Propagate particle filter using UKF velocity");
    println!("              pf.propagate(&velocity, dt_s);");
    println!("          }}");
    println!("          Event::Measurement {{ meas, .. }} => {{");
    println!("              // Update UKF with GNSS");
    println!("              ukf.update(meas.as_ref());");
    println!("              ");
    println!("              // Check if this is a geophysical measurement");
    println!("              if let Some(geo_meas) = downcast_geo_measurement(meas) {{");
    println!("                  // Update particle filter with geophysical measurement");
    println!("                  pf.update_weights(geo_meas);");
    println!("                  pf.normalize_weights();");
    println!("                  ");
    println!("                  // Resample if needed");
    println!("                  if pf.effective_sample_size() < 0.5 * pf.num_particles() {{");
    println!("                      pf.resample();");
    println!("                  }}");
    println!("              }}");
    println!("          }}");
    println!("      }}");
    println!("  }}");
    println!("  ");
    println!("  // Get position estimates");
    println!("  let ukf_position = ukf.get_estimate().rows(0, 3);");
    println!("  let pf_position = pf.get_estimate();");
    println!("  ```");
    println!("");
    
    println!("Resampling Strategies:");
    println!("  - Systematic (default): Low variance, deterministic");
    println!("  - Multinomial: Simple but higher variance");
    println!("  - Stratified: Good balance of randomness and low variance");
    println!("  - Residual: Preserves high-weight particles");
    println!("");
    
    println!("Averaging Strategies:");
    println!("  - WeightedMean (default): Statistically optimal");
    println!("  - Mean: Simple average, ignores weights");
    println!("  - HighestWeight: MAP estimate, single best particle");
    println!("");
    
    println!("Performance Considerations:");
    println!("  - Particle count: 1000-2000 recommended for real-time");
    println!("  - Monitor effective sample size (N_eff)");
    println!("  - Resample when N_eff < 50% of particle count");
    println!("  - Adjust process noise based on vehicle dynamics");
    println!("");
    
    println!("Expected Behavior:");
    println!("  - UKF provides smooth, continuous state estimate");
    println!("  - Particle filter position converges to areas matching geophysical map");
    println!("  - In GNSS-denied scenarios, PF can provide position updates");
    println!("  - Particle diversity maintained by process noise");
    println!("");
    
    println!("For a complete working example with real data, see:");
    println!("  examples/geonav-sim/");
    println!("");
    
    // Demonstrate basic API with mock data
    demonstrate_api();
}

fn demonstrate_api() {
    println!("=== API Demonstration (Mock Data) ===\n");
    
    // Initial position: 45°N, 122°W, 100m altitude
    let initial_position = DVector::from_vec(vec![
        45.0_f64.to_radians(),
        -122.0_f64.to_radians(),
        100.0,
    ]);
    
    println!("Initial position:");
    println!("  Latitude:  {} rad ({:.4}°)", initial_position[0], initial_position[0].to_degrees());
    println!("  Longitude: {} rad ({:.4}°)", initial_position[1], initial_position[1].to_degrees());
    println!("  Altitude:  {} m", initial_position[2]);
    println!("");
    
    // This would normally come from geonav crate
    // For demonstration, we show the API calls
    println!("Creating particle filter:");
    println!("  let mut pf = VelocityInformedParticleFilter::new(");
    println!("      initial_position,");
    println!("      1000,  // particles");
    println!("      1.0,   // process_noise_std");
    println!("  );");
    println!("");
    
    // Simulate velocity from UKF: 10 m/s north, 5 m/s east, 0 m/s vertical
    let velocity = DVector::from_vec(vec![10.0, 5.0, 0.0]);
    println!("Velocity from UKF: [{:.1}, {:.1}, {:.1}] m/s", velocity[0], velocity[1], velocity[2]);
    println!("");
    
    println!("Propagating particles:");
    println!("  pf.propagate(&velocity, 0.1);  // dt = 0.1s");
    println!("");
    
    // Expected position change
    use std::f64::consts::PI;
    let earth_radius = 6.371e6; // meters
    let expected_lat_change = (velocity[0] * 0.1) / earth_radius * (180.0 / PI);
    let expected_lon_change = (velocity[1] * 0.1) / (earth_radius * (initial_position[0].cos())) * (180.0 / PI);
    
    println!("Expected position change (approximate):");
    println!("  Δ Latitude:  {:.8}° ({:.3} m north)", expected_lat_change, velocity[0] * 0.1);
    println!("  Δ Longitude: {:.8}° ({:.3} m east)", expected_lon_change, velocity[1] * 0.1);
    println!("  Δ Altitude:  {:.3} m", -velocity[2] * 0.1);
    println!("");
    
    println!("Updating with geophysical measurement:");
    println!("  pf.update_weights(&gravity_measurement);");
    println!("  pf.normalize_weights();");
    println!("");
    
    println!("Checking for resampling need:");
    println!("  let n_eff = pf.effective_sample_size();");
    println!("  if n_eff < 0.5 * pf.num_particles() as f64 {{");
    println!("      pf.resample();");
    println!("  }}");
    println!("");
    
    println!("Getting position estimate:");
    println!("  let position = pf.get_estimate();");
    println!("  let covariance = pf.get_covariance();");
    println!("");
    
    println!("Configuration options:");
    println!("  - Resampling strategy: Systematic, Multinomial, Stratified, Residual");
    println!("  - Averaging strategy: WeightedMean, Mean, HighestWeight");
    println!("  - Resampling threshold: Typically 0.5 (50% of particles)");
    println!("  - Process noise: Adjust based on vehicle dynamics and uncertainty");
    println!("");
    
    println!("=== End of Example ===");
}
