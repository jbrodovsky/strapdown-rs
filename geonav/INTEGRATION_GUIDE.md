# Integrating Velocity-Informed Particle Filter with Existing Code

This guide shows how to integrate the velocity-informed particle filter into existing strapdown-rs simulations.

## Quick Integration Checklist

- [ ] Add geonav dependency to your project
- [ ] Load geophysical map (gravity or magnetic)
- [ ] Initialize particle filter with initial position
- [ ] Add particle propagation to IMU event handling
- [ ] Add particle weight update to geophysical measurement events
- [ ] Monitor and resample as needed
- [ ] Log/visualize particle filter output

## Step-by-Step Integration

### Step 1: Add Dependencies

In your `Cargo.toml`:

```toml
[dependencies]
strapdown-core = { path = "../core" }
geonav = { path = "../geonav" }
nalgebra = "0.33"
```

### Step 2: Import Required Types

```rust
use geonav::velocity_particle::{
    VelocityInformedParticleFilter,
    VelocityInformedParticle,
};
use geonav::{GeoMap, GravityMeasurement, MagneticAnomalyMeasurement};
use strapdown::kalman::UnscentedKalmanFilter;
use strapdown::particle::{ParticleResamplingStrategy, ParticleAveragingStrategy};
use nalgebra::DVector;
```

### Step 3: Load Geophysical Map

```rust
// Load gravity anomaly map
let gravity_map = GeoMap::load_geomap(
    PathBuf::from("path/to/gravity_map.nc"),
    GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute)
)?;
let gravity_map_rc = Rc::new(gravity_map);

// Or load magnetic anomaly map
let magnetic_map = GeoMap::load_geomap(
    PathBuf::from("path/to/magnetic_map.nc"),
    GeophysicalMeasurementType::Magnetic(MagneticResolution::TwoMinutes)
)?;
let magnetic_map_rc = Rc::new(magnetic_map);
```

### Step 4: Initialize Filters

```rust
// Initialize UKF (existing code)
let initial_state = InitialState {
    latitude: 45.0,
    longitude: -122.0,
    altitude: 100.0,
    // ... other fields
};

let mut ukf = UnscentedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // IMU biases
    None,
    covariance_diagonal,
    process_noise,
    alpha, beta, kappa,
);

// Initialize particle filter
let initial_position = DVector::from_vec(vec![
    initial_state.latitude.to_radians(),
    initial_state.longitude.to_radians(),
    initial_state.altitude,
]);

let mut particle_filter = VelocityInformedParticleFilter::new(
    initial_position,
    1000,  // number of particles
    1.0,   // process noise std (meters)
);
```

### Step 5: Modify Event Loop

```rust
// Add storage for results
let mut ukf_results = Vec::new();
let mut pf_results = Vec::new();

for event in event_stream {
    match event {
        Event::Imu { dt_s, imu, elapsed_s } => {
            // 1. Propagate UKF (existing)
            ukf.predict(&imu, dt_s);
            
            // 2. Extract velocity from UKF
            let ukf_state = ukf.get_estimate();
            let velocity = DVector::from_vec(vec![
                ukf_state[3],  // v_north
                ukf_state[4],  // v_east
                ukf_state[5],  // v_down
            ]);
            
            // 3. Propagate particle filter with UKF velocity
            particle_filter.propagate(&velocity, dt_s);
        }
        
        Event::Measurement { meas, elapsed_s } => {
            // Check measurement type and route accordingly
            
            // For GNSS measurements - update UKF only
            if meas.as_any().downcast_ref::<GPSPositionMeasurement>().is_some() ||
               meas.as_any().downcast_ref::<GPSVelocityMeasurement>().is_some() ||
               meas.as_any().downcast_ref::<GPSPositionAndVelocityMeasurement>().is_some() {
                ukf.update(meas.as_ref());
            }
            
            // For geophysical measurements - update particle filter
            else if let Some(gravity_meas) = meas.as_any_mut().downcast_mut::<GravityMeasurement>() {
                // Set current state for Eötvös correction
                let ukf_state = ukf.get_estimate();
                let strapdown_state: StrapdownState = (&ukf_state.as_slice()[..9]).try_into()?;
                gravity_meas.set_state(&strapdown_state);
                
                // Update particle weights
                particle_filter.update_weights(gravity_meas);
                particle_filter.normalize_weights();
                
                // Check if resampling needed
                let n_eff = particle_filter.effective_sample_size();
                if n_eff < 0.5 * particle_filter.num_particles() as f64 {
                    particle_filter.resample();
                }
            }
            
            else if let Some(mag_meas) = meas.as_any_mut().downcast_mut::<MagneticAnomalyMeasurement>() {
                // Set current state for WMM reference
                let ukf_state = ukf.get_estimate();
                let strapdown_state: StrapdownState = (&ukf_state.as_slice()[..9]).try_into()?;
                mag_meas.set_state(&strapdown_state);
                
                // Update particle weights
                particle_filter.update_weights(mag_meas);
                particle_filter.normalize_weights();
                
                // Resample if needed
                let n_eff = particle_filter.effective_sample_size();
                if n_eff < 0.5 * particle_filter.num_particles() as f64 {
                    particle_filter.resample();
                }
            }
        }
    }
    
    // Store results for later analysis
    ukf_results.push((
        elapsed_s,
        ukf.get_estimate().clone(),
        ukf.get_certainty().clone(),
    ));
    
    pf_results.push((
        elapsed_s,
        particle_filter.get_estimate().clone(),
        particle_filter.get_covariance().clone(),
        particle_filter.effective_sample_size(),
    ));
}
```

### Step 6: Save and Analyze Results

```rust
// Create output directory
std::fs::create_dir_all("output")?;

// Save UKF results (existing)
let ukf_csv = generate_ukf_csv(&ukf_results);
std::fs::write("output/ukf_trajectory.csv", ukf_csv)?;

// Save particle filter results
let pf_csv = generate_pf_csv(&pf_results);
std::fs::write("output/pf_trajectory.csv", pf_csv)?;

// Save comparison
let comparison_csv = generate_comparison_csv(&ukf_results, &pf_results);
std::fs::write("output/comparison.csv", comparison_csv)?;

fn generate_pf_csv(results: &[(f64, DVector<f64>, DMatrix<f64>, f64)]) -> String {
    let mut csv = String::from("time,latitude_deg,longitude_deg,altitude_m,\
                                lat_std_m,lon_std_m,alt_std_m,n_eff\n");
    
    for (time, position, covariance, n_eff) in results {
        let lat_deg = position[0].to_degrees();
        let lon_deg = position[1].to_degrees();
        let alt = position[2];
        
        // Convert covariance to meters
        let lat_std_m = covariance[(0, 0)].sqrt() * 111_000.0;
        let lon_std_m = covariance[(1, 1)].sqrt() * 111_000.0 * lat_deg.to_radians().cos();
        let alt_std_m = covariance[(2, 2)].sqrt();
        
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            time, lat_deg, lon_deg, alt,
            lat_std_m, lon_std_m, alt_std_m, n_eff
        ));
    }
    
    csv
}
```

## Advanced Configuration

### Custom Resampling Strategy

```rust
use strapdown::particle::ParticleResamplingStrategy;

let pf = VelocityInformedParticleFilter::new_with_config(
    initial_position,
    1000,
    1.0,
    ParticleResamplingStrategy::Residual,  // Better for multimodal distributions
    ParticleAveragingStrategy::WeightedMean,
    0.5,
);
```

### Adaptive Process Noise

```rust
// Adjust process noise based on vehicle dynamics
let speed = velocity.norm();
let adaptive_noise = if speed > 10.0 {
    2.0  // More noise at high speed
} else {
    0.5  // Less noise at low speed
};

particle_filter.set_process_noise_std(adaptive_noise);
```

### Particle Diversity Monitoring

```rust
// Check if particles are degenerating
let n_eff = particle_filter.effective_sample_size();
let n_total = particle_filter.num_particles() as f64;
let diversity_ratio = n_eff / n_total;

if diversity_ratio < 0.3 {
    eprintln!("Warning: Low particle diversity ({:.1}%)", diversity_ratio * 100.0);
    // Consider increasing process noise or resampling more frequently
}
```

## Visualization

### Plotting with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
ukf_df = pd.read_csv('output/ukf_trajectory.csv')
pf_df = pd.read_csv('output/pf_trajectory.csv')

# Plot trajectories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Position comparison
ax1.plot(ukf_df['longitude_deg'], ukf_df['latitude_deg'], 
         label='UKF', linewidth=2)
ax1.plot(pf_df['longitude_deg'], pf_df['latitude_deg'], 
         label='Particle Filter', linewidth=2, alpha=0.7)
ax1.set_xlabel('Longitude (deg)')
ax1.set_ylabel('Latitude (deg)')
ax1.set_title('Position Comparison')
ax1.legend()
ax1.grid(True)

# Uncertainty over time
ax2.plot(pf_df['time'], pf_df['lat_std_m'], label='Latitude std')
ax2.plot(pf_df['time'], pf_df['lon_std_m'], label='Longitude std')
ax2.plot(pf_df['time'], pf_df['alt_std_m'], label='Altitude std')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position Std (m)')
ax2.set_title('Particle Filter Uncertainty')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('output/trajectory_comparison.png', dpi=300)
plt.show()

# Plot effective sample size
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pf_df['time'], pf_df['n_eff'])
ax.axhline(y=pf_df['n_eff'].iloc[0] * 0.5, 
           color='r', linestyle='--', label='50% threshold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Effective Sample Size')
ax.set_title('Particle Filter Degeneracy')
ax.legend()
ax.grid(True)
plt.savefig('output/effective_sample_size.png', dpi=300)
plt.show()
```

## Troubleshooting

### Particles Diverging
**Symptoms**: Position estimate drifts away from truth
**Solutions**:
- Decrease process noise
- Increase measurement frequency
- Check geophysical map coverage
- Verify measurement noise values

### Particle Degeneracy
**Symptoms**: N_eff drops rapidly, most particles have ~zero weight
**Solutions**:
- Increase process noise
- Lower resampling threshold
- Use Residual resampling strategy
- Increase number of particles

### Poor Performance in GNSS-Denied
**Symptoms**: Position accuracy degrades without GNSS
**Solutions**:
- Ensure geophysical map covers area
- Increase geophysical measurement frequency
- Tune measurement noise to match actual sensor
- Use higher resolution maps

### High Computational Cost
**Symptoms**: Real-time performance issues
**Solutions**:
- Reduce number of particles
- Decrease geophysical measurement frequency
- Use Systematic resampling (fastest)
- Consider GPU acceleration (future)

## Best Practices

1. **Start Simple**: Begin with 1000 particles and default settings
2. **Monitor N_eff**: Keep above 50% of total particles
3. **Tune Gradually**: Change one parameter at a time
4. **Validate**: Compare against ground truth or UKF-only solution
5. **Log Everything**: Save all states, covariances, and diagnostics
6. **Visualize**: Plot trajectories, uncertainties, and particle distributions

## Support and Resources

- Documentation: `cargo doc --open --package geonav`
- Examples: `examples/velocity_particle_integration.rs`
- Full guide: `geonav/VELOCITY_PARTICLE_FILTER.md`
- Issues: https://github.com/jbrodovsky/strapdown-rs/issues
