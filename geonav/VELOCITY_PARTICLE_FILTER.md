# Velocity-Informed Particle Filter for Geophysical Navigation

## Overview

The velocity-informed particle filter is a specialized particle filter designed for geophysical navigation that leverages velocity estimates from an existing INS/GNSS filter (such as a UKF or EKF) to efficiently track position. This approach reduces the computational burden of full-state particle filtering while maintaining the ability to incorporate geophysical anomaly measurements for enhanced position estimation, especially in GNSS-denied environments.

## Architecture

```
                        IMU
                         |
                         v
        +----------------------------------+
        |   INS/GNSS Filter (UKF/EKF)    |
        |   - Position, velocity, attitude|
        |   - IMU bias estimation         |
        +----------------------------------+
                  |               |
                  |               | Velocity
                  v               | estimates
                GNSS              |
                                  v
        +----------------------------------+
        | Velocity-Informed Particle Filter|
        | - Position-only particles        |
        | - Geophysical measurements       |
        +----------------------------------+
                  |
                  v
          Enhanced position estimate
```

### Open-Loop Design

The particle filter operates in an **open-loop** manner, meaning it does not feed back corrections to the INS/GNSS filter. This design choice offers several advantages:

1. **Stability**: The INS/GNSS filter remains stable and unaffected by particle filter behavior
2. **Modularity**: Easy to enable/disable geophysical aiding without affecting primary navigation
3. **Experimentation**: Can compare INS/GNSS-only vs. geophysical-aided performance
4. **Safety**: Primary navigation is not compromised by potential particle filter issues

Future work may explore closed-loop architectures where the particle filter provides feedback.

## Key Features

### Reduced Dimensionality

- **Traditional Approach**: Full-state particles (9-15 DOF: position, velocity, attitude, biases)
- **Velocity-Informed Approach**: Position-only particles (3 DOF: latitude, longitude, altitude)

This dimensional reduction offers:
- **Computational Efficiency**: ~3-5x fewer states means significantly faster propagation
- **Fewer Particles Needed**: Position-only tracking requires fewer particles for same accuracy
- **Leverage Existing Estimates**: Uses high-quality velocity from Kalman filter

### Geophysical Measurement Integration

The particle filter can incorporate various geophysical anomaly measurements:

1. **Gravity Anomalies**
   - Free-air anomaly (requires velocity for Eötvös correction)
   - Bouguer anomaly
   - Isostatic anomaly
   
2. **Magnetic Anomalies**
   - Total field anomaly (measured - WMM reference)
   - Component anomalies
   
3. **Terrain Relief** (future)
   - Bathymetry for underwater navigation
   - Digital elevation models for airborne navigation

### Configurable Parameters

#### Number of Particles
- **Range**: 100 - 10,000
- **Typical**: 500 - 2,000 for real-time applications
- **Trade-off**: More particles = better approximation but higher computational cost

```rust
let pf = VelocityInformedParticleFilter::new(
    initial_position,
    1000,  // Recommended for real-time
    1.0,
);
```

#### Process Noise (Jitter)
- **Range**: 0.1 - 10.0 meters
- **Typical**: 0.5 - 2.0 meters
- **Purpose**: Maintains particle diversity during propagation
- **Trade-off**: 
  - Too low: Particle degeneracy (all particles converge to same state)
  - Too high: Loss of precision (particles spread too much)

```rust
pf.set_process_noise_std(1.0);  // 1 meter standard deviation
```

#### Resampling Strategy
- **Systematic** (default): Low variance, deterministic
- **Multinomial**: Simple, high variance
- **Stratified**: Balanced randomness
- **Residual**: Preserves high-weight particles

```rust
let pf = VelocityInformedParticleFilter::new_with_config(
    initial_position,
    1000,
    1.0,
    ParticleResamplingStrategy::Systematic,
    ParticleAveragingStrategy::WeightedMean,
    0.5,  // Resample when N_eff < 50% of particles
);
```

## Usage Guide

### Basic Workflow

```rust
use geonav::velocity_particle::VelocityInformedParticleFilter;
use nalgebra::DVector;

// 1. Initialize with position from INS/GNSS filter
let initial_position = DVector::from_vec(vec![
    ukf.get_estimate()[0],  // latitude (rad)
    ukf.get_estimate()[1],  // longitude (rad)
    ukf.get_estimate()[2],  // altitude (m)
]);

let mut pf = VelocityInformedParticleFilter::new(
    initial_position,
    1000,  // particles
    1.0,   // process noise (meters)
);

// 2. In your navigation loop:
for event in event_stream {
    match event {
        Event::Imu { dt_s, imu, .. } => {
            // Propagate INS/GNSS filter
            ukf.predict(&imu, dt_s);
            
            // Extract velocity
            let velocity = DVector::from_vec(vec![
                ukf.get_estimate()[3],  // v_n
                ukf.get_estimate()[4],  // v_e
                ukf.get_estimate()[5],  // v_d
            ]);
            
            // Propagate particle filter
            pf.propagate(&velocity, dt_s);
        }
        
        Event::Measurement { meas, .. } => {
            // Update INS/GNSS filter
            ukf.update(meas.as_ref());
            
            // If this is a geophysical measurement
            if let Some(geo_meas) = meas.downcast_ref::<GravityMeasurement>() {
                // Update particle weights
                pf.update_weights(geo_meas);
                pf.normalize_weights();
                
                // Resample if needed
                if pf.effective_sample_size() < 0.5 * pf.num_particles() as f64 {
                    pf.resample();
                }
            }
        }
    }
}

// 3. Get position estimates
let ukf_position = &ukf.get_estimate().rows(0, 3);
let pf_position = pf.get_estimate();
let pf_covariance = pf.get_covariance();
```

### Monitoring Performance

```rust
// Check effective sample size
let n_eff = pf.effective_sample_size();
println!("Effective particles: {:.0}/{}", n_eff, pf.num_particles());

// Rule of thumb: N_eff < 50% indicates degeneracy, time to resample
if n_eff < 0.5 * pf.num_particles() as f64 {
    pf.resample();
}

// Monitor position uncertainty
let cov = pf.get_covariance();
let pos_std_lat = cov[(0, 0)].sqrt().to_degrees() * 111_000.0;  // Convert to meters
let pos_std_lon = cov[(1, 1)].sqrt().to_degrees() * 111_000.0;
let pos_std_alt = cov[(2, 2)].sqrt();
println!("Position std: ({:.1}, {:.1}, {:.1}) m", pos_std_lat, pos_std_lon, pos_std_alt);
```

## Performance Characteristics

### Computational Cost

Approximate CPU time per update (relative to UKF):
- 100 particles: ~0.1x UKF time
- 500 particles: ~0.5x UKF time
- 1000 particles: ~1.0x UKF time
- 5000 particles: ~5.0x UKF time

### Memory Usage

Per particle overhead: ~40 bytes (position vector + weight)
- 1000 particles ≈ 40 KB
- 5000 particles ≈ 200 KB

### Accuracy

Position accuracy depends on:
1. **Geophysical map resolution**: Higher resolution = better position fix
2. **Geophysical measurement noise**: Lower noise = better weighting
3. **Number of particles**: More particles = better approximation
4. **Process noise tuning**: Proper tuning prevents degeneracy

Typical performance in good conditions:
- With gravity anomaly map (1' resolution): 50-200 m position accuracy
- With magnetic anomaly map (2' resolution): 100-500 m position accuracy

## Tuning Guidelines

### Selecting Number of Particles

Start with 1000 particles and adjust based on:
- **Increase** if: Position estimates are noisy or inconsistent
- **Decrease** if: Computational cost is too high and estimates are already good
- **Monitor**: Effective sample size should stay > 50% of total particles

### Tuning Process Noise

Process noise should match expected position uncertainty growth:

```
process_noise_std ≈ velocity_uncertainty * typical_dt
```

For example:
- Velocity uncertainty: 0.5 m/s
- Update rate: 10 Hz (dt = 0.1 s)
- Process noise: 0.5 * 0.1 = 0.05 m → Start with 0.1-0.5 m

Adjust based on particle behavior:
- **Particles converging too fast**: Increase noise
- **Particles too spread out**: Decrease noise
- **Check N_eff**: Should remain above 50% between resamples

### Resampling Threshold

Default: 0.5 (resample when N_eff < 50% of particles)

- **Lower threshold** (0.3-0.4): Less frequent resampling, faster but risks degeneracy
- **Higher threshold** (0.6-0.8): More frequent resampling, more robust but slower

## Limitations and Caveats

1. **Requires Geophysical Maps**: Cannot provide enhanced position without maps
2. **Open-Loop Only**: Current implementation does not feed back to INS/GNSS
3. **Geodetic Approximations**: Uses linearized coordinate transformations (valid for small displacements)
4. **Position-Only**: Does not estimate velocity or attitude
5. **Computational Cost**: Scales linearly with number of particles

## Future Enhancements

1. **Closed-Loop Feedback**: Allow particle filter to correct INS/GNSS filter
2. **Adaptive Particle Count**: Dynamically adjust based on uncertainty
3. **Map Matching**: Incorporate terrain/bathymetry constraints
4. **Multi-Modal Support**: Better handle ambiguous positions
5. **GPU Acceleration**: Parallel particle propagation for real-time performance

## References

1. Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems* (2nd ed.). Artech House.

2. Doucet, A., de Freitas, N., & Gordon, N. (Eds.). (2001). *Sequential Monte Carlo Methods in Practice*. Springer.

3. Canciani, A. J. (2017). *Magnetic Navigation Using Terrain-Referenced Data*. Ph.D. Dissertation, Air Force Institute of Technology.

4. Goldenberg, F. (2006). "Geomagnetic Navigation beyond the Magnetic Compass." *IEEE/ION PLANS*.

## Examples

See:
- `examples/velocity_particle_integration.rs` - API demonstration and usage patterns
- `examples/geonav-sim/` - Complete simulation with real data (requires HDF5)

## API Documentation

Full API documentation is available via:
```bash
cargo doc --open --package geonav
```

## Testing

Run tests:
```bash
cargo test --package geonav velocity_particle
```

## License

This implementation is part of the strapdown-rs project. See LICENSE file for details.
