# EKF-based Geophysical Navigation

This document describes the Extended Kalman Filter (EKF) implementation for geophysical navigation in the `strapdown-rs` project.

## Overview

The EKF-based geophysical INS provides an alternative to the Unscented Kalman Filter (UKF) for integrating inertial navigation with geophysical measurements (gravity and magnetic anomalies). While the UKF uses sigma points to propagate uncertainty through nonlinear transformations, the EKF uses first-order Taylor series approximations (Jacobians) to linearize the system dynamics and measurement models.

## Features

- **15-state EKF**: Position (3), velocity (3), attitude (3), and IMU biases (6)
- **Geophysical measurements**: Gravity and magnetic anomaly measurements from geophysical maps
- **Numerical Jacobians**: Map gradients computed using finite differences for measurement updates
- **Health monitoring**: Integrated with existing health check framework
- **Compatible with UKF**: Drop-in replacement via CLI flag

## Implementation Details

### State Vector

The EKF uses a 15-dimensional state vector (when biases are enabled):

```
x = [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
```

Where:
- `lat, lon, alt`: Position in geodetic coordinates (radians, radians, meters)
- `v_n, v_e, v_d`: Velocity in local-level frame (m/s)
- `roll, pitch, yaw`: Attitude Euler angles (radians)
- `b_ax, b_ay, b_az`: Accelerometer biases (m/s²)
- `b_gx, b_gy, b_gz`: Gyroscope biases (rad/s)

### Predict Step

The predict step propagates the state and covariance using:

1. **Nonlinear state propagation**: Uses the `forward()` function to integrate IMU measurements through the strapdown mechanization equations
2. **Linearized covariance propagation**: $\bar{P} = F P F^T + Q$ where:
   - $F$ is the state transition Jacobian (computed in `core/src/linearize.rs`)
   - $Q$ is the process noise covariance

### Update Step

The update step incorporates measurements to correct the state estimate:

1. **Measurement prediction**: $\hat{z} = h(\bar{x})$
2. **Innovation covariance**: $S = H \bar{P} H^T + R$
3. **Kalman gain**: $K = \bar{P} H^T S^{-1}$
4. **State correction**: $x = \bar{x} + K(z - \hat{z})$
5. **Covariance update**: $P = (I - KH)\bar{P}(I - KH)^T + KRK^T$ (Joseph form)

Where:
- $H$ is the measurement Jacobian
- $R$ is the measurement noise covariance

### Geophysical Measurements

#### Gravity Anomaly Measurements

Gravity measurements depend on:
- Position (lat, lon) for map lookup
- Velocity (for Eotvos correction)
- Altitude (for free-air correction)

The measurement Jacobian is computed numerically using finite differences on the geophysical map:

```rust
let (dlat, dlon) = map.get_gradient(&lat, &lon, epsilon);
```

#### Magnetic Anomaly Measurements  

Magnetic anomaly measurements depend on:
- Position (lat, lon) for map lookup
- Altitude (for reference field computation via WMM)
- Date (for WMM temporal variation)

Similar to gravity, the Jacobian is computed numerically from the map gradient.

### Numerical Gradient Computation

The `GeoMap::get_gradient()` method computes map gradients using central finite differences:

```
∂z/∂lat ≈ (z(lat+ε) - z(lat-ε)) / (2ε)
∂z/∂lon ≈ (z(lon+ε) - z(lon-ε)) / (2ε)
```

With automatic fallback to forward/backward differences at map boundaries.

## Usage

### Command Line

```bash
# Run with EKF (instead of default UKF)
strapdown-geonav \
  --input sensor_data.csv \
  --output navigation_solution.csv \
  --filter ekf \
  --geo-type gravity \
  --geo-noise-std 100.0

# Run with magnetic anomalies
strapdown-geonav \
  --input sensor_data.csv \
  --output navigation_solution.csv \
  --filter ekf \
  --geo-type magnetic \
  --geo-noise-std 50.0
```

### Programmatic API

```rust
use strapdown::kalman::{ExtendedKalmanFilter, InitialState};
use geonav::{GeoMap, build_event_stream, geo_closed_loop_ekf};

// Initialize EKF
let initial_state = InitialState { /* ... */ };
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // IMU biases
    cov_diagonal,
    process_noise,
    true,  // use 15-state with biases
);

// Load geophysical map
let geomap = GeoMap::load_geomap(map_path, measurement_type)?;

// Build event stream
let events = build_event_stream(&records, &config, geomap, noise_std, frequency);

// Run EKF navigation
let results = geo_closed_loop_ekf(&mut ekf, events)?;
```

## Performance Comparison: EKF vs UKF

| Aspect | EKF | UKF |
|--------|-----|-----|
| **Computational Cost** | Lower (linear algebra only) | Higher (sigma point propagation) |
| **Memory Usage** | Lower | Higher (stores sigma points) |
| **Accuracy** | First-order approximation | Second-order approximation |
| **Nonlinearity Handling** | Good for mildly nonlinear systems | Better for highly nonlinear systems |
| **Deterministic** | Yes | Yes |
| **Implementation Complexity** | Requires Jacobian computation | Requires careful sigma point selection |

### When to Use EKF

- Computational resources are limited
- System dynamics are mildly nonlinear
- Jacobians can be computed efficiently
- Faster iteration cycles are needed for development

### When to Use UKF

- High accuracy is paramount
- System has significant nonlinearities
- Jacobians are difficult to compute or verify
- Computational resources are available

## Testing

The EKF geophysical navigation includes comprehensive tests:

```bash
# Run all core EKF tests (31 tests)
cargo test --package strapdown-core --lib kalman

# Run geophysical EKF tests (17 tests including Jacobian tests)
cargo test --package strapdown-geonav --lib
```

Key test coverage:
- EKF construction (9-state and 15-state)
- Predict step with various motion profiles
- Update step with different measurement types
- Geophysical measurement Jacobians
- Map gradient computation
- Angle wrapping and covariance reduction

## References

1. Groves, P. D. "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition", Chapter 14.2
2. Bar-Shalom, Y., et al. "Estimation with Applications to Tracking and Navigation", Chapter 5
3. Julier, S. & Uhlmann, J. "Unscented Filtering and Nonlinear Estimation" (for UKF comparison)

## Future Enhancements

Potential improvements to the EKF geophysical navigation:

1. **Analytic Jacobians**: Compute analytic gradients for bilinear map interpolation
2. **Adaptive Process Noise**: Adjust Q based on motion dynamics
3. **Map Uncertainty**: Incorporate geophysical map uncertainty into R
4. **Multi-resolution Maps**: Support hierarchical map representations
5. **Terrain-aided Navigation**: Add terrain elevation as additional measurement
6. **Error-State Formulation**: Implement error-state EKF for improved numerical stability
