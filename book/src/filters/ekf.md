# Extended Kalman Filter (EKF)

The Extended Kalman Filter (EKF) is one of the primary navigation filters available in Strapdown-rs. It provides efficient state estimation for the INS through linearization of nonlinear system and measurement models.

## Overview

The EKF linearizes the nonlinear navigation equations using Jacobian matrices (first-order Taylor series expansion). While this approximation is less accurate than the UKF's unscented transform for highly nonlinear systems, it is computationally more efficient and works well for most practical navigation scenarios.

## Mathematical Foundation

### State Vector

The EKF supports two state configurations:

**9-State Model (Navigation Only)**:
- Latitude, longitude, altitude
- North, east, down velocities
- Roll, pitch, yaw angles

**15-State Model (Navigation + Biases)**:
- 9 navigation states (as above)
- Accelerometer biases (3)
- Gyroscope biases (3)

### Prediction Step

The prediction step propagates the state and covariance forward using IMU measurements using the strapdown mechanization and computed Jacobian matrices.

### Update Step

When GNSS or other measurements are available, the Kalman gain is computed and the state is updated with the measurement innovation.

## Features

### Advantages

1. **Computational Efficiency**: 3-5x faster than UKF
2. **Memory Efficient**: Stores only mean and covariance
3. **Well-Understood**: Extensive literature and proven track record
4. **Analytic Jacobians**: Uses pre-computed derivatives for accuracy

### Limitations

1. **Linearization Error**: Less accurate for highly nonlinear systems
2. **First-Order Approximation**: May miss higher-order effects
3. **Gaussian Assumption**: Cannot handle multimodal distributions

## Usage

### Basic Initialization

```rust
use strapdown::kalman::{ExtendedKalmanFilter, InitialState, NavigationFilter};
use nalgebra::{DMatrix, DVector};

// Define initial state
let initial_state = InitialState {
    latitude: 45.0,
    longitude: -122.0,
    altitude: 100.0,
    northward_velocity: 0.0,
    eastward_velocity: 0.0,
    vertical_velocity: 0.0,
    roll: 0.0,
    pitch: 0.0,
    yaw: 0.0,
    in_degrees: true,
    is_enu: true,
};

// Initialize 9-state EKF (no biases)
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![],  // No biases for 9-state
    vec![1e-6; 9],  // Initial covariance diagonal
    DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 9])),  // Process noise
    false,  // use_biases = false for 9-state
);
```

### 15-State with Bias Estimation

```rust
// Initialize 15-state EKF with bias estimation
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // Initial bias estimates (3 accel + 3 gyro)
    vec![1e-6; 15],  // Initial covariance diagonal
    DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 15])),  // Process noise
    true,  // use_biases = true for 15-state
);
```

### Prediction with IMU Data

```rust
use strapdown::IMUData;

let imu_data = IMUData {
    timestamp: 1234567890.0,
    gyro: [0.01, -0.02, 0.03],  // rad/s
    accel: [0.5, -0.2, 9.81],    // m/s²
};

let dt = 0.01;  // Time step in seconds
ekf.predict(&imu_data, dt);
```

### Update with GNSS

```rust
use strapdown::measurements::GPSPositionMeasurement;

let gps_measurement = GPSPositionMeasurement {
    latitude: 45.00012,
    longitude: -122.00015,
    altitude: 101.5,
    std_dev: [5.0, 5.0, 10.0],  // Measurement uncertainty
};

ekf.update(&gps_measurement);
```

## Performance

### Computational Complexity

- Typical update rate: 10,000-20,000 updates/second on modern hardware
- 3-5x faster than UKF
- Memory usage: ~2 KB for 15-state

## Comparison with UKF

| Aspect | EKF | UKF |
|--------|-----|-----|
| **Speed** | Faster (3-5x) | Slower |
| **Accuracy** | Good for mildly nonlinear | Better for highly nonlinear |
| **Implementation** | Requires Jacobians | No Jacobians needed |
| **Memory** | Lower | Higher |
| **Best For** | Real-time systems | Research/offline processing |

See [EKF vs UKF Comparison](./comparison.md) for detailed analysis.

## Best Practices

1. **Start with 9-state** unless you need bias estimation
2. **Tune conservatively**: Start with larger uncertainties and reduce
3. **Monitor innovation**: Check measurement residuals for divergence
4. **Use 15-state** for long-duration missions or low-quality IMUs
5. **Validate with dead reckoning**: Compare against open-loop results

## Next Steps

- Try the [UKF](./ukf.md) for comparison
- Learn about [Measurement Models](./measurements.md)
- See [Example: Closed-Loop Simulation](../examples/tutorial-basic.md)
