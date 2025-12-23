# Extended Kalman Filter (EKF) Usage Guide

## Overview

The Extended Kalman Filter (EKF) has been added to the strapdown-rs library as an alternative to the Unscented Kalman Filter (UKF). The EKF provides a linearized Gaussian approximation for navigation state estimation using analytic Jacobians for efficient uncertainty propagation.

## Features

- **9-state and 15-state configurations**: Support for navigation-only (9-state) or navigation + IMU biases (15-state)
- **Analytic Jacobians**: Uses pre-computed Jacobians from the `linearize` module for efficient computation
- **Multiple measurement types**: Supports GPS position, GPS velocity, combined position+velocity, and barometric altitude
- **Comprehensive testing**: 16 unit tests and 5 integration tests validate correctness
- **Well-documented**: Extensive LaTeX/KaTeX doccomments explain the mathematical foundations

## API Usage

### Basic Initialization

```rust
use strapdown::kalman::{ExtendedKalmanFilter, InitialState, NavigationFilter};
use strapdown::measurements::GPSPositionMeasurement;
use strapdown::IMUData;
use nalgebra::{DMatrix, DVector, Vector3};

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

// Initialize 15-state EKF with biases
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // IMU biases (3 accel + 3 gyro)
    vec![1e-6; 15],  // Initial covariance diagonal
    DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 15])),  // Process noise
    true,  // use_biases = true for 15-state
);

// Predict step with IMU data
let imu_data = IMUData {
    accel: Vector3::new(0.0, 0.0, 9.81),
    gyro: Vector3::zeros(),
};
ekf.predict(imu_data, 0.01);  // dt = 0.01 seconds

// Update step with GPS measurement
let gps_meas = GPSPositionMeasurement {
    latitude: 45.0,
    longitude: -122.0,
    altitude: 100.0,
    horizontal_noise_std: 5.0,
    vertical_noise_std: 2.0,
};
ekf.update(&gps_meas);

// Get state estimate
let state = ekf.get_estimate();
let covariance = ekf.get_certainty();
```

### 9-State Configuration (No Biases)

```rust
// Initialize 9-state EKF without biases
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // Biases ignored when use_biases = false
    vec![1e-6; 9],  // Initial covariance for 9 states
    DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 9])),  // Process noise for 9 states
    false,  // use_biases = false for 9-state
);
```

### Using `initialize_ekf` Helper Function

For simulation workflows, use the `initialize_ekf` helper:

```rust
use strapdown::sim::{initialize_ekf, TestDataRecord};

// Assuming you have TestDataRecord from sensor data
let initial_pose = TestDataRecord::default();

let ekf = initialize_ekf(
    initial_pose,
    None,  // Use default attitude covariance
    None,  // Use default IMU biases
    None,  // Use default IMU bias covariance
    None,  // Use default process noise
    true,  // Use 15-state configuration
);
```

## Measurement Types

The EKF supports all measurement types from the `measurements` module:

### GPS Position
```rust
use strapdown::measurements::GPSPositionMeasurement;

let meas = GPSPositionMeasurement {
    latitude: 45.0,  // degrees
    longitude: -122.0,  // degrees
    altitude: 100.0,  // meters
    horizontal_noise_std: 5.0,  // meters
    vertical_noise_std: 2.0,  // meters
};
ekf.update(&meas);
```

### GPS Velocity
```rust
use strapdown::measurements::GPSVelocityMeasurement;

let meas = GPSVelocityMeasurement {
    northward_velocity: 10.0,  // m/s
    eastward_velocity: 5.0,  // m/s
    vertical_velocity: 0.0,  // m/s
    horizontal_noise_std: 0.5,  // m/s
    vertical_noise_std: 0.5,  // m/s
};
ekf.update(&meas);
```

### Combined GPS Position + Velocity
```rust
use strapdown::measurements::GPSPositionAndVelocityMeasurement;

let meas = GPSPositionAndVelocityMeasurement {
    latitude: 45.0,
    longitude: -122.0,
    altitude: 100.0,
    northward_velocity: 10.0,
    eastward_velocity: 5.0,
    horizontal_noise_std: 5.0,
    vertical_noise_std: 2.0,
    velocity_noise_std: 0.5,
};
ekf.update(&meas);
```

### Barometric Altitude
```rust
use strapdown::measurements::RelativeAltitudeMeasurement;

let meas = RelativeAltitudeMeasurement {
    relative_altitude: 5.0,  // meters above reference
    reference_altitude: 95.0,  // meters (reference altitude)
};
ekf.update(&meas);
```

## Command Line Usage (Simulator)

The EKF is now fully integrated into the simulator and can be used via command line:

```bash
# Run closed-loop simulation with EKF (linearized Jacobians)
strapdown-sim closed-loop --filter ekf --input data.csv --output results.csv

# Run with UKF (default, sigma point propagation)
strapdown-sim closed-loop --filter ukf --input data.csv --output results.csv

# With GNSS degradation config
strapdown-sim closed-loop --filter ekf --config gnss_config.toml --input data.csv --output results.csv

# View available options
strapdown-sim closed-loop --help
```

The `--filter` option accepts either `ukf` or `ekf`, with `ukf` as the default for backward compatibility.

## Performance Characteristics

### Advantages of EKF
- **Computational efficiency**: No sigma point generation/propagation
- **Deterministic**: Produces identical results on repeated runs
- **Lower memory**: Smaller footprint than UKF
- **Well-understood**: Decades of theory and applications

### Limitations
- **Linearization errors**: First-order Taylor approximation may introduce errors for highly nonlinear systems
- **Potential divergence**: Can diverge if linearization is poor or process noise underestimated
- **Gaussian assumption**: Like UKF, assumes Gaussian distributions

### EKF vs UKF Comparison

Based on integration tests with real data:
- **Similar accuracy**: EKF and UKF achieve comparable RMS horizontal errors (typically within 50% of each other)
- **Speed**: EKF is generally faster due to no sigma point propagation
- **Stability**: Both filters maintain bounded errors over extended periods
- **GNSS outages**: Both filters handle intermittent GNSS well with appropriate process noise tuning

## Testing

### Unit Tests
Run EKF-specific unit tests:
```bash
cargo test -p strapdown-core --lib kalman::tests::ekf
```

All 16 EKF unit tests cover:
- Construction (9-state and 15-state)
- Predict/update steps
- All measurement types
- Motion profiles (free fall, hover, horizontal motion)
- Covariance reduction
- Angle wrapping

### Integration Tests
Run EKF integration tests:
```bash
cargo test -p strapdown-core --test integration_tests test_ekf
```

Integration tests include:
- End-to-end 15-state EKF simulation
- 9-state configuration verification
- EKF vs UKF performance comparison
- GNSS outage handling
- Bounded drift vs dead reckoning

**Note**: Integration tests are marked `#[ignore]` as they require test data files. Remove the ignore attribute and provide data to run them.

## Default Parameters

### Process Noise
Default process noise diagonal (for 15-state):
```rust
const DEFAULT_PROCESS_NOISE: [f64; 15] = [
    1e-6,  // latitude
    1e-6,  // longitude
    1e-4,  // altitude
    1e-3,  // velocity north
    1e-3,  // velocity east
    1e-3,  // velocity down
    1e-5,  // roll
    1e-5,  // pitch
    1e-5,  // yaw
    1e-6,  // accel bias x
    1e-6,  // accel bias y
    1e-6,  // accel bias z
    1e-8,  // gyro bias x
    1e-8,  // gyro bias y
    1e-8,  // gyro bias z
];
```

### Initial Covariance
Recommended initial covariance based on sensor accuracy:
- Position: `(horizontal_accuracy * METERS_TO_DEGREES)²`
- Velocity: `(speed_accuracy)²`
- Attitude: `1e-9` (radians²)
- Accelerometer biases: `1e-3` (m/s²)²
- Gyroscope biases: `1e-8` (rad/s)²

## Mathematical Background

The EKF implementation follows the standard formulation:

### Predict Step
$$
\begin{aligned}
\bar{x}_{k+1} &= f(x_k, u_k) \\
\bar{P}_{k+1} &= F_k P_k F_k^T + G_k Q_k G_k^T
\end{aligned}
$$

where:
- $F_k$ is the state transition Jacobian computed using `linearize::state_transition_jacobian`
- $G_k$ is the process noise Jacobian computed using `linearize::process_noise_jacobian`
- $Q_k$ is the process noise covariance

### Update Step
$$
\begin{aligned}
K_k &= \bar{P}_k H_k^T (H_k \bar{P}_k H_k^T + R_k)^{-1} \\
x_k &= \bar{x}_k + K_k (z_k - h(\bar{x}_k)) \\
P_k &= (I - K_k H_k) \bar{P}_k
\end{aligned}
$$

where:
- $H_k$ is the measurement Jacobian (e.g., `linearize::gps_position_jacobian`)
- $R_k$ is the measurement noise covariance
- $K_k$ is the Kalman gain

For detailed mathematical derivations, see:
- Groves, P. D. "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition", Chapter 14.2
- In-code documentation with LaTeX/KaTeX equations

## Troubleshooting

### Filter Divergence
If the EKF diverges (unbounded errors):
1. Increase process noise (Q matrix)
2. Check IMU bias initialization
3. Verify coordinate frame consistency (ENU vs NED)
4. Ensure measurement noise is realistic

### Poor Performance vs UKF
If EKF significantly underperforms UKF:
1. Check for highly nonlinear motion (e.g., aggressive maneuvers)
2. Verify Jacobian computations are appropriate for your scenario
3. Consider using UKF for highly nonlinear regimes

### Numerical Issues
If experiencing numerical instability:
1. Covariance regularization is applied automatically (eps = 1e-9)
2. Joseph form covariance update is used for numerical stability
3. Robust SPD solver is used for Kalman gain computation

## Future Work

- CLI integration once particle filter stubs are resolved
- Example configurations and datasets
- Performance benchmarking suite
- Extended documentation with real-world examples
- Support for additional measurement types (pseudorange, carrier phase)

## References

1. Groves, P. D. (2013). "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition"
2. Bar-Shalom, Y., et al. (2001). "Estimation with Applications to Tracking and Navigation"
3. Simon, D. (2006). "Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches"
