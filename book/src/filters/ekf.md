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
    is_enu: false, // NED, the crate default; set true for ENU data
};

// Initial covariance diagonal.
//
// Latitude and longitude are held in RADIANS and altitude in metres, so the three position
// entries are not the same unit and cannot come from one literal. `vec![1e-6; 9]` -- what
// this example used to show -- reads as a 6.4 km initial horizontal uncertainty rather than
// as a small number, and `1e-9` as a process-noise density is a 201 m one. Both are issue
// #308; see **Units on the covariance diagonals** below. Write the metres once, convert where
// they are used.
let horizontal_std_rad = 10.0 * strapdown::earth::METERS_TO_RADIANS;  // a 10 m GNSS fix
let mut initial_covariance = vec![
    horizontal_std_rad.powi(2),  // latitude, rad^2
    horizontal_std_rad.powi(2),  // longitude, rad^2
    10.0_f64.powi(2),            // altitude, m^2
];
initial_covariance.extend([0.25; 3]);  // velocity, (m/s)^2 -- a 0.5 m/s fix
initial_covariance.extend([1e-4; 3]);  // attitude, rad^2   -- ~0.6 deg

// Initialize 9-state EKF (no biases)
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![],  // No biases for 9-state
    initial_covariance,
    // The crate's own default, which is built the same way: one metric constant, converted
    // once. Nine-state filters take its leading nine entries.
    DMatrix::from_diagonal(&DVector::from_vec(
        strapdown::sim::DEFAULT_PROCESS_NOISE_DENSITY[0..9].to_vec(),
    )),
    false,  // use_biases = false for 9-state
);
```

### 15-State with Bias Estimation

```rust
// The nine-state diagonal built above, extended with the bias states. The same unit
// caveat applies to its position block, and for the same reason.
let mut initial_covariance_15 = initial_covariance.clone();
initial_covariance_15.extend([1e-3; 3]);  // accel bias, (m/s^2)^2
initial_covariance_15.extend([1e-8; 3]);  // gyro bias, (rad/s)^2

// Initialize 15-state EKF with bias estimation
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // Initial bias estimates (3 accel + 3 gyro)
    initial_covariance_15,
    DMatrix::from_diagonal(&DVector::from_vec(
        strapdown::sim::DEFAULT_PROCESS_NOISE_DENSITY.to_vec(),
    )),  // Process noise
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

## Units on the covariance diagonals

The state vector is not in one unit system, and neither $P_0$ nor $Q$ can be filled from a
single literal. Latitude and longitude are **radians**; altitude, velocity and the
accelerometer biases are metric; attitude and the gyroscope biases are radians and radians
per second. A horizontal uncertainty written in metres therefore has to pass through
`earth::METERS_TO_RADIANS` before it can go on the diagonal.

It is worth knowing what the round numbers mean once the conversion is skipped:

| written as a variance | as a horizontal standard deviation |
|---|---|
| `1e-6` rad² | 6367 m |
| `1e-9` rad² | 201 m |
| `(5 m × METERS_TO_DEGREES)²` | 286 m (degrees, not radians -- 57.3x too large) |

All three shipped in this crate, in $Q$ and in $P_0$, and are what issue #308 fixed. The
symptom is characteristic: with $Q$ that large the innovation covariance $S = HPH^T + R$ is
dominated by the filter's own prediction, so the update discards it and lands on each fix,
the solution tracks the fix noise one-for-one instead of averaging it down, and innovation
gating cannot function because a genuinely bad fix is still inside what the filter believes
possible.

`sim::DEFAULT_PROCESS_NOISE_DENSITY`, `sim::DEFAULT_INITIAL_POSITION_UNCERTAINTY_M` and the
`sim::initialize_*` helpers all do the conversion for you; `IMUQuality::auto_covariance`
derives a whole $P_0$ diagonal from an IMU grade and a reported fix accuracy.

## Best Practices

1. **Start with 9-state** unless you need bias estimation
2. **Tune conservatively**: Start with larger uncertainties and reduce
3. **Monitor innovation**: Check measurement residuals for divergence
4. **Use 15-state** for long-duration missions or low-quality IMUs
5. **Validate with dead reckoning**: Compare against open-loop results
6. **Write position uncertainties in metres** and convert once -- see
   [Units on the covariance diagonals](#units-on-the-covariance-diagonals)

## Next Steps

- Try the [UKF](./ukf.md) for comparison
- Learn about [Measurement Models](./measurements.md)
- See [Example: Closed-Loop Simulation](../examples/tutorial-basic.md)
