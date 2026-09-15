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

// Initial covariance diagonal.
//
// Latitude and longitude are held in RADIANS and altitude in metres, so the three position
// entries are not in the same unit and cannot be filled from one literal. `vec![1e-6; 15]`
// -- what this example used to show -- reads as a 6.4 km initial horizontal uncertainty, not
// as a small number; see "Default Parameters" below and issue #308. Write the uncertainty in
// metres once and convert where it is used.
let horizontal_std_rad = 10.0 * strapdown::earth::METERS_TO_RADIANS;  // a 10 m GNSS fix
let mut initial_covariance = vec![
    horizontal_std_rad.powi(2),  // latitude, rad^2
    horizontal_std_rad.powi(2),  // longitude, rad^2
    10.0_f64.powi(2),            // altitude, m^2
];
initial_covariance.extend([0.25; 3]);   // velocity, (m/s)^2   -- a 0.5 m/s fix
initial_covariance.extend([1e-4; 3]);   // attitude, rad^2     -- ~0.6 deg
initial_covariance.extend([1e-3; 3]);   // accel bias, (m/s^2)^2
initial_covariance.extend([1e-8; 3]);   // gyro bias, (rad/s)^2

// Initialize 15-state EKF with biases
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // IMU biases (3 accel + 3 gyro)
    initial_covariance.clone(),  // cloned so the 9-state example below can reuse it
    // Process noise. The crate's own default, which is built the same way -- one metric
    // constant converted once. `vec![1e-9; 15]` is a 201 m per-step horizontal term.
    DMatrix::from_diagonal(&DVector::from_vec(
        strapdown::sim::DEFAULT_PROCESS_NOISE.to_vec(),
    )),
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
// The leading nine entries of the 15-state diagonal built above -- the same unit caveat
// applies, and for the same reason.
let initial_covariance_9 = initial_covariance[0..9].to_vec();

// Initialize 9-state EKF without biases
let mut ekf = ExtendedKalmanFilter::new(
    initial_state,
    vec![0.0; 6],  // Biases ignored when use_biases = false
    initial_covariance_9,  // Initial covariance for 9 states
    DMatrix::from_diagonal(&DVector::from_vec(
        strapdown::sim::DEFAULT_PROCESS_NOISE[0..9].to_vec(),
    )),  // Process noise for 9 states
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

# Run with UKF (sigma point propagation)
strapdown-sim closed-loop --filter ukf --input data.csv --output results.csv

# Run with the ESKF (the default; omitting --filter selects it)
strapdown-sim closed-loop --filter eskf --input data.csv --output results.csv

# With GNSS degradation config
strapdown-sim closed-loop --filter ekf --config gnss_config.toml --input data.csv --output results.csv

# View available options
strapdown-sim closed-loop --help
```

The `--filter` option accepts `eskf`, `ukf` or `ekf`. Since #258 the default is `eskf`, the
15-state error-state filter: it estimates the IMU biases online rather than carrying whatever
turn-on bias the sensor happened to have, which the 9-state UKF and EKF cannot observe.

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
Default process noise diagonal (for 15-state). Note that the three position entries are
**not** in the same unit as one another: the filter holds latitude and longitude in radians
and altitude in metres, so a horizontal uncertainty written in metres has to be converted
before it can go on the diagonal. The crate does that once, from a single metric constant:

```rust
// strapdown::sim
pub const POSITION_PROCESS_NOISE_M: f64 = 0.1; // metres, per step

const HORIZONTAL: f64 = {
    let radians = POSITION_PROCESS_NOISE_M * strapdown::earth::METERS_TO_RADIANS;
    radians * radians // 2.467e-16 rad^2
};

const DEFAULT_PROCESS_NOISE: [f64; 15] = [
    HORIZONTAL,  // latitude, rad^2
    HORIZONTAL,  // longitude, rad^2
    1e-4,  // altitude, m^2 -- its own constant, see below
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

The altitude entry keeps its own constant, `VERTICAL_POSITION_PROCESS_NOISE_M2 = 1e-4`
(a 1 cm per-step standard deviation), rather than `POSITION_PROCESS_NOISE_M` squared.
That entry was already in metres and already meant what it said, so #308 left it alone:
a units fix is not the place to retune the vertical channel. Whether 1 cm per step is the
right *tuning* is a fair question and a separate one.

If you write the horizontal entries directly in rad^2, be aware what the numbers mean:
`1e-6 rad^2` is a **6.4 km** per-step standard deviation, not a small number. Writing it
next to an altitude term of `1e-4 m^2` (1 cm) is the defect issue #308 fixed -- the filter
is told its own prediction is worthless, so it discards it and lands on each fix instead of
filtering, and innovation gating cannot function. `1e-9 rad^2` is the same mistake three
orders of magnitude smaller: a 201 m per-step standard deviation.

### Initial Covariance
Everything above is about $Q$, and every word of it applies to $P_0$: latitude and longitude
are radians there too. `1e-6` as an initial variance is a 6.4 km initial horizontal
uncertainty, which is how #303 first noticed the problem.

Recommended initial covariance based on sensor accuracy. Convert metres to **radians**, not
degrees -- `METERS_TO_DEGREES` alone leaves the value 57.3x too large as a standard deviation
and 3283x in variance:

- Position: `(horizontal_accuracy * METERS_TO_RADIANS)²` for latitude and longitude,
  `vertical_accuracy²` for altitude
- Velocity: `(speed_accuracy)²`
- Attitude: `1e-9` (radians²) is what the `initialize_*` helpers use, which assumes the
  record's own attitude is trusted. Seeding from a coarse alignment instead, use something
  like the `1e-4` rad² (~0.6°) in the worked example above -- the two differ by five orders
  of magnitude because they describe different situations, not because one is wrong
- Accelerometer and gyroscope biases: `1e-3`, all six entries

`sim::initialize_ekf` and `sim::initialize_ukf` build the position and velocity blocks from a
`TestDataRecord`'s own reported accuracies and the rest from those defaults, and `sim::initialize_eskf` and
`engine::InsEngine` use `sim::DEFAULT_INITIAL_POSITION_UNCERTAINTY_M` (10 m) when there is no
record to read. For a derivation from an IMU grade rather than a hand-picked constant, see
`IMUQuality::auto_covariance`.

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
