# Integration Tests for INS Filters

This document describes the comprehensive integration tests for the strapdown inertial navigation system (INS) filters implemented in `core/tests/integration_tests.rs`.

## Overview

The integration tests validate the entire INS filter pipeline using real sensor data from the `sim/data/test_data.csv` file. Unlike unit tests that verify individual components, these tests ensure that the complete system works correctly with actual IMU and GNSS measurements.

## Test Data

The tests use real data collected from the Sensor Logger mobile application, which contains:
- IMU measurements (accelerometer and gyroscope) at ~100 Hz
- GNSS position and velocity measurements at ~1 Hz  
- Approximately 90 minutes of data with ~5366 samples

The dataset is located at: `sim/data/test_data.csv`

## Error Metrics

The tests use the following error metrics to evaluate filter performance:

### Horizontal Position Error
- **Metric**: Haversine distance between estimated and GNSS positions (meters)
- **Formula**: Great-circle distance on Earth's surface
- **Purpose**: Measures planar navigation accuracy

### Altitude Error
- **Metric**: Simple absolute difference (meters)
- **Purpose**: Measures vertical navigation accuracy

### Velocity Error
- **Metric**: Component-wise absolute differences for north, east, and down velocities (m/s)
- **Purpose**: Measures velocity estimation accuracy

## Test Suite

### 1. `test_dead_reckoning_on_real_data`

**Purpose**: Establish baseline performance for pure INS dead reckoning without GNSS corrections.

**What it tests**:
- Dead reckoning completes without crashes or errors
- Navigation solution remains finite (no NaN or Inf values)
- Provides baseline drift metrics for comparison

**Expected behavior**:
- Significant drift over time (this is normal for MEMS IMUs)
- All state values remain finite
- Errors grow unbounded (no error thresholds enforced)

**Typical results**:
- RMS horizontal error: ~7,000 km (expected drift for 90 minutes without corrections)
- Test serves as a baseline to demonstrate the value of GNSS-aided navigation

### 2. `test_ukf_closed_loop_on_real_data`

**Purpose**: Validate UKF performance with full-rate GNSS measurements.

**What it tests**:
- Closed-loop UKF completes successfully
- Position errors remain bounded
- Velocity and attitude estimates are stable
- All values remain finite

**Error thresholds**:
- RMS horizontal error < 30 m
- RMS altitude error < 20 m  
- Maximum horizontal error < 100 m

**Typical results**:
- RMS horizontal error: ~24 m
- RMS altitude error: ~4 m
- Mean velocity errors: <1 m/s

These are realistic values for consumer-grade MEMS IMU with smartphone GNSS.

### 3. `test_ukf_with_degraded_gnss`

**Purpose**: Validate UKF performance under degraded GNSS conditions (reduced update rate).

**What it tests**:
- UKF handles reduced GNSS update rate (5-second intervals)
- Errors are higher than full-rate but still bounded
- Filter doesn't diverge between GNSS updates

**Error thresholds**:
- RMS horizontal error < 50 m
- Maximum horizontal error < 600 m

**Typical results**:
- RMS horizontal error: ~28 m
- Maximum horizontal error: ~553 m
- Errors are larger due to drift between 5-second updates

### 4. `test_ukf_outperforms_dead_reckoning`

**Purpose**: Demonstrate that GNSS-aided UKF provides significant improvement over dead reckoning.

**What it tests**:
- UKF with GNSS has lower errors than dead reckoning
- GNSS corrections effectively bound error growth

**Typical results**:
- Dead reckoning RMS error: ~7,100 km
- UKF RMS error: ~24 m
- Improvement: >99.99%

## Running the Tests

### Run all integration tests:
```bash
cd core
cargo test --test integration_tests
```

### Run a specific test:
```bash
cd core
cargo test --test integration_tests test_ukf_closed_loop_on_real_data -- --nocapture
```

### Run with output visible:
```bash
cd core  
cargo test --test integration_tests -- --nocapture
```

### Run with single thread (sequential execution):
```bash
cd core
cargo test --test integration_tests -- --test-threads=1
```

## Expected Runtime

The integration tests process real sensor data and run complex filters, so they take longer than unit tests:

- `test_dead_reckoning_on_real_data`: ~5 seconds
- `test_ukf_closed_loop_on_real_data`: ~60-90 seconds
- `test_ukf_with_degraded_gnss`: ~60-90 seconds
- `test_ukf_outperforms_dead_reckoning`: ~120-180 seconds

Total runtime: ~4-5 minutes

## Implementation Details

### Error Metric Calculation

The `compute_error_metrics()` function:
1. Matches navigation results to GNSS measurements by timestamp
2. Skips invalid GNSS data (NaN values)
3. Computes error for each matched sample
4. Filters out non-finite errors
5. Calculates mean, max, and RMS statistics

### Filter Initialization

Tests use realistic initialization based on first GNSS measurement:
- Position: First GNSS lat/lon/alt
- Velocity: Computed from GNSS speed and bearing
- Attitude: From phone orientation sensors
- Covariances: Conservative initial uncertainties
- Process noise: Tuned for MEMS IMU characteristics

### Event Stream Generation

Tests use the `build_event_stream()` function to create a sequence of IMU propagation and GNSS update events from the raw data, with configurable scheduling and fault injection.

## Bug Fixes

During test implementation, the following bugs were discovered and fixed:

1. **Dead reckoning conversion bug** (sim.rs:749, 768):
   - Issue: Used wrong `From` implementation, converting to 15-element DVector instead of using StrapdownState directly
   - Fix: Changed to use `From<(&DateTime<Utc>, &StrapdownState)>` implementation
   - Impact: Dead reckoning now works correctly

## Future Enhancements

Potential improvements for the test suite:

1. **Additional test scenarios**:
   - GNSS outages (DutyCycle scheduler)
   - Measurement corruption (fault models)
   - Different motion profiles

2. **More filters**:
   - Particle filter integration tests
   - Extended Kalman Filter (when implemented)

3. **Performance benchmarks**:
   - Execution time metrics
   - Memory usage tracking
   - Scalability tests

4. **Shorter test datasets**:
   - Create focused test datasets for faster CI/CD
   - Keep full dataset for comprehensive validation

## References

- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed.
- Sensor Logger app: https://www.tszheichoi.com/sensorlogger
