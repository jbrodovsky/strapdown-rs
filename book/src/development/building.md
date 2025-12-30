# Building and Testing

## Building the Project

### System Dependencies

The project requires the following system libraries:
- `pkg-config`
- `libhdf5-dev` and `libhdf5-openmpi-dev` (for HDF5 support)
- `libnetcdf-dev` (for NetCDF geophysical data)
- `zlib1g-dev` (for compression support)

On Ubuntu/Debian systems:
```bash
sudo apt update
sudo apt install -y pkg-config libhdf5-dev libhdf5-openmpi-dev libnetcdf-dev zlib1g-dev
```

### Rust Toolchain
- Minimum Rust version: 1.70+ (stable channel)
- Required components: `clippy`, `rustfmt`

### Building

Build the entire workspace:
```bash
cargo build --workspace --all-features
```

Build a specific crate:
```bash
cargo build -p strapdown-core
cargo build -p strapdown-sim
cargo build -p strapdown-geonav
```

Build with optimizations (release mode):
```bash
cargo build --workspace --all-features --release
```

### Installing Binaries

Install the simulation binaries to your system:
```bash
# Install strapdown-sim
cargo install --path sim

# Install geonav-sim
cargo install --path geonav
```

## Testing

### Overview

The strapdown-rs project includes comprehensive unit tests, integration tests, and module-specific tests to validate the correctness of the navigation algorithms.

### Running Tests

Run all tests in the workspace:
```bash
cargo test --workspace --all-features --verbose
```

Run tests for a specific crate:
```bash
cargo test -p strapdown-core
cargo test -p strapdown-sim
cargo test -p strapdown-geonav
```

Run with output visible:
```bash
cargo test -- --nocapture
```

Run tests sequentially (single-threaded):
```bash
cargo test -- --test-threads=1
```

### Test Organization

Tests are organized into:
- **Unit tests**: Inline in source files using `#[cfg(test)]` modules
- **Integration tests**: Located in `core/tests/integration_tests.rs`
- **Module tests**: Specific to individual modules

### Integration Tests for INS Filters

The integration tests validate the entire INS filter pipeline using real sensor data. These tests are comprehensive and take longer to run than unit tests.

#### Test Data

The integration tests use real data collected from the Sensor Logger mobile application:
- IMU measurements (accelerometer and gyroscope) at ~100 Hz
- GNSS position and velocity measurements at ~1 Hz  
- Approximately 90 minutes of data with ~5366 samples
- Dataset location: `sim/data/test_data.csv`

#### Error Metrics

##### Horizontal Position Error
- **Metric**: Haversine distance between estimated and GNSS positions (meters)
- **Formula**: Great-circle distance on Earth's surface
- **Purpose**: Measures planar navigation accuracy

##### Altitude Error
- **Metric**: Simple absolute difference (meters)
- **Purpose**: Measures vertical navigation accuracy

##### Velocity Error
- **Metric**: Component-wise absolute differences for north, east, and down velocities (m/s)
- **Purpose**: Measures velocity estimation accuracy

#### Test Suite

##### 1. `test_dead_reckoning_on_real_data`

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

##### 2. `test_ukf_closed_loop_on_real_data`

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

##### 3. `test_ukf_with_degraded_gnss`

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

##### 4. `test_ukf_outperforms_dead_reckoning`

**Purpose**: Demonstrate that GNSS-aided UKF provides significant improvement over dead reckoning.

**What it tests**:
- UKF with GNSS has lower errors than dead reckoning
- GNSS corrections effectively bound error growth

**Typical results**:
- Dead reckoning RMS error: ~7,100 km
- UKF RMS error: ~24 m
- Improvement: >99.99%

#### Running Integration Tests

Run all integration tests:
```bash
cd core
cargo test --test integration_tests
```

Run a specific test:
```bash
cd core
cargo test --test integration_tests test_ukf_closed_loop_on_real_data -- --nocapture
```

Run with output visible:
```bash
cd core  
cargo test --test integration_tests -- --nocapture
```

Run with single thread (sequential execution):
```bash
cd core
cargo test --test integration_tests -- --test-threads=1
```

#### Expected Runtime

The integration tests process real sensor data and run complex filters:

- `test_dead_reckoning_on_real_data`: ~5 seconds
- `test_ukf_closed_loop_on_real_data`: ~60-90 seconds
- `test_ukf_with_degraded_gnss`: ~60-90 seconds
- `test_ukf_outperforms_dead_reckoning`: ~120-180 seconds

**Total runtime**: ~4-5 minutes

#### Implementation Details

##### Error Metric Calculation

The `compute_error_metrics()` function:
1. Matches navigation results to GNSS measurements by timestamp
2. Skips invalid GNSS data (NaN values)
3. Computes error for each matched sample
4. Filters out non-finite errors
5. Calculates mean, max, and RMS statistics

##### Filter Initialization

Tests use realistic initialization based on first GNSS measurement:
- Position: First GNSS lat/lon/alt
- Velocity: Computed from GNSS speed and bearing
- Attitude: From phone orientation sensors
- Covariances: Conservative initial uncertainties
- Process noise: Tuned for MEMS IMU characteristics

##### Event Stream Generation

Tests use the `build_event_stream()` function to create a sequence of IMU propagation and GNSS update events from the raw data, with configurable scheduling and fault injection.

### Future Test Enhancements

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

## Linting and Formatting

### Running Clippy

Clippy provides lint checks for Rust code:
```bash
cargo clippy --workspace --all-features
```

Address warnings and errors before committing.

### Running rustfmt

Format code with rustfmt:
```bash
cargo fmt --all
```

Check formatting without making changes:
```bash
cargo fmt --all -- --check
```

## Documentation

### Building API Documentation

Generate documentation for all crates:
```bash
cargo doc --workspace --all-features --no-deps
```

Open the documentation in a browser:
```bash
cargo doc --workspace --all-features --no-deps --open
```

### Building the Book

This user guide is built using mdBook. Install mdBook:
```bash
cargo install mdbook
```

Build the book:
```bash
cd book
mdbook build
```

Serve the book locally for development:
```bash
cd book
mdbook serve
```

Then open http://localhost:3000 in your browser.

## Continuous Integration

The project uses GitHub Actions for continuous integration. See `.github/workflows/` for workflow definitions:

- `rust.yml`: Builds and tests the project
- `deploy-book.yml`: Builds and deploys this documentation to GitHub Pages
- `publish.yml`: Publishes crates to crates.io

## References

- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed.
- Sensor Logger app: https://www.tszheichoi.com/sensorlogger
