# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strapdown-rs is an open-source Rust implementation of strapdown inertial navigation system (INS) algorithms. This repository implements the **Free Core** version of the product, providing researchers, students, and developers with a lightweight but reproducible simulation tool for GNSS-denied scenarios. The architecture is designed with extensibility in mind to support future commercial professional versions.

**Product Philosophy**:
- **Free Core (this repo)**: Open-source academic/research tool focused on reproducible simulations, basic sensor models, and GNSS degradation scenarios
- **Professional/Enterprise** (future): Commercial versions with advanced GNSS modeling, synthetic data generation, GUIs, and hardware integration
- **Experimental features**: Geophysical navigation capabilities (gravity/magnetic anomaly aiding) remain in this repo as experimental features for research purposes

## Architecture

This is a Cargo workspace with three main crates:

### 1. `strapdown-core` (/core)
The core library implementing strapdown INS algorithms and simulation framework:
- **lib.rs**: Library entry point and 9-state strapdown mechanization in local-level frame (NED). Implements forward propagation equations from Groves textbook (Chapter 5.4-5.5)
- **earth.rs**: WGS84 Earth ellipsoid model and geodetic calculations
- **kalman.rs**: Kalman-style navigation filters including Unscented Kalman Filter (UKF) for nonlinear state estimation
- **particle.rs**: Particle filter (Sequential Monte Carlo) implementation for non-Gaussian estimation with resampling strategies
- **measurements.rs**: Measurement models (GPS position/velocity, barometric altitude, pseudorange, carrier phase) implementing the `MeasurementModel` trait
- **messages.rs**: Event stream handling for GNSS scheduling and fault injection scenarios
- **sim.rs**: Simulation utilities, CSV data loading (Sensor Logger format), dead reckoning and closed-loop functions
- **linalg.rs**: Linear algebra utilities for matrix operations

**Key state representation**: 9-state vector [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw] in NED frame

**Design principles for Free Core**:
- Deterministic simulations with reproducible results (seeded RNG)
- Configuration-driven (YAML/JSON configs for scenarios)
- Extensible architecture for future professional features
- Clean separation between core algorithms and simulation framework

### 2. `strapdown-sim` (/sim)
Command-line tool for running INS simulations with GNSS degradation:
- Modes: open-loop (dead reckoning), closed-loop with UKF, or particle filter
- GNSS fault simulation: dropouts, reduced update rates, measurement corruption, bias injection
- Input: CSV files with IMU and GNSS measurements (Sensor Logger format)
- Output: Navigation solutions as CSV/Parquet
- Configuration: YAML/JSON scenario files or command-line arguments
- Built-in logging: Use `--log-level` and `--log-file` flags (see LOGGING.md for details)

**Free Core scope**: Basic GNSS degradation (outages, noise, reduced availability)
**Future Pro scope**: Advanced faults (spoofing, jamming, multipath, terrain masking)

### 3. `strapdown-geonav` (/geonav)
**Experimental** geophysical navigation module (research-grade):
- Loads NetCDF geophysical maps (gravity/magnetic anomaly grids)
- Integrates geophysical measurements with INS/GNSS filters
- Provides alternative PNT in GNSS-denied environments
- Built-in logging: Use `--log-level` and `--log-file` flags (see LOGGING.md for details)
- Status: Experimental feature for research, may be commercialized in future roadmap

## Common Commands

### Build & Test
```bash
# Build entire workspace in release mode
pixi run build
# Or: cargo build --workspace --release

# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test --package strapdown-core
cargo test --package strapdown-sim

# Run specific test
cargo test --package strapdown-core test_name

# Run with code coverage
pixi run coverage
# Or: cargo tarpaulin --workspace --timeout 600
```

### Lint & Format
```bash
# Run linting (includes Python ruff and Rust clippy)
pixi run lint

# Format code (Python and Rust)
pixi run fmt
```

### Running Simulations
```bash
# Open-loop (dead reckoning)
./target/release/strapdown-sim -i data/input/input.csv -o output.csv open-loop

# Closed-loop with GNSS degradation
./target/release/strapdown-sim -i data/input/input.csv -o output.csv closed-loop \
  --seed 42 \
  --dropout-start-s 100.0 --dropout-duration-s 50.0 \
  --fault-type bias --fault-magnitude 10.0

# Geophysical navigation
./target/release/geonav-sim -i data/input/input.csv -o output.csv \
  --geo-type gravity --geo-resolution one-minute
```

### Data Processing (Python scripts)
```bash
# Preprocess raw data
pixi run preprocess

# Download geophysical maps
pixi run getmaps

# Create simulation datasets
pixi run create_dataset
```

## Free Core Features (Definition of Done)

The Free Core implementation must achieve the following capabilities:

1. **Reproducible Simulations**:
   - Researchers can configure scenarios (input data, filter config, GNSS settings) and get identical results with same random seed
   - Particle filter results are statistically consistent across runs
   - Deterministic behavior for scientific reproducibility

2. **Dataset Support**:
   - Import third-party datasets (KITTI, nuScenes, Carla, MEMS-Nav, etc.) via converter tools
   - Pre-processed hosted datasets that work out-of-the-box
   - CLI/interactive tools for format conversion

3. **GNSS Degradation Modeling** (Basic):
   - Complete outages (dropouts with configurable start/duration)
   - Increased noise levels
   - Reduced satellite availability
   - Measurement corruption and bias injection

4. **Configuration-Driven**:
   - YAML/JSON scenario files describing: trajectory/data path, sensor parameters, noise models, degradation events, filter configuration, output format
   - Command-line interface with config file support and argument overrides

5. **Output Formats**:
   - CSV and Parquet export for analysis in Python/MATLAB/R
   - Navigation solution time series with position, velocity, attitude estimates

6. **Python Integration**:
   - Python bindings for core functionality (future work)
   - Enables integration with ML/AI workflows and analysis pipelines

## Development Notes

### Code Style and Conventions
- **Naming conventions**:
  - Variables and functions: `snake_case`
  - Types (structs, enums, traits): `CamelCase`
  - Constants: `SCREAMING_SNAKE_CASE`
  - Names should be descriptive (avoid abbreviations and mathematical symbols)
- **Function design**:
  - Keep functions focused on a single task
  - Break up long functions (>25 statements) into smaller private helper functions
  - Each function should have a clear purpose reflected in its name
- **Documentation**:
  - Use Rust doc comments extensively (`///` and `//!`)
  - Include examples in documentation
  - Reference Groves textbook equations by section/equation number where applicable
- **Testing**:
  - Write unit tests for each module
  - Include integration tests for full system behavior
  - Test edge cases and error handling
  - Use `assert_approx_eq` for floating-point comparisons

### Reference Material
- Primary reference: "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition" by Paul D. Groves
- Equations reference Groves by section/equation number
- Variables named by quantity (not mathematical symbols) for clarity

### Coordinate Conventions
- **Navigation frame**: NED (North-East-Down) local-level frame
  - Default convention is East-North-Up (ENU) but NED is also supported
  - Users control via `is_enu` boolean flags and sign conventions
  - Vertical velocity: positive up in ENU, positive down in NED
  - Valid altitude range: [-11,000m, 30,000m] for ENU; [11,000m, -30,000m] for NED
- **Attitude representation**: Direction cosine matrices (DCM), Euler angles (XYZ rotation)
- **Position**: WGS84 geodetic (lat/lon in degrees, altitude in meters)
- **Velocities**: Local-level frame (m/s)
- **IMU data**: Specific force (m/s²) and angular rate (rad/s) in body frame, NOT preprocessed
  - Raw IMU output includes gravitational acceleration
  - Strapdown equations handle gravity removal during propagation

### Filter Architecture
- **Common interface**: All filters implement the `NavigationFilter` trait defined in `kalman.rs`
  - Required methods: `predict()` and `update()`
  - Unified state representation via `StrapdownState`
- **Forward propagation**: Uses `StrapdownState::propagate()` with strapdown equations (Chapter 5.4-5.5)
- **Measurement models**: Implement the `MeasurementModel` trait for update step
  - Trait provides `predict_measurement()` and `innovation_covariance()` methods
  - Implemented models: GPS position, GPS velocity, barometric altitude, pseudorange, carrier phase
- **UKF implementation**:
  - Uses unscented transform with sigma points for nonlinear state estimation
  - Handles full 9-state navigation solution
- **Particle filter implementation** (`particle.rs`):
  - Extended state: 15+ states (9 nav states + 3 accel bias + 3 gyro bias + optional)
  - Resampling strategies: systematic, stratified, residual
  - Averaging strategies: mean, weighted mean, maximum weight
  - Includes vertical channel damping with altitude error feedback
  - Each particle propagates independently through strapdown equations
- **Process noise**: Default values defined in `sim::DEFAULT_PROCESS_NOISE`
- **Design for extensibility**: Core algorithms separate from simulation framework to enable future professional features

### Data Format
Input CSV must contain timestamped sensor measurements:
- Timestamps: ISO UTC format with timezone
- IMU: acc_x/y/z (m/s²), gyro_x/y/z (rad/s), mag_x/y/z (µT)
- GNSS: latitude, longitude, altitude, speed, bearing
- Orientation: roll, pitch, yaw (degrees) or quaternions

**Current dataset format**: Sensor Logger app format

**Future dataset support** (via converters): KITTI, nuScenes, Carla, MEMS-Nav

### Testing
- Extensive unit tests across core modules (use `cargo test --workspace` to run all)
- Tests use `assert_approx_eq` macro for floating-point comparisons
- Integration tests in `core/tests/integration_tests.rs` verify full simulation pipeline
- Deterministic tests verify reproducibility with seeded RNG
- Coverage reports generated with `cargo tarpaulin`

### Environment Setup
Project uses Pixi for dependency management (Python + Rust):
- Environment variables set in `pixi.toml` activation section
- HDF5 required for NetCDF support (geonav experimental features)
- Python ≥3.12, Rust ≥1.91
- Release binaries automatically added to PATH via pixi activation

## Extensibility for Professional Versions

When developing Free Core features, consider extensibility for future professional products:

**Pro features (out of scope for this repo)**:
- Synthetic data generation (vehicle dynamics, waypoint following, realistic sensor models)
- Advanced GNSS modeling (satellites, pseudoranges, DOP, spoofing, jamming, multipath, terrain masking)
- Batch runner for parameter sweeps
- GUI (Bevy-based scenario editor, 3D visualization)
- Professional reports (PDF/HTML with plots and metrics)
- Advanced data adapters (ROS bag, RINEX, standard nav formats)

**Design principles for extensibility**:
- Trait-based abstractions for sensors, filters, and measurement models
- Configuration schema that can be extended without breaking changes
- Clean separation between algorithms (strapdown-core) and application logic (strapdown-sim)
- Plugin architecture for adding new sensors/measurement models in pro versions

## Workflow Notes
- Main branch for PRs: `main`
- Create feature branches for development
- Current focus: Implementing Free Core specification features for reproducible research simulations
