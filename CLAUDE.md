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
- **earth.rs**: WGS84 Earth ellipsoid model and geodetic calculations
- **strapdown.rs**: 9-state strapdown mechanization in local-level frame (NED). Implements forward propagation equations from Groves textbook (Chapter 5.4-5.5)
- **filter.rs**: Navigation filters (Unscented Kalman Filter, Particle Filter) with measurement models (GPS position, velocity)
- **sim.rs**: Simulation utilities, CSV data loading (multiple dataset formats), dead reckoning and closed-loop functions
- **messages.rs**: Event stream handling for GNSS scheduling and fault injection
- **linalg.rs**: Linear algebra utilities for matrix operations

**Key state representation**: 9-state vector [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw] in NED frame

**Design principles for Free Core**:
- Deterministic simulations with reproducible results (seeded RNG)
- Configuration-driven (YAML/JSON configs for scenarios)
- Extensible architecture for future professional features
- Clean separation between core algorithms and simulation framework

### 2. `strapdown-sim` (/sim)
Command-line tool for running INS simulations with GNSS degradation:
- Modes: open-loop (dead reckoning) or closed-loop (loosely-coupled UKF/PF)
- GNSS fault simulation: dropouts, reduced update rates, measurement corruption, bias injection
- Input: CSV files with IMU and GNSS measurements (supports multiple dataset formats)
- Output: Navigation solutions as CSV/Parquet
- Configuration: YAML/JSON scenario files or command-line arguments

**Free Core scope**: Basic GNSS degradation (outages, noise, reduced availability)
**Future Pro scope**: Advanced faults (spoofing, jamming, multipath, terrain masking)

### 3. `strapdown-geonav` (/geonav)
**Experimental** geophysical navigation module (research-grade):
- Loads NetCDF geophysical maps (gravity/magnetic anomaly grids)
- Integrates geophysical measurements with INS/GNSS
- Provides alternative PNT in GNSS-denied environments
- Status: Experimental feature for research, may be commercialized in future roadmap (Phase 2)

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

### Reference Material
- Primary reference: "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition" by Paul D. Groves
- Equations reference Groves by section/equation number
- Variables named by quantity (not mathematical symbols) for clarity

### Coordinate Conventions
- **Navigation frame**: NED (North-East-Down) local-level frame
- **Attitude representation**: Direction cosine matrices (DCM), Euler angles (XYZ rotation)
- **Position**: WGS84 geodetic (lat/lon in degrees, altitude in meters)
- **Velocities**: Local-level frame (m/s)

### Filter Architecture
- Forward propagation uses `StrapdownState::propagate()` with strapdown equations
- Filters implement `MeasurementModel` trait for update step
- UKF uses sigma points for nonlinear state estimation
- Particle filters for non-Gaussian estimation
- Default process noise defined in `sim::DEFAULT_PROCESS_NOISE`
- **Design for extensibility**: Core algorithms separate from simulation framework to enable future professional features

### Data Format
Input CSV must contain timestamped sensor measurements:
- Timestamps: ISO UTC format with timezone
- IMU: acc_x/y/z (m/s²), gyro_x/y/z (rad/s), mag_x/y/z (µT)
- GNSS: latitude, longitude, altitude, speed, bearing
- Orientation: roll, pitch, yaw (degrees) or quaternions

**Supported dataset formats**:
- Sensor Logger app format (current)
- KITTI, nuScenes, Carla, MEMS-Nav (via converters - in progress)

### Testing
- 84+ unit tests across core modules
- Tests use `assert_approx_eq` for floating-point comparisons
- Integration tests verify full simulation pipeline
- Deterministic tests verify reproducibility with seeded RNG

### Environment Setup
Project uses Pixi for dependency management (Python + Rust):
- Environment variables set in `pixi.toml` activation
- HDF5 required for NetCDF support (geonav experimental features)
- Python ≥3.13, Rust ≥1.89

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

## Current Development
- Active branch: jbrodovsky/issue7
- Main branch for PRs: main
- Current focus: Implementing Free Core specification features for reproducible research simulations
