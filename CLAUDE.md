# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strapdown-rs is a Rust implementation of strapdown inertial navigation system (INS) algorithms. The project provides a source library and simulation tools for processing IMU and GNSS data to estimate position, velocity, and orientation. It includes experimental work on geophysical navigation as an alternative PNT solution under GNSS-denied conditions.

## Architecture

This is a Cargo workspace with three main crates:

### 1. `strapdown-core` (/core)
The core library implementing strapdown INS algorithms:
- **earth.rs**: WGS84 Earth ellipsoid model and geodetic calculations
- **strapdown.rs**: 9-state strapdown mechanization in local-level frame (NED). Implements forward propagation equations from Groves textbook (Chapter 5.4-5.5)
- **filter.rs**: Navigation filters (Unscented Kalman Filter, Particle Filter) with measurement models (GPS position, velocity)
- **sim.rs**: Simulation utilities, CSV data loading (Sensor Logger app format), dead reckoning and closed-loop functions
- **messages.rs**: Event stream handling for GNSS scheduling and fault injection
- **linalg.rs**: Linear algebra utilities for matrix operations

**Key state representation**: 9-state vector [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw] in NED frame

### 2. `strapdown-sim` (/sim)
Binary for simulating INS performance with GNSS degradation scenarios:
- Modes: open-loop (dead reckoning) or closed-loop (loosely-coupled UKF)
- GNSS fault simulation: dropouts, reduced update rates, measurement corruption, bias injection
- Input: CSV files with IMU and GNSS measurements (Sensor Logger format)
- Output: Navigation solutions as CSV

### 3. `strapdown-geonav` (/geonav)
Experimental geophysical navigation using gravity/magnetic anomaly maps:
- Loads NetCDF geophysical maps
- Integrates geophysical measurements with INS/GNSS
- Provides alternative PNT in GNSS-denied environments

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
- Default process noise defined in `sim::DEFAULT_PROCESS_NOISE`

### Data Format
Input CSV must contain (Sensor Logger app format):
- Timestamps: ISO UTC format with timezone
- IMU: acc_x/y/z (m/s²), gyro_x/y/z (rad/s), mag_x/y/z (µT)
- GNSS: latitude, longitude, altitude, speed, bearing
- Orientation: roll, pitch, yaw (degrees) or quaternions

### Testing
- 84 unit tests across core modules
- Tests use `assert_approx_eq` for floating-point comparisons
- Integration tests verify full simulation pipeline

### Environment Setup
Project uses Pixi for dependency management (Python + Rust):
- Environment variables set in `pixi.toml` activation
- HDF5 required for NetCDF support (geonav)
- Python ≥3.13, Rust ≥1.89

## Current Development
- Active branch: jbrodovsky/issue91
- Main branch for PRs: main
- Research focus: improving INS accuracy under GNSS-denied conditions using geophysical aiding
