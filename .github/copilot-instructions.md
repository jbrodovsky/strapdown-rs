# Strapdown-rs Project Instructions

You are an expert scientific programmer familiar with Rust, Python, inertial navigation systems, Bayesian state estimation, and sensor fusion. Additionly you are an experienced researcher and scientist. Please follow these guidelines when contributing to the project.

## Project Overview

Strapdown-rs is a Rust implementation of strapdown inertial navigation system (INS) algorithms. The project provides a source library and simulation tools for processing IMU and GNSS data to estimate position, velocity, and orientation. It includes experimental work on geophysical navigation as an alternative PNT solution under GNSS-denied conditions. The project's source code is written in Rust and should follow idiomatic Rust conventions as well as common software design practices concerning modularity, readability, and maintainability. The code should be well-structured, with clear separation of concerns and appropriate use of modules and crates.

This project is intended for use by researchers and developers working in the field of navigation systems, particularly those interested in strapdown INS algorithms and alternative PNT solutions. This repo contains experiments and projects that are intended to be used in production of peer-reviewed research papers, as well as the LaTeX source for those papers.

## Style

The code should be organized into modules that reflect the functionality of the system. Each module should have a clear purpose and should be named accordingly. The main module (lib.rs) should serve as an entry point for the library, while other modules should encapsulate specific functionalities such as data processing, filtering, and sensor fusion.

The code should be well-documented, with clear and concise comments explaining the purpose of each module, function, and data structure. The documentation should also include examples of how to use the various components of the system. Make extensive use of Rust's documentation features, including doc comments and examples.

The code should be written in a way that is easy to understand and follow. Variable and function names should be descriptive and meaningful, avoiding abbreviations or overly complex names. Functions should be kept short and focused on a single task to aid in debugging and testing. When writing a function that starts to get long (approximately more than 25 statements), consider breaking it up into smaller functions. These sub-functions should likely be private. Each function should have a clear purpose and should be named accordingly.

The code should be written in a way that is easy to test and debug. Unit tests should be included for each module, and integration tests should be provided for the overall system. The tests should cover a wide range of scenarios, including edge cases and error handling.

Variable names should use snake_case, while type names (structs, enums, traits) should use CamelCase. Constants should be in SCREAMING_SNAKE_CASE. Names should be descriptive and meaningful, avoiding abbreviations, overly complex names, or mathematical symbols.

## Architecture

This is a Cargo workspace with three main crates:

### 1. `strapdown-core` (/core)
The core library implementing strapdown INS algorithms:
- **earth.rs**: WGS84 Earth ellipsoid model and geodetic calculations
- **kalman.rs**: Kalman-style Navigation filters
- **lib.rs**: library entry point and 9-state strapdown mechanization in local-level frame (NED). Implements forward propagation equations from Groves textbook (Chapter 5.4-5.5)
- **linalg.rs**: Linear algebra utilities for matrix operations
- **measurements.rs**: Measurement models (position, velocity, barometric altitude, pseudorange, carrier phase, etc.)
- **messages.rs**: Event stream handling for GNSS scheduling and fault injection
- **particles.rs**: Particle filter implementation for non-Gaussian state estimation
- **sim.rs**: Simulation utilities, CSV data loading (Sensor Logger app format), dead reckoning and closed-loop functions

**Key state representation**: 9-state vector [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw] in NED frame

### 2. `strapdown-sim` (/sim)
Binary for simulating INS performance with GNSS degradation scenarios:
- Modes: 
    - Open-loop (dead reckoning) 
    - Closed-loop (loosely-coupled UKF)
    - Particle filter
- GNSS fault simulation: dropouts, reduced update rates, measurement corruption, bias injection
- Input: CSV files with IMU and GNSS measurements (Sensor Logger format)
- Output: Navigation solutions as CSV

### 3. `strapdown-geonav` (/geonav)
Experimental geophysical navigation using gravity/magnetic anomaly maps:
- Loads NetCDF geophysical maps
- Integrates geophysical measurements with INS/GNSS
- Provides alternative PNT in GNSS-denied environments

Additional core capabilities should be implemented as needed either as a new create for larger features or as new modules within an existing crate for smaller features.

## Dependencies and Prerequisites

### System Dependencies
The project requires the following system libraries for building:
- `pkg-config`
- `libhdf5-dev` and `libhdf5-openmpi-dev` (for HDF5 support)
- `libnetcdf-dev` (for NetCDF geophysical data)
- `zlib1g-dev` (for compression support)

On Ubuntu/Debian systems, install with:
```bash
sudo apt update
sudo apt install -y pkg-config libhdf5-dev libhdf5-openmpi-dev libnetcdf-dev zlib1g-dev
```

### Rust Toolchain
- Minimum Rust version: 1.70+ (stable channel)
- Required components: `clippy`, `rustfmt`

## Build and Test Commands

### Building the Project
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

### Running Tests
Run all tests in the workspace:
```bash
cargo test --workspace --all-features --verbose
```

Run tests for a specific crate:
```bash
cargo test -p strapdown-core
```

### Linting and Formatting
Run clippy for linting:
```bash
cargo clippy --workspace --all-features
```

Format code with rustfmt:
```bash
cargo fmt --all
```

### Running the Simulation
The `strapdown-sim` binary can be run with various options:
```bash
# Build and install the simulation binary
cargo install --path sim

# Run with a configuration file
strapdown-sim --config examples/configs/example.toml

# Run with specific log level
strapdown-sim --config examples/configs/example.toml --log-level debug
```

## Testing Guidelines

### Test Structure
- **Unit tests**: Inline in source files using `#[cfg(test)]` modules
- **Integration tests**: Located in `core/tests/integration_tests.rs`
- Tests should cover edge cases, error handling, and numerical accuracy

### Test Naming
- Test function names should be descriptive: `test_<functionality>_<scenario>`
- Example: `test_wgs84_geodetic_to_ecef_conversion`

### Running Specific Tests
```bash
# Run tests matching a pattern
cargo test test_wgs84

# Run a specific test
cargo test test_specific_function_name
```

## Common Workflows

### Adding a New Module
1. Create the module file in the appropriate crate's `src/` directory
2. Add the module declaration in `lib.rs` or `main.rs`
3. Include comprehensive doc comments with examples
4. Add unit tests within the module
5. Update integration tests if needed

### Working with Simulation Data
- Input data format: CSV files from Sensor Logger app or similar
- CSV columns expected: timestamp, gyro (x,y,z), accel (x,y,z), GPS data, etc.
- See `core/src/sim.rs` for data loading functions

### Adding a New Kalman Filter
- Extend `core/src/kalman.rs` module
- Implement state transition and measurement models
- Follow the existing UKF pattern for consistency
- Add comprehensive tests for filter convergence and accuracy