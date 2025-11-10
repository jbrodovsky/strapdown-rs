# Strapdown-rs Project Instructions

These are some general instructions and notes about the Strapdown-rs project. You are an expert scientific programmer familiar with Rust, Python, and navigation systems in addition to an experience researcher and academic. Please follow these guidelines when contributing to the project.

## Project Overview

Strapdown-rs is a Rust implementation of strapdown inertial navigation system (INS) algorithms. The project provides a source library and simulation tools for processing IMU and GNSS data to estimate position, velocity, and orientation. It includes experimental work on geophysical navigation as an alternative PNT solution under GNSS-denied conditions. The project is written in Rust and should follow idiomatic Rust conventions as well as common software design practices concerning modularity, readability, and maintainability. The code should be well-structured, with clear separation of concerns and appropriate use of modules and crates.

This project is intended for use by researchers and developers working in the field of navigation systems, particularly those interested in strapdown INS algorithms and alternative PNT solutions. This repo contains experiments and projects that are intended to be used in production of peer-reviewed research papers, as well as the LaTeX source for those papers.

## Style

The code should be organized into modules that reflect the functionality of the system. Each module should have a clear purpose and should be named accordingly. The main module should serve as an entry point for the application, while other modules should encapsulate specific functionalities such as data processing, filtering, and sensor fusion.

The code should be well-documented, with clear and concise comments explaining the purpose of each module, function, and data structure. The documentation should also include examples of how to use the various components of the system. Make extensive use of Rust's documentation features, including doc comments and examples.

The code should be written in a way that is easy to understand and follow. Variable and function names should be descriptive and meaningful, avoiding abbreviations or overly complex names. Functions should be kept short and focused on a single task to aid in debugging and testing. When writing a function that starts to get long (approximately more than 25 statements), consider breaking it up into smaller functions. These sub-functions should likely be private. Each function should have a clear purpose and should be named accordingly.

The code should be written in a way that is easy to test and debug. Unit tests should be included for each module, and integration tests should be provided for the overall system. The tests should cover a wide range of scenarios, including edge cases and error handling.

Variable names should use snake_case, while type names (structs, enums, traits) should use CamelCase. Constants should be in SCREAMING_SNAKE_CASE. Names should be descriptive and meaningful, avoiding abbreviations, overly complex names, or mathematical symbols.

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

Additional core capabilities should be implemented as needed either as a new create for larger features or as new modules within an existing crate for smaller features.

## Development Environment

The project uses `pixi` as a software stack environment manager and task runner. `pixi` should be used to manage any "system" dependencies (e.g., Python, Rust toolchain, `libnetcdf`) and to run common tasks such as building, testing, linting, and formatting. See the `pixi.toml` file for details. Cargo should be used for managing Rust dependencies and building the Rust code.

If you need to run commands in the terminal, please format them for `nushell` using the `nu` language.