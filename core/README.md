# Strapdown - A simple strapdown INS implementation

Strapdown-rs is a straightforward strapdown inertial navigation system (INS) implementation in Rust. It is designed to be simple and easy to understand, making it a great starting point for those interested in learning about strapdown INS algorithms. It is currently under active development.

## Installation

Add the library to your project with `cargo add strapdown-core`, or install the
simulation CLI with `cargo install strapdown-sim`.

The library is imported as `strapdown`:

```rust
use strapdown::StrapdownState;
```

## Cargo Features

`strapdown-core` builds with no features by default and requires **no system
libraries** -- a Rust toolchain is enough. The binary data formats each pull in a
system library or a sizeable dependency tree, so they are opt-in:

| Feature   | Enables                                  | Requires        |
|-----------|------------------------------------------|-----------------|
| `hdf5`    | `to_hdf5` / `from_hdf5`                  | `libhdf5`       |
| `netcdf`  | `to_netcdf` / `from_netcdf`              | `libnetcdf`     |
| `mcap`    | `to_mcap` / `from_mcap`                  | --              |
| `clap`    | `clap::ValueEnum` derives on config enums | --             |
| `full`    | all of the above                         | both libraries  |

CSV support is always available and needs no feature flag.

```toml
[dependencies]
strapdown-core = { version = "1.0", features = ["hdf5", "netcdf"] }
```

## Summary

`strapdown-rs` is a Rust-based software library for implementing strapdown inertial navigation systems (INS). It provides core functionality for processing inertial measurement unit (IMU) data to estimate position, velocity, and orientation using a strapdown mechanization model that is typical of modern systems particularly in the low size, weight, and power (low SWaP) domain (cell phones, drones, robotics, UAVs, UUVs, etc.). Additionally, it provides some basic simulation capabilities for simulating INS scenarios (e.g. dead reckoning, closed-loop INS, intermitent GPS, GPS degradation, etc.).

`strapdown-rs` prioritizes correctness, numerical stability, and performance. It is built with extensibility in mind, allowing researchers and engineers to implement additional filtering, sensor fusion, or aiding algorithms on top of the base INS framework. This library is not intended to be a full-featured INS solution, notably it does not have code for processing raw IMU or GPS signals and only implements a loosely-couple INS.

The toolbox is designed for research, teaching, and development purposes and aims to serve the broader robotics, aerospace, and autonomous systems communities. The intent is to provide a high-performance, memory-safe, and cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems. The simulation is intended to be used for testing and verifying the correctness of the INS algorithms, by providing a simple simulation that allows users to generate a "ground truth" trajectory.

## Functionality

`strapdown-rs` is intended to be both a source code library included into your INS software and simulation environment as well as very light-weight INS simulator. The library provides a set of modules modeling the WGS84 Earth ellipsoid, a common 9-state strapdown forward mechanization, and a set of navigation filters for estimating position, velocity, and orientation from inertial measurement unit (IMU) data.

The simulation program provides a simple command line interface for running various configurations of the INS. In can run in open-loop (dead reckoning) mode or closed-loop (full state loosely couple UKF) mode. It can simulate various scenarios such as intermittent GPS, GPS degradation, and more. The simulation is designed to be easy to use and provides a simple API for generating datsets for further navigation processing or research.

Both `strapdown-sim` and `geonav-sim` include built-in logging capabilities using the Rust `log` crate. Library functions use log macros for diagnostic output that can be captured by any logging backend.

## Data Formats

The library supports CSV, HDF5, netCDF, and MCAP for input and output data. CSV
works out of the box; the others are behind the feature flags listed above.

### CSV Format

CSV is the default format for compatibility and ease of inspection. The library can read and write:
- **TestDataRecord**: Input sensor data (IMU, GNSS, magnetometer, barometer) from apps like Sensor Logger
- **NavigationResult**: Output navigation solutions (position, velocity, attitude, biases, and covariances)

```rust
use strapdown::sim::{TestDataRecord, NavigationResult};

// Read CSV files
let input_data = TestDataRecord::from_csv("sensor_data.csv")?;
let results = NavigationResult::from_csv("nav_results.csv")?;

// Write CSV files
TestDataRecord::to_csv(&input_data, "sensor_data_out.csv")?;
NavigationResult::to_csv(&results, "nav_results_out.csv")?;
```

### HDF5 Format

Requires the `hdf5` feature. HDF5 provides efficient storage for large datasets with better compression and faster I/O:

```rust
use strapdown::sim::{TestDataRecord, NavigationResult};

// Read HDF5 files
let input_data = TestDataRecord::from_hdf5("sensor_data.h5")?;
let results = NavigationResult::from_hdf5("nav_results.h5")?;

// Write HDF5 files
TestDataRecord::to_hdf5(&input_data, "sensor_data_out.h5")?;
NavigationResult::to_hdf5(&results, "nav_results_out.h5")?;
```

**HDF5 File Structure:**
- TestDataRecord data is stored in a `/test_data` group
- NavigationResult data is stored in a `/navigation_results` group
- Each field is stored as a separate dataset within the group
- Timestamps are stored as RFC3339-formatted strings

The HDF5 format is particularly useful for:
- Large datasets that benefit from compression
- Integration with scientific computing workflows (Python, MATLAB, Julia)
- Parallel I/O operations
- Hierarchical data organization