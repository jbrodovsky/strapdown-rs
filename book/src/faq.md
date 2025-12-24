# Frequently Asked Questions

## General Questions

### What is Strapdown-rs?

Strapdown-rs is a Rust library for implementing strapdown inertial navigation systems (INS). It provides core functionality for processing IMU data to estimate position, velocity, and orientation.

### Who should use Strapdown-rs?

Strapdown-rs is designed for:
- Researchers working on navigation systems
- Engineers developing autonomous systems
- Students learning about inertial navigation
- Anyone needing a high-performance INS implementation in Rust

### What coordinate frames are supported?

The library primarily uses the North-East-Down (NED) local-level frame. The 9-state vector includes:
- Position: latitude, longitude, altitude
- Velocity: northward, eastward, downward
- Attitude: roll, pitch, yaw

### Is this production-ready?

Strapdown-rs is primarily intended for research and development. While the code is well-tested and prioritizes correctness, it is still under active development as part of ongoing PhD research.

## Installation and Setup

### What are the system requirements?

See the [System Requirements](./installation/requirements.md) page for detailed information. In summary:
- Rust 1.70 or later
- System libraries: HDF5, NetCDF, zlib
- Supported on Linux, macOS, and Windows

### Why do I need HDF5 and NetCDF?

These libraries are required for:
- HDF5: Data storage and processing
- NetCDF: Geophysical map data for geonav features

If you only need the core INS functionality, you can disable these features in your `Cargo.toml`.

### How do I install on Windows?

Windows support requires additional setup. We recommend either:
1. Using [vcpkg](https://vcpkg.io/) to install dependencies
2. Using WSL2 for a native Linux environment

See [Installation](./installation/installation.md) for details.

## Usage Questions

### What data format does strapdown-sim accept?

The simulation expects CSV files with IMU and GNSS data following the Sensor Logger app format. See [Input Data Format](./user-guide/data-format.md) for details.

### Which filter should I use: EKF, UKF, or Particle Filter?

- **EKF**: Fastest, works well for mildly nonlinear systems
- **UKF**: Better accuracy for highly nonlinear systems, 2-3x slower than EKF
- **Particle Filter**: Best for non-Gaussian distributions and multimodal scenarios

See [Filter Comparison](./filters/comparison.md) for detailed analysis.

### Can I use my own sensor data?

Yes! You'll need to convert your data to the expected CSV format. The library is designed to work with standard IMU (gyroscope and accelerometer) and GNSS (position) measurements.

### How do I simulate GNSS outages?

Use the GNSS fault simulation features in the configuration file:

```toml
[gnss]
dropout_probability = 0.1  # 10% chance of dropout
reduced_update_rate = 0.5  # Half the normal rate
```

See [GNSS Degradation Scenarios](./gnss/fault-simulation.md) for more options.

## Performance Questions

### How fast is Strapdown-rs?

Performance varies by filter type and configuration:
- **EKF**: ~10,000-20,000 updates/second
- **UKF**: ~3,000-5,000 updates/second
- **Particle Filter**: Depends on particle count (100 particles: ~500 updates/second)

These are approximate figures on modern hardware and will vary based on your system.

### Can I run simulations in parallel?

The current implementation processes data sequentially as navigation is inherently a sequential process. However, the particle filter implementation can utilize multiple cores for particle processing.

### How much memory does it use?

Memory usage is modest:
- Core library: ~10-50 MB
- Simulations: Depends on data size and filter configuration
- Particle filters: Linear with particle count

## Development Questions

### How can I contribute?

We welcome contributions! Please see the [Contributing Guide](./development/contributing.md) and reach out to the project maintainer before starting major work.

### Where is the API documentation?

Full API documentation is available at [docs.rs/strapdown-core](https://docs.rs/strapdown-core). This book focuses on high-level concepts and usage patterns.

### Can I use this in my commercial project?

Yes! Strapdown-rs is licensed under the MIT License, which allows commercial use. See the [LICENSE](https://github.com/jbrodovsky/strapdown-rs/blob/main/LICENSE) file for details.

### How do I cite this work?

If you use Strapdown-rs in your research, please cite the JOSS paper:

[![JOSS](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4/status.svg)](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4)

## Troubleshooting

### I'm getting linking errors during build

This usually means HDF5 or NetCDF libraries aren't found. Ensure:
1. Libraries are installed: `pkg-config --modversion hdf5`
2. `PKG_CONFIG_PATH` is set correctly
3. Development headers are installed (`-dev` packages on Linux)

See [Installation Troubleshooting](./installation/installation.md#troubleshooting) for more help.

### My simulation produces NaN values

Common causes:
- Invalid initial conditions
- IMU data with unrealistic values
- Numerical instability in filter

Check your input data and initial state. Enable debug logging with `--log-level debug` to investigate.

### The particle filter is very slow

This is expected with high particle counts. Consider:
- Reducing the number of particles
- Using the RBPF (Rao-Blackwellized) variant
- Using EKF/UKF instead if appropriate

### Where can I get help?

- Check this FAQ and the user guide
- Search existing [GitHub Issues](https://github.com/jbrodovsky/strapdown-rs/issues)
- Open a new issue if you've found a bug
- Contact the maintainer for research collaborations

## Additional Questions?

If your question isn't answered here, please:
1. Check the [User Guide](./user-guide/overview.md)
2. Review the [API Reference](./api/core.md)
3. Open an issue on [GitHub](https://github.com/jbrodovsky/strapdown-rs/issues)
