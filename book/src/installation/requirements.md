# System Requirements

This page outlines the system requirements for running Strapdown-rs.

## Hardware Requirements

### Minimum

- **CPU**: Any modern 64-bit processor (x86_64 or ARM64)
- **RAM**: 2 GB
- **Storage**: 100 MB for binaries, plus space for data files

### Recommended

- **CPU**: Multi-core processor (4+ cores) for parallel processing
- **RAM**: 8 GB or more for large simulations
- **Storage**: SSD with sufficient space for trajectory data and results

## Software Requirements

### Operating Systems

Strapdown-rs supports all major operating systems:

- **Linux**: Ubuntu 20.04+, Debian 11+, Fedora 35+, or equivalent
- **macOS**: 11.0 (Big Sur) or later
- **Windows**: Windows 10/11 with MSVC or MinGW

### Rust Toolchain

- **Minimum Rust version**: 1.70.0
- **Recommended**: Latest stable release
- **Required components**: `rustc`, `cargo`, `clippy`, `rustfmt`

Install via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### System Libraries

#### Required Libraries

- **pkg-config**: For library detection
- **HDF5**: Version 1.10 or later
- **NetCDF**: Version 4.7 or later (for geophysical navigation)
- **zlib**: Standard compression library

#### Optional Libraries

- **OpenMPI**: For parallel HDF5 support (optional but recommended)

## Platform-Specific Notes

### Linux

Most distributions include the required libraries in their package repositories. Use your package manager to install dependencies.

**Performance Note**: For best performance, consider building with native CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### macOS

Apple Silicon (M1/M2/M3) is fully supported. The ARM64 architecture provides excellent performance for navigation algorithms.

**Note**: You may need to set additional environment variables for Homebrew-installed libraries:

```bash
export CPATH=/opt/homebrew/include
export LIBRARY_PATH=/opt/homebrew/lib
```

### Windows

Windows support is available but requires additional setup for HDF5 and NetCDF. Using [vcpkg](https://vcpkg.io/) is recommended for managing C/C++ dependencies.

Alternatively, use WSL2 (Windows Subsystem for Linux) for a native Linux environment.

## Development Requirements

If you plan to contribute to Strapdown-rs development:

- **Git**: Version control
- **clippy**: Rust linter (`rustup component add clippy`)
- **rustfmt**: Code formatter (`rustup component add rustfmt`)
- **cargo-edit**: For managing dependencies (`cargo install cargo-edit`)

## Testing Your Environment

After installing dependencies, verify your environment:

```bash
# Check Rust version
rustc --version

# Verify pkg-config
pkg-config --version

# Check for HDF5
pkg-config --modversion hdf5

# Check for NetCDF
pkg-config --modversion netcdf

# Test compilation
cargo build --workspace
```

## Next Steps

- Proceed to [Installing from Crates.io](./crates-io.md)
- Or learn about [Building from Source](./building.md)
