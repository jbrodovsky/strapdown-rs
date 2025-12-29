# Installation

This page provides detailed instructions for installing Strapdown-rs on your system.

## Prerequisites

Before installing Strapdown-rs, ensure you have the following:

- **Rust**: Version 1.70 or higher (install from [rustup.rs](https://rustup.rs))
- **System Libraries**: Required for building certain dependencies

## System Dependencies

Strapdown-rs requires several system libraries for HDF5 and NetCDF support:

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y pkg-config \
  libhdf5-dev \
  libhdf5-openmpi-dev \
  libnetcdf-dev \
  zlib1g-dev
```

### Fedora/RHEL

```bash
sudo dnf install -y pkg-config \
  hdf5-devel \
  hdf5-openmpi-devel \
  netcdf-devel \
  zlib-devel
```

### macOS

```bash
brew install pkg-config hdf5 netcdf
```

### Windows

For Windows users, we recommend using [vcpkg](https://vcpkg.io/) to install dependencies:

```powershell
vcpkg install hdf5 netcdf zlib
```

## Installation Methods

### Method 1: Install from Crates.io (Recommended)

The easiest way to use Strapdown-rs is to add it as a dependency in your project:

```bash
cargo add strapdown-core
```

Or add manually to your `Cargo.toml`:

```toml
[dependencies]
strapdown-core = "0.1"
```

To install the simulation binary:

```bash
cargo install strapdown-sim
```

### Method 2: Build from Source

Clone the repository and build locally:

```bash
# Clone the repository
git clone https://github.com/jbrodovsky/strapdown-rs.git
cd strapdown-rs

# Build the entire workspace
cargo build --workspace --all-features --release

# Install the simulation binary
cargo install --path sim

# Optionally, install the geonav binary
cargo install --path geonav
```

### Method 3: Using Pixi (Experimental)

The project includes a `pixi.toml` for environment management:

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Activate the environment
pixi install
pixi shell

# Build and run
cargo build --release
```

## Verifying Installation

After installation, verify everything works:

```bash
# Check strapdown-sim version
strapdown-sim --version

# Run a simple test
cargo test -p strapdown-core
```

## Troubleshooting

### HDF5/NetCDF Linking Issues

If you encounter linking errors:

1. Ensure `pkg-config` can find the libraries:
   ```bash
   pkg-config --modversion hdf5
   pkg-config --modversion netcdf
   ```

2. Set environment variables if needed:
   ```bash
   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
   ```

### Rust Version Issues

Ensure you're using a recent Rust version:

```bash
rustc --version
rustup update stable
```

## Next Steps

- Continue to the [Quick Start](../quick-start.md) guide
- Learn about [System Requirements](./requirements.md)
- Explore [Building from Source](./building.md) in detail
