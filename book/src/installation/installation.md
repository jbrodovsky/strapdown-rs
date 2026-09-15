# Installation

This page provides detailed instructions for installing Strapdown-rs on your system.

## Prerequisites

Before installing Strapdown-rs, ensure you have the following:

- **Rust**: Version 1.91 or higher (install from [rustup.rs](https://rustup.rs))
- **A C/C++ compiler and cmake >= 3.26**, for the features that use a C library

## Build Dependencies

There are **no system libraries to install**. libhdf5, libnetcdf, zlib and freetype are
compiled from vendored sources that ship as ordinary cargo dependencies, so nothing is looked
for on your machine. You only need a toolchain to compile them with.

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y build-essential cmake
```

Check `cmake --version`: the bundled HDF5 needs **3.26 or newer**, which is more than Ubuntu
22.04 (3.22) or Debian 12 (3.25) ship. On those, install cmake from
[Kitware's APT repository](https://apt.kitware.com/), or with `pip install cmake`, or with
`snap install cmake --classic`.

### Fedora/RHEL

```bash
sudo dnf install -y gcc gcc-c++ cmake
```

### macOS

```bash
xcode-select --install
brew install cmake
```

### Windows

Install the MSVC toolchain from Visual Studio Build Tools, plus
[cmake](https://cmake.org/download/). vcpkg is not needed.

> **Note:** `strapdown-core` on its own needs none of this -- `cargo add strapdown-core` and a
> Rust toolchain are enough unless you turn on its `hdf5` or `netcdf` features.

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

# Optionally, with geophysical (gravity/magnetic) navigation. This is the variant that
# compiles libnetcdf from source, so it needs cmake.
cargo install --path sim --features geonav
```

> The repository used to ship a `pixi.toml` for environment management. It was removed once
> the C libraries began building from source, since there was nothing left for it to provide;
> `cargo build` is now the whole story.

## Verifying Installation

After installation, verify everything works:

```bash
# Check strapdown-sim version
strapdown-sim --version

# Run a simple test
cargo test -p strapdown-core
```

## Troubleshooting

### HDF5/NetCDF Build Issues

These libraries are compiled from source, so failures look like cmake or compiler errors
rather than linker errors.

1. **`CMake 3.26 or higher is required`** -- your cmake is too old. See the Build Dependencies
   section above.

2. **Confusing cmake failures inside the netCDF build** -- check whether `HDF5_DIR` is set:
   ```bash
   echo "${HDF5_DIR:-unset}"
   ```
   If it is, the HDF5 build script quietly switches to looking for a *system* library instead
   of building the vendored one, and the netCDF build is then handed the wrong headers. Unset
   it. A leftover conda or pixi shell is the usual source.

3. **`cargo install` is slow the first time** -- that is the one-off source build of libhdf5
   and libnetcdf, roughly a minute on a modern machine. It is cached afterwards.

### Rust Version Issues

Ensure you're using a recent Rust version:

```bash
rustc --version
```

Inside a clone of the repository, `rust-toolchain.toml` pins the version and rustup fetches it
for you, so this should already agree with what CI uses.

## Next Steps

- Continue to the [Quick Start](../quick-start.md) guide
- Learn about [System Requirements](./requirements.md)
- Explore [Building from Source](./building.md) in detail
