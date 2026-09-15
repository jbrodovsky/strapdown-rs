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

- **Linux**: Ubuntu 24.04+, Debian 13+, Fedora 40+, or equivalent. Older releases work too,
  but only once a cmake newer than the distribution's own is installed -- see the C toolchain
  note below.
- **macOS**: 11.0 (Big Sur) or later
- **Windows**: Windows 10/11 with MSVC or MinGW

### Rust Toolchain

- **Minimum Rust version**: 1.91
- **Required components**: `rustc`, `cargo`, `clippy`, `rustfmt`

Contributors working in a clone of the repository do not need to choose a version: a
`rust-toolchain.toml` pins 1.91 and rustup fetches it automatically on the first cargo
command.

Install via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### C Toolchain

There are **no system libraries to install**. libhdf5, libnetcdf, zlib and freetype are all
compiled from vendored sources shipped as ordinary crates.io dependencies. What you do need is
a toolchain to compile them with:

- **A C/C++ compiler**: GCC, Clang, or MSVC.
- **cmake, version 3.26 or later**: required by the bundled HDF5.

Both are needed only when you build something that touches a C library -- `strapdown-geonav`
always, `strapdown-core`'s `hdf5`/`netcdf` features, and `strapdown-sim`'s `plotting` feature
(for freetype). `cargo build -p strapdown-core` on its own needs neither.

One caveat on "nothing is searched for": inside a clone of this repository that is exactly
true, because `.cargo/config.toml` sets `FREETYPE2_NO_PKG_CONFIG` and the HDF5/netCDF/zlib
builds are pinned to their vendored sources by cargo features. `freetype-sys` on its own would
prefer a system FreeType when pkg-config reports one, so `cargo install strapdown-sim` -- which
does not see this repository's cargo config -- may link the copy on your machine instead. Both
work; only the repository build is fully hermetic.

> **cmake 3.26 is newer than some distributions ship.** Ubuntu 22.04 LTS has 3.22 and
> Debian 12 has 3.25; both fail. Install a newer cmake from
> [Kitware's APT repository](https://apt.kitware.com/), with `pip install cmake`, or with
> `snap install cmake --classic`. Ubuntu 24.04 and Fedora 40+ are new enough as shipped.

#### Runtime

- **libfontconfig**: needed at *run time* by `strapdown-sim`'s `plotting` feature, which loads
  it on demand rather than linking it. A machine without it builds fine and fails only when a
  plot is first rendered. Present by default on virtually every desktop Linux install; on a
  minimal container, `apt install libfontconfig1`.

#### A variable to keep unset

- **`HDF5_DIR`**: if this is set, the HDF5 build script looks for a system library instead of
  building the vendored one, and the netCDF build then fails against those headers with an
  error that does not name the cause. Unset it (a leftover conda or pixi shell is the usual
  source).

## Platform-Specific Notes

### Linux

Install a compiler and cmake with your package manager -- for example `apt install build-essential cmake` -- then check `cmake --version` against the 3.26 note above.

**Performance Note**: For best performance, consider building with native CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### macOS

Apple Silicon (M1/M2/M3) is fully supported. The ARM64 architecture provides excellent performance for navigation algorithms.

Xcode Command Line Tools provide the compiler (`xcode-select --install`) and `brew install cmake` provides the rest.

### Windows

The MSVC toolchain from Visual Studio Build Tools, plus cmake, is all that is required. vcpkg is no longer needed: the C libraries are built from source rather than located on the system.

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

# Check the C toolchain
cc --version
cmake --version      # must be 3.26 or newer

# Make sure no stale HDF5 override is in the environment
echo "HDF5_DIR=${HDF5_DIR:-(unset, good)}"

# Test compilation. The first run compiles libhdf5 and libnetcdf from source, which takes
# roughly a minute; later runs reuse them.
cargo build --workspace
```

## Next Steps

- Proceed to [Installing from Crates.io](./crates-io.md)
- Or learn about [Building from Source](./building.md)
