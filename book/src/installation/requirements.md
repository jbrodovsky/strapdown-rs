# System Requirements

This page lists what each part of the project needs to build and run, and how to get it on
each platform. [Installation](./installation.md) has the install commands themselves.

## Platforms

The code is platform-independent Rust plus vendored C libraries. Continuous integration builds
and tests `strapdown-core` on Linux, macOS and Windows (`ubuntu-latest`, `macos-latest`,
`windows-latest` GitHub runners), builds `strapdown-geonav` with its vendored netCDF and HDF5
on macOS and Windows, and builds and tests the whole workspace with every feature on Linux.

There are no special hardware requirements. Memory and run time scale with the length of the
trajectory and, for the particle filter, with `--num-particles`.

## Rust toolchain

- **Rust 1.91 or newer** (`rustc` and `cargo`). This is the workspace's declared
  `rust-version`.
- `clippy` and `rustfmt` only if you plan to contribute; see
  [Building and Testing](../development/building.md).

Install via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Inside a clone of the repository you do not choose a version: `rust-toolchain.toml` pins 1.91
and rustup fetches it on the first cargo command.

## C toolchain

There are **no system libraries to install**. libhdf5, libnetcdf, zlib and FreeType are
compiled from vendored sources that ship as ordinary cargo dependencies. What you may need is a
toolchain to compile them, depending on the features you build:

| Needs | When |
| --- | --- |
| Nothing beyond Rust | `strapdown-core` with default features |
| A C compiler (GCC, Clang or MSVC) | `strapdown-sim` with its default `plotting` feature (FreeType); `strapdown-core` with `mcap` (LZ4, Zstandard) |
| A C/C++ compiler **and cmake 3.26 or newer** | `strapdown-core` with `hdf5` or `netcdf`; `strapdown-sim` with `geonav`; `strapdown-geonav` always |

> **cmake 3.26 is newer than some distributions ship.** Ubuntu 22.04 LTS has 3.22 and
> Debian 12 has 3.25; both fail. Install a newer cmake from
> [Kitware's APT repository](https://apt.kitware.com/), with `pip install cmake`, or with
> `snap install cmake --classic`. Ubuntu 24.04 and Fedora 40+ are new enough as shipped.

### A variable to keep unset

**`HDF5_DIR`**: if this is set, the HDF5 build script looks for a system library instead of
building the vendored one, and the netCDF build then fails against those headers with an
error that does not name the cause. Unset it. A leftover conda or pixi shell is the usual
source.

### "Nothing is searched for" has one exception

Inside a clone of the repository the build is fully hermetic: `.cargo/config.toml` sets
`FREETYPE2_NO_PKG_CONFIG`, and the HDF5, netCDF and zlib builds are pinned to their vendored
sources by cargo features. `freetype-sys` on its own prefers a system FreeType when pkg-config
reports one, so `cargo install` from git or crates.io, which does not read the repository's
cargo configuration, may link the copy on your machine instead. Both work.

### Runtime

**libfontconfig** is needed at *run time* by `strapdown-sim`'s `plotting` feature, which loads
it on demand rather than linking it. A machine without it builds and runs normally, and fails
only when a plot is first rendered. It is present on virtually every desktop Linux install; on
a minimal container, `apt install libfontconfig1`.

## Platform-specific setup

### Linux

Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y build-essential cmake
cmake --version      # must be 3.26 or newer for hdf5/netcdf/geonav
```

Fedora or RHEL:

```bash
sudo dnf install -y gcc gcc-c++ cmake
```

### macOS

```bash
xcode-select --install     # the C/C++ compiler
brew install cmake
```

### Windows

Install the MSVC toolchain from Visual Studio Build Tools, plus
[cmake](https://cmake.org/download/). vcpkg is not needed: the C libraries are built from
source rather than located on the system. WSL2 with the Linux instructions above also works.

## Checking your environment

```bash
rustc --version      # 1.91 or newer
cc --version         # any C compiler, if you build plotting/mcap/hdf5/netcdf/geonav
cmake --version      # 3.26 or newer, if you build hdf5/netcdf/geonav
echo "HDF5_DIR=${HDF5_DIR:-(unset, good)}"
```

## Next steps

- [Installation](./installation.md): install the simulator or add the library.
- [Quick Start](../quick-start.md): run every simulation mode on a synthetic trajectory.
