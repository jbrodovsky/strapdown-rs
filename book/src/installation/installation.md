# Installation

There are two things to install, and most readers want only one of them:

- the **`strapdown-sim` binary**, to run simulations from the command line;
- the **`strapdown-core` library**, to call the mechanization and filters from your own Rust
  code. It is imported as `strapdown`.

## Version 1.0 is installed from git

At the time of writing, crates.io carries only pre-1.0 releases of these crates, whose API
differs from the one this book documents. Version 1.0.0 is published to crates.io together with
the JOSS paper. **Until then, install from the GitHub repository**, as below. Once 1.0.0 is on
crates.io, drop the `--git` argument from every command on this page.

## Before you start

You need a Rust toolchain, version **1.91 or newer**, from [rustup.rs](https://rustup.rs).
Whether you need anything else depends on which features you build:

| What you build | Needs |
| --- | --- |
| `strapdown-core` with default features | Rust only |
| `strapdown-core` with `mcap`; `strapdown-sim` with its default `plotting` feature | Rust and a C compiler |
| `strapdown-core` with `hdf5` or `netcdf`; `strapdown-sim` with `geonav`; `strapdown-geonav` | Rust, a C/C++ compiler, and **cmake 3.26 or newer** |

No system libraries are searched for. libhdf5, libnetcdf, zlib and FreeType are compiled from
vendored sources that ship as ordinary cargo dependencies, so the compiler and cmake are all
the build needs. Two details catch people out:

- **cmake 3.26 is newer than some distributions ship.** Ubuntu 22.04 has 3.22 and Debian 12 has
  3.25, and both fail. Install a newer cmake from
  [Kitware's APT repository](https://apt.kitware.com/), with `pip install cmake`, or with
  `snap install cmake --classic`. Ubuntu 24.04 and Fedora 40+ are new enough as shipped.
- **`HDF5_DIR` must be unset.** When it is set, the HDF5 build script looks for a system
  library instead of building the vendored one, even though a vendored build was requested,
  and the netCDF build then fails against those headers with an error that does not name the
  cause. A leftover conda or pixi shell is the usual source.

[System Requirements](./requirements.md) lists the per-platform commands for the compiler and
cmake.

## Install the simulator

```bash
cargo install --git https://github.com/jbrodovsky/strapdown-rs strapdown-sim
```

With experimental geophysical (gravity and magnetic anomaly) aiding, which adds the `--geo`
family of flags to `cl` and `pf`:

```bash
cargo install --git https://github.com/jbrodovsky/strapdown-rs strapdown-sim --features geonav
```

Check the result:

```console
$ strapdown-sim --version
strapdown-sim 1.0.0
```

`cargo install` puts the binary in `~/.cargo/bin`, which rustup adds to your `PATH`.

## Add the library to a project

```bash
cargo add strapdown-core --git https://github.com/jbrodovsky/strapdown-rs
```

That writes this entry to your `Cargo.toml`:

```toml
[dependencies]
strapdown-core = { git = "https://github.com/jbrodovsky/strapdown-rs", version = "1.0.0" }
```

Add features with `--features`, for example
`cargo add strapdown-core --git https://github.com/jbrodovsky/strapdown-rs --features mcap`.
In your code the crate is `strapdown`, not `strapdown_core`. [Using the
Library](../user-guide/library.md) continues from here.

## Cargo features

### `strapdown-core`

| Feature | Default | What it adds | Build needs |
| --- | --- | --- | --- |
| `clap` | no | `clap` `ValueEnum` and `Args` derives on the simulation configuration types in `strapdown::sim`, so a CLI can take them as arguments. `strapdown-sim` turns this on | Rust only |
| `hdf5` | no | `NavigationResult::to_hdf5` / `from_hdf5` and the matching `TestDataRecord` readers and writers | C/C++ compiler, cmake ≥ 3.26 |
| `netcdf` | no | `NavigationResult::to_netcdf` / `from_netcdf` and the `TestDataRecord` equivalents | C/C++ compiler, cmake ≥ 3.26 |
| `mcap` | no | `NavigationResult::to_mcap` / `from_mcap` and the `TestDataRecord` equivalents | C compiler (bundled LZ4 and Zstandard) |
| `full` | no | All four of the above | as for `hdf5` |

The default build has none of them and compiles with Rust alone. These writers are library
methods only. `strapdown-sim` writes CSV whatever features the library was built with: an
`-o` path that does not end in `.csv` is treated as a *directory*, and the results are written
into it as `.csv` files (`-o out.h5` produces `out.h5/<input name>.csv`).

### `strapdown-sim`

| Feature | Default | What it adds | Build needs |
| --- | --- | --- | --- |
| `plotting` | **yes** | Rendering for the global `--plot` flag, a performance plot against the GNSS track. Without the feature the flag is still accepted, but it logs an error and draws nothing | C compiler (bundled FreeType); libfontconfig at **run time** |
| `geonav` | no | Gravity and magnetic anomaly map aiding: `--geo`, `--gravity-*`, `--magnetic-*` and `--geo-interval-s` on `cl` and `pf`. Without the feature these flags do not exist | C/C++ compiler, cmake ≥ 3.26 |

`plotting` loads libfontconfig when it first draws text rather than linking it at build time.
A machine without it builds and runs normally, and fails only when a plot is rendered. It is
present on virtually every desktop Linux install; on a minimal container, `apt install
libfontconfig1`.

## Build from a clone

Contributors, and anyone who wants the examples and test fixtures, should clone the
repository:

```bash
git clone https://github.com/jbrodovsky/strapdown-rs.git
cd strapdown-rs
cargo build --workspace --release
```

The binary is then at `target/release/strapdown-sim`. Inside the clone, `rust-toolchain.toml`
pins the toolchain the CI uses (1.91) and rustup fetches it on the first cargo command, so you
do not choose a version yourself. `.cargo/config.toml` additionally forces the vendored
FreeType, so a repository build never links a system copy. (`cargo install` from git does not
read that file and may link a system FreeType if pkg-config reports one; both work.)

To put a binary built from your clone on your `PATH`:

```bash
cargo install --path sim
cargo install --path sim --features geonav   # with geophysical aiding
```

There is no environment manager to set up; `cargo build` is the whole setup. Running the test
suite, the lint gate and the performance baselines is covered in
[Building and Testing](../development/building.md).

## Troubleshooting

**`CMake 3.26 or higher is required`.** Your cmake is too old for the bundled HDF5. Install a
newer one as described under [Before you start](#before-you-start).

**Confusing cmake failures inside the netCDF build.** Check for a stale `HDF5_DIR`:

```bash
echo "${HDF5_DIR:-unset}"
```

If it prints a path, unset it and rebuild.

**The first build of `hdf5`, `netcdf` or `geonav` is slow.** That is the one-off source build
of libhdf5 and libnetcdf. Cargo caches it, and later builds reuse it.

**A plot fails to render while everything else works.** libfontconfig is missing at run time;
see the `plotting` row above.

**`cargo add strapdown-core` without `--git` resolved to a 0.x version.** That is the last
pre-1.0 release on crates.io. Its API is not the one in this book. Use the `--git` form until
1.0.0 is published.

## Next steps

- [Quick Start](../quick-start.md): generate a trajectory and run every simulation mode on it.
- [System Requirements](./requirements.md): platform-specific setup.
