# Frequently Asked Questions

## General

### What is strapdown-rs?

A Rust implementation of strapdown inertial navigation and of the tools to study it under GNSS
degradation. It is three crates: `strapdown-core` (the library, imported as `strapdown`), the
`strapdown-sim` command-line simulator, and the experimental `strapdown-geonav` for gravity
and magnetic map aiding. See the [Introduction](./introduction.md).

### Who is it for?

Researchers, students and engineers who want reproducible inertial-navigation experiments:
the same input, configuration and seed give the same output, and the GNSS outages, noise and
spoofing are configured rather than hand-edited into the data.

### What coordinate frames are supported?

North-East-Down (NED) is the default local-level frame, as in Groves. East-North-Up (ENU) is an
explicit opt-in: `--enu` on the command line or `is_enu = true` in a config file. The 9-state
navigation vector is:

- position: latitude, longitude, altitude (height above the ellipsoid, positive up in both
  frames);
- velocity: north, east and vertical, where vertical is positive **down** in NED and positive
  **up** in ENU (the output column is `velocity_vertical`);
- attitude: roll, pitch, yaw.

See [Coordinate Frames](./user-guide/coordinate-frames.md).

### Is this production-ready?

It is research software. It is tested and prioritises correctness, but it is developed as part
of ongoing PhD research, and `strapdown-geonav` in particular is experimental.

## Installation and setup

### What are the system requirements?

See [System Requirements](./installation/requirements.md). In summary:

- Rust 1.91 or later (`rust-toolchain.toml` fetches it);
- a C/C++ compiler and cmake 3.26 or newer, but only for features that build a C library
  (HDF5, netCDF, `geonav`); `cargo build -p strapdown-core` needs neither;
- no system libraries at build time;
- libfontconfig at *run* time, for `strapdown-sim`'s `--plot` (the `plotting` feature, on by
  default). It is loaded on demand, so a machine without it builds fine and fails at the
  first plot;
- Linux, macOS or Windows.

### Do I need to install HDF5 and NetCDF?

No. They are compiled from vendored sources that ship as ordinary cargo dependencies, along
with zlib and freetype, so nothing is searched for on your machine. What you need instead is a
C/C++ compiler and cmake 3.26 or newer to build them with. HDF5 and NetCDF back the optional
`hdf5`/`netcdf` output methods on `NavigationResult` and `TestDataRecord`, and NetCDF also
reads the geophysical maps.

### How do I install on Windows?

Install the MSVC toolchain from Visual Studio Build Tools plus
[cmake](https://cmake.org/download/). vcpkg is not needed, since nothing is looked up on the
system. WSL2 also works. See [Installation](./installation/installation.md).

## Usage

### What data format does strapdown-sim accept?

CSV files in the Sensor Logger app's format: timestamped IMU, GNSS and optional barometer and
magnetometer columns. `strapdown-sim syn` writes the same format from a synthetic trajectory.
See [Input Data Format](./user-guide/data-format.md).

### What does it write?

CSV. Every path through `strapdown-sim` ends in `NavigationResult::to_csv`. An `-o` value ending
in `.csv` is a file; one ending in another data format's extension (`.h5`, `.nc`, `.mcap`,
`.parquet` and so on) is refused; any other value is a directory to write `<input name>.csv`
into. HDF5, NetCDF and MCAP writers exist as library methods on `NavigationResult` behind cargo
features, not as CLI options. See [Output Format](./user-guide/output-format.md).

### My run stops with `InvalidConfiguration { field: "is_enu", ... }`

The data's frame does not match the one you declared. Sensor Logger exports are ENU, so pass
`--enu` (or set `is_enu = true` in the config file); `syn` output is NED, so leave it off. The
check compares the leading records' vertical specific force with what the declared frame
expects at rest, and the message says which way round it is.

### Which filter should I use?

- **ESKF** (`cl`, the default): the error-state Kalman filter. Start here.
- **EKF** (`cl --filter ekf`) and **UKF** (`cl --filter ukf`): the full-state alternatives,
  and the two Kalman filters that accept geophysical aiding.
- **Particle filter** (`pf`): the Rao-Blackwellized particle filter, the only particle filter
  in the crate. It is built for map-aided navigation, where the measurement is a nonlinear,
  possibly ambiguous function of position.

No speed comparison is published, because none has been measured. For accuracy, see
[Filter Comparison](./filters/comparison.md) and [Performance Baselines](./development/performance.md).

### Can I use my own sensor data?

Yes: convert it to the input CSV format above. Any IMU (accelerometer and gyroscope) with GNSS
position and velocity will do; barometer and magnetometer columns are optional.

### How do I simulate GNSS outages?

With a scheduler. On the command line, a duty cycle of 100 s available and 50 s denied:

```bash
strapdown-sim cl -i input.csv -o output.csv --sched duty --on-s 100 --off-s 50
```

or one fix a minute with `--sched fixed --interval-s 60`. In a config file the same goes in the
`[aiding]` section:

```toml
[aiding.scheduler]
kind = "duty_cycle"
on_s = 100.0
off_s = 50.0
start_phase_s = 0.0
```

Note that a duty cycle starts with its OFF window unless `start_phase_s` (`--duty-phase-s`)
gives an initial ON window. Noise, bias and spoofing are fault models (`--fault`). See
[Schedulers and Faults Reference](./gnss/scenarios.md) for every option.

## Performance

### How fast is it?

There are no published throughput figures, because none have been measured; nothing in CI
tracks wall-clock time yet. What is measured and gated is navigation *accuracy*: see
[Performance Baselines](./development/performance.md).

### Can I run simulations in parallel?

Across files, yes. Point `input` at a directory and set `parallel = true` in the config file
(or pass `--parallel` with `--config`), and the files are processed concurrently on a thread
pool. Each file's run is itself sequential, as navigation is.

In this release, `--parallel` takes effect only together with `--config`. With a subcommand
(`strapdown-sim --parallel cl -i dir/ -o out/`) the directory's files are processed one after
another.

### The particle filter is slow

Its cost grows with the number of particles, so reduce `--num-particles` (or `num_particles` in
`[particle_filter]`). If you do not need map aiding, a Kalman filter (`cl`) is the usual
choice.

## Development

### How can I contribute?

See [Contributing](./development/contributing.md), and contact the maintainer before starting
major work.

### Where is the API documentation?

The rustdoc is published with this book; [API Documentation](./api/index.md) links it and maps
the modules. It will also be on docs.rs once the crates are published with v1.0.0.

### Can I use this in a commercial project?

The code is MIT licensed, which allows commercial use. See the
[LICENSE](https://github.com/jbrodovsky/strapdown-rs/blob/main/LICENSE) file.

### How do I cite this work?

The JOSS paper is under review
([openjournals/joss-reviews#11377](https://github.com/openjournals/joss-reviews/issues/11377)).
Until it is published, cite the software with the repository's `CITATION.cff` (GitHub's "Cite
this repository" button). See [Publications and Links](./resources/publications.md).

[![JOSS](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4/status.svg)](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4)

## Troubleshooting

### I'm getting build errors from HDF5 or NetCDF

These are built from source, so failures are cmake or compiler errors rather than linker
errors. The two common ones:

1. **`CMake 3.26 or higher is required`**: your cmake is older than the bundled HDF5 needs.
   Ubuntu 22.04 (3.22) and Debian 12 (3.25) both hit this.
2. **A confusing cmake failure inside the netCDF build**: check `echo "${HDF5_DIR:-unset}"`.
   If it is set, the HDF5 build switches to looking for a *system* library even though a
   vendored build was requested, and netCDF is then given the wrong headers. Unset it.

See [Installation](./installation/installation.md) for more.

### My simulation produces NaN values or stops on a health check

Common causes are a wrong frame declaration (see above), unrealistic IMU values, or a filter
that has diverged. The run's health monitor stops a run whose state leaves configured bounds;
`--log-level debug` shows what happened before it did.

### Where can I get help?

- this FAQ and the [User Guide](./user-guide/overview.md);
- existing [GitHub Issues](https://github.com/jbrodovsky/strapdown-rs/issues);
- a new issue, if you have found a bug.
