# Building and Testing

## Building the Project

### Build Dependencies

There are no system libraries to install. libhdf5, libnetcdf, zlib and freetype are compiled
from vendored sources that ship as cargo dependencies, so the only requirement beyond Rust is
a toolchain to build them with:
- a C/C++ compiler
- **cmake 3.26 or newer** (more than Ubuntu 22.04 or Debian 12 ship -- see
  [System Requirements](../installation/requirements.md))

On Ubuntu/Debian systems:
```bash
sudo apt update
sudo apt install -y build-essential cmake
```

Neither is needed for `cargo build -p strapdown-core`, which uses no C library by default.

### Rust Toolchain
- Pinned to 1.91 by `rust-toolchain.toml`; rustup fetches it on the first cargo command
- Required components: `clippy`, `rustfmt` (also declared in `rust-toolchain.toml`)

### Building

Build the entire workspace:
```bash
cargo build --workspace --all-features
```

Build a specific crate:
```bash
cargo build -p strapdown-core
cargo build -p strapdown-sim
cargo build -p strapdown-geonav
```

Build with optimizations (release mode):
```bash
cargo build --workspace --all-features --release
```

### Installing Binaries

Install the simulation binaries to your system:
```bash
# Install strapdown-sim
cargo install --path sim

# ...with geophysical navigation. This is what pulls in libnetcdf, so it needs cmake.
cargo install --path sim --features geonav
```

## Testing

### Overview

The strapdown-rs project includes comprehensive unit tests, integration tests, and module-specific tests to validate the correctness of the navigation algorithms.

### Running Tests

Run all tests in the workspace:
```bash
cargo test --workspace --all-features --verbose
```

Run tests for a specific crate:
```bash
cargo test -p strapdown-core
cargo test -p strapdown-sim
cargo test -p strapdown-geonav
```

Run with output visible:
```bash
cargo test -- --nocapture
```

Run tests sequentially (single-threaded):
```bash
cargo test -- --test-threads=1
```

### Test Organization

Tests are organized into:
- **Unit tests**: Inline in source files using `#[cfg(test)]` modules
- **Integration tests**: Located in `core/tests/integration_tests.rs`
- **Module tests**: Specific to individual modules

### Integration Tests for INS Filters

The integration tests validate the entire INS filter pipeline using real sensor data. These tests are comprehensive and take longer to run than unit tests.

#### Test Data

The integration tests use real data collected from the Sensor Logger mobile application:
- IMU measurements (accelerometer and gyroscope) at ~100 Hz
- GNSS position and velocity measurements at ~1 Hz  
- Approximately 90 minutes of data with ~5366 samples
- Dataset location: `core/tests/test_data.csv`

#### Error Metrics

##### Horizontal Position Error
- **Metric**: Haversine distance between estimated and GNSS positions (meters)
- **Formula**: Great-circle distance on Earth's surface
- **Purpose**: Measures planar navigation accuracy

##### Altitude Error
- **Metric**: Simple absolute difference (meters)
- **Purpose**: Measures vertical navigation accuracy

##### Velocity Error
- **Metric**: Component-wise absolute differences for north, east, and down velocities (m/s)
- **Purpose**: Measures velocity estimation accuracy

#### Test Suite

`core/tests/integration_tests.rs` holds 23 tests, not the handful this page used to walk
through one by one. That walkthrough transcribed each test's thresholds into prose, and the
prose then drifted from the constants -- it was still quoting a 30 m horizontal limit long after
the code moved to 40 m. So this section names the groups and the constants and points at the
source, rather than copying numbers that nothing keeps honest.

**Per-filter closed-loop suites.** Each of the four filters gets the same three scenarios:

| | full-rate GNSS | degraded GNSS (5 s) | beats dead reckoning |
|---|---|---|---|
| UKF | `test_ukf_closed_loop_on_real_data` | `test_ukf_with_degraded_gnss` | `test_ukf_outperforms_dead_reckoning` |
| EKF | `test_ekf_closed_loop_on_real_data` | `test_ekf_with_degraded_gnss` | `test_ekf_outperforms_dead_reckoning` |
| ESKF | `test_eskf_closed_loop_on_real_data` | `test_eskf_with_degraded_gnss` | `test_eskf_outperforms_dead_reckoning` |
| RBPF | `test_rbpf_closed_loop_on_real_data` | `test_rbpf_with_degraded_gnss` | -- |

**Plus**: `test_dead_reckoning_on_real_data` (the unaided baseline),
`test_eskf_output_stays_valid_across_full_run`, `test_eskf_default_initialization_on_real_data`,
`test_eskf_auto_covariance_initialization_on_real_data`, `test_eskf_recovers_from_gnss_outage`,
`test_filter_output_length_matches_input`, `test_filters_are_deterministic_across_runs`,
`test_rmse_benchmark_across_filters`, `test_full_lifecycle_through_ins_engine`,
`magnetometer_yaw_aiding_source_error_matches_derivation`, and
`gating_through_the_closed_loop_no_longer_cascades`.

#### Thresholds

The limits live in `core/tests/integration_tests.rs` as named constants, each with its
derivation in a doc comment. Read those rather than any number on this page:

| constant | what bounds it |
|---|---|
| `MAX_HORIZONTAL_RMSE_M` | empirical, with headroom over the observed run |
| `MAX_VERTICAL_RMSE_M` | empirical |
| `MAX_LEVEL_ATTITUDE_RMSE_RAD` | **derived** -- gravity observability |
| `MAX_YAW_RMSE_RAD` | **derived** -- a multiple of `MAG_YAW_SOURCE_RMSE_RAD`, the magnetometer's own measured error |
| `DEAD_RECKONING_BASELINE_SAMPLES` | **derived** -- see below |
| `DEAD_RECKONING_BEAT_FACTOR` | **derived** -- how far a filter must beat the baseline for the comparison to be live |

The distinction matters. An empirical guard answers "is this worse than yesterday?"; a derived
one answers "is this physically possible?". **Do not tighten a derived bound to match a measured
number** -- if a filter beats it, the bound was wrong, and the derivation is what changes. See
`AGENTS.md` for the policy.

#### The dead-reckoning baseline is a window, not the whole run

The `*_outperforms_dead_reckoning` tests score the comparison over
`DEAD_RECKONING_BASELINE_SAMPLES` records, not the full 90-minute recording. This page
previously advertised the full-run version -- "dead reckoning RMS error ~7,100 km, improvement
>99.99%" -- and that headline is exactly what the truncation exists to kill.

Unaided dead reckoning over the whole log ends millions of metres out. A test asserting
`filter_error < dead_reckoning_error` against that is vacuous: it passes for any filter that
does not itself diverge, including a badly broken one. Truncating to a window where the baseline
drift is comparable to the filter's error makes the comparison discriminating again, and
`DEAD_RECKONING_BEAT_FACTOR` then requires a real margin rather than any margin. See #307 and
#299, and the derivation in the module header.

#### Running Integration Tests

```bash
cargo test -p strapdown-core --test integration_tests             # all of them
cargo test -p strapdown-core --test integration_tests -- --nocapture   # with printed output
cargo test -p strapdown-core --test integration_tests test_ukf_closed_loop_on_real_data -- --nocapture
cargo test -p strapdown-core --test integration_tests -- --test-threads=1   # sequential
```

#### Expected Runtime

Deliberately not itemised here. The previous figures covered four tests where there are now 22,
and predated the baseline truncation above, so they were wrong in both directions and nothing
noticed. Nothing in CI measures wall clock today -- that is the open half of #377 -- so treat
any runtime number as an observation you made, not a contract. What *is* bounded is CI itself:
every job sets `timeout-minutes` (#379).


#### Implementation Details

##### Error Metric Calculation

The `compute_error_metrics()` function:
1. Matches navigation results to GNSS measurements by timestamp
2. Skips invalid GNSS data (NaN values)
3. Computes error for each matched sample
4. Filters out non-finite errors
5. Calculates mean, max, and RMS statistics

##### Filter Initialization

Tests use realistic initialization based on first GNSS measurement:
- Position: First GNSS lat/lon/alt
- Velocity: Computed from GNSS speed and bearing
- Attitude: From phone orientation sensors
- Covariances: Conservative initial uncertainties
- Process noise: Tuned for MEMS IMU characteristics

##### Event Stream Generation

Tests use the `build_event_stream()` function to create a sequence of IMU propagation and GNSS update events from the raw data, with configurable scheduling and fault injection.

### Future Test Enhancements

Potential improvements for the test suite:

1. ~~**Additional test scenarios**~~ -- **done.** `core/tests/perf_baseline.rs` covers GNSS
   outages (`real_outage_60s`, `syn_outage_60s`, both `DutyCycle`), measurement corruption
   (`real_degraded`, an AR(1) fault model) and a synthetic cruise profile alongside the real
   recording.

2. ~~**More filters**~~ -- **done.** The EKF is implemented (`core/src/kalman.rs`) and has the
   same three-scenario integration suite as the others; the RBPF has
   `test_rbpf_closed_loop_on_real_data` and `test_rbpf_with_degraded_gnss`, and its own row in
   the performance baseline.

3. **Performance benchmarks**: *accuracy* regression is done -- `core/tests/perf_baseline.rs`
   gates every filter against a checked-in baseline in both directions, and
   [Performance Baselines](./performance.md) has the current numbers. What remains is the
   *wall-clock* half, which the accuracy gate deliberately does not measure:
   - Execution time metrics
   - Memory usage tracking
   - Scalability tests

4. **Shorter test datasets**:
   - Create focused test datasets for faster CI/CD
   - Keep full dataset for comprehensive validation

## Linting and Formatting

### Running Clippy

Clippy provides lint checks for Rust code:
```bash
cargo clippy --workspace --all-features
```

Address warnings and errors before committing.

### Running rustfmt

Format code with rustfmt:
```bash
cargo fmt --all
```

Check formatting without making changes:
```bash
cargo fmt --all -- --check
```

## Documentation

### Building API Documentation

Generate documentation for all crates:
```bash
cargo doc --workspace --all-features --no-deps
```

Open the documentation in a browser:
```bash
cargo doc --workspace --all-features --no-deps --open
```

### Building the Book

This user guide is built using mdBook. Install mdBook:
```bash
cargo install mdbook
```

Build the book:
```bash
cd book
mdbook build
```

Serve the book locally for development:
```bash
cd book
mdbook serve
```

Then open http://localhost:3000 in your browser.

## Continuous Integration

The project uses GitHub Actions for continuous integration. See `.github/workflows/` for workflow definitions:

- `rust.yml`: the blocking gate. Three jobs -- `minimal` (`strapdown-core`, no default
  features, on Linux/macOS/Windows), `full` (fmt, clippy `-D warnings`, and the whole suite
  with all features, Linux), and `vendored` (the macOS/Windows canary for the source builds of
  netCDF and HDF5)
- `deploy-book.yml`: builds and deploys this documentation to GitHub Pages
- `publish.yml`: verifies, then publishes the crates to crates.io. Takes a `dry_run` input
- `draft-pdf.yml`: renders the JOSS paper, on the `joss` branch
- `copilot-setup-steps.yml`: toolchain provisioning only

Every job sets `timeout-minutes` (#379), so a hang fails rather than running to GitHub's
360-minute default.

## References

- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed.
- Sensor Logger app: https://www.tszheichoi.com/sensorlogger
