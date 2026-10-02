# Building and Testing

## Building the project

### Build dependencies

There are no system libraries to install. libhdf5, libnetcdf, zlib and freetype are compiled
from vendored sources that ship as cargo dependencies, so the only requirement beyond Rust is
a toolchain to build them with:

- a C/C++ compiler
- **cmake 3.26 or newer** (newer than Ubuntu 22.04 or Debian 12 ship; see
  [System Requirements](../installation/requirements.md))

On Ubuntu/Debian systems with a recent enough release:

```bash
sudo apt update
sudo apt install -y build-essential cmake
```

Neither is needed for `cargo build -p strapdown-core`, which uses no C library by default.
Keep `HDF5_DIR` unset: when it is set, the HDF5 build looks for a system library even though a
vendored one was requested, and the netCDF build then fails against the wrong headers.

### Rust toolchain

`rust-toolchain.toml` pins Rust 1.91 with the `clippy`, `rustfmt`, `rust-src` and
`rust-analyzer` components; rustup fetches it on the first cargo command. The pin matters for
linting, because `clippy::pedantic` and `clippy::nursery` gain lints with every release.

### Building

```bash
cargo build --workspace --all-features            # everything
cargo build -p strapdown-core                     # the library alone, no C toolchain needed
cargo build --release -p strapdown-sim            # the CLI
cargo build --release -p strapdown-sim --features geonav   # the CLI with --geo
```

### Installing the binary

```bash
cargo install --path sim                     # from a checkout
cargo install --path sim --features geonav   # with geophysical navigation
```

## Testing

```bash
cargo test --workspace --all-features   # the whole suite, as CI's `full` job runs it
cargo test-min                          # strapdown-core with no default features, as CI's `minimal` job runs it
cargo test -p strapdown-core            # one crate
```

Both of the first two are gates in CI, and they test different code: an item used only behind
a feature is live in one configuration and dead in the other.

### Where the tests are

Unit tests sit beside the code in `#[cfg(test)]` modules in every crate. The integration tests
are:

| file | what it checks |
|---|---|
| `core/tests/integration_tests.rs` | every filter end to end on the real recording `test_data.csv` (see below) |
| `core/tests/filter_comparison.rs` | the ESKF, EKF, UKF and RBPF driven through the same `NavigationFilter` interface on one shared scenario |
| `core/tests/perf_baseline.rs` | the navigation-accuracy regression gate against `perf_baseline.json`; see [Performance Baselines](./performance.md) |
| `core/tests/aiding.rs` | innovation gating and ZUPT/ZARU aiding over whole runs |
| `core/tests/baro_bias.rs` | that the barometric bias state is actually estimated, in all three Kalman filters |
| `core/tests/engine_lever_arm.rs` | `InsEngine` and antenna lever-arm compensation, on synthetic and real data |
| `core/tests/jacobian_agreement.rs` | that the analytic transition Jacobians agree with the mechanization they linearize |
| `core/tests/ukf_conditioning.rs` | that a one-ulp change to the UKF's input cannot move its answer by a metre |
| `core/tests/rbpf_depletion.rs` | that the RBPF's particle cloud does not collapse to a point when its weights degenerate |
| `core/tests/example_configs.rs` | that every config in `examples/configs/` and `conf/` deserializes into the scenario it describes, and that paired `conf/` recipes agree |
| `core/tests/config_serde_defaults.rs` | that each configuration type's `Default` and its serde defaults agree |
| `geonav/tests/geo_closed_loop.rs` | geophysically aided closed-loop runs |
| `geonav/tests/builder_equivalence.rs` | that `geonav::build_event_stream` produces the same non-geophysical events as core's |

`core/tests/` also holds the two data files, `test_data.csv` and `perf_baseline.json`. The
Python package has its own tests, run with `uv run pytest -q` (see
[Python Analysis Tooling](./analysis.md)).

### The real-data integration tests

#### Test data

`core/tests/test_data.csv` is a Sensor Logger recording from a consumer phone:
**5,366 records at 1 Hz**, spanning about 89 minutes, with IMU, GNSS, barometer, magnetometer
and orientation columns in the ENU convention. IMU and GNSS share the 1 Hz rows.

#### Error metrics

`compute_error_metrics` in the test file matches each navigation result to the GNSS fix at
the same timestamp, skips rows without a valid fix, and reports mean, maximum and RMS of:

- horizontal error: the haversine distance between estimate and fix, in metres;
- altitude error: the absolute difference, in metres;
- velocity error: per component, in m/s.

Note that the reference here is the GNSS fix, which is also what the filters are aided by; see
the first caveat in [Performance Baselines](./performance.md) for what that does and does not
measure.

#### The suite

`core/tests/integration_tests.rs` holds 23 tests at the time of writing;
`cargo test -p strapdown-core --test integration_tests -- --list` prints the current set. This
section names the groups and points at the source rather than transcribing thresholds, which
is how an earlier version of this page came to quote a limit the code had long since changed.

**Per-filter closed-loop tests:**

| | full-rate GNSS | reduced-rate GNSS | beats dead reckoning |
|---|---|---|---|
| UKF | `test_ukf_closed_loop_on_real_data` | `test_ukf_with_degraded_gnss` (5 s) | `test_ukf_outperforms_dead_reckoning` |
| EKF | `test_ekf_closed_loop_on_real_data` | `test_ekf_with_degraded_gnss` (5 s) | `test_ekf_outperforms_dead_reckoning` |
| ESKF | `test_eskf_closed_loop_on_real_data` | `test_eskf_with_degraded_gnss` (2 s) | `test_eskf_outperforms_dead_reckoning` |
| RBPF | `test_rbpf_closed_loop_on_real_data` | `test_rbpf_with_degraded_gnss` (5 s) | -- |

**Others:** `test_dead_reckoning_on_real_data` (the unaided baseline),
`test_filter_comparison` (UKF, EKF and ESKF on the same data),
`test_eskf_output_stays_valid_across_full_run`, `test_eskf_default_initialization_on_real_data`,
`test_eskf_auto_covariance_initialization_on_real_data`, `test_eskf_recovers_from_gnss_outage`,
`test_filter_output_length_matches_input`, `test_filters_are_deterministic_across_runs`,
`test_rmse_benchmark_across_filters`, `test_full_lifecycle_through_ins_engine`,
`magnetometer_yaw_aiding_source_error_matches_derivation` and
`gating_through_the_closed_loop_no_longer_cascades`. The last is marked `#[ignore]`, with the
reason in its attribute and doc comment, so a plain run reports 22 passed and 1 ignored; add
`-- --ignored` to run it.

#### Thresholds

The limits are named constants at the top of `core/tests/integration_tests.rs`, each with its
derivation in a doc comment. Read those rather than any number on this page:

| constant | what bounds it |
|---|---|
| `MAX_HORIZONTAL_RMSE_M` | empirical, with headroom over the observed run |
| `MAX_VERTICAL_RMSE_M` | empirical |
| `MAX_LEVEL_ATTITUDE_RMSE_RAD` | **derived**: gravity observability |
| `MAX_YAW_RMSE_RAD` | **derived**: a multiple of `MAG_YAW_SOURCE_RMSE_RAD`, the magnetometer's own measured error |
| `DEAD_RECKONING_BASELINE_SAMPLES` | **derived**: see below |
| `DEAD_RECKONING_BEAT_FACTOR` | **derived**: how far a filter must beat the baseline for the comparison to mean anything |

The distinction matters. An empirical guard answers "is this worse than yesterday?"; a derived
one answers "is this physically possible?". **Do not tighten a derived bound to match a
measured number**: if a filter beats it, the derivation is what changes. `AGENTS.md` has the
policy.

#### The dead-reckoning baseline is a window, not the whole run

The `*_outperforms_dead_reckoning` tests compare over the first
`DEAD_RECKONING_BASELINE_SAMPLES` records, not the whole recording. Unaided dead reckoning over
the full 89 minutes ends millions of metres out, and a test asserting
`filter_error < dead_reckoning_error` against that passes for any filter that does not itself
diverge, including a badly broken one. Truncating to a window where the baseline drift is
comparable to the filter's error makes the comparison discriminating, and
`DEAD_RECKONING_BEAT_FACTOR` then requires a real margin. See #307 and #299.

#### Running them

```bash
cargo test -p strapdown-core --test integration_tests                   # all of them
cargo test -p strapdown-core --test integration_tests -- --nocapture    # with printed output
cargo test -p strapdown-core --test integration_tests test_ukf_closed_loop_on_real_data -- --nocapture
```

No runtime is given here because none is measured: nothing in CI tracks wall-clock time (the
open half of #377). What is bounded is CI itself: every job sets `timeout-minutes`.

## Linting and formatting

The lints are enforced at `deny`, and CI lints two feature configurations. `.cargo/config.toml`
defines an alias for each command CI runs:

| alias | expands to | CI job |
|---|---|---|
| `cargo fmt-check` | `cargo fmt --all -- --check` | `full` |
| `cargo lint` | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | `full` |
| `cargo lint-min` | `cargo clippy -p strapdown-core --all-targets --no-default-features -- -D warnings` | `minimal` |
| `cargo test-min` | `cargo test -p strapdown-core --no-default-features` | `minimal` |

Run all four, plus `cargo test --workspace --all-features`, before pushing. `cargo lint` alone
is not enough: an import used only under `--all-features` is dead under
`--no-default-features`, and only `cargo lint-min` sees it. `cargo lint-fix` applies the fixes
clippy can make itself, and `cargo fmt --all` formats.

`clippy::pedantic`, `clippy::nursery`, `missing_docs` and the zero-panic lints
(`unwrap_used`, `expect_used`, `panic`, in library code) are all denied. Relax a lint once, in
`[workspace.lints.clippy]` in the root `Cargo.toml` with a reason, rather than with a call-site
`#[allow]`. `AGENTS.md` lists the existing exceptions and why.

The Python half has its own gate: `just check-python` runs `uv run ruff check`,
`uv run ruff format --check` and `uv run pytest -q`.

## Documentation

### API documentation

```bash
cargo docs                                   # alias for cargo doc --workspace --no-deps
cargo doc --workspace --no-deps --open       # and open it in a browser
```

See [API Documentation](../api/index.md) for what each crate contains and where the published
copy lives.

### Building the book

The book is built with mdBook. CI pins version 0.4.40, so install that one:

```bash
cargo install mdbook --version 0.4.40 --locked
mdbook build book          # output in book/book/
mdbook serve book          # live-reloading server on http://localhost:3000
```

Run them from the repository root. Pages under `book/src/` include files from elsewhere in the
repository (Rust examples by anchor, `CONTRIBUTING.md`, the generated
`development/baseline-tables.md`), so an edit to those files changes the book too.

### Publishing the book

Publishing is automatic. `.github/workflows/deploy-book.yml` builds the book on every push to
`main` (and on pull requests that touch the book or the crates, without deploying), adds the
workspace rustdoc under `api/`, checks the links, and deploys the result to GitHub Pages at
<https://jbrodovsky.github.io/strapdown-rs/>.

For that to work, the repository's **Settings → Pages → Build and deployment → Source** must be
set to **GitHub Actions**; that is a one-time setting. There is no custom domain: the `cname`
line in `book/book.toml` is commented out, so the site is served from the default
`github.io` address.

## Continuous integration

The workflows in `.github/workflows/`:

| workflow | what it does |
|---|---|
| `rust.yml` | the blocking Rust gate. `minimal`: `strapdown-core` with no default features (build, `clippy -D warnings`, test) on Linux, macOS and Windows. `full`: format check, `clippy -D warnings` and the whole test suite with all features, on Linux. `vendored`: builds the vendored netCDF/HDF5 on macOS and Windows. Two advisory jobs: `platform-spread`, which reports how far the accuracy metrics differ across the three platforms, and `semver`, which runs `cargo semver-checks` against the last release |
| `python.yml` | lints and tests the `analysis` package (`ruff check`, `ruff format --check`, `pytest`), deliberately without GMT installed |
| `deploy-book.yml` | builds this book and the rustdoc, checks links, and deploys to GitHub Pages |
| `publish.yml` | verifies, then publishes the crates to crates.io on a release; a manual run takes a `dry_run` input |
| `draft-pdf.yml` | renders the JOSS paper in `papers/joss/` when it changes |
| `copilot-setup-steps.yml` | toolchain provisioning for the Copilot coding agent only |

Every job sets `timeout-minutes`, so a hang fails instead of running to GitHub's 360-minute
default.

## References

- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation
  Systems*, 2nd ed. Artech House.
- Sensor Logger app: <https://www.tszheichoi.com/sensorlogger>
