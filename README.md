# strapdown-rs

[![JOSS review status](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4/status.svg)](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4)
[![CI](https://github.com/jbrodovsky/strapdown-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/jbrodovsky/strapdown-rs/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/strapdown-core.svg)](https://crates.io/crates/strapdown-core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Strapdown inertial navigation and GNSS-degradation simulation in Rust.**

`strapdown-rs` is a library and command-line toolkit for research on inertial navigation
systems (INS), particularly when GNSS is unavailable or untrustworthy. It provides:

- local-level strapdown mechanization (Groves, *Principles of GNSS, Inertial, and Multisensor
  Integrated Navigation Systems*, 2nd ed., §5.4);
- four loosely coupled navigation filters behind one trait;
- a simulator that replays recorded or synthetic IMU/GNSS data while removing, thinning or
  corrupting the GNSS stream according to a seeded scenario file.

**📖 [User guide](https://jbrodovsky.github.io/strapdown-rs/)** ·
**📚 [API documentation](https://docs.rs/strapdown-core)** ·
**▶️ [Runnable examples](#runnable-examples)** ·
**🔧 [Example scenarios](examples/configs/)**

## Statement of need

Navigation algorithms are commonly prototyped in MATLAB or Python. There they are fast only
once rewritten in vectorized form, which is especially awkward for particle filters. Real
GNSS-denied data is expensive, regulated and impossible to repeat exactly.

`strapdown-rs` addresses both problems:
- Filters are written plainly and compiled, so the same code serves research scripts,
  experiments and applications.
- Any trajectory can be run under controlled, reproducible GNSS outages and faults, so
  algorithms are compared under identical conditions.

It is aimed at graduate students, researchers and engineers working on INS/GNSS integration
and alternative positioning, navigation and timing (PNT).

## What it does

- **Filters.** All four implement the `NavigationFilter` trait:
  - error-state Kalman filter (ESKF, the default), 15 states;
  - extended Kalman filter (EKF);
  - unscented Kalman filter (UKF);
  - Rao-Blackwellized particle filter after Canciani & Raquet (2017).
- **Aiding.** GNSS position and velocity, barometric altitude with bias estimation,
  magnetometer heading (World Magnetic Model declination), and ZUPT/ZARU. Chi-squared
  innovation gating with recovery.
- **Initialization.** Coarse alignment, IMU calibration and lever-arm compensation via the
  `InsEngine` builder.
- **GNSS degradation.**
  - Schedulers: pass-through, fixed interval, duty-cycle outages.
  - Faults: correlated AR(1) noise, slow bias drift, position hijacking, combinations.
  - Configured from TOML, YAML or JSON files, or from CLI flags.
- **Synthetic trajectories** with IMU error models by sensor grade (`strapdown-sim syn`).
- **Metrics.** RMSE, CEP, NEES and NIS.
- **Output.** The CLI writes CSV. The library also writes HDF5, NetCDF and MCAP.
- **Experimental:** gravity and magnetic anomaly map aiding (`strapdown-geonav`).

It is loosely coupled only. There are no pseudorange or carrier-phase models, and no
raw-signal processing.

## Repository layout

The repository is a two-language monorepo. The Rust crates form a Cargo workspace. The
Python package is a [uv](https://docs.astral.sh/uv/) workspace declared by the root
`pyproject.toml`. Neither is a member of the other.

| Path | Language | What it is |
|---|---|---|
| [`core/`](core/) | Rust | `strapdown-core`: mechanization, filters, measurement models, scenario engine |
| [`sim/`](sim/) | Rust | `strapdown-sim`: the command-line simulator |
| [`geonav/`](geonav/) | Rust | `strapdown-geonav`: experimental gravity and magnetic anomaly aiding |
| [`analysis/`](analysis/) | Python | the `analyze` CLI: preprocessing, measurement statistics, result comparison |
| [`examples/`](examples/) | — | scenario configuration files |
| [`conf/`](conf/) | — | the experiment recipes behind the published results |

## Installation

Rust is the only hard requirement. `rust-toolchain.toml` fetches the pinned toolchain (1.91)
on the first `cargo` command.

**Library only.** `strapdown-core` with default features needs nothing else:

```bash
cargo add strapdown-core --git https://github.com/jbrodovsky/strapdown-rs
```

The library crate is imported as `strapdown`:

```rust
use strapdown::StrapdownState;
```

**Simulator.**

```bash
cargo install --git https://github.com/jbrodovsky/strapdown-rs strapdown-sim
```

Its default `plotting` feature, and the HDF5 and NetCDF features, compile C libraries from
vendored sources. Those builds need a C/C++ compiler and **cmake 3.26 or newer**, and
`HDF5_DIR` must be unset. See the [installation guide](https://jbrodovsky.github.io/strapdown-rs/installation/installation.html).

The crates are also published to crates.io. Version 1.0.0 is released there together with the
JOSS paper; until then, install from git as above.

**Developing.** From a clone:

```bash
cargo build --workspace --release
```

```bash
uv sync
```

`just setup` runs both. The `analyze` map-drawing subcommands also need the GMT C library;
every other subcommand runs without it.

## Quick start

```bash
# A 10-minute synthetic trajectory with truth, IMU and GNSS columns
strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42

# Closed loop (ESKF) with GNSS available for 100 s, then denied for 50 s, repeating
strapdown-sim cl -i synthetic.csv -o eskf.csv --seed 42 --sched duty --on-s 100 --off-s 50

# Dead reckoning: no aiding at all
strapdown-sim dr -i synthetic.csv -o dr.csv
```

Use `--filter ukf` or `--filter ekf` to select another filter. `strapdown-sim pf` runs the
particle filter. `strapdown-sim <command> --help` lists every option. Logging is controlled
with `--log-level` and `--log-file`; see the
[logging guide](https://jbrodovsky.github.io/strapdown-rs/user-guide/logging.html).

Geophysical aiding is experimental and behind a feature:

```bash
cargo install --git https://github.com/jbrodovsky/strapdown-rs strapdown-sim --features geonav
```

```bash
strapdown-sim cl -i data.csv -o out.csv --geo --gravity-resolution one-minute
```

## Runnable examples

Two worked examples live in [`core/examples/`](core/examples/). Both generate their own
trajectory, so there is no dataset to fetch first.

```bash
# Minimal InsEngine usage: propagate IMU, fuse GNSS, read the solution.
cargo run -p strapdown-core --example basic_ins
```

```bash
# Coasting through a GNSS outage, and recovery when the signal returns.
cargo run -p strapdown-core --example gnss_outage
```

**Read `basic_ins` first.** It is the propagate/update/read loop that every application is
built around, on a trajectory simple enough to check by hand. It also spells out the
specific-force sign convention, the most common way to get a first integration wrong: an
accelerometer at rest reads **−9.81 m/s² on the down axis** in NED, because it senses the
ground pushing up.

**Then `gnss_outage`.** It shows the error staying flat while aided, growing while coasting,
and collapsing on recovery. It then explains why the coasting error is several times
*smaller* than double-integrating the accelerometer bias would predict: the filter absorbed
most of the bias into a fraction of a degree of pitch, which position and velocity aiding
cannot tell apart from bias.

## Running the tests

```bash
# Every Rust test, all features (as CI runs it)
cargo test --workspace --all-features
```

```bash
# The minimal-feature configuration CI also tests
cargo test-min
```

```bash
# The Python analysis package
uv run pytest -q
```

The accuracy regression suite (`core/tests/perf_baseline.rs`) runs every filter over
recorded and synthetic scenarios. It fails if a gated metric moves beyond tolerance. Its
results are published as the
[performance baselines](https://jbrodovsky.github.io/strapdown-rs/development/performance.html)
page. CI runs on Linux, macOS and Windows.

## Contributing and support

- Bugs and feature requests: [open an issue](https://github.com/jbrodovsky/strapdown-rs/issues).
- Questions about using the toolkit: open an issue with the *question* template.
- Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) for the setup, the lint gate and the
  pull-request checklist.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Citing

If you use `strapdown-rs` in research, please cite it using [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button). The JOSS paper is under review; its DOI will be
added on acceptance.

## License

MIT. See [LICENSE](LICENSE).
