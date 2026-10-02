# Introduction

[![JOSS review status](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4/status.svg)](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4)
[![CI](https://github.com/jbrodovsky/strapdown-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/jbrodovsky/strapdown-rs/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/strapdown-core.svg)](https://crates.io/crates/strapdown-core)
[![docs.rs](https://docs.rs/strapdown-core/badge.svg)](https://docs.rs/strapdown-core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jbrodovsky/strapdown-rs/blob/main/LICENSE)

`strapdown-rs` is a Rust library and command-line simulator for research on strapdown inertial
navigation systems (INS), with an emphasis on what happens when GNSS is unavailable or cannot be
trusted. It mechanizes recorded or synthetic IMU data, fuses it with GNSS and other aiding
sources through one of four navigation filters, and lets you remove, thin or corrupt the GNSS
stream according to a seeded, repeatable scenario.

> The crates.io and docs.rs badges show the most recent *published* release, which predates
> 1.0. Version 1.0.0 is published together with the JOSS paper; until then, install from git
> as described in [Installation](./installation/installation.md).

## What it provides

The repository is a Cargo workspace of three crates:

| Crate | Imported as | What it is |
| --- | --- | --- |
| `strapdown-core` | `strapdown` | Mechanization, Earth model, filters, measurement models, the scenario engine and CSV I/O |
| `strapdown-sim` | (binary) | The `strapdown-sim` command-line simulator |
| `strapdown-geonav` | `geonav` | **Experimental** gravity and magnetic anomaly map aiding |

The pieces that matter most:

- **Strapdown mechanization** in the local-level (north, east, vertical) frame, following
  Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd
  ed., §5.4-5.5. See [The Navigation Model](./user-guide/concepts.md).
- **Four navigation filters**, all behind the `NavigationFilter` trait:
  - the error-state Kalman filter (ESKF), the default for closed-loop runs;
  - the extended Kalman filter (EKF);
  - the unscented Kalman filter (UKF);
  - a Rao-Blackwellized particle filter (RBPF) after Canciani & Raquet (2017), the only
    particle filter in the crate.
- **Aiding**: GNSS position and velocity, barometric relative altitude (with a barometric
  bias state, on by default in the simulator) and magnetometer heading. The library adds
  zero-velocity and zero-angular-rate updates driven by a stationary detector.
- **GNSS degradation**: schedulers that decide *when* fixes arrive (pass-through, fixed
  interval, duty-cycled outages) and fault models that decide *what* they contain (AR(1)
  degradation, slow bias drift, position hijack). See [Fault Simulation](./gnss/fault-simulation.md).
- **Synthetic trajectories** with IMU error models by sensor grade (`strapdown-sim syn`), so
  every example in this book runs without a dataset.
- **Reproducibility**: the random draws are seeded (`--seed`, or `seed` in a configuration
  file), so the same inputs and seed give a byte-identical result, and a whole run can be
  described by one TOML, YAML or JSON file.

## What it is not

The boundaries, stated up front:

- **Loosely coupled only.** Aiding enters as position, velocity, altitude or heading
  measurements. There are no pseudorange or carrier-phase models, no satellite geometry and no
  raw-signal processing.
- **Not IMU firmware or a driver.** It reads logged samples (CSV) or samples you hand to the
  library; it does not talk to hardware.
- **No coning or sculling compensation.** Each IMU sample is integrated to first order, with
  the attitude averaged across the interval when the specific force is resolved. That suits
  the sample rates and dynamics the project targets. It does not suit high-rate,
  high-vibration, navigation-grade work.
- **Local-level frame only.** The mechanization is valid from 11 km below to 30 km above the
  ellipsoid; there is no ECEF or ECI mechanization.
- **The CLI writes CSV only.** HDF5, NetCDF and MCAP writers exist as library methods on
  `NavigationResult`, behind cargo features. There is no Parquet support.
- **One input format.** The simulator reads the
  [Sensor Logger](https://www.tszheichoi.com/sensorlogger) CSV layout, which `syn` also
  writes. Other datasets must be converted to that layout first; see
  [Input Data Format](./user-guide/data-format.md).
- **Geophysical navigation is experimental.** `strapdown-geonav` is a research tool held at
  0.x so its API can change.

## Who it is for

- **Researchers** comparing navigation filters or aiding strategies under identical, repeatable
  GNSS outages and faults.
- **Graduate students** learning strapdown navigation. Variables are named after the
  quantities in Groves rather than after single letters, and every equation can be run.
- **Engineers** prototyping INS/GNSS integration who want a tested reference implementation to
  compare their own against.

## How this book is organised

| Part | Read it when |
| --- | --- |
| [Installation](./installation/installation.md) and [Quick Start](./quick-start.md) | You want to run something in the next ten minutes |
| [User Guide](./user-guide/overview.md) | You want the model, the frames, the state vector, and every subcommand |
| [Navigation Filters](./filters/kalman.md) | You need to know how a particular filter works and how to tune it |
| [GNSS Degradation](./gnss/fault-simulation.md) | You are designing outage or spoofing scenarios |
| [Geophysical Navigation](./geonav/overview.md) | You work with gravity or magnetic anomaly maps (experimental) |
| [API Documentation](./api/index.md) | You are calling the library from Rust |
| [Examples and Tutorials](./examples/configurations.md) | You learn best from complete worked runs |
| [Development](./development/contributing.md) | You want to build, test or contribute |

## Getting help

- Questions and bug reports: [GitHub issues](https://github.com/jbrodovsky/strapdown-rs/issues).
- Frequently asked questions: [FAQ](./faq.md).
- Every subcommand documents its own flags: `strapdown-sim <command> --help`.

## Citation

If you use `strapdown-rs` in research, please cite it using the repository's
[`CITATION.cff`](https://github.com/jbrodovsky/strapdown-rs/blob/main/CITATION.cff) (GitHub's
"Cite this repository" button). The JOSS paper is under review
([openjournals/joss-reviews#11377](https://github.com/openjournals/joss-reviews/issues/11377));
its DOI will be added on acceptance.

## License

MIT. See [LICENSE](https://github.com/jbrodovsky/strapdown-rs/blob/main/LICENSE).
