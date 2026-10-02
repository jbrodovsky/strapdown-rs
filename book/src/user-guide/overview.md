# User Guide Overview

This part of the book explains what the toolkit computes and how to drive it. This page is a
map: what the pieces are, and which page documents each one.

## The crates

| Crate | Use it for | Documented in |
| --- | --- | --- |
| `strapdown-core` (imported as `strapdown`) | Mechanization, filters, measurement models, scenario engine, CSV I/O | [Using the Library](./library.md), [API Documentation](../api/index.md) |
| `strapdown-sim` | Running simulations from the command line or a scenario file | [Running Simulations](./simulations.md) and the pages under it |
| `strapdown-geonav` | **Experimental** gravity and magnetic anomaly map aiding | [Geophysical Navigation](../geonav/overview.md) |

The simulator is a thin layer over the library: every mode it offers is a library function
you can call from Rust.

## The model

Three pages describe what the numbers mean, and are worth reading before interpreting any
output:

- [The Navigation Model](./concepts.md): strapdown mechanization in the local-level frame,
  the gravity and Earth-rate models, and how aiding corrects the solution.
- [Coordinate Frames](./coordinate-frames.md): NED by default, ENU on request, the body frame,
  and the check that refuses a wrongly declared frame.
- [State Representation](./state-representation.md): the nine navigation states, the filters'
  15-element bias-augmented states and their extra bias states, and how covariance reaches the
  output.

## The subcommands

`strapdown-sim` has one subcommand per mode:

| Subcommand | What it does | Page |
| --- | --- | --- |
| `dr` | Dead reckoning: propagates the IMU from the initial state with no aiding | [Dead Reckoning](./dead-reckoning.md) |
| `cl` | Closed loop: a Kalman filter corrects the INS with GNSS, barometer and magnetometer measurements, and feeds the corrections back. ESKF by default; `--filter ekf` or `--filter ukf` selects the others | [Closed Loop](./closed-loop.md) |
| `pf` | Closed loop with the Rao-Blackwellized particle filter | [Particle Filter](./particle-filter.md) |
| `syn` | Generates a synthetic trajectory: noisy sensor records, or with `--no-noise`, the truth | [Synthetic Trajectories](./synthetic.md) |
| `config` | An interactive wizard that writes a scenario file | [Configuration Files](./configuration.md) |
| `ol` | **Not implemented.** Reserved for an open-loop mode; it validates its paths and writes no output | -- |

Dead reckoning is `dr`. It is not "open loop": `ol` names a different, unimplemented mode in
which a filter would estimate errors without feeding them back.

A few options are shared:

- **Global:** `--config <file>` runs a whole scenario from a TOML, YAML or JSON file instead
  of subcommand flags; `--log-level` and `--log-file` control logging ([Logging](./logging.md));
  `--parallel` processes several input files at once; `--plot` draws a performance plot
  against the GNSS track.
- **Every simulation mode:** `-i` takes a CSV file or a directory of them, `-o` a CSV file or a
  directory; `--enu` declares ENU input; health limits (`--health-*`) and execution limits
  (`--max-wall-clock-*`, `--max-no-progress-s`) stop a run that diverges or hangs.
- **`cl` and `pf`:** `--seed`, the GNSS scheduler (`--sched` and its parameters) and the GNSS
  fault model (`--fault` and its parameters).

`strapdown-sim <subcommand> --help` lists every flag with its default. The simulator writes
**CSV only**; see [Output Format](./output-format.md).

## The filters

All four implement the library's `NavigationFilter` trait, so the simulator drives them through
the same loop.

| Filter | Selected by | State it carries | Page |
| --- | --- | --- | --- |
| Error-state Kalman filter (ESKF) | `cl` (the default) | Nominal navigation state plus a 15-element error state, 16 with the barometric bias | [ESKF](../filters/eskf.md) |
| Extended Kalman filter (EKF) | `cl --filter ekf` | Full 15-element state, plus extra bias states | [EKF](../filters/ekf.md) |
| Unscented Kalman filter (UKF) | `cl --filter ukf` | Full 15-element state, plus extra bias states | [UKF](../filters/ukf.md) |
| Rao-Blackwellized particle filter (RBPF) | `pf` | Particles over horizontal position error; one shared Kalman filter for the rest. No IMU-bias states | [RBPF](../filters/rbpf.md) |

How they differ, and how they score on the reference scenarios, is on the
[Comparison](../filters/comparison.md) and [Performance Baselines](../development/performance.md)
pages. The RBPF is the only particle filter: `particle.rs` provides building blocks
(resampling, averaging), not a second filter ([Particle Building
Blocks](../filters/particle-filter.md)).

## Aiding and degradation

- **Measurement models** -- GNSS position and velocity, barometric altitude, magnetometer
  heading, and the library's zero-velocity and zero-angular-rate updates -- and innovation
  gating: [Measurement Models and Integrity](../filters/measurements.md). All aiding is loosely
  coupled; there are no pseudorange or carrier-phase models.
- **GNSS degradation** -- schedulers decide when fixes arrive (`passthrough`, `fixed`, `duty`)
  and fault models decide what they contain (`none`, `degraded`, `slowbias`, `hijack`):
  [Fault Simulation](../gnss/fault-simulation.md) and the
  [Schedulers and Faults Reference](../gnss/scenarios.md).

## Data in, data out

- [Input Data Format](./data-format.md): the Sensor Logger CSV layout that every mode reads and
  `syn` writes.
- [Output Format](./output-format.md): the `NavigationResult` columns, their units, and the
  covariance columns.
- [Configuration Files](./configuration.md): every key of a scenario file.

## A typical workflow

1. Get input: a recording in the Sensor Logger layout, or `strapdown-sim syn`.
2. Declare its frame: nothing for NED (what `syn` writes), `--enu` for a phone export.
3. Run a baseline: `cl` with GNSS uninterrupted, and `dr` for the unaided bound.
4. Run the scenario you care about: an outage (`--sched duty`), a fault (`--fault ...`), or a
   different filter, with a fixed `--seed`.
5. Write the scenario into a configuration file once it settles, so the run can be repeated
   exactly.
6. Compare the CSVs, against the truth from `syn --no-noise` when you have it.

The [Quick Start](../quick-start.md) walks through steps 1-6 on a synthetic trajectory.
