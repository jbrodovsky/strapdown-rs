# Architecture

This page describes how the workspace is put together: which crate depends on which, how a
simulation's data moves from a CSV file to a CSV file, and the rules every crate follows for
errors, determinism and optional features. For a module-by-module list, see
[API Documentation](../api/index.md).

## Crates

The repository is a Cargo workspace of three crates, plus a separate Python package.

| crate | kind | imported as | version | role |
|---|---|---|---|---|
| `strapdown-core` | library | `strapdown` | 1.0.0 | mechanization, filters, measurement models, event streams, simulation framework |
| `strapdown-sim` | binary | -- | 1.0.0 | the `strapdown-sim` command-line tool |
| `strapdown-geonav` | library | `geonav` | 0.1.0, experimental | geophysical map aiding |
| `analysis` | Python package | `analyze` CLI | -- | preprocessing and post-processing; not a Cargo member ([Python Analysis Tooling](./analysis.md)) |

```text
                    +--------------------+
                    |   strapdown-sim    |   binary: CLI, config files, CSV output,
                    |   (bin)            |   parallel runs (rayon), plots (plotters)
                    +----+----------+----+
                         |          | optional, feature "geonav"
                         |          v
                         |   +------------------+
                         |   | strapdown-geonav |   maps (netcdf), WMM,
                         |   | (lib "geonav")   |   gravity / magnetic models
                         |   +--------+---------+
                         |            |
                         v            v
                    +--------------------+
                    |  strapdown-core    |   nalgebra, serde, rand, csv, WMM;
                    |  (lib "strapdown") |   optional hdf5 / netcdf / mcap / clap
                    +--------------------+
```

`strapdown-core` depends on no other workspace crate and needs no C toolchain in its default
configuration. `strapdown-sim` enables core's `clap` feature for its argument enums.
`strapdown-geonav` always links libnetcdf (built from source), which is why `strapdown-sim`
only pulls it in behind the `geonav` feature.

## Layers inside `strapdown-core`

| layer | modules | what lives there |
|---|---|---|
| foundations | `earth`, `linalg`, `error` | WGS84 geometry and gravity, covariance square roots, `StrapdownError` |
| mechanization | crate root (`lib.rs`) | `StrapdownState`, `IMUData`, `forward`: the 9-state strapdown equations of Groves §5.4–5.5 |
| initialisation | `alignment`, `calibration`, `stationary` | coarse alignment, IMU calibration, stationary detection for ZUPT/ZARU |
| measurements | `measurements`, `gating` | the `MeasurementModel` trait and its models; innovation gating |
| estimation | `kalman`, `linearize`, `particle`, `rbpf` | ESKF, EKF, UKF, the RBPF, and the Jacobians they share |
| framework | `messages`, `sim`, `metrics`, `engine` | event streams, records and results, runners, scoring, the `InsEngine` facade |

Lower layers do not know about higher ones: the mechanization knows nothing about filters, and
the filters know nothing about files or schedules.

## Data flow

Every simulation mode is the same pipeline with a different estimator in the middle:

```text
 input.csv
    |  TestDataRecord::from_csv                      (sim)
    v
 Vec<TestDataRecord>   one row per sample: IMU, GNSS, barometer, magnetometer, ...
    |  check_declared_frame(records, is_enu)         (sim; rejects a wrong NED/ENU declaration)
    |
    |  build_event_stream(records, &AidingConfig, is_enu)            (messages)
    |    - MeasurementScheduler per channel: GNSS, barometer, magnetometer  -> *when*
    |    - GnssFaultModel on each delivered GNSS fix                        -> *what*
    |    - seeded RNG from AidingConfig::seed
    |  geonav::build_event_stream adds gravity/magnetic events on top  (geonav, optional)
    v
 EventStream { start_time, events: Vec<Event> }
    |    Event::Imu { dt_s, imu, elapsed_s }
    |    Event::Measurement { meas: Box<dyn MeasurementModel>, elapsed_s }
    |
    |  run_closed_loop(&mut filter, stream, health_limits, execution_limits)   (sim)
    |    for each event:  Imu         -> filter.predict(imu, dt)
    |                     Measurement -> filter.update(meas)   (gated, NIS reported)
    |    HealthMonitor and ExecutionMonitor can stop a diverging or stalled run
    v
 Vec<NavigationResult>   one row per timestamp: state, IMU and map biases, covariance
    |  NavigationResult::to_csv                      (sim; to_hdf5 / to_netcdf / to_mcap in the library)
    v
 output.csv
```

The variations are:

- **Dead reckoning** (`dr`) skips the event stream:
  `sim::dead_reckoning_with_limits(records, is_enu, health_limits, execution_limits)`
  mechanizes the IMU rows directly, checks the position, speed and wall-clock limits after each
  step, and reports no covariance. `sim::dead_reckoning(records, is_enu)` is the same with no
  limits.
- **Closed loop** (`cl`) builds one of the Kalman filters and calls `run_closed_loop`;
  with geophysical maps it calls `run_closed_loop_with_geo`, which also reports the map biases.
- **Particle filter** (`pf`) consumes the same `EventStream` through an event loop in
  `strapdown-sim` that drives the RBPF.
- **Synthetic** (`syn`) runs the other way: `sim::generate_synthetic` produces both a truth
  trajectory and the sensor records a vehicle on it would have logged, and writes the records
  as a CSV that the other modes can read.
- **Online use** from Rust does not need a stream at all: `InsEngine` wraps any
  `NavigationFilter` and accepts IMU samples and GNSS fixes one at a time. See
  [Using the Library](../user-guide/library.md).

### Where degradation sits

Scheduling and fault injection happen **once, in `build_event_stream`**, before any filter
runs. The filter never knows a fix was dropped or corrupted; it sees only the events it is
given. That is what makes comparisons fair: every filter run on the same records with the same
`AidingConfig` sees an identical stream, and swapping the filter changes nothing upstream. It
is also why `strapdown-geonav` delegates to core's builder rather than reimplementing it, and
why `geonav/tests/builder_equivalence.rs` checks that the two produce the same non-geophysical
events. See [Schedulers and Faults Reference](../gnss/scenarios.md).

## Trait boundaries

| trait | defined in | implemented by | contract |
|---|---|---|---|
| `NavigationFilter` | crate root | ESKF, EKF, UKF, RBPF | `predict(&dyn InputModel, dt)`, `update(&dyn MeasurementModel)` returning an `UpdateOutcome`, `get_estimate`, `get_certainty`, and optional gating hooks |
| `InputModel` | crate root | `IMUData`, `ImuSample`, `VelocityData` | the control input to `predict`, taken as a trait object so that `NavigationFilter` is object-safe |
| `MeasurementModel` | `measurements` | GNSS, barometer, magnetometer, ZUPT, ZARU; geonav's gravity and magnetic models | the measurement, its prediction from a state, its noise and its Jacobian |
| `Particle` | `particle` | nothing in the workspace; it is for user-built particle filters | the hooks the generic resampling and averaging strategies in `particle` need |

Adding a sensor means implementing `MeasurementModel` and emitting events for it; adding an
estimator means implementing `NavigationFilter`. Neither requires touching the other, the
event stream, or the runner.

## Errors

Library code never panics. `unwrap_used`, `expect_used` and `panic` are denied for library code
(tests are exempt), and every fallible function returns `Result<_, StrapdownError>`, documented
with an `# Errors` section. `StrapdownError` is one enum with variants for shape errors,
non-finite values, invalid configuration, map loading, divergence, sensor gaps and timeouts.

`StrapdownError::is_recoverable` separates the conditions a run should skip from the ones that
should stop it. An estimate off a geophysical map, a measurement that cannot be evaluated, and
an external model (the World Magnetic Model) refusing a query are recoverable: the runner
skips that measurement and counts it. Everything else ends the run with the error.

The lints that enforce this are workspace-wide and at `deny`, including `clippy::pedantic`,
`clippy::nursery` and `missing_docs`; see [Building and Testing](./building.md#linting-and-formatting).

## Determinism

A run is a function of its inputs and its seeds. There are no wall-clock or thread-dependent
inputs to the navigation solution:

- the GNSS fault model draws from a `StdRng` seeded from `AidingConfig::seed` (`[aiding] seed`
  in a config file, falling back to the top-level `seed` through
  `SimulationConfig::resolved_aiding`; `--seed` on the command line; 42 when nothing sets it);
- the RBPF seeds its own `StdRng` from its configuration's seed;
- `generate_synthetic` takes the seeded RNG it draws from.

`core/tests/integration_tests.rs` checks that repeated runs are identical, and the
[performance baseline](./performance.md) is gated on all three CI platforms, which catches
results that depend on the platform. `--parallel` processes *files* concurrently; each file's
run is still sequential.

## Feature gating

| crate | feature | default | adds |
|---|---|---|---|
| `strapdown-core` | `clap` | off | `clap` derives on configuration enums |
| `strapdown-core` | `hdf5`, `netcdf`, `mcap` | off | binary I/O on `TestDataRecord` and `NavigationResult` |
| `strapdown-core` | `full` | off | all four |
| `strapdown-sim` | `plotting` | **on** | the `--plot` figure; loads libfontconfig at run time |
| `strapdown-sim` | `geonav` | off | `--geo` and the `[geophysical]` section |

Because an item used only behind a feature is dead code without it, CI lints and tests both
`--all-features` and core's `--no-default-features`; see
[Building and Testing](./building.md#linting-and-formatting).
