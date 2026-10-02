# Using the Library

`strapdown-sim` is one consumer of `strapdown-core`; your own program can be another. This page
covers the library's user-facing layer: the `InsEngine` most programs should start from, the
pieces that sit in front of it (calibration, coarse alignment, an initial covariance from the
IMU grade, the antenna lever arm), error handling, and scoring a run against truth.

The package is `strapdown-core` and the crate is imported as **`strapdown`**. Until v1.0.0 is on
crates.io it is added from git; [Installation](../installation/installation.md) has the exact
`Cargo.toml` line and the feature list.

Every Rust snippet on this page is included from an example under `core/examples/` that is
compiled and linted with the rest of the workspace:

| example | shows |
| --- | --- |
| `basic_ins` | the `InsEngine` loop |
| `gnss_outage` | coasting through a GNSS outage |
| `lever_arm` | calibration, alignment, `auto_covariance`, the lever arm, recoverable errors |
| `kalman_filters` | the three Kalman filters behind `NavigationFilter` |
| `eskf` | `initialize_eskf`, an event stream, `run_closed_loop`, `metrics::evaluate` |
| `aiding` | ZUPT, ZARU, barometer, magnetometer, innovation gating |
| `score_run` | scoring a `strapdown-sim` output CSV against a truth CSV |

```bash
cargo run -p strapdown-core --example lever_arm
```

## The `InsEngine`

`engine::InsEngine` (re-exported at the crate root) owns a `NavigationFilter` (the 15-state
[ESKF](../filters/eskf.md) by default), advances it at the IMU rate, corrects it whenever aiding
arrives, and reports a `NavSolution` in degrees, metres and metres per second rather than the
radians the filter carries. The loop is the same whatever the data source:

1. build an engine from an initial state;
2. `predict` for every IMU sample;
3. `update_gnss` (or `update`) whenever a measurement arrives;
4. read `nav_solution` whenever you need the estimate.

From [`core/examples/basic_ins.rs`](https://github.com/jbrodovsky/strapdown-rs/blob/main/core/examples/basic_ins.rs),
which [Tutorial: Basic INS](../examples/tutorial-basic.md) walks through:

```rust,ignore
{{#include ../../../core/examples/basic_ins.rs:build}}
```

```rust,ignore
{{#include ../../../core/examples/basic_ins.rs:loop}}
```

Aiding is asynchronous: the engine's clock advances only on `predict`, and an update applies at
whatever epoch the last `predict` left it at. Propagate up to a measurement's time of validity
before applying it.

### Builder and configuration

`InsEngine::builder()` returns an `InsEngineBuilder`; every setter is optional.

| builder method | sets |
| --- | --- |
| `with_initial_state(InitialState)` | the initial state; its `is_enu` becomes the engine's frame unless one was named |
| `with_frame(is_enu)` | the frame explicitly; a contradicting initial state is then a build error |
| `with_config(InsEngineConfig)` | the whole configuration (counts as naming a frame) |
| `with_lever_arm([x, y, z])` | antenna offset, metres, body frame |
| `with_initial_covariance(Vec<f64>)` | $P_0$, 15 entries |
| `with_process_noise(Vec<f64>)` | the process-noise density $q$, 15 entries |
| `with_imu_biases(accel, gyro)` | initial bias estimates, m/s² and rad/s |
| `with_filter(Box<dyn NavigationFilter>)` | drive a filter you built yourself instead of the default ESKF |
| `build()` | returns `Result<InsEngine, StrapdownError>` |

`build` returns `StrapdownError::InvalidConfiguration` for a non-finite lever arm or one longer
than 100 m, a covariance or noise diagonal that is not 15 finite non-negative entries, a
contradiction between the configured frame and the initial state's, or a supplied filter with
fewer than nine states. With `with_filter`, the covariance, noise and bias settings are not
applied (the filter was built with its own); the frame and lever arm still are.

`InsEngineConfig` holds the same settings as a serializable struct (`is_enu`, `lever_arm`,
`process_noise_diagonal`, `initial_covariance_diagonal`, `initial_accel_bias`,
`initial_gyro_bias`). It is `#[non_exhaustive]`: start from `InsEngineConfig::default()` (NED, no
lever arm, default noise) and assign fields. Without `with_initial_covariance`, the default ESKF
starts from a fixed $P_0$: 10 m of horizontal and vertical position, $10^{-3}$ (m/s)² of
velocity, $10^{-5}$ rad² of attitude, and $10^{-6}$ (m/s²)² and $10^{-8}$ (rad/s)² on the
biases. For a $P_0$ that describes your sensor, pass `IMUQuality::auto_covariance`
([below](#imu-grades)).

### Running the engine

| method | does |
| --- | --- |
| `predict(&ImuSample)` | propagate by one sample of increments |
| `predict_rates(&IMUData, dt)` | the same from rates, through `ImuSample::from_rates` |
| `update_gnss(&GnssFix)` | apply an antenna-referred fix: position, then velocity if `GnssFix::with_velocity` gave one |
| `update(&dyn MeasurementModel)` | apply any other [measurement model](../filters/measurements.md) |
| `set_innovation_gate`, `set_gate_recovery` | install a gate and tune its recovery ([gating](../filters/measurements.md#innovation-gating)) |
| `nav_solution()` | the current `NavSolution` |
| `covariance()` | the filter's covariance, in its own units |
| `set_lever_arm([x, y, z])` | change the antenna offset on a running engine, validated like the builder |
| `filter()` | borrow the underlying `&dyn NavigationFilter` |

`update_gnss` and `update` return an [`UpdateOutcome`](../filters/kalman.md#outputs-updateoutcome).
When a gate rejects the position leg of a fix, its velocity leg is skipped too.

`NavSolution` carries `elapsed_s`; `latitude` and `longitude` (degrees); `altitude` (m, above
the ellipsoid); `velocity_north`, `velocity_east` and `velocity_vertical` (m/s; vertical is
positive down in NED); `roll`, `pitch` and `yaw` (degrees, yaw on -180..180, with
`heading_deg()` for 0..360); `accel_bias` and `gyro_bias` (zero if the filter carries no bias
states); `position_std_m` and `velocity_std_mps` (one-sigma north, east and vertical, converted
to metres); and `is_enu`.

## Antenna lever arm

The filter estimates the IMU's position, but a receiver reports its antenna's. The two differ by
the body-frame offset $r^b_{ant}$ rotated into the navigation frame, and a rotating vehicle adds
a velocity difference:

$$
p_{GNSS} = p_{IMU} + C_b^n r^b_{ant}, \qquad
v_{GNSS} = v_{IMU} + C_b^n \left(\omega^b_{ib} \times r^b_{ant}\right) .
$$

Set the lever arm (`with_lever_arm`, `InsEngineConfig::lever_arm` or `set_lever_arm`) and
`update_gnss` inverts both relations before the filter sees the fix, using the current attitude
estimate and the bias-corrected angular rate from the last `predict`. The offset is measured
from the IMU centre to the antenna phase centre, in metres, along the IMU's own axes (x forward,
y right, z down). A zero lever arm, the default, makes the compensation an exact identity.
`update` applies no lever arm: another sensor's mounting geometry belongs to its own model. The
free functions `engine::lever_arm_position_offset`, `lever_arm_velocity_offset` and
`shift_position_by_offset` expose the geometry.

[`core/examples/lever_arm.rs`](https://github.com/jbrodovsky/strapdown-rs/blob/main/core/examples/lever_arm.rs)
puts the antenna 1.5 m ahead of and 0.8 m above a stationary, tilted IMU, and feeds the same
antenna fixes to an engine that knows the lever arm and to one that does not:

```rust,ignore
{{#include ../../../core/examples/lever_arm.rs:engine}}
```

```rust,ignore
{{#include ../../../core/examples/lever_arm.rs:run}}
```

The compensated engine places the IMU where it is; the uncompensated one places it at the
antenna (the [output](#output-of-lever_armrs) is at the end of this page).

## IMU calibration

`calibration::ImuCalibration` removes known, deterministic sensor errors before the
mechanization. It uses Groves's sensor model (Section 4.4.1, Eq. 4.17 and 4.18),

$$
\tilde f^b = b + (I + M) f^b + w ,
$$

stored as the forward parameters a calibration report quotes: a bias $b$, per-axis scale-factor
errors (the diagonal of $M$) and misalignment coefficients (the off-diagonal of $M$, with a zero
diagonal). The model is inverted once, at construction, and `SensorCalibration::new` returns an
error for non-finite values, a non-zero misalignment diagonal or a singular $I + M$. If you have
the whole $M$ as one matrix, use `SensorCalibration::from_error_matrix`. The random noise $w$
cannot be removed; that is what the filter's process noise is for.

```rust,ignore
{{#include ../../../core/examples/lever_arm.rs:calibration}}
```

`correct(&ImuSample)` applies it to increments, where the bias enters as $b\ \Delta t$;
`correct_rates(&IMUData)` to rates. Neither the engine nor `strapdown-sim` calls it: apply it to
each sample yourself before `predict`, as `lever_arm.rs` does.

## Coarse alignment

A strapdown solution is relative to the attitude it starts from. `alignment` implements the
self-alignment estimators of Groves Section 5.6.3:

| function | estimates | needs |
| --- | --- | --- |
| `average_imu(&[IMUData])` | the mean of a window | samples you trust to be stationary |
| `coarse_leveling(&specific_force)` | roll and pitch (`LevelAttitude`) | a stationary accelerometer reading |
| `gyrocompassing(&imu, latitude_deg, IMUQuality, &GyrocompassConfig)` | heading and its one-sigma (`HeadingEstimate`) | a stationary gyro good enough to see Earth rate |
| `heading_from_velocity(v_n, v_e, minimum_speed)` | heading | the vehicle moving faster than `minimum_speed` |
| `attitude_from_level_and_heading(level, heading)` | a `Rotation3` | |

**Every function is NED**: a level IMU reads $-g$ on its down axis, and an ENU reading has to be
reflected first. `gyrocompassing` predicts its own heading error from the grade's gyro bias
instability and refuses, with `MeasurementUnavailable`, when that exceeds
`GyrocompassConfig::maximum_heading_uncertainty_radians` (45° by default). A consumer or
industrial MEMS gyro cannot gyrocompass, and the function says so instead of returning a number.
It also refuses above 75° of latitude by default. The functions take one averaged sample and do
not check for motion themselves; choosing stationary samples is the
[`StationaryDetector`](../filters/measurements.md#zero-velocity-and-zero-angular-rate-updates)'s
job:

```rust,ignore
{{#include ../../../core/examples/lever_arm.rs:alignment}}
```

`lever_arm.rs` then seeds the engine from that attitude and widens the yaw entry of $P_0$ by the
heading uncertainty the gyrocompass reported (the `engine` snippet above).

## IMU grades

`IMUQuality` (`Consumer`, the default; `Industrial`; `Tactical`; `Navigation`; `Strategic`)
carries representative figures for each grade:

| grade | gyro bias instability (°/h) | gyro ARW (°/√h) | accel bias instability (m/s²) | accel VRW (m/s/√h) |
| --- | --- | --- | --- | --- |
| `Consumer` | 100 | 1.0 | 0.1 | 0.1 |
| `Industrial` | 50 | 0.1 | 0.05 | 0.03 |
| `Tactical` | 1 | 0.01 | 0.001 | 0.01 |
| `Navigation` | 0.01 | 0.005 | 0.0001 | 0.005 |
| `Strategic` | 0.0001 | 0.0005 | 0.00001 | 0.0001 |

The accessors return the gyro figures in radians (`gyro_bias_instability_rad_per_hour`,
`gyro_angle_random_walk`). The grade is used in three places:

- `initial_bias_covariance()`: the six bias variances a filter should open with, each the
  grade's bias instability squared (the gyro converted to rad/s). The `initialize_*` helpers use
  it through their `imu_quality` field.
- `auto_covariance(InitialUncertainty, latitude_deg, altitude_m)`: a whole fifteen-entry $P_0$,
  derived as tabulated on [Kalman Filters](../filters/kalman.md#the-constructors-with-p0-from-the-imu-grade).
  `InitialUncertainty::default()` is 2.5 m horizontal, 5 m vertical and 0.5 m/s.
- `strapdown-sim syn --imu-grade`: the noise and bias levels of a synthetic trajectory.

`velocity_process_noise(dt)` and `attitude_process_noise(dt)` return the variance the grade's
random walks accumulate over `dt` seconds. The filters take a variance **per second**
([process noise](../filters/kalman.md#process-noise-is-a-spectral-density)), which is the value
these return at `dt = 1.0`. `gyro_process_noise` and `accel_process_noise` are deprecated: a bias
instability belongs in $P_0$, not in $Q$.

## Errors

Every fallible function returns `StrapdownError` (`core/src/error.rs`, re-exported at the root).
Library code does not unwrap, expect or panic; the lints that forbid it are set to deny. The
enum is `#[non_exhaustive]`, and one method sorts its variants into the two kinds a caller has to
tell apart:

| `is_recoverable()` | variants | meaning |
| --- | --- | --- |
| `true` | `OutOfMapBounds`, `MeasurementUnavailable`, `ExternalModel` | this one measurement could not be used; the filter state is untouched and valid, so skip it and continue |
| `false` | `DimensionMismatch`, `NotSquare`, `OutOfRange`, `NonFinite`, `UnsupportedInput`, `SingularMatrix`, `InconsistentTimestep`, `InvalidConfiguration`, `MapLoad`, `CovarianceDiagonal`, `FilterDiverged`, `SensorStreamGap`, `Timeout` | bad input or configuration, or a state that can no longer be trusted; stop rather than emit numbers that look like a trajectory |

A gated-out measurement is not an error at all: it is `Ok` with `accepted == false`.
`sim::run_closed_loop` follows the same rule, counting and skipping recoverable failures. In your
own loop:

```rust,ignore
{{#include ../../../core/examples/lever_arm.rs:recoverable}}
```

`StrapdownError` converts into `anyhow::Error`, so `?` composes with the file-I/O functions in
`sim`.

## Scoring a run against truth

`metrics::evaluate(&estimates, &truth, MetricOptions)` scores a `Vec<NavigationResult>` against a
truth series and returns `AccuracyMetrics`: horizontal RMSE, CEP50, CEP95 and maximum; vertical
RMSE and signed bias; horizontal and vertical velocity RMSE; roll, pitch, yaw and geodesic
attitude RMSE; the position NEES and its diagonal-only form; and three-sigma containment. Each is
an `Option`, `None` when the run cannot support it (a dead-reckoning run has no covariance). It
is the same reduction `core/tests/perf_baseline.rs` gates on.

The truth comes from one of two places, and they are not equally good:

- `metrics::truth_from_trajectory(&truth)` takes the exact trajectory `sim::generate_synthetic`
  returns, or that `strapdown-sim syn --no-noise` writes. This is ground truth.
- `metrics::truth_from_records(&records)` takes the GNSS fixes of a recorded log. On any log
  whose fixes also aided the filter, the result measures agreement with the aid, and it cannot
  fall below the receiver's own noise.

Estimates and truth are matched by timestamp; `MetricOptions` sets the largest gap allowed
(`max_match_gap_s`, zero by default) and a number of leading samples to discard
(`warmup_samples`). From `core/examples/eskf.rs`:

```rust,ignore
{{#include ../../../core/examples/eskf.rs:score}}
```

`core/examples/score_run.rs` does the same for two CSV files on disk; see
[Tutorial: Particle Filter](../examples/tutorial-particle-filter.md#comparing-against-truth).

## Below the engine

- **Filters directly.** Construct any filter and drive it through `NavigationFilter`; see
  [Kalman Filters](../filters/kalman.md) and the [RBPF](../filters/rbpf.md).
- **Simulation runners in `sim`.** `TestDataRecord::from_csv` loads a Sensor Logger CSV;
  `dead_reckoning(&records, is_enu)` mechanizes it unaided; `messages::build_event_stream` plus
  `run_closed_loop`, with a filter from `initialize_eskf`, `initialize_ekf` or `initialize_ukf`,
  is the closed loop `strapdown-sim cl` runs; `generate_synthetic` makes a trajectory and its
  truth. [ESKF](../filters/eskf.md#running-it-end-to-end) runs the whole chain.
- **Output.** `NavigationResult::to_csv` is always available. `to_hdf5`, `to_netcdf` and
  `to_mcap` exist behind the `hdf5`, `netcdf` and `mcap` features (`full` enables all three
  plus `clap`); the default build has no features and needs only a Rust toolchain. The CLI
  writes CSV only.

## Output of `lever_arm.rs`

```text
consumer grade: gyrocompassing could not produce a measurement: a Consumer-grade gyro at latitude 39.950 deg gives a predicted one-sigma heading error of 496.9 deg, above the 45.0 deg limit; its bias instability is larger than the Earth rate it would have to measure
aligned from 2852 stationary samples: roll 2.000, pitch -1.000, heading 60.000 deg (claimed 1-sigma 5.0 deg)
with lever arm   : IMU position off by 0.00 m horizontally, 0.00 m vertically
without lever arm: IMU position off by 1.51 m horizontally, 0.77 m vertically
skipped: GnssFix could not produce a measurement: fix contains a non-finite component
set_lever_arm: invalid configuration for `lever_arm`: magnitude 150.000 m exceeds the 100 m limit; the offset is measured in metres in the body frame
```
