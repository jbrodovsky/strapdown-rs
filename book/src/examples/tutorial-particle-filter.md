# Tutorial: Particle Filter

This tutorial runs the [Rao-Blackwellized particle filter](../filters/rbpf.md) end to end with
`strapdown-sim pf`: generate a synthetic drive and its exact truth, run the filter from flags and
then from a configuration file adapted from the `conf/rbpf_*` recipes, score both against truth,
and see why the simulator does not use the paper's horizontal process noise. Every command and
every block of output below was produced by the `strapdown-sim` 1.0.0 binary and the examples in
`core/examples/`; your timestamps will differ.

## A trajectory and its truth

`syn` with the same flags and seed writes the noisy sensor log and, with `--no-noise`, the exact
trajectory it was derived from. This one starts in Philadelphia and drives due north at 15 m/s
for ten minutes with a consumer-grade IMU sampled at 10 Hz (the `syn` defaults). A negative
value has to be attached with `=`, or the parser reads it as a flag.

```bash
strapdown-sim syn -o drive.csv --duration-s 600 --seed 42 \
  --latitude-deg 39.95 --longitude-deg=-75.16 --altitude-m 50 --velocity-north-mps 15
strapdown-sim syn -o drive_truth.csv --duration-s 600 --seed 42 \
  --latitude-deg 39.95 --longitude-deg=-75.16 --altitude-m 50 --velocity-north-mps 15 --no-noise
```

```text
2026-10-01 20:52:19.901 [INFO] - Generated 6000 synthetic records (600.0 s at 10 Hz)
2026-10-01 20:52:19.912 [INFO] - Sensor records written to drive.csv
2026-10-01 20:52:19.993 [INFO] - Generated 6000 synthetic records (600.0 s at 10 Hz)
2026-10-01 20:52:20.000 [INFO] - Truth trajectory written to drive_truth.csv
```

`syn` output is NED, which is the default for every subcommand, so no `--enu` is needed. (The
`conf/` recipes set `is_enu = true` because they read ENU Sensor Logger exports.)

## Run `pf` with its defaults

```bash
strapdown-sim pf -i drive.csv -o pf_out.csv
```

```text
2026-10-01 20:52:20.005 [INFO] - Processing file: drive.csv
2026-10-01 20:52:20.029 [INFO] - Read 6000 records from drive.csv
2026-10-01 20:52:20.030 [INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.775 m/s^2 against a local gravity of 9.801 m/s^2
2026-10-01 20:52:21.164 [INFO] - Results written to pf_out.csv
2026-10-01 20:52:21.164 [INFO] - Particle filter simulation complete
```

The "Mechanizing input as NED" line is the frame check: the mean vertical specific force has the
sign NED predicts, so the declared frame is accepted.

With no flags the filter runs 100 particles, uses a horizontal process noise of `[1, 1]` m/√s and
takes every GNSS fix in the file, here one per record (10 Hz). `pf_out.csv` has the same columns
as a `cl` run (see [Output Format](../user-guide/output-format.md)), one row per IMU sample. The
six IMU-bias columns are zero: the RBPF carries no bias states.

## Comparing against truth

`core/examples/score_run.rs` loads an output CSV and a truth CSV and scores them with
`metrics::evaluate`, the reduction the accuracy baseline is gated on:

```rust,ignore
{{#include ../../../core/examples/score_run.rs:score}}
```

```bash
cargo run -p strapdown-core --example score_run -- pf_out.csv drive_truth.csv
```

```text
aligned samples                            5990
horizontal RMSE                           0.988 m
horizontal CEP50                          0.808 m
horizontal CEP95                          1.751 m
horizontal max                            3.375 m
vertical RMSE                             1.958 m
horizontal velocity RMSE                  0.157 m/s
yaw RMSE                                  0.409 deg
position NEES (consistent at 3)          13.332
3-sigma horizontal containment            1.000
```

Rows are matched by timestamp, which works because `syn` always starts at
2025-01-01T00:00:00Z; the first ten aligned samples (one second) are skipped as initialization.
Read the two consistency lines together, as
[metrics](../user-guide/library.md#scoring-a-run-against-truth) explains: the NEES compares the
error with the full position covariance, and containment counts how often the error lies inside
three sigma.

## A configuration adapted from `conf/rbpf_truth.toml`

The recipes under `conf/` describe the repository's own experiments: ENU phone recordings in
`data/input`, logs written to `log/`, map biases in `[geophysical]`. For a `syn` trajectory, keep
their `[particle_filter]` section and change the rest: the input and output paths, `is_enu =
false`, no `[geophysical]` section, and an `[aiding]` scheduler that thins the 10 Hz synthetic
fixes to 1 Hz, the rate the recordings' receivers actually deliver. `[aiding]` is the section's
name; the recipes' `[gnss_degradation]` is accepted as an alias.

```toml
# rbpf_syn.toml: particle filter on a synthetic NED trajectory, adapted from conf/rbpf_truth.toml.
input = "drive.csv"
output = "pf_config_out.csv"
mode = "particle-filter"
is_enu = false          # `syn` writes NED; the conf/ recipes read ENU Sensor Logger exports
seed = 42

[particle_filter]
num_particles = 1000
position_init_std_m = [10.0, 10.0, 5.0]
velocity_init_std_mps = 1.0
attitude_init_std_rad = 0.1
velocity_process_noise_std_mps = 1e-3
attitude_process_noise_std_rad = 0.01
horizontal_process_noise_std_m = [1.0, 1.0]
baro_loop_time_constant_s = 10.0
baro_error_std_m = 8.3
baro_error_time_constant_s = 3600.0
vertical_accel_error_init_std_mps2 = 0.1
effective_sample_threshold = 1.0
roughening_factor = 0.2

[aiding]
seed = 42

[aiding.scheduler]
kind = "fixed_interval"
interval_s = 1.0
phase_s = 0.0

[aiding.fault]
kind = "none"
```

A configuration file supplies the whole run, so it is passed on its own:

```bash
strapdown-sim --config rbpf_syn.toml
cargo run -p strapdown-core --example score_run -- pf_config_out.csv drive_truth.csv
```

```text
2026-10-01 20:52:39.076 [INFO] - Loading configuration from rbpf_syn.toml
2026-10-01 20:52:39.076 [INFO] - Configuration loaded successfully
2026-10-01 20:52:39.076 [INFO] - Mode: ParticleFilter
...
2026-10-01 20:52:39.104 [INFO] - Running particle filter simulation
2026-10-01 20:52:43.908 [INFO] - Results written to pf_config_out.csv
```

```text
aligned samples                            5990
horizontal RMSE                           2.362 m
horizontal CEP50                          1.994 m
horizontal CEP95                          3.962 m
horizontal max                            6.800 m
vertical RMSE                             2.676 m
horizontal velocity RMSE                  0.570 m/s
yaw RMSE                                  0.416 deg
position NEES (consistent at 3)           7.880
3-sigma horizontal containment            0.999
```

The [RBPF configuration table](../filters/rbpf.md#configuration) explains every key. Three are
worth knowing before changing anything:

- `horizontal_process_noise_std_m`: see the next section.
- `effective_sample_threshold = 1.0` resamples after every update, as Canciani and Raquet do.
- `roughening_factor = 0.2` jitters the cloud after resampling so it cannot collapse onto copies
  of one particle; `0.0` reproduces the paper.

The `[geophysical]` section and the `*_variation_*` keys only matter for map-aided runs, which
need gravity or magnetic anomaly maps; see [Geophysical Navigation](../geonav/overview.md).

## Why not the paper's zero horizontal noise

Canciani and Raquet put no process noise on the sampled horizontal position (their eq. 19). The
library's `RbpfConfig` keeps that default; `strapdown-sim` does not. On the same trajectory with
1 Hz fixes:

```bash
strapdown-sim pf -i drive.csv -o pf_zero.csv --sched fixed --interval-s 1 \
  --horizontal-process-noise-std-m 0 0
```

```text
2026-10-01 20:52:45.263 [INFO] - Processing file: drive.csv
2026-10-01 20:52:45.290 [INFO] - Read 6000 records from drive.csv
2026-10-01 20:52:45.291 [INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.775 m/s^2 against a local gravity of 9.801 m/s^2
Error: OutOfRange { what: "speed", value: 500.1655942836976, min: 0.0, max: 500.0 }
```

The run diverges until the health monitor's speed limit (`--health-speed-mps-max`, 500 m/s by
default) stops it, and no output file is written. With zero noise each epoch's time update is a
noiseless observation of the velocity and tilt errors, which moves their uncertainty out of the
shared covariance and into the particles' spread; resampling on metre-level fixes then destroys
that spread. [RBPF](../filters/rbpf.md#horizontal-process-noise-why-the-default-is-1-ms) gives the
full account.

## The same trajectory through the ESKF

For a reference point, run the default Kalman filter on the same input with the same 1 Hz
schedule and score it the same way:

```bash
strapdown-sim cl -i drive.csv -o cl_out.csv --sched fixed --interval-s 1
cargo run -p strapdown-core --example score_run -- cl_out.csv drive_truth.csv
```

```text
aligned samples                            5990
horizontal RMSE                           2.072 m
horizontal CEP50                          1.738 m
horizontal CEP95                          3.519 m
horizontal max                            5.070 m
vertical RMSE                             2.336 m
horizontal velocity RMSE                  0.450 m/s
yaw RMSE                                  0.335 deg
position NEES (consistent at 3)           5.192
3-sigma horizontal containment            1.000
```

One synthetic straight-line drive is not a ranking of the two filters. The RBPF exists for
map-aided navigation, where the measurement is a nonlinear, often multimodal function of
position; with GNSS fixes alone, as here, it has nothing to exploit that a Kalman filter cannot.
The gated comparison across scenarios, including the RBPF's own row, is on
[Performance Baselines](../development/performance.md), and
[RBPF](../filters/rbpf.md#sparse-gnss-what-this-structure-cannot-recover-from) documents a regime
(one fix a minute) in which it diverges where the EKF does not.

For the unaided baseline, `strapdown-sim dr -i drive.csv -o dr_out.csv` scored the same way
reports no consistency metrics (a dead-reckoning run has no covariance) and a horizontal error
that grows to tens of kilometres over the ten minutes on this consumer-grade IMU.

## Where next

- [RBPF](../filters/rbpf.md): structure, configuration keys and departures from the paper.
- [Particle Filter](../user-guide/particle-filter.md) in the user guide: the `pf` subcommand's
  flags.
- [Schedulers and Faults](../gnss/scenarios.md): outages, reduced rates and spoofing for any of
  the runs above.
