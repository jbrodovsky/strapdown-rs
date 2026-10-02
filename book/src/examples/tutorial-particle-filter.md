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
2026-10-01 21:44:03.088 [INFO] - Generated 6000 synthetic records (600.0 s at 10 Hz)
2026-10-01 21:44:03.097 [INFO] - Sensor records written to drive.csv
2026-10-01 21:44:03.155 [INFO] - Generated 6000 synthetic records (600.0 s at 10 Hz)
2026-10-01 21:44:03.160 [INFO] - Truth trajectory written to drive_truth.csv
```

`syn` output is NED, which is the default for every subcommand, so no `--enu` is needed. (The
`conf/` recipes set `is_enu = true` because they read ENU Sensor Logger exports.)

## Run `pf` with its defaults

```bash
strapdown-sim pf -i drive.csv -o pf_out.csv
```

```text
2026-10-01 21:44:03.164 [INFO] - Processing file: drive.csv
2026-10-01 21:44:03.185 [INFO] - Read 6000 records from drive.csv
2026-10-01 21:44:03.186 [INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.775 m/s^2 against a local gravity of 9.801 m/s^2
2026-10-01 21:44:04.269 [INFO] - Results written to pf_out.csv
2026-10-01 21:44:04.269 [INFO] - Particle filter simulation complete
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
horizontal RMSE                           0.920 m
horizontal CEP50                          0.758 m
horizontal CEP95                          1.619 m
horizontal max                            2.706 m
vertical RMSE                             0.888 m
horizontal velocity RMSE                  0.193 m/s
yaw RMSE                                  0.404 deg
position NEES (consistent at 3)           3.534
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
2026-10-01 21:44:12.099 [INFO] - Loading configuration from rbpf_syn.toml
2026-10-01 21:44:12.099 [INFO] - Configuration loaded successfully
2026-10-01 21:44:12.099 [INFO] - Mode: ParticleFilter
...
2026-10-01 21:44:12.117 [INFO] - Running particle filter simulation
2026-10-01 21:44:16.836 [INFO] - Results written to pf_config_out.csv
```

```text
aligned samples                            5990
horizontal RMSE                           1.845 m
horizontal CEP50                          1.613 m
horizontal CEP95                          3.007 m
horizontal max                            4.470 m
vertical RMSE                             1.415 m
horizontal velocity RMSE                  0.479 m/s
yaw RMSE                                  0.418 deg
position NEES (consistent at 3)           3.056
3-sigma horizontal containment            1.000
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
cargo run -p strapdown-core --example score_run -- pf_zero.csv drive_truth.csv
```

```text
aligned samples                            5990
horizontal RMSE                           2.025 m
horizontal CEP50                          1.624 m
horizontal CEP95                          3.550 m
horizontal max                            6.068 m
vertical RMSE                             1.409 m
horizontal velocity RMSE                  0.509 m/s
yaw RMSE                                  0.603 deg
position NEES (consistent at 3)           9.786
3-sigma horizontal containment            0.929
```

The same command with the default `1 1` (drop the last flag) scores a horizontal RMSE of
1.870 m, a NEES of 3.141 and a containment of 1.000. On this short, clean synthetic drive the
zero-noise run finishes and its error is only slightly larger, but its covariance is no longer
honest: a NEES three times the ideal and 93% three-sigma containment mean the filter claims more
certainty than it has. With zero noise each epoch's time update is a noiseless observation of
the velocity and tilt errors, which moves their uncertainty out of the shared covariance and
into the particles' spread, and resampling on metre-level fixes then destroys that spread. On
the reference recording that overconfidence grows until the solution runs kilometres off;
[RBPF](../filters/rbpf.md#horizontal-process-noise-why-the-default-is-1-ms) gives the full
account.

## The same trajectory through the ESKF

For a reference point, run the default Kalman filter on the same input with the same 1 Hz
schedule and score it the same way:

```bash
strapdown-sim cl -i drive.csv -o cl_out.csv --sched fixed --interval-s 1
cargo run -p strapdown-core --example score_run -- cl_out.csv drive_truth.csv
```

```text
aligned samples                            5990
horizontal RMSE                           1.574 m
horizontal CEP50                          1.380 m
horizontal CEP95                          2.653 m
horizontal max                            3.994 m
vertical RMSE                             1.183 m
horizontal velocity RMSE                  0.371 m/s
yaw RMSE                                  0.498 deg
position NEES (consistent at 3)           2.879
3-sigma horizontal containment            0.999
```

One synthetic straight-line drive is not a ranking of the two filters. The RBPF exists for
map-aided navigation, where the measurement is a nonlinear, often multimodal function of
position; with GNSS fixes alone, as here, it has nothing to exploit that a Kalman filter cannot.
The gated comparison across scenarios, including the RBPF's own row, is on
[Performance Baselines](../development/performance.md), and
[RBPF](../filters/rbpf.md#sparse-gnss-what-this-structure-cannot-recover-from) documents a regime
(one fix a minute) in which it diverges where the EKF does not.

For the unaided baseline, run
`strapdown-sim dr -i drive.csv -o dr_out.csv`. Scored the same way, it reports no consistency metrics (a
dead-reckoning run has no covariance) and a horizontal error that reaches 108.8 km by the end, a
horizontal RMSE of 41.3 km, on this consumer-grade IMU.

## Where next

- [RBPF](../filters/rbpf.md): structure, configuration keys and departures from the paper.
- [Particle Filter](../user-guide/particle-filter.md) in the user guide: the `pf` subcommand's
  flags.
- [Schedulers and Faults](../gnss/scenarios.md): outages, reduced rates and spoofing for any of
  the runs above.
