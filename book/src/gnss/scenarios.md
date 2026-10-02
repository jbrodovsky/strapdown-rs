# Schedulers and Faults Reference

This page lists every scheduler and fault model in `strapdown::messages`, with each field, its
unit and default, and how to set it from a configuration file and from the command line. For
why the shipped recipes use the values they do, see [Fault Simulation](./fault-simulation.md).

The two are orthogonal. A **scheduler** (`MeasurementScheduler`) decides *when* a measurement
reaches the filter. A **fault model** (`GnssFaultModel`) decides *what* is wrong with each GNSS
fix that does. Both are applied once, when `build_event_stream` turns the input records into
the time-ordered stream of IMU and measurement events, so every filter sees exactly the same
degraded stream.

## Where they are configured

Everything on this page sits in one struct, `AidingConfig`, which is the `[aiding]` section of
a configuration file. The section was called `[gnss_degradation]` before the v1.0 freeze and
that spelling is still accepted as an alias; the recipes under `conf/` use it.

| key | type | default | meaning |
|---|---|---|---|
| `scheduler` | scheduler | `pass_through` | when GNSS fixes are delivered |
| `fault` | fault model | `none` | how GNSS fixes are corrupted |
| `baro_scheduler` | scheduler | `fixed_interval`, 1 s | when barometric altitude is delivered |
| `magnetometer_scheduler` | scheduler | `fixed_interval`, 1 s | when magnetometer heading is delivered |
| `baro_noise_std_m` | m (1σ) | 2.236 (√5) | barometric altitude noise standard deviation |
| `baro_bias_index` | integer or unset | unset | which filter state holds the barometric bias (library use; see below) |
| `seed` | integer | 42 | seed for the fault model's random draws |
| `max_imu_gap_s` | s | 5.0 | longest inertial gap tolerated before the run is refused; `<= 0` disables |

A complete example that runs as written (`input`/`output` pointed at a synthetic trajectory):

```toml
mode = "closed-loop"
input = "synthetic.csv"
output = "results/baro.csv"

[aiding]
seed = 7
baro_noise_std_m = 3.0
max_imu_gap_s = 5.0

[aiding.scheduler]          # GNSS: 100 s available, 50 s denied
kind = "duty_cycle"
on_s = 100.0
off_s = 50.0
start_phase_s = 0.0

[aiding.baro_scheduler]     # an independent barometer outage
kind = "duty_cycle"
on_s = 60.0
off_s = 60.0
start_phase_s = 0.0

[aiding.magnetometer_scheduler]
kind = "pass_through"
```

Three behaviours of the parser are worth knowing:

- **A misspelled `kind` is an error.** `kind = "fixed"` fails with
  ``unknown variant `fixed`, expected one of `pass_through`, `fixed_interval`, `duty_cycle` ``.
- **A missing field is an error.** Scheduler and fault fields have no defaults, except the two
  optional time constants on `degraded`; leave one out and the run stops with
  ``missing field `...` ``.
- **A misspelled section name is not.** Unknown top-level keys are ignored, so
  `[gnss_degredation]` parses, is discarded, and the run proceeds with no degradation at all.
  `core/tests/example_configs.rs` guards the shipped files against this; your own files are
  not guarded, so check section names against this page.

**Seeds.** In a configuration file the fault model is seeded by `[aiding] seed` when it is
given, and otherwise by the top-level `seed`, so two files differing only in the top-level
`seed` produce two fault realizations. On the command line, `--seed` seeds the fault model.

**`baro_bias_index`** tells the barometer measurement which state of the filter holds a
barometric bias. `strapdown-sim` overwrites it from the filter it builds (and clears it for the
particle filter, which carries no barometric bias state), so setting it in a `strapdown-sim`
config file has no effect. It matters when you call `build_event_stream` from Rust with your
own filter.

## Schedulers

The configuration file names the variant with `kind` in snake case. The command line uses
shorter names: `--sched passthrough|fixed|duty`. The GNSS scheduler is the only one with flags;
the barometer and magnetometer schedulers are set from a configuration file or from Rust, and
otherwise take their 1 Hz default.

### `PassThrough`

`kind = "pass_through"`, `--sched passthrough` (the CLI default). Every record that carries a
complete fix delivers it. No fields.

### `FixedInterval`

`kind = "fixed_interval"`, `--sched fixed`.

| field | unit | CLI flag | CLI default | meaning |
|---|---|---|---|---|
| `interval_s` | s | `--interval-s` | 1.0 | interval between delivered fixes |
| `phase_s` | s | `--phase-s` | 0.0 | time of the first tick |

Ticks fall on exact multiples of `interval_s` after `phase_s`, so they do not creep over a long
run. It is a **rate limit, not a resampler**: it delivers the first record at or after each
tick, so a log sampled more slowly than `interval_s` delivers every record. After a gap in the
log the clock jumps past the gap rather than bursting to catch up. A non-positive or non-finite
`interval_s` behaves as `PassThrough` rather than silently withholding every fix.

### `DutyCycle`

`kind = "duty_cycle"`, `--sched duty`.

| field | unit | CLI flag | CLI default | meaning |
|---|---|---|---|---|
| `on_s` | s | `--on-s` | 10.0 | length of each ON (available) window |
| `off_s` | s | `--off-s` | 10.0 | length of each OFF (denied) window |
| `start_phase_s` | s | `--duty-phase-s` | 0.0 | length of an initial ON window |

The timeline is `start_phase_s` of ON, then `off_s` OFF and `on_s` ON, repeating. **The cycle
starts with its OFF window**, so with `start_phase_s = 0` the run begins in an outage:

```text
start_phase_s = 0:    OFF [0, off_s)  ON [off_s, off_s+on_s)  OFF ...
start_phase_s = p:    ON [0, p)  OFF [p, p+off_s)  ON [p+off_s, p+off_s+on_s)  OFF ...
```

Set `start_phase_s` (on the command line, `--duty-phase-s`; note it is not `--phase-s`, which
belongs to `fixed`) to let the filter converge on some fixes before the first outage. Every
record inside an ON window delivers its fix. A cycle whose `on_s + off_s` is not positive and
finite behaves as `PassThrough`.

## Fault models

The configuration file names the variant with `kind` in snake case. The command line spells it
`--fault none|degraded|slowbias|hijack`. Every field is required in a configuration file except
`tau_pos_s` and `tau_vel_s`; the CLI defaults below apply only on the command line.

Faults apply to GNSS only. Barometer and magnetometer measurements are scheduled but never
corrupted. Each fault advances once per *delivered* fix, with `dt` the time since the previous
delivered fix.

### `None`

`kind = "none"`, `--fault none` (the CLI default). Fixes reach the filter unchanged.

### `Degraded`

`kind = "degraded"`, `--fault degraded`. AR(1)-correlated error added to position (north, east
and up) and to north/east velocity, and the advertised accuracies inflated.

| field | unit | CLI flag | CLI default | meaning |
|---|---|---|---|---|
| `rho_pos` | -- | `--rho-pos` | 0.99 | AR(1) coefficient for position error, per fix |
| `sigma_pos_m` | m | `--sigma-pos-m` | 3.0 | position innovation σ per fix; steady-state σ if `tau_pos_s` is set |
| `rho_vel` | -- | `--rho-vel` | 0.95 | AR(1) coefficient for velocity error, per fix |
| `sigma_vel_mps` | m/s | `--sigma-vel-mps` | 0.3 | velocity innovation σ per fix; steady-state σ if `tau_vel_s` is set |
| `r_scale` | -- | `--r-scale` | 5.0 | multiplies the advertised horizontal and velocity 1σ |
| `tau_pos_s` | s | `--tau-pos-s` | unset | position correlation time (optional) |
| `tau_vel_s` | s | `--tau-vel-s` | unset | velocity correlation time (optional) |

Each error state $x$ follows

$$x_k = \rho\  x_{k-1} + w_k, \qquad w_k \sim \mathcal{N}(0, s^2).$$

Without a time constant, $\rho$ is `rho_pos` and $s$ is `sigma_pos_m`, applied per delivered
fix, so the correlation time depends on the schedule and the error settles at
$s/\sqrt{1-\rho^2}$, not at $s$. With `tau_pos_s` set, $\rho = e^{-\Delta t/\tau}$ for the
actual interval $\Delta t$ between fixes and $s = \sigma\sqrt{1-\rho^2}$, which holds the
steady-state standard deviation at `sigma_pos_m` whatever the schedule. The time constants were
added in commit `96a47f4` for exactly this reason; [Fault
Simulation](./fault-simulation.md#correlation-time-and-the-fix-interval) has the argument. A
non-positive or non-finite `tau` falls back to the per-fix form.

`r_scale` scales the advertised **standard deviations**, which the GNSS measurement model
squares to build $R$, so $R$ grows by `r_scale²`. The advertised vertical accuracy is not
scaled, although the altitude does receive the position error.

### `SlowBias`

`kind = "slow_bias"`, `--fault slowbias`. A soft spoof: a north/east position offset that
drifts at a set rate, with a matching velocity offset so that position and velocity stay
mutually consistent.

| field | unit | CLI flag | CLI default | meaning |
|---|---|---|---|---|
| `drift_n_mps` | m/s | `--drift-n-mps` | 0.02 | northward drift rate of the offset |
| `drift_e_mps` | m/s | `--drift-e-mps` | 0.0 | eastward drift rate of the offset |
| `q_bias` | m²/s | `--q-bias` | 1e-6 | random-walk intensity on the offset; 0 disables |
| `rotate_omega_rps` | rad/s | `--rotate-omega-rps` | 0.0 | rotation rate of the drift direction; 0 keeps it fixed |

At each fix the offset grows by the drift rate times $\Delta t$, plus a zero-mean draw with
variance `q_bias`$\ \Delta t$ per axis. With `rotate_omega_rps` non-zero, the drift vector is
rotated by $\omega t$ before it is applied. The reported velocity is offset by the current drift
rate. The advertised accuracies are unchanged.

### `Hijack`

`kind = "hijack"`, `--fault hijack`. A hard spoof: a constant north/east offset over a fixed
window, with velocity left alone.

| field | unit | CLI flag | CLI default | meaning |
|---|---|---|---|---|
| `offset_n_m` | m | `--hijack-offset-n-m` | 50.0 | northward offset |
| `offset_e_m` | m | `--hijack-offset-e-m` | 0.0 | eastward offset |
| `start_s` | s | `--hijack-start-s` | 120.0 | start of the window, from the first record |
| `duration_s` | s | `--hijack-duration-s` | 60.0 | length of the window |

The offset applies to fixes with elapsed time in $[t_\text{start}, t_\text{start} +
t_\text{duration}]$, endpoints included, and to none outside it.

### `Combo`

`kind = "combo"`. Several fault models applied in sequence to the same fix, the output of each
feeding the next. It is available from a configuration file and from Rust; `--fault` has no
`combo` choice. The members go in a `faults` array, each a complete fault model of its own:

```toml
[aiding.fault]
kind = "combo"

[[aiding.fault.faults]]
kind = "slow_bias"
drift_n_mps = 0.02
drift_e_mps = 0.0
q_bias = 1e-6
rotate_omega_rps = 0.0

[[aiding.fault.faults]]
kind = "hijack"
offset_n_m = 50.0
offset_e_m = 0.0
start_s = 120.0
duration_s = 60.0
```

The members share one fault state. Combine *different* kinds: two `degraded` members step the
same AR(1) states twice per fix rather than adding two independent errors.

## Example commands

Each of these was run against a 600 s trajectory from `strapdown-sim syn -o synthetic.csv
--duration-s 600 --seed 42`:

```bash
# One fix a minute, otherwise clean
strapdown-sim cl -i synthetic.csv -o fixed.csv --sched fixed --interval-s 60 --fault none

# 100 s on / 50 s off, with time-constant AR(1) degradation on the fixes that remain
strapdown-sim cl -i synthetic.csv -o duty.csv --seed 42 \
  --sched duty --on-s 100 --off-s 50 \
  --fault degraded --sigma-pos-m 35 --tau-pos-s 500 --sigma-vel-mps 1.5 --tau-vel-s 100

# Soft spoof, drift direction rotating at 1 mrad/s
strapdown-sim cl -i synthetic.csv -o slowbias.csv \
  --fault slowbias --drift-n-mps 0.05 --rotate-omega-rps 0.001

# Hard spoof: 50 m north between t = 120 s and t = 180 s
strapdown-sim cl -i synthetic.csv -o hijack.csv \
  --fault hijack --hijack-offset-n-m 50 --hijack-start-s 120 --hijack-duration-s 60

# The particle filter takes the same flags
strapdown-sim pf -i synthetic.csv -o pf.csv --sched fixed --interval-s 10 --num-particles 200
```

## Shipped recipes

- `examples/configs/*.yaml` covers each scheduler and fault on its own and in pairs; see
  [Example Configurations](../examples/configurations.md).
- `conf/{ukf,ekf,rbpf}_degraded.toml` is the experiment matrix's single interference profile:
  `fixed_interval` at 60 s with `degraded` at 35 m and 1.5 m/s steady state, `tau_pos_s = 500`,
  `tau_vel_s = 100`, `r_scale = 5`. The `conf/*_truth.toml` recipes are the unscheduled,
  unfaulted counterparts, and `conf/*_{grav,mag,both}.toml` repeat the degraded profile with
  geophysical aiding added. `conf/real/` holds the same fifteen recipes for the frozen
  real-sensor arm of the experiment.
- No recipe ships for total denial. `--sched duty --on-s 30 --off-s 120 --duty-phase-s 30
  --fault none` gives 30 s of fixes in every 150 s with availability as the only variable.
