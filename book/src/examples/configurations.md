# Example Configurations

Ready-to-run scenario files live in
[`examples/configs/`](https://github.com/jbrodovsky/strapdown-rs/tree/main/examples/configs).
Each is a complete `SimulationConfig`; for the schema itself see
[Configuration Files](../user-guide/configuration.md), and for every scheduler and fault field
see [Schedulers and Faults Reference](../gnss/scenarios.md).

Three things apply to all of them:

- **`--config` supplies the whole run.** The subcommand's `-i`, `-o` and `--seed` are ignored
  alongside it, so set `input` and `output` inside the file. Most of these files omit both, and
  then read `input.csv` and write `output.csv` in the working directory.
- **They declare the NED frame** by not setting `is_enu`. That is right for `strapdown-sim syn`
  output. Sensor Logger exports are ENU, so add `is_enu: true` (or `is_enu = true`) to run one
  against a phone recording; a wrong declaration is rejected before the run starts.
- **Duty cycles start with their OFF window** unless `start_phase_s` gives an initial ON
  window (see [`DutyCycle`](../gnss/scenarios.md#dutycycle)). Several files below set
  `start_phase_s: 0.0`, so their runs begin in an outage.

Every file in the tables below was run against a 600 s trajectory from
`strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42`, with `input` and `output`
added, and completed. `geonav_example.toml` needs a binary built with `--features geonav` and
map files beside the input.

## Scenario files

All of these are `mode: closed-loop` with the default filter (the ESKF), unless noted.

### Baseline

| File | Scheduler | Fault | What it models |
|---|---|---|---|
| `baseline.yaml` | `pass_through` | `none` | no degradation; the reference to compare the others against |

### Availability

| File | Scheduler | Fault | What it models |
|---|---|---|---|
| `sched_10s.yaml` | `fixed_interval`, 10 s | `none` | a reduced update rate |
| `duty_10on_2off.yaml` | `duty_cycle`, 10 s on / 2 s off, initial 1 s on | `none` | brief periodic outages |
| `simple_dropout.yaml` | `duty_cycle`, 120 s on / 30 s off, no initial on | `none` | bridges and tall buildings; starts with a 30 s outage |
| `extended_gnss_denied.yaml` | `duty_cycle`, 60 s on / 300 s off, no initial on | `none` | tunnels and sustained jamming; starts with a 300 s outage |

`extended_gnss_denied.yaml` shows dead-reckoning quality most clearly: five minutes is long
enough for consumer-MEMS drift to dominate. Because it sets no initial ON window, the first
five minutes are an outage before the filter has seen any fix; set `start_phase_s` if you want
the filter to converge first. See [Tutorial: GNSS Degradation](./tutorial-gps-degradation.md).

### Accuracy

| File | Scheduler | Fault | What it models |
|---|---|---|---|
| `degraded_fullrate.yaml` | `pass_through` | `degraded` (ρ 0.99, σ 3 m; ρ 0.95, σ 0.3 m/s; `r_scale` 5) | AR(1)-correlated error at full rate |
| `degraded_5s.yaml` | `fixed_interval`, 5 s | `degraded`, same values | reduced rate and corruption together |
| `gnss_degradation.yaml` | `fixed_interval`, 10 s | `degraded`, same values | the same idea at 10 s |
| `gnss_degradation.toml` | as above | as above | the same scenario in TOML |
| `gnss_degradation.json` | as above | as above | the same scenario in JSON |
| `combo.yaml` | `fixed_interval`, 5 s | `degraded` (ρ 0.995, σ 4 m; ρ 0.97, σ 0.35 m/s; `r_scale` 5) | reduced rate plus degraded accuracy |
| `particle_filter_comparison.yaml` | `fixed_interval`, 5 s | `degraded` (σ 5 m, 0.5 m/s) | the closed-loop half of a filter comparison; see below |

None of these sets `tau_pos_s` or `tau_vel_s`, so their `sigma_*` values are per-fix
innovations and their correlation times depend on the fix interval; see
[Fault Simulation](../gnss/fault-simulation.md#correlation-time-and-the-fix-interval).

`combo.yaml` combines a scheduler with a fault. It is **not** the `combo` fault kind, which
chains several fault models on the same fix; no shipped example uses that kind, and
[Schedulers and Faults Reference](../gnss/scenarios.md#combo) shows its syntax.

`particle_filter_comparison.yaml` is, despite its name, a closed-loop file. Its comments
describe the comparison: copy it, change the copy's `mode` to `particle-filter` and both
files' `output`, and run each with `--config`.

### Spoofing

| File | Scheduler | Fault | What it models |
|---|---|---|---|
| `slowbias.yaml` | `pass_through` | `slow_bias`, 0.02 m/s north, `q_bias` 1e-6 | soft spoofing: each fix looks plausible |
| `slowbias_rot.yaml` | `pass_through` | `slow_bias`, as above with `q_bias` 5e-6 and the direction rotating at 1e-3 rad/s | the same, drifting in a turning direction |
| `hijack.yaml` | `pass_through` | `hijack`, 50 m north from 120 s for 60 s | hard spoofing: an abrupt offset |
| `combo_duty_hijack.yaml` | `duty_cycle`, 10 s on / 3 s off | `hijack`, 10 m north and 10 m east from 150 s for 120 s | duty-cycled availability plus hard spoofing |

Spoofing scenarios are worth pairing with an innovation gate (`--gate-confidence` on the
command line). A `hijack` produces a large normalized innovation squared the moment it starts,
which a chi-squared gate can reject, while a `slow_bias` is designed to stay under that
threshold.

### Filter, I/O and geophysical settings

| File | What it shows |
|---|---|
| `closed-loop-ukf.toml` | selecting the UKF in `[closed_loop]`, with `[logging]` and an undegraded `[aiding]` section |
| `parallel_example.toml` | a directory as `input` and `output`, `parallel = true` to process its files concurrently, and a log file |
| `geonav_example.toml` | a `[geophysical]` section with gravity and magnetic aiding on the UKF; see [Geophysical Navigation](../geonav/overview.md) |

## Other files

- `README.md` in the directory repeats this catalogue with each scheduler and fault option
  written out.
- `json/` holds ten `{"name", "args"}` files (`baseline`, `combo`, `combo_duty_hijack`,
  `degraded_5s`, `degraded_fullrate`, `duty_10on_2off`, `hijack`, `sched_10s`, `slowbias`,
  `slowbias_rot`). They are **not** configuration files: each is a list of command-line
  arguments for the same scenario, in the command line's vocabulary (`--sched fixed`,
  `--fault slowbias`) rather than the configuration file's (`fixed_interval`, `slow_bias`).
  Nothing in the repository reads them, and `--config` would reject them.

## Running the set

Because `--config` ignores `-i`/`-o`, a batch varies the paths *in the file*. This writes a
copy of each YAML scenario pointed at one input and a separate output, then runs it:

```bash
mkdir -p results .scenarios
for config in examples/configs/*.yaml; do
  name=$(basename "$config" .yaml)
  { cat "$config"
    echo "input: data/input.csv"
    echo "output: results/${name}.csv"
  } > ".scenarios/${name}.yaml"
  strapdown-sim --config ".scenarios/${name}.yaml"
done
```

Passing `-o "results/${name}.csv"` instead would be silently ignored, and every scenario would
write to the same `output.csv`.

## The experiment recipes in `conf/`

`conf/` holds the fifteen configurations the project's own experiments run through the
`justfile`: `{ukf,ekf,rbpf}_{truth,degraded,grav,mag,both}.toml`, plus the same fifteen under
`conf/real/` for the frozen real-sensor arm. They read a directory of preprocessed recordings
(`data/input`, which is not distributed), declare `is_enu = true`, and carry long comments
explaining every value. `*_truth` is undegraded, `*_degraded` is the single interference
profile described in [Fault Simulation](../gnss/fault-simulation.md), and
`*_{grav,mag,both}` add geophysical aiding to that profile. They are worth reading as worked
examples even without the data.
