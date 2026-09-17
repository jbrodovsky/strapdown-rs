# Example Configuration Files

This directory contains example scenario configuration files for use with `strapdown-sim`. Each one is a complete `SimulationConfig` document: until the v1.0 freeze most were bare aiding fragments, which `--config` could not accept at all -- it parses a `SimulationConfig` and rejected them with ``missing field `mode` ``. These scenarios demonstrate the various GNSS denial and degradation modes supported by the simulation.

## Configuration File Format

Each file is a `SimulationConfig`. `mode` is the only field without a default, and the aiding
settings live under an **`aiding`** block:

```yaml
mode: closed-loop
aiding:
  scheduler: { kind: pass_through }
  fault: { kind: none }
  seed: 42
```

Inside `aiding`:
- **scheduler**: Controls when GNSS measurements are available
- **fault**: Controls how GNSS measurements are corrupted
- **baro_scheduler**, **magnetometer_scheduler**: the same scheduling for the other two aiding
  channels, each on its own clock. Both default to `fixed_interval` at 1 Hz when omitted, so
  none of the files here sets them; neither channel is ever corrupted.
- **seed**: Random seed for reproducibility

`aiding` was called `gnss_degradation` before the v1.0 freeze -- it now carries the barometer
and magnetometer too, so the name no longer fitted. The old spelling still parses
(`#[serde(alias)]`), so existing config files keep working.

None of these files sets `is_enu`, so each one declares the **NED** default. Sensor Logger
exports are ENU, and `--enu` is ignored alongside `--config` like every other subcommand
argument, so add `is_enu: true` to the file when running one of these against Sensor Logger
data. The frame is checked rather than guessed: declaring the wrong one is rejected up front
with an `InvalidConfiguration` naming what to set, not silently mechanized at 2 g.

`input` and `output` may be omitted, in which case they are `input.csv` and `output.csv` in the
working directory. **`--config` supplies the entire run**, so the subcommand and its `-i`/`-o`
and `--seed` arguments are ignored when it is present -- the CLI says as much in
`--help` ("ignored if --config is provided"). None of the scenario files here sets `input` or
`output`, so point a scenario at your own data by adding those two keys to the file (or copying
it), not by passing paths on the command line.

See the [User Guide](../../docs/USER_GUIDE.md) for detailed documentation.

---

## Available Scenarios

### Baseline (No Degradation)

**File:** `baseline.yaml`

No GNSS degradation - all measurements pass through unchanged. Use this as a reference for comparing degraded scenarios.

```bash
strapdown-sim --config baseline.yaml
```

---

### Degraded Accuracy Scenarios

#### Full Rate with Degraded Accuracy

**File:** `degraded_fullrate.yaml`

GNSS measurements at full rate but with AR(1)-correlated position and velocity errors. Models low-SNR or multipath conditions.

#### Reduced Rate with Degraded Accuracy

**File:** `degraded_5s.yaml`

GNSS measurements every 5 seconds with degraded accuracy. Combines scheduling and fault injection.

---

### Scheduling Scenarios

#### Fixed Interval

**File:** `sched_10s.yaml`

GNSS updates every 10 seconds with no measurement corruption. Models reduced update rate.

#### Duty Cycle

**File:** `duty_10on_2off.yaml`

Alternates between 10 seconds of GNSS availability and 2 seconds of denial. Models periodic outages.

---

### Spoofing Scenarios

#### Slow Bias (Soft Spoofing)

**File:** `slowbias.yaml`

Slowly drifting position bias that appears plausible to the filter. Simulates gradual trajectory manipulation.

**File:** `slowbias_rot.yaml`

Same as above but with rotating drift direction.

#### Hijack (Hard Spoofing)

**File:** `hijack.yaml`

Abrupt position offset during a fixed time window. Simulates hard spoofing attack.

---

### Combined Scenarios

#### Combo: Reduced Rate + Degraded

**File:** `combo.yaml`

Combines reduced update rate (5s) with degraded measurement accuracy.

#### Combo: Duty Cycle + Hijack

**File:** `combo_duty_hijack.yaml`

Combines duty-cycled availability with hard spoofing.

---

## Usage Examples

### Command Line

```bash
# Using a config file. `input` and `output` come from the file; a subcommand and its
# arguments would be ignored here, so none is given.
strapdown-sim --config examples/configs/degraded_5s.yaml
```

To run a scenario against your own data, or with a different seed, set it in the file --
`--seed` on the command line is ignored alongside `--config` just as `-i`/`-o` are:

```yaml
input: data/input.csv
output: results/degraded_5s.csv
aiding:
  seed: 123
```

### Batch Processing

Because `--config` ignores `-i`/`-o`, a batch has to vary the paths *in the file*. Write a
copy per scenario and run that:

```bash
#!/bin/bash
# Run all scenarios, each against data/input.csv and into its own output file.
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

Passing `-o "results/${name}.csv"` instead would be silently ignored and every scenario would
write to the same `output.csv`.

---

## Alternative Formats

The same configurations are also available in JSON and TOML formats:

- `gnss_degradation.json` - JSON format example
- `gnss_degradation.toml` - TOML format example
- `json/` subdirectory - Additional JSON examples

---

## Creating Custom Scenarios

To create a custom scenario:

1. Copy an existing configuration as a starting point
2. Modify the scheduler and/or fault sections
3. Save with a `.yaml`, `.json`, or `.toml` extension

### Scheduler Options

These all sit under the `aiding:` block shown above.

```yaml
# All measurements pass through
scheduler:
  kind: pass_through

# Fixed interval updates
scheduler:
  kind: fixed_interval
  interval_s: 10.0
  phase_s: 0.0

# Duty cycle (ON/OFF periods)
scheduler:
  kind: duty_cycle
  on_s: 30.0
  off_s: 60.0
  start_phase_s: 0.0   # note: not `phase_s`, which belongs to fixed_interval
```

The same three kinds configure `baro_scheduler` and `magnetometer_scheduler`. Both default to
one measurement per second rather than to `pass_through`: before #375 they were emitted once
per record, which tied their rate to the log's rather than the sensor's — 1 Hz on a Sensor
Logger export, 50 Hz on a synthetic trajectory. Set `kind: pass_through` explicitly to get one
per record back.

```yaml
baro_scheduler:
  kind: duty_cycle       # a barometer outage, independent of the GNSS one
  on_s: 60.0
  off_s: 60.0
  start_phase_s: 0.0
```

> **Config files and CLI flags use different names for the same thing.** In a config file the
> scheduler kinds are `pass_through`, `fixed_interval` and `duty_cycle`, matching the
> `MeasurementScheduler` variants. The equivalent CLI flags are `--sched passthrough|fixed|duty`, and
> the duty phase is `--duty-phase-s` rather than `--phase-s`. A misspelled `kind` does not
> error: every field has a default, so the section is silently dropped and the run proceeds
> with no degradation at all. `core/tests/example_configs.rs` guards the files in this
> directory against exactly that.

### Fault Model Options

These all sit under the `aiding:` block shown above.

```yaml
# No corruption
fault:
  kind: none

# AR(1) correlated noise
fault:
  kind: degraded
  rho_pos: 0.99
  sigma_pos_m: 3.0
  rho_vel: 0.95
  sigma_vel_mps: 0.3
  r_scale: 5.0

# Slow drifting bias
fault:
  kind: slow_bias
  drift_n_mps: 0.02
  drift_e_mps: 0.0
  q_bias: 1e-6
  rotate_omega_rps: 0.0

# Hard spoofing window
fault:
  kind: hijack
  offset_n_m: 50.0
  offset_e_m: 0.0
  start_s: 120.0
  duration_s: 60.0
```

---

## Reproducibility Notes

- Always specify the `seed` value for reproducible results
- The same seed with the same configuration will produce identical results
- Use different seeds to generate multiple realizations for Monte Carlo analysis
