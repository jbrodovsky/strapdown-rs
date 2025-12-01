# Example Configuration Files

This directory contains example GNSS degradation configuration files for use with `strapdown-sim`. These scenarios demonstrate the various GNSS denial and degradation modes supported by the simulation.

## Configuration File Format

All configurations use YAML format and contain three main sections:
- **scheduler**: Controls when GNSS measurements are available
- **fault**: Controls how GNSS measurements are corrupted
- **seed**: Random seed for reproducibility

See the [User Guide](../../docs/USER_GUIDE.md) for detailed documentation.

---

## Available Scenarios

### Baseline (No Degradation)

**File:** `baseline.yaml`

No GNSS degradation - all measurements pass through unchanged. Use this as a reference for comparing degraded scenarios.

```bash
strapdown-sim -i input.csv -o baseline.csv closed-loop --config baseline.yaml
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
# Using a config file
strapdown-sim -i data/input.csv -o results/output.csv closed-loop \
  --config examples/configs/degraded_5s.yaml

# Override seed for different realization
strapdown-sim -i data/input.csv -o results/output.csv closed-loop \
  --config examples/configs/degraded_5s.yaml \
  --seed 123
```

### Batch Processing

```bash
#!/bin/bash
# Run all scenarios
for config in examples/configs/*.yaml; do
  name=$(basename "$config" .yaml)
  strapdown-sim -i data/input.csv -o "results/${name}.csv" closed-loop \
    --config "$config"
done
```

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

```yaml
# All measurements pass through
scheduler:
  kind: passthrough

# Fixed interval updates
scheduler:
  kind: fixed_interval
  interval_s: 10.0
  phase_s: 0.0

# Duty cycle (ON/OFF periods)
scheduler:
  kind: duty
  on_s: 30.0
  off_s: 60.0
  phase_s: 0.0
```

### Fault Model Options

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
  kind: slowbias
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
