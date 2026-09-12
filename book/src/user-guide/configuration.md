# Configuration Files

A scenario configuration describes how GNSS behaves during a simulation: **when** fixes are
delivered and **how** they are corrupted. It is a single document with three sections, and it
can be written as YAML, JSON or TOML — the format is chosen by the file extension.

```yaml
scheduler:        # when fixes arrive
  kind: duty_cycle
  on_s: 120.0
  off_s: 30.0
  start_phase_s: 0.0
fault:            # how they are corrupted
  kind: none
seed: 42          # for reproducibility
```

```bash
strapdown-sim -i input.csv -o output.csv closed-loop --config scenario.yaml
```

## Names must be exact

Every field has a default, so a misspelled `kind` **does not error**. The section is silently
dropped and the run proceeds with no degradation at all — a scenario that looks like it ran
and simulated nothing. This is the single most common way to get a wrong result out of the
simulator.

Config files use the names of the `GnssScheduler` and `GnssFaultModel` variants:

| Section | Valid `kind` values |
|---|---|
| `scheduler` | `pass_through`, `fixed_interval`, `duty_cycle` |
| `fault` | `none`, `degraded`, `slow_bias`, `hijack`, `combo` |

**The CLI flags use different spellings for the same things**: `--sched passthrough|fixed|duty`
and `--fault none|degraded|slowbias|hijack`. The duty-cycle phase is `--duty-phase-s` on the
command line and `start_phase_s` in a config file, while `phase_s` in a config file belongs to
`fixed_interval` only.

`core/tests/example_configs.rs` guards every file in `examples/configs/` against this class of
mistake: it parses each one and asserts the result is the scenario the file describes, rather
than a silently-defaulted pass-through.

## Schedulers

### `pass_through`

Every fix reaches the filter. The baseline for comparing degraded runs.

```yaml
scheduler:
  kind: pass_through
```

### `fixed_interval`

One fix every `interval_s` seconds, the rest discarded. Models a reduced update rate.

```yaml
scheduler:
  kind: fixed_interval
  interval_s: 10.0
  phase_s: 0.0       # offset before the first emitted fix
```

### `duty_cycle`

Alternating windows of availability and denial. The timeline is `start_phase_s` of initial
availability, then `off_s` denied and `on_s` available, repeating.

```yaml
scheduler:
  kind: duty_cycle
  on_s: 120.0          # 2 minutes available
  off_s: 30.0          # 30 seconds denied
  start_phase_s: 0.0
```

Every fix inside an ON window is delivered, and none inside an OFF window.

## Fault models

Faults corrupt the content of a fix; the scheduler decides whether it arrives at all. The two
compose, so a duty cycle can deliver degraded fixes during its ON windows.

```yaml
# No corruption
fault:
  kind: none

# AR(1)-correlated position and velocity error, with inflated reported covariance.
# Models low signal-to-noise or multipath.
fault:
  kind: degraded
  rho_pos: 0.99        # correlation of the position error, per step
  sigma_pos_m: 3.0
  rho_vel: 0.95
  sigma_vel_mps: 0.3
  r_scale: 5.0         # factor the advertised covariance is inflated by

# A slowly drifting offset. Models soft spoofing: the trajectory is nudged
# gradually, and each individual fix looks plausible.
fault:
  kind: slow_bias
  drift_n_mps: 0.02
  drift_e_mps: 0.0
  q_bias: 1e-6
  rotate_omega_rps: 0.0

# A hard offset during a fixed window. Models hard spoofing.
fault:
  kind: hijack
  offset_n_m: 50.0
  offset_e_m: 0.0
  start_s: 120.0
  duration_s: 60.0
```

`combo` takes a list of the above and applies them in sequence, each feeding into the next.

## Reproducibility

`seed` drives every stochastic part of the fault models. The same seed with the same
configuration produces identical output, which is what makes a reported error metric
meaningful. Vary the seed to generate realizations for Monte Carlo work.

Worked files for all of these live in
[`examples/configs/`](https://github.com/jbrodovsky/strapdown-rs/tree/main/examples/configs).
