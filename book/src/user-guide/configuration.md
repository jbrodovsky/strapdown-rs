# Configuration Files

A scenario configuration describes how the aiding sensors behave during a simulation: **when**
their measurements are delivered and, for GNSS, **how** they are corrupted. It is a single
document, and it can be written as YAML, JSON or TOML — the format is chosen by the file
extension.

```yaml
scheduler:        # when GNSS fixes arrive
  kind: duty_cycle
  on_s: 120.0
  off_s: 30.0
  start_phase_s: 0.0
fault:            # how they are corrupted
  kind: none
baro_scheduler:           # when barometric altitude arrives; 1 Hz if omitted
  kind: fixed_interval
  interval_s: 1.0
  phase_s: 0.0
magnetometer_scheduler:   # when a magnetometer heading arrives; 1 Hz if omitted
  kind: fixed_interval
  interval_s: 1.0
  phase_s: 0.0
baro_noise_std_m: 2.23606797749979   # barometer one-sigma, metres; this is the default
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
| `scheduler`, `baro_scheduler`, `magnetometer_scheduler` | `pass_through`, `fixed_interval`, `duty_cycle` |
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

## The other two aiding channels

### The barometer's noise

`baro_noise_std_m` is the one-sigma barometric altitude uncertainty in **metres**, and it is
the only aiding-noise knob a config file has: the Sensor Logger format carries no
pressure-accuracy column, so unlike GNSS — whose noise comes from each record's
`horizontal_accuracy` — nothing in a log can supply one. Set it to model a good barometer or a
bad one.

It is a *standard deviation*. The value it replaced lived in a trait impl as `diag([5.0])`,
which is a **variance**, so the default here is its square root and the filter sees the same
$R$ it always did. Setting `baro_noise_std_m: 5.0` is not the old behaviour — it is five times
looser.

### The barometer's bias

A barometer does not read altitude, it reads pressure, and the reference pressure it is
converted against drifts — roughly a hectopascal an hour, which is about 8.3 m. A filter that
models the reading as unbiased has nowhere to put that drift except into altitude.

`estimate_baro_bias` gives the filter a state for it. It belongs to the **`closed_loop`**
section, not to the aiding config above, because it changes the filter rather than the sensor:

```yaml
closed_loop:
  filter: ukf
  estimate_baro_bias: true
```

or `--estimate-baro-bias` on the command line. The output then carries `baro_bias` and
`baro_bias_cov` columns; without it both are empty, which is how a reader tells "no bias state"
from "bias estimated at zero".

Measured on `core/tests/test_data.csv`, switching it on moves the vertical channel in all
three Kalman filters at once:

| | 3σ containment | vertical bias | vertical RMSE |
|---|---|---|---|
| off | 0.40 | +0.40 m | 2.58 m |
| on | 0.84 | −0.02 m | 1.41 m |

Containment is the fraction of epochs whose true altitude lies inside the filter's own 3σ
band, so its ideal is 0.9973. The three filters converge independently on a bias of about
−0.6 m, which is the corroboration a single filter could not give.

It is **off by default**: it widens the state vector by one, and that is a default to settle at
the 1.0 API freeze rather than alongside the state itself.

`baro_scheduler` and `magnetometer_scheduler` take the same three kinds and the same fields,
each with its own independent clock, so a GNSS outage, a barometer outage and a heading outage
can be configured separately or made to overlap. Neither channel is corrupted: `fault` governs
GNSS only.

**Both default to `fixed_interval` at 1 Hz, not to `pass_through`.** Before #375 they were not
scheduled at all — one measurement per record, whatever the log's sample rate happened to be.
On a 1 Hz Sensor Logger export that is right by coincidence; on a 50 Hz synthetic trajectory it
delivered fifty pressure readings and fifty derived headings every second, each entering the
filter as an independent fix with full weight. A heading re-read from the same field vector
fifty times is one measurement counted fifty times, and the UKF diverges on it.

Set `kind: pass_through` to get the old behaviour back, when your log's rate really is the
sensor's rate:

```yaml
magnetometer_scheduler:
  kind: pass_through
```

There are no CLI flags for these two; they are configurable from a file only.

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
