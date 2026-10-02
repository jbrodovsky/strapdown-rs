# Particle Filter

`strapdown-sim pf` runs the Rao-Blackwellized particle filter (RBPF), after Canciani & Raquet
(IEEE TAES 53(1), 2017). It is the only particle filter the binary offers. Horizontal position
error is carried by particles; altitude, velocity, tilt, the barometer loop's states and any map
biases form one Kalman filter whose covariance all particles share. The filter exists for
map-aided navigation, where a gravity or magnetic map makes the measurement a multimodal function
of position; with GNSS alone it is a working, if more expensive, alternative to the Kalman
filters. The structure, the equations and the departures from the paper are on the
[Rao-Blackwellized Particle Filter](../filters/rbpf.md) page; this page is about running it.

```bash
strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42
strapdown-sim pf -i synthetic.csv -o results/pf.csv
```

```console
[INFO] - Processing file: synthetic.csv
[INFO] - Read 6000 records from synthetic.csv
[INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.754 m/s^2 against a local gravity of 9.780 m/s^2
[INFO] - Results written to results/pf.csv
[INFO] - Particle filter simulation complete
```

(Timestamps trimmed.) The shared flags -- `-i`, `-o`, `--enu`, the execution and health limits --
are described in [Running Simulations](./simulations.md). `pf` also takes the same GNSS
scheduler and fault flags as `cl` (`--sched`, `--fault` and their parameters, including
`--tau-pos-s`/`--tau-vel-s`), with the same meanings and defaults; they are documented once, on
the [Closed Loop](./closed-loop.md#gnss-scheduling-when-fixes-arrive) page.

## What differs from `cl`

- **Measurements.** GNSS fixes and magnetometer headings are measurement updates, as in `cl`.
  Barometer readings are not: they drive a third-order barometer loop inside the mechanization,
  which holds the vertical channel, so they never reweight the particles.
- **No IMU bias states.** The filter follows the paper's thirteen-state model. Adding bias states
  made it diverge on degraded-GNSS runs. The output's `acc_bias_*` and `gyro_bias_*` columns and
  their covariances are therefore zero.
- **No barometric bias state.** The `baro_bias` and `baro_bias_cov` columns are empty; the
  barometer loop's own error states take that role.
- **No innovation gate.** The library's RBPF accepts one, but `strapdown-sim` installs none for
  `pf`: `--gate-confidence` and its recovery flags belong to `cl`, and a `[closed_loop]` gate in a
  configuration file is not read in particle-filter mode. The health monitor checks the state
  and covariance bounds after every event, but no NIS is passed to it, so `--nis-pos-max` and
  `--nis-pos-consec-fail` are accepted and have no effect on `pf`.
- **One seed for two things.** `--seed` seeds both the GNSS fault models and the filter's own
  sampling.

## Flags

### Filter and initial uncertainty

| Flag | Default | Meaning |
|---|---|---|
| `--filter-type <TYPE>` | `rao-blackwellized` | the only value |
| `--num-particles <N>` | `100` | particle count |
| `--seed <SEED>` | `42` | seeds the particles and the fault models |
| `--position-std <M>` | `10` | initial position standard deviation, metres, applied to north, east and up alike |
| `--velocity-std <MPS>` | `1` | initial velocity standard deviation, m/s |
| `--attitude-std <RAD>` | `0.1` | initial tilt standard deviation, radians |

`--position-std` sets all three axes to one value. A configuration file's `position_init_std_m`
takes the three separately and defaults to `[10, 10, 5]`, so a flag run and a default config run
start from slightly different altitude priors.

### Process noise

| Flag | Default | Meaning |
|---|---|---|
| `--horizontal-process-noise-std-m <N> <E>` | `1 1` | random walk on the sampled horizontal position error, north then east, m/√s |
| `--velocity-process-noise-std-mps <V>` | `0.001` | velocity random walk, m/s per √s |
| `--attitude-process-noise-std-rad <A>` | `0.01` | attitude random walk, rad per √s |

All three are rates per root second: the per-step standard deviation is the value times
$\sqrt{\Delta t}$, so the variance grows linearly in elapsed time whatever the log's sample rate.

**The horizontal default is not the paper's.** Canciani & Raquet's eq. 19 puts no process noise
on horizontal position, and the library's `RbpfConfig` keeps that zero. With GNSS-rate fixes on
MEMS data a zero-noise cloud collapses and the filter diverges, so `strapdown-sim` and its
configuration file default to 1 m/√s on each axis, the value the repository's `conf/` recipes
use. Pass `0 0` to run the paper's model. On the stationary 600 s synthetic trajectory above all
three settings finish: scored against `syn --no-noise` truth with the
[Quick Start](../quick-start.md#7-score-the-runs)'s script, `0 0` gives a horizontal RMS of
1.516 m, the default `1 1` 0.921 m and `0.5 0.5` 0.704 m (`0 0` with roughening off as well
also finishes, at 1.715 m). The divergence the default guards against is the one the library
documents for GNSS-rate fixes on recorded MEMS data; a stationary synthetic run does not show it.

`--horizontal-process-noise-std-m` takes exactly two values, north then east, separated by a
comma or a space: `--horizontal-process-noise-std-m 0.5,0.5` and
`--horizontal-process-noise-std-m 0.5 0.5` are the same. A single value is refused.

### Barometer loop

| Flag | Default | Meaning |
|---|---|---|
| `--baro-loop-time-constant-s <S>` | `10` | time constant of the barometer loop in the mechanization |
| `--baro-error-std-m <M>` | `8.3` | steady-state standard deviation of the barometer-aiding error, the hourly drift the Kalman filters' barometric bias is sized from |
| `--baro-error-time-constant-s <S>` | `3600` | correlation time of the barometer-aiding error |
| `--vertical-accel-error-init-std-mps2 <A>` | `0.1` | initial standard deviation of the loop's vertical-acceleration error, the consumer-grade accelerometer bias instability |

### Resampling

| Flag | Default | Meaning |
|---|---|---|
| `--effective-sample-threshold <F>` | `1.0` | resample when the effective sample size falls below this fraction of the particle count; `1.0` resamples after every update, as the paper does |
| `--roughening-factor <K>` | `0.2` | jitter applied after resampling, as a fraction of the cloud's extent; `0` turns it off, as the paper has it |

Roughening is on by default because resampling makes exact copies: when the weights degenerate
onto one particle the cloud collapses to a point and its reported covariance to floating-point
noise. The [RBPF](../filters/rbpf.md) page has the detail.

### Map biases (geophysical builds only)

With `--features geonav`, `pf` takes the same `--geo`, `--gravity-*`, `--magnetic-*` and
`--geo-interval-s` flags as `cl`, and four of its own that shape each map bias's temporal
variation:

| Flag | Default | Meaning |
|---|---|---|
| `--gravity-variation-std <MGAL>` | derived from `--gravity-bias-process-noise-std` | steady-state sigma of the gravity bias's time-varying part |
| `--gravity-variation-time-constant-s <S>` | `300` | its correlation time |
| `--magnetic-variation-std <NT>` | derived from `--magnetic-bias-process-noise-std` | the same for the magnetic bias; the paper uses 5 nT |
| `--magnetic-variation-time-constant-s <S>` | `300` | its correlation time; the paper's is 300 s |

See [Geophysical Navigation](../geonav/overview.md).

### Removed flags

Four flags from before the Canciani & Raquet restructure are still recognized, so that a script
passing one fails with a message saying where the setting went rather than a bare parse error:

| Removed flag | Replacement |
|---|---|
| `--geo-bias-init-std` | `--gravity-bias-init-std` (mGal) and `--magnetic-bias-init-std` (nT) |
| `--geo-bias-process-noise-std` | `--gravity-bias-process-noise-std` and `--magnetic-bias-process-noise-std` |
| `--process-noise-std-m` | `--horizontal-process-noise-std-m`; altitude has no process noise (eq. 20) |
| `--zero-vertical-velocity`, `--zero-vertical-velocity-std-mps` | the barometer loop; see `--baro-loop-time-constant-s` |

```console
$ strapdown-sim pf -i synthetic.csv -o results/pf_x.csv --process-noise-std-m 1,1,1
Error: "`--process-noise-std-m` was removed: the particle filter follows Canciani & Raquet, whose altitude error has no process noise (eq. 20). Set the horizontal pair with `--horizontal-process-noise-std-m north,east` (m per sqrt(s))"
```

## Examples

```bash
# More particles, a duty-cycled outage that starts after 100 s of GNSS, and a degraded fix
strapdown-sim pf -i synthetic.csv -o results/pf_s.csv --num-particles 200 \
  --sched duty --on-s 100 --off-s 50 --duty-phase-s 100 --fault degraded

# Tighter horizontal process noise
strapdown-sim pf -i synthetic.csv -o results/pf_h.csv --horizontal-process-noise-std-m 0.5 0.5
```

The same run from a configuration file uses the `[particle_filter]` section; see
[Configuration Files](./configuration.md#particle_filter).

## Accuracy and cost

The accuracy suite runs one RBPF scenario, `real_rbpf_slice__rbpf`: the first 1,200 records of
the reference recording with GNSS every epoch and 500 particles. Its numbers are in the tables
under [Performance Baselines: Current values](../development/performance.md#current-values); its
consistency metrics are reported but not yet gated.

Runtime is not documented here. The cost of a run depends on the particle count, the trajectory's
length and rate, and the machine, and no repeatable measurement of it has been made for this
book.
