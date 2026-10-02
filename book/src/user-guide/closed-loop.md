# Closed Loop (Kalman Filters)

`strapdown-sim cl` runs the INS with a Kalman filter that corrects it from aiding measurements,
feeding each correction back into the navigation solution. It is the mode for GNSS-aided
navigation and for studying what happens when the GNSS is withheld or corrupted.

```bash
strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42
strapdown-sim cl -i synthetic.csv -o results/eskf.csv
```

```console
[INFO] - Running in closed-loop mode with Error-State Kalman Filter (ESKF)
[INFO] - Read 6000 records from synthetic.csv
[INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.754 m/s^2 against a local gravity of 9.780 m/s^2
[INFO] - Initialized event stream with 13198 events
[INFO] - Initialized ESKF
[INFO] - Starting closed-loop navigation filter with 13198 events
...
[INFO] - Results written to results/eskf.csv
```

(Timestamps trimmed; at `info` the filter also logs a progress line every few events, which
`--log-level warn` silences.) The shared flags -- `-i`, `-o`, `--enu`, the execution and health
limits -- are described in [Running Simulations](./simulations.md). This page covers the flags
specific to `cl`.

## Choosing the filter

| Flag | Default | Values |
|---|---|---|
| `--filter <FILTER>` | `eskf` | `eskf`, `ukf`, `ekf` |

- **`eskf`** -- the 15-state error-state Kalman filter with a multiplicative attitude error. The
  default. See [ESKF](../filters/eskf.md).
- **`ekf`** -- the extended Kalman filter. See [EKF](../filters/ekf.md).
- **`ukf`** -- the unscented Kalman filter. See [UKF](../filters/ukf.md).

All three estimate accelerometer and gyroscope biases (the `acc_bias_*` and `gyro_bias_*` output
columns are populated for each) and, by default, a barometric bias. How they compare is on the
[Comparison](../filters/comparison.md) page.

```bash
strapdown-sim cl -i synthetic.csv -o results/ukf.csv --filter ukf
strapdown-sim cl -i synthetic.csv -o results/ekf.csv --filter ekf
```

### UKF sigma-point parameters

Read only with `--filter ukf`.

| Flag | Default | Meaning |
|---|---|---|
| `--ukf-alpha <ALPHA>` | `0.1` | sigma-point spread |
| `--ukf-beta <BETA>` | `2` | prior-distribution parameter; 2 is optimal for a Gaussian |
| `--ukf-kappa <KAPPA>` | `0` | secondary spread parameter |

`--ukf-alpha` defaults to `0.1`, the same value a configuration file uses, not the textbook
`1e-3`. At `1e-3` the weighted sigma-point mean is formed from terms about $10^6$ times the
answer, which costs six significant digits to cancellation on every step; the measurement behind
the choice is recorded on `ClosedLoopConfig::ukf_alpha` in the API documentation.

## What aids the filter

`cl` builds its measurements from the input records. Which columns feed which measurement is
covered in [Input Data Format](./data-format.md#which-columns-drive-what); in summary:

| Measurement | Built from | Rate | Noise |
|---|---|---|---|
| GNSS position and velocity | `latitude`, `longitude`, `altitude`, `speed`, `bearing` | every record that carries all five, subject to `--sched` | the record's `horizontalAccuracy`, `verticalAccuracy`, `speedAccuracy`, then `--fault` |
| Barometric altitude | `relativeAltitude`, referenced to the first record's altitude | 1 Hz | 2.236 m one-sigma |
| Magnetometer heading | `mag_x`, `mag_y`, `mag_z`, with declination from the World Magnetic Model at the record's date | 1 Hz | 0.2 rad one-sigma |

The GNSS channel is the one the command line can degrade. The barometer and magnetometer
schedules, and the barometer's noise, have **no command-line flags**: they take the defaults
above, and a configuration file can change them (`aiding.baro_scheduler`,
`aiding.magnetometer_scheduler`, `aiding.baro_noise_std_m`; see
[Configuration Files](./configuration.md#the-barometer-and-magnetometer)). Neither is ever
corrupted; fault models apply to GNSS only.

### The barometric bias state

A barometer converts pressure to altitude against a reference pressure that drifts, and a filter
that models the reading as unbiased has nowhere to put that drift except into altitude. `cl`
therefore estimates a barometric bias as an extra filter state, **on by default**, and writes it
to the `baro_bias` and `baro_bias_cov` output columns.

| Flag | Effect |
|---|---|
| `--no-estimate-baro-bias` | do not carry the barometric bias state; the two output columns are left empty |

The flag is spelled negatively because the default is on. The old `--estimate-baro-bias` was
removed rather than kept as a no-op, so a script still passing it fails with clap's "unexpected
argument" error instead of silently meaning nothing. The measurements behind the default, and the
one case where the state hurts (an AR(1)-degraded GNSS fault, issue #410), are on the
[Configuration Files](./configuration.md#the-barometric-bias) page.

## GNSS scheduling: when fixes arrive

The scheduler decides which GNSS fixes reach the filter at all. A fix the scheduler withholds is
simply absent; the filter coasts on the IMU, still aided by the barometer and magnetometer.

| Flag | Default | Used by | Meaning |
|---|---|---|---|
| `--sched <KIND>` | `passthrough` | -- | `passthrough`, `fixed` or `duty` |
| `--interval-s <S>` | `1` | `fixed` | seconds between delivered fixes |
| `--phase-s <S>` | `0` | `fixed` | offset of the first delivered fix |
| `--on-s <S>` | `10` | `duty` | length of each available window |
| `--off-s <S>` | `10` | `duty` | length of each denied window |
| `--duty-phase-s <S>` | `0` | `duty` | length of an initial available window before the cycle starts |

- **`passthrough`** delivers every fix.
- **`fixed`** delivers at most one fix per interval: the first usable fix at or after each tick.
  It models a reduced update rate.
- **`duty`** alternates denial and availability. The timeline is `--duty-phase-s` of
  availability, then `--off-s` denied and `--on-s` available, repeating. **With the default
  phase of 0 the run therefore starts in an outage**: `--sched duty --on-s 100 --off-s 50` denies
  GNSS for the first 50 s, delivers it from 50 s to 150 s, denies it from 150 s to 200 s, and so
  on. Set `--duty-phase-s` to start with GNSS available.

```bash
strapdown-sim cl -i synthetic.csv -o results/fixed.csv --sched fixed --interval-s 5
strapdown-sim cl -i synthetic.csv -o results/duty.csv  --sched duty --on-s 100 --off-s 50
```

On the 600 s, 10 Hz trajectory above, the pass-through run's event stream holds 13,198 events
and the duty-cycle run's 11,199 -- the difference is the 1,999 GNSS fixes the two outages
withhold, as `Initialized event stream with ... events` reports at `info` level.

## GNSS faults: what the fixes say

A fault model corrupts the content of each fix the scheduler lets through. Scheduler and fault
compose: a duty cycle can deliver degraded fixes during its available windows.

| Flag | Default | Meaning |
|---|---|---|
| `--fault <KIND>` | `none` | `none`, `degraded`, `slowbias` or `hijack` |

**`degraded`** adds AR(1)-correlated error to position and velocity and inflates the accuracy the
fix advertises. It models low signal-to-noise or multipath.

| Flag | Default | Meaning |
|---|---|---|
| `--rho-pos` | `0.99` | AR(1) correlation coefficient of the position error, per fix |
| `--sigma-pos-m` | `3` | position error, metres: the per-fix innovation, or the steady-state sigma with `--tau-pos-s` |
| `--rho-vel` | `0.95` | AR(1) correlation coefficient of the velocity error, per fix |
| `--sigma-vel-mps` | `0.3` | velocity error, m/s, read the same way as `--sigma-pos-m` |
| `--r-scale` | `5` | factor on the advertised one-sigma accuracies, so $R$ grows by its **square** (25 by default) |
| `--tau-pos-s` | unset | position-error correlation time, seconds |
| `--tau-vel-s` | unset | velocity-error correlation time, seconds |

Without `--tau-pos-s`, `--rho-pos` is applied once per delivered fix, so the error's correlation
time is $-\Delta t / \ln \rho$ with $\Delta t$ the fix interval: changing `--interval-s` changes
the error model too. With `--tau-pos-s`, the coefficient is $e^{-\Delta t/\tau}$ for the actual
interval between fixes and `--sigma-pos-m` becomes the steady-state standard deviation, so the
error keeps the same timescale whatever the schedule. `--tau-vel-s` does the same for velocity.
Leave both unset to reproduce results made before they existed.

**`slowbias`** adds a north/east offset that grows steadily, so each fix looks plausible while
the trajectory is nudged away. It models a soft spoof.

| Flag | Default | Meaning |
|---|---|---|
| `--drift-n-mps` | `0.02` | northward drift rate of the offset, m/s |
| `--drift-e-mps` | `0` | eastward drift rate of the offset, m/s |
| `--q-bias` | `1e-6` | random-walk density of the offset, m²/s: each step adds variance `q_bias * dt` |
| `--rotate-omega-rps` | `0` | rate at which the drift direction rotates, rad/s |

**`hijack`** applies a constant north/east offset over a fixed window. It models a hard spoof.

| Flag | Default | Meaning |
|---|---|---|
| `--hijack-offset-n-m` | `50` | northward offset, metres |
| `--hijack-offset-e-m` | `0` | eastward offset, metres |
| `--hijack-start-s` | `120` | window start, seconds from the first record |
| `--hijack-duration-s` | `60` | window length, seconds |

```bash
strapdown-sim cl -i synthetic.csv -o results/degraded.csv \
  --fault degraded --sigma-pos-m 10 --tau-pos-s 60 --tau-vel-s 30
strapdown-sim cl -i synthetic.csv -o results/hijack.csv \
  --fault hijack --hijack-offset-n-m 100 --hijack-start-s 200 --hijack-duration-s 60
strapdown-sim cl -i synthetic.csv -o results/both.csv \
  --sched duty --on-s 100 --off-s 50 --fault degraded --sigma-pos-m 10.0 --seed 7
```

Only one fault can be chosen on the command line. Several applied in sequence -- a slow drift
followed by a hijack, say -- is the `combo` fault, which exists only in a configuration file; see
[Configuration Files](./configuration.md#fault-models). Every random draw a fault makes comes from
`--seed` (default `42`), so the same seed and flags give the same corrupted fixes. The
[GNSS Degradation](../gnss/fault-simulation.md) chapter describes the models in more depth.

## Innovation gating

By default every measurement is applied. An innovation gate rejects a measurement whose
normalized innovation squared (NIS) is implausibly large for the filter's own covariance.

| Flag | Default | Meaning |
|---|---|---|
| `--gate-confidence <P>` | unset (no gate) | reject a measurement whose NIS exceeds the $\chi^2$ quantile at probability `P`, for that measurement's own degrees of freedom. Must lie strictly inside (0, 1) |
| `--gate-inflation <FACTOR>` | `2` | each rejection multiplies the filter's uncertainty by this factor, in the directions the rejected measurement observed. At least 1.0; 1.0 disables it |
| `--gate-force-after <COUNT>` | `5` | after this many consecutive rejections, apply the next measurement regardless. `0` disables it; `1` is refused, because it would be no gate at all |

Because the threshold is the $\chi^2$ quantile at each measurement's own dimension, one
confidence level means the same thing for a 1-DOF barometer reading and a multi-DOF GNSS fix.
The two recovery flags exist because a gate with no way back is self-reinforcing: a filter that
rejects one fix keeps drifting while the covariance it judges the next one against stays put, so
one rejection leads to the next. The recovery flags are validated even when no gate is
installed.

```console
$ strapdown-sim cl -i synthetic.csv -o results/gated.csv --gate-confidence 0.999 --log-level warn
[WARN] - closed-loop run completed with 78 of 13198 events gated out by the innovation test
$ strapdown-sim cl -i synthetic.csv -o results/bad.csv --gate-confidence 1.5
Error: OutOfRange { what: "innovation gate confidence", value: 1.5, min: 0.0, max: 1.0 }
```

The gate is distinct from the run-level `--nis-pos-max`/`--nis-pos-consec-fail` health check,
which does not reject anything: it fails the whole run when outliers persist. See
[Measurement Models and Integrity](../filters/measurements.md) for the gate's theory.

## Seeds and reproducibility

| Flag | Default | Seeds |
|---|---|---|
| `--seed <SEED>` | `42` | the GNSS fault models |

With `--fault none` nothing in a `cl` run is random, and the output depends only on the input
and the flags.

## Geophysical aiding (experimental)

A binary built with `--features geonav` adds `--geo` and the `--gravity-*`, `--magnetic-*` and
`--geo-interval-s` flags, which aid the UKF or EKF with gravity- or magnetic-anomaly
measurements against a map. The ESKF has no geophysical implementation, so `--geo` requires
`--filter ukf` or `--filter ekf`. See [Geophysical Navigation](../geonav/overview.md).
