# Measurement Models and Integrity

Every filter's `update` takes a `&dyn MeasurementModel`, so any model below can be applied to
the ESKF, EKF, UKF or RBPF, and a new aiding source is a new type implementing the trait, with no
change to the filters. The models live in `core/src/measurements.rs`; the stationary detector
in `core/src/stationary.rs`; innovation gating in `core/src/gating.rs`.

The integration is **loosely coupled**: GNSS enters as a position (and velocity) solution
produced by the receiver. There are no pseudorange or carrier-phase models in this crate.

## The `MeasurementModel` trait

| method | returns | meaning |
| --- | --- | --- |
| `get_dimension()` | `usize` | length of the measurement vector |
| `get_measurement(&state)` | `Result<DVector<f64>, StrapdownError>` | the observed $z$; `state` is passed so a model can use it (the magnetometer levels itself with the estimated roll and pitch) |
| `get_expected_measurement(&state)` | `DVector<f64>` | the prediction $h(x)$ |
| `get_noise()` | `DMatrix<f64>` | the measurement covariance $R$, a **variance** |
| `get_jacobian(&state)` | `Result<DMatrix<f64>, StrapdownError>` | $H = \partial h/\partial x$ |
| `wrap_residual(&mut residual)` | | wraps angular innovations; the default does nothing |

Three conventions hold for every model:

- `state` is the [shared layout](./kalman.md#state-layout-and-units): latitude and longitude in
  radians, attitude as roll, pitch and yaw.
- The attitude columns of $H$ are derivatives with respect to **roll, pitch and yaw**. The EKF
  uses them directly; the ESKF converts them to its body-frame rotation-vector error
  ([ESKF](./eskf.md#update-injection-and-reset)) and the RBPF to its nav-frame tilt.
- Every `*_noise_std` field is a one-sigma standard deviation; the model squares it to form
  $R$.

`get_measurement` and `get_jacobian` return `Result` because some models can legitimately fail
to produce a value; a geophysical model queried off its map is the main case, and its error is
*recoverable* (skip the measurement, carry on).

## The models

| model | dim | $z$ | noise fields | observes |
| --- | :-: | --- | --- | --- |
| `GPSPositionMeasurement` | 3 | latitude, longitude (given in degrees, converted to radians), altitude | `horizontal_noise_std`, `vertical_noise_std` (m) | position |
| `GPSVelocityMeasurement` | 3 | north, east, vertical velocity (vertical in the state's frame convention) | `horizontal_noise_std`, `vertical_noise_std` (m/s) | velocity |
| `GPSPositionAndVelocityMeasurement` | 5 | latitude, longitude, altitude, north and east velocity; **no vertical velocity** | `horizontal_noise_std`, `vertical_noise_std` (m), `velocity_noise_std` (m/s) | position, horizontal velocity |
| `RelativeAltitudeMeasurement` | 1 | `relative_altitude + reference_altitude` | `noise_std` (m) | altitude, and a barometric bias if `bias_index` names one |
| `MagnetometerYawMeasurement` | 1 | tilt-compensated heading from a body-frame field, optionally corrected by WMM declination | `noise_std` (rad) | yaw |
| `ZuptMeasurement` | 3 | zero velocity | `velocity_noise_std` (m/s) | velocity |
| `ZaruMeasurement` | 3 | the raw body-frame gyro reading | `angular_rate_noise_std` (rad/s) | gyroscope bias |

The GNSS position noise is a ground distance in metres. It is converted to latitude and
longitude variances through the WGS84 radii of curvature at the fix, separately for each axis,
because a metre of easting is a larger angle of longitude than a metre of northing is of
latitude.

### What `strapdown-sim` feeds the filter

`messages::build_event_stream` turns each input record into events, and the simulator's filters
see three measurement types:

| source | model | rate | noise |
| --- | --- | --- | --- |
| GNSS | `GPSPositionAndVelocityMeasurement` | every record with a fix, then thinned by the `[aiding]` scheduler and corrupted by its fault model | the record's `horizontalAccuracy`, `verticalAccuracy`, `speedAccuracy` |
| barometer | `RelativeAltitudeMeasurement` | `baro_scheduler`, 1 Hz by default | `baro_noise_std_m`, default $\sqrt{5}$ m |
| magnetometer | `MagnetometerYawMeasurement`, declination on | `magnetometer_scheduler`, 1 Hz by default | `MAG_YAW_NOISE`, 0.2 rad, fixed |

[Schedulers and Faults](../gnss/scenarios.md) covers the scheduler and fault options. ZUPT and
ZARU are not applied by the simulator; they are library building blocks.

### Barometric altitude

`RelativeAltitudeMeasurement` predicts `altitude + bias`, where the bias is `state[bias_index]`
when `bias_index` is `Some` and zero otherwise. An index that falls inside the fifteen
navigation and IMU-bias states, or past the end of the state, is refused with an error rather
than read as a bias. The default noise, `BAROMETRIC_ALTITUDE_NOISE_M`, is $\sqrt5 \approx 2.24$
m: it keeps the variance the model used before the field existed, which was 5 m².

### Magnetometer yaw

`MagnetometerYawMeasurement` levels the body-frame field with the state's roll and pitch (never
its yaw, which would make the measurement depend on the quantity it measures), takes the
heading with the `atan2` appropriate to the frame, and, with `apply_declination`, adds the World
Magnetic Model declination at the state's position for `year` and `day_of_year`. Declination
enters with opposite signs in NED and ENU, and the `is_enu` field **must** match the state's
frame: a mismatch reflects the heading rather than degrading it. `get_declination(lat_deg,
lon_deg, alt_m)` returns the declination in radians, positive east.

```rust,ignore
{{#include ../../../core/examples/aiding.rs:magnetometer}}
```

### Zero-velocity and zero-angular-rate updates

A stationary vehicle is free information. `ZuptMeasurement` asserts the velocity is zero
(Groves 15.2.1); because velocity error is correlated with accelerometer bias error, the
correction reaches the bias too. `ZaruMeasurement` asserts that a stationary gyro reads only its
own bias plus the Earth's rotation, which makes it a direct observation of the gyro bias (Groves
15.2.2). It predicts $b_g + C_n^b\ \omega_{ie}^n$ (`set_earth_rate_compensation(false)` drops
the Earth-rate term), and it **requires** a filter with gyro-bias states: on a nine-state filter
it returns `StrapdownError::DimensionMismatch` instead of silently correcting nothing.

Neither model decides whether the vehicle is still. That is `stationary::StationaryDetector`'s
job: it keeps a sliding window of IMU samples and declares the platform stationary when all four
hold across the window:

1. specific-force variance (trace) below `accel_variance_threshold`, (m/s²)²;
2. mean specific-force magnitude within `gravity_tolerance_mps2` of 9.807 m/s²;
3. angular-rate variance (trace) below `gyro_variance_threshold`, (rad/s)²;
4. mean angular-rate magnitude below `gyro_mean_threshold`, rad/s. This separates a stop from a
   steady turn, which has small gyro *variance* and a large mean.

It latches only after `min_stationary_samples` consecutive stationary windows and releases on
the first sample that fails. `StationaryConfig::default()` (a 100-sample window, latching after
50 more) is tuned for a consumer MEMS IMU at about 100 Hz. Constant-velocity cruise on a smooth
road can pass all four tests; prefer conservative thresholds, since a false stop injects a hard
zero-velocity constraint into a moving solution.

[`core/examples/aiding.rs`](https://github.com/jbrodovsky/strapdown-rs/blob/main/core/examples/aiding.rs)
parks an ESKF-backed `InsEngine` for 30 s with a gyro bias the filter has not been told about:

```bash
cargo run -p strapdown-core --example aiding
```

```rust,ignore
{{#include ../../../core/examples/aiding.rs:stationary}}
```

### Geophysical models

The experimental `strapdown-geonav` crate adds gravity-anomaly, magnetic-anomaly and combined
models (`GravityMeasurement`, `MagneticAnomalyMeasurement`, `CombinedGeophysicalMeasurement`)
that implement the same trait against a loaded map. See
[Maps and Measurement Models](../geonav/maps.md).

## Innovation gating

A Kalman update trusts its measurement unconditionally. Under the filter's own model the
innovation is $\nu \sim \mathcal N(0, S)$ with $S = HPH^\top + R$, so the normalized innovation
squared

$$
d^2 = \nu^\top S^{-1} \nu
$$

is $\chi^2$-distributed with as many degrees of freedom as the measurement has components
(Bar-Shalom, Li and Kirubarajan, Section 5.4). Rejecting an update whose NIS exceeds a $\chi^2$
quantile discards what the model says should almost never happen. Every filter reports the NIS
in its [`UpdateOutcome`](./kalman.md#outputs-updateoutcome) whether or not a gate is installed.

`gating::InnovationGate` has two variants:

| variant | threshold | constructor |
| --- | --- | --- |
| `ChiSquared { confidence }` | the `confidence` quantile of $\chi^2$ at the measurement's own dimension | `InnovationGate::chi_squared(confidence)`, refusing values outside $(0, 1)$ |
| `Fixed { threshold }` | the same number for every measurement | `InnovationGate::fixed(threshold)` |

Prefer the $\chi^2$ form: a filter mixing 1-D barometer, 3-D position and 5-D position and
velocity updates needs a different threshold for each. `gating::chi_squared_quantile(p, dof)`
and `chi_squared_cdf(x, dof)` are public. Gating is **off** by default everywhere: in the
library, in `ClosedLoopConfig` and on the command line.

### A gate needs a way back

Rejecting a fix leaves the state where it was, but the filter keeps propagating and keeps
accumulating error, while the covariance the next fix is judged against does not grow. The next
innovation is larger, fails by more, and one rejection can become permanent dead reckoning
(#340). `gating::GateRecovery` is the way back, and it is on by default once a gate is
installed:

| field | default | effect |
| --- | --- | --- |
| `rejection_inflation` | 2.0 | each rejection multiplies the filter's uncertainty in the directions that measurement observes, so $HPH^\top$ grows by this factor; 1.0 disables it |
| `forced_update_after` | `Some(5)` | when this many consecutive measurements in a row fail the gate, the last of them is applied anyway and reported with `forced: true`; `None` disables it, and values below 2 are refused by `GateRecovery::new` |

The streak counts every sensor and resets on any accepted update. `GateRecovery::none()` turns
both mechanisms off.

The third scene of `aiding.rs` installs a 0.999 gate on a vehicle driving north at 10 m/s, then
from t = 61 s feeds fixes 200 m east of the truth:

```rust,ignore
{{#include ../../../core/examples/aiding.rs:gate}}
```

```rust,ignore
{{#include ../../../core/examples/aiding.rs:outcome}}
```

### Configuring the gate in `strapdown-sim`

| `cl` flag | config key in `[closed_loop]` | meaning |
| --- | --- | --- |
| `--gate-confidence <P>` | `innovation_gate = { chi_squared = { confidence = P } }` | install a $\chi^2$ gate |
| `--gate-inflation <F>` | `gate_recovery = { rejection_inflation = F }` | inflation per rejection, default 2 |
| `--gate-force-after <N>` | `gate_recovery = { forced_update_after = N }` | forced-update streak, default 5 |

`innovation_gate = { fixed = { threshold = T } }` selects the fixed variant in a file. Gating
rejects individual measurements; a filter that has genuinely diverged produces a large NIS on
every fix, and the run's health monitor (`--nis-pos-max`, `--nis-pos-consec-fail`) is what stops
it rather than letting it degrade silently to dead reckoning.

## Output of `aiding.rs`

```text
parked for 30 s, 286 ZUPT/ZARU pairs
  gyro bias estimate z: 0.00200 rad/s (injected 0.00200)
  speed: 0.0000 m/s, altitude 12.00 m
magnetometer: declination -11.85 deg, true heading 30.00 deg, measured 30.00 deg
chi-squared 0.999 threshold, 1 dof: 10.83
chi-squared 0.999 threshold, 3 dof: 16.27
chi-squared 0.999 threshold, 5 dof: 20.52
t = 59 s  NIS        0.0  accepted true   forced false
t = 60 s  NIS        0.0  accepted true   forced false
t = 61 s  NIS     2805.9  accepted false  forced false
t = 62 s  NIS     1564.2  accepted false  forced false
t = 63 s  NIS      674.9  accepted false  forced false
t = 64 s  NIS      256.0  accepted false  forced false
t = 65 s  NIS       93.6  accepted true   forced true
t = 66 s  NIS       43.5  accepted false  forced false
t = 67 s  NIS      107.0  accepted false  forced false
t = 68 s  NIS      114.9  accepted false  forced false
t = 69 s  NIS       88.9  accepted false  forced false
t = 70 s  NIS       58.8  accepted true   forced true
```

The ZARU updates recover the injected gyro bias; the magnetometer model returns the heading the
field was built for; and the gate rejects the spoofed fixes while each rejection inflates the
covariance, so the NIS falls, until the fifth consecutive failure is forced through. Against a
constant 200 m offset the forced updates pull the solution toward the spoofed track a step at a
time: a gate bounds how long the filter disagrees with its sensor, it does not detect a
spoofer.
