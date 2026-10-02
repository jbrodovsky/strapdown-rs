# Synthetic Trajectories

`strapdown-sim syn` generates a trajectory from an initial kinematic state and writes it as CSV:
either noisy sensor records that every other mode can read, or the exact truth those records were
drawn from. Since no sample dataset ships with the repository, `syn` is the reproducible input
that every example in this book runs on:

```bash
strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42
```

```console
[INFO] - Generated 6000 synthetic records (600.0 s at 10 Hz)
[INFO] - Sensor records written to synthetic.csv
```

## The motion model

The vehicle moves at a **constant navigation-frame velocity** while the body turns at a
**constant body-frame angular rate**: zero linear and angular acceleration. The angular rate turns
the body, not the path -- with `--velocity-north-mps 10 --angular-velocity-z-dps 1` the vehicle
spins slowly while still travelling due north. Every trajectory is therefore a straight line (a
rhumb line over the ellipsoid) at constant altitude rate.

Each epoch's ideal IMU output is computed by *inverse* mechanization: the specific force and
angular rate that keep the navigation-frame velocity constant, including the Earth-rate,
transport-rate and Coriolis terms, so that mechanizing the perfect IMU reproduces the truth. The
truth itself is propagated through the library's own mechanization (Groves §5.4-5.5) from those
perfect samples. Noise is then added to produce the sensor records.

## Flags

### Output and timing

| Flag | Default | Meaning |
|---|---|---|
| `-o`, `--output <FILE>` | required | output CSV path; its parent directory is created |
| `--duration-s <S>` | `300` | trajectory length, seconds |
| `--sample-rate-hz <HZ>` | `10` | sample rate of every channel, Hz |
| `--seed <SEED>` | `42` | seeds every noise draw |
| `--no-noise` | off | write the truth instead of sensor records; see [below](#truth-with---no-noise) |
| `--enu` | off | emit the trajectory in ENU rather than NED |

`syn` takes no `-i`. Its output path is always one file, and must end in `.csv`: any other
extension (`-o x.parquet`) or none is refused rather than given CSV under its name. Unlike the
simulation subcommands, it never treats the path as a directory.

### Initial state

| Flag | Default | Meaning |
|---|---|---|
| `--latitude-deg` | `0` | WGS84 latitude, degrees |
| `--longitude-deg` | `0` | WGS84 longitude, degrees |
| `--altitude-m` | `0` | height above the ellipsoid, metres |
| `--velocity-north-mps` | `0` | north velocity, m/s |
| `--velocity-east-mps` | `0` | east velocity, m/s |
| `--velocity-down-mps` | `0` | vertical velocity, m/s: positive **down** by default, positive **up** with `--enu` (the flag keeps its NED name) |
| `--roll-deg`, `--pitch-deg`, `--yaw-deg` | `0` | initial attitude, degrees |
| `--angular-velocity-x-dps`, `-y-dps`, `-z-dps` | `0` | constant body rates about x, y and z, degrees per second |

With every default the vehicle sits still at latitude 0, longitude 0, altitude 0. Pass negative
values with `=`: `--longitude-deg=-75`.

### Sensor noise

| Flag | Default | Meaning |
|---|---|---|
| `--imu-grade <GRADE>` | `consumer` | IMU noise and bias levels; see [IMU grades](#imu-grades) |
| `--gnss-horizontal-noise-m <M>` | `2.5` | one-sigma white noise on the GNSS horizontal position, metres |
| `--gnss-vertical-noise-m <M>` | `5` | one-sigma white noise on the GNSS altitude, metres |
| `--gnss-velocity-noise-mps <MPS>` | `0.5` | one-sigma white noise on the north and east velocity behind `speed` and `bearing`, m/s per axis |
| `--baro-noise-std-pa <PA>` | about `26.86` | one-sigma white noise on the barometric pressure, pascals. `relativeAltitude` is computed from the noisy pressure, at about 0.083 m per pascal near sea level |
| `--mag-noise-std-ut <UT>` | `0.5` | one-sigma white noise on each magnetometer axis, microtesla |
| `--mag-hard-iron-std-ut <UT>` | `0` | standard deviation of a hard-iron offset drawn once per run and held, microtesla per axis |

The barometric default is not a round number because it is derived: it is the pressure noise
whose altitude equivalent at sea level is the 2.24 m ($\sqrt 5$ m) one-sigma the filters assume
for a barometer (`measurements::BAROMETRIC_ALTITUDE_NOISE_M`), so a default filter's $R$
describes a default synthetic barometer.

Hard iron is off by default on purpose: a constant body-frame field biases the computed heading in
a way no filter here can observe, so with it on the yaw error measures the offset rather than the
filter.

## IMU grades

Each grade sets four numbers, from the table in `IMUQuality` (`core/src/lib.rs`, after Groves
Table 4.1). The **bias** is drawn once per run per axis from a zero-mean normal with the bias
instability as its standard deviation, and held constant. The **white noise** is drawn every
sample, with the random-walk coefficient scaled to the sample rate as
$\sigma = \text{RW}\sqrt{f_s/3600}$.

| `--imu-grade` | gyro bias instability (deg/h) | gyro angle random walk (deg/√h) | accel bias instability (m/s²) | accel velocity random walk (m/s/√h) |
|---|---:|---:|---:|---:|
| `consumer` | 100 | 1 | 0.1 | 0.1 |
| `industrial` | 50 | 0.1 | 0.05 | 0.03 |
| `tactical` | 1 | 0.01 | 0.001 | 0.01 |
| `navigation` | 0.01 | 0.005 | 0.0001 | 0.005 |
| `strategic` | 0.0001 | 0.0005 | 0.00001 | 0.0001 |

Running [`dr`](./dead-reckoning.md) on the same trajectory at different grades shows what each
grade means for unaided drift.

## What it writes

### Sensor records (the default)

Without `--no-noise`, the output is a `TestDataRecord` CSV: the same 31 columns, in the same
order, as a Sensor Logger export, readable by `dr`, `cl` and `pf` without conversion. Column
definitions are on [Input Data Format](./data-format.md). What `syn` puts in them:

| Columns | Content |
|---|---|
| `time` | RFC 3339 UTC timestamps starting at `2025-01-01T00:00:00Z`, one per sample |
| `acc_*`, `gyro_*` | perfect IMU plus the grade's constant bias and white noise, body frame |
| `latitude`, `longitude`, `altitude` | truth plus white GNSS noise, **on every row** |
| `speed`, `bearing` | ground speed (m/s) and track (degrees) of the true north/east velocity plus `--gnss-velocity-noise-mps` white noise on each axis |
| `horizontalAccuracy` | the `--gnss-horizontal-noise-m` value |
| `verticalAccuracy` | the `--gnss-vertical-noise-m` value |
| `speedAccuracy` | the `--gnss-velocity-noise-mps` value |
| `bearingAccuracy` | $\operatorname{atan2}(\sigma_v, \text{speed})$ in degrees: the angle the velocity noise subtends at the row's speed, which tends to 90° as the platform stops |
| `pressure` | isothermal-atmosphere pressure at the true altitude plus `--baro-noise-std-pa` noise, **hPa** (the unit a Sensor Logger export uses) |
| `relativeAltitude` | the height of the noisy pressure relative to the **first** noisy pressure, through the same isothermal atmosphere, m. It is `0` on the first row |
| `mag_*` | the World Magnetic Model field at the true position and the record's date, rotated into the body frame, plus noise and any hard iron, µT |
| `grav_*` | local gravity rotated into the body frame, m/s² |
| `qw`, `qx`, `qy`, `qz`, `roll`, `pitch`, `yaw` | the true attitude, as a quaternion and as Euler angles in radians |

**There is no truth in this file.** It contains only what a sensor would report.

Three properties of these records matter when you use them:

- **GNSS is on every row**, so with the default `--sched passthrough` a filter receives GNSS at
  the full sample rate -- 10 Hz by default. For a realistic 1 Hz receiver, run the filter with
  `--sched fixed --interval-s 1`, as the accuracy suite's `syn_cruise_1hz` scenario does.
- **The barometer is an independent sensor with a constant bias.** `relativeAltitude` comes
  from the noisy pressure, not from the GNSS altitude, so its noise is independent of the GNSS
  fix and set by `--baro-noise-std-pa`. Because it is referenced to the first noisy reading, as
  a phone's is, that reading's error becomes a constant offset in every later row: the
  barometric bias the Kalman filters estimate.
- **The velocity aiding is noisy, and says by how much.** `speed` and `bearing` carry
  `--gnss-velocity-noise-mps` of noise per axis, and `speedAccuracy` reports that one-sigma,
  which is what the filters weight the velocity fix with. On the stationary default the noise is
  all there is, so `speed` is a small positive number and `bearing` is random.

### Truth with `--no-noise`

With `--no-noise`, `syn` writes the truth instead, as a `NavigationResult` CSV -- the same 40
columns every simulation mode writes, described on [Output Format](./output-format.md). Position,
velocity and attitude are exact; bias and covariance columns are zero and the six optional bias
columns are empty.

The truth depends only on the kinematic flags (initial state, rates, duration, sample rate,
frame). It does not depend on `--seed`, `--imu-grade` or any noise flag, because it is propagated
from the perfect IMU. So the truth for a noisy run is obtained by repeating the command with
`--no-noise` added:

```bash
strapdown-sim syn -o synthetic.csv       --duration-s 600 --seed 42
strapdown-sim syn -o synthetic_truth.csv --duration-s 600 --seed 42 --no-noise
```

Both files have one row per sample with identical timestamps, which is also how the output of
`dr`, `cl` and `pf` on `synthetic.csv` is laid out, so a result can be scored against the truth
row by row.

## A moving trajectory

The stationary default is enough to check that a filter holds still. For anything else, give the
vehicle a velocity. This is the trajectory the accuracy suite's synthetic scenarios use -- a
50 m/s cruise north-east at 50 Hz -- written for 120 s:

```bash
strapdown-sim syn -o cruise.csv --duration-s 120 --sample-rate-hz 50 \
  --latitude-deg 40 --longitude-deg=-75 --altitude-m 200 \
  --velocity-north-mps 40 --velocity-east-mps 30 --yaw-deg 36.86989764584402 \
  --gnss-horizontal-noise-m 3 --gnss-vertical-noise-m 5 --baro-noise-std-pa 30 --seed 42
```

The yaw of 36.87 degrees points the body along the velocity vector, so the vehicle travels
nose-first.

## Frames

`syn` writes NED by default: at rest the accelerometer reads about $-9.8$ m/s² on the down axis,
which is what `dr`, `cl` and `pf` expect without `--enu`. `syn --enu` writes ENU records, which
pair with `--enu` on the simulation subcommands. Mixing the two is refused before propagation;
see [Input Data Format](./data-format.md#frame-convention).

## Reproducibility

The same flags and seed produce a byte-identical file: two runs of
`syn -o s.csv --duration-s 60 --seed 42` compare equal with `cmp`. Three things make that hold:

- every random draw comes from a generator seeded by `--seed`;
- the start time is fixed at `2025-01-01T00:00:00Z` rather than taken from the clock, which also
  fixes the date the magnetic model is evaluated at;
- the magnetometer noise, the hard-iron offset and the GNSS velocity noise draw from their own
  streams, so changing `--mag-noise-std-ut` changes only the three `mag_*` columns, and changing
  `--gnss-velocity-noise-mps` changes only `speed`, `bearing`, `speedAccuracy` and
  `bearingAccuracy`. The IMU, GNSS position and barometer noise are untouched by either.

Combined with the exact truth from `--no-noise`, that makes a synthetic trajectory the one input
on which an error can be attributed to the estimator rather than to an unknown reference.

## The `[synthetic]` configuration section

A configuration file with `mode = "synthetic"` runs `syn` from a `[synthetic]` section instead of
from flags. The keys are the flags' names with two differences: the IMU grade is `imu_quality`,
and the initial state sits in its own `[synthetic.initial_state]` table.

```toml
mode = "synthetic"

[synthetic]
output = "cfg_out/syn_cruise.csv"   # required
duration_s = 300.0                   # required
sample_rate_hz = 50.0                # default 10
imu_quality = "consumer"             # consumer | industrial | tactical | navigation | strategic
seed = 42
no_noise = false
gnss_horizontal_noise_m = 3.0        # default 2.5
gnss_vertical_noise_m = 5.0          # default 5
gnss_velocity_noise_mps = 0.5        # default 0.5
baro_noise_std_pa = 30.0             # default about 26.86
mag_noise_std_ut = 0.5
mag_hard_iron_std_ut = 0.0

[synthetic.initial_state]
latitude_deg = 40.0                  # these three are required
longitude_deg = -75.0                # once the table is present
altitude_m = 200.0
velocity_north_mps = 40.0            # the rest default to zero
velocity_east_mps = 30.0
velocity_down_mps = 0.0
yaw_deg = 36.86989764584402
is_enu = false
```

```console
$ strapdown-sim --config syn.toml
[INFO] - Mode: Synthetic
[INFO] - Generated 15000 synthetic records (300.0 s at 50 Hz)
[INFO] - Sensor records written to cfg_out/syn_cruise.csv
```

`output` and `duration_s` have no defaults: a `[synthetic]` section without them is refused with
``missing field `duration_s` ``. Leave out `[synthetic.initial_state]` entirely and the vehicle
starts at rest at 0, 0, 0; include it and `latitude_deg`, `longitude_deg` and `altitude_m` must
all be given. The other keys of a configuration file (`input`, `[closed_loop]` and so on) are
ignored in synthetic mode, except `[logging]`, which sets the log level and file as in any mode.
