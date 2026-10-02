# Input Data Format

Every simulation mode reads one CSV layout: the export format of the
[Sensor Logger](https://www.tszheichoi.com/sensorlogger) phone app, with one row per sample. In
the library a row is a `TestDataRecord` (`strapdown::sim`), and `TestDataRecord::from_csv` is
the loader `strapdown-sim` calls for every input file. `strapdown-sim syn` writes the same
layout, so a synthetic trajectory and a phone recording are interchangeable inputs.

## Columns

The header must name **all 31 columns**; their order does not matter, and extra columns are
ignored. A *cell* may be empty, `NaN` or `null`, all of which read as NaN, so a channel a
recording lacks is expressed by leaving its cells blank -- not by dropping its column. A file
missing a column parses no rows at all and is refused with `No usable records read from ...`;
run with `--log-level warn` to see the per-row reason (``missing field `bearingAccuracy` ``, for
example).

| Column | Unit | Meaning |
|---|---|---|
| `time` | -- | timestamp with a UTC offset, see [Timestamps](#timestamps) |
| `latitude`, `longitude` | degrees | GNSS position, WGS84 |
| `altitude` | m | GNSS height |
| `speed` | m/s | GNSS ground speed |
| `bearing` | degrees | GNSS ground track, clockwise from north |
| `horizontalAccuracy` | m | one-sigma horizontal position accuracy the receiver reports |
| `verticalAccuracy` | m | one-sigma vertical position accuracy |
| `speedAccuracy` | m/s | one-sigma speed accuracy |
| `bearingAccuracy` | degrees | one-sigma bearing accuracy |
| `acc_x`, `acc_y`, `acc_z` | m/s² | accelerometer specific force, body frame, gravity **included** |
| `gyro_x`, `gyro_y`, `gyro_z` | rad/s | gyroscope angular rate, body frame |
| `qw`, `qx`, `qy`, `qz` | -- | device attitude quaternion |
| `roll`, `pitch`, `yaw` | rad | device attitude as Euler angles, in the app's own convention |
| `mag_x`, `mag_y`, `mag_z` | µT | magnetometer, body frame |
| `relativeAltitude` | m | barometric altitude change since the recording started |
| `pressure` | hPa (mbar) | barometric pressure; `syn` writes hPa too |
| `grav_x`, `grav_y`, `grav_z` | m/s² | the device's gravity estimate, body frame |

The IMU data is raw: the accelerometer columns include gravity, and the strapdown mechanization
removes it during propagation. Do not feed it gravity-compensated "linear acceleration".

## Which columns drive what

Not every column is read by an estimator. This is what each one feeds, taken from
`build_event_stream` (`core/src/messages.rs`), which turns records into the events `cl` and `pf`
process, and from the initialization code all three modes share.

| Columns | Used for | When missing (NaN) |
|---|---|---|
| `acc_*`, `gyro_*` | one IMU propagation step per record | `cl`/`pf`: the step is skipped and the filter coasts across the gap; a gap longer than `aiding.max_imu_gap_s` (5 s) is an error. `dr`: the run fails |
| `latitude`, `longitude`, `altitude`, `speed`, `bearing` | one GNSS position-and-velocity fix, subject to the GNSS scheduler and fault model; also the initial position and horizontal velocity | no fix from that record -- all five must be present -- and the scheduler waits for the next usable one |
| `horizontalAccuracy` | the fix's horizontal one-sigma, floored at 1 mm | 15 m |
| `verticalAccuracy` | the fix's vertical one-sigma, floored at 1 mm | 1000 m |
| `speedAccuracy` | the fix's velocity one-sigma, floored at 0.1 m/s | 100 m/s |
| `relativeAltitude` | one barometric altitude measurement, referenced to the first record's `altitude`, 1 Hz by default | no barometer measurement from that record |
| `mag_x`, `mag_y`, `mag_z` | one magnetometer heading measurement, with declination from the World Magnetic Model at the record's date, 1 Hz by default | no heading measurement from that record |
| `qw`, `qx`, `qy`, `qz` | the initial attitude, from the first record | an all-NaN or zero quaternion gives the identity attitude |
| `grav_*` | gravity-anomaly measurements, geophysical builds only (`--geo`) | -- |
| `bearingAccuracy`, `pressure`, `roll`, `pitch`, `yaw` | nothing; read and carried, never used by an estimator | -- |

Points worth knowing:

- **The accuracy columns are standard deviations**, not variances: the measurement model
  squares them to build $R$. They are the only way per-fix GNSS quality reaches the filter, and
  the `degraded` fault model works by scaling them (see
  [Closed Loop](./closed-loop.md#gnss-faults-what-the-fixes-say)). A column of zeros does not
  produce a singular $R$, because of the floors above; a column of NaN makes every fix very
  loose.
- **The barometer's noise is not in the file.** There is no pressure-accuracy column, so the
  barometric one-sigma comes from configuration (`aiding.baro_noise_std_m`, default 2.236 m).
- **Attitude comes from the quaternion.** The `roll`/`pitch`/`yaw` columns are radians but not in
  the intrinsic XYZ sequence the library uses, and the two disagree by more than a sign on real
  data, so they are not read.
- **The first record seeds every mode**: its position, its `speed`/`bearing` as north and east
  velocity (zero if either is NaN), zero vertical velocity, and its quaternion.
- **Rates are taken from the timestamps.** Each IMU step uses the time since the previous
  record. Barometer and magnetometer measurements are scheduled at 1 Hz by default whatever the
  record rate, and GNSS at every record that carries a fix unless a scheduler says otherwise.

## Timestamps

`time` must carry a UTC offset. Both RFC 3339 (`2025-01-01T00:00:00Z`, which `syn` writes) and
the space-separated form Sensor Logger exports (`2025-01-01 00:00:00+00:00`) are accepted, and any
offset is converted to UTC -- an input stamped `-05:00` comes out five hours later, in UTC, in the
output. A timestamp with no offset fails to parse, and so does its row.

Timestamps must increase strictly: the time step between records is the IMU integration
interval, and a duplicate or out-of-order timestamp gives a non-positive step, which the
mechanization rejects.

## Rows that cannot be read

A row that fails to parse -- a malformed number, a bad timestamp -- is skipped with a warning
(`Skipping row N due to parse error: ...`) and the rest of the file is read. A file that yields
no rows at all is an error, so a wrong schema cannot silently produce an empty result.

## Synthetic data

`strapdown-sim syn` writes exactly this layout, in NED, with GNSS on every row and the accuracy
columns set from the configured noise levels. Its truth is a separate file, written with
`--no-noise` in the [output format](./output-format.md) rather than this one. The details of what
`syn` puts in each column -- including that its `relativeAltitude` is derived from a noisy
pressure independent of the GNSS altitude, and that its `speed`/`bearing` carry their own
velocity noise, reported in `speedAccuracy` -- are on
[Synthetic Trajectories](./synthetic.md#what-it-writes).

## Frame Convention

A CSV carries no frame tag, so the tool cannot tell from the file which local-level frame its
IMU readings are expressed in. You have to say, and the default is **NED** (north-east-down),
matching the library and `strapdown-sim syn`:

| Source | Frame | Vertical specific force at rest | Flag |
| --- | --- | --- | --- |
| Sensor Logger export | ENU | `+9.8` m/s^2 along up | `--enu` |
| `strapdown-sim syn` output | NED | `-9.8` m/s^2 along down | *(default)* |

```bash
# A Sensor Logger recording
strapdown-sim dr -i recording.csv -o results/ --enu

# Anything generated by `syn` -- NED, so no flag
strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42
strapdown-sim dr -i synthetic.csv -o results/
```

The flag applies to `dr`, `ol`, `cl` and `pf` alike, and the same setting is available as
`is_enu = true` at the top level of a config file. `syn --enu` generates ENU data, so
`syn --enu` and `dr --enu` pair up the way plain `syn` and plain `dr` do.

Getting this wrong is not a small error: mechanizing NED records as ENU adds the gravity model
to the sensed specific force instead of cancelling it, and the solution falls at 2 g. So the
declaration is checked rather than trusted. Before propagating, the tool rotates the leading
records' accelerometer readings into the navigation frame and compares the vertical component
against the sign the declared frame requires; if they contradict each other it stops. Declaring
`syn` output as ENU, for example:

```console
$ strapdown-sim dr -i synthetic.csv -o results/wrong.csv --enu
Error: InvalidConfiguration { field: "is_enu", reason: "mean vertical specific force over the first 10 record(s) is -9.75 m/s^2, but ENU mechanization expects +9.78 m/s^2 at rest: these look like NED records. Mechanizing them as ENU would double-count gravity and integrate at 2 g. Declare the frame that matches the data (drop `--enu`, or set `is_enu = false` in the config file), or re-record it in ENU." }
```

The check only rejects; it never guesses. It is deliberately generous -- it takes roughly 1.5 g
of sustained downward acceleration for a correctly declared file to trip it, so free fall and
ordinary manoeuvring are fine -- and it passes anything ambiguous through.

## Collecting data

Sensor Logger is available for iOS and Android. Its export holds one file per sensor, which has
to be merged onto a single timeline with one row per sample, in the column set above, before
`strapdown-sim` can read it. The repository's `analysis` package does that merge -- along with
resampling and splitting recordings at IMU dropouts -- for its own recordings; see
[Python Analysis Tooling](../development/analysis.md).
