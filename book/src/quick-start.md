# Quick Start

This page runs every simulation mode on one synthetic trajectory, start to finish, in about
ten commands. Nothing needs downloading: `strapdown-sim syn` generates the input. Every command
and every line of output below was produced by `strapdown-sim 1.0.0`.

If you have not installed the simulator yet:

```bash
cargo install --git https://github.com/jbrodovsky/strapdown-rs strapdown-sim
```

See [Installation](./installation/installation.md) for the details and the build requirements.

## 1. Generate a trajectory

```console
$ strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42
2026-10-01 21:34:29.872 [INFO] - Generated 6000 synthetic records (600.0 s at 10 Hz)
2026-10-01 21:34:29.880 [INFO] - Sensor records written to synthetic.csv
```

With no other flags, `syn` writes ten minutes of a **stationary** platform at 0° N, 0° E and
zero altitude, sampled at 10 Hz, with a consumer-grade IMU error model (`--imu-grade consumer`)
and a noisy GNSS fix on every row (2.5 m horizontal, 5 m vertical and 0.5 m/s per velocity
axis, one sigma). The barometer is an independent sensor: `relativeAltitude` is computed from a
noisy pressure (about 27 Pa, or 2.24 m, one sigma by default). The flags to
move it, rotate it, change the IMU grade or the sample rate are listed by
`strapdown-sim syn --help`; see [Synthetic Trajectories](./user-guide/synthetic.md).

The file has the same 31 columns as a Sensor Logger export, which is the format every
simulation mode reads:

```text
time,bearingAccuracy,speedAccuracy,verticalAccuracy,horizontalAccuracy,speed,bearing,altitude,longitude,latitude,qz,qy,qx,qw,roll,pitch,yaw,acc_z,acc_y,acc_x,gyro_z,gyro_y,gyro_x,mag_z,mag_y,mag_x,relativeAltitude,pressure,grav_z,grav_y,grav_x
```

The ones the filters use most:

| Columns | Meaning |
| --- | --- |
| `time` | ISO 8601 UTC timestamp, e.g. `2025-01-01T00:00:00.100Z` |
| `acc_x`, `acc_y`, `acc_z` | Specific force in the body frame, m/s², **including** the reaction to gravity |
| `gyro_x`, `gyro_y`, `gyro_z` | Angular rate in the body frame, rad/s |
| `latitude`, `longitude`, `altitude` | GNSS position: degrees, degrees, metres |
| `speed`, `bearing` | GNSS ground speed (m/s) and course (degrees), turned into a north/east velocity fix |
| `horizontalAccuracy`, `verticalAccuracy`, `speedAccuracy` | GNSS one-sigma accuracies, used as the measurement noise |
| `qw`, `qx`, `qy`, `qz` | Attitude quaternion, used for the initial attitude |
| `relativeAltitude` | Barometric altitude change, m |
| `mag_x`, `mag_y`, `mag_z` | Magnetometer, µT, used for heading |

A blank or `NaN` cell means "no measurement on this row". [Input Data
Format](./user-guide/data-format.md) documents every column.

**This file is in the NED frame**, the library default: at rest its accelerometer reads about
−9.8 m/s² on `acc_z`, the down axis. Sensor Logger exports from a phone are ENU and need
`--enu` on every command. Declaring the wrong frame is refused before anything is integrated;
[Coordinate Frames](./user-guide/coordinate-frames.md) shows the error.

## 2. Closed loop: the ESKF

```bash
strapdown-sim cl -i synthetic.csv -o eskf.csv
```

`cl` corrects the inertial solution with GNSS position and velocity, barometric altitude and
magnetometer heading. Its default filter is the error-state Kalman filter. As `strapdown-sim`
configures it, that filter carries **16 error states**: the 15 of position, velocity,
attitude, accelerometer bias and gyroscope bias, plus a barometric altitude bias. The last is
on by default and `--no-estimate-baro-bias` removes it. [State
Representation](./user-guide/state-representation.md) lays the states out.

The log begins:

```text
2026-10-01 21:34:29.887 [INFO] - Running in closed-loop mode with Error-State Kalman Filter (ESKF)
2026-10-01 21:34:29.887 [INFO] - Processing file: synthetic.csv
2026-10-01 21:34:29.903 [INFO] - Read 6000 records from synthetic.csv
2026-10-01 21:34:29.903 [INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.754 m/s^2 against a local gravity of 9.780 m/s^2
2026-10-01 21:34:29.904 [INFO] - Initialized event stream with 13198 events
2026-10-01 21:34:29.904 [INFO] - Initialized ESKF
2026-10-01 21:34:29.904 [INFO] - Starting closed-loop navigation filter with 13198 events
```

followed by a progress line every ten events and, at the end, `Results written to eskf.csv`.
(One line, `Using aiding config: ...`, is omitted above for width.) The progress lines run to
over a thousand on this file. `--log-level warn` silences them, and the commands below use it:

```bash
strapdown-sim --log-level warn cl -i synthetic.csv -o eskf.csv
```

### What came out

Every mode writes the same CSV layout. The first ten columns are the navigation solution; the
last row of `eskf.csv` is:

```text
timestamp,latitude,longitude,altitude,velocity_north,velocity_east,velocity_vertical,roll,pitch,yaw
2025-01-01T00:09:59.900Z,-1.244253386640393e-6,-1.7581270456255539e-6,-0.42194145354743773,0.015484151883163197,0.0568049993997672,0.038121929489373735,0.009803423792142807,0.006982138212381999,-0.002044723575180731
```

Latitude and longitude are in degrees, altitude in metres (positive up), velocities in m/s and
the angles in radians. `velocity_vertical` follows the frame: positive **down** here, because
the run is NED. The remaining 30 columns are the IMU bias estimates, the covariance, and the
optional bias states (`gravity_bias`, `magnetic_bias`, `baro_bias` and their variances). The
map-bias cells are empty on this run because it carried no maps. [Output
Format](./user-guide/output-format.md) lists them all.

### The same seed gives the same file

Runs are deterministic. Running the same command again produces a byte-identical file:

```console
$ strapdown-sim --log-level warn cl -i synthetic.csv -o eskf_again.csv
$ cmp eskf.csv eskf_again.csv && echo identical
identical
```

## 3. The other Kalman filters

```bash
strapdown-sim --log-level warn cl -i synthetic.csv -o ekf.csv --filter ekf
strapdown-sim --log-level warn cl -i synthetic.csv -o ukf.csv --filter ukf
```

The EKF and UKF carry the full navigation state rather than an error state, with the same 15
elements plus the barometric bias. See [Kalman Filters](./filters/kalman.md).

## 4. Dead reckoning

```bash
strapdown-sim --log-level warn dr -i synthetic.csv -o dr.csv --health-speed-mps-max 1000
```

`dr` propagates the IMU from the first record with no aiding at all, so its error grows without
bound. It applies the same health limits as the filters, and on this file the unaided velocity
passes the default 500 m/s ceiling about 580 s in: without `--health-speed-mps-max 1000` the run
stops with `Error: OutOfRange { what: "speed", value: 500.12376354277296, min: 0.0, max: 500.0 }`,
exits 1 and writes no file. (`ol` is a different subcommand, reserved for an open-loop mode that is **not
implemented**: it writes no output. Use `dr` for dead reckoning.)

## 5. The particle filter

```bash
strapdown-sim --log-level warn pf -i synthetic.csv -o pf.csv
```

`pf` runs the Rao-Blackwellized particle filter (100 particles by default, `--num-particles`
to change it). Its output has the same columns; its six IMU-bias columns are zero because the
filter does not estimate IMU biases. See [Rao-Blackwellized Particle Filter](./filters/rbpf.md).

## 6. A GNSS outage

```bash
strapdown-sim --log-level warn cl -i synthetic.csv -o eskf_outage.csv \
  --sched duty --on-s 100 --off-s 50
```

The duty-cycle scheduler withholds GNSS fixes in 50 s windows separated by 100 s of normal
service. With the default `--duty-phase-s 0` the cycle **starts with an outage**: fixes are
withheld over 0-50 s, 150-200 s, 300-350 s and 450-500 s. At the default log level the
difference is visible in the event count: 11199 events here against 13198 for the
uninterrupted run, which is 200 s of 10 Hz GNSS fixes. The barometer and magnetometer are
scheduled separately and keep arriving. [Fault Simulation](./gnss/fault-simulation.md) covers
the other schedulers and the fault models (`--fault degraded|slowbias|hijack`).

## 7. Score the runs

`syn --no-noise` writes the truth for the same trajectory, in the output format:

```bash
strapdown-sim syn -o truth.csv --duration-s 600 --seed 42 --no-noise
```

This short Python script (standard library only) compares each run's horizontal position with
the truth, row by row:

```python
# score.py
import csv, math, sys

R = 6_371_000.0  # mean Earth radius, m; adequate for metre-level errors


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


truth = load("truth.csv")
for path in sys.argv[1:]:
    est = load(path)
    errors = []
    for t, e in zip(truth, est):
        d_north = math.radians(float(e["latitude"]) - float(t["latitude"])) * R
        d_east = (
            math.radians(float(e["longitude"]) - float(t["longitude"]))
            * R
            * math.cos(math.radians(float(t["latitude"])))
        )
        errors.append(math.hypot(d_north, d_east))
    rms = math.sqrt(sum(x * x for x in errors) / len(errors))
    print(f"{path:16s} horizontal RMS {rms:10.3f} m   final {errors[-1]:10.3f} m")
```

```console
$ python3 score.py dr.csv eskf.csv ekf.csv ukf.csv pf.csv eskf_outage.csv
dr.csv           horizontal RMS  41364.109 m   final 109129.025 m
eskf.csv         horizontal RMS      0.524 m   final      0.239 m
ekf.csv          horizontal RMS      0.522 m   final      0.239 m
ukf.csv          horizontal RMS      0.522 m   final      0.240 m
pf.csv           horizontal RMS      0.921 m   final      0.528 m
eskf_outage.csv  horizontal RMS     15.671 m   final      0.240 m
```

Read these as a demonstration of the workflow, not as a comparison of the filters: they come
from one stationary trajectory and one seed. Unaided, the consumer-grade IMU drifts by about
110 km in ten minutes. Every aided run stays at the metre level. The outage run's final error
matches the uninterrupted run's to within a millimetre because the last 100 s have GNSS again,
while its RMS carries the four outages. The filters' measured accuracy on the reference scenarios is on the
[Performance Baselines](./development/performance.md) page.

## 8. The same run from a scenario file

Everything above can be written into one TOML, YAML or JSON file, which is the reproducible
record of an experiment. This is the outage run of step 6:

```toml
# outage.toml: the duty-cycle run above, as a scenario file
mode = "closed-loop"
input = "synthetic.csv"
output = "eskf_outage_from_config.csv"
seed = 42

[logging]
level = "warn"

[closed_loop]
filter = "eskf"

[aiding.scheduler]
kind = "duty_cycle"
on_s = 100.0
off_s = 50.0
start_phase_s = 0.0

[aiding.fault]
kind = "none"
```

```console
$ strapdown-sim --config outage.toml
$ cmp eskf_outage.csv eskf_outage_from_config.csv && echo identical
identical
```

`--config` supplies the whole run, so no subcommand is given. The `[aiding]` section is also
accepted under its older name, `[gnss_degradation]`. Every field has a default, so a misspelled
section name is not an error: it is silently ignored and the run uses defaults. Check a new
file against the keys in [Configuration Files](./user-guide/configuration.md).

## Next steps

- [User Guide Overview](./user-guide/overview.md): the crates, subcommands and filters, and
  where each is documented.
- [The Navigation Model](./user-guide/concepts.md): what the mechanization computes.
- [Example Configurations](./examples/configurations.md): scenario files for every scheduler
  and fault.
- [Using the Library](./user-guide/library.md): the same machinery from Rust.
