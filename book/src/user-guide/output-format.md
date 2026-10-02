# Output Format

Every simulation mode -- `dr`, `cl`, `pf` -- and `syn --no-noise` write the same thing: a CSV
with one `NavigationResult` (`strapdown::sim`) per row, written by `NavigationResult::to_csv`.
The schema is fixed at 40 columns whatever the mode, so files from different estimators line up
column for column; a mode that has nothing to put in a column leaves a defined placeholder rather
than dropping it.

**The command line writes CSV and nothing else.** HDF5, NetCDF and MCAP writers exist as library
methods (see [below](#other-formats-library-only)); no `strapdown-sim` flag or file extension
reaches them. There is no Parquet support at either layer.

## Rows

There is one row per input record, stamped with that record's time in UTC
(`2025-01-01T00:09:59.900Z`). A row stamped $t_k$ holds the estimate after every event at or
before $t_k$ -- the IMU step ending at $t_k$ and any measurement updates applied there -- and none
after. The first row is the initial state, taken from the first record. On the 600 s, 10 Hz
synthetic trajectory used throughout this guide, `dr`, `cl` and `pf` each write 6,000 rows.

## Columns

In file order:

| Column(s) | Unit | Meaning |
|---|---|---|
| `timestamp` | RFC 3339, UTC | epoch of the row |
| `latitude`, `longitude` | degrees | WGS84 position |
| `altitude` | m | height above the WGS84 ellipsoid, positive up in both frames |
| `velocity_north`, `velocity_east` | m/s | horizontal velocity |
| `velocity_vertical` | m/s | vertical velocity: positive **down** for an NED run, positive **up** for an ENU run |
| `roll`, `pitch`, `yaw` | rad | attitude; roll and yaw on $[-\pi, \pi]$, yaw negative west of north |
| `acc_bias_x`, `acc_bias_y`, `acc_bias_z` | m/s² | estimated accelerometer bias, body frame |
| `gyro_bias_x`, `gyro_bias_y`, `gyro_bias_z` | rad/s | estimated gyroscope bias, body frame |
| `latitude_cov`, `longitude_cov` | rad² | position variances |
| `altitude_cov` | m² | altitude variance |
| `latitude_longitude_cov` | rad² | position covariance |
| `latitude_altitude_cov`, `longitude_altitude_cov` | rad·m | position covariances |
| `velocity_n_cov`, `velocity_e_cov`, `velocity_v_cov` | (m/s)² | velocity variances |
| `roll_cov`, `pitch_cov`, `yaw_cov` | rad² | attitude variances |
| `acc_bias_x_cov`, `acc_bias_y_cov`, `acc_bias_z_cov` | (m/s²)² | accelerometer-bias variances |
| `gyro_bias_x_cov`, `gyro_bias_y_cov`, `gyro_bias_z_cov` | (rad/s)² | gyroscope-bias variances |
| `gravity_bias`, `gravity_bias_cov` | mGal, mGal² | gravity-map bias and its variance (geophysical runs only) |
| `magnetic_bias`, `magnetic_bias_cov` | nT, nT² | magnetic-map bias and its variance (geophysical runs only) |
| `baro_bias`, `baro_bias_cov` | m, m² | barometric bias and its variance |

Two things to watch:

- **Position and its covariance are in different units.** `latitude` and `longitude` are
  degrees, but `latitude_cov` and the other position covariances are the filter's raw variances
  in radians. To score a position error against its covariance, convert the error to radians;
  do not convert the variance to degrees.
- **Only the position block is a full covariance.** The six position entries are the whole
  symmetric 3×3 block, which is what a proper NEES needs. Every other `_cov` column is a
  diagonal element only.

## Empty and placeholder cells

The last six columns are optional, and an empty cell means "this run did not estimate that
state", which is a different fact from an estimate of zero.

| Written by | Bias columns | Covariance columns | `gravity_*`, `magnetic_*` | `baro_bias*` |
|---|---|---|---|---|
| `dr` | `0` | `NaN` -- dead reckoning has no covariance | empty | empty |
| `cl` | estimated | estimated | empty unless `--geo` | estimated; empty with `--no-estimate-baro-bias` |
| `pf` | `0`, with `0` variance -- the RBPF has no IMU-bias states | estimated | empty unless `--geo` | empty -- the RBPF has no barometric bias state |
| `syn --no-noise` | `0` | `0` | empty | empty |

## Reading results back

`NavigationResult::from_csv` reads one of these files back into the library type. The example
`core/examples/score_run.rs` uses it to score a run against the truth `syn --no-noise` writes,
with the same metrics the accuracy suite reports:

```bash
strapdown-sim syn -o synthetic.csv       --duration-s 600 --seed 42
strapdown-sim syn -o synthetic_truth.csv --duration-s 600 --seed 42 --no-noise
strapdown-sim cl  -i synthetic.csv -o results/eskf.csv
cargo run -p strapdown-core --example score_run -- results/eskf.csv synthetic_truth.csv
```

The metrics it prints are defined on [Performance Baselines](../development/performance.md#the-metrics).
Any CSV reader works too: the header is a single row of the column names above, and the empty
optional cells read as missing values in pandas or MATLAB.

## Other formats (library only)

`NavigationResult` has three more writers, each with a matching reader, each behind a cargo
feature of `strapdown-core` that is off by default:

| Method | Reader | Feature | Layout |
|---|---|---|---|
| `NavigationResult::to_hdf5` | `from_hdf5` | `hdf5` | group `navigation_results`, one 1-D dataset per column; `timestamp` as RFC 3339 strings |
| `NavigationResult::to_netcdf` | `from_netcdf` | `netcdf` | dimension `time`, one variable per column; `timestamp` as fractional Unix seconds |
| `NavigationResult::to_mcap` | `from_mcap` | `mcap` | channel `navigation_results`, one MessagePack-encoded record per message, logged at the row's timestamp |

All three carry the same 40 columns. HDF5 and NetCDF have no optional type, so an absent bias
column is written as `NaN` and read back as absent. The NetCDF `timestamp` is a double holding
Unix seconds with their fraction, which keeps microsecond resolution; `from_netcdf` rounds it back
to the microsecond, so a 10 Hz run's rows keep distinct times through the round trip.

Enable a writer on the dependency, from git until 1.0 is published:

```toml
[dependencies]
strapdown-core = { git = "https://github.com/jbrodovsky/strapdown-rs", features = ["hdf5"] }
```

`hdf5` and `netcdf` compile libhdf5 and libnetcdf from vendored sources, so they need a C
compiler and cmake 3.26 or newer; `mcap` is Rust only. The `full` feature turns on all three
plus `clap`. See [System Requirements](../installation/requirements.md).

`TestDataRecord`, the input record, has the same three writers and readers, behind the same
features.
