# Geophysical Navigation Overview

> **Experimental.** The `strapdown-geonav` crate is held at version 0.1 so that its API can
> still change, and its results are research results, not a validated product. Everything on
> this page describes what the code does today.

Geophysical navigation aids an inertial solution with a quantity the Earth itself provides: the
local **gravity anomaly** or the local **magnetic anomaly**. Both vary with position and are
published as global grids. A sensor on the vehicle measures the anomaly, the filter reads the
same quantity off a map at its estimated position, and the difference corrects the position
estimate. Unlike GNSS, nothing has to be received, so there is nothing to jam or spoof. The
price is that a map measurement is a highly nonlinear, often ambiguous function of position,
and that the signal (tens of milligal, tens to hundreds of nanotesla) is small against the
sensor errors of anything cheap.

The work this module supports is described in two Institute of Navigation papers, one using the
UKF and one the particle filter (see [Publications and Links](../resources/publications.md)):

- J. Brodovsky and P. Dames, "Navigation in GNSS-denied environments using MEMS-grade sensors
  and geophysical anomalies: A UKF approach," *ION ITM 2026*, pp. 155–164,
  [doi:10.33012/2026.20508](https://doi.org/10.33012/2026.20508).
- J. Brodovsky and P. Dames, "Navigation in GNSS-Denied Environments Using MEMS-Grade Sensors
  and Geophysical Anomalies: A Particle Filter Approach," *ION Pacific PNT 2026*, pp. 399–410,
  [doi:10.33012/2026.20618](https://doi.org/10.33012/2026.20618).

[Maps and Measurement Models](./maps.md) has the measurement equations, the bias states and
where the maps come from.

## Building

Geophysical aiding is a feature of `strapdown-sim`, off by default because it compiles libnetcdf
(and with it HDF5) from source, which needs a C compiler and cmake 3.26 or newer:

```bash
cargo build --release -p strapdown-sim --features geonav
# or, to install the binary
cargo install --git https://github.com/jbrodovsky/strapdown-rs strapdown-sim --features geonav
```

A binary built without the feature has no `--geo` flag at all. (There used to be a separate
`geonav-sim` binary; it was folded into `strapdown-sim`.)

## Which filters

| mode | filter | geophysical aiding |
|---|---|---|
| `cl --filter ukf` | UKF | yes |
| `cl --filter ekf` | EKF | yes |
| `cl` (default `--filter eskf`) | ESKF | **no**: the run stops with "ESKF is not yet implemented for geophysical navigation" |
| `pf` | Rao-Blackwellized particle filter | yes |
| `dr` | dead reckoning | no |

Because the ESKF is the closed-loop default, `cl --geo` must name `--filter ukf` or
`--filter ekf`, and a config file must set `filter` in `[closed_loop]`.

## Command-line flags

All of these require `--geo`, and at least one of `--gravity-resolution` or
`--magnetic-resolution` must be given; a run with `--geo` and neither stops with "geophysical
navigation needs at least one map". The same flags are available on `cl` and `pf`.

| flag | unit | default | meaning |
|---|---|---|---|
| `--geo` | -- | off | enable geophysical aiding |
| `--gravity-resolution` | -- | unset | enable gravity aiding; one of `one-degree` ... `one-second` (see below) |
| `--gravity-map-file` | path | `<input stem>_gravity.nc` | gravity anomaly map |
| `--gravity-noise-std` | mGal | 100 | gravity measurement noise σ |
| `--gravity-bias` | mGal | 0 | initial value of the gravity map-bias state |
| `--gravity-bias-init-std` | mGal | `--gravity-noise-std` | initial σ of the gravity map-bias state |
| `--gravity-bias-process-noise-std` | mGal/√s | init σ / √3600 | random-walk rate of the gravity map bias |
| `--magnetic-resolution` | -- | unset | enable magnetic aiding; same choices |
| `--magnetic-map-file` | path | `<input stem>_magnetic.nc` | magnetic anomaly map |
| `--magnetic-noise-std` | nT | 150 | magnetic measurement noise σ |
| `--magnetic-bias` | nT | 0 | initial value of the magnetic map-bias state |
| `--magnetic-bias-init-std` | nT | `--magnetic-noise-std` | initial σ of the magnetic map-bias state |
| `--magnetic-bias-process-noise-std` | nT/√s | init σ / √3600 | random-walk rate of the magnetic map bias |
| `--geo-interval-s` | s | every record | seconds **between** geophysical measurements (alias `--geo-frequency-s`) |

The particle filter adds four more, for the temporal-variation part of its map-bias model (see
[Rao-Blackwellized Particle Filter](../filters/rbpf.md)): `--gravity-variation-std`,
`--gravity-variation-time-constant-s` (default 300 s), `--magnetic-variation-std` and
`--magnetic-variation-time-constant-s` (default 300 s). The two standard deviations default to
values derived from the corresponding `--*-bias-process-noise-std`.

The 100 mGal and 150 nT noise defaults are placeholders; nothing has measured them against
these maps. Set the noise and the bias prior for your sensor. `--magnetic-bias-init-std` in
particular wants to be large for a magnetometer inside a vehicle, whose own field the bias
state exists to absorb.

**The resolution flags are labels, not selectors.** The grid used is whatever the map file
contains; the resolution is recorded with the map but changes nothing numerically. Gravity
labels stop at `one-minute` and magnetic labels at `two-minutes`, and finer choices are
recorded as those.

## Configuration file

The same settings, in snake case, go in a `[geophysical]` section. This file ran as written
against a synthetic trajectory with map files beside it:

```toml
input = "synthetic.csv"
output = "results/geo_cfg.csv"
mode = "closed-loop"

[closed_loop]
filter = "ekf"

[geophysical]
gravity_resolution = "one_minute"
gravity_noise_std = 50.0
gravity_bias_init_std = 100.0
magnetic_resolution = "two_minutes"
magnetic_noise_std = 20.0
magnetic_bias_init_std = 500.0
geo_interval_s = 5.0

[aiding.scheduler]
kind = "duty_cycle"
on_s = 100.0
off_s = 200.0
start_phase_s = 0.0
```

| key | CLI counterpart |
|---|---|
| `gravity_resolution`, `magnetic_resolution` | `--gravity-resolution`, `--magnetic-resolution` (values in snake case: `one_minute`, `two_minutes`, ...) |
| `gravity_map_file`, `magnetic_map_file` | `--gravity-map-file`, `--magnetic-map-file` |
| `gravity_noise_std`, `magnetic_noise_std` | `--gravity-noise-std`, `--magnetic-noise-std` |
| `gravity_bias`, `magnetic_bias` | `--gravity-bias`, `--magnetic-bias` |
| `gravity_bias_init_std`, `magnetic_bias_init_std` | `--gravity-bias-init-std`, `--magnetic-bias-init-std` |
| `gravity_bias_process_noise_std`, `magnetic_bias_process_noise_std` | `--gravity-bias-process-noise-std`, `--magnetic-bias-process-noise-std` |
| `geo_interval_s` (alias `geo_frequency_s`) | `--geo-interval-s` |

The particle filter's variation settings go in `[particle_filter]` as
`gravity_variation_std`, `gravity_variation_time_constant_s`, `magnetic_variation_std` and
`magnetic_variation_time_constant_s`. The recipes `conf/{ukf,ekf,rbpf}_{grav,mag,both}.toml`
are complete worked examples, with the source of every number in their comments.

## How maps are found

Unless a map file is given explicitly, `strapdown-sim` looks for it **beside the input CSV,
named after it**: for `data/input/drive.csv` it opens `data/input/drive_gravity.nc` and
`data/input/drive_magnetic.nc`. A directory input is processed file by file, so every
trajectory needs its own pair. A missing map stops the run with
`Gravity map file not found: <path>` (or the magnetic equivalent).

A map is a NetCDF grid with one-dimensional `lat` and `lon` variables in degrees and a
two-dimensional `z` variable of anomaly values, gravity in milligal and magnetic in nanotesla,
which is the layout GMT writes. `analyze preprocess --getmaps` in the Python tooling downloads
both maps for each trajectory and writes them under exactly these names; see
[Maps and Measurement Models](./maps.md#where-the-maps-come-from).

## Output

The output CSV always carries `gravity_bias`, `gravity_bias_cov`, `magnetic_bias` and
`magnetic_bias_cov` columns (see [Output Format](../user-guide/output-format.md)). For each
channel a geophysically aided run uses, they hold the filter's estimate of the map bias, in
the map's unit, and its variance; for a channel the run does not use, they are `NaN`.

## Example

The command below ran against `strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42`,
with `synthetic_gravity.nc` and `synthetic_magnetic.nc` beside it. Those two maps were written
by a short Python script as a smooth artificial field covering the track: `syn` does not
produce maps, and its `grav_*` and `mag_*` columns are not sampled from any map, so this shows
the mechanics of a run, not geophysical navigation performance.

```bash
strapdown-sim cl -i synthetic.csv -o geo_both.csv --filter ekf \
  --geo --gravity-resolution one-minute --magnetic-resolution two-minutes \
  --gravity-noise-std 50 --magnetic-noise-std 20 --magnetic-bias-init-std 500

strapdown-sim pf -i synthetic.csv -o geo_pf.csv \
  --geo --gravity-resolution one-minute --geo-interval-s 5 --num-particles 200
```

## Status and limitations

- **Experimental API.** `strapdown-geonav` is 0.1 and may change in any release.
- **No ESKF.** Only the UKF, the EKF and the particle filter accept geophysical measurements.
- **Leaving the map.** With the EKF and the particle filter, a measurement whose estimate is
  off the map tile is skipped with a warning, so a map that is too small quietly stops aiding.
  With the UKF, an off-map sigma point currently stops the run with
  `NonFinite { what: "filter state" }`. Pad the map well beyond the track either way; see
  [Maps and Measurement Models](./maps.md#reading-a-value-bilinear-interpolation).
- **The sensor matters more than the filter.** The gravity observation is the magnitude of the
  record's `grav_x/y/z` columns and the magnetic observation the magnitude of `mag_x/y/z`.
  Phone recordings from Sensor Logger are poor sources for both: the magnitude of the `grav_*`
  channel is pinned to a device constant, so it carries no gravity information, and the
  magnetometer is dominated by the vehicle's own field. The project's own experiments therefore
  replace those two channels with simulated readings of dedicated low-cost sensors
  (`analyze preprocess --synthetic`, documented in `analysis/src/analysis/synthetic.py`).
- **The noise defaults are not measurements.** `analyze geostats` in the
  [Python tooling](../development/analysis.md) measures bias, noise and signal-to-noise for a
  data set against its maps.
- **Relief maps are not used.** `GeoMap` can represent a terrain-relief grid, and the
  preprocessing downloads one for plotting, but no measurement model consumes it.
