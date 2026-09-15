# Strapdown Geonav - Geophysical Navigation Library

This crate provides geophysical navigation capabilities for strapdown inertial navigation systems, enabling gravity and magnetic anomaly aiding for improved navigation accuracy, particularly in GNSS-denied environments.

## Overview

`strapdown-geonav` is a library crate that extends the basic strapdown navigation simulation by incorporating geophysical measurements (gravity and magnetic anomalies) to aid navigation.

**Note**: The standalone `geonav-sim` binary has been consolidated into `strapdown-sim`. Use `strapdown-sim` with the `--features geonav` flag to access geophysical navigation capabilities.

## Usage

### As a Simulation Tool

Build and run with geophysical navigation support:

```bash
# Build strapdown-sim with geonav feature
cargo build --release --package strapdown-sim --features geonav

# Run geophysical navigation simulation
strapdown-sim cl --input data.csv --output out/ \
    --geo \
    --gravity-resolution one-minute \
    --filter ukf

# Magnetic navigation
strapdown-sim cl --input data.csv --output out/ \
    --geo \
    --magnetic-resolution two-minutes \
    --magnetic-noise-std 150.0
```

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
strapdown-geonav = { path = "../geonav" }
```

Use in your code:

```rust
use std::rc::Rc;
use strapdown::sim::run_closed_loop;
use geonav::{GeoMap, GeophysicalMeasurementType, GravityResolution};
use geonav::build_event_stream;

// Load a gravity map
let measurement_type = GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute);
let map = GeoMap::load_geomap("gravity_map.nc", measurement_type)?;

// Build event stream with geophysical measurements
let events = build_event_stream(
    &records,
    &gnss_config,
    Some(Rc::new(map)),
    Some(100.0),  // gravity noise std (mGal)
    None,         // no magnetic map
    None,
    None,
)?;

// Run simulation. The geonav-specific `geo_closed_loop_*` drivers were removed in favour
// of the one driver in `strapdown-core`, which takes any `NavigationFilter`.
let results = run_closed_loop(&mut ukf, events, None, None)?;
```

## Command Line Options

When using `strapdown-sim --features geonav`:

### Enabling Geophysical Navigation
- `--geo`: Enable geophysical navigation mode (required)

### Gravity Configuration
- `--gravity-resolution`: Map resolution (one-degree to one-minute)
- `--gravity-bias`: Measurement bias in mGal
- `--gravity-noise-std`: Noise standard deviation in mGal (default: 100)
- `--gravity-map-file`: Custom map file path

### Magnetic Configuration
- `--magnetic-resolution`: Map resolution (one-degree to two-minutes)
- `--magnetic-bias`: Measurement bias in nT
- `--magnetic-noise-std`: Noise standard deviation in nT (default: 150)
- `--magnetic-map-file`: Custom map file path

### Common Options
- `--geo-frequency-s`: Geophysical measurement frequency in seconds

## Geophysical Map Files

Maps should be in NetCDF format:

- **Gravity maps**: `{input_name}_gravity.nc` or specified via `--gravity-map-file`
- **Magnetic maps**: `{input_name}_magnetic.nc` or specified via `--magnetic-map-file`

### Map File Structure
Maps should contain:
- `lat` variable: latitude coordinates (degrees)
- `lon` variable: longitude coordinates (degrees)
- `z` variable: anomaly data (mGal for gravity, nT for magnetic)

Only those three variables are read, and the whole grid is loaded into memory. `lat` and `lon`
are assumed to be ascending; no CF attributes, scale factors or `_FillValue` handling are
applied.

### Where to get the grids

No map data is vendored in this repository, and there is no fetcher yet (tracked in
[#85](https://github.com/jbrodovsky/strapdown-rs/issues/85)). The grids these tools were
developed against are the GMT remote datasets, most easily retrieved through
[PyGMT](https://www.pygmt.org/) in a separate Python environment -- PyGMT needs the GMT C
library, which is why it is not part of this repository's own toolchain:

```python
import pygmt

# region is [min_lon, max_lon, min_lat, max_lat]; inflate your trajectory's
# bounding box by ~0.25 deg so interpolation near the edges has data.
grav = pygmt.datasets.load_earth_free_air_anomaly("01m", region=region)
mag = pygmt.datasets.load_earth_magnetic_anomaly("02m", region=region)

grav.to_netcdf("<input_stem>_gravity.nc")
mag.to_netcdf("<input_stem>_magnetic.nc")
```

The resolution strings match the `--gravity-resolution` / `--magnetic-resolution` flags (see
the `Display` impls on `GravityResolution` and `MagneticResolution`, which emit the same GMT
tokens). Save the files next to your input CSV under the `{input_stem}_gravity.nc` /
`{input_stem}_magnetic.nc` names above so `strapdown-sim` finds them automatically, or pass
`--gravity-map-file` / `--magnetic-map-file` explicitly.

## Example Scenarios

### GNSS-Denied Navigation with Gravity Aiding
```bash
strapdown-sim cl --input urban_canyon.csv --output out/ \
    --geo --gravity-resolution one-minute \
    --dropout-start-s 0 --dropout-duration-s 999999
```

### Intermittent GNSS with Magnetic Aiding
```bash
strapdown-sim cl --input flight_data.csv --output out/ \
    --geo --magnetic-resolution five-minutes \
    --sched duty --on-s 10 --off-s 40
```

## API Documentation

See the Rust API documentation for detailed information on:
- `GeoMap` - Geophysical map loading and interpolation
- `GravityMeasurement` - Gravity anomaly measurement model
- `MagneticAnomalyMeasurement` - Magnetic anomaly measurement model
- `build_event_stream` - Event stream construction with geophysical measurements
- `strapdown::sim::run_closed_loop` - the simulation driver, which takes any
  `NavigationFilter`. The geonav-specific `geo_closed_loop_ukf` / `geo_closed_loop_ekf` /
  `geo_closed_loop_rbpf` wrappers no longer exist

## Performance Notes

- **Map Resolution**: Higher resolution maps provide better accuracy but require more memory
- **Typical Noise Levels**:
  - Gravity: 10-100 mGal standard deviation
  - Magnetic: 50-200 nT standard deviation
- **Coverage**: Ensure geophysical maps cover the entire trajectory area
