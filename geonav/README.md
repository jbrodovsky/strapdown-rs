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

# Magnetic navigation. The default filter, the ESKF, has no geophysical arm: pick the
# UKF or the EKF.
strapdown-sim cl --input data.csv --output out/ \
    --geo \
    --magnetic-resolution two-minutes \
    --magnetic-noise-std 150.0 \
    --filter ekf
```

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
strapdown-geonav = { path = "../geonav" }
```

Use in your code. `build_event_stream` takes the records, the aiding configuration, the
records' frame and a `GeophysicalAiding` describing the maps:

```rust
use std::path::Path;
use std::rc::Rc;

use geonav::{
    GeoMap, GeophysicalAiding, GeophysicalMeasurementType, GravityResolution, build_event_stream,
};
use strapdown::messages::{AidingConfig, EventStream};
use strapdown::sim::TestDataRecord;

fn gravity_aided_events(records: &[TestDataRecord]) -> Result<EventStream, Box<dyn std::error::Error>> {
    let map = GeoMap::load_geomap(
        Path::new("gravity_map.nc"),
        GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute),
    )?;
    let geophysical = GeophysicalAiding {
        gravity_map: Some(Rc::new(map)),
        gravity_noise_std: Some(100.0), // mGal
        interval_s: Some(10.0),         // one gravity measurement every 10 s
        // No magnetic map, and a filter that carries no map-bias state.
        ..GeophysicalAiding::default()
    };
    Ok(build_event_stream(records, &AidingConfig::default(), false, &geophysical)?)
}
```

Feed the stream to any `NavigationFilter` through `strapdown::sim::run_closed_loop_with_geo`.
A filter that estimates the map bias also needs a `GeoBiasLayout` in
`GeophysicalAiding::bias_layout` and the matching `ExtraStateLayout` on the runner; the rustdoc
for both, and `strapdown-sim`'s geophysical runner (`run_geo_closed_loop_file` in
`sim/src/main.rs`), show the whole arrangement.

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
- `--geo-interval-s`: Seconds *between* geophysical measurements (`--geo-frequency-s` is
  accepted as an alias)

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
from pathlib import Path

import pandas as pd
import pygmt

# The trajectory strapdown-sim will be run on. The maps have to cover it.
input_csv = Path("data/input/flight.csv")
track = pd.read_csv(input_csv)

# PyGMT wants [min_lon, max_lon, min_lat, max_lat]. Pad the trajectory's bounding box so
# interpolation near the edges still has data on both sides.
pad = 0.25  # degrees
region = [
    track["longitude"].min() - pad,
    track["longitude"].max() + pad,
    track["latitude"].min() - pad,
    track["latitude"].max() + pad,
]

grav = pygmt.datasets.load_earth_free_air_anomaly("01m", region=region)
mag = pygmt.datasets.load_earth_magnetic_anomaly("02m", region=region)

# These are the names strapdown-sim looks for next to the input CSV.
stem = input_csv.with_suffix("")
grav.to_netcdf(f"{stem}_gravity.nc")
mag.to_netcdf(f"{stem}_magnetic.nc")
```

A fixed region works just as well if you already know the area -- for example
`region = [-76.0, -75.0, 39.5, 40.5]` for the Philadelphia area.

The resolution strings match the `--gravity-resolution` / `--magnetic-resolution` flags (see
the `Display` impls on `GravityResolution` and `MagneticResolution`, which emit the same GMT
tokens). Writing the files beside the input CSV under those names is what lets `strapdown-sim`
find them automatically; otherwise pass `--gravity-map-file` / `--magnetic-map-file`
explicitly.

## Example Scenarios

### GNSS-Denied Navigation with Gravity Aiding
```bash
# One GNSS fix at t = 0 and none after it.
strapdown-sim cl --input urban_canyon.csv --output out/ \
    --geo --gravity-resolution one-minute --filter ukf \
    --sched fixed --interval-s 1000000
```

### Intermittent GNSS with Magnetic Aiding
```bash
strapdown-sim cl --input flight_data.csv --output out/ \
    --geo --magnetic-resolution five-minutes --filter ukf \
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
