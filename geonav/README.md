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
use geonav::{GeoMap, GeophysicalMeasurementType, GravityResolution};
use geonav::{build_event_stream, geo_closed_loop_ukf};

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
);

// Run simulation
let results = geo_closed_loop_ukf(&mut ukf, events)?;
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
- `geo_closed_loop_ukf` / `geo_closed_loop_ekf` - Simulation functions

## Performance Notes

- **Map Resolution**: Higher resolution maps provide better accuracy but require more memory
- **Typical Noise Levels**:
  - Gravity: 10-100 mGal standard deviation
  - Magnetic: 50-200 nT standard deviation
- **Coverage**: Ensure geophysical maps cover the entire trajectory area
