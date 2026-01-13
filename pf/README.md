# Geophysical Particle Filter (pf)

A Python package for velocity-based particle filter navigation using geophysical anomaly measurements (gravity and magnetic fields) for map-matching.

## Overview

This package implements a velocity-based particle filter that provides alternative positioning estimates in GNSS-denied scenarios. The filter:

1. **Propagates particles** using velocity estimates from an INS/GNSS navigation solution
2. **Updates particle weights** based on geophysical anomaly measurements matched against pre-loaded maps
3. **Provides position estimates** through weighted particle averaging

This is a prototype/research tool designed to complement the `strapdown-geonav` Rust crate.

## Installation

```bash
cd pf
uv sync
```

For development dependencies:
```bash
uv sync --all-extras
```

## Usage

### Command-Line Interface

The package provides a CLI tool `pf-sim` for running simulations:

```bash
pf-sim --ins-dir ./data/ins --imu-dir ./data/raw \
       --output-dir ./results --geo-type gravity \
       --num-particles 1000 --seed 42
```

#### Required Arguments

- `--ins-dir`: Directory containing INS solution CSV files
- `--imu-dir`: Directory containing IMU/GNSS CSV files
- `--output-dir`: Directory for output results
- `--geo-type`: Type of geophysical measurement (`gravity`, `magnetic`, or `combined`)

#### Map Options

Maps are loaded from PyGMT's earth datasets:
- Gravity: Earth free-air anomaly grid
- Magnetic: World Digital Magnetic Anomaly Map (WDMAM)

- `--map-region`: Region bounds for map loading in degrees (west, east, south, north). If not specified, derived from input data with 0.1 degree padding.
- `--gravity-resolution`: Resolution for gravity maps (default: `01m` = 1 arc-minute)
- `--magnetic-resolution`: Resolution for magnetic maps (default: `03m` = 3 arc-minutes)

#### Filter Options

- `--num-particles`: Number of particles (default: 1000)
- `--gravity-noise-std`: Gravity measurement noise std in mGal (default: 100.0)
- `--magnetic-noise-std`: Magnetic measurement noise std in nT (default: 150.0)
- `--position-jitter-std`: Position process noise std in degrees (default: 0.0001)
- `--altitude-jitter-std`: Altitude process noise std in meters (default: 5.0)
- `--resampling-strategy`: Resampling strategy (`systematic`, `stratified`, `residual`)
- `--averaging-strategy`: State estimation strategy (`mean`, `weighted_mean`, `map`)
- `--seed`: Random seed for reproducibility

### Python API

```python
from pf import (
    GeophysicalParticleFilter,
    ParticleFilterConfig,
    GeoMap,
    GeophysicalMeasurementType,
    NavigationState,
    GeophysicalMeasurement,
)

# Load gravity map from PyGMT (1 arc-minute resolution)
gravity_map = GeoMap.load_gravity(
    region=(-74, -73, 40, 41),  # (west, east, south, north)
    resolution="01m"
)

# Or load magnetic map from PyGMT (3 arc-minute resolution, WDMAM data)
magnetic_map = GeoMap.load_magnetic(
    region=(-74, -73, 40, 41),
    resolution="03m",
    data_source="wdmam"
)

# Or create from an existing xarray DataArray
# gravity_map = GeoMap.from_dataarray(my_dataarray, GeophysicalMeasurementType.GRAVITY)

# Create filter configuration
config = ParticleFilterConfig(
    num_particles=1000,
    measurement_type=GeophysicalMeasurementType.GRAVITY,
    gravity_noise_std=100.0,
    seed=42
)

# Initialize filter
pf = GeophysicalParticleFilter(config, gravity_map=gravity_map)
pf.initialize((40.5, -73.5, 100.0))  # (lat, lon, alt)

# Run filter step
nav_state = NavigationState(
    timestamp=1234567890.0,
    latitude=40.5,
    longitude=-73.5,
    altitude=100.0,
    velocity_north=5.0,
    velocity_east=3.0,
    velocity_vertical=0.0,
)

measurement = GeophysicalMeasurement(
    timestamp=1234567890.0,
    gravity_x=0.1,
    gravity_y=0.1,
    gravity_z=9.8,
)

est_lat, est_lon, est_alt = pf.step(nav_state, measurement, dt=1.0)

# Get state covariance
cov = pf.get_covariance()

# Get effective sample size
n_eff = pf.get_effective_sample_size()
```

## Data Format

### INS Input CSV

Expected columns:
- `timestamp`: ISO format datetime or Unix timestamp
- `latitude`: Latitude in degrees
- `longitude`: Longitude in degrees
- `altitude`: Altitude in meters
- `velocity_north`: Northward velocity in m/s
- `velocity_east`: Eastward velocity in m/s
- `velocity_vertical`: Vertical velocity in m/s (positive up)

### IMU/GNSS Input CSV

Expected columns:
- `time`: ISO format datetime or Unix timestamp
- `latitude`, `longitude`, `altitude`: Position (for ground truth)
- `speed`, `bearing`: Speed and heading
- `grav_x`, `grav_y`, `grav_z`: Gravity vector components (m/s^2)
- `mag_x`, `mag_y`, `mag_z`: Magnetic field components (uT)

## Algorithm Details

### Prediction Step

Particles are propagated using simplified local-level frame position update equations:

```
dlat = v_north * dt / R_meridian
dlon = v_east * dt / (R_transverse * cos(lat))
dalt = v_vertical * dt
```

Process noise is added to each particle position.

### Measurement Update

For gravity measurements:
1. Compute observed gravity anomaly using Eötvös correction
2. Look up predicted anomaly from map at each particle position
3. Compute likelihood using Gaussian measurement model
4. Update particle weights

For magnetic measurements:
1. Compute observed magnetic anomaly (field magnitude)
2. Look up predicted anomaly from map
3. Compute likelihood and update weights

### Resampling

Resampling is triggered when effective sample size falls below threshold:
- Systematic resampling (default)
- Stratified resampling
- Residual resampling

### State Estimation

- Weighted mean (default): Sum of weighted particle positions
- Simple mean: Unweighted average
- MAP: Maximum a posteriori (highest weight particle)

## References

- Groves, P.D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*. 2nd Edition.
- Geophysical anomaly calculations follow `geonav/src/lib.rs` in the strapdown-rs repository.

## License

MIT License - See repository root for details.
