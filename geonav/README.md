# Geonav-Sim Usage Guide

This guide demonstrates how to use the `geonav-sim` program for geophysical navigation simulations.

## Overview

`geonav-sim` extends the basic strapdown navigation simulation by incorporating geophysical measurements (gravity and magnetic anomalies) to aid navigation, particularly in GNSS-denied environments.

## Basic Usage

```bash
# Basic gravity-aided navigation simulation
geonav-sim -i data/input.csv -o results/output.csv --geo-type gravity

# Magnetic-aided navigation simulation with higher resolution
geonav-sim -i data/input.csv -o results/output.csv \
    --geo-type magnetic \
    --geo-resolution two-minutes \
    --geo-noise-std 50.0

# Custom map file specification
geonav-sim -i data/input.csv -o results/output.csv \
    --geo-type gravity \
    --map-file data/custom_gravity_map.nc
```

## Input File Requirements

### CSV Data File
The input CSV must contain standard IMU/GNSS data with the following columns:
- `time` - ISO UTC timestamp (YYYY-MM-DD HH:mm:ss.ssss+HH:MM)
- `latitude`, `longitude`, `altitude` - Position measurements
- `speed`, `bearing` - Velocity measurements  
- `acc_x`, `acc_y`, `acc_z` - Accelerometer data (m/s²)
- `gyro_x`, `gyro_y`, `gyro_z` - Gyroscope data (rad/s)
- `grav_x`, `grav_y`, `grav_z` - Gravity measurements (m/s²) [for gravity-aided navigation]
- `mag_x`, `mag_y`, `mag_z` - Magnetometer data (μT) [for magnetic-aided navigation]

### Geophysical Map Files
Maps should be in NetCDF format with specific naming conventions:

- **Gravity maps**: `{input_name}_gravity.nc`
- **Magnetic maps**: `{input_name}_magnetic.nc`

For example, if your input CSV is `flight_data.csv`, the program will look for:
- `flight_data_gravity.nc` (for gravity-aided navigation)
- `flight_data_magnetic.nc` (for magnetic-aided navigation)

#### Map File Structure
Maps should contain:
- `lat` variable: latitude coordinates (degrees)
- `lon` variable: longitude coordinates (degrees)  
- `z` variable: anomaly data (mGal for gravity, nT for magnetic)

## Command Line Options

### Geophysical Measurement Configuration
- `--geo-type`: Choose between `gravity` or `magnetic` measurements
- `--geo-resolution`: Map resolution (from `one-degree` to `one-second`)
- `--geo-noise-std`: Measurement noise standard deviation
- `--map-file`: Custom map file path (overrides auto-detection)

### GNSS Degradation Configuration
The program inherits all GNSS degradation options from `strapdown-sim`:

#### Scheduler Options (controls GNSS availability)
- `--sched passthrough`: Use all available GNSS measurements
- `--sched fixed --interval-s 10`: GNSS fixes every 10 seconds
- `--sched duty --on-s 30 --off-s 60`: 30s ON, 60s OFF cycles

#### Fault Models (corrupts GNSS measurements)
- `--fault none`: No GNSS corruption
- `--fault degraded`: AR(1) correlated errors
- `--fault slowbias`: Slow-drifting bias
- `--fault hijack`: Position spoofing attack

## Example Scenarios

### 1. GNSS-Denied Navigation with Gravity Aiding
```bash
# Simulate complete GNSS denial with gravity measurements
geonav-sim -i urban_canyon.csv -o gravity_aided.csv \
    --geo-type gravity \
    --geo-noise-std 50.0 \
    --sched fixed --interval-s 999999  # Effectively no GNSS
```

### 2. Intermittent GNSS with Magnetic Aiding
```bash
# GNSS available 20% of the time, magnetic aiding throughout
geonav-sim -i flight_data.csv -o magnetic_aided.csv \
    --geo-type magnetic \
    --geo-resolution five-minutes \
    --sched duty --on-s 10 --off-s 40
```

### 3. GNSS Spoofing with Geophysical Validation
```bash
# Simulate spoofing attack with gravity measurements for validation
geonav-sim -i test_route.csv -o spoofing_test.csv \
    --geo-type gravity \
    --fault hijack --hijack-offset-n-m 100 --hijack-start-s 300
```

### 4. High-Precision Survey with Multiple Aiding
```bash
# High-resolution gravity aiding with minimal GNSS degradation
geonav-sim -i survey_data.csv -o high_precision.csv \
    --geo-type gravity \
    --geo-resolution one-minute \
    --geo-noise-std 10.0 \
    --fault degraded --sigma-pos-m 1.0
```

## Output

The program generates a CSV file with navigation results including:
- Estimated position, velocity, and attitude
- Covariance diagonal elements  
- Input measurements and derived quantities
- Geophysical anomaly values used for aiding

## Performance Considerations

- **Map Resolution**: Higher resolution maps provide better accuracy but require more memory and computation
- **Noise Levels**: Typical values:
  - Gravity: 10-100 mGal standard deviation
  - Magnetic: 50-200 nT standard deviation
- **Coverage**: Ensure geophysical maps cover the entire trajectory area

## Troubleshooting

### Common Issues
1. **Map file not found**: Ensure proper naming convention or use `--map-file`
2. **Out of bounds errors**: Trajectory extends beyond map coverage
3. **Memory issues**: Very high-resolution maps may require significant RAM

### Error Messages
- `"Map file not found"`: Check file naming and paths
- `"Latitude out of bounds"`: Trajectory extends beyond map area
- `"Failed to read config"`: GNSS config file format issues

## Integration with Analysis Tools

Output CSV files are compatible with standard navigation analysis tools and can be processed with:
- Python pandas/numpy for analysis
- MATLAB navigation toolboxes
- R for statistical analysis
- Excel for basic visualization

Example Python analysis:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('output.csv')

# Plot trajectory
plt.figure(figsize=(10, 8))
plt.scatter(results['longitude'], results['latitude'], 
           c=results['timestamp'], cmap='viridis')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title('Geophysical Navigation Trajectory')
plt.colorbar(label='Time')
plt.show()
```