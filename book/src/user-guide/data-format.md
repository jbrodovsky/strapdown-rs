# Input Data Format

## Overview

The strapdown-rs project uses CSV data files for simulation and testing. The primary data format is compatible with the **Sensor Logger** mobile application, which records IMU, GNSS, and other sensor measurements.

## CSV Column Descriptions

The data files contain the following columns (not all columns may be present in every file):

### Timestamp
* **time** - ISO UTC timestamp of the form `YYYY-MM-DD HH:mm:ss.ssss+HH:MM` (where the `+HH:MM` is the timezone offset)

### GNSS Measurements
* **latitude** - Latitude measurement in degrees
* **longitude** - Longitude measurement in degrees
* **altitude** - Altitude measurement in meters
* **speed** - Speed measurement in meters per second
* **bearing** - Bearing measurement in degrees

### IMU Measurements (Body Frame)
* **acc_x** - Acceleration in the X direction (meters per second squared)
* **acc_y** - Acceleration in the Y direction (meters per second squared)
* **acc_z** - Acceleration in the Z direction (meters per second squared)
* **gyro_x** - Angular velocity around the X axis (radians per second)
* **gyro_y** - Angular velocity around the Y axis (radians per second)
* **gyro_z** - Angular velocity around the Z axis (radians per second)

### Attitude (Orientation)
* **roll** - Roll angle in degrees
* **pitch** - Pitch angle in degrees
* **yaw** - Yaw angle in degrees
* **qw** - Quaternion scalar component
* **qx** - Quaternion X component
* **qy** - Quaternion Y component
* **qz** - Quaternion Z component

### Magnetometer
* **mag_x** - Magnetic field strength in the X direction (micro teslas)
* **mag_y** - Magnetic field strength in the Y direction (micro teslas)
* **mag_z** - Magnetic field strength in the Z direction (micro teslas)

### Barometric Pressure
* **pressure** - Atmospheric pressure measurement (milli bar)
* **relativeAltitude** - Relative altitude measurement (meters)

### Gravity Vector (Computed)
* **grav_x** - Gravitational acceleration in the X direction (meters per second squared)
* **grav_y** - Gravitational acceleration in the Y direction (meters per second squared)
* **grav_z** - Gravitational acceleration in the Z direction (meters per second squared)

## Data Directory Structure

The project organizes data files into several directories based on processing type and GNSS condition:

### Input Data
- **`input/`**: Contains pre-processed input files ready for processing, along with route visualization images. Raw recordings are available upon request but not stored in version control due to size.

### Ground Truth
- **`truth/`**: Contains ground truth data files of processed trajectories used for validation and testing. These files serve as reference data for comparing alternative processing methods and navigation algorithms.

### GNSS Degradation Scenarios

The following directories contain processed trajectory data that simulate various degraded GNSS conditions:

- **`degraded/`**: Simulates degraded GPS conditions with reduced accuracy
- **`spoofed/`**: Simulates GPS spoofing with fixed position offsets to mislead the navigation system
- **`intermittent/`**: Simulates intermittent GPS with periodic outages
- **`combo/`**: Simulates a combination of degraded, spoofed, and intermittent GPS conditions

## Usage

### Loading Data

The `strapdown-core` library provides functions to load CSV data:

```rust
use strapdown_core::sim::load_test_data;

let data = load_test_data("path/to/data.csv")?;
```

### Testing with Degraded GNSS

To test alternative processing methods and navigation algorithms under degraded conditions:

1. Design your navigation algorithm to leverage the available data in the input files
2. Configure your experiment's `GnssDegradationConfiguration` to match the specific characteristics of the degraded condition you want to simulate
3. Process your experiment accordingly

### Example Usage

```bash
# Run closed-loop simulation with input data
strapdown-sim closed-loop -i data/input/trajectory.csv -o results/output.csv

# Test with degraded GNSS
strapdown-sim closed-loop -i data/degraded/trajectory.csv -o results/degraded_output.csv
```

## Data Collection

Data can be collected using the **Sensor Logger** mobile application:
- Available for iOS and Android
- Records synchronized IMU and GNSS data
- Exports in CSV format compatible with this project
- Website: https://www.tszheichoi.com/sensorlogger

## Notes

- Not all columns need to be present in every file
- Missing values are typically represented as empty strings or `NaN`
- The simulation functions will skip rows with invalid or missing critical data (e.g., IMU measurements)
- Timestamps should be monotonically increasing
- IMU data is typically recorded at ~100 Hz
- GNSS data is typically recorded at ~1 Hz
