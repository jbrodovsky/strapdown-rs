# Quick Start

This guide will get you up and running with Strapdown-rs in minutes.

## Installation

Add Strapdown-rs to your `Cargo.toml`:

```bash
cargo add strapdown-core
```

Or install the simulation binary:

```bash
cargo install strapdown-sim
```

## Running Your First Simulation

### 1. Prepare Input Data

Strapdown-sim expects CSV files with IMU and GNSS data. The format follows the [Sensor Logger](https://github.com/tszheichoi/awesome-sensor-logger) app convention:

```csv
timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,latitude,longitude,altitude
1234567890.0,0.01,-0.02,0.03,0.5,-0.2,9.81,45.5,-122.6,100.0
...
```

### 2. Run a Simulation

**Dead Reckoning (Open-Loop)**:
```bash
strapdown-sim open-loop -i data/input.csv -o results/output.csv
```

**Closed-Loop with EKF**:
```bash
strapdown-sim closed-loop -i data/input.csv -o results/output.csv --filter ekf
```

**Closed-Loop with UKF**:
```bash
strapdown-sim closed-loop -i data/input.csv -o results/output.csv --filter ukf
```

**Particle Filter**:
```bash
strapdown-sim particle-filter -i data/input.csv -o results/output.csv --particles 100
```

### 3. View Results

The output CSV contains the estimated navigation state:

```csv
timestamp,latitude,longitude,altitude,velocity_north,velocity_east,velocity_down,roll,pitch,yaw
1234567890.0,45.50001,-122.59999,100.1,1.2,0.5,-0.1,0.01,0.02,90.5
...
```

## Using Configuration Files

For more complex scenarios, use TOML configuration files:

```toml
# config.toml
[simulation]
mode = "closed-loop"
filter = "ekf"

[data]
input_file = "data/trajectory.csv"
output_file = "results/navigation.csv"

[gnss]
enabled = true
dropout_probability = 0.1
update_rate = 1.0

[filter]
use_biases = true
initial_position_uncertainty = 10.0
initial_velocity_uncertainty = 1.0
initial_attitude_uncertainty = 0.1
```

Run with:
```bash
strapdown-sim --config config.toml
```

## Next Steps

- Read the [User Guide](./user-guide/overview.md) for detailed usage instructions
- Explore [Example Configurations](./examples/configurations.md)
- Learn about [Navigation Filters](./filters/kalman.md)
- Check the [API Reference](./api/core.md) for programmatic usage
