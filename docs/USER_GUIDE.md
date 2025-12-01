# Strapdown-rs User Guide

This guide covers the use of `strapdown-rs` for researchers, students, and developers working with strapdown inertial navigation systems (INS) and GNSS-denied navigation scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration File Format](#configuration-file-format)
4. [Filter Selection and Tuning](#filter-selection-and-tuning)
5. [GNSS Degradation Scenarios](#gnss-degradation-scenarios)
6. [Dataset Conversion Workflow](#dataset-conversion-workflow)
7. [Output Analysis](#output-analysis)
8. [Example Scenarios](#example-scenarios)
9. [Reproducibility Best Practices](#reproducibility-best-practices)
10. [API Documentation](#api-documentation)

---

## Quick Start

Get up and running with `strapdown-rs` in under 30 minutes.

### Prerequisites

- Rust toolchain (1.91+): Install via [rustup](https://rustup.rs/)
- (Optional) Python 3.12+ for analysis scripts
- (Optional) [pixi](https://pixi.sh/) for environment management

### Step 1: Install the Simulation Binary

```bash
# Install from crates.io (recommended)
cargo install strapdown-sim

# Or build from source
git clone https://github.com/jbrodovsky/strapdown-rs.git
cd strapdown-rs
cargo build --release -p strapdown-sim
```

### Step 2: Prepare Input Data

Create or obtain a CSV file with IMU and GNSS measurements. The expected format matches the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) app output. See [Data Format](#data-format) for details.

### Step 3: Run Your First Simulation

```bash
# Baseline closed-loop simulation (no degradation)
strapdown-sim -i data/input.csv -o results/baseline.csv closed-loop

# With GNSS degradation using a config file
strapdown-sim -i data/input.csv -o results/degraded.csv closed-loop \
  --config examples/configs/gnss_degradation.yaml
```

### Step 4: Analyze Results

The output CSV contains navigation solutions with position, velocity, attitude, and covariance estimates. Use your preferred analysis tool (Python, MATLAB, R, Excel) to compare against ground truth.

---

## Installation

### As a Library

Add `strapdown-core` to your Rust project:

```bash
cargo add strapdown-core
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
strapdown-core = "0.5"
```

### As a Command-Line Tool

```bash
cargo install strapdown-sim
```

### From Source (Development)

```bash
git clone https://github.com/jbrodovsky/strapdown-rs.git
cd strapdown-rs

# Using pixi (recommended for reproducible environments)
pixi run build

# Or using cargo directly
cargo build --workspace --release
```

---

## Configuration File Format

Simulations can be configured using YAML, JSON, or TOML files. The configuration specifies GNSS scheduling (when measurements are available) and fault models (how measurements are corrupted).

### Basic Structure

```yaml
# GNSS scheduling (controls availability)
scheduler:
  kind: passthrough  # or fixed_interval, duty

# Fault model (controls measurement corruption)
fault:
  kind: none  # or degraded, slowbias, hijack

# Random seed for reproducibility
seed: 42
```

### Scheduler Options

#### PassThrough (Default)
All GNSS measurements are delivered at their native rate.

```yaml
scheduler:
  kind: passthrough
```

#### Fixed Interval
Deliver GNSS measurements at fixed intervals (simulates reduced update rate).

```yaml
scheduler:
  kind: fixed_interval
  interval_s: 10.0   # Seconds between updates
  phase_s: 0.0       # Initial phase offset
```

#### Duty Cycle
Alternate between ON (GNSS available) and OFF (GNSS denied) periods.

```yaml
scheduler:
  kind: duty
  on_s: 30.0         # Duration of ON period (seconds)
  off_s: 60.0        # Duration of OFF period (seconds)
  phase_s: 0.0       # Initial phase offset
```

### Fault Model Options

#### None (Baseline)
No measurement corruption.

```yaml
fault:
  kind: none
```

#### Degraded
AR(1)-correlated noise on position and velocity with inflated covariance. Models low-SNR or multipath conditions.

```yaml
fault:
  kind: degraded
  rho_pos: 0.99       # AR(1) correlation for position
  sigma_pos_m: 3.0    # Position noise std dev (meters)
  rho_vel: 0.95       # AR(1) correlation for velocity
  sigma_vel_mps: 0.3  # Velocity noise std dev (m/s)
  r_scale: 5.0        # Measurement covariance inflation factor
```

#### Slow Bias (Soft Spoofing)
Slowly drifting position bias that appears plausible to the filter.

```yaml
fault:
  kind: slowbias
  drift_n_mps: 0.02   # Northward drift rate (m/s)
  drift_e_mps: 0.0    # Eastward drift rate (m/s)
  q_bias: 1e-6        # Random walk PSD (m²/s³)
  rotate_omega_rps: 0.0  # Optional rotation of drift direction
```

#### Hijack (Hard Spoofing)
Apply a constant offset during a specified time window.

```yaml
fault:
  kind: hijack
  offset_n_m: 50.0    # North offset (meters)
  offset_e_m: 0.0     # East offset (meters)
  start_s: 120.0      # Start time (seconds)
  duration_s: 60.0    # Duration (seconds)
```

### Complete Example

```yaml
# examples/configs/gnss_degradation.yaml
scheduler:
  kind: fixed_interval
  interval_s: 10.0
  phase_s: 0.0
fault:
  kind: degraded
  rho_pos: 0.99
  sigma_pos_m: 3.0
  rho_vel: 0.95
  sigma_vel_mps: 0.3
  r_scale: 5.0
seed: 42
```

---

## Filter Selection and Tuning

`strapdown-rs` provides two filter implementations:

### Unscented Kalman Filter (UKF)

The default filter for closed-loop navigation. Best for:
- Gaussian measurement noise
- Real-time performance requirements
- Standard GNSS-aided INS applications

```bash
strapdown-sim -i input.csv -o output.csv closed-loop
```

### Particle Filter

Alternative for non-Gaussian distributions. Best for:
- Highly non-linear measurement models
- Multi-modal state distributions
- Research on filter comparison

```bash
strapdown-sim -i input.csv -o output.csv particle-filter \
  --num-particles 100 \
  --position-std 10.0 \
  --velocity-std 1.0 \
  --attitude-std 0.1 \
  --averaging-strategy weighted-average
```

#### Particle Filter Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-particles` | 100 | Number of particles |
| `--position-std` | 10.0 | Initial position uncertainty (meters) |
| `--velocity-std` | 1.0 | Initial velocity uncertainty (m/s) |
| `--attitude-std` | 0.1 | Initial attitude uncertainty (radians) |
| `--averaging-strategy` | weighted-average | How to compute solution (weighted-average, unweighted-average, highest-weight) |

### Tuning Process Noise

Default process noise values are defined in `DEFAULT_PROCESS_NOISE`. For custom tuning, provide process noise via the API:

```rust
use strapdown::sim::{initialize_ukf, DEFAULT_PROCESS_NOISE};
use nalgebra::{DMatrix, DVector};

// Start with defaults
let mut process_noise = DEFAULT_PROCESS_NOISE.to_vec();

// Increase position noise for lower-quality IMU
process_noise[0] *= 10.0;  // lat
process_noise[1] *= 10.0;  // lon
process_noise[2] *= 10.0;  // alt

let process_noise_matrix = DMatrix::from_diagonal(
    &DVector::from_vec(process_noise)
);
```

---

## GNSS Degradation Scenarios

The simulation supports various GNSS degradation modes for testing navigation algorithm robustness.

### Scenario 1: Complete Outage (Extended GNSS-Denied)

Simulate periods with no GNSS availability:

```yaml
# No GNSS for 5 minutes, then resume
scheduler:
  kind: duty
  on_s: 300.0    # 5 min GNSS available
  off_s: 300.0   # 5 min GNSS denied
  phase_s: 0.0
fault:
  kind: none
seed: 42
```

### Scenario 2: Reduced Update Rate

Simulate degraded GNSS with slower updates:

```yaml
scheduler:
  kind: fixed_interval
  interval_s: 30.0  # Update every 30 seconds
  phase_s: 0.0
fault:
  kind: none
seed: 42
```

### Scenario 3: Degraded Accuracy

Simulate multipath or low-SNR conditions:

```yaml
scheduler:
  kind: passthrough
fault:
  kind: degraded
  rho_pos: 0.99
  sigma_pos_m: 5.0
  rho_vel: 0.95
  sigma_vel_mps: 0.5
  r_scale: 10.0
seed: 42
```

### Scenario 4: Spoofing Attack

Simulate position hijacking:

```yaml
scheduler:
  kind: passthrough
fault:
  kind: hijack
  offset_n_m: 100.0   # Pull 100m north
  offset_e_m: 50.0    # and 50m east
  start_s: 120.0      # Starting at t=2min
  duration_s: 180.0   # For 3 minutes
seed: 42
```

### Scenario 5: Combined Effects

Combine scheduling and fault models:

```yaml
scheduler:
  kind: duty
  on_s: 30.0
  off_s: 30.0
  phase_s: 0.0
fault:
  kind: degraded
  rho_pos: 0.99
  sigma_pos_m: 3.0
  rho_vel: 0.95
  sigma_vel_mps: 0.3
  r_scale: 5.0
seed: 42
```

---

## Dataset Conversion Workflow

### Input Data Format

The expected CSV format matches the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) app:

| Column | Unit | Description |
|--------|------|-------------|
| `time` | ISO 8601 | Timestamp (YYYY-MM-DD HH:mm:ss.ssss+HH:MM) |
| `latitude` | degrees | WGS84 latitude |
| `longitude` | degrees | WGS84 longitude |
| `altitude` | meters | Altitude above ellipsoid |
| `speed` | m/s | Ground speed |
| `bearing` | degrees | Heading |
| `acc_x`, `acc_y`, `acc_z` | m/s² | Accelerometer (body frame) |
| `gyro_x`, `gyro_y`, `gyro_z` | rad/s | Gyroscope (body frame) |
| `roll`, `pitch`, `yaw` | radians | Attitude angles |
| `horizontalAccuracy` | meters | GPS horizontal accuracy |
| `verticalAccuracy` | meters | GPS vertical accuracy |
| `speedAccuracy` | m/s | GPS speed accuracy |

### Converting Third-Party Datasets

Use the preprocessing script to convert other formats:

```bash
# Using pixi
pixi run preprocess --input_dir data/raw/ --output_dir data/input/

# Or directly
python scripts/preprocess.py --input_dir data/raw/ --output_dir data/input/
```

### Creating Test Datasets

Generate navigation solutions for various scenarios:

```bash
pixi run create_dataset --input data/input/ --output-root data/
```

This creates:
- `data/baseline/` - Ground truth from clean GNSS
- `data/degraded/` - Degraded GNSS accuracy
- `data/intermittent/` - Intermittent GNSS availability

---

## Output Analysis

### Output Format

The navigation solution CSV contains:

| Column | Unit | Description |
|--------|------|-------------|
| `timestamp` | ISO 8601 | Solution timestamp |
| `latitude` | degrees | Estimated latitude |
| `longitude` | degrees | Estimated longitude |
| `altitude` | meters | Estimated altitude |
| `velocity_north` | m/s | Northward velocity |
| `velocity_east` | m/s | Eastward velocity |
| `velocity_down` | m/s | Downward velocity |
| `roll`, `pitch`, `yaw` | radians | Estimated attitude |
| `*_cov` | varies | Covariance diagonals |

### Python Analysis Example

```python
import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
import matplotlib.pyplot as plt

# Load results
truth = pd.read_csv('data/baseline/baseline.csv', index_col=0, parse_dates=True)
result = pd.read_csv('results/degraded.csv', index_col=0, parse_dates=True)

# Calculate 2D position error
truth_coords = truth[['latitude', 'longitude']].values
result_coords = result[['latitude', 'longitude']].values
error_m = haversine_vector(truth_coords, result_coords, Unit.METERS)

# Compute statistics
print(f"RMS Error: {np.sqrt(np.mean(error_m**2)):.2f} m")
print(f"Max Error: {np.max(error_m):.2f} m")
print(f"95th Percentile: {np.percentile(error_m, 95):.2f} m")

# Plot error over time
time_s = (result.index - result.index[0]).total_seconds()
plt.figure(figsize=(12, 4))
plt.plot(time_s, error_m)
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Navigation Error Over Time')
plt.grid(True)
plt.savefig('error_analysis.png', dpi=150)
plt.show()
```

### MATLAB Analysis Example

```matlab
% Load data
truth = readtable('data/baseline/baseline.csv');
result = readtable('results/degraded.csv');

% Calculate error using haversine formula
R = 6371000;  % Earth radius in meters
dlat = deg2rad(result.latitude - truth.latitude);
dlon = deg2rad(result.longitude - truth.longitude);
lat1 = deg2rad(truth.latitude);
lat2 = deg2rad(result.latitude);

a = sin(dlat/2).^2 + cos(lat1).*cos(lat2).*sin(dlon/2).^2;
error_m = 2 * R * atan2(sqrt(a), sqrt(1-a));

% Statistics
fprintf('RMS Error: %.2f m\n', rms(error_m));
fprintf('Max Error: %.2f m\n', max(error_m));

% Plot
figure;
plot(error_m);
xlabel('Sample');
ylabel('Position Error (m)');
title('Navigation Error');
grid on;
```

---

## Example Scenarios

### Baseline (No Degradation)

Test filter performance with ideal GNSS:

```bash
strapdown-sim -i data/input.csv -o results/baseline.csv closed-loop \
  --config examples/configs/baseline.yaml
```

Config (`examples/configs/baseline.yaml`):
```yaml
scheduler:
  kind: passthrough
fault:
  kind: none
seed: 42
```

### Simple Dropout

30-second GNSS outage:

```bash
strapdown-sim -i data/input.csv -o results/dropout.csv closed-loop \
  --sched duty --on-s 30 --off-s 30
```

### Extended GNSS-Denied

5-minute outage for testing dead reckoning:

```yaml
scheduler:
  kind: duty
  on_s: 60.0      # 1 min GNSS
  off_s: 300.0    # 5 min denied
  phase_s: 0.0
fault:
  kind: none
seed: 42
```

### Particle Filter Comparison

Compare UKF vs particle filter performance:

```bash
# UKF baseline
strapdown-sim -i data/input.csv -o results/ukf.csv closed-loop \
  --config examples/configs/degraded_5s.yaml

# Particle filter
strapdown-sim -i data/input.csv -o results/pf.csv particle-filter \
  --num-particles 200 \
  --config examples/configs/degraded_5s.yaml
```

---

## Reproducibility Best Practices

### 1. Use Explicit Random Seeds

Always specify a seed for reproducible stochastic behavior:

```yaml
seed: 42  # In config files
```

```bash
strapdown-sim ... --seed 42  # On command line
```

### 2. Version Control Configurations

Store all config files alongside your data:

```
project/
├── data/
│   ├── input/
│   └── results/
├── configs/
│   ├── baseline.yaml
│   ├── experiment_1.yaml
│   └── experiment_2.yaml
└── analysis/
    └── compare_results.py
```

### 3. Record Environment

Capture software versions:

```bash
# Record Rust/cargo version
cargo --version > environment.txt
rustc --version >> environment.txt

# Record strapdown version
strapdown-sim --version >> environment.txt

# If using pixi
pixi list >> environment.txt
```

### 4. Use Configuration Files Over CLI Arguments

For reproducibility, prefer config files:

```bash
# ✓ Reproducible
strapdown-sim -i data.csv -o results.csv closed-loop --config experiment.yaml

# ✗ Harder to reproduce
strapdown-sim -i data.csv -o results.csv closed-loop \
  --sched fixed --interval-s 10 --fault degraded --rho-pos 0.99 ...
```

### 5. Document Dataset Provenance

Record how input data was collected and processed:

```markdown
# data/README.md
## Dataset: Urban Drive 2024-01-15

- Device: iPhone 14 Pro
- App: Sensor Logger v2.4.0
- Duration: 45 minutes
- Route: Downtown loop
- Preprocessing: scripts/preprocess.py v1.2
```

### 6. Statistical Significance

For stochastic experiments (particle filter), run multiple trials:

```bash
for seed in 1 2 3 4 5; do
  strapdown-sim -i data.csv -o "results_${seed}.csv" particle-filter \
    --seed $seed --config experiment.yaml
done
```

---

## API Documentation

### Rust API (rustdoc)

Full API documentation is available via rustdoc:

```bash
cargo doc --open -p strapdown-core
```

Or view online at [docs.rs/strapdown-core](https://docs.rs/strapdown-core).

### Key Types

#### `StrapdownState`
9-state navigation solution: position (lat, lon, alt), velocity (N, E, D), attitude (roll, pitch, yaw).

#### `UnscentedKalmanFilter`
Full-state UKF for loosely-coupled INS/GNSS integration.

#### `ParticleFilter`
Bootstrap particle filter for non-Gaussian estimation.

#### `GnssDegradationConfig`
Configuration for GNSS scheduling and fault injection.

#### `TestDataRecord`
CSV data structure matching Sensor Logger format.

#### `NavigationResult`
Output structure for navigation solutions.

### Example: Custom Simulation

```rust
use strapdown::sim::{TestDataRecord, NavigationResult, initialize_ukf, closed_loop};
use strapdown::messages::{GnssDegradationConfig, build_event_stream};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load input data
    let records = TestDataRecord::from_csv("data/input.csv")?;
    
    // Load configuration
    let config = GnssDegradationConfig::from_yaml("config.yaml")?;
    
    // Build event stream
    let events = build_event_stream(&records, &config);
    
    // Initialize filter
    let mut ukf = initialize_ukf(records[0].clone(), None, None, None, None, None, None);
    
    // Run simulation
    let results = closed_loop(&mut ukf, events)?;
    
    // Save results
    NavigationResult::to_csv(&results, "output.csv")?;
    
    Ok(())
}
```

---

## Further Resources

- [GitHub Repository](https://github.com/jbrodovsky/strapdown-rs)
- [API Documentation (docs.rs)](https://docs.rs/strapdown-core)
- [LOGGING.md](../LOGGING.md) - Logging configuration
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contributing guidelines

### Reference Text

The implementation follows:

> Groves, P.D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd Edition. Artech House.

Equations are referenced by chapter and equation number in code comments.
