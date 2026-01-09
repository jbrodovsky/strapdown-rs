# Geophysical Measurement Configuration

This document describes how to configure geophysical measurements (gravity and magnetic anomaly) in the strapdown-sim configuration wizard.

## Overview

The `strapdown-sim config` command now supports interactive configuration of geophysical navigation parameters. This allows you to set up simulations that incorporate gravity and/or magnetic anomaly measurements for GNSS-denied navigation.

## Using the Configuration Wizard

When running `strapdown-sim config`, you'll be prompted with a new section for geophysical navigation configuration after the GNSS degradation settings:

### Step 1: Enable Geophysical Navigation

```text
Enable geophysical navigation (gravity/magnetic anomaly measurements)?
  (y)es
  (n)o
Choice:
```

Select 'y' to configure geophysical measurements, or 'n' to skip this section.

### Step 2: Configure Gravity Measurements (Optional)

If enabled, you'll be prompted to configure gravity anomaly measurements:

```text
Enable gravity anomaly measurements?
  (y)es
  (n)o
Choice:
```

If you select 'yes', you'll configure:

- **Map Resolution**: Choose from 15 resolution options (1 degree down to 1 arcsecond)
- **Measurement Bias**: Bias in milliGals (mGal), default is 0.0
- **Noise Standard Deviation**: Measurement noise in mGal, default is 100.0
- **Map File Path**: Path to the gravity map NetCDF file (leave empty for auto-detection)

### Step 3: Configure Magnetic Measurements (Optional)

Similarly, you can configure magnetic anomaly measurements:

```text
Enable magnetic anomaly measurements?
  (y)es
  (n)o
Choice:
```

If you select 'yes', you'll configure:

- **Map Resolution**: Choose from 15 resolution options (1 degree down to 1 arcsecond)
- **Measurement Bias**: Bias in nanoTeslas (nT), default is 0.0
- **Noise Standard Deviation**: Measurement noise in nT, default is 150.0
- **Map File Path**: Path to the magnetic map NetCDF file (leave empty for auto-detection)

### Step 4: Set Measurement Frequency

Finally, specify the frequency for geophysical measurements:

```text
Geophysical measurement frequency (seconds) [auto]:
```

Enter a positive number for the measurement interval in seconds, or press Enter to use automatic frequency.

## Configuration File Format

The geophysical configuration section in the generated TOML file looks like this:

```toml
[geophysical]
# Gravity anomaly measurements
gravity_resolution = "OneMinute"
gravity_bias = 0.0
gravity_noise_std = 100.0
gravity_map_file = "path/to/gravity_map.nc"  # Optional

# Magnetic anomaly measurements
magnetic_resolution = "OneMinute"
magnetic_bias = 0.0
magnetic_noise_std = 150.0
magnetic_map_file = "path/to/magnetic_map.nc"  # Optional

# Measurement frequency (seconds)
geo_frequency_s = 1.0
```

## Resolution Options

Available map resolutions (higher numbers = finer resolution):

1. One Degree (1°)
2. Thirty Minutes (30')
3. Twenty Minutes (20')
4. Fifteen Minutes (15')
5. Ten Minutes (10')
6. Six Minutes (6')
7. Five Minutes (5')
8. Four Minutes (4')
9. Three Minutes (3')
10. Two Minutes (2')
11. **One Minute (1')** - Default
12. Thirty Seconds (30")
13. Fifteen Seconds (15")
14. Three Seconds (3")
15. One Second (1")

## Default Values

- **Gravity noise standard deviation**: 100.0 mGal
- **Magnetic noise standard deviation**: 150.0 nT
- **Measurement bias**: 0.0 (for both)
- **Map resolution**: One Minute
- **Measurement frequency**: Auto-detected from data

## Example Usage

1. Run the configuration wizard:

   ```bash
   strapdown-sim config
   ```

2. Follow the prompts to enable geophysical navigation and configure gravity and/or magnetic measurements.

3. The wizard will generate a configuration file that can be used with:

   ```bash
   strapdown-sim --config your_config.toml
   ```

## Notes

- At least one measurement type (gravity or magnetic) must be enabled if geophysical navigation is activated
- If both map file paths are left empty, the system will attempt to auto-detect them based on the input file location
- The `geonav` feature must be enabled when building strapdown-sim to use these features

## Example Configuration

See [examples/configs/geonav_example.toml](../configs/geonav_example.toml) for a complete example configuration with geophysical measurements.
