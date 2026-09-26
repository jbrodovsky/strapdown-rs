# Geophysical Measurement Configuration

This document describes how to configure geophysical measurements (gravity and magnetic anomaly) in the strapdown-sim configuration wizard.

## Overview

Geophysical measurements are configured in one of two equivalent ways: a `[geophysical]`
section in a scenario file, or the `--geo` flags on the `cl` subcommand. Both resolve to the
same settings and run the same code, so a run is reproducible from either.

```bash
strapdown-sim --config conf/ukf_both.toml        # a scenario file
strapdown-sim cl --geo -i ... --filter ukf ...   # the equivalent flags
```

A closed-loop scenario file carrying a `[geophysical]` section used to be **refused**: the
geophysical runner was reachable only from the command line, so `mode = "closed-loop"` plus
maps was an error and only `mode = "particle-filter"` could carry them. The nine recipes under
`conf/` that describe UKF and EKF geophysical runs are live again as a result.

Two things are worth knowing before writing one:

- **`filter` must be `ukf` or `ekf`.** The ESKF has no geophysical implementation, and it is
  the default -- so a `[geophysical]` section in a file that omits `filter` is refused.
- **At least one of `gravity_resolution` / `magnetic_resolution` must be set.** A
  `[geophysical]` section naming neither configures nothing, and is refused rather than run.

`strapdown-sim config` also writes these sections interactively, and reconciles the two rules
above for you.

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
gravity_resolution = "one_minute"
gravity_bias = 0.0                  # seed of the map-bias state, mGal
gravity_noise_std = 100.0           # measurement noise, mGal
gravity_bias_init_std = 100.0       # prior on the bias state; defaults to gravity_noise_std
gravity_bias_process_noise_std = 1.7  # random walk, mGal per sqrt(s); defaults to the prior over an hour
gravity_map_file = "path/to/gravity_map.nc"  # Optional

# Magnetic anomaly measurements
magnetic_resolution = "one_minute"
magnetic_bias = 0.0
magnetic_noise_std = 150.0
magnetic_bias_init_std = 150.0
magnetic_bias_process_noise_std = 2.5
magnetic_map_file = "path/to/magnetic_map.nc"  # Optional

# Measurement interval in seconds -- a period, not a frequency.
geo_interval_s = 1.0
```

The four `*_bias_init_std` / `*_bias_process_noise_std` keys are the configuration-file
counterparts of `--gravity-bias-init-std` and friends. They matter most for the magnetic
channel: a recording made inside a vehicle carries thousands of nT of the vehicle's own field,
and the bias state is what absorbs it, so the default prior of `magnetic_noise_std` is usually
far too tight.

**The noise defaults are defaults, not measurements.** 100 mGal and 150 nT have never been
compared against the maps they are differenced from. `analyze geostats` in the `analysis`
package measures the residual directly and writes a paste-ready block; run `just geo-stats`.

## Resolution Options

Available map resolutions, finest last. **The config file and the command line spell these
differently** -- a config file takes the `snake_case` form and `--gravity-resolution` /
`--magnetic-resolution` take the `kebab-case` one:

| Resolution | In a config file | On the command line |
|---|---|---|
| One Degree (1°) | `"one_degree"` | `one-degree` |
| Thirty Minutes (30') | `"thirty_minutes"` | `thirty-minutes` |
| Twenty Minutes (20') | `"twenty_minutes"` | `twenty-minutes` |
| Fifteen Minutes (15') | `"fifteen_minutes"` | `fifteen-minutes` |
| Ten Minutes (10') | `"ten_minutes"` | `ten-minutes` |
| Six Minutes (6') | `"six_minutes"` | `six-minutes` |
| Five Minutes (5') | `"five_minutes"` | `five-minutes` |
| Four Minutes (4') | `"four_minutes"` | `four-minutes` |
| Three Minutes (3') | `"three_minutes"` | `three-minutes` |
| Two Minutes (2') | `"two_minutes"` | `two-minutes` |
| **One Minute (1')** - default | `"one_minute"` | `one-minute` |
| Thirty Seconds (30") | `"thirty_seconds"` | `thirty-seconds` |
| Fifteen Seconds (15") | `"fifteen_seconds"` | `fifteen-seconds` |
| Three Seconds (3") | `"three_seconds"` | `three-seconds` |
| One Second (1") | `"one_second"` | `one-second` |

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
