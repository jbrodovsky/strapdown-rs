# Summary: Geophysical Measurement Configuration for strapdown-sim

## Changes Made

This update adds interactive configuration support for geophysical measurements (gravity and magnetic anomaly) to the `strapdown-sim config` command.

### 1. Core Library Changes (`core/src/sim.rs`)

#### Added `geophysical` field to `SimulationConfig`
- New optional field: `pub geophysical: Option<GeophysicalConfig>`
- Updated `Default` implementation to include `geophysical: None`
- This allows the main simulation configuration to include geophysical parameters

### 2. CLI Changes (`sim/src/main.rs`)

#### New Prompt Functions
Added five new interactive prompt functions:

1. **`prompt_enable_geophysical()`** - Asks if user wants to enable geophysical navigation
2. **`prompt_geo_resolution(measurement_type: &str)`** - Prompts for map resolution (15 options from 1° to 1")
3. **`prompt_gravity_config()`** - Configures gravity anomaly measurements:
   - Map resolution
   - Measurement bias (mGal)
   - Noise standard deviation (mGal, default: 100.0)
   - Map file path (optional, auto-detect if empty)
4. **`prompt_magnetic_config()`** - Configures magnetic anomaly measurements:
   - Map resolution
   - Measurement bias (nT)
   - Noise standard deviation (nT, default: 150.0)
   - Map file path (optional, auto-detect if empty)
5. **`prompt_geo_measurement_frequency()`** - Sets measurement frequency in seconds

#### Updated `create_config_file()` Function
- Added "Geophysical Navigation Configuration" section
- Calls new prompt functions after GNSS degradation config
- Validates that at least one measurement type is enabled
- Builds and includes `GeophysicalConfig` in final `SimulationConfig`

### 3. Documentation

#### New Documentation File (`docs/GEOPHYSICAL_CONFIG.md`)
Comprehensive guide covering:
- Overview of geophysical configuration
- Step-by-step wizard usage
- Configuration file format
- Available resolution options
- Default values
- Example usage
- Notes and requirements

### 4. Example Configuration (`examples/configs/geonav_example.toml`)
Complete example configuration demonstrating:
- Gravity anomaly measurements setup
- Magnetic anomaly measurements setup
- Integration with closed-loop simulation
- TOML format with comments

### 5. Test Script (`test_geo_config_wizard.sh`)
Automated test script to verify the wizard functionality with simulated user input.

## Features

### User-Friendly Configuration
- Interactive prompts with clear options
- Default values for common parameters
- Input validation with helpful error messages
- Option to quit at any point with 'q'

### Flexible Measurement Configuration
- Enable/disable gravity measurements independently
- Enable/disable magnetic measurements independently
- Must enable at least one type if geophysical navigation is activated
- Auto-detection of map files if not specified

### Resolution Options
15 resolution levels available for both gravity and magnetic maps:
- From coarse (1 degree) to fine (1 arcsecond)
- Default: One Minute (good balance of accuracy and file size)

### Integration with Existing Workflow
- Seamlessly integrated into existing config wizard
- Follows same patterns as other configuration sections
- Compatible with all simulation modes
- Works with existing TOML/JSON/YAML config formats

## Usage Example

```bash
# Run the configuration wizard
strapdown-sim config

# When prompted:
# 1. Enable geophysical navigation: y
# 2. Enable gravity measurements: y
#    - Select resolution: 11 (OneMinute)
#    - Set bias: 0.0
#    - Set noise std: 100.0
#    - Map file: (press Enter for auto-detect)
# 3. Enable magnetic measurements: y
#    - Select resolution: 11 (OneMinute)
#    - Set bias: 0.0
#    - Set noise std: 150.0
#    - Map file: (press Enter for auto-detect)
# 4. Set measurement frequency: 1.0 (seconds)

# Use the generated config
strapdown-sim --config your_config.toml
```

## Configuration File Example

```toml
[geophysical]
gravity_resolution = "OneMinute"
gravity_bias = 0.0
gravity_noise_std = 100.0
magnetic_resolution = "OneMinute"
magnetic_bias = 0.0
magnetic_noise_std = 150.0
geo_frequency_s = 1.0
```

## Benefits

1. **Easier Setup**: No need to manually edit configuration files for geophysical measurements
2. **Reduced Errors**: Input validation prevents common configuration mistakes
3. **Better UX**: Interactive wizard guides users through all options
4. **Documentation**: Clear prompts explain each parameter
5. **Flexibility**: Users can enable only the measurements they need

## Testing

- ✅ Code compiles without errors
- ✅ No clippy warnings in modified files
- ✅ All existing tests pass
- ✅ Markdown documentation passes linting
- ✅ Integration with existing config wizard verified

## Files Modified

1. `/home/james/Code/strapdown-rs/core/src/sim.rs`
2. `/home/james/Code/strapdown-rs/sim/src/main.rs`

## Files Added

1. `/home/james/Code/strapdown-rs/docs/GEOPHYSICAL_CONFIG.md`
2. `/home/james/Code/strapdown-rs/examples/configs/geonav_example.toml`
3. `/home/james/Code/strapdown-rs/test_geo_config_wizard.sh`

## Next Steps

Users can now:
1. Run `strapdown-sim config` to create configurations with geophysical measurements
2. Refer to `docs/GEOPHYSICAL_CONFIG.md` for detailed documentation
3. Use `examples/configs/geonav_example.toml` as a template
4. Test the wizard with `test_geo_config_wizard.sh`
