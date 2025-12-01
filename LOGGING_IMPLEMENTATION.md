# Logging System Implementation Summary

## Overview

A comprehensive logging system has been implemented for the strapdown-rs project using the Rust `log` crate with `env_logger` as the backend. This provides a Python-like logging experience with configurable log levels and output destinations.

## Changes Made

### 1. Dependencies Added

**strapdown-core** (`core/Cargo.toml`):
- Added `log = "0.4"` for logging macros

**strapdown-sim** (`sim/Cargo.toml`):
- Added `log = "0.4"` for logging macros
- Added `env_logger = "0.11"` for the logging backend
- Added `chrono = "0.4"` for timestamp formatting

**strapdown-geonav** (`geonav/Cargo.toml`):
- Added `log = "0.4"` for logging macros
- Added `env_logger = "0.11"` for the logging backend

### 2. CLI Arguments

Both `strapdown-sim` and `geonav-sim` now support:
- `--log-level <LEVEL>`: Set log level (off, error, warn, info, debug, trace)
  - Default: `info`
- `--log-file <PATH>`: Write logs to a file instead of stderr

### 3. Logger Initialization

Added `init_logger()` function to both executables that:
- Configures the log level based on CLI input
- Sets up custom log formatting with timestamps
- Redirects output to a file if specified
- Uses append mode for log files
- Provides user-friendly error messages for invalid log levels

### 4. Log Message Replacements

All `println!` and `eprintln!` statements were replaced with appropriate log macros:

| Original | Replacement | Use Case |
|----------|-------------|----------|
| `println!("Info...")` | `info!("Info...")` | General informational messages |
| `eprintln!("Error...")` | `error!("Error...")` | Error conditions |
| `eprintln!("Warning...")` | `warn!("Warning...")` | Warning conditions |
| `println!("Debug info...")` | `debug!("Debug info...")` | Detailed debug information |

**Files Updated:**
- `sim/src/main.rs`: All logging statements in the simulation binary
- `core/src/sim.rs`: Library-level logging for simulation functions
- `geonav/src/main.rs`: All logging statements in geophysical navigation binary
- `geonav/src/lib.rs`: Library-level logging for geophysical functions

### 5. Documentation

Created comprehensive documentation:
- `LOGGING.md`: Complete logging usage guide with examples
- Updated `README.md`: Added note about logging capabilities
- Updated `core/README.md`: Mentioned logging in library context

## Log Format

```
YYYY-MM-DD HH:MM:SS.mmm [LEVEL] - message
```

Example:
```
2025-12-01 14:30:45.123 [INFO] - Read 1000 records from data/input.csv
2025-12-01 14:30:50.789 [INFO] - Results written to output.csv
```

## Usage Examples

### Default (info level to stderr)
```bash
strapdown-sim closed-loop -i input.csv -o output.csv
```

### Debug logging to file
```bash
strapdown-sim closed-loop -i input.csv -o output.csv --log-level debug --log-file sim.log
```

### Disable logging
```bash
strapdown-sim closed-loop -i input.csv -o output.csv --log-level off
```

## Benefits

1. **Consistent Output**: All diagnostic messages use a consistent format
2. **Configurable Verbosity**: Users can control the amount of output
3. **File Logging**: Logs can be saved for later analysis
4. **Python-like Experience**: Similar to Python's logging module
5. **Production Ready**: Proper separation of informational, warning, and error messages
6. **Library Support**: Core library functions also use logging for diagnostic output

## Testing

The implementation has been tested with:
- Default settings (info level to stderr)
- Debug level logging to a file
- Logging disabled (off level)
- Various command-line scenarios

All tests passed successfully, and the logging system is working as expected.

## Backward Compatibility

The changes maintain backward compatibility:
- Log output defaults to `info` level, providing similar output to previous versions
- Log messages go to stderr by default, not interfering with CSV output to stdout
- All existing functionality remains intact
- No breaking changes to the API or command-line interface (only additions)
