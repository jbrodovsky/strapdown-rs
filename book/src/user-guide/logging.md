# Logging Guide

This project uses the Rust `log` crate with `env_logger` as the backend, providing a Python-like logging experience.

## Usage

### Command-Line Options

The `strapdown-sim` executable supports the following logging options:

- `--log-level <LEVEL>`: Set the log level (off, error, warn, info, debug, trace)
  - Default: `info`
- `--log-file <PATH>`: Write logs to a file instead of stderr
  - If not specified, logs are written to stderr

### Examples

#### Basic usage with default settings (info level to stderr):
```bash
strapdown-sim cl -i input.csv -o output.csv
```

#### Set log level to debug:
```bash
strapdown-sim cl -i input.csv -o output.csv --log-level debug
```

#### Write logs to a file:
```bash
strapdown-sim cl -i input.csv -o output.csv --log-file simulation.log
```

#### Combine log level and file output:
```bash
strapdown-sim cl -i input.csv -o output.csv --log-level debug --log-file debug.log
```

#### Disable logging:
```bash
strapdown-sim cl -i input.csv -o output.csv --log-level off
```

## Log Levels

The following log levels are available, from least to most verbose:

- **off**: No logging output
- **error**: Only errors that prevent operation
- **warn**: Warnings about potentially problematic situations
- **info**: General informational messages (default)
- **debug**: Detailed information useful for debugging
- **trace**: Very detailed trace information

## Log Format

Log messages are formatted as:
```
YYYY-MM-DD HH:MM:SS.mmm [LEVEL] - message
```

Example:
```
2025-12-01 14:30:45.123 [INFO] - Read 1000 records from data/input.csv
2025-12-01 14:30:45.456 [INFO] - Running particle filter with 100 particles
2025-12-01 14:30:50.789 [INFO] - Results written to output.csv
```

## `RUST_LOG` is not read

This page used to document `RUST_LOG` as an override that `--log-level` took precedence over.
It is not an override and there is nothing to take precedence over: **`RUST_LOG` has no effect
at all.**

`init_logger` (`sim/src/common.rs`) builds the logger with `env_logger::Builder::new()` and
`builder.filter_level(level)`. `Builder::new()` does not consult the environment -- only
`from_env` and `from_default_env` do -- and nothing in the crate calls `parse_env`. Verified:

```console
$ RUST_LOG=off strapdown-sim dr -i input.csv -o output.csv
2026-09-17 23:45:46.856 [INFO] - Running in Dead Reckoning mode with input: input.csv
```

`--log-level` is the only control, and because `filter_level` sets one global level there are
no per-module directives either -- `RUST_LOG=strapdown=debug,strapdown_sim=trace` has no
equivalent here.

## Using Logging in Code

For developers extending the project, use the logging macros from the `log` crate:

```rust
use log::{trace, debug, info, warn, error};

// Informational messages
info!("Processing {} records", count);

// Warnings
warn!("Skipping row {} due to parse error", row_num);

// Errors
error!("Failed to read config file: {}", err);

// Debug information
debug!("UKF state: {:?}", ukf.get_mean());

// Detailed traces
trace!("Entering function with params: {:?}", params);
```

## Python Logger Comparison

This logging system provides a similar experience to Python's logging module:

| Python                          | Rust                                |
|---------------------------------|-------------------------------------|
| `logging.info("message")`       | `info!("message")`                  |
| `logging.warning("message")`    | `warn!("message")`                  |
| `logging.error("message")`      | `error!("message")`                 |
| `logging.debug("message")`      | `debug!("message")`                 |
| `--log-level info`              | `--log-level info`                  |
| Writing to file with FileHandler| `--log-file path/to/file.log`       |

## Notes

- Log files are opened in append mode, so multiple runs will append to the same log file
- Timestamps use the local system timezone
- The logger is initialized once at program startup
- Log messages from the core library (`strapdown-core`) are also captured and formatted consistently
