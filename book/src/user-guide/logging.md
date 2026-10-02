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

Example, from `strapdown-sim pf -i synthetic.csv -o results/pf.csv`:
```
2026-10-01 20:33:18.027 [INFO] - Read 6000 records from synthetic.csv
2026-10-01 20:33:18.028 [INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.754 m/s^2 against a local gravity of 9.780 m/s^2
2026-10-01 20:33:22.501 [INFO] - Results written to results/pf.csv
```

At `info`, a closed-loop run also logs a progress line every few events (position, velocity
and their sigmas), which makes `info` verbose on long runs; `warn` keeps the end-of-run
summaries -- such as how many measurements an innovation gate rejected -- and the per-row
warnings about input rows that could not be parsed.

## `RUST_LOG` is not read

This page used to document `RUST_LOG` as an override that `--log-level` took precedence over.
It is not an override and there is nothing to take precedence over: **`RUST_LOG` has no effect
at all.**

`init_logger` (`sim/src/common.rs`) builds the logger with `env_logger::Builder::new()` and
`builder.filter_level(level)`. `Builder::new()` does not consult the environment -- only
`from_env` and `from_default_env` do -- and nothing in the crate calls `parse_env`. Verified:

```console
$ RUST_LOG=off strapdown-sim dr -i input.csv -o output.csv 2>&1 | head -1
2026-10-01 21:42:29.555 [INFO] - Running in Dead Reckoning mode with input: input.csv
```

`--log-level` is the only control, and because `filter_level` sets one global level there are
no per-module directives either -- `RUST_LOG=strapdown=debug,strapdown_sim=trace` has no
equivalent here.

## Using Logging in Code

For developers extending the project, use the logging macros from the `log` crate:

```rust,ignore
use log::{trace, debug, info, warn, error};

info!("Processing {} records", count);
warn!("Skipping row {} due to parse error", row_num);
error!("Failed to read config file: {}", err);
debug!("filter state: {:?}", state);
trace!("Entering function with params: {:?}", params);
```

The library emits its own messages through the same macros, so anything `strapdown-core` logs
appears in `strapdown-sim`'s output, formatted the same way. A program that uses the library
directly sees those messages only if it installs a `log` backend of its own.

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

## Logging from a configuration file

A configuration file can set the same two things in a `[logging]` section:

```toml
[logging]
level = "debug"
file = "results/run.log"
```

When a run uses `--config`, the command line and the file are combined:

- `--log-file` on the command line overrides `file`.
- `--log-level` overrides `level` **only when it is not `info`**. The flag's default is `info`,
  so passing `--log-level info` explicitly cannot be told apart from not passing it, and the
  file's level wins. To quieten a file that says `debug`, pass `--log-level warn`, not
  `--log-level info`.

## Notes

- Log files are opened in append mode, so multiple runs append to the same file. Missing parent
  directories are created.
- Timestamps use the local system timezone.
- An unrecognized level (`--log-level loud`) prints `Invalid log level 'loud', defaulting to
  'info'` and runs at `info`.
- The logger is initialized once at program startup.
