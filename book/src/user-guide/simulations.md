# Running Simulations

`strapdown-sim` is the command-line front end to the library. It reads sensor records from CSV,
runs one of the estimators over them, and writes the navigation solution back out as CSV. This
page is the reference for the binary as a whole: the options every command shares, how input and
output paths are resolved, and the limits that stop a run. Each mode has its own page:

| Subcommand | What it does | Page |
|---|---|---|
| `dr` | Dead reckoning: the IMU alone, from the first record's state | [Dead Reckoning](./dead-reckoning.md) |
| `cl` | Closed loop: the ESKF (default), EKF or UKF, aided by GNSS, barometer and magnetometer | [Closed Loop](./closed-loop.md) |
| `pf` | The Rao-Blackwellized particle filter | [Particle Filter](./particle-filter.md) |
| `syn` | Generate a synthetic trajectory, as sensor records or as truth | [Synthetic Trajectories](./synthetic.md) |
| `config` | Interactive wizard that writes a configuration file | [below](#config-the-configuration-wizard) |
| `ol` | **Not implemented.** Validates its paths, writes nothing, prints `Open-loop mode is not yet fully implemented` | [below](#ol-not-implemented) |

Everything here was derived from the clap definitions in `sim/src/main.rs` and checked against
`strapdown-sim --help` and `strapdown-sim <subcommand> --help` on the 1.0.0 binary. Those help
pages are the authority if this page and the binary ever disagree.

The geophysical flags (`--geo` and the `--gravity-*`/`--magnetic-*` family on `cl` and `pf`)
exist only in a binary built with `--features geonav`; see
[Geophysical Navigation](../geonav/overview.md).

## Two ways to run

A run is described either by a subcommand and its flags, or by a configuration file:

```bash
# Flags
strapdown-sim cl -i synthetic.csv -o results/eskf.csv

# A file
strapdown-sim --config scenario.toml
```

The two are exclusive. When `--config` is given, **the subcommand and every flag that belongs to
it are ignored**, and the file supplies the whole run, including `input`, `output` and the mode
(see [Configuration Files](./configuration.md)). `--config` is a global option, but put it before
any subcommand: `strapdown-sim cl --config x.toml` is parsed as a `cl` invocation first and fails
on the missing `-i`/`-o`.

Run with no subcommand and no `--config`, the binary exits with status 1 and
`Error: No command provided`.

## Global options

These are accepted before or after the subcommand.

| Option | Default | Meaning |
|---|---|---|
| `-c`, `--config <FILE>` | -- | Run entirely from a TOML, YAML or JSON file. The format is chosen by extension (`.toml`, `.yaml`/`.yml`, `.json`). |
| `--log-level <LEVEL>` | `info` | `off`, `error`, `warn`, `info`, `debug` or `trace`. An unrecognized value prints a warning and falls back to `info`. See [Logging](./logging.md). |
| `--log-file <PATH>` | stderr | Append log lines to this file instead of stderr. Parent directories are created. |
| `--parallel` | off | Process the files of a directory input concurrently. Honoured by `dr`, `cl`, `pf` and `--config`. |
| `--plot` | off | Write a PNG performance plot beside each result. Honoured by `dr`, `cl`, `pf` and `--config`. |

`--parallel` and `--plot` work the same way on a subcommand as on the configuration-file path,
where they force `parallel = true` and `generate_plot = true`. `--parallel` matters only for a
directory input; a single file runs as before. `syn`, `config` and `ol` have nothing to plot or
parallelize, and refuse either flag with an error rather than ignoring it.

The plot is drawn for every dead-reckoning, closed-loop and particle-filter result. It lands
beside the CSV with the extension replaced by `.png`, and needs the `plotting` feature, which is on
by default. Without the feature, `--plot` is refused before the run starts; a configuration file's
`generate_plot = true` is logged as an error and the run goes on without the plot. A failure to
draw a plot is logged and does not fail the run.

## Input and output paths

`dr`, `cl`, `pf` and `ol` share the same two required arguments.

| Argument | Accepts |
|---|---|
| `-i`, `--input <PATH>` | A single `.csv` file, or a directory. A directory is searched (not recursively) for files with the extension `csv`, which are processed in sorted name order. |
| `-o`, `--output <PATH>` | A file path ending in `.csv`, or a directory. |

The rules, all from `sim/src/common.rs`:

- **An input file must have the extension `csv`.** Anything else is refused with
  `Input file '...' is not a CSV file.` A directory with no CSV files in it is refused too.
- **Whether `--output` is a file or a directory is decided by its extension.** A path ending in
  `.csv` (any case) names a file; any other path names a directory. An *existing* directory
  always counts as a directory, even one named `results.csv`.
- **Output is CSV only.** There is no flag that selects another format. A path whose extension
  names another data format -- `.h5`, `.hdf5`, `.nc`, `.netcdf`, `.mcap`, `.parquet`, `.json`,
  `.yaml`, `.yml`, `.toml` or `.txt` -- is refused, unless it is an existing directory, rather
  than made into a directory of CSV files. HDF5, NetCDF and MCAP exist only as library methods;
  see [Output Format](./output-format.md).
- **Missing directories are created**: the output directory itself, or the parent of an output
  file.
- **Naming.** With one input file, a file output is written exactly where it points, and a
  directory output gets the input's file name. With several inputs, a directory output gets one
  file per input under the input's name, and a file output `-o run.csv` becomes
  `run_<input stem>.csv` for each input, beside where `run.csv` would be.
- **Results are never written over an input.** If the resolved output path is the input file,
  or any other file this run reads (compared after resolving symlinks and `..`), the run stops.
  So `dr -i batch -o batch` is refused rather than overwriting the recordings.

```console
$ strapdown-sim --log-level warn dr -i batch -o out_file/run.csv
$ ls out_file
run_a.csv  run_b.csv
$ strapdown-sim --log-level warn dr -i batch/a.csv -o batch/a.csv
Error: "Refusing to write results to 'batch/a.csv': that is the input file. Pass a different --output path."
```

**Batches.** When the input is a directory, a file that fails -- because it yields no usable
records, or because its run fails -- is logged and counted, and the rest of the batch still runs.
If any file failed, the command then exits with status 1 and
`Error: "N of M file(s) failed to process"`, on every subcommand and on a `--config` run alike. A
run that fails writes no output file for that input. With a single input file, its own error is
reported.

**Negative numbers** need the `=` form, because clap otherwise reads a leading `-` as a flag:
`--longitude-deg=-75`, not `--longitude-deg -75`.

## Frame declaration

`dr`, `cl`, `pf` and `ol` take `--enu`, which declares the input records to be in the ENU
convention rather than the default NED. Sensor Logger exports are ENU; `syn` output is NED unless
it was generated with `syn --enu`. The declaration is checked against the data before anything is
propagated, and a wrong one is refused with an `InvalidConfiguration` error naming the flag to
change. See [Input Data Format](./data-format.md#frame-convention).

## Execution limits

Every simulation subcommand carries three wall-clock budgets, so a run that hangs or slows to a
crawl ends with an error instead of occupying the machine. Each is disabled by a value of `0` or
less; whichever active one trips first ends the run.

| Flag | Default | Fails the run when |
|---|---|---|
| `--max-wall-clock-ratio <R>` | `0.25` | wall-clock time exceeds `R` times the simulated duration (a 600 s trajectory gets 150 s) |
| `--max-wall-clock-s <S>` | `1200` | wall-clock time for one trajectory exceeds `S` seconds |
| `--max-no-progress-s <S>` | `600` | `S` seconds pass with no event processed |

The failure is a `Timeout` error, and no output file is written for that trajectory.

## Health limits

The health monitor checks the estimate after every event and fails the run when it leaves a
plausible envelope. The defaults are deliberately permissive -- they catch numerical divergence,
not a poor result -- so narrow them to the scenario when you want them to act as a gate.

| Flag | Default | Fails the run when |
|---|---|---|
| `--health-lat-min-deg`, `--health-lat-max-deg` | `-90`, `90` | latitude leaves the band (degrees) |
| `--health-lon-min-deg`, `--health-lon-max-deg` | `-180`, `180` | longitude leaves the band (degrees) |
| `--health-alt-min-m`, `--health-alt-max-m` | `-1e8`, `1e8` | altitude above the ellipsoid leaves the band (metres) |
| `--health-speed-mps-max` | `500` | the magnitude of the full velocity vector, vertical included, exceeds this (m/s) |
| `--health-cov-diag-max` | `1e15` | any variance on the covariance diagonal exceeds this |
| `--nis-pos-max` | `100` | (sets the threshold above which an update's NIS counts as an outlier) |
| `--nis-pos-consec-fail` | `20` | this many outliers occur in a row; one in-bounds update resets the count |

The NIS test applies to every measurement update, barometer and magnetometer included, despite
the `_pos` in the name. It is a run-level divergence check, separate from the
[innovation gate](./closed-loop.md#innovation-gating) that `cl` can install to reject individual
measurements.

```console
$ strapdown-sim --log-level warn cl -i cruise.csv -o hl/out.csv --health-speed-mps-max 10
[ERROR] - Health fail after propagate at 2025-01-01 00:00:00.020 UTC (#0): speed = 49.79808422315322 is outside the valid range [0, 10]
[ERROR] - Error running closed-loop simulation: speed = 49.79808422315322 is outside the valid range [0, 10]
Error: OutOfRange { what: "speed", value: 49.79808422315322, min: 0.0, max: 10.0 }
```

(`cruise.csv` is a 50 m/s synthetic trajectory; see
[Synthetic Trajectories](./synthetic.md#a-moving-trajectory). The speed is not exactly 50 m/s
because the initial velocity is taken from the first record's noisy `speed` and `bearing`.)

Not every limit can apply to every mode, and each flag's `--help` says which modes apply it:

- **`dr` applies the execution limits and the position and speed bounds**, checked after every
  propagation step. It carries no covariance and makes no measurement update, so
  `--health-cov-diag-max` and the NIS pair do not apply to it. An unaided MEMS arc can pass the
  default 500 m/s speed bound on a long recording; raise `--health-speed-mps-max` for those.
- **`pf` applies every limit except the NIS pair**: its loop checks the state and covariance
  bounds but computes no NIS, so `--nis-pos-max` and `--nis-pos-consec-fail` have no effect on it.
- **`ol` runs nothing**, so none of them applies.

## Seeds

Every stochastic part of a run is seeded, and the default seed everywhere is `42`, so the same
command on the same input produces the same output.

| Flag | Seeds |
|---|---|
| `cl --seed` | the GNSS fault models (AR(1) noise, slow-bias random walk) |
| `pf --seed` | the fault models and the particle filter's own sampling |
| `syn --seed` | the IMU, GNSS, barometer and magnetometer noise of the generated trajectory |

`dr` takes no seed: there is nothing random in it. In a configuration file the two roles of `pf
--seed` are separate keys; see [Configuration Files](./configuration.md#seeds).

## `config`: the configuration wizard

`strapdown-sim config` takes no arguments. It asks a series of questions on the terminal -- file
name and directory, mode, input and output paths, seed, frame, parallel processing, log level and
file, the filter (for closed loop), a GNSS scheduler and fault model, and optional geophysical
aiding -- and writes a complete configuration file in the format its extension names. Answer `q`
at any prompt to quit. The wizard needs a terminal: if standard input closes before it has its
answers (`strapdown-sim config < /dev/null`), it says so on stderr and exits with status 1.

The file it writes spells out every section it covers with default values filled in, which makes
it a reasonable starting point to edit by hand:

```console
$ strapdown-sim config
...
✓ Configuration file successfully created: wiz/template.toml

You can now run the simulation with:
  strapdown-sim --config wiz/template.toml
```

The wizard offers dead reckoning, closed loop, particle filter and synthetic generation. It does
not offer open loop, which is not implemented. For synthetic generation it asks only for the
output file (which must end in `.csv`), the seed and the logging, and writes a `[synthetic]`
section at the `syn` defaults for you to edit. When geophysical aiding is enabled on top of the
ESKF, the wizard switches the filter to the UKF, because the ESKF has no geophysical
implementation.

## `ol`: not implemented

`ol` takes the same arguments as `dr`. It checks that the input exists, creates the output
directory, prints `Open-loop mode is not yet fully implemented`, writes no result and exits with
status 0. It is not dead reckoning; use `dr` for that.
