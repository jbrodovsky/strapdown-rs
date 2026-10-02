# Python Analysis Tooling

> **Experimental research tooling.** The `analysis` package is what the project's own
> experiments run on. It is tested in CI, but its interface is not covered by the 1.0 API
> commitments the Rust crates make.

The repository is a two-language monorepo. Beside the Cargo workspace sits `analysis/`, a
Python package (Python 3.13 or newer) that preprocesses recordings into simulator input and
scores simulator output. It is a member of a **uv** workspace declared by the root
`pyproject.toml`; it is not a Cargo member, and nothing in the Rust crates depends on it.

## Setup

From the repository root:

```bash
uv sync                  # creates .venv and installs analysis with its dependencies
uv run analyze --help    # the CLI, through the `analyze` entry point
```

`just setup` runs `cargo fetch` and `uv sync` together.

**GMT is optional.** PyGMT loads the GMT C library when it is imported, so every import of it
is inside the function that uses it, and the package, its CLI and its tests all work without
GMT installed (CI's `python.yml` deliberately installs none). GMT and network access are only
needed to download maps with `analyze preprocess --getmaps`.

## The `analyze` command

| subcommand | module | what it does |
|---|---|---|
| `preprocess` | `preprocess.py` | rebuilds simulator input from Sensor Logger exports: resampling to a fixed rate (`-f`), splitting recordings at IMU dropouts, and with `--getmaps` downloading each trajectory's `_gravity.nc`, `_magnetic.nc` and `_relief.nc` maps; `--synthetic` replaces the phone's gravity and magnetometer readings with simulated dedicated sensors (`synthetic.py`) |
| `dataset-summary` | `__init__.py` | distance travelled and duration of each trajectory |
| `performance` | `compare.py`, `plotting.py` | error statistics, LaTeX tables and plots for a directory of results against the GNSS reference |
| `geoperformance` | `compare.py`, `plotting.py` | the same for geophysically aided runs, against a degraded unaided baseline |
| `compare-filters` | `compare.py` | side-by-side comparison of several result directories |
| `geostats` | `geostats.py` | bias, noise, signal-to-noise and de-correlation length of the geophysical readings against their maps; `--apply-to conf` writes the measured values into the configs |

`uv run analyze <subcommand> --help` lists each one's options. `geostats` mirrors the
anomaly computations in `core/src/earth.rs` and `geonav/src/lib.rs` term for term and checks
that it agrees with them before computing anything; see
[Maps and Measurement Models](../geonav/maps.md#the-python-mirror-of-these-models).

## The `justfile` recipes

The experiment pipeline is a set of [`just`](https://github.com/casey/just) recipes at the
repository root:

| recipe | does |
|---|---|
| `setup` | `cargo fetch` and `uv sync` |
| `build` | `cargo build --release --workspace --all-features` |
| `check-python` | `ruff check`, `ruff format --check` and `pytest`, as CI runs them |
| `preprocess` | rebuilds `data/input` from `data/raw` at 10 Hz, with maps and synthetic geophysical sensors |
| `preprocess-1hz` | the same at 1 Hz |
| `geo-stats` | `analyze geostats` over `data/input`; writes a report and leaves `conf/` alone |
| `geo-adopt` | `analyze geostats --apply-to conf`: rewrites the geophysical configs with measured values |
| `truth`, `degraded`, `ukf-geo`, `ekf-geo`, `rbpf-sim` | run the `conf/` recipes through `strapdown-sim --config` |
| `postprocess` | `dataset-summary` and `performance` over every result directory |
| `geoperf-all`, `geoperf-rbpf` | `geoperformance` for each geophysically aided run against its baseline |
| `clean` | deletes `data/input`, `data/output` and `log` |
| `pipeline` | `clean build preprocess geo-stats truth degraded ukf-geo ekf-geo rbpf-sim postprocess geoperf-all`, in order |

**`data/raw` is not distributed.** The recordings the pipeline starts from are not in the
repository (`data/` is ignored by git), so `just preprocess` and `just pipeline` only run where
those recordings exist. The other subcommands take any directory of simulator input and
output: `performance` and `dataset-summary` were run for this page on a trajectory from
`strapdown-sim syn` and its closed-loop result, matched by file name. `geostats` also needs
the `_gravity.nc` and `_magnetic.nc` maps beside each trajectory.

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run pytest -q
```

These are what `.github/workflows/python.yml` runs, and what `just check-python` wraps. Ruff
is configured in the root `pyproject.toml`.
