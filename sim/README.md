# strapdown-sim

Command-line simulator for strapdown inertial navigation systems, built on
[`strapdown-core`](https://crates.io/crates/strapdown-core).

## What it does

- **Dead reckoning / open loop** — propagate IMU data with no aiding.
- **Closed loop** — fuse loosely coupled GNSS, barometric altitude, and
  magnetometer heading through an EKF, UKF, or error-state KF.
- **Particle filtering** — standard and Rao-Blackwellized particle filters.
- **GNSS degradation** — scheduled dropouts, reduced update rates, and injected
  measurement faults, all reproducible from a seed.
- **Synthetic data** — generate IMU, GNSS, and barometric measurements from a
  defined initial kinematic state.

## Install

```sh
cargo install strapdown-sim
```

## Usage

```sh
# Dead reckoning
strapdown-sim -i input.csv -o output.csv open-loop

# Closed loop with a GNSS outage
strapdown-sim -i input.csv -o output.csv closed-loop \
  --seed 42 --dropout-start-s 100.0 --dropout-duration-s 50.0

# Generate a synthetic trajectory
strapdown-sim syn -o synthetic.csv --duration-s 600 --seed 42
```

Scenarios can also be described in a TOML/JSON/YAML file and run with
`--config`. See the [documentation](https://www.strapdown.rs) for the full
configuration schema.

## Experimental features

Geophysical anomaly navigation (gravity and magnetic map aiding) is available
behind the `geonav` feature flag and is considered experimental.

## License

MIT
