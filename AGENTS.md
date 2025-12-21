# Repository Guidelines

## Project Structure & Module Organization
This is a Cargo workspace with three primary crates plus supporting data/scripts.
- `core/`: `strapdown-core` library (INS algorithms, filters, simulation utilities).
- `sim/`: `strapdown-sim` CLI for open/closed-loop runs and GNSS degradation.
- `geonav/`: experimental geophysical navigation module.
- `data/`: sample datasets and scenarios (e.g., `data/input/*.csv`).
- `docs/`, `papers/`, `spec.md`: design notes and research docs.
- `scripts/`, `examples/`: helper workflows and usage examples.

## Build, Test, and Development Commands
Use Pixi when available; Cargo works directly too.
- `pixi run build` / `cargo build --workspace --release`: build all crates.
- `cargo test --workspace`: run all tests.
- `cargo test --package strapdown-core`: test a single crate.
- `pixi run lint`: run Rust clippy and Python ruff.
- `pixi run fmt`: run rustfmt and Python formatting.
- `pixi run coverage` / `cargo tarpaulin --workspace --timeout 600`: coverage.
- Example run: `./target/release/strapdown-sim -i data/input/input.csv -o output.csv open-loop`.

## Coding Style & Naming Conventions
- Rust formatting via rustfmt (4-space indentation); keep functions focused and small.
- Naming: `snake_case` for functions/vars, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Prefer descriptive names over symbols; add Rust doc comments (`///`) and cite Groves equations when relevant.
- Use `assert_approx_eq` for floating-point comparisons in tests.

## Testing Guidelines
- Unit tests live alongside modules; integration tests live in `core/tests/integration_tests.rs`.
- Tests should be deterministic; seed RNGs when applicable.
- Name test functions in `snake_case` and keep fixtures minimal.

## Commit & Pull Request Guidelines
- Commit subjects are short, imperative, and plain (e.g., "Update RBPF documentation..."). Use `Fixes #123` when closing issues.
- PRs should include a concise description, linked issue(s), and any new flags/configs or dataset notes. Add tests when behavior changes.

## Environment & Configuration
- Pixi manages Python/Rust deps (`pixi.toml`); Rust >=1.91 and Python >=3.12 are expected. HDF5 is required for `geonav`.
- Scenarios use YAML/JSON configs; CSV inputs follow Sensor Logger-style IMU/GNSS columns.
