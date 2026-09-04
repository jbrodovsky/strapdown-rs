# Repository Guidelines

When asked to execute on an issue or new high-level feature request, ask the user if this should be completed on the current branch or to create a new branch. This project is still in pre-1.0 development, so breaking changes may be introduced at any time and backwards compatibility need not be maintained.

## Project Structure & Module Organization
This is a Cargo workspace with three crates.
- `core/`: `strapdown-core` library (INS algorithms, filters, simulation utilities).
- `sim/`: `strapdown-sim` CLI for open/closed-loop runs and GNSS degradation.
- `geonav/`: experimental geophysical navigation module.
- `docs/`, `book/`: design notes and the mdBook user guide.
- `papers/joss/`: the JOSS submission manuscript.
- `examples/`: usage examples and sample scenario configs.
- Test fixtures live in `core/tests/`; sample datasets are not vendored.

## Build, Test, and Development Commands
Use Pixi when available; Cargo works directly too.
- `pixi run build` / `cargo build --workspace --release`: build all crates.
- `cargo test --workspace`: run all tests.
- `cargo test --package strapdown-core`: test a single crate.
- `pixi run lint`: run clippy as CI does; `pixi run lint-fix` applies fixes.
- `pixi run fmt`: run rustfmt; `pixi run fmt-check` checks without writing.
- `pixi run coverage` / `cargo tarpaulin --workspace --timeout 600`: coverage.
- Example run: `./target/release/strapdown-sim -i input.csv -o output.csv open-loop`.

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
