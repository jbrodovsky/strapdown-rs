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

## Code Quality Gates

These are **enforced**, not advisory. `pixi run lint` and the blocking CI job both run
`cargo clippy --workspace --all-targets --all-features -- -D warnings`, and every lint level
in `[workspace.lints]` is `deny`, so a plain `cargo build` fails the same way CI does. Run
`pixi run lint` and `pixi run fmt-check` before pushing; there is no warning-level grace period.

**What is on** (`Cargo.toml`, `[workspace.lints]`):
- `clippy::pedantic` and `clippy::nursery`, both at `deny`.
- `rust_2018_idioms`, `unreachable_pub`, `missing_debug_implementations`, `missing_docs`.
- The zero-panic policy of #254: `clippy::unwrap_used`, `clippy::expect_used` and
  `clippy::panic` are denied **in library code**. `clippy.toml` exempts tests, which
  legitimately unwrap in order to assert.
- `clippy::missing_errors_doc` and `clippy::missing_panics_doc`.
- `clippy::needless_pass_by_value`.

**What this means when you write code:**
- No `unwrap()`, `expect()` or `panic!()` in a library crate. Return `StrapdownError`
  (`core/src/error.rs`) instead; it has a variant for every failure the mechanization and the
  filters can reach, and adding one is cheaper than adding a panic.
- Every `pub` item needs a doc comment, including struct fields and enum variants. A
  `fn -> Result` needs an `# Errors` section saying when each variant is returned.
- New public functions take references unless the callee stores the value.
- Derive or implement `Debug` on every public type.

**Adding an exception.** If a lint is genuinely wrong for this codebase, relax it **once, in
`[workspace.lints.clippy]`, with a comment giving the reason** -- not with a scattered
`#[allow]` at the call site. The existing allows are all written that way and each carries its
justification (`suboptimal_flops` changes floating-point results in the mechanization;
`similar_names` contradicts the Groves-derived naming; `doc_markdown` cannot tell LaTeX from
Rust in the KaTeX maths, see #329). Two `#[allow]`s exist at call sites; both carry a `reason`.

**Toolchain pinning.** `pedantic` and `nursery` are lint *groups*, so a newer clippy can add a
lint this code has never been checked against. CI therefore pins `dtolnay/rust-toolchain@1.91`
to match `rust-version` in `Cargo.toml`, the `rust` pin in `pixi.toml` and `msrv` in
`clippy.toml`. Keep all four in sync; raising the MSRV is a breaking change. A forward-looking
`allow` for a lint the pinned toolchain does not know about is made inert by
`unknown_lints = "allow"` in `[workspace.lints.rust]`.

## Testing Guidelines
- Unit tests live alongside modules; integration tests live in `core/tests/integration_tests.rs`.
- Tests should be deterministic; seed RNGs when applicable.
- Name test functions in `snake_case` and keep fixtures minimal.

## Commit & Pull Request Guidelines
- Commit subjects are short, imperative, and plain (e.g., "Update RBPF documentation..."). Use `Fixes #123` when closing issues.
- PRs should include a concise description, linked issue(s), and any new flags/configs or dataset notes. Add tests when behavior changes.

## Environment & Configuration
- Pixi manages the toolchain and system libraries (`pixi.toml`); Rust >=1.91 is expected. HDF5
  and netCDF are required for `geonav`, and the freetype/fontconfig stack for the `plotting`
  feature of `strapdown-sim`.
- The repository is **Rust-only**. The Python post-processing package under `analysis/` was
  untracked in `c5f72c6` when the repo was scoped to the v1.0 crate set, so there is no Python
  source, no `pyproject.toml` and no `ruff`/`ty` configuration to maintain. If Python bindings
  or an analysis package return (`strapdown_py` is a commented-out workspace member), they get
  their own tooling gate at that point.
- Scenarios use YAML/JSON configs; CSV inputs follow Sensor Logger-style IMU/GNSS columns.
