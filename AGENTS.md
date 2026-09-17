# Repository Guidelines

When asked to execute on an issue or new high-level feature request, ask the user if this should be completed on the current branch or to create a new branch. This project is at 1.0, so `strapdown-core` and `strapdown-sim` follow semantic versioning: a breaking change to a `pub` item needs a major bump and cannot ride along with a fix. `strapdown-geonav` is deliberately held at 0.x (see `geonav/Cargo.toml`) and its API may still move. When a change would break a published API, say so and propose the additive form instead.

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
Plain Cargo, no environment manager. `rust-toolchain.toml` pins the toolchain and
`.cargo/config.toml` defines the aliases below.
- `cargo build --workspace --release`: build all crates.
- `cargo test --workspace`: run all tests.
- `cargo test --package strapdown-core`: test a single crate.
- `cargo lint`: clippy over the workspace with all features; `cargo lint-fix` applies what
  it can. **Not CI parity on its own** -- see the gates section below.
- `cargo lint-min` / `cargo test-min`: the `-p strapdown-core --no-default-features`
  configuration CI also gates on, which `cargo lint` cannot see.
- `cargo fmt --all`: run rustfmt; `cargo fmt-check` checks without writing.
- `cargo docs`: build the API docs with KaTeX (aliased, because cargo ignores an alias that
  shadows a built-in subcommand such as `doc`).
- `cargo coverage`: coverage; needs `cargo install cargo-tarpaulin` first.
- Example run: `./target/release/strapdown-sim dr -i input.csv -o output.csv`. The
  subcommand comes first -- `-i` before it is rejected -- and there is no `open-loop`
  subcommand: the modes are `dr`, `ol`, `cl`, `pf`, `config` and `syn`. Use `dr` for dead
  reckoning; `ol` is not implemented and writes no output.

## Coding Style & Naming Conventions
- Rust formatting via rustfmt (4-space indentation); keep functions focused and small.
- Naming: `snake_case` for functions/vars, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Prefer descriptive names over symbols; add Rust doc comments (`///`) and cite Groves equations when relevant.
- Use `assert_approx_eq` for floating-point comparisons in tests.

## Code Quality Gates

These are **enforced**, not advisory. `cargo lint` and one blocking CI job run
`cargo clippy --workspace --all-targets --all-features -- -D warnings`, and every lint level
in `[workspace.lints]` is `deny`, so a plain `cargo build` fails the same way CI does.

**`cargo lint` is half the gate.** A second blocking job lints
`-p strapdown-core --all-targets --no-default-features`, and the two configurations disagree
about what is dead: an item used only behind a feature gate is live under `--all-features` and
orphaned without it. That is `cargo lint-min`, with `cargo test-min` as its test counterpart.

Run `cargo lint`, `cargo lint-min` and `cargo fmt-check` before pushing; there is no
warning-level grace period.

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
lint this code has never been checked against. The pin therefore lives in five places that must
agree: `channel` in `rust-toolchain.toml`, `rust-version` in `Cargo.toml`, `msrv` in
`clippy.toml`, and the `dtolnay/rust-toolchain@1.91` pins in `.github/workflows/rust.yml` and
`publish.yml`. `rust-toolchain.toml` is what makes a local `cargo clippy` give the same answer
as CI, which a floating `stable` cannot. Raising the MSRV is a breaking change. A
forward-looking `allow` for a lint the pinned toolchain does not know about is made inert by
`unknown_lints = "allow"` in `[workspace.lints.rust]`. `deploy-book.yml` deliberately uses
`cargo +stable install mdbook`, since a third-party tool need not honour our MSRV.

## Testing Guidelines
- Unit tests live alongside modules; integration tests live in `core/tests/integration_tests.rs`.
- Tests should be deterministic; seed RNGs when applicable.
- Name test functions in `snake_case` and keep fixtures minimal.
- Navigation accuracy is gated, not just asserted. `core/tests/perf_baseline.rs` scores every
  filter over a fixed scenario matrix and compares the result against
  `core/tests/perf_baseline.json`; it fails both when a metric gets worse and when it gets
  better, the second asking for the baseline to be re-blessed. It rides in
  `cargo test --workspace --all-features`, so there is no extra command to run before pushing.
  `CONTRIBUTING.md` has the bless workflow and `book/src/development/performance.md` the
  current numbers. Note that this is *accuracy*, not wall-clock: nothing here measures runtime.
- Do not tighten `core/tests/integration_tests.rs`'s thresholds to match a baseline number.
  Several of them are derived physical bounds with their derivations written out, and they
  answer a different question -- "is this physically possible?" rather than "is this worse than
  yesterday?".

## Commit & Pull Request Guidelines
- Commit subjects are short, imperative, and plain (e.g., "Update RBPF documentation..."). Use `Fixes #123` when closing issues.
- PRs should include a concise description, linked issue(s), and any new flags/configs or dataset notes. Add tests when behavior changes.

## Environment & Configuration
- **No environment manager.** `git clone && cargo build` is the whole setup. `rust-toolchain.toml`
  fetches the pinned toolchain; `.cargo/config.toml` carries the env vars and aliases that
  `pixi.toml` used to. Pixi was removed in #335.
- **Build prerequisites** beyond Rust: a C/C++ compiler and **cmake >= 3.26**, needed only when a
  feature that touches a C library is on (`geonav` always; `strapdown-core`'s `hdf5`/`netcdf`;
  `strapdown-sim`'s `plotting`, for freetype). libhdf5, libnetcdf, zlib and freetype are all
  compiled from vendored sources, so no system library is ever searched for. Ubuntu 22.04 (cmake
  3.22) and Debian 12 (3.25) are too old and need a newer cmake.
- **`HDF5_DIR` must stay unset.** `hdf5-metno-sys`'s build script prefers a system library
  whenever that variable is set, *even with the `static` feature on*, and the vendored netCDF
  build then fails confusingly against those headers. A leftover `pixi shell` is the likely way
  to hit this.
- libfontconfig is a **runtime** dependency of the `plotting` feature (it is `dlopen`ed), not a
  build-time one: a machine without it builds fine and fails when it first renders text.
- **`gh` must be 2.71 or newer** -- not a build prerequisite (nothing here needs it to build,
  test or lint), but the PR and issue workflow does. Older clients request the `projectCards`
  field GitHub removed with Projects (classic): `gh pr view`/`gh issue view` exit 1 on a
  deprecation notice, and `gh pr edit` exits 1 while leaving the body **unchanged**. No Debian
  or Ubuntu package is new enough (Ubuntu 26.04 and Debian trixie ship 2.46, bookworm 2.23), so
  take it from [cli.github.com](https://github.com/cli/cli/blob/trunk/docs/install_linux.md) or
  put the release binary on `PATH`. Diagnosed in #365; `gh api` was never affected.
- The repository is **Rust-only**. The Python post-processing package under `analysis/` was
  untracked in `c5f72c6` when the repo was scoped to the v1.0 crate set, and the two analysis
  notebooks under `examples/` followed in #335 -- they import pygmt, cartopy and filterpy
  against the environment pixi used to provide, and nothing here can run them. So there is no
  Python source, no `pyproject.toml` and no `ruff`/`ty` configuration to maintain. If Python bindings
  or an analysis package return (`strapdown_py` is a commented-out workspace member), they get
  their own tooling gate at that point.
- Scenarios use YAML/JSON configs; CSV inputs follow Sensor Logger-style IMU/GNSS columns.
