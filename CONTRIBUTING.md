# Contribution Guidelines

Thank you for considerating contributing to this project. At time of writing this is being developed as part of a PhD dissertation. The goal is to make this a high-performance, memory-safe, and cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems. The project is open to contributions from the community, and any feedback or suggestions for improvement are welcome.

If you are a developer, researcher, or enthusiast interested in contributing to this project, please first reach out to [James Brodovsky](mailto:jbrodovsky@temple.edu). Contribution are welcome, but primary authorship will remain with the original author.

If you find a bug while using this software please open an issue and report it as such. If you have a feature request please similarly open an issue to request it.

## Setting up

There is no environment manager to install -- `git clone && cargo build` is the whole setup.
`rust-toolchain.toml` fetches the pinned toolchain automatically on the first cargo command
(a few hundred MB the first time), and `.cargo/config.toml` provides the aliases below.

Beyond Rust you need a **C/C++ compiler and cmake >= 3.26**, because libhdf5, libnetcdf, zlib
and freetype are compiled from vendored sources rather than linked from the system. That is
what lets every crate build and document anywhere without `apt install` (#335). Two things to
watch for:

- **Ubuntu 22.04 ships cmake 3.22 and Debian 12 ships 3.25 -- both too old.** Install a newer
  one from [Kitware's APT repository](https://apt.kitware.com/), or `pip install cmake`, or
  `snap install cmake --classic`. Ubuntu 24.04 and Fedora 40+ are fine as shipped.
- **Make sure `HDF5_DIR` is not set** in your shell. Its build script prefers a system library
  whenever that variable exists, even when asked for a vendored build, and the netCDF build
  then fails against those headers in a way that does not name the cause.

`strapdown-core` on its own needs none of this: `cargo build -p strapdown-core` uses no C
toolchain at all.

Nothing here needs the [GitHub CLI](https://cli.github.com/) to build, test or lint, so this is
only relevant if you use `gh` to open and review pull requests. If you do, it must be **2.71 or
newer**. Older clients ask for the `projectCards` field that GitHub removed along with Projects
(classic), so `gh pr view` and `gh issue view` exit 1 on a deprecation notice, and `gh pr edit`
exits 1 while leaving the pull request body **unchanged** -- easy to miss. No Debian or Ubuntu
release ships a new enough package (Ubuntu 26.04 and Debian trixie are on 2.46, Debian 12 on
2.23), so install it from
[cli.github.com's apt repository](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)
or put the release binary on your `PATH`.

## Before you open a pull request

The workspace enforces a strict lint gate rather than a warning-level one. Both of these must
be clean, and CI runs exactly the same commands on the same pinned 1.91 toolchain:

```bash
cargo fmt-check   # cargo fmt --all -- --check
cargo lint        # cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

`clippy::pedantic` and `clippy::nursery` are denied workspace-wide, as are `missing_docs` and
the zero-panic lints (`unwrap_used`, `expect_used`, `panic`) in library code -- return a
`StrapdownError` instead. Every `pub` item, including struct fields and enum variants, needs a
doc comment.

If a lint is genuinely wrong for this codebase, relax it once in `[workspace.lints.clippy]` in
the root `Cargo.toml` with a comment explaining why, rather than adding an `#[allow]` at the
call site. `AGENTS.md` has the full policy, including which lints are already relaxed and on
what grounds.

### Performance baselines

`core/tests/perf_baseline.rs` scores every filter over a fixed set of scenarios and compares
the result against `core/tests/perf_baseline.json`, a checked-in record of what the navigation
solution measured when it was last blessed. It runs as part of `cargo test --workspace
--all-features`, so the three commands above already cover it -- there is no fourth command to
remember. `cargo perf` runs it alone and prints the table.

It fails in **both** directions. A metric more than 10% worse than its baseline is a
regression. A metric more than 25% better is *also* a failure, and the message says to
re-bless: an improvement that is not recorded is one the next change can silently undo.

```bash
UPDATE_PERF_BASELINE=1 cargo test -p strapdown-core --test perf_baseline
# PowerShell:
$env:UPDATE_PERF_BASELINE=1; cargo test -p strapdown-core --test perf_baseline
```

Review the diff before committing it -- every changed number is a claim about the navigation
solution, and the pull request description should say which change produced it. Blessing
preserves the per-metric `note`, `gated` and tolerance overrides already in the file, so
annotating a number is not undone by the next re-bless. See the module documentation in
`core/tests/perf_baseline.rs` for what these numbers do and do not measure, and
`book/src/development/performance.md` for the current values.
