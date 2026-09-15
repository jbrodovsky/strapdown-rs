# Contribution Guidelines

Thank you for considerating contributing to this project. At time of writing this is being developed as part of a PhD dissertation. The goal is to make this a high-performance, memory-safe, and cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems. The project is open to contributions from the community, and any feedback or suggestions for improvement are welcome.

If you are a developer, researcher, or enthusiast interested in contributing to this project, please first reach out to [James Brodovsky](mailto:jbrodovsky@temple.edu). Contribution are welcome, but primary authorship will remain with the original author.

If you find a bug while using this software please open an issue and report it as such. If you have a feature request please similarly open an issue to request it.

## Before you open a pull request

The workspace enforces a strict lint gate rather than a warning-level one. Both of these must
be clean, and CI runs exactly the same commands on the pinned 1.91 toolchain:

```bash
pixi run fmt-check   # cargo fmt --all -- --check
pixi run lint        # cargo clippy --workspace --all-targets --all-features -- -D warnings
pixi run test        # cargo test --workspace --all-features
```

`clippy::pedantic` and `clippy::nursery` are denied workspace-wide, as are `missing_docs` and
the zero-panic lints (`unwrap_used`, `expect_used`, `panic`) in library code -- return a
`StrapdownError` instead. Every `pub` item, including struct fields and enum variants, needs a
doc comment.

If a lint is genuinely wrong for this codebase, relax it once in `[workspace.lints.clippy]` in
the root `Cargo.toml` with a comment explaining why, rather than adding an `#[allow]` at the
call site. `AGENTS.md` has the full policy, including which lints are already relaxed and on
what grounds.
