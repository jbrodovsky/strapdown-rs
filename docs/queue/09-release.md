# Queue 9 -- Docs, examples, release automation + strict-lint deny flip

| | |
|---|---|
| **Branch** | `v1/09-release` |
| **Base** | `main` (queue 8 merged ahead of it) |
| **Issues** | #265, #253 |
| **Queue position** | 9 |

## Why

#253b (the deny flip and mechanical sweep) lands here on purpose: by this point nothing is stacked on top of it, so its churn never has to be rebased through.

## Acceptance criteria

- [x] `examples/basic_ins.rs` and `examples/gnss_outage.rs` -- written as
      `core/examples/`, which is where Cargo discovers examples; the workspace root is a
      virtual manifest and its `examples/` directory holds configs and notebooks, not targets
- [x] `cargo doc --workspace --no-deps` warning-free -- 30 warnings fixed
- [x] README updated with a runnable-examples section
- [~] mdBook updated -- the four pages this PR's work produces are written
      (`examples/tutorial-basic`, `examples/tutorial-gps-degradation`,
      `examples/configurations`, `user-guide/configuration`). **37 of the book's 55 pages
      remain one-line placeholders** and filling them is its own project, not a queue item
      that also ships examples, a scheduler fix and a lint flip
- [x] Lints flipped warn -> deny (11 levels); `unwrap_used`/`expect_used`/`panic` were
      already present and CI already ran `-D warnings`, so what changed is that a local
      `cargo clippy` now fails the same way CI does
- [x] CI green on Linux, macOS and Windows
- [ ] crates.io release workflow verified via dry run -- **deliberately not run.** See below

## Release verification

Left for the maintainer rather than attempted. `publish.yml` dry-runs all three crates with
`--locked`, but `strapdown-core` is on crates.io at 0.5.0 and the workspace is at 1.0.0, while
`strapdown-geonav` has never been published. The dependent crates therefore try to resolve
`strapdown-core = 1.0.0` from the registry and cannot: a dependent cannot be dry-run before
its dependency is actually published. Only `strapdown-core` itself can be verified this way,
and the rest become verifiable the moment 1.0.0 is live.

## Defects found and fixed here

| Issue | What |
|---|---|
| #312 | `GnssScheduler::DutyCycle` emitted GNSS only at window toggles -- `on_s: 1800, off_s: 600` delivered **two** fixes across an 89-minute recording. Fixed, with the state now derived from the elapsed clock rather than stepped |
| -- | **Nine of the shipped example configs did not deserialize at all**: `duty`, `passthrough` and `slowbias` are not the variant names, and `json/baseline.json` was malformed. Because every field has a default, a misspelled `kind` silently produced a pass-through config rather than an error, so these scenarios simulated nothing |

The second is guarded by `core/tests/example_configs.rs`, which asserts every config parses
*into the scenario it describes* rather than merely parsing.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).
