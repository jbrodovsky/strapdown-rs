# Queue 9 -- Docs, examples, release automation + strict-lint deny flip

| | |
|---|---|
| **Branch** | `v1/09-release` |
| **Base** | `v1/08-validation` |
| **Issues** | #265, #253 |
| **Queue position** | 9 |

## Why

#253b (the deny flip and mechanical sweep) lands here on purpose: by this point nothing is stacked on top of it, so its churn never has to be rebased through.

## Acceptance criteria

- [ ] `examples/basic_ins.rs` and `examples/gnss_outage.rs`
- [ ] Crate docs, README and mdBook updated; `cargo doc --workspace --no-deps` warning-free
- [ ] Lints flipped warn -> deny, `unwrap_used`/`expect_used`/`panic` added, CI blocking
- [ ] CI green on Linux, macOS and Windows
- [ ] crates.io release workflow verified via dry run

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

