# Queue 104 -- CI and release workflow

| | |
|---|---|
| **Branch** | `v1/p-release-automation` |
| **Base** | `main` |
| **Issues** | #265 |
| **Queue position** | 104 |
| **Merge point** | any time |

## Why

The CI half of #265, independent of every Rust change.

## Acceptance criteria

- [ ] crates.io publish workflow verified by dry run
- [ ] CI matrix covers Linux, macOS and Windows
- [ ] MSRV 1.91 pinned in CI to match Cargo.toml

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

