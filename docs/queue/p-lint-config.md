# Queue 100 -- Lint config (warn-level) + feature gating

| | |
|---|---|
| **Branch** | `v1/p-lint-config` |
| **Base** | `main` |
| **Issues** | #253, #263 |
| **Queue position** | 100 |
| **Merge point** | before everything |

## Why

Landing `-D pedantic` early would touch nearly every hunk PRs 2-9 also touch, and denying `unwrap_used` before #254 removes the unwraps forces `#[allow]` scaffolding that #254 then deletes. So: config and signal now at `warn`, deny flip in PR 9.

## Acceptance criteria

- [ ] `[workspace.lints]`, `rustfmt.toml`, `clippy.toml` at warn level
- [ ] Non-blocking CI lint job
- [ ] `--no-default-features` and `--all-features` both build clean
- [ ] MIT licensing reaffirmed across all crates

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

