# Queue 7 -- InsEngine builder + lever-arm compensation

| | |
|---|---|
| **Branch** | `v1/07-engine` |
| **Base** | `v1/06-aiding` |
| **Issues** | #262 |
| **Queue position** | 7 |

## Why

Depends on the object-safety decision made in PR 2.

## Acceptance criteria

- [ ] `InsEngine::builder().with_config(..).build()`
- [ ] High-rate `predict`, async `update_gnss`, `nav_solution()`
- [ ] Antenna lever-arm compensation for position and velocity
- [ ] Builder documented with doctests

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

