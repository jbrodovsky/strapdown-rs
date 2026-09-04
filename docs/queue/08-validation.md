# Queue 8 -- Integration suite + ground-truth validation

| | |
|---|---|
| **Branch** | `v1/08-validation` |
| **Base** | `v1/07-engine` |
| **Issues** | #264 |
| **Queue position** | 8 |

## Why

Where PR 0's runtime work pays off.

## Acceptance criteria

- [ ] End-to-end trajectory estimation on real data
- [ ] Horizontal/vertical/attitude RMSE benchmarked across all filters
- [ ] Full lifecycle: alignment -> dead reckoning -> GNSS fusion -> ZUPT -> outage recovery
- [ ] Deterministic across runs
- [ ] Error metrics documented against theoretical bounds

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

