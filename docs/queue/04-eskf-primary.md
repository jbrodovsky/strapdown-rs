# Queue 4 -- 15-state ESKF as the default filter

| | |
|---|---|
| **Branch** | `v1/04-eskf-primary` |
| **Base** | `v1/03-ned-default` |
| **Issues** | #258 |
| **Queue position** | 4 |

## Why

Much smaller than the issue implies now that #266 is fixed and Joseph form already exists.

## Acceptance criteria

- [ ] ESKF propagates on `ImuSample`
- [ ] Closed-loop error injection resets the error state to zero
- [ ] Bias estimates stay bounded over the full 5,366-sample run
- [ ] Convergence and stability tests pass

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

