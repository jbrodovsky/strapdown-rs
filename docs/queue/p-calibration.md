# Queue 102 -- ImuCalibration

| | |
|---|---|
| **Branch** | `v1/p-calibration` |
| **Base** | `main` |
| **Issues** | #256 |
| **Queue position** | 102 |
| **Merge point** | rebase onto PR 2 for the final signature |

## Why

New module, purely additive.

## Acceptance criteria

- [ ] `ImuCalibration` with biases, scale factors and misalignment matrices
- [ ] `correct(&self, raw: &ImuSample) -> ImuSample`
- [ ] serde round-trip (JSON/YAML/TOML)
- [ ] Unit tests for identity and non-trivial corrections

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

