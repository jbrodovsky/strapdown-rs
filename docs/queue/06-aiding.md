# Queue 6 -- NIS gating + ZUPT/ZARU

| | |
|---|---|
| **Branch** | `v1/06-aiding` |
| **Base** | `v1/05-filters` |
| **Issues** | #260, #261 |
| **Queue position** | 6 |

## Why

`sim::health::HealthMonitor` already has `nis_pos_max` and a consecutive-exceedance counter but is passed `None` at both call sites. This PR supplies the producer.

## Acceptance criteria

- [ ] NIS/Mahalanobis gate with configurable chi-squared threshold, wired into `update`
- [ ] Rejected-measurement metrics logged; `HealthMonitor` receives a real NIS
- [ ] Variance-based stationary detector
- [ ] `ZuptMeasurement`/`ZaruMeasurement` + Jacobians in `linearize.rs`
- [ ] ZUPT/ZARU prevent drift over prolonged stops

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

