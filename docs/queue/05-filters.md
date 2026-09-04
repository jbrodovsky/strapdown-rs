# Queue 5 -- EKF, UKF, PF/RBPF on the new API

| | |
|---|---|
| **Branch** | `v1/05-filters` |
| **Base** | `v1/04-eskf-primary` |
| **Issues** | #259 |
| **Queue position** | 5 |

## Why

Scope decision needed first: `core/src/particle.rs` is 466 lines of traits with zero concrete impls, while `sim.rs` already advertises a `ParticleFilterType::Standard` that does not exist.

## Acceptance criteria

- [ ] EKF/UKF Jacobians and sigma-point propagation updated for Delta-v/Delta-theta
- [ ] RBPF prediction and state representation updated
- [ ] All filters implement the updated trait with uniform `Result` returns
- [ ] Side-by-side comparison test across filters

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

