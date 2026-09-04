# Queue 1 -- ESKF vertical-channel divergence

| | |
|---|---|
| **Branch** | `v1/01-eskf-fix` |
| **Base** | `v1/00-test-speed` |
| **Issues** | #266 |
| **Queue position** | 1 |

## Why

Three confirmed defects: position error is radians in `F` and `H` but metres in `inject_error_state`; the velocity-attitude block of `F` uses a global attitude-error convention while attitude propagation, gyro-bias coupling and injection all use local; and a scalar `1e-9` covariance jitter is added to all 15 diagonal entries regardless of units.

## Acceptance criteria

- [ ] ESKF altitude bounded on `test_data.csv` with full GNSS aiding
- [ ] ESKF altitude bounded under degraded GNSS
- [ ] All three `#[ignore]`d ESKF tests re-enabled
- [ ] Thresholds restated as physical bounds, not golden numbers
- [ ] Finite-difference test of `error_state_transition_jacobian` with non-identity `C_bn`
- [ ] Regression test: filter insensitive to ~1e-12 input perturbation
- [ ] `apply_eskf_correction` and `inject_error_state` unified into one function
- [ ] RBPF unaffected (it shares `apply_eskf_correction`)

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

