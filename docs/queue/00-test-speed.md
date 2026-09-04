# Queue 0 -- Test-suite runtime

| | |
|---|---|
| **Branch** | `v1/00-test-speed` |
| **Base** | `main` |
| **Issues** | -- |
| **Queue position** | 0 |

## Why

`cargo test` takes ~241 s. 11 of 13 integration tests finish in 0.78 s; the two RBPF tests take 224 s and 210 s because they run at `num_particles: 5000`, 10x the `RbpfConfig::default()` of 500. The build profile is already fully optimised (`opt-level=3` for nalgebra, `strapdown` and the test harness alike), so this is a test-configuration problem, not a Cargo profile problem.

## Acceptance criteria

- [ ] Integration suite runs in well under a minute
- [ ] RBPF error metrics stay inside the existing assertion thresholds with margin
- [ ] Assertion messages state the threshold they actually enforce
- [ ] No test changes pass/fail status

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

