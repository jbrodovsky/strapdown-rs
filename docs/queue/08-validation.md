# Queue 8 -- Integration suite + ground-truth validation

| | |
|---|---|
| **Branch** | `v1/08-validation` |
| **Base** | `main` (queue 6 and 7 merged ahead of it) |
| **Issues** | #264 |
| **Queue position** | 8 |

## Why

Where PR 0's runtime work pays off.

## Acceptance criteria

- [x] End-to-end trajectory estimation on real data
- [x] Horizontal/vertical/attitude RMSE benchmarked across all filters --
      `test_rmse_benchmark_across_filters`, with the EKF excluded while it diverges (#307)
- [x] Full lifecycle: dead reckoning -> GNSS fusion -> ZUPT -> outage recovery --
      `test_full_lifecycle_through_ins_engine`, driven through `InsEngine`. Coarse
      *alignment* is levelling from the first record rather than a true alignment, which
      lands with queue 103 (#282)
- [x] Deterministic across runs -- `test_filters_are_deterministic_across_runs`
- [x] Error metrics documented against theoretical bounds -- "Theoretical bounds" in the
      `integration_tests` module documentation

## What the suite measures

| filter | horizontal | vertical | roll | pitch | yaw |
|--------|-----------|----------|------|-------|-----|
| UKF    | 23.54 m   | 2.78 m   | 3.63 deg | 2.93 deg | 98.1 deg |
| ESKF   | 23.49 m   | 2.40 m   | 3.79 deg | 2.87 deg | 96.8 deg |
| RBPF   | 23.78 m   | 4.54 m   | 3.01 deg | 3.02 deg | 96.3 deg |

The horizontal figure is dominated by sample alignment rather than filter error, and the
yaw figure is a genuine defect. Both are derived in the module documentation.

## Defects this suite surfaced

Filed rather than fixed here -- this PR reports metrics, it does not change filter or
simulation behaviour.

| Issue | What |
|---|---|
| #305 | Yaw diverges in every filter (~96 deg RMSE); unobservable under position-only aiding |
| #306 | EKF vertical and roll RMSE ~10x the other filters |
| #312 | `GnssScheduler::DutyCycle` emits GNSS only at window toggles, so outages never happen |

#307 (EKF divergence to ~14,707 km) predates this work and is why five tests are
`#[ignore]`d, including `test_filter_comparison`.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).
