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

- [x] ESKF propagates on `ImuSample`
- [x] Closed-loop error injection resets the error state to zero
- [x] Bias estimates stay bounded over the full 5,366-sample run
- [x] Convergence and stability tests pass

## Outcome

`FilterType` now defaults to `Eskf`, so an unqualified `strapdown-sim closed-loop` run and a
`ClosedLoopConfig` with no `filter` key both select the 15-state error-state filter. The UKF
and EKF are unchanged and still selectable with `--filter`.

`ErrorStateKalmanFilter::predict` takes an `ImuSample` and mechanizes the increments
directly, correcting them by `bias * dt` rather than converting back to rates first. `IMUData`
is still accepted and routed through `ImuSample::from_rates`, which is the same rectangular
integration the filter previously performed internally -- `eskf_predicts_identically_from_a_sample_and_from_rates`
holds the two paths to bit equality, so no existing caller moves. A sample whose own `dt`
disagrees with the `dt` argument is rejected with `StrapdownError::InconsistentTimestep`
rather than one being silently preferred; that variant was added in queue 2 for this.

### Bias boundedness

`assert_bias_estimates_bounded` checks every sample of the run, not just the last. The
distinction is the point: `inject_error_state` clamps the estimates, so a final-sample
assertion tests the clamp rather than the estimator, and the clamp fires often enough on the
test-local tuning to matter -- 76 of the 5,366 samples, including the last one, where gyro
bias y sits at exactly the 0.05 rad/s cap.

The default tuning does better. `test_eskf_default_initialization_on_real_data` covers the
`initialize_eskf` path that a `strapdown-sim` user actually gets -- five orders of magnitude
tighter on the bias covariance, and previously exercised by no test at all -- and there the
clamp never engages: peak gyro bias 0.028 rad/s against the 0.05 cap, peak accelerometer bias
0.62 m/s² against 2.0. The test asserts that, so the bound is a statement about the estimator
rather than about the clamp.

| Run | horizontal rms | horizontal max | altitude rms | altitude max |
|---|---|---|---|---|
| Default initialization (`initialize_eskf`) | 23.53 m | 37.88 m | 3.02 m | 42.59 m (sample 3), 12.30 m after settling |
| Test-local tuning, full-rate GNSS | 23.49 m | 37.89 m | 2.40 m | 9.23 m |
| Test-local tuning, 2 s GNSS | 23.56 m | 39.94 m | 3.62 m | 12.90 m |

The 42.59 m altitude peak is the vertical channel settling: the filter starts with zero
vertical velocity and no accelerometer-bias knowledge and cannot separate the two until it has
a few GNSS fixes. It occurs on sample 3 of this 1 Hz recording and the error never again
exceeds 12.30 m across the remaining 5,336 samples. The test bounds the transient and the
steady state separately rather than excluding the transient, which would hide a divergence.

### Left open

The test-local `ESKF_INITIAL_COVARIANCE` is loose enough on the bias states to drive the
anti-windup clamp, and the two ESKF tunings in the tree now differ by five orders of magnitude
with no note saying which is intended. Reconciling them is tuning work, not a correctness fix.

**Resolved** in [`04a-eskf-tuning`](04a-eskf-tuning.md), which collapses the two into the
single documented `sim::ESKF_INITIAL_ERROR_COVARIANCE`. The tables above describe the tree as
of this queue position and are superseded there.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

