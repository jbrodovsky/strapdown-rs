# Queue 6 -- NIS gating + ZUPT/ZARU

| | |
|---|---|
| **Branch** | `v1/06-aiding` |
| **Base** | `main` (queue 5 merged in #274) |
| **Issues** | #260, #261 |
| **Queue position** | 6 |

## Why

`sim::health::HealthMonitor` already has `nis_pos_max` and a consecutive-exceedance counter but is passed `None` at both call sites. This PR supplies the producer.

## Acceptance criteria

- [x] NIS/Mahalanobis gate with configurable chi-squared threshold, wired into `update`
- [x] Rejected-measurement metrics logged; `HealthMonitor` receives a real NIS
- [x] Variance-based stationary detector
- [x] `ZuptMeasurement`/`ZaruMeasurement` + Jacobians in `linearize.rs`
- [x] ZUPT/ZARU prevent drift over prolonged stops

## What landed

New modules:

- `core/src/gating.rs` -- `InnovationGate` (chi-squared quantile at the measurement's own
  DOF, or a fixed threshold), `normalized_innovation_squared`, `UpdateOutcome`, and the
  chi-squared CDF/quantile they are built on. No new dependency: the incomplete gamma
  function is implemented here and validated against published tables.
- `core/src/stationary.rs` -- `StationaryDetector`, a sliding-window detector testing
  specific-force variance, specific-force magnitude, angular-rate variance *and*
  angular-rate mean. The last two are what separate a stop from a steady turn, which bare
  variance cannot see.

`NavigationFilter::update` now returns `Result<UpdateOutcome, StrapdownError>` and the trait
gains `set_innovation_gate`. All four filters (ESKF, EKF, UKF, RBPF) honour the gate and
report a NIS whether or not one is installed -- `HealthMonitor` consumes it either way.

`ZuptMeasurement` and `ZaruMeasurement` with `zupt_jacobian`/`zaru_jacobian`. ZARU observes
the gyro bias directly, so its Jacobian is 3x15 rather than the 3x9 every other model
returns; `expand_measurement_jacobian` in `kalman.rs` now widens 9-column Jacobians instead
of each filter open-coding the padding, and the ESKF's nominal vector was widened to 15 so
there is a bias to predict from.

Gating is **opt-in** (`--gate-confidence`, or `innovation_gate` in a scenario file).
Switching it on by default changes the trajectory of every existing scenario, which is a
decision for the ground-truth validation suite in queue 8, not a side effect of adding the
capability.

## Measured

ZUPT/ZARU over a 600 s stop with an unmodelled 0.02 m/s^2 accelerometer bias and 1e-3 rad/s
gyro bias (`core/tests/aiding.rs`):

| | position error | attitude error |
|---|---|---|
| unaided | 497 579 m | 0.864 rad |
| ZUPT only | 0.008 m | 0.586 rad |
| ZARU only | 5 242 m | 0.000 59 rad |
| both | 0.006 m | 0.002 rad |

Each aids what it observes and not the other, which is the expected split: ZUPT makes the
accelerometer bias observable through velocity, ZARU makes the gyro bias observable directly,
and neither substitutes for the other.

Gating, one outlier at 50x the filter's own reported position sigma:

| filter | ungated | gated |
|---|---|---|
| ESKF | 3.95 m | 0.21 m |
| EKF | 22 527 m | 4.00 m |
| UKF | 10 074 m | 3.98 m |

## Found while doing this, not fixed here

Three pre-existing defects, all filed. None is caused by this work and none is fixed by it.

**#303 (reopened)** -- closed as completed but still failing on the tip of `main`, including
its own reproducer `filter_comparison.rs::all_filters_converge_from_a_displaced_seed`. New
evidence: no seed error is needed. Ordinary GNSS fix noise, at *half a metre* rms, diverges
the ESKF vertical channel to 1e8 m over 300 s, and the ESKF's divergence is nearly
independent of the noise amplitude -- the signature of an unstable mode rather than of noise
propagation. `filter_comparison.rs` misses it because it feeds fixes taken noise-free from
truth. Reproducer: `core/tests/aiding.rs::a_note_on_filter_consistency` (`#[ignore]`d,
asserting the healthy behaviour so it turns green when fixed).

**#307** -- the EKF ends `test_data.csv` ~14,707 km from truth, and the existing assertions
accept it because they compare against dead reckoning, which is worse. Wiring the real NIS
into `HealthMonitor` is what exposed this: the monitor's `nis_pos_max`/`nis_pos_consec_fail`
limits were always there, but nothing produced a NIS for them to judge. With one, the run
aborts on `NIS = 16728` against a limit of 100. Five integration tests are `#[ignore]`d
against it here; the ESKF, UKF and RBPF legs of the same suite pass.

**#308** -- `sim::DEFAULT_PROCESS_NOISE` gives latitude and longitude `1e-6` **rad^2** per
step (a 6.4 km standard deviation) next to an altitude term of `1e-4` m^2, while the initial
covariance built immediately below it converts metres to radians correctly. The filter
therefore does not filter: it jumps onto each fix rather than averaging, and NIS on honest
fixes comes out at ~0.004 where it should be ~1. Unchanged here because it moves the
trajectory of every existing scenario and every tuned bound -- that belongs with queue 8
(#277). `core/tests/aiding.rs` writes every position quantity in metres and converts once.

### Why this bounds what #260 can be shown to do

A filter whose reported uncertainty does not match its actual error cannot gate. On `main`
the EKF reports a position sigma of roughly 500 m after converging to metres, so a genuine
200 m multipath fix is *within* what it believes possible and is correctly accepted. The gate
itself is validated against published chi-squared tables and rejects what it is asked to, but
end-to-end outlier rejection in metres will only work once #303/#308 are resolved. The tests
here size their outliers in *sigmas read back from the filter* for exactly this reason: that
tests the gate, where a fixed displacement in metres would test the tuning.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).
