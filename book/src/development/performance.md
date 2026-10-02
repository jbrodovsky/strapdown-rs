# Performance Baselines

"Performance" on this page means **navigation accuracy**, not wall-clock speed: how far the
solution was from truth, and whether the filter's covariance knew it. Execution time, memory
and scalability are not measured anywhere yet.

Every number here is recorded in `core/tests/perf_baseline.json` and checked on every CI run by
`core/tests/perf_baseline.rs`. The check fails in **both** directions:

- a metric more than **10%** worse than its baseline is a regression, and CI goes red;
- a metric more than **25%** better is *also* a failure, and the message asks for the baseline
  to be re-blessed. An improvement nobody records is one the next change can silently undo.

Each scenario also records the number of aligned samples it was scored over and the number of
channel-samples dropped as non-finite, both compared for exact equality before any metric is.
The second is zero everywhere. It is recorded because dropping a sample does not lower the
first, but does make the metric it was dropped from look better.

## Running and re-blessing

```bash
cargo perf     # run the suite and print the table; writes nothing
UPDATE_PERF_BASELINE=1 cargo test -p strapdown-core --test perf_baseline   # re-bless
```

In PowerShell the second is `$env:UPDATE_PERF_BASELINE=1; cargo test -p strapdown-core --test
perf_baseline`. Re-blessing rewrites `core/tests/perf_baseline.json` *and* regenerates
`book/src/development/baseline-tables.md`, which this page includes below, and the gate fails if
the two disagree, so there is no second step to forget. It preserves the per-metric `note`,
`gated` and tolerance overrides already in the file. Review the diff before committing it:
every changed number is a claim about the navigation solution, and the pull request should
say which change produced it. [Contributing](./contributing.md) has the workflow.

## Scenarios

`real` is `core/tests/test_data.csv`: 5,366 records at 1 Hz over 89 minutes, ENU, from a
consumer phone. `syn` is `sim::generate_synthetic`: 300 s at 50 Hz, NED, a 50 m/s
constant-velocity cruise (40 m/s north, 30 m/s east) from 40° N, 75° W with a consumer-grade
IMU, seed 42, scored against its own exact truth.

| id | data | GNSS | estimators |
|---|---|---|---|
| `real_clean` | real | every epoch | UKF, EKF, ESKF |
| `real_sparse_5s` | real | every 5 s | UKF, ESKF |
| `real_outage_60s` | real | duty cycle: 60 s off, 120 s on, repeating | ESKF |
| `real_degraded` | real | every epoch, AR(1) position and velocity fault ($\rho$ 0.98, $\sigma$ 3 m and 0.3 m/s, `r_scale` 1) | UKF, ESKF |
| `syn_cruise_1hz` | syn | every 1 s | UKF, EKF, ESKF |
| `syn_outage_60s` | syn | duty cycle: 60 s off, 60 s on, repeating | UKF, ESKF |
| `syn_dead_reckoning` | syn, first 120 s | none | unaided mechanization |
| `real_rbpf_slice` | real, first 1,200 records | every epoch | RBPF, 500 particles |

The two outage scenarios use `start_phase_s = 0`, so each begins with its OFF window (see
[`DutyCycle`](../gnss/scenarios.md#dutycycle)). The barometer and magnetometer are scheduled at
their 1 Hz default in every scenario, so the synthetic vertical and yaw columns are those of a
1 Hz barometer and magnetometer even though the IMU runs at 50 Hz.

Two scenarios are deliberately absent. **Full-run dead reckoning** ends millions of metres out,
at altitudes where floating-point results have already been seen to differ between platforms;
the 120 s synthetic window is the safe version of the same measurement. **A second
particle-filter scenario** is left out to keep the suite's CI time down.

The suite is not behind a feature or `#[ignore]`, so it runs on all three CI platforms. That is
deliberate: the three legs double as a portability check on the accuracy numbers themselves.

## The metrics

| metric | unit | definition | ideal |
|---|---|---|---|
| `horizontal_rmse_m` | m | RMS great-circle distance from truth | 0 |
| `horizontal_cep50_m` | m | median radial error: the empirical, nearest-rank CEP, *not* the Rayleigh form | 0 |
| `horizontal_cep95_m` | m | 95th-percentile radial error, same convention | 0 |
| `horizontal_max_m` | m | largest single-sample radial error | 0 |
| `vertical_rmse_m` | m | RMS altitude error | 0 |
| `vertical_bias_m` | m | **signed** mean altitude error; catches a slow one-sided drift while it is still small | 0 |
| `velocity_horizontal_rmse_mps` | m/s | RMS north/east velocity error | 0 |
| `velocity_vertical_rmse_mps` | m/s | RMS vertical velocity error | 0 |
| `roll_rmse_deg`, `pitch_rmse_deg` | deg | RMS per-axis error, wrapped onto $[-\pi, \pi]$ | 0 |
| `yaw_rmse_deg` | deg | RMS heading error, wrapped | 0 |
| `attitude_geodesic_rmse_deg` | deg | RMS rotation angle of $R_{est}^{-1} R_{true}$: the only attitude error independent of the Euler sequence | 0 |
| `nees_position` | -- | mean normalized estimation error squared over the full 3×3 position block, $\frac{1}{N}\sum_k e_k^\top P_k^{-1} e_k$ | 3.0 |
| `npes_position` | -- | the same, computed as though $P$ were diagonal: $\frac{1}{N}\sum_k \left(\frac{e_{lat}^2}{P_{lat}} + \frac{e_{lon}^2}{P_{lon}} + \frac{e_{alt}^2}{P_{alt}}\right)$ | 3.0 |
| `containment_3sigma_horizontal` | fraction | share of latitude and longitude samples inside $\pm 3\sigma$ | 0.9973 |
| `containment_3sigma_vertical` | fraction | share of altitude samples inside $\pm 3\sigma$ | 0.9973 |

`nees_position` is the real consistency statistic; it became computable when
`NavigationResult` started keeping the position block's off-diagonal terms
([#376](https://github.com/jbrodovsky/strapdown-rs/issues/376)). `npes_position` is kept so its
history stays comparable. It equals the NEES when the position states are uncorrelated and is
optimistic when they are not, which is during a free-inertial coast: compare the two columns on
the outage rows.

Read either beside the containment columns, never alone. Together they separate the two
failure modes either one hides:

| containment | `nees_position` | diagnosis |
|---|---|---|
| ~1.0 | far below 3 | over-conservative: the filter is right but does not believe it |
| below 0.99 | far above 3 | over-confident, the dangerous one |
| ~0.9973 | ~3 | consistent |

There is deliberately no Wasserstein metric; the reasoning is in the `strapdown::metrics`
module documentation.

## Read these caveats first

The numbers mean little without them.

1. **On the real-data scenarios, "truth" is the GNSS fix, which is also the filters' aiding
   source.** They measure agreement with the aid, not independent accuracy, and they cannot
   fall below the receiver's own error. The receiver advertises a 3.81 m horizontal 1σ on
   average over this recording (`GNSS_REPORTED_HORIZONTAL_ACCURACY_M` in
   `core/tests/integration_tests.rs`, checked against the data on every run). The `syn_*`
   scenarios are scored against an exact synthetic trajectory and have no such floor.
2. **On a full-rate real-data row, the filter is scored against a fix it has just been
   given.** The row at $t_k$ contains $t_k$'s GNSS update, so on `real_clean` the horizontal
   columns measure how a filter weighs its own aiding against its prediction. The real-data
   rows that measure navigation are the ones where the filter has to predict between or
   without fixes (`real_sparse_5s`, `real_outage_60s`, `real_degraded`), together with the
   `syn_*` rows and their exact truth.
3. **The consistency columns mean different things on the two sources.** On the synthetic
   rows they are a real measurement of whether a filter believes the right thing. On the
   real-data rows the reference is itself noisy and the filters' models of the phone are
   approximate, so read them as a drift detector rather than a verdict.
4. **`real_degraded` tells the filter nothing about the fault.** Its fault uses
   `r_scale = 1`, so the filter's measurement noise is not inflated while each fix carries an
   AR(1) error with a steady-state σ of $3/\sqrt{1-0.98^2} \approx 15$ m. Its NEES and
   containment columns are what a filter looks like when its $R$ is wrong, which is the
   point of the scenario. Under that fault the GNSS altitude is itself biased, so the
   barometric bias state cannot be separated from it, and the vertical columns suffer.
5. **The baseline is blessed on one platform and gated on three.** The JSON records the
   platform it was blessed on (`blessed_on`). Most metrics agree across Linux, macOS and Windows
   to well inside the tolerance; the advisory `Cross-platform accuracy spread` job in
   `rust.yml` measures the spread on every run. `real_rbpf_slice__rbpf` is the exception: its
   `horizontal_cep95_m`, `nees_position` and `npes_position` are recorded but **not gated**,
   with the reasons in each metric's `note`, until a cross-platform tolerance exists
   ([#386](https://github.com/jbrodovsky/strapdown-rs/issues/386)). Check a Windows run before
   calling a re-bless done.
6. **A one-ulp change upstream used to move the UKF rows by percents.** At the old
   `ukf_alpha = 1e-3` the weighted sigma-point mean lost about six significant digits to
   cancellation every step ([#399](https://github.com/jbrodovsky/strapdown-rs/issues/399)).
   `ukf_alpha` is now 0.1, and `core/tests/ukf_conditioning.rs` guards the result. If a
   re-bless moves rows the change cannot reach, compile the new code without calling it, bless
   that as a control, and diff the real change against the control.

## Current values

These tables are **generated** from `core/tests/perf_baseline.json` by the test that gates it,
and the gate fails if they drift from it. Do not edit them; change them by re-blessing.

{{#include ./baseline-tables.md}}

## What the tables are saying

Read the rows rather than any summary here; these are the comparisons worth making.

- **`real_clean` measures agreement with the aid.** Compare its `horiz RMSE` column with the
  receiver's 3.81 m advertised accuracy (caveat 1). The three Kalman filters land within a few
  centimetres of each other there, because on this row they are all mostly reporting the fix
  they were just handed (caveat 2).
- **`syn_cruise_1hz` is where the filters can actually separate**, because it is scored against
  exact truth. Compare its three rows column by column; the attitude table shows what a 1 Hz
  magnetometer and GNSS velocity make of yaw on a synthetic trajectory.
- **The outage rows are dominated by the outages.** On `real_outage_60s__eskf` and both
  `syn_outage_60s` rows, CEP50 stays within a few metres of the full-rate rows while RMSE, CEP95
  and the maximum, an order of magnitude larger, are set by the coasts. That is the shape free-inertial drift gives: most
  epochs are aided and accurate, and the error concentrates at the ends of the outages.
- **`syn_dead_reckoning` is the unaided reference.** It is only 120 s long, and its
  consistency columns are empty because dead reckoning reports no covariance.
- **The real-data consistency columns say the filters are over-confident.** Their horizontal
  and vertical containment sit well below 0.9973 and their NEES well above 3, while the
  synthetic rows sit much closer to ideal. That is caveat 3 in numbers; `real_degraded` is the
  extreme case for the reason in caveat 4.
- **Several numbers record known defects rather than good behaviour**, which is what a two-sided
  gate is for. Each is annotated with a `note` in `perf_baseline.json`. When one is fixed the
  improvement side trips and asks for the diff that records it.
