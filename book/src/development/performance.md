# Performance Baselines

"Performance" on this page means **navigation accuracy**, not wall-clock speed. How far the
solution was from truth, and whether the filter's covariance knew it. Execution time, memory
and scalability are tracked nowhere yet; see [Building and Testing](./building.md) for what
remains.

Every number below is recorded in `core/tests/perf_baseline.json` and checked on every CI run
by `core/tests/perf_baseline.rs`. The check fails in **both** directions:

- a metric more than **10%** worse than its baseline is a regression, and CI goes red;
- a metric more than **25%** better is *also* a failure, and the message asks for the baseline
  to be re-blessed. An improvement nobody records is one the next change can silently undo.

Re-bless with `UPDATE_PERF_BASELINE=1 cargo test -p strapdown-core --test perf_baseline`, then
commit the diff; `cargo perf` runs the suite and prints the table without writing anything.
[CONTRIBUTING.md](https://github.com/jbrodovsky/strapdown-rs/blob/main/CONTRIBUTING.md) has the
workflow.

## Read these four caveats first

The numbers are meaningless without them, and none of them is a defect in the harness.

1. **On the real-data scenarios, "truth" is the GNSS fix -- which is also the filters' aiding
   source.** They measure agreement with the aid, not independent accuracy, and cannot fall
   below the receiver's own 3.81 m horizontal noise however good the filter is. The `syn_*`
   scenarios are scored against an exact synthetic trajectory and have no such floor, which is
   why their horizontal figures are five times smaller.
2. **Every horizontal figure carries a one-step propagation offset.** `sim::run_closed_loop`
   pushes an output row *after* applying the next event, so the row labelled $t_k$ holds a
   state already propagated through the first event of $t_{k+1}$. On the 1 Hz recording at
   21.19 m/s that is 21.2 m of along-track error on its own -- very nearly the whole of the
   ~23.5 m the `real_clean` rows show, and the reason all three filters land within 0.2 m of
   each other there rather than spreading out by tuning. Tracked in
   [#367](https://github.com/jbrodovsky/strapdown-rs/issues/367); the synthetic scenarios run
   at 50 Hz specifically so the same offset is about 1 m.
3. **The synthetic scenarios carry no magnetometer, and the yaw column says which filters need
   one.** Nothing aids heading there but the GNSS velocity fix on a moving trajectory, which
   turns out to be enough: the EKF holds 0.97 deg and the ESKF 2.17 deg on `syn_cruise_1hz`
   while the UKF sits at 48.7 deg. That gap is not observability -- all three see the same
   measurements -- it is the UKF averaging sigma-point Euler angles linearly. Modelling a
   real field is tracked in
   [#369](https://github.com/jbrodovsky/strapdown-rs/issues/369).
4. **The consistency columns mean different things on the two sources.** On the synthetic
   scenarios `npes` lands at 3.4 to 5.7 against an ideal of 3.0, which is a real measurement of
   whether a filter believes the right thing. On the real-data scenarios it reaches the
   hundreds, because caveat 2's offset enters the numerator while the covariance in the
   denominator models none of it. Those values are kept as a drift detector, not read as a
   consistency verdict.

## The metrics

| metric | unit | definition | ideal |
|---|---|---|---|
| `horizontal_rmse_m` | m | RMS great-circle distance from truth | 0 |
| `horizontal_cep50_m` | m | median radial error -- the empirical, non-parametric CEP, *not* the Rayleigh form | 0 |
| `horizontal_cep95_m` | m | 95th percentile radial error | 0 |
| `horizontal_max_m` | m | largest single-sample radial error | 0 |
| `vertical_rmse_m` | m | RMS altitude error | 0 |
| `vertical_bias_m` | m | **signed** mean altitude error -- catches a slow one-sided drift while it is still small | 0 |
| `velocity_horizontal_rmse_mps` | m/s | RMS north/east velocity error | 0 |
| `velocity_vertical_rmse_mps` | m/s | RMS vertical velocity error | 0 |
| `roll_rmse_deg`, `pitch_rmse_deg` | deg | RMS per-axis error, wrapped onto $[-\pi, \pi]$ | 0 |
| `yaw_rmse_deg` | deg | RMS heading error, wrapped | 0 |
| `attitude_geodesic_rmse_deg` | deg | RMS rotation angle of $R_{est}^{-1} R_{true}$ -- the only attitude error independent of the Euler sequence | 0 |
| `npes_position` | -- | mean normalized position error squared, $\overline{\epsilon} = \frac{1}{N}\sum_k \left(\frac{e_{lat}^2}{P_{lat}} + \frac{e_{lon}^2}{P_{lon}} + \frac{e_{alt}^2}{P_{alt}}\right)$ | 3.0 |
| `containment_3sigma_horizontal` | fraction | share of latitude and longitude samples inside $\pm 3\sigma$ | 0.9973 |
| `containment_3sigma_vertical` | fraction | share of altitude samples inside $\pm 3\sigma$ | 0.9973 |

`npes_position` is deliberately **not** called NEES. `NavigationResult` keeps only the
covariance diagonal, so the true $e^T P^{-1} e$ is not computable from it; this form equals the
NEES only when the position block is diagonal, and is optimistic when it is positively
correlated. Read it beside the containment row, never alone -- together they separate the two
failure modes either one alone hides:

| containment | `npes_position` | diagnosis |
|---|---|---|
| ~1.0 | far below 3 | over-conservative: the filter is right but does not believe it |
| below 0.99 | far above 3 | over-confident -- the dangerous one |
| ~0.9973 | ~3 | consistent |

There is no Wasserstein-2 metric, though the original issue floated one. Against a Dirac at
truth it reduces to squared error plus total variance, which is strictly less information than
reporting the error and the consistency separately. Against the empirical error distribution it
is not a distance between samples of anything, because errors along one trajectory are strongly
autocorrelated. The CEP columns carry the "the shape of the error distribution moved" signal
instead, and a human reading a failure message can interpret them. The reasoning is recorded in
full in the `strapdown::metrics` module documentation.

## Scenarios

`real` is `core/tests/test_data.csv`: 5,366 samples at 1 Hz over 89 minutes, ENU, from a
consumer MEMS IMU. `syn` is `sim::generate_synthetic`: 300 s at 50 Hz, NED, 50 m/s off-axis
cruise, consumer-grade IMU, seed 42, scored against its own exact truth.

| id | data | GNSS | estimators |
|---|---|---|---|
| `real_clean` | real | every epoch | UKF, EKF, ESKF |
| `real_sparse_5s` | real | every 5 s | UKF, ESKF |
| `real_outage_60s` | real | 120 s on, 60 s off | ESKF |
| `real_degraded` | real | every epoch, AR(1) position and velocity fault | UKF, ESKF |
| `syn_cruise_1hz` | syn | every 1 s | UKF, EKF, ESKF |
| `syn_outage_60s` | syn | 60 s on, 60 s off | UKF, ESKF |
| `syn_dead_reckoning` | syn, first 120 s | none | unaided mechanization |
| `real_rbpf_slice` | real, first 1,200 samples | every epoch | RBPF, 500 particles |

Two scenarios are deliberately absent. **Full-run dead reckoning** ends millions of metres out
by way of an altitude at which the `r_e + altitude` denominator in `earth::transport_rate` is
within a few percent of zero, where finiteness is a floating-point accident that has already
differed between Windows and Linux; the 120 s synthetic window is the safe version of the same
measurement. **A second particle-filter scenario** is affordable in principle and not in CI
time: the RBPF is the only estimator here whose cost is measured in seconds.

The whole suite runs in about 5.5 seconds on Linux and is not gated behind a feature or an
`#[ignore]`, so it runs on all three CI platforms. That is deliberate: the three legs are a
free portability canary for the accuracy numbers themselves, which this project has been bitten
by before.

## Current values

Measured on `ubuntu-latest`, rustc 1.91. Regenerate with `cargo perf`.

### Position and velocity

| scenario | samples | horiz RMSE (m) | CEP50 (m) | CEP95 (m) | horiz max (m) | vert RMSE (m) | vert bias (m) | horiz vel RMSE (m/s) | vert vel RMSE (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `real_clean__ukf` | 5,366 | 23.534 | 23.108 | 34.517 | 37.723 | 2.702 | 0.339 | 1.543 | 0.545 |
| `real_clean__ekf` | 5,366 | 23.553 | 23.193 | 34.472 | 37.804 | 3.663 | -2.379 | 1.576 | 0.839 |
| `real_clean__eskf` | 5,366 | 23.723 | 22.924 | 35.425 | 41.870 | 2.724 | 0.333 | 1.538 | 0.553 |
| `real_sparse_5s__ukf` | 5,366 | 25.069 | 24.094 | 36.614 | 125.123 | 5.665 | 0.772 | 3.304 | 0.672 |
| `real_sparse_5s__eskf` | 5,366 | 25.678 | 23.604 | 37.870 | 150.052 | 5.765 | 0.775 | 3.391 | 0.715 |
| `real_outage_60s__eskf` | 5,366 | 291.407 | 28.699 | 705.003 | 2,338.920 | 5.009 | 0.313 | 12.266 | 0.670 |
| `real_degraded__ukf` | 5,366 | 30.675 | 26.083 | 51.159 | 82.186 | 10.864 | 0.585 | 2.811 | 0.808 |
| `real_degraded__eskf` | 5,366 | 50.601 | 42.348 | 86.844 | 136.795 | 10.873 | 0.563 | 2.796 | 0.818 |
| `syn_cruise_1hz__ukf` | 15,000 | 4.424 | 3.672 | 7.825 | 10.151 | 0.681 | 0.065 | 0.496 | 0.381 |
| `syn_cruise_1hz__ekf` | 15,000 | 4.421 | 3.688 | 7.679 | 10.184 | 0.667 | 0.039 | 0.049 | 0.368 |
| `syn_cruise_1hz__eskf` | 15,000 | 3.411 | 2.832 | 5.966 | 9.262 | 0.665 | 0.068 | 1.173 | 0.322 |
| `syn_outage_60s__ukf` | 15,000 | 221.670 | 6.335 | 412.956 | 2,337.140 | 0.763 | -0.024 | 13.623 | 0.595 |
| `syn_outage_60s__eskf` | 15,000 | 137.659 | 1.507 | 268.475 | 1,885.740 | 0.671 | 0.065 | 8.038 | 0.335 |
| `syn_dead_reckoning` | 6,000 | 316.860 | 110.233 | 715.856 | 835.634 | 86.591 | -64.255 | 9.363 | 1.896 |
| `real_rbpf_slice__rbpf` | 1,200 | 19.789 | 15.687 | 28.276 | 94.413 | 3.846 | -3.208 | 2.308 | 1.141 |

### Attitude

| scenario | samples | roll RMSE (deg) | pitch RMSE (deg) | yaw RMSE (deg) | geodesic RMSE (deg) |
|---|---:|---:|---:|---:|---:|
| `real_clean__ukf` | 5,366 | 3.442 | 3.000 | 22.772 | 23.208 |
| `real_clean__ekf` | 5,366 | 3.134 | 2.553 | 22.611 | 22.945 |
| `real_clean__eskf` | 5,366 | 3.334 | 2.882 | 26.149 | 26.497 |
| `real_sparse_5s__ukf` | 5,366 | 3.817 | 3.280 | 18.874 | 19.535 |
| `real_sparse_5s__eskf` | 5,366 | 3.815 | 3.172 | 19.732 | 20.339 |
| `real_outage_60s__eskf` | 5,366 | 3.734 | 3.326 | 22.887 | 23.408 |
| `real_degraded__ukf` | 5,366 | 3.518 | 3.145 | 22.351 | 22.831 |
| `real_degraded__eskf` | 5,366 | 3.423 | 3.043 | 26.398 | 26.776 |
| `syn_cruise_1hz__ukf` | 15,000 | 0.977 | 0.512 | 48.735 | 48.751 |
| `syn_cruise_1hz__ekf` | 15,000 | 0.114 | 0.069 | 0.973 | 0.982 |
| `syn_cruise_1hz__eskf` | 15,000 | 1.068 | 0.955 | 2.175 | 2.603 |
| `syn_outage_60s__ukf` | 15,000 | 3.509 | 3.848 | 105.085 | 105.204 |
| `syn_outage_60s__eskf` | 15,000 | 1.231 | 1.500 | 1.060 | 2.211 |
| `syn_dead_reckoning` | 6,000 | 0.468 | 1.106 | 0.494 | 1.301 |
| `real_rbpf_slice__rbpf` | 1,200 | 2.825 | 3.169 | 20.730 | 21.156 |

### Consistency

| scenario | samples | npes (ideal 3.0) | 3-sigma horiz (ideal 0.9973) | 3-sigma vert (ideal 0.9973) |
|---|---:|---:|---:|---:|
| `real_clean__ukf` | 5,366 | 17.318 | 1.000 | 0.451 |
| `real_clean__ekf` | 5,366 | 35.967 | 1.000 | 0.401 |
| `real_clean__eskf` | 5,366 | 1,250.370 | 0.161 | 0.441 |
| `real_sparse_5s__ukf` | 5,366 | 39.208 | 1.000 | 0.209 |
| `real_sparse_5s__eskf` | 5,366 | 340.081 | 0.252 | 0.186 |
| `real_outage_60s__eskf` | 5,366 | 812.289 | 0.275 | 0.377 |
| `real_degraded__ukf` | 5,366 | 291.252 | 1.000 | 0.142 |
| `real_degraded__eskf` | 5,366 | 5,709.580 | 0.057 | 0.143 |
| `syn_cruise_1hz__ukf` | 15,000 | 3.565 | 1.000 | 0.880 |
| `syn_cruise_1hz__ekf` | 15,000 | 3.504 | 1.000 | 0.882 |
| `syn_cruise_1hz__eskf` | 15,000 | 5.653 | 0.996 | 0.880 |
| `syn_outage_60s__ukf` | 15,000 | 4.375 | 1.000 | 0.845 |
| `syn_outage_60s__eskf` | 15,000 | 5.552 | 0.992 | 0.869 |
| `syn_dead_reckoning` | 6,000 | -- | -- | -- |
| `real_rbpf_slice__rbpf` | 1,200 | 183.531 | 0.480 | 0.657 |

## What the table is saying

- **The three healthy filters agree to within 0.2 m on `real_clean`**, which is caveat 2 in
  action: the shared 21 m offset dominates, so the row is a regression detector rather than a
  ranking.
- **`syn_cruise_1hz` is where the filters actually separate**, and no one of them wins. The
  ESKF leads on position (3.41 m against 4.42 m for both others) and loses on velocity
  (1.17 m/s against the EKF's 0.049 m/s and the UKF's 0.496 m/s); the UKF trails on attitude
  for the reason in caveat 3.
- **The outage rows are dominated by attitude, not by position.** 60 s of free inertial turns a
  few degrees of heading error into hundreds of metres, which is why `syn_outage_60s__eskf` at
  1.06 deg of yaw coasts to 137 m while the UKF at 105 deg reaches 222 m.
- **The two consistency columns disagree with each other on real data, and that is the
  finding.** The UKF and EKF report a flat 1.000 horizontal containment -- a covariance so
  conservative it cannot be wrong -- beside a vertical containment of 0.40, where six samples
  in ten fall outside three sigma. Their `npes` of 17 and 36 is almost entirely that vertical
  channel. The ESKF is the mirror image: 0.16 horizontal containment, an `npes` of 1,250, and
  caveat 2 in the numerator.
- **Several numbers here therefore record known defects rather than good behaviour**, which is
  what a two-sided gate is for. Each is annotated in the baseline file. When one is fixed the
  improvement side trips and asks for the diff that records it.
