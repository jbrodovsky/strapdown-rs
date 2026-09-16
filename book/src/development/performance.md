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

Each scenario also records the number of aligned samples it was scored over and the number of
channel-samples dropped as non-finite, both compared for exact equality before any metric is.
The second is zero everywhere and is recorded because dropping a sample does not lower the
first, but does make the metric it was dropped from look better.

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
   turns out to be enough: the EKF holds 0.97 deg and the ESKF 2.07 deg on `syn_cruise_1hz`
   while the UKF sits at 42.7 deg. That gap is not observability -- all three see the same
   measurements -- it is
   [#371](https://github.com/jbrodovsky/strapdown-rs/issues/371): the UKF means its
   sigma-point attitudes by summing Euler triples linearly, which is not the mean rotation. Modelling a
   real field is tracked in
   [#369](https://github.com/jbrodovsky/strapdown-rs/issues/369).
4. **The consistency columns mean different things on the two sources.** On the synthetic
   scenarios `npes` lands between 2.6 and 4.8 against an ideal of 3.0, which is a real
   measurement of whether a filter believes the right thing. On the real-data scenarios it reaches the
   hundreds, because caveat 2's offset enters the numerator while the covariance in the
   denominator models none of it. Those values are kept as a drift detector, not read as a
   consistency verdict.

## The metrics

| metric | unit | definition | ideal |
|---|---|---|---|
| `horizontal_rmse_m` | m | RMS great-circle distance from truth | 0 |
| `horizontal_cep50_m` | m | median radial error -- the empirical, non-parametric CEP, *not* the Rayleigh form. Nearest-rank, so it is a value some epoch actually had | 0 |
| `horizontal_cep95_m` | m | 95th percentile radial error, same convention | 0 |
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
| `real_clean__ukf` | 5,366 | 23.534 | 23.105 | 34.519 | 37.719 | 2.671 | 0.341 | 1.543 | 0.541 |
| `real_clean__ekf` | 5,366 | 23.553 | 23.188 | 34.472 | 37.804 | 3.655 | -2.398 | 1.575 | 0.869 |
| `real_clean__eskf` | 5,366 | 23.723 | 22.925 | 35.428 | 41.865 | 2.692 | 0.334 | 1.538 | 0.550 |
| `real_sparse_5s__ukf` | 5,366 | 25.067 | 24.085 | 36.614 | 124.727 | 5.656 | 0.776 | 3.301 | 0.671 |
| `real_sparse_5s__eskf` | 5,366 | 25.674 | 23.604 | 37.868 | 149.698 | 5.755 | 0.779 | 3.388 | 0.714 |
| `real_outage_60s__eskf` | 5,366 | 290.644 | 28.701 | 701.329 | 2,349.680 | 4.992 | 0.316 | 12.226 | 0.671 |
| `real_degraded__ukf` | 5,366 | 30.675 | 26.076 | 51.158 | 82.184 | 10.839 | 0.585 | 2.810 | 0.795 |
| `real_degraded__eskf` | 5,366 | 50.601 | 42.344 | 86.848 | 136.791 | 10.847 | 0.564 | 2.795 | 0.806 |
| `syn_cruise_1hz__ukf` | 15,000 | 4.423 | 3.665 | 7.821 | 10.152 | 0.846 | 0.065 | 0.489 | 0.316 |
| `syn_cruise_1hz__ekf` | 15,000 | 4.421 | 3.687 | 7.673 | 10.183 | 0.835 | 0.040 | 0.050 | 0.302 |
| `syn_cruise_1hz__eskf` | 15,000 | 3.412 | 2.835 | 5.964 | 9.284 | 0.836 | 0.069 | 1.170 | 0.239 |
| `syn_outage_60s__ukf` | 15,000 | 266.681 | 5.914 | 762.191 | 2,109.820 | 0.939 | -0.049 | 15.268 | 1.068 |
| `syn_outage_60s__eskf` | 15,000 | 20.224 | 1.454 | 44.447 | 247.874 | 0.847 | 0.067 | 1.067 | 0.243 |
| `syn_dead_reckoning` | 6,000 | 316.860 | 110.128 | 715.856 | 835.634 | 86.591 | -64.255 | 9.363 | 1.896 |
| `real_rbpf_slice__rbpf` | 1,200 | 19.789 | 15.679 | 28.276 | 94.413 | 3.846 | -3.208 | 2.308 | 1.141 |

### Attitude

| scenario | samples | roll RMSE (deg) | pitch RMSE (deg) | yaw RMSE (deg) | geodesic RMSE (deg) |
|---|---:|---:|---:|---:|---:|
| `real_clean__ukf` | 5,366 | 3.448 | 2.999 | 22.773 | 23.210 |
| `real_clean__ekf` | 5,366 | 3.133 | 2.553 | 22.610 | 22.944 |
| `real_clean__eskf` | 5,366 | 3.332 | 2.880 | 26.153 | 26.501 |
| `real_sparse_5s__ukf` | 5,366 | 3.818 | 3.278 | 18.866 | 19.527 |
| `real_sparse_5s__eskf` | 5,366 | 3.813 | 3.171 | 19.737 | 20.344 |
| `real_outage_60s__eskf` | 5,366 | 3.727 | 3.326 | 22.880 | 23.399 |
| `real_degraded__ukf` | 5,366 | 3.525 | 3.143 | 22.347 | 22.829 |
| `real_degraded__eskf` | 5,366 | 3.421 | 3.042 | 26.406 | 26.784 |
| `syn_cruise_1hz__ukf` | 15,000 | 0.887 | 0.637 | 42.665 | 42.683 |
| `syn_cruise_1hz__ekf` | 15,000 | 0.115 | 0.069 | 0.972 | 0.982 |
| `syn_cruise_1hz__eskf` | 15,000 | 1.064 | 0.946 | 2.067 | 2.509 |
| `syn_outage_60s__ukf` | 15,000 | 4.341 | 3.853 | 21.461 | 22.228 |
| `syn_outage_60s__eskf` | 15,000 | 0.242 | 0.205 | 1.181 | 1.223 |
| `syn_dead_reckoning` | 6,000 | 0.468 | 1.106 | 0.494 | 1.301 |
| `real_rbpf_slice__rbpf` | 1,200 | 2.825 | 3.169 | 20.730 | 21.156 |

### Consistency

| scenario | samples | npes (ideal 3.0) | 3-sigma horiz (ideal 0.9973) | 3-sigma vert (ideal 0.9973) |
|---|---:|---:|---:|---:|
| `real_clean__ukf` | 5,366 | 15.896 | 1.000 | 0.466 |
| `real_clean__ekf` | 5,366 | 33.366 | 1.000 | 0.412 |
| `real_clean__eskf` | 5,366 | 1,248.920 | 0.161 | 0.460 |
| `real_sparse_5s__ukf` | 5,366 | 37.562 | 1.000 | 0.211 |
| `real_sparse_5s__eskf` | 5,366 | 338.329 | 0.252 | 0.190 |
| `real_outage_60s__eskf` | 5,366 | 810.816 | 0.274 | 0.390 |
| `real_degraded__ukf` | 5,366 | 272.127 | 1.000 | 0.146 |
| `real_degraded__eskf` | 5,366 | 5,690.210 | 0.057 | 0.148 |
| `syn_cruise_1hz__ukf` | 15,000 | 2.690 | 1.000 | 0.932 |
| `syn_cruise_1hz__ekf` | 15,000 | 2.649 | 1.000 | 0.934 |
| `syn_cruise_1hz__eskf` | 15,000 | 4.795 | 0.996 | 0.933 |
| `syn_outage_60s__ukf` | 15,000 | 3.344 | 1.000 | 0.897 |
| `syn_outage_60s__eskf` | 15,000 | 4.627 | 0.992 | 0.923 |
| `syn_dead_reckoning` | 6,000 | -- | -- | -- |
| `real_rbpf_slice__rbpf` | 1,200 | 183.531 | 0.480 | 0.657 |

## What the table is saying

- **The three healthy filters agree to within 0.2 m on `real_clean`**, which is caveat 2 in
  action: the shared 21 m offset dominates, so the row is a regression detector rather than a
  ranking.
- **`syn_cruise_1hz` is where the filters actually separate**, and no one of them wins. The
  ESKF leads on position (3.41 m against 4.42 m for both others) and loses on velocity
  (1.17 m/s against the EKF's 0.050 m/s and the UKF's 0.489 m/s); the UKF trails on attitude
  for the reason in caveat 3.
- **The outage rows are dominated by attitude, not by position.** 60 s of free inertial turns a
  fraction of a degree of tilt error into hundreds of metres. `syn_outage_60s__eskf` holds
  0.20 deg of pitch and coasts to 20 m; the UKF, which does not hold its attitude through the
  coast, reaches 267 m.
- **The two consistency columns disagree with each other on real data, and that is the
  finding.** The UKF and EKF report a flat 1.000 horizontal containment -- a covariance so
  conservative it cannot be wrong -- beside a vertical containment of 0.47 and 0.41, where
  better than half the altitude errors fall outside three sigma. Their `npes` of 16 and 33 is
  almost entirely that vertical channel. The ESKF is the mirror image: 0.16 horizontal
  containment, an `npes` of 1,249, and caveat 2 in the numerator.
- **Several numbers here therefore record known defects rather than good behaviour**, which is
  what a two-sided gate is for. Each is annotated in the baseline file. When one is fixed the
  improvement side trips and asks for the diff that records it.

## The vertical process-noise retune

These numbers already include one deliberate change the gate was built to catch, and it is
worth reading as a worked example of the workflow.

`DEFAULT_PROCESS_NOISE`'s altitude entry was `1e-4` m² per step -- a 1 cm standard deviation,
a tenth of the horizontal channel's 0.1 m, an asymmetry nothing ever justified. Issue #308 had
fixed the *units* of the horizontal entries and deliberately left this one alone, recording
that whether `1e-4` was the right *tuning* was a separate question. The consistency columns
answered it: at `1e-4` three-sigma altitude containment was 0.88 on the synthetic trajectory
and 0.44 on the recording, against an ideal of 0.9973. More than half the altitude errors on
real data fell outside the uncertainty the filter reported for them.

Three entries could have been blamed. Sweeping each alone over four decades, only the altitude
position entry moves containment the right way -- the vertical-velocity entry is flat and costs
a vertical-velocity RMSE growing from 0.34 to 4.37 m/s, and the accelerometer-bias entry makes
containment *worse* while taking the real 60 s-outage horizontal RMSE from 291 m to 443 m. The
entry is now `POSITION_PROCESS_NOISE_M²`, which makes the position block isotropic in per-step
standard deviation and lands on the knee of the measured curve.

What it bought, and what it cost:

| | before | after |
|---|---:|---:|
| `syn_cruise_1hz__eskf` 3-sigma vertical | 0.880 | **0.933** |
| `syn_cruise_1hz__eskf` `npes` (ideal 3.0) | 5.653 | **4.795** |
| `syn_outage_60s__eskf` horizontal RMSE | 137.66 m | **20.22 m** |
| `syn_outage_60s__eskf` pitch RMSE | 1.500 deg | **0.205 deg** |
| `syn_cruise_1hz__eskf` vertical RMSE | **0.665 m** | 0.836 m |
| `syn_outage_60s__ukf` horizontal RMSE | **221.67 m** | 266.68 m |

The vertical RMSE cost is the honest price: the old value bought that number by under-reporting
its own error, and on the reference recording it is not paid at all -- vertical RMSE there
improved slightly. The `syn_outage_60s__ukf` row is recorded but not endorsed: across the sweep
its horizontal RMSE runs 222, 174, 267, 344 m while its yaw runs 105, 98, 21, 34 deg, so
position degrades exactly where attitude improves fivefold. That row is
[#371](https://github.com/jbrodovsky/strapdown-rs/issues/371) coasting an outage, and it does
not respond monotonically to this or any other knob.
