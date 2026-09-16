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
2. **On a full-rate real-data row, the filter is scored against a fix it has already been
   given.** This used to read differently: every horizontal figure carried a one-step
   propagation offset, because `sim::run_closed_loop` pushed each row *after* applying the
   next event, so the row labelled $t_k$ held a state already propagated to $t_{k+1}$. At
   1 Hz and 21.19 m/s that was 21.2 m of along-track error -- very nearly the whole of the
   ~23.5 m the `real_clean` rows then showed, and the reason all three filters landed within
   0.2 m of each other. [#367](https://github.com/jbrodovsky/strapdown-rs/issues/367) fixed
   it, and every horizontal metric improved at once.

   What that uncovered is caveat 1 with nothing left diluting it. A row at $t_k$ now contains
   $t_k$'s GNSS update, so on a `PassThrough` schedule these columns measure how completely a
   filter absorbs its own aiding: `real_clean__ukf` reads 0.014 m and `real_clean__ekf`
   0.0001 m, both far below the receiver's own 3.81 m. That is not accuracy. It is a Kalman
   gain of about 1 -- see [#373](https://github.com/jbrodovsky/strapdown-rs/issues/373), where
   an absolute covariance floor in radian units leaves both filters copying their fixes rather
   than filtering them. The ESKF, which floors its covariance *relatively*, sits at 5.1 m.

   **The real-data rows that still measure navigation are the ones where the filter has to
   predict between fixes** -- `real_sparse_5s`, `real_outage_60s`, `real_degraded` -- together
   with the `syn_*` scenarios and their exact independent truth.
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

Measured on `ubuntu-latest`, rustc 1.91. Read-only with `cargo perf`; re-blessed with
`UPDATE_PERF_BASELINE=1 cargo test -p strapdown-core --test perf_baseline`.

The three tables below are **generated** from `core/tests/perf_baseline.json` by the same test
that gates it, and the gate fails if they drift from it. They used to be maintained here by
hand, which meant every re-bless was a command plus 45 transcribed rows, with nothing checking
the transcription -- and a wrong number on this page is harder to notice than a wrong number in
the gated file, because nothing asserts it (#380). Edit `baseline-tables.md` and the next run
will tell you off; change the numbers by re-blessing.

{{#include ./baseline-tables.md}}

## What the table is saying

- **The `real_clean` horizontal column is no longer a navigation measurement.** The UKF reads
  0.014 m and the EKF 0.0001 m against a reference whose own noise is 3.81 m -- caveat 2. Both
  are reproducing the fix they were handed one event earlier. The ESKF's 5.1 m is the only
  figure in that column that means anything, and it is the only one of the three whose
  covariance floor is relative rather than absolute.
- **`syn_cruise_1hz` is where the filters actually separate**, and no one of them wins. The
  ESKF leads on position and loses on velocity; the UKF trails on attitude for the reason in
  caveat 3. These rows are scored against exact truth, so nothing here is circular.
- **The outage rows are dominated by attitude, not by position.** 60 s of free inertial turns a
  fraction of a degree of tilt error into hundreds of metres. `syn_outage_60s__eskf` holds
  0.20 deg of pitch and coasts to 20 m; the UKF, which does not hold its attitude through the
  coast, reaches 267 m.
- **The two consistency columns disagree with each other on real data, and that is the
  finding.** The UKF and EKF still report a flat 1.000 horizontal containment -- a covariance
  so conservative it cannot be wrong -- beside a vertical containment of 0.40 and 0.42, where
  better than half the altitude errors fall outside three sigma. Both halves of that are
  [#373](https://github.com/jbrodovsky/strapdown-rs/issues/373)'s absolute covariance floor:
  201 m of fabricated horizontal sigma is what makes the horizontal figure unfalsifiable, and
  the same constant is a rounding error in the vertical channel, which is therefore left
  bare.
- **The ESKF's numbers moved furthest when #367 landed**, and in the right direction:
  horizontal containment 0.161 to 0.497, `npes` 1,249 to 68. That is the offset leaving the
  numerator. It is still not consistent -- 0.497 against an ideal of 0.9973 -- and what
  remains is a real finding rather than an artefact.
- **`real_rbpf_slice__rbpf`'s `npes` is not gated.** It reads 5.7e25 because the particle
  cloud collapses to a horizontal sigma of nanometres on 17 of 1,200 epochs, which a mean of
  $e^2/P$ cannot survive. Exposed rather than caused by #367, which stopped sampling the cloud
  one propagation step after each fix had re-inflated it. Tracked as
  [#385](https://github.com/jbrodovsky/strapdown-rs/issues/385).
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
