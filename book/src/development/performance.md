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

## Read these six caveats first

The numbers are meaningless without them. The first four are properties of the measurement
rather than defects in the harness; the fifth is a defect, in the gate.

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
   filter absorbs its own aiding.

   That absorption used to be total: `real_clean__ukf` read 0.014 m and `real_clean__ekf`
   0.0001 m, far below the receiver's own 3.81 m. Not accuracy — a Kalman gain of about 1,
   caused by an absolute covariance floor applied to a latitude variance in radians².
   [#373](https://github.com/jbrodovsky/strapdown-rs/issues/373) **fixed it**, and the three
   filters now read **4.78, 5.22 and 5.12 m** — above the receiver's own noise rather than
   reproducing it, and in agreement with the ESKF, which had floored its covariance relatively
   since #266.

   **The real-data rows that still measure navigation are the ones where the filter has to
   predict between fixes** -- `real_sparse_5s`, `real_outage_60s`, `real_degraded` -- together
   with the `syn_*` scenarios and their exact independent truth.
3. **The synthetic scenarios now carry a real magnetic field, and the yaw column says which
   filters can use it.** `generate_synthetic` evaluates the World Magnetic Model at each
   epoch's position and date and rotates it into the body frame through the truth attitude
   ([#369](https://github.com/jbrodovsky/strapdown-rs/issues/369)); before that it wrote
   nothing, and heading was observable only through the GNSS velocity fix. With a 1 Hz heading
   aid all three filters now hold a fraction of a degree of yaw on `syn_cruise_1hz` -- the EKF
   0.28, the ESKF 0.18, the UKF 0.16, the last down from 108.3 deg without a magnetometer at
   all.

   **`syn_outage_60s__ukf` used to be where that stopped, and the stated reason was wrong.**
   With the field added that row read 72.68 deg of yaw, 809 m horizontal and a NEES of 44.67,
   against the ESKF's 0.255 deg and 24.0 m on the identical stream, and this page cited that
   pair as evidence for [#371](https://github.com/jbrodovsky/strapdown-rs/issues/371). It was
   not evidence for anything: the three filter constructors shipped initial gyro-bias variances
   five orders of magnitude apart -- `1e-3` for the UKF and the EKF against `1e-8` for the
   ESKF -- so the two filters were never handed the same prior. `1e-3` is a gyro-bias sigma of
   1.81 deg/s against the 0.028 deg/s a consumer-grade part actually has.

   Sizing all three from the IMU grade takes that row to **0.222 deg of yaw, 27.4 m
   horizontal, NEES 4.21 and 3-sigma horizontal containment 0.997** against an ideal of 0.9973
   -- past the ESKF's own numbers on the same stream, so the UKF is not incapable of carrying
   an attitude through a coast. `syn_outage_60s__eskf` moved the other way and for the same
   reason: its `1e-8` was about 5x *tighter* than the truth it was modelling, and correcting it
   costs 24.0 -> 26.3 m of position and 0.255 -> 0.292 deg of yaw while taking its NEES
   4.64 -> 3.87 and its vertical containment 0.866 -> 0.925. Less accurate, more honest.

   #371 was a real defect -- a linear mean over sigma-point Euler triples sends 31 rotations
   inside a 0.24 deg cone to a point 34 deg outside it -- but it is not what this row was
   measuring. Fixing it, so the UKF perturbs, averages, correlates and injects attitude on
   SO(3), moved `syn_outage_60s__ukf` yaw by **0.007 deg** (0.222 to 0.230) and improved its
   position (27.39 to 27.17 m). That near-nil result on the rows with exact truth is the
   finding. The two `__ekf` rows did not move at all for the prior change either, which is
   also a finding rather than a null result: the EKF's bias states are structurally inert
   ([#394](https://github.com/jbrodovsky/strapdown-rs/issues/394)), so its prior cannot be
   falsified by any measurement.

   **The `real_*` UKF rows are the exception, and the yaw column there cannot adjudicate it.**
   `real_clean__ukf` yaw went 22.72 to 25.83 deg and `real_degraded__ukf` 22.18 to 26.01 when
   the chart was fixed, which reads as a 14-17% regression. Three measurements say what it
   actually is. The UKF's disagreement with the **ESKF** -- the other filter here that does its
   attitude arithmetic on a manifold, and which this change does not touch -- fell from 8.40 to
   2.57 deg RMS on the same recording. The recording's own magnetometer, levelled by its own
   roll and pitch, disagrees with its yaw column by **16.8 deg RMS after removing every
   constant offset**, which is the floor for any filter aided by it, so the 3 deg the UKF moved
   is inside what that column can resolve. And on the synthetic rows, where truth is exact, the
   change is neutral. Caveat 1 again, in the attitude channel.
4. **The consistency columns mean different things on the two sources.** On the synthetic
   scenarios `npes` lands between 3.9 and 12.1 against an ideal of 3.0, which is a real
   measurement of whether a filter believes the right thing. On the real-data scenarios it reaches the
   hundreds, because caveat 2's offset enters the numerator while the covariance in the
   denominator models none of it. Those values are kept as a drift detector, not read as a
   consistency verdict.

   The synthetic figures roughly doubled when
   [#375](https://github.com/jbrodovsky/strapdown-rs/issues/375) scheduled the barometer, and
   the vertical columns say why: what a normalised statistic now sees is a systematic altitude
   bias of -0.6 to -2.0 m that nothing models. That is
   [#372](https://github.com/jbrodovsky/strapdown-rs/issues/372), measured rather than masked.

5. **The baseline is blessed on one platform and gated on three.** `rust.yml` runs this suite
   on Linux, macOS and Windows, and nothing in the gate knows that -- there is no
   cross-platform tolerance floor. For most rows it does not matter: the same commit agrees to
   about 1% across platforms. It matters where a metric is a *tail* statistic of something
   unstable. `real_rbpf_slice__rbpf`'s `horizontal_cep95_m` measured 34.02 m on Linux against
   24.33 m on Windows -- a 28.5% spread against a 25% improve band -- so a Linux bless failed
   the Windows leg of a run that was behaving identically. That metric is ungated with the
   numbers recorded in its note; the general problem is
   [#386](https://github.com/jbrodovsky/strapdown-rs/issues/386).

   **Two parts of that are now closed.** The baseline records the operating system it was
   blessed on (`blessed_on`), and the improve-side failure no longer tells a contributor their
   baseline is stale when the run is simply on a different platform -- it names both platforms
   and says to check the blessing one first. Re-blessing on the wrong leg was the specific
   damage the old message caused.

   **The tolerance floor is deliberately not set yet, because nobody has measured it.** Every
   matrix leg now writes its raw numbers with `PERF_EMIT_MEASURED` and uploads them, and the
   advisory `Cross-platform accuracy spread` job prints the per-metric spread across all three.
   Setting a floor before reading that output would be gating on a guess, which is the mistake
   #386 exists to correct. Read the job's table, then set the floor from it.

   Note also that the 28.5% figure above predates
   [#385](https://github.com/jbrodovsky/strapdown-rs/issues/385). That row's tail *was* the
   epochs where the particle cloud collapsed, and it no longer collapses, so the spread may
   already be much smaller. It has not been re-measured. Until then, **check a Windows run
   before calling a re-bless done.**
6. **The `syn_*` rows run at 50 Hz and the `real_*` rows at 1 Hz, and the aiding sensors no
   longer follow that.** Until
   [#375](https://github.com/jbrodovsky/strapdown-rs/issues/375) the barometer and the
   magnetometer were emitted once per record, outside the scheduler, so their update rate was
   the log's: 1 Hz on `test_data.csv` and 50 Hz on the synthetic trajectories, where fifty
   pressure readings a second each entered the filter as an independent fix. Both channels are
   now scheduled at 1 Hz by default. Every `real_*` row came through bit-identical -- they were
   already at 1 Hz -- and all five aided `syn_*` rows moved, almost entirely in the vertical
   channel: horizontal RMSE on `syn_cruise_1hz` moved by under half a percent. Read the
   synthetic vertical columns as a 1 Hz barometer's, not a 50 Hz one's, and the yaw
   columns as a 1 Hz magnetometer's.

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
| `nees_position` | -- | mean normalized estimation error squared over the full 3x3 position block, $\frac{1}{N}\sum_k e_k^\top P_k^{-1} e_k$ | 3.0 |
| `npes_position` | -- | the same, computed as though $P$ were diagonal: $\overline{\epsilon} = \frac{1}{N}\sum_k \left(\frac{e_{lat}^2}{P_{lat}} + \frac{e_{lon}^2}{P_{lon}} + \frac{e_{alt}^2}{P_{alt}}\right)$ | 3.0 |
| `containment_3sigma_horizontal` | fraction | share of latitude and longitude samples inside $\pm 3\sigma$ | 0.9973 |
| `containment_3sigma_vertical` | fraction | share of altitude samples inside $\pm 3\sigma$ | 0.9973 |

`nees_position` is the real statistic. It became computable in
[#376](https://github.com/jbrodovsky/strapdown-rs/issues/376), which stopped `NavigationResult`
discarding the position block's off-diagonal terms; before that the crate could only report the
diagonal-only `npes_position`, which equals the NEES when the position states are uncorrelated
and is **optimistic** when they are not.

Both are kept. `npes_position` stays so the baseline it has accumulated remains comparable
across that change, not as a claim about consistency.

**How much did the diagonal-only form actually understate?** Less than #376 feared, and
precisely where you would expect:

| scenario | `npes` | `nees` | ratio |
|---|---:|---:|---:|
| every row with GNSS every epoch | — | — | **1.00–1.01** |
| `real_outage_60s__eskf` | 73.43 | 83.04 | **1.13** |
| `syn_outage_60s__ukf` | 6.54 | 6.82 | **1.04** |

Where the filter is aided every epoch the two agree to within 1%, so the historical `npes`
numbers were not misleading. The gap opens on the **outage** rows -- `syn_outage_60s__ukf` read
15.12 against 18.45, a ratio of 1.22, until #375 brought that coast back under control -- and the mechanism is the
obvious one: during a free-inertial coast the position error accumulates along a correlated
direction with no fix to break it up, and a diagonal-only statistic cannot see that. The
contrived worst case is far larger — at a latitude-longitude correlation of 0.9 the two read
2.0 and 20.0 — so the modest ratios here are a measurement about these trajectories, not a
general reassurance.

Read either beside the containment row, never alone -- together they separate the two failure
modes either one alone hides:

| containment | `nees_position` | diagnosis |
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

- **The `real_clean` horizontal column measures agreement with the aid, not accuracy.** All
  three filters now read 4.78, 5.22 and 5.12 m against a reference whose own noise is 3.81 m,
  so none of them is below its own reference any more — #373 fixed the absolute covariance
  floor that had the UKF and EKF reproducing the fix they were handed (0.014 m and 0.0001 m).
  What remains is caveat 1's circularity, which no fix to the filters can remove: the score is
  against the aiding source itself.
- **`syn_cruise_1hz` is where the filters actually separate**, and no one of them wins. The
  ESKF leads on vertical error, the EKF on horizontal, and all three now hold a fraction of a
  degree of yaw — the UKF's, once the worst column in the table at 108.3 deg with no
  magnetometer and 2.01 deg with one but a mis-sized bias prior, is now the best of the three.
  These rows are scored against exact truth, so nothing here is circular.
- **The outage rows are dominated by attitude, not by position.** 60 s of free inertial turns a
  fraction of a degree of tilt error into tens of metres. Both filters now hold about a quarter
  of a degree of yaw through the coast and land within a couple of metres of each other, around
  26–27 m. The UKF used to reach 809 m on this row; that was its initial gyro-bias prior, not
  its attitude algebra, and the story is in caveat 3.
- **The two consistency columns disagree with each other on real data, and that is the
  finding.** Horizontal three-sigma containment sits near 0.50 across the three filters and
  vertical near 0.40, where better than half the altitude errors fall outside three sigma.
  Neither is the old flat 1.000 — [#373](https://github.com/jbrodovsky/strapdown-rs/issues/373)
  removed the absolute covariance floor that made the horizontal figure unfalsifiable, so what
  is left in both columns is a real measurement of over-confidence rather than an artefact. The
  vertical half is [#372](https://github.com/jbrodovsky/strapdown-rs/issues/372).
- **The ESKF's numbers moved furthest when #367 landed**, and in the right direction:
  horizontal containment 0.161 to 0.497, `npes` 1,249 to 68. That is the offset leaving the
  numerator. It is still not consistent -- 0.497 against an ideal of 0.9973 -- and what
  remains is a real finding rather than an artefact.
- **`real_rbpf_slice__rbpf`'s consistency metrics are not gated.** `npes` reads 5.7e25 and the
  real `nees` larger still -- both meaningless, and deliberately quoted without a ratio -- because the particle
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

`DEFAULT_PROCESS_NOISE_DENSITY`'s altitude entry was `1e-4` -- a 1 cm standard deviation,
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
entry is now `POSITION_PROCESS_NOISE_M_PER_ROOT_S²`, which makes the position block isotropic in
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
improved slightly. (These synthetic figures have since moved again, and further, under #375 --
see caveat 6. The numbers in this table are the ones that change measured, kept as its
record.) The `syn_outage_60s__ukf` row is recorded but not endorsed: across the sweep
its horizontal RMSE runs 222, 174, 267, 344 m while its yaw runs 105, 98, 21, 34 deg, so
position degrades exactly where attitude improves fivefold. That row is
[#371](https://github.com/jbrodovsky/strapdown-rs/issues/371) coasting an outage, and it does
not respond monotonically to this or any other knob.
