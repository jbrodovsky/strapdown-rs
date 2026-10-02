# Fault Simulation

This chapter is the experiment-design rationale behind the recipes in `conf/`: which GNSS
conditions are worth simulating, what the published measurements say those conditions look
like, and which knobs in `GnssFaultModel` correspond to which. Every variant, field, default
and flag is listed in [Schedulers and Faults Reference](./scenarios.md); this page explains why
the recipes use the values they do.

Two pieces of the library are involved, and they are deliberately separate:

- a **scheduler** (`MeasurementScheduler`) decides *when* a GNSS fix reaches the filter;
- a **fault model** (`GnssFaultModel`) decides *what* is wrong with each fix that does.

Both live in the `[aiding]` section of a configuration file. That section was called
`[gnss_degradation]` before the v1.0 freeze, and the old name is still accepted as an alias,
which is why the recipes under `conf/` still spell it that way.

## What jamming actually does

The dominant, repeatedly measured effect of jamming is **loss of fix**, not graceful
degradation. At Jammertest, the open field trial run annually at Andøya by Norwegian public
agencies, the jamming signal was in most runs strong enough to cause complete GNSS signal
loss, and a smartphone roughly ten metres from a jammer lost lock on benign signals entirely.
Receivers reacquired when terrain or buildings shadowed them from the jammer and lost lock
again on returning to line of sight.

That matters for how a scenario is built. A receiver inside the jammed area does not report a
noisier position; it reports nothing. A model that keeps delivering fixes with more error on
them describes the *fringe* of the jammed area, not the inside of it. The inside is what an
inertial system, or a geophysical aid, exists to carry you through.

The scale is not marginal. IATA recorded a 220% rise in GPS signal-loss events between 2021
and 2024; European flights encountering GNSS interference rose from roughly 200 per day in Q1
2024 to around 900 per day in Q2, and EUROCONTROL logged more than 2,500 jamming events in
2024 alone (see the sources at the end of this page).

## The interference tiers

| Tier | What it models | Scheduler | Fault | Recipes |
|---|---|---|---|---|
| **Denial** | inside a jammed area: no fix at all | `duty_cycle` | `none` | none shipped; use `--sched duty` |
| **Degraded** | the fringe of a jammed area | `fixed_interval` | `degraded` | `conf/*_degraded.toml` |
| **Spoof** | a deceptive signal | either | `slow_bias` / `hijack` | none shipped; use `--fault slowbias` or `hijack` |

The experiment matrix in `conf/` carries one interference profile, Degraded. Keeping one
profile, used identically everywhere a geophysically aided run is scored, was worth more than
covering both regimes at the cost of doubling the config matrix and the pairing logic that
keeps it honest (see [Choosing a baseline](#choosing-a-baseline-to-compare-against) below).

The Denial tier is still one command away. This gives 30 s of fixes in every 150 s, with
availability as the only variable:

```bash
strapdown-sim cl -i input.csv -o denial.csv \
  --sched duty --on-s 30 --off-s 120 --duty-phase-s 30 --fault none
```

`--duty-phase-s` is the length of an initial ON window. Without it the cycle starts with its
OFF window, so the filter would begin with a 120 s outage before it had seen a single fix; see
[the duty-cycle timeline](./scenarios.md#dutycycle).

### Degraded

`conf/*_degraded.toml` keeps **one fix every 60 s** (`fixed_interval`, `interval_s = 60.0`)
from a roughly 1 Hz receiver, and puts AR(1)-correlated error on each one, with the advertised
accuracies inflated:

```toml
[gnss_degradation.scheduler]      # the same as [aiding.scheduler]
kind = "fixed_interval"
interval_s = 60.0
phase_s = 0.0

[gnss_degradation.fault]          # the same as [aiding.fault]
kind = "degraded"
rho_pos = 0.99          # ignored while tau_pos_s is set
sigma_pos_m = 35.0      # steady-state wander, metres
tau_pos_s = 500.0
rho_vel = 0.95          # ignored while tau_vel_s is set
sigma_vel_mps = 1.5     # steady-state wander, m/s
tau_vel_s = 100.0
r_scale = 5.0           # advertised sigmas x5, so R x25
```

The magnitude is anchored to measurement rather than to a round number. Smartphone GNSS is
3–5 m under good multipath and over 10 m in harsh multipath, with semi-urban CEP95 around
10–12 m. Under jamming the tracked satellite count falls and GDOP rises, so true error and
advertised accuracy inflate together. 35 m of steady-state wander is roughly an order of
magnitude above the few metres a phone advertises in open conditions: a receiver still
producing fixes, but bad ones.

`r_scale` inflates the *advertised 1σ*, which the measurement model then squares, so **R moves
by `r_scale²`**: the standard 5.0 is a 25× R, not a 5× one.

### Spoofing

`slow_bias` and `hijack` are implemented and not used in the current experiment matrix. They
differ in detectability by construction. A `hijack` steps the position with no matching
velocity change, so it produces a large normalized innovation squared (NIS) the moment it
starts, which an innovation gate (`--gate-confidence`, off by default) can reject. A
`slow_bias` moves position and velocity consistently, which is what is meant to keep it under
the gate; that is the whole design of a soft spoof.

## Correlation time and the fix interval

Without a time constant, `rho_pos` and `rho_vel` are applied **once per emitted fix**, with no
reference to elapsed time. The error's correlation time is then

$$\tau = -\frac{\Delta t_\text{fix}}{\ln \rho}$$

which is a property of the *scheduler*, not of the error model. At `rho_pos = 0.99`, a 5 s fix
interval is a 498 s correlation time and a 30 s interval is 2985 s. An experiment that sweeps
the fix interval is therefore also sweeping the error timescale, and the two cannot be
separated in the result.

`tau_pos_s` and `tau_vel_s` (added in commit `96a47f4`) fix this. When set, the coefficient
becomes $\rho = e^{-\Delta t/\tau}$ for the actual interval $\Delta t$ between fixes, and
`sigma_pos_m` is reinterpreted as the **steady-state** standard deviation rather than the
per-step innovation. This is the first-order Gauss-Markov process, whose autocorrelation is
$R(\Delta t) = \sigma^2 e^{-|\Delta t|/\tau}$.

They are optional and additive. Omit them and the per-fix form applies exactly as before, so
every configuration written earlier, and every result recorded from one, still means what it
meant. Prefer them for anything that varies the schedule.

Note also that without a time constant, `sigma_pos_m` is the per-step innovation, not the
error you get. An AR(1) settles at $\sigma / \sqrt{1 - \rho^2}$, so `sigma_pos_m = 3.0` with
`rho_pos = 0.99` is 21 m of wander, not 3 m.

The same commit fixed the clock the fault advances on. A fault is applied once per *emitted*
fix, and its `dt` is now the time since the previous emitted fix rather than the record
spacing. Before that, a `slow_bias` under a 5 s schedule integrated one second of drift per
five seconds of run time.

## Choosing a baseline to compare against

`just geoperf-all` scores each geophysically aided run against that filter's unaided run, so
the two must carry an **identical** GNSS profile. If they differ, the reported "improvement" is
the difference between two degradations and has nothing to do with the aid. The aiding
(`[gnss_degradation]`) and `[health_limits]` sections of
`conf/{ukf,ekf,rbpf}_{grav,mag,both}.toml` are duplicated from the matching
`conf/*_degraded.toml` for exactly this reason, and `core/tests/example_configs.rs` loads the
`conf/` recipes so that a drift between them fails a test.

All nine geophysical recipes are scored against their own filter's
`conf/{ukf,ekf,rbpf}_degraded.toml`. The RBPF's geo-aided runs are additionally scored against
`conf/ekf_degraded.toml` as the canonical INS baseline (`just geoperf-all` and
`just geoperf-rbpf`). Keeping a single profile, applied identically everywhere, is what lets
that pairing be a filename convention rather than a per-tier lookup.

## Sources

- [Jammertest 2024 results (Septentrio)](https://www.septentrio.com/en/learn-more/insights/most-resilient-gnss-receiver-results-jammertest-2024)
- [Navigating through interference at Jammertest (ESA)](https://www.esa.int/Applications/Satellite_navigation/Navigating_through_interference_at_Jammertest)
- [GNSS jamming and spoofing detection at Jammertest 2025 (u-blox)](https://www.u-blox.com/en/blogs/tech/gnss-jamming-spoofing-detection-jammertest-2025-andoya)
- [Using mobile phones for participatory detection and localization of a GNSS jammer](https://arxiv.org/pdf/2305.02038)
- [A comparative analysis of the response of GNSS receivers under vertical and horizontal L1/E1 chirp jamming, *Sensors* 21(4):1446](https://doi.org/10.3390/s21041446)
- [A comprehensive analysis of smartphone GNSS range errors in realistic environments, *Sensors* 23(3):1631](https://doi.org/10.3390/s23031631)
- [Improving smartphone GNSS positioning accuracy using contextual information, *Sensors* 26(11):3346](https://doi.org/10.3390/s26113346)
- [EASA and IATA plan to mitigate the risks of GNSS interference](https://www.iata.org/en/pressroom/2025-releases/2025-06-18-01/)
- [IATA safety risk assessment: GNSS interference](https://ic.iata.org/sites/default/files/iata_sih_document_attachment/IATA%20Safety%20Risk%20Assessment%20-%20GNSS%20Interference%20V5.pdf)
- [EUROCONTROL GNSS interference testing guide v2.0](https://www.eurocontrol.int/sites/default/files/2023-03/eurocontrol-gnss-interference-testing-guide-v2-0.pdf)
- [RTCA DO-235C, *Assessment of radio frequency interference relevant to the GNSS L1 frequency band*](https://standards.globalspec.com/std/14557649/do-235c)
- [Overbounding the effect of uncertain Gauss-Markov noise in Kalman filtering, *NAVIGATION* 68(2):259](https://navi.ion.org/content/68/2/259)
- [GNSS error simulator for farm machinery navigation](https://www.sciencedirect.com/science/article/abs/pii/S0168169915003348), the source of the first-order Gauss-Markov form the time constants implement
