# Fault Simulation

This chapter is the experiment-design rationale behind `conf/`: which GNSS conditions are
worth simulating, what the published measurements say those conditions look like, and which
knobs in `GnssFaultModel` correspond to which.

## What jamming actually does

The dominant, repeatedly measured effect of jamming is **loss of fix**, not graceful
degradation. At Jammertest — the open field trial run annually at Andøya by the Norwegian
Public Roads Administration, Communications Authority, Defence Research Establishment,
Metrology Service and Space Agency — the jamming signal was in most runs strong enough to
cause complete GNSS signal loss, and a smartphone roughly ten metres from a jammer lost lock
on benign signals entirely. Receivers reacquired when terrain or buildings shadowed them from
the jammer and lost lock again on returning to line of sight.

That matters for how a scenario is built. A receiver inside the jammed area does not report a
noisier position; it reports nothing. A model that keeps delivering fixes with more error on
them describes the *fringe* of the jammed area, not the inside of it — and it is the inside
that an inertial system, or a geophysical aid, exists to carry you through.

The scale is not marginal. IATA recorded a 220% rise in GPS signal-loss events between 2021
and 2024; European flights encountering GNSS interference rose from roughly 200 per day in Q1
2024 to around 900 per day in Q2, and EUROCONTROL logged more than 2,500 jamming events in
2024 alone.

## The three tiers

| Tier | What it models | Scheduler | Fault | Configs |
|---|---|---|---|---|
| **Denial** | inside the jammed area | `duty_cycle` | `none` | `conf/*_denied.toml` |
| **Degraded** | the fringe of it | `fixed_interval` | `degraded` | `conf/*_jammed.toml` |
| **Spoof** | a deceptive signal | either | `slow_bias` / `hijack` | — |

### Denial

`conf/*_denied.toml`: 30 s of fixes in every 150, fault `none`.

The only variable is availability, which is the point — the fixes that do arrive are the
receiver's own, so any change in the result is attributable to the outage rather than to a
noise model. 30 s on is long enough for the filter to re-converge between outages, so each
one starts from a comparable state; 120 s off is where a MEMS platform's drift becomes the
dominant error term.

Sweep `off_s` over 60 / 120 / 300 / 600 to trace how an aid's contribution grows with outage
length. This is the tier where an aid can be shown to earn its keep, because during an outage
it is the only correction there is.

### Degraded

`conf/*_jammed.toml`: fixes every 5 s, with AR(1)-correlated error on position and velocity
and the advertised accuracies inflated.

Anchoring the magnitude to measurement rather than to a round number: smartphone GNSS is 3–5 m
under good multipath and over 10 m in harsh multipath, with semi-urban CEP95 around 10–12 m.
Under jamming the tracked satellite count falls and GDOP rises, so true error and advertised
accuracy inflate together. `sigma_pos_m = 35.0` is about 14× this dataset's 2.58 m median
advertised accuracy — a receiver still producing fixes, but bad ones.

`r_scale` inflates the *advertised 1σ*, which the measurement model then squares, so **R moves
by `r_scale²`**: the standard 5.0 is a 25× R, not a 5× one.

### Spoofing

`slow_bias` and `hijack` are implemented and unused in the current experiment matrix. They
differ in detectability by construction: a `hijack` steps the position with no matching
velocity change, so it produces a large NIS the moment it starts and a chi-squared gate
rejects it. A `slow_bias` moves position and velocity consistently, which is what keeps it
under the gate — that is the whole design of a soft spoof.

## Correlation time and the fix interval

`rho_pos` and `rho_vel` are applied **once per emitted fix**, with no reference to elapsed
time. The error's correlation time is therefore `-interval_s / ln(rho)` — a property of the
*scheduler*, not of the error model. At `rho_pos = 0.99`, a 5 s fix interval is a 498 s
correlation time and a 30 s interval is 2985 s.

So an experiment that sweeps the fix interval is also sweeping the error timescale, and the
two cannot be separated in the result.

`tau_pos_s` and `tau_vel_s` fix this. When set, the coefficient becomes `exp(-dt / tau)` for
the actual interval between fixes, and `sigma_pos_m` is reinterpreted as the **steady-state**
standard deviation in metres rather than the per-step innovation:

```toml
[gnss_degradation.fault]
kind = "degraded"
sigma_pos_m = 35.0      # steady-state wander, metres, whatever the schedule
tau_pos_s = 500.0       # correlation time in seconds
```

They are optional and additive. Omit them and the per-fix form applies exactly as before, so
every configuration written earlier — and every result recorded from one — still means what it
meant. Prefer them for anything that varies the schedule.

Note also that without a time constant, `sigma_pos_m` is the per-step innovation, not the
error you get: an AR(1) settles at `sigma / sqrt(1 - rho²)`, so `sigma_pos_m = 3.0` with
`rho_pos = 0.99` is 21 m of wander, not 3 m.

## Choosing a baseline to compare against

`geoperf-all` scores each geophysically-aided run against that filter's unaided run, so the
two must carry an **identical** GNSS profile. If they differ, the "improvement" reported is
the difference between two degradations and has nothing to do with the aid. The
`[gnss_degradation]` and `[health_limits]` sections of `conf/{ukf,ekf}_{grav,mag,both}.toml`
are duplicated from `conf/*_degraded.toml` for exactly this reason, and
`core/tests/example_configs.rs` covers `conf/` so a drift in them fails a test.

`conf/*_degraded.toml` is kept as it was — 21 m steady state — because it is the baseline
every result so far was measured against. `conf/*_jammed.toml` is the recalibrated profile to
prefer for new work. Switch deliberately rather than by having the old one change underfoot.

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
- [GNSS error simulator for farm machinery navigation](https://www.sciencedirect.com/science/article/abs/pii/S0168169915003348) — the first-order Gauss-Markov form `R(Δt) = σ² e^(-|Δt|/τ)` these time constants implement
