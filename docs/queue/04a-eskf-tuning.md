# Queue 4a -- One ESKF tuning, with the reasoning attached

| | |
|---|---|
| **Branch** | `claude/eskf-initial-covariance-retune` |
| **Base** | `v1/04-eskf-primary` |
| **Issues** | follow-up to #258; units follow #266, clamp follows #286 |
| **Queue position** | 4a (follow-up, not a milestone issue) |

## Why

Queue 4 left two ESKF initial covariances in the tree and no note saying which was intended:
`initialize_eskf`'s, at 1e-6 accelerometer-bias and 1e-8 gyroscope-bias variance, and the
integration suite's `ESKF_INITIAL_COVARIANCE`, at 0.08 and 0.008 -- five orders of magnitude
looser. The suite's tuning ran the gyro-bias estimate into the `inject_error_state`
anti-windup clamp on 76 of the 5,366 samples of `test_data.csv`, including the last one, so
what the bias assertions demonstrated there was the clamp rather than the estimator.

Two entries also disagreed with their own comments: `initialize_eskf`'s position covariance
was labelled `(m²)` when #266 established that the horizontal position error state is
radians, and `ESKF_INITIAL_COVARIANCE`'s altitude entry of `8.0` was labelled "8m" when
8.0 m² is a 2.83 m standard deviation.

## What the measurements say

Everything below is `core/tests/test_data.csv`, 5,366 samples at 1 Hz, full GNSS aiding.

**The clamp was the process noise, not the initial covariance.** Holding the initial
covariance at the suite's values and scaling only the six bias entries of Q reproduces the
saturation -- 37 clamped samples at 8x, none at 4x -- while scaling only the nine
navigation-state entries produces none. The initial bias covariance moves *when* saturation
first appears (sample 1 at 0.008, sample 1,615 at 1e-8) but not *whether* it appears. So the
loose prior alone was never the whole story, and swapping it without also dropping the 8x
process noise would have left the tests measuring the clamp.

**A sensor-class bias prior does not survive contact with this estimator.** A consumer-MEMS
gyroscope turn-on bias is roughly 0.5-1 °/s (0.009-0.017 rad/s, ~1e-4 rad²/s²) and an
accelerometer's ~0.1 m/s² (~1e-2 (m/s²)²). Those are the numbers an honest prior would carry.
Given them, the filter parks the estimates at the anti-windup caps: the clamp engages by
sample 2 at a sensor-class gyro prior, and the accelerometer bias runs to the 2.0 m/s² cap
and stays there at a sensor-class accel prior. The bias states here are observed only through
`bias -> tilt -> velocity` and absorb whatever else is unmodelled -- initial attitude error,
lever arm, time sync -- so freedom is spent on those rather than on the bias.

| gyro-bias prior (rad²/s²) | peak abs gyro bias | clamped samples |
|---|---|---|
| 1e-8 (shipped) | 0.0284 | 0 |
| 1e-7 | 0.0284 | 0 |
| 3e-7 | 0.0330 | 0 |
| 1e-6 | 0.0500 | 0 |
| 1e-5 | at cap | 1 |
| 1.7e-4 (sensor class) | at cap | 14 |

| accel-bias prior ((m/s²)²) | peak abs accel bias | clamped samples |
|---|---|---|
| 1e-6 (shipped) | 0.623 | 0 |
| 1e-5 | 0.623 | 0 |
| 1e-4 | 0.730 | 0 |
| 1e-3 | at cap | 4 |
| 1e-2 (sensor class) | at cap | 6 |

The shipped priors sit roughly two decades below both cliffs. That is the choice this branch
makes, and it is a regularisation choice, not a datasheet reading: the prior is deliberately
tighter than the sensor class because the estimator's bias observability on this data does not
support a sensor-class prior, and an estimate held by a clamp is not an estimate. The 0.008
the suite carried is wrong in the other direction and by more -- an 0.089 rad/s (5.1 °/s)
standard deviation, five times the *worst* consumer-MEMS turn-on bias.

Note that peak accelerometer bias is 0.62 m/s² even at the tightest prior, against a
plausible turn-on bias near 0.1. The accelerometer-bias state is absorbing something that is
not accelerometer bias. That is a real open question about the model, not about this tuning.

## What changed

`sim::ESKF_INITIAL_ERROR_COVARIANCE` is now the single initial error covariance, public and
documented entry by entry. `initialize_eskf` starts from it and splices in the caller's
attitude and bias overrides; the integration suite aliases it and the library's
`DEFAULT_PROCESS_NOISE` rather than keeping its own constants.

Corrected entries, with what they replace:

| entry | was | now | why |
|---|---|---|---|
| horizontal position | `1e-6`, labelled m² | `2.5e-12 rad²` = (10 m)² | consumed as rad² (#266), so `1e-6` was a 6.4 km initial uncertainty |
| altitude | `1e-4 m²` | `225 m²` = (15 m)² | 1e-4 m² is a 1 cm standard deviation on a filter initialised from a GNSS fix |
| velocity | `1e-3` | `1.0 m²/s²` = (1 m/s)² | 1e-3 is 0.03 m/s, tighter than any GNSS speed accuracy (this recording reports 0.39 m/s) |
| accel bias | 1e-6 / 0.08 split | `1e-6` | see above |
| gyro bias | 1e-8 / 0.008 split | `1e-8` | see above |

Attitude stays at 1e-5 rad² (0.18°), which is optimistic for an attitude initialised from a
handset's own orientation estimate. It is the one entry still carrying an unexamined number
and is called out as such in the constant's documentation.

## Results

No integration-test bound moved. Measured before -> after:

| run | horizontal rms | horizontal max | altitude rms | altitude max | peak abs gyro bias |
|---|---|---|---|---|---|
| `test_eskf_closed_loop_on_real_data` | 23.49 -> 23.53 m | 37.89 -> 37.88 m | 2.40 -> 2.77 m | 9.23 -> 12.30 m | 0.05000 (at cap) -> 0.02835 |
| `test_eskf_default_initialization_on_real_data` | 23.53 m | 37.88 m | 3.02 -> 2.88 m | 42.59 -> 19.16 m | 0.02835 |
| `test_eskf_with_degraded_gnss` (2 s fixes) | 23.56 -> 23.67 m | 39.94 -> 40.00 m | 3.62 -> 3.90 m | 12.90 -> 14.45 m | 0.02829 |

The closed-loop run gives up 0.37 m of vertical rms and 3.1 m of vertical peak. That is what
the 8x bias-state random walk was buying, and it was buying it by letting the bias estimates
run into the clamp; the filter now sits alongside the UKF's 2.8 m / 12.1 m instead of below
it. The vertical bound keeps ~3.5x margin against the 385 m peak the pre-#286 divergence
produced, so it still fails loudly on a real regression.

The default-initialisation run improves in the vertical channel for the opposite reason: its
42.6 m settling transient on sample 3 was the 1 cm initial altitude prior refusing the
correction the filter needed, and it is now 19.2 m. Steady state is unchanged at 12.30 m.

## Left open

- Process noise is still a set of undocumented numbers with no stated derivation.
  `DEFAULT_PROCESS_NOISE` deserves the same treatment this branch gave the initial
  covariance, derived from IMU noise densities rather than chosen. There is a measured
  opportunity waiting there: 8x on the nine navigation-state entries with the bias entries
  left at 1x gives 2.42 m altitude rms / 9.62 m peak *and* no clamping (peak gyro bias
  0.0225), better than either tuning this branch compared. Doing it properly belongs with
  #257's `auto_covariance`, not here, because picking `8x` again would just be a third
  undocumented number.
- The accelerometer-bias state absorbing 0.62 m/s² is unexplained.
- `test_eskf_stability_high_dynamics` does not apply a high-dynamics tuning. Two unused
  `_`-prefixed vectors implied it did; they are removed. Whether the test should re-tune, or
  be renamed for what it checks (finiteness and quaternion normalisation), is open.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).
