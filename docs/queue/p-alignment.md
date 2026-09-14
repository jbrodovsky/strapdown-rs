# Queue 103 -- Coarse alignment and initialisation

| | |
|---|---|
| **Branch** | `v1/p-alignment` |
| **Base** | `main` |
| **Issues** | #257 |
| **Queue position** | 103 |
| **Merge point** | rebase onto PR 2 |

## Why

New module, purely additive. `earth.rs` already has the primitives.

## Acceptance criteria

- [x] `coarse_leveling`, `gyrocompassing`, `heading_from_velocity`
- [x] Unit tests under known orientations and stationary noise

## What landed

`core/src/alignment.rs`. Every function is NED and says so; nothing in the module
inspects an `is_enu` flag.

| Item | Role |
|---|---|
| `coarse_leveling` | roll and pitch from a stationary specific-force vector |
| `gyrocompassing` | heading from sensed Earth rate, with three refusal paths |
| `heading_from_velocity` | course over ground, refused below a speed threshold |
| `attitude_from_level_and_heading` | assembles the two into an NED `C_b_n` |
| `average_imu` | reduces a stationary window to the one sample the estimators want |
| `GyrocompassConfig`, `LevelAttitude`, `HeadingEstimate` | limits and results |

Gyrocompassing refuses rather than guesses on three independent grounds: latitude above
`maximum_latitude_degrees`, a predicted one-sigma heading error for the declared
`IMUQuality` above `maximum_heading_uncertainty_radians` (which rules out consumer and
industrial MEMS outright, per Groves 5.6.3), and an observed horizontal rate that is not
credible as Earth rotation. All three are `MeasurementUnavailable`, i.e. recoverable, so
the caller can fall back to `heading_from_velocity`.

Kept orthogonal to `stationary.rs`: these are pure functions over one averaged sample and
do not own a detector. `StationaryDetector` selects the window; this module interprets it.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

