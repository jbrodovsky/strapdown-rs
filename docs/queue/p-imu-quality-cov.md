# Queue 101 -- auto_covariance from IMUQuality

| | |
|---|---|
| **Branch** | `v1/p-imu-quality-cov` |
| **Base** | `main` |
| **Issues** | #257 |
| **Queue position** | 101 |
| **Merge point** | before PR 1 |

## Why

Pulled ahead of PR 1 on purpose: fixing #266 means retuning the ESKF covariance constants, and this is the principled way to do it instead of hand-picking new golden numbers.

## Acceptance criteria

- [x] `auto_covariance` derives P0 from `IMUQuality` + position/velocity uncertainty
- [x] Builds on the existing `IMUQuality` accessors rather than new tables

## What landed

`IMUQuality::auto_covariance(uncertainty, latitude_degrees, altitude_m)` in `core/src/lib.rs`,
alongside the `*_process_noise` helpers it is the initial-condition counterpart to, plus
`InitialUncertainty` for the two terms an IMU grade cannot supply (how well the initial
position and velocity are known).

Each block of the 15-element diagonal is traceable to an existing accessor:

| States | Derived from |
|---|---|
| position | `InitialUncertainty`, converted to rad² through the WGS84 principal radii |
| velocity | `InitialUncertainty`, widened by `accel_velocity_random_walk` over one initialisation interval |
| attitude | `gyro_angle_random_walk` over the same interval, plus the `b/g` levelling error of an unknown accelerometer bias (Groves §5.6.3) |
| accel bias | `accel_bias_instability_mps2` squared |
| gyro bias | `gyro_bias_instability_dph`, converted from rad/h to rad/s, squared |

## Two design decisions

**It lives on `IMUQuality`, not on the ESKF.** The accessors it composes are all there, it is
filter-agnostic (the 15-state EKF, the ESKF and the particle filter all want the same numbers),
and putting it on a filter constructor would mean either duplicating the grade table or taking
an `IMUQuality` parameter anyway.

**It is opt-in; `DEFAULT_INITIAL_COVARIANCE` is unchanged.** Two reasons. Mechanically, the
default is a `const` array with no position and no fix quality available to it, so making it
the default would require inventing default values for both -- trading one set of hidden
assumptions for another. Empirically, the shipped default and `sim::DEFAULT_PROCESS_NOISE` are
*the same fifteen numbers*: P0 was seeded from the process noise, which is why its position
terms claim 1e-3 rad (about 6 km) of latitude error and 1 cm of altitude error at once. That
is worth replacing, but replacing it is the #266 retune's job, with its own blast radius.

Measured on `core/tests/test_data.csv` with process noise and initial state held fixed, phone
recording against `IMUQuality::Consumer` and the receiver's own reported 3.81 m accuracy:

| Metric | Default P0 | `auto_covariance` P0 |
|---|---|---|
| Horizontal rms | 23.53 m | 23.53 m |
| Horizontal peak | 37.88 m | 37.88 m |
| Altitude rms | 3.02 m | 3.05 m |
| Altitude peak (settling transient) | 42.56 m | 21.43 m |
| Altitude peak after settling | 12.30 m | 14.58 m |

Pinned by `test_eskf_auto_covariance_initialization_on_real_data`, held to the same bounds as
the default-initialisation test next to it.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

