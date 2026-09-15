# Queue 3 -- Default frame to NED

| | |
|---|---|
| **Branch** | `v1/03-ned-default` |
| **Base** | `v1/02-filter-api` (merged; this PR now sits on `main`) |
| **Issues** | #255 |
| **Queue position** | 3 |

## Why

Split out of #255 deliberately. Flipping `is_enu` touches ~77 sites and shifts integration numbers; keeping it separate from the Delta-v/Delta-theta change means you know which one moved them.

## Acceptance criteria

- [x] Default frame is NED throughout; explicit ENU conversions supported
- [x] `measurements.rs` no longer hardcodes `is_enu: false` against an ENU-by-default library
- [x] Integration deltas attributable to this change alone

## What landed

**The default flip.** `StrapdownState::default()`, `StrapdownState::new(.., None)` and
`InitialState::new(.., None)` all produce NED states. `InitialState` already derived
`Default` as NED while its constructor said ENU; those now agree.

**A vertical-channel bug the flip would otherwise have shipped.** `position_update`
integrated `velocity_vertical` into `altitude` without consulting the frame. That is right
in ENU and inverted in NED: a body in free fall *gained* 4.9 m in the first second. Harmless
while everything defaulted to ENU, load-bearing the moment NED became the default. `altitude`
is height above the ellipsoid -- positive up -- in both frames; only `velocity_vertical` and
the gravity sign change. Guarded by `free_fall_loses_altitude_in_both_frames` and
`synthetic_ned_descent_loses_altitude`, both of which fail against the pre-fix code.

**Explicit conversions.** `StrapdownState::to_ned` / `to_enu`. The nav frame here is ordered
(north, east, vertical), so ENU and NED differ by a vertical reflection -- improper on its
own. The conversion flips the body frame's vertical axis alongside the nav frame's, which
composes to a proper rotation and matches the physical case (a z-up sensor resolved in a
z-up frame becoming a z-down sensor in a z-down frame).

**`measurements.rs`.** `jacobian_state` now inherits the crate default via
`..StrapdownState::default()` instead of restating a bare `is_enu: false`. The filter state
vector carries no frame tag, so the default is the only convention available there -- it just
happened to agree with nothing while the crate defaulted to ENU.

## Integration deltas

**Zero.** Every metric in `core/tests/integration_tests.rs` is bit-identical across the
change:

| Filter | RMS horizontal | RMS altitude | Delta |
|---|---|---|---|
| Dead reckoning | 12894566.23 m | -- | none |
| UKF closed-loop | 23.54 m | 2.78 m | none |
| EKF closed-loop | 27.64 m | 37.27 m | none |
| ESKF closed-loop | 23.49 m | 2.40 m | none |
| RBPF closed-loop | 23.78 m | 4.54 m | none |

That is the intended result, not a missing test. `test_data.csv` is a Sensor Logger export
whose accelerometer reads +g along the device's up-axis at rest -- ENU-convention data. The
suite now pins `is_enu: true` explicitly rather than leaning on the default, so the frame
flip cannot move it and any future movement is attributable to something else.

## Known gaps, deliberately left

- ~~**`sim::dead_reckoning` and `sim::initialize_{ukf,ekf,eskf}` still hardcode ENU.**~~
  Closed by #296. The frame is now a caller-supplied option on all four entry points --
  `dead_reckoning` takes it as an argument, the three initialisers take it on
  `UkfConfig`/`EkfConfig`/`EskfConfig` -- defaulting to NED, with `--enu` on the CLI and
  `is_enu` in the config file. `TestDataRecord` still carries no frame tag, so the half of the
  objection the flag does not answer is handled by `sim::check_declared_frame`: it rotates the
  leading records' specific force into the navigation frame and rejects a declaration the data
  contradicts (+9.7 m/s^2 for Sensor Logger, -9.8 for `syn`) with an
  `InvalidConfiguration` naming the flag to pass. It rejects; it does not guess. The symptom
  is gone: `syn -> dr` on 60 s of stationary navigation-grade truth held altitude to 0.03 m,
  against the 35 km it reached before.
- **`earth::transport_rate` disagrees with Groves 5.44 in both frames** -- sign-flipped on all
  three components, `R_N`/`R_E` swapped, and the third component built from `v_N` where Groves
  uses `v_E`. Verified against a numerical differentiation of `C_n^e` along the trajectory.
  Wrong independently of the frame default, so fixing it here would have moved every
  integration number for an unrelated reason. Tracked in #297.
- ~~**ENU support is only skin-deep.**~~ Closed by #321: `velocity_update` now converts to NED
  once, propagates there and reflects the result back, so an ENU run is exact rather than
  approximate. (The original gap: only the gravity term consulted `is_enu`, so the two views of
  one state drifted apart by the Coriolis asymmetry, ~2e-5 m of altitude over a 0.1 s step at
  20 m/s.) Part of why NED is the default now rather than the alternative.

Also filed while working this queue position, unrelated to the frame: #298, where
`strapdown-sim` creates its `-o` path as a directory and overwrites the input CSV with the
results.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).
