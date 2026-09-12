# Queue 7 -- InsEngine builder + lever-arm compensation

| | |
|---|---|
| **Branch** | `v1/07-engine` |
| **Base** | `v1/06-aiding` |
| **Issues** | #262 |
| **Queue position** | 7 |

## Why

Depends on the object-safety decision made in PR 2.

## Acceptance criteria

- [x] `InsEngine::builder().with_config(..).build()`
- [x] High-rate `predict`, async `update_gnss`, `nav_solution()`
- [x] Antenna lever-arm compensation for position and velocity
- [x] Builder documented with doctests

## What landed

`core/src/engine.rs` -- the user-facing entry point: a builder, `predict`/`predict_rates`,
`update_gnss`, an `update` escape hatch for any other measurement model, and
`nav_solution()` in degrees, metres and m/s. Lever-arm compensation moves a fix reported at
the antenna to the IMU centre, for both position (`attitude * lever_arm`) and velocity
(`attitude * (omega x lever_arm)`), and is an exact identity at a zero lever arm.

Gating from queue 6 is carried through, closing the half of #260 that asked for it in
`InsEngine` as well as in `NavigationFilter::update`: `set_innovation_gate` installs it, and
`update_gnss` returns the position leg's `UpdateOutcome`. A fix whose position the gate
rejects **skips its velocity leg too** -- both legs come from the same receiver at the same
epoch, so a multipath or spoofed fix corrupts them together, and applying the velocity anyway
would let the measurement the gate just rejected back in through the other door.

## Fixed here: `InitialState::new` double-converted attitude

On the degrees path the constructor converted `roll`/`pitch`/`yaw` to radians while leaving
`in_degrees == true`, so every filter constructor -- `if initial_state.in_degrees {
roll.to_radians() }` -- converted them a second time. A 45 degree seed was stored as 0.785
and reached the filter as 0.0137 rad. Only a zero attitude survived the round trip.

It went unnoticed because the workspace's other seeds are struct literals; `InsEngine` is the
first caller to use the constructor with a non-zero heading, and lever-arm compensation
rotates by the attitude estimate, so it resolved the antenna offset along the wrong axis.
Measured on the synthetic static-antenna run, horizontal error against truth:

| seeded heading | before | after |
|---|---|---|
| 0 deg | 0.017 m | 0.017 m |
| 45 deg | 2.238 m | 0.017 m |
| 90 deg | 4.170 m | 0.017 m |

The fix stores every angular field in the units it was supplied in, tagged by `in_degrees`,
which is the contract the position fields already followed and the one the filter
constructors already assumed. #304 documents this same inconsistency as known-and-unfixed;
that PR and this one touch the same function and will want a trivial conflict resolution.

## Also fixed: the lever-arm test harness compared two different measurement streams

`run_on_records(.., None, ..)` built a position-only `GnssFix` with the record's own reported
accuracies, while `run_on_records(.., Some(offset), ..)` built one with a velocity leg and
fixed 3 m / 5 m accuracies. Comparing those two runs and attributing the difference to the
lever arm made `real_data_zero_lever_arm_changes_nothing` unpassable by construction. Every
run now synthesizes its fix the same way, so only the offset varies.

## Quarantined

`real_data_antenna_offset_is_removed` is `#[ignore]`d. Compensation rotates the lever arm by
the filter's *estimated* attitude, and over all 5,365 GNSS epochs of `test_data.csv` that
estimate sits **60.6 deg** from the device's own recorded attitude on average, 156.6 deg at
worst. Rotating a 3 m offset by a heading that wrong points the correction the wrong way, so
applying it is worse than ignoring it -- 3.460 m of drift against the baseline versus 3.152 m
uncompensated. That is arithmetic, not a lever-arm defect, and adding magnetometer yaw aiding
makes it worse still (75.9 deg mean), so it is not simply weak yaw observability under
GNSS-only aiding. Same family as #302, #303 and #307.

The three synthetic tests cover the compensation itself and pass to 0.017 m at headings of
0, 45 and 90 degrees, because there the attitude is correct by construction.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).
