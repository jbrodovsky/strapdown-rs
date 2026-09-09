# Queue 2 -- StrapdownError + ImuSample (breaking API)

| | |
|---|---|
| **Branch** | `v1/02-filter-api` |
| **Base** | `v1/01-eskf-fix` |
| **Issues** | #254, #255, #292 |
| **Queue position** | 2 |

## Why

#254 and #255 both break `NavigationFilter`; doing them separately means touching three filter impls, ~14 doc examples and 139 unit tests twice. Two commits inside one PR.

## Acceptance criteria

- [ ] `StrapdownError` (thiserror) in `core/src/error.rs`, exported
- [ ] `predict`/`update` return `Result`; `predict` takes `&dyn InputModel` so the trait is object-safe
- [ ] Zero `panic!`/`.unwrap()` in library code
- [ ] `ImuSample { delta_v, delta_theta, dt }` + `mechanize()`; `ImuSample::from_rates` helper
- [ ] `forward()` kept as a deprecated wrapper
- [ ] Decide explicitly whether `MeasurementModel::get_jacobian` also returns `Result`
- [x] #292: `principal_radii` given degrees at every call site

## #292 -- radii units

Folded in here because it is a mechanization units bug and this PR owns the
mechanization API. The issue named one call site; there were three.

| site | defect |
|---|---|
| `lib.rs` `position_update` (x2) | `state.latitude` is radians, `principal_radii` takes degrees. ~0.5% error in the lat/lon integration rate, shared by every filter -- which is why no test caught it. |
| `linearize.rs` `state_transition_jacobian` | Same bug, on the line directly above three calls that *do* convert with `to_degrees()`. |
| `earth.rs` `eotvos` | Internally inconsistent: passed degrees to `principal_radii` but took `latitude.cos()` as if radians, and documented radians while its only production caller (`gravity_anomaly`) documents and passes degrees. At 45 deg that is `cos(45 rad) = 0.525` instead of `0.707` -- a ~26% error in the Eotvos correction, feeding geonav gravity aiding. Doc corrected to degrees, cosine now converts. |

Two regression tests in `lib.rs`: one pins `position_update`'s integration rate
against radii taken at the right latitude (verified to fail when the bug is
reintroduced), one asserts the invariant every caller shares -- meridian radius
increases toward the poles, which a radians argument violates by landing near
the equator.

**Fallout:** `rbpf_runs_on_scenario_stationary` is `#[ignore]`d under #295. The
correct radii move its stationary altitude error from 12.49 m to 28.71 m against
a 15 m bound with 1.2x margin over the buggy baseline, so the RBPF vertical
channel is compensating for the old units. Quarantined rather than tuned around,
per the #267 precedent.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

