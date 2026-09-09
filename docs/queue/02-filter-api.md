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

- [x] `StrapdownError` (thiserror) in `core/src/error.rs`, exported
- [x] `predict`/`update` return `Result`; `predict` takes `&dyn InputModel` so the trait is object-safe
- [x] Zero `panic!`/`.unwrap()` in library code -- 49 -> 0, verified by lint, not inspection
- [x] `ImuSample { delta_v, delta_theta, dt }` + `mechanize()`; `ImuSample::from_rates` helper
- [x] `forward()` kept as a deprecated wrapper
- [x] Decide explicitly whether `MeasurementModel::get_jacobian` also returns `Result` -- **yes**

## Decisions

**`get_jacobian` returns `Result`.** The five core models are analytic and infallible, but the
three geophysical ones read a loaded map and genuinely fail when the estimate leaves its
bounds. Returning `Result` alone would have been cosmetic, though: the panic was two layers
down in `GeoMap::get_point`, so that had to become fallible in the same change.

**RBPF keeps its inherent methods** rather than adopting `NavigationFilter`; that is queue 5
(#259). It did have to become fallible, because `matrix_square_root` and `get_measurement`
now are.

**`update` failures are skipped and counted, not fatal** (`run_closed_loop`). An off-map
geophysical sample at a tile edge is the condition that aiding exists to handle. A circuit
breaker at 100 consecutive rejections keeps that from silently degrading to dead reckoning,
which would still pass an accuracy assertion by coincidence. `is_recoverable()` is one
function rather than a match per call site, so #260's gating variant extends the policy in
one place.

**Four asserts are kept**, documented, in `NavigationResult`'s conversions: they are fed only
by `filter.get_estimate()`, so a wrong shape is a crate invariant violation, not user input.
Converting them would push `?` through result assembly for a failure that cannot occur.

## Numerical result of the `ImuSample` refactor

The risk was that increments change the arithmetic in a crate that has diverged from a 1e-14
perturbation before (#266, #286). Measured rather than assumed:

| path | change |
|---|---|
| attitude | **bit-identical** -- `skew(w) * dt` and `skew(w * dt)` differ only by an exactly-representable negation |
| velocity | differs by association alone: `v + (f + g - c) * dt` became `v + dv_nav + (g - c) * dt` |

`mechanize_agrees_with_rate_form_to_rounding` pins both: `assert_eq!` on the attitude
components, and a 16 ULP bound on velocity, so a future change that alters the *equation*
rather than the association fails loudly. The integration suite is unchanged, so the
difference does not compound across 5,366 samples of closed-loop feedback.

That test also earned its keep immediately. Four tests failed on the first full run, one with
a 139 km horizontal error -- far too large to be association error, which pointed at a caller
rather than the arithmetic. It was a `#[cfg(test)]` scenario generator that had duplicated
the mechanization equations instead of calling `forward`, and went stale when they changed.
It now calls `mechanize`.

**Not applied: coning/sculling compensation.** `from_rates` is a first-order rectangular
integration, exact only for rates constant across the interval. Genuine compensation is what
makes an increment interface worth having at high rotation rates; it is follow-up work.
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

