# Queue 2 -- StrapdownError + ImuSample (breaking API)

| | |
|---|---|
| **Branch** | `v1/02-filter-api` |
| **Base** | `v1/01-eskf-fix` |
| **Issues** | #254, #255 |
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

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

