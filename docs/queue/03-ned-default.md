# Queue 3 -- Default frame to NED

| | |
|---|---|
| **Branch** | `v1/03-ned-default` |
| **Base** | `v1/02-filter-api` |
| **Issues** | #255 |
| **Queue position** | 3 |

## Why

Split out of #255 deliberately. Flipping `is_enu` touches ~77 sites and shifts integration numbers; keeping it separate from the Delta-v/Delta-theta change means you know which one moved them.

## Acceptance criteria

- [ ] Default frame is NED throughout; explicit ENU conversions supported
- [ ] `measurements.rs` no longer hardcodes `is_enu: false` against an ENU-by-default library
- [ ] Integration deltas attributable to this change alone

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

