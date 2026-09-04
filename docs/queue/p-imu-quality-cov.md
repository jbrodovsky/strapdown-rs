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

- [ ] `auto_covariance` derives P0 from `IMUQuality` + position/velocity uncertainty
- [ ] Builds on the existing `IMUQuality` accessors rather than new tables

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

