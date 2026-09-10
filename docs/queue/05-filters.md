# Queue 5 -- EKF, UKF, PF/RBPF on the new API

| | |
|---|---|
| **Branch** | `v1/05-filters` |
| **Base** | `main` (retargeted after #273 merged) |
| **Issues** | #259 |
| **Queue position** | 5 |

## Why

Scope decision needed first: `core/src/particle.rs` is 466 lines of traits with zero concrete impls, while `sim.rs` already advertises a `ParticleFilterType::Standard` that does not exist.

## Acceptance criteria

- [x] EKF/UKF Jacobians and sigma-point propagation updated for Delta-v/Delta-theta
- [x] RBPF prediction and state representation updated
- [x] All filters implement the updated trait with uniform `Result` returns
- [x] Side-by-side comparison test across filters

## Outcome

### The scope decision

`ParticleFilterType` advertised three variants and had one implementation. `Standard` was the
`clap` **default**, so `strapdown-sim particle-filter` failed on its own defaults with "Only
Rao-Blackwellized particle filter is implemented in this mode" -- a runtime error on the path
a user reaches by typing the subcommand and nothing else.

`Standard` and `Velocity` were removed rather than implemented. `particle.rs`'s own module
documentation calls it a "template-style library" of building blocks -- the resampling
strategies and the `Particle` trait -- for users assembling their own filter, which is a
defensible thing for it to be and not a filter. Implementing a bootstrap PF to satisfy an
enum variant would have widened queue 5 well past #259's stated criteria, and #259 asks only
for the RBPF's prediction and state representation.

The enum is kept at one variant rather than collapsed away, and `run_particle_filter` now
`match`es on it, so adding a second concrete filter is an extension of that match rather than
a rediscovery that the flag was never read.

### Delta-v / Delta-theta

The EKF and UKF moved from `imu_from_input` (rates only) to `imu_sample_from_input`, which
the ESKF has used since queue 4 and which accepts either form. `imu_from_input` had no
remaining callers and was deleted.

Both filters now correct for bias in the increment domain -- `delta_v - b_a*dt`,
`delta_theta - b_g*dt` -- and hand the corrected `ImuSample` straight to `mechanize`. The old
order was to correct rates and *then* integrate via `ImuSample::from_rates`, which is
arithmetically the same for a sample that arrived as rates but forces a genuine delta-v /
delta-theta sample through a lossy division by `dt` first. The UKF applies the correction per
sigma point, since each sigma point carries its own bias hypothesis in states 9..15.

Where a Jacobian is still derived in the rate domain -- `state_transition_jacobian` for the
EKF, and the RBPF's use of the same function -- the corrected sample is converted back with
`to_rates()` for that call only. `rate_and_increment_inputs_are_equivalent` holds the two
input paths to 1e-12 relative agreement across the ESKF, EKF and UKF with non-zero seeded
biases, which is the property the reordering could have broken.

### The RBPF on the trait

`RaoBlackwellizedParticleFilter` now implements `NavigationFilter`. Its inherent
`predict(&IMUData, f64)` and generic `update<M>` became private bodies (`predict_sample`,
`update_with`) and the trait impl is the input-resolution shim; `update_with` stays generic
because the downcasts inside it are what select the specialised position/velocity paths.
Callers are unchanged apart from needing the trait in scope -- `sim/src/main.rs` already
imported it, under a `#[cfg(feature = "geonav")]` that is now unconditional.

`imu_sample_from_input` was widened from private to `pub(crate)` so `rbpf.rs` can share it.

### The comparison test

`core/tests/filter_comparison.rs` runs all four filters over one scenario through
`&mut dyn NavigationFilter`, and asserts finiteness, agreement with truth, and pairwise
agreement -- three assertions because a diverged filter goes non-finite before it goes
merely wrong, and because a filter can sit inside its own truth bound while drifting to the
opposite edge of it from the others.

| filter | horizontal | altitude | velocity |
|---|---|---|---|
| ESKF | 0.000 m | 0.000 m | 0.0000 m/s |
| EKF | 0.000 m | 0.000 m | 0.0000 m/s |
| UKF | 0.019 m | 0.000 m | 0.0305 m/s |
| RBPF | 0.222 m | 0.613 m | 0.1172 m/s |

Bounds are set at ~4.5x the worst measured value, the margin #288 rederived the ESKF
integration bounds to. The whole file runs in 0.3 s.

Truth is defined as the mechanization's own integral of the inertial stream rather than
solved for independently, so the generator and the filters cannot disagree about what the
data means. `lib.rs` already records that a hand-rolled copy of the propagation equations in
a scenario generator went silently wrong when the mechanization moved to increments.

Two traps found while building it, both recorded in the file:

- `GPSPositionMeasurement::get_noise` converts metres to radians itself, and
  `build_event_stream` feeds it metres. `generate_scenario_data` pre-multiplies by
  `METERS_TO_DEGREES`, which squares the conversion and asks the filters to trust a 45 um
  GPS. The RBPF's scenario tests run against that.
- Taking each fix from the state *before* its inertial sample rather than after leaves every
  measurement one sample stale. At 10 m/s that is a 2 m along-track bias -- 0.4 sigma against
  a 5 m fix, but *persistent*, and it diverged the ESKF to 10^33 m within 600 samples.

### Found, not fixed: #303

`all_filters_converge_from_a_displaced_seed` is `#[ignore]`d. The EKF and ESKF do not
converge from **any** non-zero seed error on this scenario -- 1 m of position, 1 m of
altitude, 0.05 m/s of velocity or 0.001 rad of attitude all run the vertical channel away
within 2-5 minutes -- while the UKF converges from all of them and all four are exact when
seeded on truth. The UKF is the only one of the three that does not use `linearize.rs`'s
analytic Jacobians, which is what localises it.

This predates queue 5: the ESKF's divergence samples are bit-identical before and after this
branch, and the ESKF was already consuming `ImuSample` from queue 4. Quarantined rather than
tuned around, per the precedent set by #295. Full characterisation and a suggested starting
point are in #303.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).
