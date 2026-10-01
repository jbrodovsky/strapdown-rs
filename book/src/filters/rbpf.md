# Rao-Blackwellized Particle Filter

The RBPF is the marginalized particle filter of Schön, Gustafsson & Nordlund (2005), built as
Canciani & Raquet describe it for airborne magnetic-anomaly navigation ("Airborne Magnetic Anomaly
Navigation", IEEE TAES 53(1):67-80, 2017). It exists for map-aided navigation. There, a gravity
or magnetic map makes the measurement a highly nonlinear, often multimodal function of position,
while everything else about the navigation error stays linear.

## Structure

An INS nominal trajectory is mechanized with **barometer aiding in the mechanization**. This is
a third-order loop that feeds the difference between the barometer and the INS altitude back
into altitude, vertical velocity and a vertical-acceleration correction. The filter estimates
the errors in that aided solution:

| partition | states |
| --- | --- |
| **sampled** (particles) | latitude and longitude error |
| **linear** (one Kalman filter, shared covariance) | altitude error; velocity error (3); nav-frame tilt (3); barometer-aiding error `δh_a`; the loop's vertical-acceleration error `δâ`; and per map channel, a temporal variation `V` and a constant offset `c` |

For one map channel that is the paper's thirteen states, with no IMU bias states. Only horizontal
position is sampled. The linear states' conditional covariance depends on nothing
particle-specific, so it is held once for the whole cloud.

- **Time update, once per measurement epoch.** Between measurements only the nominal is
  mechanized, while the error-state transition and process noise accumulate. At the next
  measurement, the particles are drawn from `N = A^n_l P A^n_lᵀ + Qⁿ` and their linear states
  move by what the draw implies about them (the paper's eqs. 30-35).
- **Measurement update, one path for every measurement.** The measurement's Jacobian, carried
  onto the linear partition, gives `C`. Each particle is weighted by its residual under
  `S = C P Cᵀ + R`, the likelihood with the linear states integrated out (eq. 24). Its linear
  states then take a Kalman step with a gain shared by the cloud (eqs. 26-29). For a map, `C`
  selects `V + c`, which is the paper's update exactly.
- **Barometer readings** are the loop's input, not measurements. They don't reweight the cloud
  or start an epoch.
- **Feedback.** After every update, the weighted-mean error is folded into the nominal and the
  cloud re-centred. Position, velocity, the loop's states and the map states are added; tilt is
  composed.

`estimate()` reports the same layout as the Kalman filters:
`[lat, lon, alt, v(3), roll, pitch, yaw, b_a(3), b_g(3)]`, followed by one total map bias
`V + c` per channel. The six IMU-bias rows are zero, since the filter carries no bias states.

## Configuration

A `[particle_filter]` section holds the particle count, the initial uncertainties, and the
following:

| key | meaning | default |
| --- | --- | --- |
| `velocity_process_noise_std_mps`, `attitude_process_noise_std_rad` | VRW and ARW, per √s (eq. 20) | 1e-3, 0.01 |
| `horizontal_process_noise_std_m` | `[north, east]` random walk on the sampled position, m/√s | `[0, 0]`, the paper's eq. 19 |
| `baro_loop_time_constant_s` | the loop's gains place its three poles at `-1/τ` | 10 |
| `baro_error_std_m`, `baro_error_time_constant_s` | the barometer-aiding error as a Gauss-Markov process (`σ_b`, `τ_b`) | 8.3 m, 3600 s |
| `vertical_accel_error_init_std_mps2` | prior on `δâ` | 0.1 |
| `effective_sample_threshold` | resample below this fraction of N; `1.0` resamples every update, as the paper does | 1.0 |
| `roughening_factor` | post-resample jitter; `0.0` as in the paper | 0.2 |
| `gravity_variation_std`, `magnetic_variation_std` | steady-state sigma of each channel's `V` | from `[geophysical] *_bias_process_noise_std` |
| `gravity_variation_time_constant_s`, `magnetic_variation_time_constant_s` | correlation time of `V` | 300 (the paper's) |

The map bias seed and total prior come from `[geophysical]`, as for the Kalman filters. `V`
starts from its stationary distribution, and `c` takes the rest of `*_bias_init_std`. The
paper's magnetic temporal variation is `magnetic_variation_std = 5` with
`magnetic_variation_time_constant_s = 300`.

## Horizontal process noise: why the recipes set 1 m/√s

The paper's zero (eq. 19) makes every epoch's time update a noiseless observation of the
velocity and tilt errors. That moves their uncertainty out of the shared covariance and into
the particles' spread. Resampling on GNSS-rate fixes at metre level then destroys that spread.

On the reference recording the effect is severe:

- the conditional velocity sigma fell to millimetres per second;
- GNSS velocity fixes stopped correcting anything;
- the solution ran 17 km off.

At 1 m/√s, the setting the `conf/` recipes use, the same recording scores 3.2 m horizontal and
0.8 m vertical RMSE against the GNSS track, with 95% 3σ containment. The paper's
navigation-grade INS and magnetometer-only updates never pushed it into that regime.

## Sparse GNSS: what this structure cannot recover from

With one GNSS fix every 60 s (`conf/rbpf_degraded.toml`, `rbpf_both.toml`) the filter diverges.
On a 9-trajectory subset the median horizontal RMSE is hundreds of kilometres, against about
0.5 km for the EKF. The cause is structural, not a tuning slip. Traced on
2025-06-18_15-09-25:

1. **Heading drags the solution.** Pitch is unobservable between fixes, and a 4° pitch error
   levels the magnetometer wrongly. At this site's dip, that biases the heading measurement by
   several degrees. With the recipe's attitude noise, the heading fixes pull yaw 10-25° off,
   and reweighting on heading shifts the velocity mean through the cloud.
2. **The first fix lands outside the cloud.** The reported uncertainty is conserved: velocity
   σ grows to 30-55 m/s by t = 60 s. But the mean is about 10σ off, so the weights collapse
   onto one ancestor.
3. **Nothing brings velocity back.** Only horizontal position is sampled, so a position residual
   never enters the Kalman step. Its correlation with velocity and tilt lives only in the
   particle spread, which the collapse has just removed. All that is left is `P_vv` of about
   0.6 m/s, which a velocity residual of about 140 m/s barely moves. The EKF carries that
   correlation in `P`: its 0.9 km error at t = 60 s drops to 50 m in one fix.

Ablations that did not rescue it (subset medians):

- the time update per sample: 326 km;
- `Qⁿ = 3 m/√s`: 280 km;
- `effective_sample_threshold = 0.5`: 323 km;
- a slower barometer loop: 658 km;
- a ZVV pseudo-measurement: 4-7 km;
- the Kalman filters' VRW and ARW densities: 25 km.

The paper's setting never tests this: a navigation-grade INS, map fixes every epoch, and no GNSS.

## Departures from Canciani & Raquet

1. **Closed loop.** They ran a navigation-grade INS open loop, and noted feedback "may be
   required with a less accurate INS". A MEMS nominal needs it.
2. **Barometer loop.** The paper prints the loop's coupling into the error model but omits its
   feedback on the altitude error, and gives no gains. Both are completed from the standard
   third-order loop (Titterton & Weston).
3. **WGS84 discrete-time error Jacobian.** It stands in for the spherical-Earth continuous
   Pinson matrices; the structure is the same.
4. **Other fixes.** GNSS and magnetometer-heading fixes are supported alongside the map.
5. **Roughening** after resampling (`roughening_factor = 0.0` removes it).
6. **Process noise across an epoch.** The accumulated process noise carries each state's own
   decay across the epoch, so a Gauss-Markov state stays stationary over long gaps between
   sparse fixes.
7. **Configurable horizontal process noise.** The default is the paper's zero.

The implementation, with equation references throughout, is `core/src/rbpf.rs`.
