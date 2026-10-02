# Extended Kalman Filter (EKF)

`kalman::ExtendedKalmanFilter` is a full-state EKF: position, velocity and attitude (as roll,
pitch and yaw) are the state vector itself, and the filter linearizes the mechanization about
its current estimate with analytic Jacobians (Groves Chapter 14.2). Run it from the command line
with `strapdown-sim cl --filter ekf`. For when to prefer it over the default
[ESKF](./eskf.md), see [Comparison](./comparison.md).

## State

| configuration | states |
| --- | --- |
| 9-state (`use_biases: false`) | latitude, longitude, altitude, velocity (3), roll, pitch, yaw |
| 15-state (`use_biases: true`, the `EkfConfig` default) | the nine above, then accelerometer bias (3) and gyroscope bias (3) |
| extended | the fifteen, then any `other_states` (geophysical map biases), then the barometric bias if `estimate_baro_bias` is set |

Units and order are the [shared layout](./kalman.md#state-layout-and-units): latitude and
longitude in radians, attitude in radians on the principal branch. Extra states and the
barometric bias require `use_biases`; `initialize_ekf` refuses them otherwise.

## Predict

1. Bias-correct the increments with the current bias estimate, as the ESKF does:
   $\Delta v^b = \Delta\tilde v^b - b_a\Delta t$, $\Delta\theta^b = \Delta\tilde\theta^b - b_g\Delta t$.
2. Compute the transition Jacobian $F$ at the **pre-propagation** state with
   `linearize::euler_state_transition_jacobian`, from the bias-corrected average rates.
3. For the 15-state filter, widen $F$ with `linearize::widen_with_imu_bias_coupling`: the
   blocks through which the accelerometer bias reaches velocity (and position, through the
   trapezoidal half-step) and the gyroscope bias reaches attitude. Without them the bias states
   exist but never correlate with anything a measurement observes, and the filter estimates nine
   states while carrying fifteen; that was the EKF until #394. Extra states get an identity
   (random-walk) diagonal.
4. Mechanize the state with `mechanize`.
5. $P \leftarrow F P F^\top + q\ \Delta t$, then regularize. There is no separate noise-input
   matrix $G$: $q$ is taken to be the process-noise density already expressed in state space.

### The Jacobian is in the Euler chart

The EKF's attitude states are Euler angles, so its Jacobian differentiates with respect to
roll, pitch and yaw. That is a different matrix from the rotation-vector form an error-state
filter uses: the two differ by the Euler-rate matrix $E(\Phi)$, and the difference is as large
as the terms themselves. `linearize` names the choice through `AttitudeParametrization`
(`Euler` for this filter, `RotationVector` for the error-state form) rather than leaving it
implied by which function is called. Using the rotation-vector form in the EKF was #307.

## Update

1. Evaluate the measurement Jacobian $H$ from the model's `get_jacobian` first: a geophysical
   model whose estimate has left its map reports that as an error there, before any NaN reaches
   the innovation. Every model writes its attitude columns as $\partial h/\partial$(roll, pitch,
   yaw), which is exactly this filter's chart, so no conversion is needed. A nine-column
   Jacobian is padded with zeros to the state width.
2. $S = HPH^\top + R$, innovation $\nu = z - h(x)$, wrapped by the model for angles.
3. Gate, if a gate is installed ([gating](./measurements.md#innovation-gating)).
4. $K = PH^\top S^{-1}$ (through an SPD solve), $x \leftarrow x + K\nu$, then wrap the Euler
   angles back onto the principal branch.
5. Joseph-form covariance update, $P \leftarrow (I-KH)P(I-KH)^\top + KRK^\top$, then
   regularize.

## Construction

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:ekf_new}}
```

The arguments are an `InitialState`, a slice of initial bias estimates (used only when
`use_biases` is `true`, with any further entries seeding extra states), the covariance
diagonal, the process-noise density $q$ as a matrix, and `use_biases`. The state's width is the
length of the covariance diagonal; states beyond the nine navigation states and the supplied
seeds start at zero. `kalman_filters.rs` builds the diagonal with
`IMUQuality::auto_covariance`; see [Kalman Filters](./kalman.md#the-constructors-with-p0-from-the-imu-grade).

`sim::initialize_ekf(&record, EkfConfig)` builds the same filter from a `TestDataRecord`, which
is what `strapdown-sim` does; [Kalman Filters](./kalman.md#the-siminitialize_-helpers) lists
the `EkfConfig` fields and the $P_0$ it derives. `EkfConfig::default()` is the 15-state filter;
its `Default` is written by hand precisely so that it does not silently become the 9-state one.

Driving it is the same as any filter: `predict` with an `ImuSample` or `IMUData`, `update` with
a [measurement model](./measurements.md), each returning a `Result`
([Driving a filter](./kalman.md#driving-a-filter)).

## Units on the covariance diagonals

The state vector is not in one unit system, and neither $P_0$ nor $Q$ can be filled from a
single literal. Latitude and longitude are **radians**; altitude, velocity and the
accelerometer biases are metric; attitude and the gyroscope biases are radians and radians
per second. A horizontal uncertainty written in metres therefore has to pass through
`earth::METERS_TO_RADIANS` before it can go on the diagonal.

It is worth knowing what the round numbers mean once the conversion is skipped:

| written as a variance | as a horizontal standard deviation |
|---|---|
| `1e-6` rad² | 6367 m |
| `1e-9` rad² | 201 m |
| `(5 m × METERS_TO_DEGREES)²` | 286 m (degrees, not radians: 57.3x too large) |

All three shipped in this crate, in $Q$ and in $P_0$, and are what issue #308 fixed. The
symptom is characteristic: with $Q$ that large the innovation covariance $S = HPH^\top + R$ is
dominated by the filter's own prediction, so the update discards it and lands on each fix,
the solution tracks the fix noise one-for-one instead of averaging it down, and innovation
gating cannot function because a genuinely bad fix is still inside what the filter believes
possible.

`sim::DEFAULT_PROCESS_NOISE_DENSITY`, `sim::DEFAULT_INITIAL_POSITION_UNCERTAINTY_M` and the
`sim::initialize_*` helpers all do the conversion for you; `IMUQuality::auto_covariance`
derives a whole $P_0$ diagonal from an IMU grade and a reported fix accuracy. Remember too that
the diagonal of $Q$ is a density: the filter multiplies it by $\Delta t$
([Process noise](./kalman.md#process-noise-is-a-spectral-density)).

## Analytic Jacobians, and how they are checked

The EKF's correctness rests on its Jacobians matching the function they linearize, and a wrong
block does not crash: it makes the gain wrong, and the filter degrades in a way that looks like
bad tuning. `core/tests/jacobian_agreement.rs` therefore checks the analytic transition
Jacobians directly against central finite differences of the mechanization, rather than
inferring their correctness from whether a filter converges:

- `euler_jacobian_matches_the_mechanization_in_both_frames` bounds every entry of the Euler
  Jacobian against the finite difference, in NED and ENU, at a derived tolerance of $2\times10^{-5}$.
- `altitude_row_follows_the_frame` pins the one entry whose sign the frame decides:
  $\partial\ \text{alt}/\partial v_\text{vertical}$ is $-\Delta t$ in NED, where vertical velocity is
  positive down, and $+\Delta t$ in ENU. A wrong sign turns the altitude/vertical-velocity pair
  into positive feedback.
- `the_two_parametrisations_differ_in_every_block_that_touches_attitude` checks that the Euler
  and rotation-vector forms really are different matrices.

```bash
cargo test -p strapdown-core --test jacobian_agreement
```

The measurement Jacobians live beside them in `core/src/linearize.rs` (`gps_position_jacobian`,
`gps_velocity_jacobian`, `relative_altitude_jacobian`, `magnetometer_yaw_jacobian`, `zupt_jacobian`,
`zaru_jacobian` and others).

## Troubleshooting

**The solution copies every fix, and gating never rejects anything.** Check the units of $P_0$
and $Q$ against the table above. A horizontal variance written as if it were m² is thousands of
metres in radians.

**The vertical channel runs away, or roll converges to 180°.** The data's frame disagrees with
the declared one. In NED a level IMU reads $-g$ on its down axis; Sensor Logger exports are ENU
and need `is_enu = true` (`--enu`). `sim::check_declared_frame` catches a wrong declaration
before propagation, and `strapdown-sim` calls it on every run.

**Bias estimates do not move.** A 9-state filter has nowhere to put them; use `use_biases:
true`. On a 15-state filter, position and velocity fixes observe the biases only through the
coupling blocks, and an accelerometer bias and a small tilt produce the same horizontal
signature, so they are not separately observable from position and velocity alone
([Tutorial: GNSS Degradation](../examples/tutorial-gps-degradation.md) shows this). A ZARU
pseudo-measurement observes the gyro bias directly
([Measurement Models](./measurements.md#zero-velocity-and-zero-angular-rate-updates)).

**An update returns an error.** `StrapdownError::is_recoverable` distinguishes a measurement
that could not be evaluated (skip it) from a filter that can no longer continue.

## References

- Groves, P. D., *Principles of GNSS, Inertial, and Multisensor Integrated Navigation
  Systems*, 2nd ed., Chapter 14.2.
- Bar-Shalom, Y., Li, X.-R. and Kirubarajan, T., *Estimation with Applications to Tracking and
  Navigation*, Chapter 5.
