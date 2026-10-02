# Error-State Kalman Filter (ESKF)

`kalman::ErrorStateKalmanFilter` is the default filter: `strapdown-sim cl` runs it unless told
otherwise (`FilterType::default()` is `Eskf`), and `InsEngine::builder().build()` constructs
one. It is the indirect, error-state formulation of Groves Chapter 14 and Solà (2017): a
nonlinear **nominal** trajectory is mechanized at full fidelity, and a linear Kalman filter
estimates the small **error** in it.

## Why it is the default

The ESKF became the default in #258 for reasons that are structural rather than tuning:

- **It estimates the IMU biases.** The fifteen-element error state always carries
  accelerometer and gyroscope bias, coupled to velocity and attitude in the transition matrix,
  so a turn-on bias is estimated online instead of integrated as motion. A nine-state EKF has no
  bias states to put it in.
- **Attitude is a quaternion, corrected multiplicatively.** The nominal attitude is a unit
  quaternion; the error is a three-element rotation vector. Nothing is linearized about Euler
  angles, so there is no Euler-chart singularity in the propagation and no Euler addition in the
  correction.
- **The linearization point is good by construction.** The error is reset to zero after every
  update, so the linear model only ever has to describe a small perturbation.

The EKF and UKF remain selectable and are unchanged by this default. For measured accuracy of
all three on the same scenarios see [Comparison](./comparison.md).

## Nominal and error states

The **nominal** state is held as typed quantities, not a vector: latitude and longitude
(radians), altitude (m), north/east/vertical velocity (m/s), a unit quaternion for the
body-to-navigation attitude, the two bias vectors and, when carried, a barometric bias.

The **error** state $\delta x$ is

| index | error in | unit |
| --- | --- | --- |
| 0, 1 | latitude, longitude | rad |
| 2 | altitude | m |
| 3, 4, 5 | north, east, vertical velocity | m/s |
| 6, 7, 8 | attitude, as a body-frame rotation vector $\delta\theta$ | rad |
| 9, 10, 11 | accelerometer bias | m/s² |
| 12, 13, 14 | gyroscope bias | rad/s |
| 15 (optional) | barometric bias | m |

The position errors are in the filter's own units (radians for latitude and longitude), not
metres. The error-state transition matrix and the GNSS position Jacobian are both written in
those units, so the correction is added to the nominal latitude without conversion; dividing
by the radii of curvature there was #266.

The filter's width is the length of the covariance diagonal it is constructed with: fifteen,
or sixteen when the last entry is a barometric bias. `get_estimate` returns the *nominal* state
in the [shared layout](./kalman.md#state-layout-and-units), converting the quaternion to Euler
angles; `get_certainty` returns the *error-state* covariance.

## Predict

1. **Bias-correct the increments.** Biases are rates and the increments are their integrals:
   $$
   \Delta v^b = \Delta \tilde v^b - b_a \Delta t, \qquad
   \Delta\theta^b = \Delta\tilde\theta^b - b_g \Delta t .
   $$
2. **Mechanize the nominal** with `mechanize`, the same strapdown equations every filter
   uses, and renormalize the quaternion.
3. **Propagate the error covariance** with the error-state transition matrix from
   `linearize::error_state_transition_jacobian` (Groves 14.2.4, Solà 2017 Section 6.3),
   evaluated at the propagated nominal with the bias-corrected average rates:
   $$
   P \leftarrow F_{\delta x}\  P\  F_{\delta x}^\top + q\ \Delta t .
   $$
   $F_{\delta x}$ couples the accelerometer bias into velocity and the gyroscope bias into
   attitude, which is what makes the biases observable from position and velocity fixes. A
   barometric bias is a random walk with no coupling: it reaches the navigation states only
   through the barometer's measurement Jacobian.

## Update, injection and reset

1. **Predict the measurement** from the nominal state, assembled into the
   [shared layout](./kalman.md#state-layout-and-units) including the bias entries (ZARU reads
   them).
2. **Form $H$.** Every measurement model writes its attitude columns as $\partial h /
   \partial(\text{roll, pitch, yaw})$. The ESKF's attitude error is a body-frame rotation
   vector, so those columns are converted by the chain rule,
   $$
   \frac{\partial h}{\partial \delta\theta^b} =
   \frac{\partial h}{\partial \Phi}\ \frac{\partial \Phi}{\partial \delta\theta^b},
   $$
   with the second factor from `linearize::body_rotation_vector_to_euler_jacobian`. It is the
   identity only at zero roll. Near gimbal lock (about 89.45° of pitch) no finite matrix is
   right, and the attitude columns are zeroed: the update declines to correct attitude from
   that measurement rather than apply an enormous row.
3. **Gate**, if a gate is installed (see [gating](./measurements.md#innovation-gating)). A
   rejection inflates the covariance in the observed directions and returns before anything
   is injected; there is no undo after injection starts.
4. **Solve** $\delta x = K\nu$ with $K = P H^\top (H P H^\top + R)^{-1}$.
5. **Inject** the error into the nominal:
   - position, velocity and biases are added;
   - attitude is composed, $q \leftarrow q \otimes \operatorname{Exp}(\delta\theta)$, using the
     exact exponential map rather than the first-order quaternion $[1, \tfrac12\delta\theta]$;
   - the barometric bias, when carried, is added.
6. **Clamp the IMU biases** (anti-windup): each accelerometer bias to ±2.0 m/s² and each gyro
   bias to ±0.05 rad/s. These are orders of magnitude above a consumer MEMS turn-on bias and
   exist so that a persistently faulty aiding source cannot drag the biases to physically
   impossible values (#286). The barometric bias is not clamped: it does not feed the
   mechanization.
7. **Reset** $\delta x \leftarrow 0$.
8. **Update the covariance** in Joseph form, using the same prior $P$ the gain was computed
   from: $P \leftarrow (I-KH)P(I-KH)^\top + KRK^\top$.
9. **Transport the covariance** onto the tangent space of the attitude just composed on:
   $P \leftarrow G P G^\top$, where $G$ is the identity except for the attitude block, which
   is the right Jacobian $J_r(\delta\theta)$ from `linearize::attitude_reset_jacobian`. The
   whole covariance is transformed, so attitude's correlations with the other states move with
   it (#398).
10. **Regularize**: symmetrize and add the relative jitter described on
    [Kalman Filters](./kalman.md#the-update-and-keeping-the-covariance-healthy).

## Construction

The direct constructor takes an `InitialState`, six initial bias estimates, the covariance
diagonal and the process-noise density $q$:

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:eskf_new}}
```

`initialize_eskf` builds the same filter from a `TestDataRecord` and an `EskfConfig`; its
default $P_0$ is tabulated on [Kalman Filters](./kalman.md#the-siminitialize_-helpers). For the
sixteen-state form pass a sixteen-element diagonal and $q$, and tell the barometer model where
the bias is through `AidingConfig::baro_bias_index`; `EskfConfig::estimate_baro_bias` does the
first half for you, and `EskfConfig::baro_bias_index()` returns the index (always 15) for the
second.

## Running it end to end

[`core/examples/eskf.rs`](https://github.com/jbrodovsky/strapdown-rs/blob/main/core/examples/eskf.rs)
is the library route to what `strapdown-sim cl` does: a five-minute synthetic drive due north at
15 m/s from a consumer-grade IMU, GNSS for 120 s, a 60 s outage, then GNSS again, scored against
the exact trajectory.

```bash
cargo run -p strapdown-core --example eskf
```

Build the filter with the barometric-bias state `strapdown-sim` turns on by default:

```rust,ignore
{{#include ../../../core/examples/eskf.rs:build}}
```

Build the event stream. A `DutyCycle` scheduler starts with `start_phase_s` of availability,
then alternates `off_s` of outage and `on_s` of availability; with a zero phase the run would
start in an outage.

```rust,ignore
{{#include ../../../core/examples/eskf.rs:stream}}
```

Run the closed loop and score it with `metrics::evaluate`:

```rust,ignore
{{#include ../../../core/examples/eskf.rs:run}}
```

```rust,ignore
{{#include ../../../core/examples/eskf.rs:score}}
```

Read the bias estimates back off the state vector:

```rust,ignore
{{#include ../../../core/examples/eskf.rs:biases}}
```

It printed:

```text
scored 3000 samples against truth
horizontal error: RMSE 8.49 m, max 54.72 m
end of outage: error 52.98 m, filter 1-sigma 378.53 m north, 379.08 m east
final accelerometer bias estimate: [-0.0541, -0.0320, -0.0694] m/s^2
final gyroscope bias estimate:     [0.00005, -0.00025, 0.00036] rad/s
final barometric bias estimate:    3.280 m
```

The "end of outage" line is the last sample before GNSS returns. On this run the filter's own
horizontal uncertainty at that point is several times its actual error: it is conservative,
not over-confident, while coasting. Whether that holds in general is what the consistency
metrics on [Performance Baselines](../development/performance.md) measure: the NEES and the
three-sigma containment columns, read together.

## References

- Groves, P. D., *Principles of GNSS, Inertial, and Multisensor Integrated Navigation
  Systems*, 2nd ed., Chapter 14 (14.2.4 for the error-state transition matrix).
- Solà, J., "Quaternion kinematics for the error-state Kalman filter", 2017.
- Trawny, N. and Roumeliotis, S., "Indirect Kalman Filter for 3D Attitude Estimation", 2005.

The implementation, with the reasoning behind each step in its comments, is
`ErrorStateKalmanFilter` in `core/src/kalman.rs` and `error_state_transition_jacobian` in
`core/src/linearize.rs`.
