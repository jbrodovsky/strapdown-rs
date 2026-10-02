# Kalman Filters

`strapdown-core` ships three Kalman-family filters in `core/src/kalman.rs`. All three are
loosely coupled: they take GNSS position and velocity fixes (and the other aiding sources on
[Measurement Models and Integrity](./measurements.md)), not pseudoranges or carrier phase. They
share one interface, one set of measurement models and one propagation core: the strapdown
mechanization of Groves Chapter 5.4-5.5 described in
[The Navigation Model](../user-guide/concepts.md).

| filter | type | state | attitude carried as | `cl --filter` |
| --- | --- | --- | --- | --- |
| [Error-State Kalman Filter](./eskf.md) | `kalman::ErrorStateKalmanFilter` | 15-element error state, 16 with a barometric bias | unit quaternion, corrected multiplicatively | `eskf` (the default) |
| [Extended Kalman Filter](./ekf.md) | `kalman::ExtendedKalmanFilter` | 9 or 15 full states, plus optional extra states | Euler angles in the state vector | `ekf` |
| [Unscented Kalman Filter](./ukf.md) | `kalman::UnscentedKalmanFilter` | 15 full states through `initialize_ukf` (9 is possible from the constructor), plus optional extra states | Euler angles, perturbed and averaged on the rotation group | `ukf` |

The [Comparison](./comparison.md) page says when to reach for which. The
[Rao-Blackwellized particle filter](./rbpf.md) implements the same trait and is documented
separately.

## The `NavigationFilter` trait

Every filter implements `strapdown::NavigationFilter`, which lives at the crate root
(`core/src/lib.rs`), not in `kalman`. The trait is object-safe, so a
`Box<dyn NavigationFilter>` can hold any of them; that is what lets the
[`InsEngine`](../user-guide/library.md) drive a filter it did not construct.

| method | returns | what it does |
| --- | --- | --- |
| `predict(&mut self, control_input: &dyn InputModel, dt: f64)` | `Result<(), StrapdownError>` | Propagate by one inertial sample. |
| `update(&mut self, measurement: &dyn MeasurementModel)` | `Result<UpdateOutcome, StrapdownError>` | Correct with one measurement. |
| `get_estimate(&self)` | `DVector<f64>` | The current state, in the filter's native units. |
| `get_certainty(&self)` | `DMatrix<f64>` | The current covariance. For the ESKF this is the *error-state* covariance. |
| `set_innovation_gate(&mut self, gate: Option<InnovationGate>)` | `bool` | Install or clear an innovation gate; `true` if the filter honours it. |
| `set_gate_recovery(&mut self, recovery: GateRecovery)` | `bool` | Tune how a rejection is recovered from; `true` if honoured. |
| `baro_bias_index(&self)` | `Option<usize>` | Where the barometric bias lives in `get_estimate`, if the filter carries one. |

The last three have default bodies (`false`, `false`, `None`). All three Kalman filters and the
RBPF override the two gating methods.

### Inputs: `InputModel`

`predict` takes a `&dyn InputModel` and accepts the two inertial forms:

- `ImuSample`: integrated increments ($\Delta v$, $\Delta\theta$) over its own `dt`. This is
  what an IMU emits and what `mechanize` consumes. Its `dt` must agree with the `dt` argument
  to a relative tolerance of $10^{-9}$, or `predict` returns
  `StrapdownError::InconsistentTimestep` rather than silently preferring one of the two.
- `IMUData`: instantaneous specific force (m/s²) and angular rate (rad/s), which the filter
  integrates over `dt` with `ImuSample::from_rates`.

Anything else that implements `InputModel` (`VelocityData`, for example) is refused with
`StrapdownError::UnsupportedInput`. Accelerometer readings are raw specific force, gravity
included: in NED a level, stationary IMU reads $-g$ on its down axis. See
[Tutorial: Basic INS](../examples/tutorial-basic.md#the-sign-convention-that-catches-everyone).

### Outputs: `UpdateOutcome`

`update` returns an `UpdateOutcome` (defined in `gating`, re-exported at the crate root) rather
than `()`, because three cases have to be told apart and only one of them is an error:

| field | meaning |
| --- | --- |
| `nis` | normalized innovation squared, $\nu^\top S^{-1}\nu$ with $S = HPH^\top + R$ |
| `dof` | the measurement's dimension, the degrees of freedom of the test |
| `accepted` | whether the correction was applied |
| `forced` | applied *despite* failing the gate, because the recovery policy forced it |

A filter with no gate installed (the default) accepts every measurement and still reports the
NIS. A gated-out measurement is `Ok` with `accepted == false`: the state is unchanged apart from
the covariance inflation described under [gating](./measurements.md#innovation-gating). An
`Err` means the measurement could not be evaluated at all, and
`StrapdownError::is_recoverable` says whether to skip it and carry on (see
[Using the Library](../user-guide/library.md#errors)).

### Driving a filter

[`core/examples/kalman_filters.rs`](https://github.com/jbrodovsky/strapdown-rs/blob/main/core/examples/kalman_filters.rs)
builds all three filters two ways, puts the six behind the trait, and runs one loop over a
two-minute synthetic drive with a GNSS position fix once a second:

```bash
cargo run -p strapdown-core --example kalman_filters
```

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:trait_objects}}
```

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:loop}}
```

It printed:

```text
ESKF, initialize_eskf: 15 states, 2.763 m from truth at the end, last update NIS 3.530 (3 dof, accepted: true)
EKF,  initialize_ekf: 15 states, 2.761 m from truth at the end, last update NIS 3.530 (3 dof, accepted: true)
UKF,  initialize_ukf: 15 states, 2.759 m from truth at the end, last update NIS 3.533 (3 dof, accepted: true)
ESKF, constructor: 15 states, 2.762 m from truth at the end, last update NIS 3.529 (3 dof, accepted: true)
EKF,  constructor: 15 states, 2.762 m from truth at the end, last update NIS 3.529 (3 dof, accepted: true)
UKF,  constructor: 15 states, 2.760 m from truth at the end, last update NIS 3.532 (3 dof, accepted: true)
```

On a short straight drive with a 2.5 m fix every second, all six end up where the fixes put
them. That is a property of this easy scenario, not a ranking; the gated comparison is on
[Comparison](./comparison.md).

## State layout and units

All three filters report the same layout from `get_estimate`:

| index | state | unit |
| --- | --- | --- |
| 0, 1 | latitude, longitude | **radians** |
| 2 | altitude above the WGS84 ellipsoid, positive up in both frames | m |
| 3, 4, 5 | north, east, vertical velocity (vertical is positive down in NED, up in ENU) | m/s |
| 6, 7, 8 | roll, pitch, yaw (XYZ Euler sequence, body to navigation) | rad |
| 9, 10, 11 | accelerometer bias, body frame | m/s² |
| 12, 13, 14 | gyroscope bias, body frame | rad/s |
| 15 onward | extra states: geophysical map biases, then the barometric bias | their own units |

A nine-state EKF stops at index 8. Latitude and longitude are radians and altitude is metres,
so no covariance diagonal, $P_0$ or $Q$, can be filled with one number for all three position
entries. [Units on the covariance diagonals](./ekf.md#units-on-the-covariance-diagonals) shows
what goes wrong when it is.

The frame is a property of the data. Every filter is NED unless its `InitialState` or config
sets `is_enu`; see [Coordinate Frames](../user-guide/coordinate-frames.md).

## Process noise is a spectral density

Every filter forms the per-step process noise as

$$
Q_k = q \  \Delta t
$$

from a diagonal $q$ of **variances per second**, and adds it to the propagated covariance. The
noise a run accumulates therefore depends on elapsed time, not on how often the IMU was sampled.
Before #374 the same numbers were added once per sample, which made the effective noise
proportional to the sample rate. The `process_noise` argument of every constructor, and every
`process_noise_diagonal` config field, is $q$, not $Q_k$.

The default is `sim::DEFAULT_PROCESS_NOISE_DENSITY`, fifteen entries in the state order above:

| states | density $q$ |
| --- | --- |
| latitude, longitude | $(0.1\ \text{m} \cdot \texttt{METERS\\_TO\\_RADIANS})^2$ rad²/s, i.e. 0.1 m/√s |
| altitude | $10^{-2}$ m²/s, i.e. 0.1 m/√s |
| velocity (3) | $10^{-3}$ (m/s)²/s |
| attitude (3) | $10^{-5}$ rad²/s |
| accelerometer bias (3) | $10^{-6}$ (m/s²)²/s |
| gyroscope bias (3) | $10^{-8}$ (rad/s)²/s |

The position entries come from one constant, `sim::POSITION_PROCESS_NOISE_M_PER_ROOT_S`
(0.1 m/√s), converted to radians for the horizontal pair. The rest are hand-tuned values, not
derived from a sensor model. A nine-state EKF takes the leading nine entries. A barometric bias
state walks at `sim::BARO_BIAS_PROCESS_NOISE_M2_PER_S`, derived from 8.3 m (about one
hectopascal) of reference drift per hour.

## The update, and keeping the covariance healthy

The measurement update is the standard one (Groves Chapter 3):
$K = P H^\top (H P H^\top + R)^{-1}$, with the gain solved through a symmetric
positive-definite solver rather than an explicit inverse. Angular innovations (magnetometer
yaw) are wrapped onto $[-\pi, \pi)$ by the model's `wrap_residual` before they are used.

- The **EKF and ESKF** update the covariance in **Joseph form**,
  $P^+ = (I - KH) P (I - KH)^\top + K R K^\top$, which stays symmetric and positive
  semi-definite under rounding where $(I - KH)P$ does not.
- The **UKF** subtracts $K S K^\top$, its sigma-point equivalent.

After every predict and update each filter symmetrizes $P$ and adds a jitter proportional to
each diagonal entry's own scale: a relative $10^{-9}$, with the process-noise diagonal as a
floor. An absolute jitter cannot work on a state whose units span twelve orders of magnitude.
The absolute $10^{-9}$ the UKF and EKF used to add was about 200 m of horizontal standard
deviation on a radian-valued latitude (#373).

## Initialization

A filter needs an initial state, an initial covariance $P_0$ and a process-noise density $q$.
There are two routes.

### The `sim::initialize_*` helpers

`initialize_eskf`, `initialize_ekf` and `initialize_ukf` (in `core/src/sim.rs`) build a filter
from one `TestDataRecord`, which is how `strapdown-sim cl` starts every run:

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:helpers}}
```

The record supplies the initial state through `TestDataRecord::initial_state(is_enu)`: position
from its GNSS columns, attitude from its quaternion, horizontal velocity from its `speed` and
`bearing`, and zero vertical velocity. Each helper takes a config struct. All three are
`#[non_exhaustive]`, so outside the crate you start from `default()` and assign fields.

| field | `EskfConfig` | `EkfConfig` | `UkfConfig` | meaning |
| --- | :-: | :-: | :-: | --- |
| `is_enu` | yes | yes | yes | frame of the records; `false` (NED) by default |
| `imu_quality` | yes | yes | yes | IMU grade the bias prior is derived from; `Consumer` by default |
| `imu_biases` | yes | yes | yes | initial bias **estimate**, 6 values; zero by default |
| `imu_biases_covariance` | yes | yes | yes | initial bias variances, 6 values; default `imu_quality.initial_bias_covariance()` |
| `attitude_covariance` | yes | yes | yes | initial attitude variances, 3 values, rad² |
| `process_noise_diagonal` | yes | yes | yes | $q$, one entry per state; default `DEFAULT_PROCESS_NOISE_DENSITY` |
| `estimate_baro_bias` | yes | yes | yes | add a barometric-bias state; `false` by default |
| `use_biases` | | yes | | 15 states (`true`, the default) or 9 |
| `other_states`, `other_states_covariance` | | yes | yes | extra states appended after the fifteen |
| `ukf_alpha`, `ukf_beta`, `ukf_kappa` | | | yes | sigma-point parameters; see [UKF](./ukf.md) |

The helpers do not derive $P_0$ identically:

| $P_0$ block | `initialize_eskf` | `initialize_ekf`, `initialize_ukf` |
| --- | --- | --- |
| horizontal position | `DEFAULT_INITIAL_POSITION_UNCERTAINTY_M` (10 m), as rad² | the record's `horizontal_accuracy`, as rad² |
| altitude | (10 m)² | the record's `vertical_accuracy`, squared |
| velocity | $10^{-3}$ (m/s)² | the record's `speed_accuracy`, squared |
| attitude | $10^{-5}$ rad² | $10^{-9}$ rad² |
| IMU biases | `imu_quality.initial_bias_covariance()` | the same |

`estimate_baro_bias` defaults to `false` on the three library configs but to `true` in
`strapdown-sim` (`ClosedLoopConfig`; `--no-estimate-baro-bias` turns it off). A filter that
carries the bias state must also have it *named* to the barometer model, through
`AidingConfig::baro_bias_index`, or nothing observes it. Read the index from the config's
`baro_bias_index()` method; [ESKF](./eskf.md#running-it-end-to-end) shows the pair.

The helpers do not check the declared frame against the data, because one record cannot tell a
frame error from a manoeuvre. Call `sim::check_declared_frame` over the whole record slice, as
`strapdown-sim` does.

### The constructors, with P0 from the IMU grade

Each filter can also be built directly from a `kalman::InitialState`. `InitialState` holds its
latitude, longitude and Euler angles in whatever unit its `in_degrees` flag names, and the
constructors convert exactly when that flag is set.

`IMUQuality::auto_covariance` derives a fifteen-element $P_0$ from two things known at start-up:
the IMU grade, and how well the first fix placed the vehicle (an `InitialUncertainty` holding
one-sigma horizontal and vertical position in metres and velocity in m/s). Each block comes back
in the filter's own units:

| states | derived from |
| --- | --- |
| position | the uncertainty, converted to rad² through the WGS84 radii at the given latitude (Groves Eq. 2.105-2.106) |
| velocity | the uncertainty, plus the grade's velocity random walk over a one-minute alignment |
| attitude | the grade's angle random walk over the same minute, plus the levelling error a bias-instability-sized accelerometer bias causes (Groves 5.6.3) |
| accelerometer bias | the grade's bias instability, squared |
| gyroscope bias | the grade's bias instability, converted from rad/h to rad/s, squared |

It treats heading like roll and pitch, so widen entry 8 when the initial yaw came from a
magnetometer or a gyrocompass, as
[Using the Library](../user-guide/library.md#coarse-alignment) does.

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:initial_covariance}}
```

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:constructors}}
```

The constructors are infallible and take the filter's width from the length of the covariance
diagonal; they do not check it. The `initialize_*` helpers do, and return
`StrapdownError::InvalidConfiguration` on a mismatch.

## Where next

- [ESKF](./eskf.md): the default filter, its error state and its reset.
- [EKF](./ekf.md): analytic Jacobians, and the units traps on $P_0$ and $Q$.
- [UKF](./ukf.md): sigma points, and why $\alpha = 0.1$.
- [Measurement Models and Integrity](./measurements.md): what `update` accepts, and gating.
- [Closed Loop](../user-guide/closed-loop.md): running these filters from the command line.
