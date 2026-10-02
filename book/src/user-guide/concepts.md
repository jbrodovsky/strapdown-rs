# The Navigation Model

This page describes what `strapdown-core` computes: the strapdown mechanization that turns IMU
samples into position, velocity and attitude, the Earth model it relies on, and how aiding
measurements correct it. The equations are those of Groves, *Principles of GNSS, Inertial, and
Multisensor Integrated Navigation Systems*, 2nd ed., §5.4-5.5, and they are written here as the
code implements them (`mechanize`, `attitude_update`, `position_update` and the private
`velocity_update_ned` in `core/src/lib.rs`).

## Strapdown mechanization

In a *strapdown* system the inertial sensors are fixed rigidly to the vehicle, so they measure
in the vehicle's own **body frame** ($b$):

- the gyroscopes measure the **angular rate** $\boldsymbol\omega_{ib}^b$ of the body relative
  to inertial space;
- the accelerometers measure the **specific force** $\mathbf f_{ib}^b$: the non-gravitational
  force per unit mass acting on the body. This is not acceleration. An accelerometer at rest
  on a table senses the table pushing up against gravity; one in free fall senses nothing.

*Mechanization* integrates these measurements forward in time. The attitude comes from the
angular rate, the velocity from the specific force once it is resolved into the navigation
frame and gravity is restored, and the position from the velocity. No external information is
used, so any error in the measurements or the initial state is integrated too, and the
solution drifts without bound. That is why the filters exist. The [Quick
Start](../quick-start.md) shows the effect: unaided, its consumer-grade synthetic IMU drifts
about 110 km in ten minutes, while every aided run stays at the metre level.

## The navigation state

The mechanization works in the **local-level navigation frame** ($n$): north, east and down
(NED) axes attached to the vehicle's position on the WGS84 ellipsoid. It propagates:

| Symbol | Quantity | Units in the library |
| --- | --- | --- |
| $L$, $\lambda$ | Geodetic latitude and longitude | radians |
| $h$ | Height above the ellipsoid, positive up | m |
| $\mathbf v_{eb}^n = (v_N, v_E, v_D)$ | Velocity relative to the Earth, resolved in $n$ | m/s |
| $\mathbf C_b^n$ | Attitude: the direction cosine matrix from body to navigation axes | -- |

This is the library's `StrapdownState`. [Coordinate Frames](./coordinate-frames.md) covers the
frame conventions, including the ENU option, and [State
Representation](./state-representation.md) covers how the filters extend this state.

## Inputs: increments over an interval

The mechanization consumes an `ImuSample`: the integrated specific force
$\Delta\mathbf v^b = \int \mathbf f_{ib}^b dt$, the integrated angular rate
$\Delta\boldsymbol\theta^b = \int \boldsymbol\omega_{ib}^b dt$, and the interval $\tau$ they
were accumulated over. A CSV record carries instantaneous rates, which `ImuSample::from_rates`
turns into increments by rectangular integration: $\Delta\mathbf v^b = \mathbf f_{ib}^b \tau$,
$\Delta\boldsymbol\theta^b = \boldsymbol\omega_{ib}^b \tau$. That is exact only for rates
constant across the interval. No coning or sculling compensation is applied.

A filter that estimates IMU biases subtracts its current estimate, integrated over $\tau$,
from both increments before mechanizing.

## The Earth model

Three Earth quantities appear in the update equations (Groves, Chapter 2; `core/src/earth.rs`).

**Radii of curvature.** With equatorial radius $a = 6378137$ m and eccentricity
$e = 0.0818191908425$, the meridian (north-south) and transverse (east-west) radii are

$$
R_N(L) = \frac{a (1 - e^2)}{\left(1 - e^2 \sin^2 L\right)^{3/2}}, \qquad
R_E(L) = \frac{a}{\sqrt{1 - e^2 \sin^2 L}} .
$$

**Earth rate.** The Earth turns at $\omega_{ie} = 7.2921159 \times 10^{-5}$ rad/s
(`earth::RATE`). Resolved in NED axes:

$$
\boldsymbol\omega_{ie}^n = \omega_{ie} \left(\cos L, 0, -\sin L\right)^\top .
$$

**Transport rate.** Moving over the curved Earth rotates the local-level frame itself, at
(Groves eq. 5.44)

$$
\boldsymbol\omega_{en}^n = \left( \frac{v_E}{R_E(L) + h}, \quad \frac{-v_N}{R_N(L) + h}, \quad \frac{-v_E \tan L}{R_E(L) + h} \right)^\top .
$$

Only the horizontal velocity enters it. Each rate has a skew-symmetric matrix form,
$\boldsymbol\Omega = [\boldsymbol\omega \times]$, so that $\boldsymbol\Omega \mathbf x =
\boldsymbol\omega \times \mathbf x$.

**Gravity.** `earth::gravity` gives the magnitude of normal gravity from Somigliana's formula,
reduced with height by the free-air gradient:

$$
g(L, h) = g_e \frac{1 + k \sin^2 L}{\sqrt{1 - e^2 \sin^2 L}} - 3.08 \times 10^{-6} h,
\qquad k = \frac{b g_p - a g_e}{a g_e},
$$

with $g_e = 9.7803253359$ m/s² at the equator, $g_p = 9.8321849378$ m/s² at the poles, and
polar radius $b = 6356752.31425$ m. Somigliana's formula describes *normal gravity*: the
gravitational attraction combined with the centrifugal effect of the Earth's rotation, which
is the $\mathbf g_b^n$ the velocity update needs. It acts along the ellipsoid normal, so in NED
it is $\mathbf g^n = (0, 0, g)$: positive down. At 0° latitude and zero height it is
9.780 m/s², the "local gravity" the simulator logs when it checks the input frame.

## The update equations

One mechanization step takes the state at the start of an interval, marked $(-)$, to its end,
marked $(+)$. The steps run in this order because each one uses the result of the previous one.

### Attitude (Groves eq. 5.46)

$$
\mathbf C_b^n(+) \approx \mathbf C_b^n(-)\left(\mathbf I_3 + [\Delta\boldsymbol\theta^b \times]\right) - \left(\boldsymbol\Omega_{ie}^n + \boldsymbol\Omega_{en}^n\right) \mathbf C_b^n(-) \tau
$$

The first term applies the body's sensed rotation, $[\Delta\boldsymbol\theta^b\times] =
\boldsymbol\Omega_{ib}^b \tau$. The second removes the part of that rotation that is only the
navigation frame turning underneath it: the Earth's rotation, plus the transport rate. Both
rates are evaluated at the $(-)$ state. The first-order result is not exactly orthonormal, so
it is projected back onto a rotation matrix before it is stored.

### Specific force resolution (Groves eq. 5.47)

$$
\Delta\mathbf v^n = \tfrac{1}{2}\left(\mathbf C_b^n(-) + \mathbf C_b^n(+)\right)\Delta\mathbf v^b
$$

Averaging the attitude over the interval is the mechanization's one second-order term.

### Velocity (Groves eq. 5.54)

$$
\mathbf v_{eb}^n(+) \approx \mathbf v_{eb}^n(-) + \Delta\mathbf v^n + \left[\mathbf g^n - \left(\boldsymbol\Omega_{en}^n + 2 \boldsymbol\Omega_{ie}^n\right)\mathbf v_{eb}^n(-)\right]\tau
$$

The sensed increment is added directly; gravity and the Coriolis and transport-rate terms
accumulate over the interval. The last term is a plain cross product of the local-level rates
with the local-level velocity, $(\boldsymbol\omega_{en}^n + 2\boldsymbol\omega_{ie}^n) \times
\mathbf v_{eb}^n$; no frame transformation is applied to it. The unit test
`velocity_update_coriolis_term_is_a_bare_cross_product` checks it against an explicit cross
product at 45° N.

### Position (Groves eq. 5.56)

Height, latitude and longitude are updated in that order, each from the trapezoidal average of
the old and new velocity:

$$
h(+) = h(-) - \tfrac{1}{2}\left(v_D(-) + v_D(+)\right)\tau
$$

$$
L(+) = L(-) + \frac{\tau}{2}\left(\frac{v_N(-)}{R_N(L(-)) + h(-)} + \frac{v_N(+)}{R_N(L(-)) + h(+)}\right)
$$

$$
\lambda(+) = \lambda(-) + \frac{\tau}{2}\left(\frac{v_E(-)}{\left(R_E(L(-)) + h(-)\right)\cos L(-)} + \frac{v_E(+)}{\left(R_E(L(+)) + h(+)\right)\cos L(+)}\right)
$$

The minus sign in the height update is there because $h$ is positive *up* while $v_D$ is
positive *down*. The code guards $\cos L$ away from zero near the poles, wraps the longitude to
$[-\pi, \pi]$ and keeps the latitude on $[-\pi/2, \pi/2]$.

## The specific force sign convention

This is the most common way to get a first integration wrong. At rest, the accelerometer
senses the support force, so the specific force is $\mathbf f = -\mathbf g$. **In NED, an
accelerometer at rest reads about −9.8 m/s² on its down axis.** The velocity update then adds
$\mathbf g^n = (0, 0, +g)$, and the two cancel: the velocity stays at zero, as it should. In
free fall the accelerometer reads zero, and the velocity update integrates $+g$ downward.

The `syn` output follows this convention: its stationary records carry `acc_z` ≈ −9.76 m/s²
with level attitude. The input must not be gravity-compensated beforehand. Data with gravity
already removed would make a parked vehicle fall at $1g$.

The ENU convention flips the sign of the vertical axis, so there the same accelerometer reads
+9.8 m/s² on its up axis. The library does not branch on the frame inside these equations. An
ENU state and sample are reflected into NED, propagated, and reflected back, so there is
exactly one implementation of each equation. [Coordinate Frames](./coordinate-frames.md)
describes the reflection and the check that refuses data declared in the wrong frame.

## Aiding: loosely coupled filtering

Every filter here is **loosely coupled**: aiding sensors contribute already-solved quantities
(position, velocity, altitude, heading), not raw GNSS observables. There are no pseudorange or
carrier-phase models.

A filter alternates two steps:

- **Predict.** Mechanize the IMU sample as above, and propagate the uncertainty with it. For
  the Kalman filters, $\mathbf P \leftarrow \mathbf F \mathbf P \mathbf F^\top + \mathbf Q_k$,
  where $\mathbf F$ is the error-state transition matrix (the analytic Jacobians in
  `core/src/linearize.rs`; the UKF propagates sigma points instead).
- **Update.** When a measurement $\mathbf z$ arrives, compare it with the prediction
  $\hat{\mathbf z} = h(\hat{\mathbf x})$, weigh the residual $\boldsymbol\nu = \mathbf z -
  \hat{\mathbf z}$ against the predicted uncertainty $\mathbf S = \mathbf H \mathbf P
  \mathbf H^\top + \mathbf R$, and correct the state with the gain
  $\mathbf K = \mathbf P \mathbf H^\top \mathbf S^{-1}$.

In **closed loop** (`cl`, `pf`) the correction is fed back into the mechanization. The ESKF
injects its estimated errors into the nominal state and resets them to zero; the EKF and UKF
update their full state directly; the RBPF folds its weighted-mean error into its nominal
trajectory. The Kalman filters also apply their updated IMU-bias estimates to the next sample.
`dr` only mechanizes.

In `cl` the measurements come from the input records:

| Source | Model | Measures | Rate |
| --- | --- | --- | --- |
| GNSS | `GPSPositionAndVelocityMeasurement` | $L$, $\lambda$, $h$, $v_N$, $v_E$ (velocity from `speed` and `bearing`) | as the GNSS scheduler allows (every fix by default) |
| Barometer | `RelativeAltitudeMeasurement` | Height: the record's `relativeAltitude` added to the first record's `altitude`, compared with $h$ plus the barometric bias when that state is carried | 1 Hz |
| Magnetometer | `MagnetometerYawMeasurement` | Heading, corrected by World Magnetic Model declination | 1 Hz |

The particle filter takes GNSS and magnetometer heading the same way, but uses the barometer
differently: its readings drive an altitude-aiding loop inside the mechanization rather than
entering as measurement updates ([RBPF](../filters/rbpf.md)).

The GNSS measurement noise comes from the record's `horizontalAccuracy`, `verticalAccuracy` and
`speedAccuracy` columns. The GNSS degradation machinery acts on this stream: the scheduler
decides which fixes reach the filter, and the fault model alters the ones that do ([Fault
Simulation](../gnss/fault-simulation.md)). The barometer and magnetometer have their own
schedulers and are unaffected by GNSS outages. All the models, and the innovation gating that
can reject an outlier, are documented in [Measurement Models and
Integrity](../filters/measurements.md).

## Process noise is a spectral density

Process noise models what the mechanization does not: sensor noise, unmodelled dynamics, bias
drift. The library specifies it as a **spectral density** $q$, a variance per second, and
every Kalman filter forms the per-step covariance as

$$
\mathbf Q_k = \mathbf q \tau .
$$

This makes the uncertainty a trajectory accumulates a function of elapsed time rather than of
how often it was sampled. Until #374 the same numbers were added once per IMU sample with no
$\tau$, so a 50 Hz recording received fifty times the process noise of a 1 Hz one from the same
constant.

The defaults are `sim::DEFAULT_PROCESS_NOISE_DENSITY`, one entry per state of the 15-element
layout:

| States | Density | Units |
| --- | --- | --- |
| Latitude, longitude | $(0.1\ \text{m})^2$/s converted to radians: `(0.1 * METERS_TO_RADIANS)^2` | rad²/s |
| Altitude | $10^{-2}$ | m²/s |
| Velocity (3) | $10^{-3}$ | (m/s)²/s |
| Attitude (3) | $10^{-5}$ | rad²/s |
| Accelerometer bias (3) | $10^{-6}$ | (m/s²)²/s |
| Gyroscope bias (3) | $10^{-8}$ | (rad/s)²/s |

The barometric bias state, when carried, adds a random walk of
$(8.3\ \text{m})^2$ per hour, `sim::BARO_BIAS_PROCESS_NOISE_M2_PER_S`, which is the drift of
about one hectopascal of reference pressure per hour. The particle filter's noise parameters
are configured separately, as standard deviations per $\sqrt{\text{s}}$; see the
[RBPF page](../filters/rbpf.md).
