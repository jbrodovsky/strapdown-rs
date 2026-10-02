# Coordinate Frames

Every number in an input or output file is resolved in some frame, and a wrong frame does not
produce a slightly worse answer: it produces one that falls out of the sky at twice gravity.
This page lists the frames the library uses, the one choice you have to make (NED or ENU),
and the check that catches a wrong choice.

## The frames

Following Groves' notation:

| Frame | Symbol | Used for |
| --- | --- | --- |
| Inertial | $i$ | The reference the IMU measures against: $\boldsymbol\omega_{ib}^b$ and $\mathbf f_{ib}^b$ are motion of the body relative to inertial space |
| Earth-fixed | $e$ | The rotating Earth. Velocity is Earth-relative, $\mathbf v_{eb}^n$; the Earth's rotation enters as $\boldsymbol\omega_{ie}$ |
| Local-level navigation | $n$ | Where position, velocity and attitude are expressed: axes north, east and vertical at the vehicle's position on the WGS84 ellipsoid |
| Body | $b$ | The IMU's own sensor axes, in which `acc_*` and `gyro_*` are measured |

Position is geodetic: latitude and longitude on the WGS84 ellipsoid, and height above that
ellipsoid. The library holds latitude and longitude in **radians** (`StrapdownState`); CSV
inputs and outputs carry them in **degrees**. The local-level mechanization is meant for
positions near the Earth's surface, roughly 11 km below to 30 km above the ellipsoid;
`StrapdownState::new` rejects an altitude beyond ±30 km. The simulator's default altitude
health limits are far wider (±10⁸ m) and are set with `--health-alt-min-m` and
`--health-alt-max-m`. There is no ECEF or ECI mechanization.

## NED is the default; ENU is an opt-in

The local-level axes are always ordered **north, east, vertical**. The frame choice only
decides which way the vertical axis points:

| | NED (default) | ENU (opt-in) |
| --- | --- | --- |
| Vertical axis points | down | up |
| `velocity_vertical` | positive **down**: a climbing vehicle has negative vertical velocity | positive **up** |
| Gravity in the navigation frame | $(0, 0, +g)$ | $(0, 0, -g)$ |
| Accelerometer at rest, level | ≈ −9.8 m/s² on the body's vertical axis | ≈ +9.8 m/s² on the body's vertical axis |
| `altitude` | height above the ellipsoid, positive **up** | height above the ellipsoid, positive **up** |

Note the last row: **`altitude` is positive up in both frames.** It is a height, not a "down"
coordinate. Only the vertical velocity, the gravity vector and the vertical sensor axis change
sign with the frame. Note also that "ENU" here keeps the north-then-east order of the
horizontal axes; it does not swap them to east-then-north. The horizontal velocities are
`velocity_north` and `velocity_east` in both.

NED matches Groves and aerospace practice, and it is what `strapdown-sim syn` writes. Data
from the Sensor Logger phone app is ENU: at rest its specific force lands on the device's up
axis at +9.8 m/s².

### Declaring the frame

A CSV file carries no frame tag, so the frame is always the caller's declaration:

| Where | NED (default) | ENU |
| --- | --- | --- |
| `strapdown-sim dr`, `cl`, `pf` | omit the flag | `--enu` |
| `strapdown-sim syn` (to *write* ENU records) | omit the flag | `--enu` |
| Scenario file | `is_enu = false`, or omit it | `is_enu = true` |
| `StrapdownState` | `is_enu: false` (the `Default`) | `is_enu: true` |
| `kalman::InitialState::new` | last argument `None` or `Some(false)` | `Some(true)` |
| `sim::UkfConfig`, `EkfConfig`, `EskfConfig` | `is_enu: false` (the default) | `is_enu: true` |
| `sim::dead_reckoning(records, is_enu)` | `false` | `true` |

`syn --enu` output is what `dr --enu` expects, and plain `syn` output is what plain `dr`
expects.

### Converting a state

`StrapdownState::to_ned` and `StrapdownState::to_enu` reinterpret an existing state in the
other convention, and are no-ops when the state is already in the target frame. Converting:

- negates `velocity_vertical`;
- leaves `altitude`, `latitude`, `longitude` and the horizontal velocities untouched;
- conjugates the attitude by the reflection $\mathbf F = \mathrm{diag}(1, 1, -1)$:
  $\mathbf C' = \mathbf F \mathbf C \mathbf F$. The reflection is applied to the navigation
  frame's vertical axis *and* to the body frame's, which keeps $\mathbf C'$ a proper rotation.
  In Euler angles this leaves yaw unchanged and negates roll and pitch;
- flips the `is_enu` flag.

Internally the mechanization uses exactly this: an ENU state and its IMU sample are reflected
into NED, propagated with the NED equations of [The Navigation Model](./concepts.md), and
reflected back. There is one implementation of each equation rather than a frame branch
inside each.

## The frame check

Declaring the wrong frame is refused before anything is integrated. For every mode,
`sim::check_declared_frame` takes the first 10 records (`sim::FRAME_CHECK_SAMPLES`), rotates
each one's specific force into the navigation frame using the record's attitude quaternion,
and averages the vertical component. At rest that should be $-g$ in NED and $+g$ in ENU. If
the mean lies on the wrong side by more than half of local gravity
(`sim::FRAME_CHECK_MARGIN_G = 0.5`), the run stops with `StrapdownError::InvalidConfiguration`
on the field `is_enu`, and the message names the flag to change.

`syn` output is NED. Declaring it ENU:

```console
$ strapdown-sim --log-level warn cl --enu -i synthetic.csv -o eskf_enu.csv
2026-10-01 20:42:51.596 [ERROR] - Error running closed-loop simulation on synthetic.csv: invalid configuration for `is_enu`: mean vertical specific force over the first 10 record(s) is -9.75 m/s^2, but ENU mechanization expects +9.78 m/s^2 at rest: these look like NED records. Mechanizing them as ENU would double-count gravity and integrate at 2 g. Declare the frame that matches the data (drop `--enu`, or set `is_enu = false` in the config file), or re-record it in ENU.
Error: InvalidConfiguration { field: "is_enu", reason: "mean vertical specific force over the first 10 record(s) is -9.75 m/s^2, but ENU mechanization expects +9.78 m/s^2 at rest: these look like NED records. Mechanizing them as ENU would double-count gravity and integrate at 2 g. Declare the frame that matches the data (drop `--enu`, or set `is_enu = false` in the config file), or re-record it in ENU." }
```

The command exits with status 1 and writes no output file. `dr` and `pf` refuse the same way.
When the declaration is right, the check logs what it measured at the `info` level:

```text
[INFO] - Mechanizing input as NED: mean vertical specific force over the first 10 record(s) is -9.754 m/s^2 against a local gravity of 9.780 m/s^2
```

The reason for refusing rather than warning: mechanizing NED records as ENU adds the gravity
model to the sensed specific force instead of cancelling it, so a stationary solution falls at
$2g$. The check fails *open* on data it cannot judge (empty input, non-finite values), and the
half-gravity margin leaves room for a vehicle that is not at rest when the recording starts.

## The body frame

The body frame is whatever the IMU's sensor axes are; the library does not assume
forward-right-down. The attitude $\mathbf C_b^n$ maps a body-frame vector into the
navigation frame. With all three Euler angles zero, the body axes coincide with the navigation
axes, so in NED a level, north-facing IMU has $x$ north, $y$ east and $z$ down. That is why the
`syn` records, which are level with zero heading, read about −9.76 m/s² on `acc_z`.

The initial attitude of a run comes from the first record's quaternion (`qw`, `qx`, `qy`,
`qz`), via `TestDataRecord::attitude`, not from its `roll`, `pitch` and `yaw` columns. In
Sensor Logger data those columns use the app's own angle convention, which disagrees with the
one below. An all-NaN or zero-norm quaternion is read as the identity.

## Attitude: DCM and Euler angles

Attitude is held as a direction cosine matrix $\mathbf C_b^n$ (a `nalgebra::Rotation3`; the
ESKF carries a unit quaternion internally). Euler angles are used for input and output only.
They are roll $\phi$, pitch $\theta$ and yaw $\psi$, with

$$
\mathbf C_b^n = \mathbf R_z(\psi) \mathbf R_y(\theta) \mathbf R_x(\phi),
$$

the convention of `nalgebra::Rotation3::from_euler_angles(roll, pitch, yaw)`: roll about $x$
is applied first, then pitch about $y$, then yaw about $z$, all about fixed axes. This is the
same matrix as Groves' yaw-pitch-roll Euler convention for $\mathbf C_b^n$.

In output files the angles are radians, on the branch `Rotation3::euler_angles` returns: roll
and yaw on $[-\pi, \pi]$, pitch on $[-\pi/2, \pi/2]$. Yaw is measured from north; in NED a
negative yaw is west of north. The EKF and UKF carry pitch as a plain state element, so their
pitch is wrapped to $[-\pi, \pi]$ rather than limited to $\pm\pi/2$.
