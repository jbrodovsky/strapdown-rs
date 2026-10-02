# Glossary

- **ARW / VRW** (angle / velocity random walk): The integrated effect of white noise on a
  gyroscope or accelerometer. Integrating white angular-rate noise gives an attitude error whose
  standard deviation grows with the square root of time; ARW is that growth, in rad/√s (often
  quoted in deg/√h). VRW is the same for velocity, in m/s/√s. The RBPF's velocity and attitude
  process noise are VRW and ARW.
- **CEP** (circular error probable): The radius of a circle, centred on truth, containing a
  given share of horizontal position errors. CEP50 contains half of them, CEP95 95%.
  `strapdown::metrics` computes the empirical (nearest-rank) value from the errors, not the
  Rayleigh approximation.
- **DCM** (direction cosine matrix): The 3×3 rotation matrix between two frames, here body to
  navigation frame. The attitude representation the mechanization propagates; roll, pitch and
  yaw are derived from it.
- **EKF** (extended Kalman filter): A Kalman filter for a nonlinear system that propagates the
  full state through the nonlinear model and the covariance through its Jacobians. See
  [EKF](../filters/ekf.md).
- **ENU / NED**: The two local-level frame conventions this crate supports. NED
  (North-East-Down) is the default. "ENU" is opt-in with `--enu` or `is_enu`, and here it is
  **not** the textbook East-North-Up: it keeps the north-then-east order of the horizontal
  axes (`velocity_north`, `velocity_east` in both) and flips only the vertical axis to point
  up, which is what `StrapdownState::to_enu` does. Altitude is positive up in both; vertical
  velocity is positive down in NED and up in "ENU". See [Coordinate
  Frames](../user-guide/coordinate-frames.md).
- **ESKF** (error-state Kalman filter): A Kalman filter that estimates the *error* in a
  separately propagated nominal solution, folds the estimate back in after each update, and
  resets. This crate's ESKF carries a multiplicative attitude error and is the closed-loop
  default. See [ESKF](../filters/eskf.md).
- **Gauss-Markov process** (first order): A random process whose autocorrelation decays
  exponentially, $R(\Delta t) = \sigma^2 e^{-|\Delta t|/\tau}$, with steady-state standard
  deviation $\sigma$ and correlation time $\tau$. In discrete time, an AR(1) process with $\rho
  = e^{-\Delta t/\tau}$. Used for correlated GNSS errors (`degraded` with `tau_pos_s`) and for
  map-bias variation.
- **GNSS** (global navigation satellite system): GPS, Galileo, GLONASS, BeiDou and the like.
  Here, the source of the position and velocity fixes that aid the INS, and the thing the
  degradation scenarios take away.
- **IMU** (inertial measurement unit): An accelerometer triad and a gyroscope triad measuring
  specific force and angular rate in the body frame.
- **INS** (inertial navigation system): An IMU plus the computation that integrates its output
  into position, velocity and attitude.
- **Lever arm**: The vector from the IMU to another sensor on the same body, typically the GNSS
  antenna. A rotating vehicle moves the two points differently, so a fix is translated through
  the lever arm before it is compared with the INS solution. `InsEngine` applies this
  compensation.
- **Loosely coupled**: An integration that aids the INS with the GNSS receiver's *solution*
  (position and velocity) rather than with its raw pseudorange and carrier-phase measurements
  (tightly coupled). This crate is loosely coupled only.
- **MEMS** (micro-electro-mechanical systems): The silicon sensor technology of consumer and
  low-cost IMUs, such as the one in a phone. Cheap and small, with biases and noise far larger
  than navigation-grade sensors.
- **NEES / NIS** (normalized estimation error squared / normalized innovation squared):
  Consistency statistics. NEES is $e^\top P^{-1} e$ for the true estimation error $e$ and the
  filter's covariance $P$, so it needs truth; its expected value is the dimension of $e$. NIS is
  $\nu^\top S^{-1} \nu$ for the innovation $\nu$ and its predicted covariance $S$, so it needs
  no truth and can be computed online; innovation gating thresholds it against a chi-squared
  quantile.
- **PNT** (positioning, navigation and timing): The services GNSS provides, and the usual name
  for the field of providing them by other means when GNSS is unavailable.
- **RBPF** (Rao-Blackwellized particle filter): A particle filter that samples only the strongly
  nonlinear part of the state (here, horizontal position) and handles the rest with a Kalman
  filter conditioned on each sample. The crate's only particle filter. See
  [RBPF](../filters/rbpf.md).
- **Specific force**: What an accelerometer measures: the non-gravitational force per unit mass,
  that is, acceleration minus gravitational acceleration. At rest it points up with magnitude
  $g$. The mechanization adds the gravity model back.
- **Strapdown**: An INS whose sensors are fixed ("strapped down") to the vehicle body, so that
  attitude is computed by integrating the gyroscopes rather than maintained by a gimballed
  platform.
- **UKF** (unscented Kalman filter): A Kalman filter that propagates a set of deterministically
  chosen sigma points through the nonlinear model instead of linearizing it. See
  [UKF](../filters/ukf.md).
- **ZUPT / ZARU** (zero-velocity update / zero angular-rate update): Pseudo-measurements applied
  while the vehicle is known to be stationary: velocity is zero, and the gyroscopes should read
  only Earth rotation. `strapdown::stationary` detects such periods and
  `strapdown::measurements` provides both models.

