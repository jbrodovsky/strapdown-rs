//! Coarse alignment and initialisation of the strapdown attitude solution.
//!
//! A strapdown mechanization integrates; it does not estimate. Everything it produces is
//! relative to the attitude it started from, so the initial direction cosine matrix is not a
//! convenience — a one-degree initial heading error rotates every subsequent specific-force
//! measurement by one degree, and the resulting horizontal acceleration error of
//! `g * sin(1 deg)` is 0.17 m/s^2, which is roughly two orders of magnitude larger than the
//! accelerometer bias of the consumer-grade hardware this crate is usually pointed at. Coarse
//! alignment is what supplies that starting attitude before any aiding measurement arrives.
//!
//! This module implements the three classical self-alignment estimators of Groves, 2nd ed.,
//! Section 5.6.3, plus the two small helpers needed to use them:
//!
//! | Function | Estimates | Needs | Practical limit |
//! |----------|-----------|-------|-----------------|
//! | [`coarse_leveling`] | roll, pitch | a stationary accelerometer triad | none worth worrying about; gravity is 9.8 m/s^2 |
//! | [`gyrocompassing`] | heading | a stationary gyro triad and latitude | tactical grade or better, away from the poles |
//! | [`heading_from_velocity`] | heading | horizontal velocity above a speed threshold | needs the platform to be moving, and assumes it moves the way it points |
//!
//! [`attitude_from_level_and_heading`] assembles the results into a `Rotation3`, and
//! [`average_imu`] reduces a stationary window of samples to the single averaged sample all
//! three estimators want.
//!
//! # Frame convention
//!
//! **Every function in this module is NED.** It consumes NED-convention body-frame
//! measurements and produces an NED attitude: roll about the forward axis, pitch about the
//! right axis, heading clockwise from true north. Nothing here inspects an `is_enu` flag or
//! guesses.
//!
//! The one place this bites is the accelerometer sign. Under the specific-force convention
//! that [`mechanize`](crate::mechanize) actually implements (Groves eq. 5.54, where the
//! gravity term is down-positive and the sensed increment is added to it), a *level*
//! stationary IMU reads
//!
//! ```text
//! accel_body = [0, 0, -g]      // NED
//! ```
//!
//! not `[0, 0, +g]`. The sensed specific force is the reaction to gravity, so it points *up*,
//! and the NED third axis points *down*. Feeding this module an ENU-convention specific force
//! silently produces a solution flipped through the horizontal, so reflect the third component
//! first. To go the other way — an ENU state out of an NED alignment — build the
//! [`StrapdownState`](crate::StrapdownState) with `is_enu: Some(false)` and call
//! [`to_enu`](crate::StrapdownState::to_enu), which handles the paired body/nav reflection
//! correctly.
//!
//! # Relationship to [`stationary`](crate::stationary)
//!
//! Deliberately orthogonal: these are pure functions over one averaged sample, and they do not
//! own a detector. [`StationaryDetector`](crate::stationary::StationaryDetector) already
//! answers "is the platform still?" over a sliding window, and that is exactly the question
//! whose answer selects the samples to hand to [`average_imu`]. Coupling the two would force
//! this module to carry a window and a set of thresholds it has no independent opinion about,
//! and would make alignment untestable without also driving a detector. The intended
//! composition is shown in the example below; the division of labour is that the detector
//! decides *which* samples are usable and this module decides *what they mean*.
//!
//! One consequence worth stating: the estimators here perform no plausibility check on the
//! total angular-rate magnitude. A platform in a steady turn produces a perfectly
//! self-consistent — and completely wrong — gyrocompass heading. Rejecting that is condition 4
//! of [`StationaryConfig`](crate::stationary::StationaryConfig), and it is the caller's job to
//! have run it.
//!
//! # Example
//!
//! Aligning from a stationary window, with the detector gating the samples:
//!
//! ```rust
//! use nalgebra::{Rotation3, Vector3};
//! use strapdown::alignment::{
//!     attitude_from_level_and_heading, average_imu, coarse_leveling, gyrocompassing,
//!     GyrocompassConfig,
//! };
//! use strapdown::stationary::{StationaryConfig, StationaryDetector};
//! use strapdown::{earth, IMUData, IMUQuality};
//!
//! // Ground truth for the example: a navigation-grade unit sitting at 40 deg N, tilted.
//! let latitude_degrees = 40.0;
//! let truth = Rotation3::from_euler_angles(0.05, -0.03, 1.2);
//! let gravity = earth::gravity(&latitude_degrees, &0.0);
//! let at_rest = IMUData {
//!     // Specific force is the reaction to gravity: up, hence -g on the NED down axis.
//!     accel: truth.inverse() * Vector3::new(0.0, 0.0, -gravity),
//!     gyro: truth.inverse() * earth::earth_rate_lla(&latitude_degrees),
//! };
//!
//! // Collect only the samples the detector vouches for.
//! let mut detector = StationaryDetector::new(StationaryConfig::default());
//! let mut window = Vec::new();
//! for _ in 0..200 {
//!     if detector.push(&at_rest) {
//!         window.push(at_rest);
//!     }
//! }
//! let averaged = average_imu(&window).unwrap();
//!
//! let level = coarse_leveling(&averaged.accel).unwrap();
//! let heading = gyrocompassing(
//!     &averaged,
//!     latitude_degrees,
//!     IMUQuality::Navigation,
//!     &GyrocompassConfig::default(),
//! )
//! .unwrap();
//!
//! let attitude = attitude_from_level_and_heading(level, heading.heading_radians);
//! let (roll, pitch, yaw) = attitude.euler_angles();
//! assert!((roll - 0.05).abs() < 1e-6);
//! assert!((pitch + 0.03).abs() < 1e-6);
//! assert!((yaw - 1.2).abs() < 1e-6);
//! ```
//!
//! # References
//!
//! - Groves, P. D., *Principles of GNSS, Inertial, and Multisensor Integrated Navigation
//!   Systems*, 2nd ed., Section 5.6.3 (coarse alignment: levelling and gyrocompassing) and
//!   Section 5.4 (the mechanization whose sign conventions these must match).
//! - Groves, Section 5.6.3 also gives the gyrocompassing error budget reproduced in
//!   [`HeadingEstimate::uncertainty_radians`].

use nalgebra::{Rotation3, Vector3};
use serde::{Deserialize, Serialize};

use crate::error::StrapdownError;
use crate::{IMUData, IMUQuality, earth};

/// Smallest specific-force magnitude [`coarse_leveling`] will accept, m/s^2.
///
/// Levelling divides by the sensed magnitude, so a platform in free fall — or an
/// accelerometer that has failed to a constant — has no vertical reference at all and the
/// answer is the `atan2` of noise. This bound is deliberately far below local gravity
/// (~9.8 m/s^2) rather than close to it: rejecting a *plausible but wrong* magnitude is
/// condition 2 of [`StationaryConfig`](crate::stationary::StationaryConfig), and duplicating
/// it here with a second, differently-tuned threshold would just give the caller two places to
/// disagree with itself. This one only catches the case where levelling is meaningless.
pub const MINIMUM_SPECIFIC_FORCE_MPS2: f64 = 1.0;

/// Default latitude bound for [`gyrocompassing`], degrees.
///
/// The observable horizontal Earth rate is `RATE * cos(latitude)`, so the heading error for a
/// given gyro error grows as `sec(latitude)`: it is 1.9x worse at 58 deg than at the equator,
/// 3.9x worse at 75 deg, and unbounded at the pole, where the Earth-rate vector is vertical and
/// heading is simply not observable. 75 deg is the conventional cut-off and costs a factor of
/// four over the equatorial case.
pub const DEFAULT_MAXIMUM_LATITUDE_DEGREES: f64 = 75.0;

/// Default fractional tolerance on the observed horizontal Earth rate.
///
/// The measured horizontal rate magnitude must agree with the predicted `RATE * cos(latitude)`
/// to within this fraction, or the sample is not Earth rotation and the heading derived from it
/// is meaningless. A tactical-grade bias instability of 1 deg/h is 6.7% of Earth rate, so 25%
/// leaves room for bias plus residual noise while still rejecting the cases that matter:
/// a slow turn, a badly wrong latitude, or a gyro whose scale factor or axis order is off.
pub const DEFAULT_EARTH_RATE_TOLERANCE: f64 = 0.25;

/// Default speed below which [`heading_from_velocity`] refuses to answer, m/s.
///
/// Course over ground is `atan2(v_east, v_north)`; as the speed approaches the velocity noise
/// floor that expression becomes uniformly distributed over the circle while remaining
/// perfectly well-defined numerically. 1 m/s is a walking pace and comfortably above the
/// ~0.05 m/s noise of a standalone GNSS velocity solution.
pub const DEFAULT_MINIMUM_SPEED_MPS: f64 = 1.0;

/// Roll and pitch from [`coarse_leveling`], radians, NED.
///
/// Heading is absent by construction: gravity is invariant under rotation about the vertical,
/// so the accelerometer triad carries no heading information whatsoever. That is the entire
/// reason [`gyrocompassing`] and [`heading_from_velocity`] exist.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LevelAttitude {
    /// Rotation about the body forward (x) axis, radians. Positive is right-wing-down.
    pub roll_radians: f64,
    /// Rotation about the body right (y) axis, radians. Positive is nose-up. Confined to
    /// `[-pi/2, pi/2]`, which is where the Euler parameterisation is unique.
    pub pitch_radians: f64,
}

/// A heading estimate and the uncertainty the estimator itself claims for it.
///
/// The uncertainty is not optional decoration. Gyrocompassing is the one estimator in this
/// crate whose answer can be numerically fine and physically worthless, and the difference
/// between those two cases is visible only in this field.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadingEstimate {
    /// Heading clockwise from true north, radians, wrapped to `(-pi, pi]`. NED.
    pub heading_radians: f64,
    /// One-sigma heading uncertainty, radians.
    ///
    /// For [`gyrocompassing`] this is the Groves Section 5.6.3 error budget, whose dominant
    /// term is the residual gyro bias divided by the observable horizontal Earth rate:
    ///
    /// ```text
    /// sigma_heading = gyro_bias_instability / (RATE * cos(latitude))
    /// ```
    ///
    /// with both rates in the same units. Earth rate is 15.04 deg/h, so at mid-latitudes the
    /// denominator is about 11 deg/h and the numerator is the whole story. Evaluated from
    /// [`IMUQuality`] at 45 deg latitude:
    ///
    /// | Grade | Gyro bias instability | One-sigma heading |
    /// |-------|----------------------|-------------------|
    /// | Consumer | 100 deg/h | 540 deg — no information |
    /// | Industrial | 50 deg/h | 270 deg — no information |
    /// | Tactical | 1 deg/h | 5.4 deg |
    /// | Navigation | 0.01 deg/h | 0.054 deg |
    /// | Strategic | 0.0001 deg/h | 0.00054 deg |
    ///
    /// The first two rows are why [`GyrocompassConfig::maximum_heading_uncertainty_radians`]
    /// exists and defaults to 45 deg: a MEMS gyro whose bias is several times Earth rate cannot
    /// gyrocompass, and returning a number anyway would be the worst available behaviour.
    pub uncertainty_radians: f64,
}

/// Limits under which [`gyrocompassing`] is willing to answer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GyrocompassConfig {
    /// Refuse above this absolute latitude, degrees. See [`DEFAULT_MAXIMUM_LATITUDE_DEGREES`].
    pub maximum_latitude_degrees: f64,
    /// Fractional agreement required between observed and predicted horizontal Earth rate.
    /// See [`DEFAULT_EARTH_RATE_TOLERANCE`].
    pub earth_rate_tolerance: f64,
    /// Refuse when the predicted one-sigma heading error exceeds this, radians.
    ///
    /// Defaults to `pi/4`. This is the check that turns "consumer MEMS cannot gyrocompass"
    /// from a sentence in the documentation into a returned error.
    pub maximum_heading_uncertainty_radians: f64,
}

impl Default for GyrocompassConfig {
    fn default() -> Self {
        Self {
            maximum_latitude_degrees: DEFAULT_MAXIMUM_LATITUDE_DEGREES,
            earth_rate_tolerance: DEFAULT_EARTH_RATE_TOLERANCE,
            maximum_heading_uncertainty_radians: std::f64::consts::FRAC_PI_4,
        }
    }
}

impl GyrocompassConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] if any field is not finite and strictly
    /// positive, or if `maximum_latitude_degrees` reaches 90. At the pole `cos(latitude)` is
    /// zero and the uncertainty formula divides by it, so the bound has to exclude it rather
    /// than merely discourage it.
    pub fn validate(&self) -> Result<(), StrapdownError> {
        let positive = [
            self.maximum_latitude_degrees,
            self.earth_rate_tolerance,
            self.maximum_heading_uncertainty_radians,
        ];
        if !positive
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(StrapdownError::InvalidConfiguration {
                field: "gyrocompass config",
                reason: "every limit must be finite and strictly positive".to_owned(),
            });
        }
        if self.maximum_latitude_degrees >= 90.0 {
            return Err(StrapdownError::InvalidConfiguration {
                field: "gyrocompass maximum_latitude_degrees",
                reason: format!(
                    "must be strictly less than 90 degrees, got {}; \
                     heading is not observable at the pole",
                    self.maximum_latitude_degrees
                ),
            });
        }
        Ok(())
    }
}

/// Average a window of IMU samples into one.
///
/// Coarse alignment is a static estimator applied to a single measurement, and the way to make
/// that measurement useful is to average a stationary window: white sensor noise falls as
/// `1/sqrt(n)`, while gravity and Earth rate are constant and survive untouched. This is what
/// makes gyrocompassing possible at all — the per-sample angular random walk of even a
/// navigation-grade gyro is comparable to Earth rate itself, so minutes of averaging are
/// required, not milliseconds.
///
/// Averaging does nothing for bias instability, which is common to every sample in the window.
/// That residual is exactly what [`HeadingEstimate::uncertainty_radians`] reports.
///
/// # Errors
/// * [`StrapdownError::DimensionMismatch`] if `samples` is empty; the mean is undefined.
/// * [`StrapdownError::NonFinite`] if any component of any sample is `NaN` or infinite. Checked
///   per sample rather than only on the result, because a single infinity poisons the mean into
///   `NaN` and a matched pair of infinities would poison it into `NaN` while looking like
///   arithmetic.
///
/// # Example
/// ```rust
/// use nalgebra::Vector3;
/// use strapdown::alignment::average_imu;
/// use strapdown::IMUData;
///
/// let samples = [
///     IMUData { accel: Vector3::new(0.0, 0.0, -9.0), gyro: Vector3::zeros() },
///     IMUData { accel: Vector3::new(0.0, 0.0, -11.0), gyro: Vector3::zeros() },
/// ];
/// let mean = average_imu(&samples).unwrap();
/// assert!((mean.accel[2] + 10.0).abs() < 1e-12);
/// ```
pub fn average_imu(samples: &[IMUData]) -> Result<IMUData, StrapdownError> {
    if samples.is_empty() {
        return Err(StrapdownError::DimensionMismatch {
            what: "alignment IMU window",
            expected: 1,
            got: 0,
        });
    }
    let mut accel = Vector3::zeros();
    let mut gyro = Vector3::zeros();
    let mut count = 0.0_f64;
    for sample in samples {
        if !sample.accel.iter().all(|value| value.is_finite()) {
            return Err(StrapdownError::NonFinite {
                what: "alignment specific force sample",
            });
        }
        if !sample.gyro.iter().all(|value| value.is_finite()) {
            return Err(StrapdownError::NonFinite {
                what: "alignment angular rate sample",
            });
        }
        accel += sample.accel;
        gyro += sample.gyro;
        count += 1.0;
    }
    Ok(IMUData {
        accel: accel / count,
        gyro: gyro / count,
    })
}

/// Estimate roll and pitch from the specific force sensed by a stationary IMU.
///
/// Groves, Section 5.6.3. A stationary accelerometer triad senses the reaction to gravity,
/// which in the NED navigation frame is `f_nav = [0, 0, -g]` — up, because the third axis
/// points down. Resolving that into the body frame through the attitude matrix
/// `C_b_n = R_z(heading) * R_y(pitch) * R_x(roll)` gives, with `f = accel`:
///
/// ```text
/// f_x =  g * sin(pitch)
/// f_y = -g * sin(roll) * cos(pitch)
/// f_z = -g * cos(roll) * cos(pitch)
/// ```
///
/// which inverts without knowing `g`, and without knowing the heading, to
///
/// ```text
/// pitch = atan2( f_x, hypot(f_y, f_z) )
/// roll  = atan2(-f_y, -f_z)
/// ```
///
/// Using `hypot(f_y, f_z)` for pitch rather than `asin(f_x / g)` avoids needing the local
/// gravity magnitude, is numerically better conditioned near the horizon, and degrades
/// gracefully when the accelerometers carry a common scale-factor error.
///
/// Heading is not recoverable here at any quality of hardware: rotating the body about the
/// navigation vertical leaves `f_nav` unchanged, so the measurement is literally independent of
/// it. The returned pitch lies in `[-pi/2, pi/2]` and the roll in `(-pi, pi]`, which is the
/// unique branch of the Euler parameterisation away from the pitch singularity.
///
/// # Arguments
/// * `specific_force_body` — sensed specific force in the body frame, m/s^2, **NED sign
///   convention**: a level stationary IMU reads `[0, 0, -g]`, not `[0, 0, +g]`. See the
///   [module documentation](self#frame-convention).
///
/// # Errors
/// * [`StrapdownError::NonFinite`] if any component is `NaN` or infinite.
/// * [`StrapdownError::MeasurementUnavailable`] if the magnitude is below
///   [`MINIMUM_SPECIFIC_FORCE_MPS2`], meaning there is no usable vertical reference. This
///   variant is [recoverable](StrapdownError::is_recoverable): the caller should wait for a
///   better window rather than abort.
///
/// # Example
/// ```rust
/// use nalgebra::Vector3;
/// use strapdown::alignment::coarse_leveling;
///
/// // Nose up 30 degrees, wings level. The forward axis picks up g * sin(30) = 4.9 m/s^2.
/// let level = coarse_leveling(&Vector3::new(4.905, 0.0, -8.496)).unwrap();
/// assert!((level.pitch_radians.to_degrees() - 30.0).abs() < 0.01);
/// assert!(level.roll_radians.abs() < 0.01);
/// ```
pub fn coarse_leveling(
    specific_force_body: &Vector3<f64>,
) -> Result<LevelAttitude, StrapdownError> {
    if !specific_force_body.iter().all(|value| value.is_finite()) {
        return Err(StrapdownError::NonFinite {
            what: "specific force (body frame)",
        });
    }
    let magnitude = specific_force_body.norm();
    if magnitude < MINIMUM_SPECIFIC_FORCE_MPS2 {
        return Err(StrapdownError::MeasurementUnavailable {
            model: "coarse_leveling",
            reason: format!(
                "specific force magnitude {magnitude:.4} m/s^2 is below \
                 {MINIMUM_SPECIFIC_FORCE_MPS2} m/s^2; there is no vertical reference to level \
                 against (free fall, or a failed accelerometer triad)"
            ),
        });
    }
    let forward = specific_force_body[0];
    let right = specific_force_body[1];
    let down = specific_force_body[2];
    Ok(LevelAttitude {
        roll_radians: (-right).atan2(-down),
        pitch_radians: forward.atan2(right.hypot(down)),
    })
}

/// Estimate heading from the Earth rotation rate sensed by a stationary gyro triad.
///
/// Groves, Section 5.6.3. After levelling, rotate the sensed angular rate into the levelled
/// (heading-free) frame with `C_b_l = R_y(pitch) * R_x(roll)`. A stationary platform senses
/// only Earth rotation, whose NED components are
/// `omega_nav = [RATE * cos(latitude), 0, -RATE * sin(latitude)]` — note the zero east
/// component, because Earth rotation has none. Heading is the rotation about the vertical that
/// takes the levelled vector onto that, so with `w = C_b_l * gyro`:
///
/// ```text
/// heading = atan2(-w_y, w_x)
/// ```
///
/// and the horizontal magnitude `hypot(w_x, w_y)` must come out as `RATE * cos(latitude)`,
/// which is what [`GyrocompassConfig::earth_rate_tolerance`] checks.
///
/// # This does not work on most hardware
///
/// The entire signal is `RATE * cos(latitude)`: 15.04 deg/h at the equator, 10.6 deg/h at
/// 45 deg, zero at the pole. A consumer MEMS gyro's bias instability alone is ~100 deg/h — the
/// signal being measured is a *rounding error* on the instrument. Only tactical grade and
/// better can gyrocompass at all, and even a tactical unit needs minutes of averaging (see
/// [`average_imu`]) to bring the random walk down to the level of its bias. The per-grade
/// arithmetic is tabulated on [`HeadingEstimate::uncertainty_radians`].
///
/// This function therefore refuses rather than guesses, on three independent grounds: latitude,
/// the predicted uncertainty for the declared [`IMUQuality`], and whether the observed rate is
/// credible as Earth rotation at all. When it refuses, [`heading_from_velocity`] is the
/// practical fallback — which is why the refusals are all
/// [recoverable](StrapdownError::is_recoverable).
///
/// # Arguments
/// * `imu` — an averaged stationary sample. `accel` is used for levelling and carries the
///   **NED sign convention** described in the [module documentation](self#frame-convention);
///   `gyro` is the sensed angular rate in rad/s, body frame.
/// * `latitude_degrees` — WGS84 latitude, degrees. The estimator needs it to predict the
///   observable Earth rate; a wrong latitude shows up as a tolerance failure, not as a silently
///   biased heading.
/// * `quality` — the grade of the instrument, used only to predict the uncertainty and decide
///   whether to answer.
/// * `config` — the limits to enforce; [`GyrocompassConfig::default`] is the usual choice.
///
/// # Errors
/// * [`StrapdownError::InvalidConfiguration`] if `config` does not validate.
/// * [`StrapdownError::NonFinite`] if `latitude_degrees` or any angular-rate component is not
///   finite.
/// * [`StrapdownError::MeasurementUnavailable`], recoverable, when gyrocompassing is refused:
///   the latitude exceeds [`GyrocompassConfig::maximum_latitude_degrees`]; the predicted
///   one-sigma heading error for `quality` exceeds
///   [`GyrocompassConfig::maximum_heading_uncertainty_radians`]; or the observed horizontal
///   rate disagrees with `RATE * cos(latitude)` by more than
///   [`GyrocompassConfig::earth_rate_tolerance`].
/// * Anything propagated from [`coarse_leveling`], which runs first.
///
/// # Example
/// ```rust
/// use nalgebra::{Rotation3, Vector3};
/// use strapdown::alignment::{gyrocompassing, GyrocompassConfig};
/// use strapdown::{earth, IMUData, IMUQuality};
///
/// let latitude_degrees = 45.0;
/// let truth = Rotation3::from_euler_angles(0.0, 0.0, 0.7);
/// let at_rest = IMUData {
///     accel: truth.inverse() * Vector3::new(0.0, 0.0, -earth::gravity(&latitude_degrees, &0.0)),
///     gyro: truth.inverse() * earth::earth_rate_lla(&latitude_degrees),
/// };
///
/// let heading = gyrocompassing(
///     &at_rest,
///     latitude_degrees,
///     IMUQuality::Navigation,
///     &GyrocompassConfig::default(),
/// )
/// .unwrap();
/// assert!((heading.heading_radians - 0.7).abs() < 1e-9);
///
/// // The same measurement from a consumer MEMS unit is refused, not answered.
/// let refused = gyrocompassing(
///     &at_rest,
///     latitude_degrees,
///     IMUQuality::Consumer,
///     &GyrocompassConfig::default(),
/// );
/// assert!(refused.is_err());
/// ```
pub fn gyrocompassing(
    imu: &IMUData,
    latitude_degrees: f64,
    quality: IMUQuality,
    config: &GyrocompassConfig,
) -> Result<HeadingEstimate, StrapdownError> {
    config.validate()?;
    if !latitude_degrees.is_finite() {
        return Err(StrapdownError::NonFinite {
            what: "latitude (degrees)",
        });
    }
    if !imu.gyro.iter().all(|value| value.is_finite()) {
        return Err(StrapdownError::NonFinite {
            what: "angular rate (body frame)",
        });
    }
    if latitude_degrees.abs() > config.maximum_latitude_degrees {
        return Err(StrapdownError::MeasurementUnavailable {
            model: "gyrocompassing",
            reason: format!(
                "latitude {:.3} deg exceeds the {:.3} deg limit; the observable horizontal \
                 Earth rate is RATE * cos(latitude), which vanishes at the pole",
                latitude_degrees, config.maximum_latitude_degrees
            ),
        });
    }
    let cos_latitude = latitude_degrees.to_radians().cos();
    let uncertainty_radians = predicted_heading_uncertainty(quality, cos_latitude);
    if uncertainty_radians > config.maximum_heading_uncertainty_radians {
        return Err(StrapdownError::MeasurementUnavailable {
            model: "gyrocompassing",
            reason: format!(
                "a {quality:?}-grade gyro at latitude {latitude_degrees:.3} deg gives a \
                 predicted one-sigma heading error of {:.1} deg, above the {:.1} deg limit; \
                 its bias instability is larger than the Earth rate it would have to measure",
                uncertainty_radians.to_degrees(),
                config.maximum_heading_uncertainty_radians.to_degrees()
            ),
        });
    }
    let level = coarse_leveling(&imu.accel)?;
    // Undo roll and pitch only. What remains is the heading rotation, which is precisely what
    // is being solved for, so it must not be applied here.
    let body_to_level = Rotation3::from_euler_angles(level.roll_radians, level.pitch_radians, 0.0);
    let levelled_rate = body_to_level * imu.gyro;
    let observed_horizontal_rate = levelled_rate[0].hypot(levelled_rate[1]);
    let expected_horizontal_rate = earth::RATE * cos_latitude;
    let discrepancy = (observed_horizontal_rate - expected_horizontal_rate).abs();
    if discrepancy > config.earth_rate_tolerance * expected_horizontal_rate {
        return Err(StrapdownError::MeasurementUnavailable {
            model: "gyrocompassing",
            reason: format!(
                "observed horizontal angular rate {observed_horizontal_rate:.3e} rad/s differs \
                 from the Earth rate {expected_horizontal_rate:.3e} rad/s predicted at \
                 latitude {latitude_degrees:.3} deg by more than {:.0}%; the platform is not \
                 stationary, the latitude is wrong, or the gyro is not trustworthy",
                config.earth_rate_tolerance * 100.0
            ),
        });
    }
    Ok(HeadingEstimate {
        heading_radians: (-levelled_rate[1]).atan2(levelled_rate[0]),
        uncertainty_radians,
    })
}

/// The Groves Section 5.6.3 gyrocompassing error budget, reduced to its dominant term.
///
/// `sigma_heading = gyro_bias_instability / (RATE * cos(latitude))`, with both rates expressed
/// per hour so the ratio is dimensionless — that ratio *is* the heading error in radians.
/// `cos_latitude` is strictly positive because [`GyrocompassConfig::validate`] keeps the
/// latitude bound below 90 degrees.
fn predicted_heading_uncertainty(quality: IMUQuality, cos_latitude: f64) -> f64 {
    const SECONDS_PER_HOUR: f64 = 3600.0;
    let earth_rate_per_hour = earth::RATE * SECONDS_PER_HOUR;
    // Both rates are per *hour* here, so they divide directly: `earth::RATE` is scaled up to
    // the hour above, and the accessor is already radians per hour.
    quality.gyro_bias_instability_rad_per_hour() / (earth_rate_per_hour * cos_latitude)
}

/// Estimate heading from horizontal velocity: course over ground.
///
/// The practical fallback when [`gyrocompassing`] refuses, which on MEMS hardware is always.
/// It trades one impossible requirement for one merely inconvenient one: instead of an
/// instrument good enough to see Earth rotation, it needs the platform to be moving, and to be
/// moving in the direction it points.
///
/// ```text
/// heading = atan2(velocity_east, velocity_north)
/// ```
///
/// The second assumption is the one that bites. Course over ground equals heading only when
/// sideslip and crab are negligible — fine for a road vehicle, wrong for an aircraft in a
/// crosswind or a boat in a current, and meaningless for a platform that can translate without
/// rotating. This function cannot detect any of that; it is the caller's knowledge of the
/// vehicle that makes the substitution legitimate.
///
/// The first assumption is enforced. Below `minimum_speed_mps` the velocity vector is dominated
/// by estimation noise, and `atan2` of noise is a uniformly distributed angle that arrives with
/// no indication that anything is wrong, so this refuses instead.
///
/// # Frame
///
/// The navigation frame here is ordered (north, east, vertical), so the two horizontal
/// components are identical under NED and ENU — only the third differs. This function is
/// therefore safe to call with velocities from a state in either convention, and the heading it
/// returns is clockwise from true north either way.
///
/// # Arguments
/// * `velocity_north`, `velocity_east` — horizontal velocity in the local-level frame, m/s.
/// * `minimum_speed_mps` — refuse below this horizontal speed. [`DEFAULT_MINIMUM_SPEED_MPS`] is
///   a reasonable starting point for a GNSS velocity solution; raise it for a noisier source.
///
/// # Returns
/// Heading clockwise from true north, radians, wrapped to `(-pi, pi]`.
///
/// # Errors
/// * [`StrapdownError::NonFinite`] if either velocity component is `NaN` or infinite.
/// * [`StrapdownError::InvalidConfiguration`] if `minimum_speed_mps` is negative or not finite.
///   Zero is permitted but means the refusal is disabled, which puts the burden of judging the
///   answer entirely on the caller.
/// * [`StrapdownError::MeasurementUnavailable`], recoverable, if the horizontal speed is below
///   `minimum_speed_mps`.
///
/// # Example
/// ```rust
/// use strapdown::alignment::{heading_from_velocity, DEFAULT_MINIMUM_SPEED_MPS};
///
/// // Due north-east at 14 m/s.
/// let heading = heading_from_velocity(10.0, 10.0, DEFAULT_MINIMUM_SPEED_MPS).unwrap();
/// assert!((heading.to_degrees() - 45.0).abs() < 1e-9);
///
/// // Parked: refused rather than answered from noise.
/// assert!(heading_from_velocity(0.01, -0.02, DEFAULT_MINIMUM_SPEED_MPS).is_err());
/// ```
pub fn heading_from_velocity(
    velocity_north: f64,
    velocity_east: f64,
    minimum_speed_mps: f64,
) -> Result<f64, StrapdownError> {
    if !velocity_north.is_finite() || !velocity_east.is_finite() {
        return Err(StrapdownError::NonFinite {
            what: "horizontal velocity (local-level frame)",
        });
    }
    if !minimum_speed_mps.is_finite() || minimum_speed_mps < 0.0 {
        return Err(StrapdownError::InvalidConfiguration {
            field: "heading_from_velocity minimum_speed_mps",
            reason: format!("must be finite and non-negative, got {minimum_speed_mps}"),
        });
    }
    let speed = velocity_north.hypot(velocity_east);
    if speed < minimum_speed_mps {
        return Err(StrapdownError::MeasurementUnavailable {
            model: "heading_from_velocity",
            reason: format!(
                "horizontal speed {speed:.4} m/s is below the {minimum_speed_mps:.4} m/s \
                 threshold; course over ground would be the atan2 of velocity noise"
            ),
        });
    }
    Ok(velocity_east.atan2(velocity_north))
}

/// Assemble a body-to-navigation attitude matrix from a levelling and a heading solution.
///
/// Returns `C_b_n = R_z(heading) * R_y(pitch) * R_x(roll)` in the **NED** convention, ready to
/// hand to [`StrapdownState::new`](crate::StrapdownState::new) with `is_enu: Some(false)`. For
/// an ENU state, build the NED state first and call
/// [`to_enu`](crate::StrapdownState::to_enu), which applies the paired body-and-navigation
/// reflection that a bare sign flip would get wrong.
///
/// # Example
/// ```rust
/// use strapdown::alignment::{attitude_from_level_and_heading, LevelAttitude};
///
/// let level = LevelAttitude { roll_radians: 0.1, pitch_radians: -0.2 };
/// let attitude = attitude_from_level_and_heading(level, 1.5);
/// let (roll, pitch, yaw) = attitude.euler_angles();
/// assert!((roll - 0.1).abs() < 1e-12);
/// assert!((pitch + 0.2).abs() < 1e-12);
/// assert!((yaw - 1.5).abs() < 1e-12);
/// ```
pub fn attitude_from_level_and_heading(
    level: LevelAttitude,
    heading_radians: f64,
) -> Rotation3<f64> {
    Rotation3::from_euler_angles(level.roll_radians, level.pitch_radians, heading_radians)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImuSample, StrapdownState, mechanize};
    use assert_approx_eq::assert_approx_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    /// Latitude the synthetic cases sit at, degrees. Mid-latitude, so `cos(latitude)` is
    /// comfortably away from both the equatorial best case and the polar singularity.
    const TEST_LATITUDE_DEGREES: f64 = 40.0;
    /// Altitude the synthetic cases sit at, meters.
    const TEST_ALTITUDE_M: f64 = 0.0;

    /// Non-trivial orientations to exercise. Pitch stays well inside `(-90, 90)` degrees, where
    /// the Euler parameterisation is unique, and heading away from `+/-180` so the comparison
    /// does not have to wrap.
    const KNOWN_ORIENTATIONS_DEGREES: [(f64, f64, f64); 5] = [
        (0.0, 0.0, 0.0),
        (20.0, -15.0, 35.0),
        (-40.0, 30.0, -120.0),
        (5.0, 55.0, 170.0),
        (-175.0, -2.0, 90.0),
    ];

    /// Build the exact IMU output of a perfect, perfectly stationary NED IMU at a known
    /// orientation.
    ///
    /// This is the forward model the estimators invert: rotate the true navigation-frame
    /// gravity reaction and Earth-rate vectors into the body frame with the known attitude.
    /// Note the `-gravity`: see `stationary_specific_force_sign_matches_the_mechanization`,
    /// which pins that sign against [`mechanize`] rather than against this helper's opinion.
    fn stationary_imu(attitude: &Rotation3<f64>, latitude_degrees: f64) -> IMUData {
        let gravity = earth::gravity(&latitude_degrees, &TEST_ALTITUDE_M);
        let body_from_nav = attitude.inverse();
        IMUData {
            accel: body_from_nav * Vector3::new(0.0, 0.0, -gravity),
            gyro: body_from_nav * earth::earth_rate_lla(&latitude_degrees),
        }
    }

    fn attitude_from_degrees(roll: f64, pitch: f64, yaw: f64) -> Rotation3<f64> {
        Rotation3::from_euler_angles(roll.to_radians(), pitch.to_radians(), yaw.to_radians())
    }

    /// Add zero-mean white noise to a sample, as a real instrument would.
    fn add_noise(
        imu: &IMUData,
        accel_std: f64,
        gyro_std: f64,
        rng: &mut StdRng,
    ) -> Result<IMUData, StrapdownError> {
        let accel_noise =
            Normal::new(0.0, accel_std).map_err(|error| StrapdownError::InvalidConfiguration {
                field: "accel noise std",
                reason: error.to_string(),
            })?;
        let gyro_noise =
            Normal::new(0.0, gyro_std).map_err(|error| StrapdownError::InvalidConfiguration {
                field: "gyro noise std",
                reason: error.to_string(),
            })?;
        Ok(IMUData {
            accel: imu.accel + Vector3::from_fn(|_, _| accel_noise.sample(rng)),
            gyro: imu.gyro + Vector3::from_fn(|_, _| gyro_noise.sample(rng)),
        })
    }

    // --- Sign convention, pinned against the mechanization itself ---------------------------

    #[test]
    fn stationary_specific_force_sign_matches_the_mechanization() {
        // The whole module rests on "a stationary NED IMU reads [0, 0, -g]". Assert it against
        // `mechanize` rather than against a second copy of the same belief: a genuinely
        // stationary platform, propagated through the real mechanization with the specific
        // force this module's forward model produces, must not accelerate. Get the sign wrong
        // and the sensed force adds to gravity instead of cancelling it, which shows up here as
        // 2g of vertical velocity and nowhere else.
        for (roll, pitch, yaw) in KNOWN_ORIENTATIONS_DEGREES {
            let truth = attitude_from_degrees(roll, pitch, yaw);
            let imu = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
            let mut state = StrapdownState::new(
                TEST_LATITUDE_DEGREES,
                -75.0,
                TEST_ALTITUDE_M,
                0.0,
                0.0,
                0.0,
                truth,
                true,
                Some(false),
            )
            .unwrap();
            for _ in 0..100 {
                mechanize(&mut state, &ImuSample::from_rates(&imu, 0.01)).unwrap();
            }
            let speed = Vector3::new(
                state.velocity_north,
                state.velocity_east,
                state.velocity_vertical,
            )
            .norm();
            assert!(
                speed < 1e-9,
                "a stationary platform at ({roll}, {pitch}, {yaw}) deg accelerated to \
                 {speed} m/s in 1 s; the specific-force sign disagrees with `mechanize`"
            );
        }
    }

    #[test]
    fn the_opposite_specific_force_sign_does_not_stay_stationary() {
        // The negative control for the test above: if `[0, 0, +g]` also produced a stationary
        // solution, that test would prove nothing.
        let truth = Rotation3::identity();
        let gravity = earth::gravity(&TEST_LATITUDE_DEGREES, &TEST_ALTITUDE_M);
        let inverted = IMUData {
            accel: Vector3::new(0.0, 0.0, gravity),
            gyro: earth::earth_rate_lla(&TEST_LATITUDE_DEGREES),
        };
        let mut state = StrapdownState::new(
            TEST_LATITUDE_DEGREES,
            -75.0,
            TEST_ALTITUDE_M,
            0.0,
            0.0,
            0.0,
            truth,
            true,
            Some(false),
        )
        .unwrap();
        for _ in 0..100 {
            mechanize(&mut state, &ImuSample::from_rates(&inverted, 0.01)).unwrap();
        }
        // Two g for one second, because the sensed force now adds to gravity.
        assert_approx_eq!(state.velocity_vertical, 2.0 * gravity, 1e-3);
    }

    // --- coarse_leveling --------------------------------------------------------------------

    #[test]
    fn levels_a_perfectly_level_imu() {
        let imu = stationary_imu(&Rotation3::identity(), TEST_LATITUDE_DEGREES);
        let level = coarse_leveling(&imu.accel).unwrap();
        assert_approx_eq!(level.roll_radians, 0.0, 1e-12);
        assert_approx_eq!(level.pitch_radians, 0.0, 1e-12);
    }

    #[test]
    fn recovers_known_roll_and_pitch() {
        for (roll, pitch, yaw) in KNOWN_ORIENTATIONS_DEGREES {
            let truth = attitude_from_degrees(roll, pitch, yaw);
            let imu = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
            let level = coarse_leveling(&imu.accel).unwrap();
            assert_approx_eq!(level.roll_radians.to_degrees(), roll, 1e-9);
            assert_approx_eq!(level.pitch_radians.to_degrees(), pitch, 1e-9);
        }
    }

    #[test]
    fn leveling_is_independent_of_heading() {
        // Gravity is invariant under rotation about the navigation vertical, so every heading
        // at a fixed roll and pitch must give the same answer. This is the property that makes
        // `gyrocompassing` necessary rather than optional.
        let mut first: Option<LevelAttitude> = None;
        for heading in [-180.0, -90.0, 0.0, 45.0, 179.0] {
            let truth = attitude_from_degrees(12.0, -7.0, heading);
            let level =
                coarse_leveling(&stationary_imu(&truth, TEST_LATITUDE_DEGREES).accel).unwrap();
            match first {
                None => first = Some(level),
                Some(reference) => {
                    assert_approx_eq!(level.roll_radians, reference.roll_radians, 1e-12);
                    assert_approx_eq!(level.pitch_radians, reference.pitch_radians, 1e-12);
                }
            }
        }
    }

    #[test]
    fn leveling_survives_stationary_consumer_grade_noise() {
        // Consumer MEMS white noise, 0.05 m/s^2 per sample, averaged over a 10 s window at
        // 100 Hz. The 1/sqrt(n) reduction takes the residual to ~1.6e-3 m/s^2, i.e. ~0.01 deg
        // of tilt; assert an order of magnitude looser than that so the bound is about the
        // physics rather than about this particular seed.
        let mut rng = StdRng::seed_from_u64(20_260_914);
        let truth = attitude_from_degrees(8.0, -13.0, 62.0);
        let clean = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
        let window: Vec<IMUData> = (0..1000)
            .map(|_| add_noise(&clean, 0.05, 0.002, &mut rng).unwrap())
            .collect();
        let level = coarse_leveling(&average_imu(&window).unwrap().accel).unwrap();
        assert_approx_eq!(level.roll_radians.to_degrees(), 8.0, 0.05);
        assert_approx_eq!(level.pitch_radians.to_degrees(), -13.0, 0.05);
    }

    #[test]
    fn leveling_refuses_free_fall() {
        let error = coarse_leveling(&Vector3::new(0.01, -0.02, 0.005)).unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::MeasurementUnavailable { model, .. } if model == "coarse_leveling"
        ));
        // Recoverable: the caller should wait for a better window, not abort the run.
        assert!(error.is_recoverable());
    }

    #[test]
    fn leveling_refuses_non_finite_input() {
        assert!(matches!(
            coarse_leveling(&Vector3::new(0.0, f64::NAN, -9.8)).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
        assert!(matches!(
            coarse_leveling(&Vector3::new(f64::INFINITY, 0.0, -9.8)).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
    }

    // --- gyrocompassing ---------------------------------------------------------------------

    #[test]
    fn recovers_known_heading() {
        for (roll, pitch, yaw) in KNOWN_ORIENTATIONS_DEGREES {
            let truth = attitude_from_degrees(roll, pitch, yaw);
            let imu = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
            let heading = gyrocompassing(
                &imu,
                TEST_LATITUDE_DEGREES,
                IMUQuality::Navigation,
                &GyrocompassConfig::default(),
            )
            .unwrap();
            assert_approx_eq!(heading.heading_radians.to_degrees(), yaw, 1e-8);
        }
    }

    #[test]
    fn recovers_the_full_attitude_matrix() {
        // End to end: the two estimators together must reconstruct the DCM the measurements
        // were synthesized from, element by element.
        let truth = attitude_from_degrees(20.0, -15.0, 35.0);
        let imu = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
        let level = coarse_leveling(&imu.accel).unwrap();
        let heading = gyrocompassing(
            &imu,
            TEST_LATITUDE_DEGREES,
            IMUQuality::Navigation,
            &GyrocompassConfig::default(),
        )
        .unwrap();
        let recovered = attitude_from_level_and_heading(level, heading.heading_radians);
        for (got, expected) in recovered.matrix().iter().zip(truth.matrix().iter()) {
            assert_approx_eq!(*got, *expected, 1e-9);
        }
    }

    #[test]
    fn gyrocompassing_survives_stationary_navigation_grade_noise() {
        // Navigation-grade angle random walk at 100 Hz is ~1.5e-5 rad/s per sample, which is
        // itself a quarter of Earth rate -- the point of the exercise. Averaging 400 s of it
        // drops the residual to ~7e-8 rad/s, giving a predicted heading error of
        // 7e-8 / (RATE * cos(40 deg)) ~ 1.2e-3 rad ~ 0.07 deg. Assert 0.5 deg.
        let mut rng = StdRng::seed_from_u64(20_260_915);
        let truth = attitude_from_degrees(3.0, -4.0, 110.0);
        let clean = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
        let window: Vec<IMUData> = (0..40_000)
            .map(|_| add_noise(&clean, 8.3e-4, 1.45e-5, &mut rng).unwrap())
            .collect();
        let averaged = average_imu(&window).unwrap();
        let heading = gyrocompassing(
            &averaged,
            TEST_LATITUDE_DEGREES,
            IMUQuality::Navigation,
            &GyrocompassConfig::default(),
        )
        .unwrap();
        assert_approx_eq!(heading.heading_radians.to_degrees(), 110.0, 0.5);
        // The claimed uncertainty must bracket the error actually observed.
        assert!(
            heading.uncertainty_radians.to_degrees() < 0.5,
            "navigation grade should claim better than 0.5 deg, claimed {}",
            heading.uncertainty_radians.to_degrees()
        );
    }

    #[test]
    fn gyrocompassing_refuses_high_latitude() {
        let truth = attitude_from_degrees(0.0, 0.0, 30.0);
        let latitude_degrees = 82.0;
        let imu = stationary_imu(&truth, latitude_degrees);
        let error = gyrocompassing(
            &imu,
            latitude_degrees,
            IMUQuality::Strategic,
            &GyrocompassConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::MeasurementUnavailable { model, .. } if model == "gyrocompassing"
        ));
        assert!(error.is_recoverable());
        // The same measurement at a latitude inside the bound is accepted, so the refusal is
        // about the geometry and not about the measurement being malformed.
        assert!(
            gyrocompassing(
                &stationary_imu(&truth, 60.0),
                60.0,
                IMUQuality::Strategic,
                &GyrocompassConfig::default(),
            )
            .is_ok()
        );
    }

    #[test]
    fn gyrocompassing_refuses_grades_that_cannot_see_earth_rate() {
        let truth = attitude_from_degrees(0.0, 0.0, 30.0);
        let imu = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
        for quality in [IMUQuality::Consumer, IMUQuality::Industrial] {
            let error = gyrocompassing(
                &imu,
                TEST_LATITUDE_DEGREES,
                quality,
                &GyrocompassConfig::default(),
            )
            .unwrap_err();
            assert!(
                matches!(error, StrapdownError::MeasurementUnavailable { .. }),
                "{quality:?} should be refused, got {error:?}"
            );
        }
        // Tactical is the grade at which this becomes viable, per Groves 5.6.3.
        for quality in [
            IMUQuality::Tactical,
            IMUQuality::Navigation,
            IMUQuality::Strategic,
        ] {
            assert!(
                gyrocompassing(
                    &imu,
                    TEST_LATITUDE_DEGREES,
                    quality,
                    &GyrocompassConfig::default(),
                )
                .is_ok(),
                "{quality:?} should be able to gyrocompass"
            );
        }
    }

    #[test]
    fn predicted_uncertainty_degrades_with_latitude_and_grade() {
        let equator = predicted_heading_uncertainty(IMUQuality::Tactical, 0.0_f64.cos());
        let high = predicted_heading_uncertainty(IMUQuality::Tactical, 75.0_f64.to_radians().cos());
        assert!(high > equator, "sec(latitude) growth is missing");
        // 1 deg/h against a 15.04 deg/h Earth rate is 0.0665 rad at the equator.
        assert_approx_eq!(
            equator,
            1.0_f64.to_radians() / (earth::RATE * 3600.0),
            1e-12
        );
        assert!(
            predicted_heading_uncertainty(IMUQuality::Navigation, 1.0)
                < predicted_heading_uncertainty(IMUQuality::Tactical, 1.0)
        );
    }

    #[test]
    fn gyrocompassing_refuses_an_implausible_rate() {
        // A slow rotation about a horizontal body axis, two Earth rates in size. The levelling
        // is unaffected, the heading arithmetic is perfectly well-defined, and the answer would
        // be silently wrong -- so it has to be the rate magnitude that catches this.
        let truth = attitude_from_degrees(0.0, 0.0, 30.0);
        let mut imu = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
        imu.gyro[1] += 1.0e-4;
        let error = gyrocompassing(
            &imu,
            TEST_LATITUDE_DEGREES,
            IMUQuality::Navigation,
            &GyrocompassConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::MeasurementUnavailable { model, .. } if model == "gyrocompassing"
        ));
    }

    #[test]
    fn gyrocompassing_refuses_a_wrong_latitude() {
        // Measurements taken at 10 deg, told they were taken at 65 deg: the predicted Earth
        // rate no longer matches what the gyros saw.
        let truth = attitude_from_degrees(0.0, 0.0, 30.0);
        let imu = stationary_imu(&truth, 10.0);
        assert!(
            gyrocompassing(
                &imu,
                65.0,
                IMUQuality::Navigation,
                &GyrocompassConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn gyrocompassing_refuses_non_finite_inputs() {
        let truth = attitude_from_degrees(0.0, 0.0, 0.0);
        let imu = stationary_imu(&truth, TEST_LATITUDE_DEGREES);
        assert!(matches!(
            gyrocompassing(
                &imu,
                f64::NAN,
                IMUQuality::Navigation,
                &GyrocompassConfig::default()
            )
            .unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
        let mut bad = imu;
        bad.gyro[2] = f64::INFINITY;
        assert!(matches!(
            gyrocompassing(
                &bad,
                TEST_LATITUDE_DEGREES,
                IMUQuality::Navigation,
                &GyrocompassConfig::default()
            )
            .unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
    }

    #[test]
    fn gyrocompass_config_rejects_unusable_limits() {
        assert!(GyrocompassConfig::default().validate().is_ok());
        for bad in [
            GyrocompassConfig {
                maximum_latitude_degrees: 90.0,
                ..GyrocompassConfig::default()
            },
            GyrocompassConfig {
                maximum_latitude_degrees: 0.0,
                ..GyrocompassConfig::default()
            },
            GyrocompassConfig {
                earth_rate_tolerance: -0.1,
                ..GyrocompassConfig::default()
            },
            GyrocompassConfig {
                maximum_heading_uncertainty_radians: f64::NAN,
                ..GyrocompassConfig::default()
            },
        ] {
            assert!(matches!(
                bad.validate().unwrap_err(),
                StrapdownError::InvalidConfiguration { .. }
            ));
        }
    }

    // --- heading_from_velocity --------------------------------------------------------------

    #[test]
    fn heading_from_velocity_matches_known_courses() {
        let cases = [
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 90.0),
            (-10.0, 0.0, 180.0),
            (0.0, -10.0, -90.0),
            (10.0, 10.0, 45.0),
            (-5.0, 5.0, 135.0),
        ];
        for (north, east, expected_degrees) in cases {
            let heading = heading_from_velocity(north, east, DEFAULT_MINIMUM_SPEED_MPS).unwrap();
            assert_approx_eq!(heading.to_degrees(), expected_degrees, 1e-9);
        }
    }

    #[test]
    fn heading_from_velocity_refuses_below_the_speed_threshold() {
        let error = heading_from_velocity(0.3, -0.4, DEFAULT_MINIMUM_SPEED_MPS).unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::MeasurementUnavailable { model, .. }
                if model == "heading_from_velocity"
        ));
        assert!(error.is_recoverable());
        // Exactly at the threshold is accepted: 0.6/0.8 is a 3-4-5 triangle of speed 1.0.
        assert!(heading_from_velocity(0.6, 0.8, DEFAULT_MINIMUM_SPEED_MPS).is_ok());
    }

    #[test]
    fn heading_from_velocity_rejects_bad_arguments() {
        assert!(matches!(
            heading_from_velocity(f64::NAN, 1.0, DEFAULT_MINIMUM_SPEED_MPS).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
        assert!(matches!(
            heading_from_velocity(10.0, 10.0, -1.0).unwrap_err(),
            StrapdownError::InvalidConfiguration { .. }
        ));
    }

    #[test]
    fn heading_from_velocity_survives_gnss_velocity_noise() {
        // A 20 m/s course with a standalone-GNSS-grade 0.05 m/s velocity noise. The angular
        // error is ~noise/speed = 2.5e-3 rad = 0.14 deg.
        let mut rng = StdRng::seed_from_u64(20_260_916);
        let noise = Normal::new(0.0, 0.05).unwrap();
        let truth_degrees = 33.0_f64;
        let speed = 20.0_f64;
        let north = speed * truth_degrees.to_radians().cos();
        let east = speed * truth_degrees.to_radians().sin();
        for _ in 0..200 {
            let heading = heading_from_velocity(
                north + noise.sample(&mut rng),
                east + noise.sample(&mut rng),
                DEFAULT_MINIMUM_SPEED_MPS,
            )
            .unwrap();
            assert_approx_eq!(heading.to_degrees(), truth_degrees, 1.0);
        }
    }

    // --- helpers ----------------------------------------------------------------------------

    #[test]
    fn attitude_round_trips_through_euler_angles() {
        for (roll, pitch, yaw) in KNOWN_ORIENTATIONS_DEGREES {
            let level = LevelAttitude {
                roll_radians: roll.to_radians(),
                pitch_radians: pitch.to_radians(),
            };
            let attitude = attitude_from_level_and_heading(level, yaw.to_radians());
            let (got_roll, got_pitch, got_yaw) = attitude.euler_angles();
            assert_approx_eq!(got_roll.to_degrees(), roll, 1e-9);
            assert_approx_eq!(got_pitch.to_degrees(), pitch, 1e-9);
            assert_approx_eq!(got_yaw.to_degrees(), yaw, 1e-9);
        }
    }

    #[test]
    fn average_imu_returns_the_mean() {
        let samples = [
            IMUData {
                accel: Vector3::new(1.0, 2.0, -9.0),
                gyro: Vector3::new(0.1, 0.2, 0.3),
            },
            IMUData {
                accel: Vector3::new(3.0, 4.0, -11.0),
                gyro: Vector3::new(0.3, 0.4, 0.5),
            },
        ];
        let mean = average_imu(&samples).unwrap();
        assert_approx_eq!(mean.accel[0], 2.0, 1e-12);
        assert_approx_eq!(mean.accel[1], 3.0, 1e-12);
        assert_approx_eq!(mean.accel[2], -10.0, 1e-12);
        assert_approx_eq!(mean.gyro[0], 0.2, 1e-12);
        assert_approx_eq!(mean.gyro[2], 0.4, 1e-12);
    }

    #[test]
    fn average_imu_rejects_an_empty_window() {
        assert!(matches!(
            average_imu(&[]).unwrap_err(),
            StrapdownError::DimensionMismatch {
                expected: 1,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn average_imu_rejects_non_finite_samples() {
        let samples = [IMUData {
            accel: Vector3::new(0.0, 0.0, f64::NAN),
            gyro: Vector3::zeros(),
        }];
        assert!(matches!(
            average_imu(&samples).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
        let samples = [IMUData {
            accel: Vector3::new(0.0, 0.0, -9.8),
            gyro: Vector3::new(0.0, f64::INFINITY, 0.0),
        }];
        assert!(matches!(
            average_imu(&samples).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
    }
}
