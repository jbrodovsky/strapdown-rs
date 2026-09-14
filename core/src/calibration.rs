//! Deterministic IMU calibration: systematic error removal ahead of the mechanization.
//!
//! An inertial sensor triad does not report the quantity the strapdown equations want. Its
//! output carries a turn-on bias, a scale factor error on each axis, and cross-coupling
//! between axes caused by the sensitive axes not being exactly orthogonal or exactly aligned
//! with the body frame. Groves gives the model for both sensor types in Section 4.4.1
//! (Eq. 4.17 for accelerometers, Eq. 4.18 for gyroscopes):
//!
//! $$
//! \tilde{f}^b_{ib} = b_a + (I + M_a) f^b_{ib} + w_a
//! $$
//!
//! where \\(\tilde{f}\\) is what the sensor reports, \\(f\\) is the true specific force,
//! \\(`b_a`\\) is the bias, and \\(`M_a`\\) is the scale factor and cross-coupling matrix whose
//! diagonal holds the per-axis scale factor errors and whose off-diagonal elements hold the
//! misalignment coefficients. The gyroscope model is identical in form, with angular rate in
//! place of specific force. The random noise \\(w\\) is not something a deterministic
//! calibration can remove; it is what the filter's process noise is for.
//!
//! # \\(M\\) is stored as two fields, not one
//!
//! Groves writes \\(M\\) as a single 3x3. [`SensorErrorModel`] splits it: the diagonal goes
//! in `scale_factor`, the off-diagonal in `misalignment`, and `misalignment`'s own diagonal
//! must be zero. The split keeps every number in exactly one place -- a scale factor error
//! written into both fields would leave the two silently disagreeing -- and it matches how
//! the two quantities are measured and quoted separately.
//!
//! So if what you have is a full \\(M\\), from a lab report or a datasheet, do **not** pass
//! it as `misalignment`; that is rejected at construction. Use
//! [`SensorErrorModel::from_error_matrix`] or [`SensorCalibration::from_error_matrix`],
//! which do the split for you, and [`SensorErrorModel::error_matrix`] to read the assembled
//! \\(M\\) back out.
//!
//! # What is stored, and why
//!
//! This module stores the **forward error model** -- \\(b\\) and \\(M\\) exactly as Groves
//! writes them -- and inverts it once, at construction, to obtain the correction actually
//! applied. The alternative, storing the already-inverted correction matrix, was rejected
//! because the forward parameters are what a calibration procedure produces and what a
//! datasheet quotes: a turn-on bias in m/s², a scale factor error in parts per million, a
//! misalignment in milliradians. A configuration file full of the elements of an inverted
//! matrix is not reviewable by the engineer who measured them.
//!
//! Inverting the model means
//!
//! $$
//! \hat{f}^b_{ib} = (I + M_a)^{-1} \left( \tilde{f}^b_{ib} - b_a \right)
//! $$
//!
//! # Increments, not rates
//!
//! [`ImuSample`] carries integrated increments -- delta-v and delta-theta over `dt` -- rather
//! than instantaneous rates. The correction is applied in that domain directly. Scale factor
//! and cross-coupling are linear operators, so they commute with the integration and apply
//! unchanged; bias does not, because a constant bias integrated over `dt` contributes
//! \\(b \\, \\Delta t\\) to the increment:
//!
//! $$
//! \Delta\hat{v} = (I + M_a)^{-1} \left( \Delta\tilde{v} - b_a \, \Delta t \right)
//! $$
//!
//! Applying the bias without the `dt` factor is the obvious way to get this wrong, and is
//! what the `bias_scales_with_the_sample_interval` unit test exists to catch.
//!
//! # Fallibility
//!
//! [`ImuCalibration::correct`] is infallible. The only operation that can fail is inverting
//! \\(I + M\\), and that is done once in [`SensorCalibration::new`], which returns a
//! `Result`. The fields of a [`SensorCalibration`] are private so the validated invariant --
//! "the stored correction matrix is finite and is the true inverse of the stored forward
//! model" -- cannot be broken after the fact. Deserialization goes through the same
//! constructor via `#[serde(try_from = ...)]`, so a scenario file cannot smuggle in a
//! singular calibration either.
//!
//! This is the right trade for a function called once per IMU sample at hundreds of hertz:
//! the inversion happens once at startup rather than per sample, and the caller in the
//! mechanization loop has no error branch to handle on a failure that, if it were going to
//! happen at all, would have happened at load time.
//!
//! # Example
//!
//! ```rust
//! use strapdown::calibration::{ImuCalibration, SensorCalibration};
//! use strapdown::{IMUData, ImuSample};
//! use nalgebra::Vector3;
//!
//! # fn main() -> Result<(), strapdown::StrapdownError> {
//! // A 0.05 m/s^2 accelerometer bias on the x axis and a 1% scale factor error on y.
//! let accelerometer = SensorCalibration::new(
//!     [0.05, 0.0, 0.0],
//!     [0.0, 0.01, 0.0],
//!     [[0.0; 3]; 3],
//! )?;
//! let calibration = ImuCalibration::new(accelerometer, SensorCalibration::identity());
//!
//! let raw = ImuSample::from_rates(
//!     &IMUData { accel: Vector3::new(0.05, 1.01, 0.0), gyro: Vector3::zeros() },
//!     0.01,
//! );
//! let corrected = calibration.correct(&raw);
//!
//! // The bias and the scale factor error are both removed.
//! assert!((corrected.delta_v[0] - 0.0).abs() < 1e-12);
//! assert!((corrected.delta_v[1] - 0.01).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

use crate::error::StrapdownError;
use crate::{IMUData, ImuSample};

/// Smallest determinant of \\(I + M\\) accepted as invertible.
///
/// A physically plausible scale factor and cross-coupling matrix is a small perturbation of
/// zero -- parts per million to a few percent -- so \\(\\det(I + M)\\) sits very close to
/// one. A determinant near zero does not describe a sensor; it describes a typo, and
/// inverting it would produce an enormous correction matrix that silently destroys the
/// navigation solution rather than failing. The threshold is deliberately far below any
/// realistic value so that it rejects only the genuinely degenerate case.
const MINIMUM_CORRECTION_DETERMINANT: f64 = 1e-9;

/// The forward error-model parameters for one three-axis inertial sensor triad.
///
/// This is the plain-data, serializable form: the numbers a calibration procedure reports,
/// with no derived quantities. It is the wire format of [`SensorCalibration`], and is also
/// what [`SensorCalibration::parameters`] hands back.
///
/// All three fields default to zero, which is the identity calibration -- no bias, no scale
/// factor error, no misalignment.
///
/// # Units
/// * `bias` -- m/s² for an accelerometer, rad/s for a gyroscope. Body frame.
/// * `scale_factor` -- dimensionless fractional error per axis. `0.01` means the sensor
///   reads 1% high. This is the diagonal of Groves' \\(M\\) (Section 4.4.1).
/// * `misalignment` -- dimensionless cross-coupling coefficients, the off-diagonal elements
///   of Groves' \\(M\\). Element `[row][column]` is the contribution of the true `column`
///   axis to the measured `row` axis. For small angles these are the misalignment angles in
///   radians. The diagonal must be zero: per-axis scale factor errors belong in
///   `scale_factor`, and accepting them in both places would leave the two silently
///   disagreeing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SensorErrorModel {
    /// Additive bias, body frame. m/s² for accelerometers, rad/s for gyroscopes.
    pub bias: [f64; 3],
    /// Fractional scale factor error per axis; the diagonal of Groves' \\(M\\).
    pub scale_factor: [f64; 3],
    /// Cross-coupling / misalignment coefficients; the off-diagonal of Groves' \\(M\\).
    /// The diagonal must be zero.
    pub misalignment: [[f64; 3]; 3],
}

impl SensorErrorModel {
    /// Build the parameters from a full Groves \\(M\\), splitting it across the two fields.
    ///
    /// This is the constructor to reach for when a calibration report gives \\(M\\) as one
    /// 3x3 matrix, which is how Groves Section 4.4.1 writes it. The diagonal becomes
    /// `scale_factor` and the off-diagonal becomes `misalignment`; passing the same matrix
    /// directly as `misalignment` would instead be rejected, since that field's diagonal
    /// must be zero.
    ///
    /// Element `[row][column]` of `error_matrix` is the contribution of the true `column`
    /// axis to the measured `row` axis, so the diagonal is the per-axis scale factor error.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::calibration::SensorErrorModel;
    ///
    /// // A full M: 1% scale factor error on x, 2 mrad of x-into-y cross-coupling.
    /// let m = [
    ///     [0.01, 0.0, 0.0],
    ///     [0.002, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0],
    /// ];
    /// let parameters = SensorErrorModel::from_error_matrix([0.0; 3], m);
    ///
    /// assert_eq!(parameters.scale_factor, [0.01, 0.0, 0.0]);
    /// assert_eq!(parameters.misalignment[1][0], 0.002);
    /// assert_eq!(parameters.misalignment[0][0], 0.0);
    /// // Round-trips back to the matrix it came from.
    /// assert_eq!(parameters.error_matrix(), m);
    /// ```
    #[must_use]
    pub const fn from_error_matrix(bias: [f64; 3], error_matrix: [[f64; 3]; 3]) -> Self {
        Self {
            bias,
            scale_factor: [error_matrix[0][0], error_matrix[1][1], error_matrix[2][2]],
            misalignment: [
                [0.0, error_matrix[0][1], error_matrix[0][2]],
                [error_matrix[1][0], 0.0, error_matrix[1][2]],
                [error_matrix[2][0], error_matrix[2][1], 0.0],
            ],
        }
    }

    /// Reassemble Groves' \\(M\\) from the two stored fields.
    ///
    /// The inverse of [`Self::from_error_matrix`]. Use it to compare against a source that
    /// quotes \\(M\\) whole, or to hand the model to code expecting the single-matrix form.
    #[must_use]
    pub const fn error_matrix(&self) -> [[f64; 3]; 3] {
        [
            [
                self.scale_factor[0],
                self.misalignment[0][1],
                self.misalignment[0][2],
            ],
            [
                self.misalignment[1][0],
                self.scale_factor[1],
                self.misalignment[1][2],
            ],
            [
                self.misalignment[2][0],
                self.misalignment[2][1],
                self.scale_factor[2],
            ],
        ]
    }
}

/// A validated, invertible calibration for one three-axis inertial sensor triad.
///
/// Constructed from a [`SensorErrorModel`], which is checked for finiteness and
/// invertibility once. The inverse \\((I + M)^{-1}\\) is computed at that point and cached,
/// so applying the calibration to a sample is a subtraction and a matrix-vector product.
///
/// The fields are private on purpose: the cached correction matrix is only meaningful as the
/// inverse of the stored forward model, and public fields would let a caller change one
/// without the other. Use [`Self::new`] or [`Self::from_parameters`] to build one and
/// [`Self::parameters`] to read the forward model back.
///
/// # Example
///
/// ```rust
/// use strapdown::calibration::SensorCalibration;
/// use nalgebra::Vector3;
///
/// # fn main() -> Result<(), strapdown::StrapdownError> {
/// let gyroscope = SensorCalibration::new(
///     [1e-4, 0.0, -2e-4], // rad/s of turn-on bias
///     [0.002, 0.0, 0.0],  // 0.2% scale factor error on x
///     [[0.0; 3]; 3],
/// )?;
///
/// // Applied to instantaneous rates rather than increments.
/// let corrected = gyroscope.correct_rate(&Vector3::new(1e-4, 0.0, -2e-4));
/// assert!(corrected.norm() < 1e-15);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SensorErrorModel", into = "SensorErrorModel")]
pub struct SensorCalibration {
    /// The forward error model, exactly as supplied.
    parameters: SensorErrorModel,
    /// `bias`, in vector form, so `correct_*` does not rebuild it per sample.
    bias: Vector3<f64>,
    /// \\((I + M)^{-1}\\), validated finite at construction.
    correction: Matrix3<f64>,
}

impl SensorCalibration {
    /// Build a calibration from the forward error-model parameters.
    ///
    /// See [`SensorErrorModel`] for the units and the sign convention of each argument.
    ///
    /// # Errors
    /// * [`StrapdownError::NonFinite`] if any supplied coefficient is `NaN` or infinite.
    /// * [`StrapdownError::InvalidConfiguration`] if any diagonal element of `misalignment`
    ///   is non-zero; scale factor errors belong in `scale_factor`.
    /// * [`StrapdownError::SingularMatrix`] if \\(I + M\\) is not invertible, which in
    ///   practice means a scale factor error near `-1` (a sensor that reports nothing) or a
    ///   mistyped misalignment coefficient orders of magnitude too large.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::calibration::SensorCalibration;
    /// # fn main() -> Result<(), strapdown::StrapdownError> {
    /// let calibration = SensorCalibration::new([0.1, 0.0, 0.0], [0.0; 3], [[0.0; 3]; 3])?;
    /// assert_eq!(calibration.bias(), [0.1, 0.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        bias: [f64; 3],
        scale_factor: [f64; 3],
        misalignment: [[f64; 3]; 3],
    ) -> Result<Self, StrapdownError> {
        Self::from_parameters(SensorErrorModel {
            bias,
            scale_factor,
            misalignment,
        })
    }

    /// Build a calibration from a [`SensorErrorModel`], validating it.
    ///
    /// # Errors
    /// The same three conditions as [`Self::new`], of which this is the struct-literal form.
    pub fn from_parameters(parameters: SensorErrorModel) -> Result<Self, StrapdownError> {
        validate_finite(&parameters)?;
        validate_zero_misalignment_diagonal(&parameters)?;

        let forward = Matrix3::identity() + scale_and_coupling_matrix(&parameters);
        let determinant = forward.determinant();
        if determinant.abs() < MINIMUM_CORRECTION_DETERMINANT {
            return Err(StrapdownError::SingularMatrix {
                what: "IMU calibration scale factor and cross-coupling matrix",
                dim: 3,
            });
        }
        let correction = forward
            .try_inverse()
            .ok_or(StrapdownError::SingularMatrix {
                what: "IMU calibration scale factor and cross-coupling matrix",
                dim: 3,
            })?;

        Ok(Self {
            parameters,
            bias: Vector3::from(parameters.bias),
            correction,
        })
    }

    /// Build a calibration from a bias and a full Groves \\(M\\).
    ///
    /// Convenience for [`SensorErrorModel::from_error_matrix`] followed by
    /// [`Self::from_parameters`]: it splits `error_matrix` into the scale factor diagonal and
    /// the cross-coupling off-diagonal, so a matrix with a non-zero diagonal is accepted here
    /// where [`Self::new`] would reject it as `misalignment`.
    ///
    /// # Errors
    /// [`StrapdownError::NonFinite`] if any coefficient is `NaN` or infinite, or
    /// [`StrapdownError::SingularMatrix`] if \\(I + M\\) is not invertible. The
    /// misalignment-diagonal check cannot fire, since the split fills that diagonal with
    /// zeros by construction.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::calibration::SensorCalibration;
    /// # fn main() -> Result<(), strapdown::StrapdownError> {
    /// let m = [[0.01, 0.0, 0.0], [0.002, 0.0, 0.0], [0.0, 0.0, 0.0]];
    /// let calibration = SensorCalibration::from_error_matrix([0.0; 3], m)?;
    /// assert_eq!(calibration.scale_factor(), [0.01, 0.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_error_matrix(
        bias: [f64; 3],
        error_matrix: [[f64; 3]; 3],
    ) -> Result<Self, StrapdownError> {
        Self::from_parameters(SensorErrorModel::from_error_matrix(bias, error_matrix))
    }

    /// The identity calibration: removes nothing, changes nothing.
    ///
    /// Useful as a default and as the control case in tests.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            parameters: SensorErrorModel::default(),
            bias: Vector3::zeros(),
            correction: Matrix3::identity(),
        }
    }

    /// The forward error model this calibration inverts.
    #[must_use]
    pub const fn parameters(&self) -> SensorErrorModel {
        self.parameters
    }

    /// The additive bias, in the sensor's own units, body frame.
    #[must_use]
    pub const fn bias(&self) -> [f64; 3] {
        self.parameters.bias
    }

    /// The fractional scale factor error per axis.
    #[must_use]
    pub const fn scale_factor(&self) -> [f64; 3] {
        self.parameters.scale_factor
    }

    /// The cross-coupling / misalignment coefficients.
    #[must_use]
    pub const fn misalignment(&self) -> [[f64; 3]; 3] {
        self.parameters.misalignment
    }

    /// The cached correction matrix \\((I + M)^{-1}\\).
    ///
    /// Exposed for inspection and for callers building their own correction pipeline; the
    /// bias is *not* folded into it, since the bias term is the one that does not commute
    /// with integration over the sample interval.
    #[must_use]
    pub const fn correction_matrix(&self) -> Matrix3<f64> {
        self.correction
    }

    /// Correct an instantaneous rate: specific force in m/s², or angular rate in rad/s.
    ///
    /// Applies \\((I + M)^{-1} (\\tilde{x} - b)\\) from Groves Section 4.4.1.
    #[must_use]
    pub fn correct_rate(&self, raw: &Vector3<f64>) -> Vector3<f64> {
        self.correction * (raw - self.bias)
    }

    /// Correct an integrated increment accumulated over `dt` seconds.
    ///
    /// Identical to [`Self::correct_rate`] except that the bias is scaled by `dt` first, the
    /// bias being the one term in the error model that does not commute with integration.
    /// See the module documentation for the derivation.
    #[must_use]
    pub fn correct_increment(&self, raw: &Vector3<f64>, dt: f64) -> Vector3<f64> {
        self.correction * (raw - self.bias * dt)
    }
}

impl Default for SensorCalibration {
    fn default() -> Self {
        Self::identity()
    }
}

impl TryFrom<SensorErrorModel> for SensorCalibration {
    type Error = StrapdownError;

    fn try_from(parameters: SensorErrorModel) -> Result<Self, Self::Error> {
        Self::from_parameters(parameters)
    }
}

impl From<SensorCalibration> for SensorErrorModel {
    fn from(calibration: SensorCalibration) -> Self {
        calibration.parameters
    }
}

/// A deterministic calibration for a complete inertial measurement unit.
///
/// Holds one independently validated [`SensorCalibration`] per sensor triad, because an
/// accelerometer triad and a gyroscope triad are separate instruments with separate error
/// models and separate units. [`Self::correct`] applies both to an [`ImuSample`].
///
/// Serializes to a two-key table, each key a [`SensorErrorModel`]; an omitted key defaults
/// to the identity calibration, so a configuration file can correct the accelerometers alone
/// without spelling out an identity gyroscope block.
///
/// # Example
///
/// ```rust
/// use strapdown::calibration::ImuCalibration;
///
/// // Only the accelerometer block is given; the gyroscope defaults to identity.
/// let yaml = "
/// accelerometer:
///   bias: [0.02, -0.01, 0.005]
///   scale_factor: [0.001, 0.001, 0.002]
/// ";
/// let calibration: ImuCalibration = serde_yaml::from_str(yaml).unwrap();
/// assert_eq!(calibration.accelerometer.bias(), [0.02, -0.01, 0.005]);
/// assert_eq!(calibration.gyroscope.bias(), [0.0; 3]);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImuCalibration {
    /// Error model for the accelerometer triad. Bias in m/s².
    pub accelerometer: SensorCalibration,
    /// Error model for the gyroscope triad. Bias in rad/s.
    pub gyroscope: SensorCalibration,
}

impl ImuCalibration {
    /// Assemble a calibration from two already-validated sensor calibrations.
    ///
    /// Infallible: each argument was validated when it was built.
    #[must_use]
    pub const fn new(accelerometer: SensorCalibration, gyroscope: SensorCalibration) -> Self {
        Self {
            accelerometer,
            gyroscope,
        }
    }

    /// The identity calibration: [`Self::correct`] returns its input unchanged.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            accelerometer: SensorCalibration::identity(),
            gyroscope: SensorCalibration::identity(),
        }
    }

    /// Whether this calibration is the identity, and so has no effect.
    ///
    /// Lets a caller skip the correction entirely when none was configured.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.parameters_are_default()
    }

    fn parameters_are_default(&self) -> bool {
        let default = SensorErrorModel::default();
        self.accelerometer.parameters == default && self.gyroscope.parameters == default
    }

    /// Remove the modelled systematic errors from a raw inertial sample.
    ///
    /// The accelerometer calibration is applied to `delta_v` and the gyroscope calibration to
    /// `delta_theta`, each in the increment domain with the bias scaled by the sample's own
    /// `dt`. The `dt` itself is carried through untouched.
    ///
    /// Infallible by construction -- see the module documentation. Non-finite input is
    /// propagated rather than rejected, matching [`ImuSample::from_rates`]; the finiteness
    /// check belongs at the point the sample enters the crate ([`ImuSample::new`]) and at the
    /// point it is integrated ([`crate::mechanize`]), not in between.
    ///
    /// # Example
    /// ```rust
    /// use strapdown::calibration::{ImuCalibration, SensorCalibration};
    /// use strapdown::ImuSample;
    /// use nalgebra::Vector3;
    ///
    /// # fn main() -> Result<(), strapdown::StrapdownError> {
    /// // A pure 1 rad/s gyro bias on the z axis, observed over a 0.2 s interval.
    /// let gyroscope = SensorCalibration::new([0.0, 0.0, 1.0], [0.0; 3], [[0.0; 3]; 3])?;
    /// let calibration = ImuCalibration::new(SensorCalibration::identity(), gyroscope);
    ///
    /// let raw = ImuSample::new(Vector3::zeros(), Vector3::new(0.0, 0.0, 0.2), 0.2)?;
    /// let corrected = calibration.correct(&raw);
    ///
    /// // 1 rad/s of bias over 0.2 s is 0.2 rad of the increment, so nothing is left.
    /// assert!(corrected.delta_theta.norm() < 1e-15);
    /// assert_eq!(corrected.dt, 0.2);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn correct(&self, raw: &ImuSample) -> ImuSample {
        ImuSample {
            delta_v: self.accelerometer.correct_increment(&raw.delta_v, raw.dt),
            delta_theta: self.gyroscope.correct_increment(&raw.delta_theta, raw.dt),
            dt: raw.dt,
        }
    }

    /// Remove the modelled systematic errors from raw instantaneous rates.
    ///
    /// The rate-domain counterpart of [`Self::correct`], for callers still working with
    /// [`IMUData`]. Because scale factor and cross-coupling are linear, correcting rates and
    /// then integrating gives the same answer as integrating and then correcting increments.
    #[must_use]
    pub fn correct_rates(&self, raw: &IMUData) -> IMUData {
        IMUData {
            accel: self.accelerometer.correct_rate(&raw.accel),
            gyro: self.gyroscope.correct_rate(&raw.gyro),
        }
    }
}

/// Assemble Groves' \\(M\\) from its separately stored diagonal and off-diagonal parts.
const fn scale_and_coupling_matrix(parameters: &SensorErrorModel) -> Matrix3<f64> {
    let scale = parameters.scale_factor;
    let coupling = parameters.misalignment;
    Matrix3::new(
        scale[0],
        coupling[0][1],
        coupling[0][2],
        coupling[1][0],
        scale[1],
        coupling[1][2],
        coupling[2][0],
        coupling[2][1],
        scale[2],
    )
}

/// Reject `NaN` and infinities before they reach the matrix inversion.
fn validate_finite(parameters: &SensorErrorModel) -> Result<(), StrapdownError> {
    if !parameters.bias.iter().all(|value| value.is_finite()) {
        return Err(StrapdownError::NonFinite {
            what: "IMU calibration bias",
        });
    }
    if !parameters
        .scale_factor
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(StrapdownError::NonFinite {
            what: "IMU calibration scale factor",
        });
    }
    if !parameters
        .misalignment
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        return Err(StrapdownError::NonFinite {
            what: "IMU calibration misalignment",
        });
    }
    Ok(())
}

/// Enforce the split between `scale_factor` and `misalignment`.
fn validate_zero_misalignment_diagonal(
    parameters: &SensorErrorModel,
) -> Result<(), StrapdownError> {
    for axis in 0..3 {
        let diagonal = parameters.misalignment[axis][axis];
        if diagonal != 0.0 {
            return Err(StrapdownError::InvalidConfiguration {
                field: "misalignment",
                reason: format!(
                    "diagonal element [{axis}][{axis}] is {diagonal}, but must be zero; \
                     per-axis scale factor errors belong in `scale_factor`. If this is a \
                     full Groves M, build it with `from_error_matrix`, which splits the \
                     diagonal out for you"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ImuCalibration, SensorCalibration, SensorErrorModel, scale_and_coupling_matrix};
    use crate::error::StrapdownError;
    use crate::{IMUData, ImuSample};
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::{Matrix3, Vector3};

    const TOLERANCE: f64 = 1e-12;

    fn sample(delta_v: [f64; 3], delta_theta: [f64; 3], dt: f64) -> ImuSample {
        ImuSample {
            delta_v: Vector3::from(delta_v),
            delta_theta: Vector3::from(delta_theta),
            dt,
        }
    }

    /// A calibration exercising all three error sources on both triads at once.
    fn nontrivial_calibration() -> ImuCalibration {
        let accelerometer = SensorCalibration::new(
            [0.05, -0.02, 0.011],
            [0.01, -0.005, 0.002],
            [
                [0.0, 0.003, -0.001],
                [-0.002, 0.0, 0.004],
                [0.0015, -0.0025, 0.0],
            ],
        )
        .expect("accelerometer parameters are well conditioned");
        let gyroscope = SensorCalibration::new(
            [1e-3, -5e-4, 2e-4],
            [-0.004, 0.006, 0.001],
            [
                [0.0, -0.0012, 0.0021],
                [0.0009, 0.0, -0.0017],
                [-0.0004, 0.0031, 0.0],
            ],
        )
        .expect("gyroscope parameters are well conditioned");
        ImuCalibration::new(accelerometer, gyroscope)
    }

    /// Apply the *forward* model of Groves Eq. 4.17 -- the thing `correct` inverts.
    fn apply_forward_model(
        calibration: &SensorCalibration,
        truth: &Vector3<f64>,
        dt: f64,
    ) -> Vector3<f64> {
        let parameters = calibration.parameters();
        let forward = Matrix3::identity() + scale_and_coupling_matrix(&parameters);
        forward * truth + Vector3::from(parameters.bias) * dt
    }

    // --- Identity -----------------------------------------------------------------

    #[test]
    fn identity_calibration_is_a_no_op() {
        let calibration = ImuCalibration::identity();
        let raw = sample([1.5, -2.25, 9.81], [0.01, -0.02, 0.03], 0.02);
        let corrected = calibration.correct(&raw);
        assert_eq!(corrected, raw);
        assert!(calibration.is_identity());
    }

    #[test]
    fn default_calibration_is_the_identity_calibration() {
        assert_eq!(ImuCalibration::default(), ImuCalibration::identity());
        assert_eq!(
            SensorCalibration::default().correction_matrix(),
            Matrix3::identity()
        );
        assert!(ImuCalibration::default().is_identity());
        assert!(!nontrivial_calibration().is_identity());
    }

    #[test]
    fn identity_leaves_rates_alone() {
        let raw = IMUData {
            accel: Vector3::new(0.1, 0.2, -9.8),
            gyro: Vector3::new(-0.01, 0.0, 0.05),
        };
        let corrected = ImuCalibration::identity().correct_rates(&raw);
        assert_eq!(corrected.accel, raw.accel);
        assert_eq!(corrected.gyro, raw.gyro);
    }

    // --- Hand-computed corrections ------------------------------------------------

    /// Bias and a diagonal scale factor only, so the expected values are arithmetic:
    /// `dt = 0.5`, so the bias contributes `[0.1, -0.2, 0.05]` to the increment, leaving
    /// `[0.9, 2.2, 2.95]`; dividing by `1 + s = [1.25, 0.5, 1.0]` gives the expectation.
    #[test]
    fn bias_and_scale_factor_match_hand_computed_values() {
        let accelerometer =
            SensorCalibration::new([0.2, -0.4, 0.1], [0.25, -0.5, 0.0], [[0.0; 3]; 3]).unwrap();
        let calibration = ImuCalibration::new(accelerometer, SensorCalibration::identity());

        let corrected = calibration.correct(&sample([1.0, 2.0, 3.0], [0.0; 3], 0.5));

        assert_approx_eq!(corrected.delta_v[0], 0.72, TOLERANCE);
        assert_approx_eq!(corrected.delta_v[1], 4.4, TOLERANCE);
        assert_approx_eq!(corrected.delta_v[2], 2.95, TOLERANCE);
        assert_approx_eq!(corrected.dt, 0.5, TOLERANCE);
    }

    /// A single off-diagonal term makes the inverse hand-computable: with `M[0][1] = 0.5`,
    /// `I + M` is unit upper triangular, so its inverse is the same matrix with that element
    /// negated. Correcting `[1, 2, 3]` therefore gives `[1 - 0.5 * 2, 2, 3]`.
    #[test]
    fn misalignment_matches_hand_computed_values() {
        let mut misalignment = [[0.0; 3]; 3];
        misalignment[0][1] = 0.5;
        let gyroscope = SensorCalibration::new([0.0; 3], [0.0; 3], misalignment).unwrap();

        assert_approx_eq!(gyroscope.correction_matrix()[(0, 1)], -0.5, TOLERANCE);

        let calibration = ImuCalibration::new(SensorCalibration::identity(), gyroscope);
        let corrected = calibration.correct(&sample([0.0; 3], [1.0, 2.0, 3.0], 0.1));

        assert_approx_eq!(corrected.delta_theta[0], 0.0, TOLERANCE);
        assert_approx_eq!(corrected.delta_theta[1], 2.0, TOLERANCE);
        assert_approx_eq!(corrected.delta_theta[2], 3.0, TOLERANCE);
    }

    /// The general case: push a known truth through the forward model of Eq. 4.17 and check
    /// that `correct` recovers it. This is the property the module exists to provide, and it
    /// exercises the full non-diagonal inverse that the two tests above only sample.
    #[test]
    fn correct_inverts_the_forward_error_model() {
        let calibration = nontrivial_calibration();
        let dt = 0.02;
        let true_delta_v = Vector3::new(0.031, -0.0072, 0.1962);
        let true_delta_theta = Vector3::new(-2.1e-4, 8.0e-5, 1.3e-3);

        let raw = ImuSample {
            delta_v: apply_forward_model(&calibration.accelerometer, &true_delta_v, dt),
            delta_theta: apply_forward_model(&calibration.gyroscope, &true_delta_theta, dt),
            dt,
        };

        // The raw sample really is corrupted -- otherwise this would prove nothing.
        assert!((raw.delta_v - true_delta_v).norm() > 1e-6);
        assert!((raw.delta_theta - true_delta_theta).norm() > 1e-8);

        let corrected = calibration.correct(&raw);
        for axis in 0..3 {
            assert_approx_eq!(corrected.delta_v[axis], true_delta_v[axis], TOLERANCE);
            assert_approx_eq!(
                corrected.delta_theta[axis],
                true_delta_theta[axis],
                TOLERANCE
            );
        }
        assert_approx_eq!(corrected.dt, dt, TOLERANCE);
    }

    // --- The delta form of the bias -----------------------------------------------

    /// The bias term is the only part of the model that does not commute with integration:
    /// a constant bias `b` contributes `b * dt` to an increment. Doubling `dt` must double
    /// the amount of bias removed.
    #[test]
    fn bias_scales_with_the_sample_interval() {
        let accelerometer =
            SensorCalibration::new([0.4, 0.0, 0.0], [0.0; 3], [[0.0; 3]; 3]).unwrap();
        let calibration = ImuCalibration::new(accelerometer, SensorCalibration::identity());

        let short = calibration.correct(&sample([1.0, 0.0, 0.0], [0.0; 3], 0.25));
        let long = calibration.correct(&sample([1.0, 0.0, 0.0], [0.0; 3], 0.5));

        // 0.4 * 0.25 = 0.1 removed, versus 0.4 * 0.5 = 0.2.
        assert_approx_eq!(short.delta_v[0], 0.9, TOLERANCE);
        assert_approx_eq!(long.delta_v[0], 0.8, TOLERANCE);
    }

    /// Correcting rates then integrating equals integrating then correcting increments,
    /// across a range of `dt`. This is what `bias * dt` buys: if the bias were applied
    /// without the `dt` factor, only `dt == 1.0` would agree.
    #[test]
    fn increment_and_rate_corrections_agree_for_any_interval() {
        let calibration = nontrivial_calibration();
        let rates = IMUData {
            accel: Vector3::new(0.35, -1.2, 9.79),
            gyro: Vector3::new(0.004, -0.011, 0.002),
        };

        for dt in [0.001, 0.01, 0.1, 0.5, 1.0, 2.5] {
            let via_rates = calibration.correct_rates(&rates);
            let via_increments = calibration.correct(&ImuSample::from_rates(&rates, dt));

            for axis in 0..3 {
                assert_approx_eq!(
                    via_increments.delta_v[axis],
                    via_rates.accel[axis] * dt,
                    TOLERANCE
                );
                assert_approx_eq!(
                    via_increments.delta_theta[axis],
                    via_rates.gyro[axis] * dt,
                    TOLERANCE
                );
            }
        }
    }

    /// Scale factor and cross-coupling are linear, so they alone *do* commute with `dt`.
    /// Guards against "fixing" the bias handling by scaling the whole correction.
    #[test]
    fn scale_and_misalignment_do_not_scale_with_the_interval() {
        let accelerometer = SensorCalibration::new(
            [0.0; 3],
            [0.02, 0.0, 0.0],
            [[0.0, 0.01, 0.0], [0.0; 3], [0.0; 3]],
        )
        .unwrap();
        let calibration = ImuCalibration::new(accelerometer, SensorCalibration::identity());

        let raw = [1.0, 2.0, 3.0];
        let short = calibration.correct(&sample(raw, [0.0; 3], 0.01));
        let long = calibration.correct(&sample(raw, [0.0; 3], 10.0));

        for axis in 0..3 {
            assert_approx_eq!(short.delta_v[axis], long.delta_v[axis], TOLERANCE);
        }
    }

    // --- Validation ----------------------------------------------------------------

    #[test]
    fn a_scale_factor_of_negative_one_is_rejected() {
        let error = SensorCalibration::new([0.0; 3], [-1.0, 0.0, 0.0], [[0.0; 3]; 3]).unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::SingularMatrix { dim: 3, .. }
        ));
    }

    #[test]
    fn a_non_finite_coefficient_is_rejected() {
        assert!(matches!(
            SensorCalibration::new([f64::NAN, 0.0, 0.0], [0.0; 3], [[0.0; 3]; 3]).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
        assert!(matches!(
            SensorCalibration::new([0.0; 3], [f64::INFINITY, 0.0, 0.0], [[0.0; 3]; 3]).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
        let mut misalignment = [[0.0; 3]; 3];
        misalignment[1][2] = f64::NEG_INFINITY;
        assert!(matches!(
            SensorCalibration::new([0.0; 3], [0.0; 3], misalignment).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
    }

    #[test]
    fn from_error_matrix_splits_a_full_groves_matrix() {
        // The exact matrix that `new` rejects when handed to `misalignment`.
        let m = [[0.01, 0.003, 0.0], [0.002, -0.02, 0.0], [0.0, 0.004, 0.005]];

        assert!(SensorCalibration::new([0.0; 3], [0.0; 3], m).is_err());

        let parameters = SensorErrorModel::from_error_matrix([0.1, 0.2, 0.3], m);
        assert_eq!(parameters.bias, [0.1, 0.2, 0.3]);
        assert_eq!(parameters.scale_factor, [0.01, -0.02, 0.005]);
        assert_eq!(
            parameters.misalignment,
            [[0.0, 0.003, 0.0], [0.002, 0.0, 0.0], [0.0, 0.004, 0.0]]
        );
        // The split is lossless.
        assert_eq!(parameters.error_matrix(), m);
    }

    #[test]
    fn from_error_matrix_agrees_with_the_split_constructor() {
        let scale_factor = [0.01, -0.02, 0.005];
        let misalignment = [[0.0, 0.003, 0.0], [0.002, 0.0, 0.0], [0.0, 0.004, 0.0]];
        let bias = [0.05, 0.0, -0.01];

        let split = SensorCalibration::new(bias, scale_factor, misalignment).unwrap();
        let whole =
            SensorCalibration::from_error_matrix(bias, split.parameters().error_matrix()).unwrap();

        assert_eq!(split.parameters(), whole.parameters());

        // Same correction applied to a real sample, not just the same stored numbers.
        let raw = ImuSample::from_rates(
            &IMUData {
                accel: Vector3::new(0.3, -0.2, 9.7),
                gyro: Vector3::zeros(),
            },
            0.01,
        );
        let by_split = split.correct_increment(&raw.delta_v, raw.dt);
        let by_whole = whole.correct_increment(&raw.delta_v, raw.dt);
        assert_approx_eq!(by_split[0], by_whole[0], 1e-15);
        assert_approx_eq!(by_split[1], by_whole[1], 1e-15);
        assert_approx_eq!(by_split[2], by_whole[2], 1e-15);
    }

    #[test]
    fn error_matrix_round_trips_through_identity() {
        let identity = SensorErrorModel::default();
        assert_eq!(identity.error_matrix(), [[0.0; 3]; 3]);
        assert_eq!(
            SensorErrorModel::from_error_matrix([0.0; 3], [[0.0; 3]; 3]),
            identity
        );
    }

    #[test]
    fn a_non_zero_misalignment_diagonal_is_rejected() {
        let mut misalignment = [[0.0; 3]; 3];
        misalignment[2][2] = 0.01;
        let error = SensorCalibration::new([0.0; 3], [0.0; 3], misalignment).unwrap_err();
        match error {
            StrapdownError::InvalidConfiguration { field, ref reason } => {
                assert_eq!(field, "misalignment");
                assert!(reason.contains("scale_factor"), "got: {reason}");
            }
            other => panic!("expected InvalidConfiguration, got {other:?}"),
        }
    }

    // --- serde ---------------------------------------------------------------------

    #[test]
    fn json_round_trip_preserves_the_calibration() {
        let calibration = nontrivial_calibration();
        let encoded = serde_json::to_string(&calibration).unwrap();
        let decoded: ImuCalibration = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, calibration);
    }

    #[test]
    fn yaml_round_trip_preserves_the_calibration() {
        let calibration = nontrivial_calibration();
        let encoded = serde_yaml::to_string(&calibration).unwrap();
        let decoded: ImuCalibration = serde_yaml::from_str(&encoded).unwrap();
        assert_eq!(decoded, calibration);
    }

    #[test]
    fn toml_round_trip_preserves_the_calibration() {
        let calibration = nontrivial_calibration();
        let encoded = toml::to_string(&calibration).unwrap();
        let decoded: ImuCalibration = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded, calibration);
    }

    /// A round trip has to survive the *behaviour*, not just the parameters: the cached
    /// correction matrix is rebuilt on deserialization rather than transported.
    #[test]
    fn a_deserialized_calibration_corrects_identically() {
        let calibration = nontrivial_calibration();
        let decoded: ImuCalibration =
            serde_json::from_str(&serde_json::to_string(&calibration).unwrap()).unwrap();
        let raw = sample([0.4, -0.2, 0.19], [0.003, -0.001, 0.002], 0.02);
        assert_eq!(decoded.correct(&raw), calibration.correct(&raw));
    }

    #[test]
    fn omitted_fields_deserialize_to_the_identity() {
        let decoded: ImuCalibration = serde_json::from_str("{}").unwrap();
        assert_eq!(decoded, ImuCalibration::identity());

        let partial: ImuCalibration =
            serde_json::from_str(r#"{"accelerometer": {"bias": [1.0, 2.0, 3.0]}}"#).unwrap();
        assert_eq!(partial.accelerometer.bias(), [1.0, 2.0, 3.0]);
        assert_eq!(partial.accelerometer.scale_factor(), [0.0; 3]);
        assert_eq!(partial.gyroscope, SensorCalibration::identity());
    }

    /// Validation is not bypassable through the deserializer: `#[serde(try_from = ...)]`
    /// routes a config file through the same constructor as Rust callers.
    #[test]
    fn deserialization_rejects_a_singular_calibration() {
        let encoded = r#"{"gyroscope": {"scale_factor": [0.0, -1.0, 0.0]}}"#;
        let error = serde_json::from_str::<ImuCalibration>(encoded).unwrap_err();
        assert!(error.to_string().contains("singular"), "got: {error}");
    }

    #[test]
    fn parameters_survive_the_conversion_to_and_from_the_wire_form() {
        let calibration = nontrivial_calibration().accelerometer;
        let parameters: SensorErrorModel = calibration.into();
        assert_eq!(parameters, calibration.parameters());
        assert_eq!(
            SensorCalibration::from_parameters(parameters).unwrap(),
            calibration
        );
        assert_eq!(calibration.misalignment(), parameters.misalignment);
    }
}
