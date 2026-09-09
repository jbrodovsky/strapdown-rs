//! Unified error type for the strapdown navigation library.
//!
//! Before this module the crate carried four incompatible error styles — `&'static str`
//! from [`StrapdownState::try_from`](crate::StrapdownState), `String` from geonav's map
//! loader, [`anyhow`] across the simulation I/O layer, and `Box<dyn Error>` in the CLI —
//! with panics filling every remaining gap. The practical cost was that a filter whose
//! estimate drifted off a loaded geophysical map tile aborted the entire run, which is
//! precisely the condition map-matching navigation exists to recover from.
//!
//! [`StrapdownError`] replaces the panics and the first two of those styles. The `anyhow`
//! layer in [`crate::sim`] is deliberately kept for file I/O, where `anyhow`'s context
//! chaining is genuinely more useful than a fixed enum; `StrapdownError` converts into
//! `anyhow::Error` automatically, so `?` composes across the boundary.
//!
//! # Recoverable versus fatal
//!
//! The distinction that matters operationally is [`StrapdownError::is_recoverable`]: a
//! recoverable error concerns one measurement and leaves the filter state valid, so the
//! caller may skip that update and continue. Everything else means the state is no longer
//! trustworthy and the run should stop rather than emit numbers that look like a
//! trajectory and are not.

use std::path::PathBuf;

use thiserror::Error;

/// Errors produced by the strapdown navigation library.
///
/// Marked `#[non_exhaustive]`: issue #260 (innovation gating) adds a rejection variant and
/// issue #262 (lever-arm compensation) adds geometry variants, and neither should be a
/// breaking change for downstream `match` statements.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum StrapdownError {
    // --- Shape and conversion. Caller error; fatal. ---
    /// A vector or matrix did not have the length the operation requires.
    #[error("dimension mismatch in {what}: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The quantity whose length was wrong.
        what: &'static str,
        /// The required length.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },

    /// A matrix operation that requires a square matrix was given a rectangular one.
    #[error("{what} requires a square matrix, got {rows}x{cols}")]
    NotSquare {
        /// The operation that requires squareness.
        what: &'static str,
        /// Row count of the offending matrix.
        rows: usize,
        /// Column count of the offending matrix.
        cols: usize,
    },

    // --- Physical range. Bad input; fatal at construction. ---
    /// A physical quantity fell outside the range in which the mechanization is valid.
    #[error("{what} = {value} is outside the valid range [{min}, {max}]")]
    OutOfRange {
        /// The quantity that was out of range.
        what: &'static str,
        /// The offending value.
        value: f64,
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },

    /// A value was `NaN` or infinite where a finite number is required.
    ///
    /// Worth a dedicated variant rather than folding into [`Self::OutOfRange`]: every
    /// comparison against `NaN` is false, so `NaN` silently passes range checks written as
    /// `min <= x && x <= max`. It has to be rejected explicitly or not at all.
    #[error("{what} is not finite")]
    NonFinite {
        /// The quantity that was not finite.
        what: &'static str,
    },

    // --- Filter runtime. ---
    /// A filter was handed an [`InputModel`](crate::InputModel) it does not support.
    #[error("{filter} does not support this input model; expected {expected}")]
    UnsupportedInput {
        /// The filter that rejected the input.
        filter: &'static str,
        /// The input type the filter accepts.
        expected: &'static str,
    },

    /// A matrix could not be factorized or inverted, even after diagonal jitter.
    ///
    /// In practice this means a diverged or collapsed covariance. The filter cannot
    /// continue from it, but the caller may be able to reinitialize.
    #[error("{what}: matrix is singular and could not be solved (dimension {dim})")]
    SingularMatrix {
        /// The operation that failed.
        what: &'static str,
        /// Dimension of the offending matrix.
        dim: usize,
    },

    /// An [`ImuSample`](crate::ImuSample)'s own `dt` disagreed with the `dt` argument.
    ///
    /// Silently preferring one over the other would reproduce the class of defect that
    /// issue #292 was: an integration rate that is quietly wrong rather than loudly absent.
    #[error(
        "timestep mismatch: sample carries dt = {sample_dt} s but was propagated with {arg_dt} s"
    )]
    InconsistentTimestep {
        /// The `dt` carried by the sample.
        sample_dt: f64,
        /// The `dt` passed to the propagation call.
        arg_dt: f64,
    },

    // --- Aiding and measurements. Recoverable: skip this update, keep going. ---
    /// A queried coordinate fell outside the bounds of a loaded geophysical map.
    ///
    /// Routine rather than exceptional: a filter estimate near a tile edge, or any particle
    /// in the tail of the distribution, lands off-map regularly.
    #[error("{axis} {value} is outside the map bounds [{min}, {max}]")]
    OutOfMapBounds {
        /// Which axis was exceeded, `"latitude"` or `"longitude"`.
        axis: &'static str,
        /// The offending coordinate.
        value: f64,
        /// Lower bound of the map along that axis.
        min: f64,
        /// Upper bound of the map along that axis.
        max: f64,
    },

    /// A measurement model could not produce a value for the current state.
    #[error("{model} could not produce a measurement: {reason}")]
    MeasurementUnavailable {
        /// The measurement model that failed.
        model: &'static str,
        /// Why it could not produce a value.
        reason: String,
    },

    /// An external geophysical model rejected the query.
    ///
    /// Chiefly the World Magnetic Model, which has hard validity ranges in position and a
    /// coefficient epoch that expires.
    #[error("{model} is unavailable here: {detail}")]
    ExternalModel {
        /// The external model that failed.
        model: &'static str,
        /// Detail reported by the model.
        detail: String,
    },

    // --- Configuration. Fatal. ---
    /// A user-supplied configuration value was invalid.
    #[error("invalid configuration for `{field}`: {reason}")]
    InvalidConfiguration {
        /// The configuration key at fault.
        field: &'static str,
        /// Why the value is unusable.
        reason: String,
    },

    // --- Map I/O (geonav). Fatal. ---
    /// A geophysical map file could not be loaded.
    #[error("failed to load map from {}: {detail}", path.display())]
    MapLoad {
        /// Path that was being loaded.
        path: PathBuf,
        /// What went wrong.
        detail: String,
    },
}

impl StrapdownError {
    /// Whether the caller may skip the current measurement and continue.
    ///
    /// A recoverable error concerns a single measurement and leaves the filter state
    /// untouched and valid. A fatal one means the state itself is no longer trustworthy.
    ///
    /// This exists as one function rather than as a `match` at each call site so that
    /// adding a variant — issue #260's innovation gating is the next one — extends the
    /// policy in a single place instead of silently defaulting to "fatal" wherever a
    /// caller forgot to update its pattern.
    ///
    /// # Examples
    ///
    /// ```
    /// use strapdown::StrapdownError;
    ///
    /// let off_map = StrapdownError::OutOfMapBounds {
    ///     axis: "latitude",
    ///     value: 41.0,
    ///     min: 39.0,
    ///     max: 40.0,
    /// };
    /// assert!(off_map.is_recoverable());
    ///
    /// let singular = StrapdownError::SingularMatrix { what: "kalman gain", dim: 9 };
    /// assert!(!singular.is_recoverable());
    /// ```
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::OutOfMapBounds { .. }
                | Self::MeasurementUnavailable { .. }
                | Self::ExternalModel { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::StrapdownError;

    #[test]
    fn measurement_failures_are_recoverable() {
        assert!(
            StrapdownError::OutOfMapBounds {
                axis: "longitude",
                value: -70.0,
                min: -75.0,
                max: -74.0,
            }
            .is_recoverable()
        );
        assert!(
            StrapdownError::ExternalModel {
                model: "WMM",
                detail: "epoch expired".to_owned(),
            }
            .is_recoverable()
        );
    }

    #[test]
    fn state_corrupting_failures_are_fatal() {
        assert!(
            !StrapdownError::SingularMatrix {
                what: "robust_spd_solve",
                dim: 15,
            }
            .is_recoverable()
        );
        assert!(!StrapdownError::NonFinite { what: "innovation" }.is_recoverable());
        assert!(
            !StrapdownError::DimensionMismatch {
                what: "state",
                expected: 9,
                got: 3,
            }
            .is_recoverable()
        );
    }

    #[test]
    fn messages_name_the_offending_quantity() {
        let e = StrapdownError::OutOfRange {
            what: "latitude",
            value: 2.0,
            min: -std::f64::consts::FRAC_PI_2,
            max: std::f64::consts::FRAC_PI_2,
        };
        let rendered = e.to_string();
        assert!(rendered.contains("latitude"), "got: {rendered}");
        assert!(rendered.contains('2'), "got: {rendered}");
    }

    /// `NaN` defeats range checks written as comparisons, which is how it reached the
    /// `position(..).unwrap()` in the geophysical map lookup. The dedicated variant is
    /// the reminder that it has to be tested for explicitly.
    #[test]
    fn nan_defeats_comparison_based_range_checks() {
        let nan = f64::NAN;
        assert!(!(-1.0..=1.0).contains(&nan));
        assert!(!(nan >= -1.0 && nan <= 1.0));
        assert!(!nan.is_finite());
    }
}
