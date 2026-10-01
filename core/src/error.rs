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
//! That boundary is where the code actually sits as of the v1.0 API freeze. Until then it
//! described an intent rather than a fact: `run_closed_loop`, `run_closed_loop_with_geo`,
//! `HealthMonitor::check` and `ExecutionMonitor::check` are the compute layer, not file I/O,
//! and all four returned `anyhow::Result` while `dead_reckoning` and
//! [`NavigationFilter`](crate::NavigationFilter)'s `predict`/`update` beside them were
//! already typed. A caller could not ask *why* a run aborted without matching on a string.
//! Their signatures are frozen by the tag, so they were typed here or never.
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
    ///
    /// When it is *not* routine -- every update on a trajectory failing this way -- the map
    /// does not extend far enough past the recorded track to cover where the filter wandered,
    /// which is what the message points at. It shows up first under GNSS denial, where the
    /// solution runs unaided for a whole outage before the next fix pulls it back.
    #[error(
        "{axis} {value} is outside the map bounds [{min}, {max}]. If every update on this \
         trajectory fails this way, the map does not cover where the filter went: re-run \
         `just preprocess` with a larger `--margin-km`"
    )]
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

    // --- Run monitoring. Fatal: the run is abandoned. ---
    /// A covariance diagonal entry left the range a healthy filter keeps it in.
    ///
    /// Separate from [`Self::OutOfRange`] because the **index** is the useful half: knowing
    /// that some variance went negative says the filter broke, knowing *which* state says
    /// where. `OutOfRange::what` is a `&'static str` and cannot carry it.
    #[error("covariance diagonal [{index}] = {value} is outside [{min}, {max}]")]
    CovarianceDiagonal {
        /// Index on the covariance diagonal.
        index: usize,
        /// The offending variance.
        value: f64,
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },

    /// A filter rejected enough consecutive measurements to be considered diverged.
    ///
    /// One rejected fix is ordinary and is not this; a run of them means the filter's
    /// covariance no longer describes its error, and later results are meaningless.
    /// `detail` carries whatever identifies the most recent rejection -- the normalised
    /// innovation squared when a NIS streak tripped it, or the rejecting error and its
    /// timestamp when a run of unusable measurements did. The two callers have different
    /// things to say and neither could supply the other's number.
    #[error(
        "filter diverged: {consecutive_rejections} consecutive rejections (limit {limit}); {detail}"
    )]
    FilterDiverged {
        /// How many consecutive measurements were rejected.
        consecutive_rejections: usize,
        /// The limit that was passed.
        limit: usize,
        /// What identifies the most recent rejection.
        detail: String,
    },

    /// A sensor stopped reporting for longer than the run tolerates.
    ///
    /// Raised while the event stream is built, before a single filter step, because only the
    /// builder can tell "the recording has a hole" from "this sensor is simply slower than the
    /// others": it sees a source epoch whose columns for `sensor` were unusable, where the
    /// runner sees nothing at all. Detecting it late is what produced the defect this exists
    /// for -- an IMU that stopped mid-recording left the filter unable to propagate, and the
    /// frozen estimate fell far enough behind the vehicle to surface as an absurd NIS, so
    /// [`Self::FilterDiverged`] blamed the filter for missing data.
    ///
    /// `sensor` is the discriminant that keeps this distinct from a deliberately GNSS-denied
    /// run, which is a scenario rather than a fault and never reaches here.
    #[error(
        "{sensor} stream gap: {duration_s:.1} s without usable data, from t={start_s:.1} s to \
         t={end_s:.1} s ({epochs} source epochs present with unusable {sensor} columns)"
    )]
    SensorStreamGap {
        /// Which sensor stopped reporting.
        sensor: &'static str,
        /// Elapsed time of the last usable sample before the gap, in seconds.
        start_s: f64,
        /// Elapsed time at which the sensor resumed, or the stream ended.
        end_s: f64,
        /// Length of the gap in seconds.
        duration_s: f64,
        /// How many source epochs fell inside the gap.
        epochs: usize,
    },

    /// A run passed one of its execution limits.
    ///
    /// `context` is a `String` rather than a `&'static str` because it is supplied by the
    /// caller describing what it was doing, unlike the library-chosen constants every other
    /// variant carries.
    #[error(
        "execution timeout ({context}): {what} reached {elapsed_s:.2} s against a limit of {limit_s:.2} s"
    )]
    Timeout {
        /// What the caller was doing when the limit was reached.
        context: String,
        /// Which limit: wall clock, or time without progress.
        what: &'static str,
        /// Elapsed seconds measured.
        elapsed_s: f64,
        /// The configured limit, in seconds.
        limit_s: f64,
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

    /// The run-monitoring variants are fatal, and say so here rather than by omission.
    ///
    /// `is_recoverable` is an allow-list, so a new variant is fatal by default -- which is
    /// right for all three of these but is a property of the list's shape rather than a
    /// decision anyone recorded. A variant that ought to be recoverable would be silently
    /// wrong, and nothing else would notice.
    #[test]
    fn run_monitoring_failures_are_fatal() {
        assert!(
            !StrapdownError::CovarianceDiagonal {
                index: 3,
                value: -1.0,
                min: 0.0,
                max: 1e12,
            }
            .is_recoverable(),
            "a negative variance means the covariance is no longer a covariance"
        );
        assert!(
            !StrapdownError::FilterDiverged {
                consecutive_rejections: 21,
                limit: 20,
                detail: "last NIS 214.8".to_owned(),
            }
            .is_recoverable(),
            "a rejection streak means the covariance no longer describes the error"
        );
        assert!(
            !StrapdownError::Timeout {
                context: "closed loop".to_owned(),
                what: "wall clock",
                elapsed_s: 61.0,
                limit_s: 60.0,
            }
            .is_recoverable(),
            "a run that passed its limit has no partial result worth continuing from"
        );
        assert!(
            !StrapdownError::SensorStreamGap {
                sensor: "IMU",
                start_s: 411.0,
                end_s: 1319.0,
                duration_s: 908.0,
                epochs: 908,
            }
            .is_recoverable(),
            "an inertial stream with a hole in it cannot be propagated across the hole"
        );
    }

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
        assert!(!(-1.0..=1.0).contains(&nan));
        assert!(!nan.is_finite());
    }
}
