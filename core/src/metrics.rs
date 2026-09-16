//! Navigation-accuracy metrics for scoring a solution against a reference trajectory.
//!
//! This module answers one question: *how far wrong was the navigation solution, and did the
//! filter know it?* It takes the `Vec<NavigationResult>` that
//! [`sim::run_closed_loop`](crate::sim::run_closed_loop) and
//! [`sim::dead_reckoning`](crate::sim::dead_reckoning) already produce, aligns it against a
//! series of [`TruthSample`]s, and reduces the pair to the fixed set of scalars in
//! [`MetricId`].
//!
//! It exists because the same reductions -- RMS of a wrapped angle, a haversine position
//! error, a percentile of a float slice -- were written four times across this workspace's
//! test targets, each copy slightly different. `core/tests/perf_baseline.rs` gates on these
//! numbers, so they need one definition with one set of documented edge cases.
//!
//! # What "truth" means here
//!
//! Two sources reduce to a [`TruthSample`], and they are not equally good:
//!
//! * [`truth_from_trajectory`] takes the exact, noise-free trajectory returned as the first
//!   element of [`sim::generate_synthetic`](crate::sim::generate_synthetic). This is real
//!   ground truth -- the mechanization's own integral of a known input -- and an error scored
//!   against it is the filter's error.
//! * [`truth_from_records`] takes the GNSS fixes out of a
//!   [`TestDataRecord`] log. On any log this crate ships, those
//!   fixes are **also the filter's aiding source**, so the resulting metrics measure agreement
//!   with the aid rather than independent accuracy, and they cannot fall below the receiver's
//!   own noise however good the filter is.
//!
//! # Consistency: the NEES, and the diagonal-only form beside it
//!
//! [`MetricId::NeesPosition`] is the real statistic -- the mean of $e^\top P^{-1} e$ over the
//! 3x3 position block, off-diagonals included, consistent at 3.0. It became computable in #376,
//! which stopped [`NavigationResult`] discarding the position off-diagonals when a row is
//! built.
//!
//! [`MetricId::NpesPosition`] is what this crate could measure before that:
//!
//! $$ \overline{\epsilon} = \frac{1}{N} \sum_k \left( \frac{e_{lat,k}^2}{P_{lat,k}} +
//!    \frac{e_{lon,k}^2}{P_{lon,k}} + \frac{e_{alt,k}^2}{P_{alt,k}} \right) $$
//!
//! It equals the NEES only when the position block is genuinely diagonal, and is **optimistic**
//! -- too small -- whenever the states are correlated, which after a GNSS update they are. The
//! size of that gap is not subtle: at a latitude-longitude correlation of 0.9, an error along
//! the unlikely direction scores 2.0 on the diagonal form and 20.0 on the real one.
//!
//! Both are kept. The NEES is the one to believe; `npes_position` stays so that the baseline it
//! has accumulated remains comparable across the change, and it is gated in
//! `core/tests/perf_baseline.rs` on that basis rather than as a claim about consistency.
//!
//! Read it **beside** [`MetricId::Containment3SigmaHorizontal`], never alone. The two together
//! separate the failure modes that either one alone hides:
//!
//! | containment | `npes_position` | diagnosis |
//! |---|---|---|
//! | ~1.0 | much less than 3 | over-conservative: the filter is right but does not believe it |
//! | below 0.99 | much greater than 3 | over-confident -- the dangerous one |
//! | ~0.9973 | ~3 | consistent |
//!
//! # Why there is no Wasserstein-2 metric
//!
//! It was considered and deliberately left out, because none of its readings earns a place
//! beside the metrics above. Against a Dirac at truth the closed form is
//! $W_2^2 = \lVert \hat{x} - x \rVert^2 + \mathrm{tr}(P)$ -- squared error plus total
//! variance, strictly less information than reporting the error and the consistency
//! separately, and it forces an arbitrary weighting to mix metres, m/s and radians into one
//! scalar. Against the empirical distribution of the error sequence it is not a distance
//! between samples of anything: the errors along one trajectory are strongly autocorrelated,
//! so the statistic drifts with run length and has no stable interpretation. Against a
//! reference posterior it is genuinely the right tool for comparing Monte-Carlo
//! approximations, but there is no reference posterior here.
//!
//! The one shape that would fit a regression harness is the one-dimensional $W_2$ between a
//! run's sorted horizontal-error vector and a stored baseline quantile vector. That is a
//! metric between two *runs*, not a property of one, so it cannot be a scalar baseline entry;
//! it would need a quantile vector per scenario. [`MetricId::HorizontalCep50`] and
//! [`MetricId::HorizontalCep95`] capture the same "the shape of the error distribution moved"
//! signal at a fraction of the cost, and a human reading a failure message can interpret them.
use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use nalgebra::{Matrix3, Rotation3, Vector3};
use serde::{Deserialize, Serialize};

use crate::earth::haversine_distance;
use crate::error::StrapdownError;
use crate::sim::{NavigationResult, TestDataRecord};
use crate::wrap_to_pi;

/// Degrees of freedom in the position channel.
///
/// The value a consistent filter scores on [`MetricId::NeesPosition`], and on
/// [`MetricId::NpesPosition`] when the position block happens to be diagonal.
pub const POSITION_DEGREES_OF_FREEDOM: f64 = 3.0;

/// Fraction of a Gaussian inside plus or minus three standard deviations.
///
/// The target for both containment metrics. A filter above it is conservative, one below it
/// over-confident.
pub const THREE_SIGMA_CONTAINMENT: f64 = 0.997_300_203_936_740;

/// One sample of reference truth, in the units [`NavigationResult`] uses at runtime.
///
/// Both truth sources this crate can produce reduce to this, so nothing downstream has to ask
/// which one it is holding. Latitude and longitude are **degrees**, matching
/// [`NavigationResult::latitude`]; this is the single place that question is settled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TruthSample {
    /// UTC timestamp this sample is valid at.
    pub timestamp: DateTime<Utc>,
    /// Reference latitude, degrees (WGS84).
    pub latitude_deg: f64,
    /// Reference longitude, degrees (WGS84).
    pub longitude_deg: f64,
    /// Reference height above the ellipsoid, metres, positive up in both frames.
    pub altitude_m: f64,
    /// Reference north velocity, m/s.
    pub velocity_north_mps: f64,
    /// Reference east velocity, m/s.
    pub velocity_east_mps: f64,
    /// Reference vertical velocity, m/s, in the source's own sign convention.
    pub velocity_vertical_mps: f64,
    /// Reference attitude, or `None` when the source carries no usable one.
    ///
    /// A `None` here removes that sample from the four attitude metrics and from nothing else.
    pub attitude: Option<Rotation3<f64>>,
}

/// Truth series from an exact synthetic trajectory.
///
/// Takes the first element of [`sim::generate_synthetic`](crate::sim::generate_synthetic),
/// which is the noise-free state the noisy sensor records were derived from.
///
/// Samples whose position is not finite are dropped rather than carried: a `NaN` latitude
/// makes every downstream comparison `false`, which would silently shrink the sample count
/// instead of failing.
#[must_use]
pub fn truth_from_trajectory(truth: &[NavigationResult]) -> Vec<TruthSample> {
    truth
        .iter()
        .filter(|row| {
            row.latitude.is_finite() && row.longitude.is_finite() && row.altitude.is_finite()
        })
        .map(|row| TruthSample {
            timestamp: row.timestamp,
            latitude_deg: row.latitude,
            longitude_deg: row.longitude,
            altitude_m: row.altitude,
            velocity_north_mps: row.velocity_north,
            velocity_east_mps: row.velocity_east,
            velocity_vertical_mps: row.velocity_vertical,
            attitude: Some(Rotation3::from_euler_angles(row.roll, row.pitch, row.yaw)),
        })
        .collect()
}

/// Truth series from a sensor log, using the GNSS fix as the reference.
///
/// Velocity comes from [`TestDataRecord::ground_track_velocity`], which converts the bearing
/// from degrees once so that callers stop forgetting to. Attitude comes from
/// [`TestDataRecord::attitude`] -- the quaternion, not the record's `roll`/`pitch`/`yaw`
/// columns, which are a different Euler convention and disagree by more than a sign.
///
/// A record whose quaternion is unusable gets `None` rather than what
/// [`TestDataRecord::attitude`] returns for one, which is the identity rotation. That fallback
/// is right for seeding a filter and wrong for scoring one: carried into a truth series it
/// would be indistinguishable from a genuine level-and-north-facing reference, and every
/// estimate would be scored against an attitude the log never recorded.
///
/// Records whose position is not finite are dropped. The GNSS fix is also the filters' aiding
/// source on every log this crate ships, so see the module documentation before reading an
/// absolute number off these metrics.
#[must_use]
pub fn truth_from_records(records: &[TestDataRecord]) -> Vec<TruthSample> {
    records
        .iter()
        .filter(|r| r.latitude.is_finite() && r.longitude.is_finite() && r.altitude.is_finite())
        .map(|r| {
            let (velocity_north_mps, velocity_east_mps) = r.ground_track_velocity();
            TruthSample {
                timestamp: r.time,
                latitude_deg: r.latitude,
                longitude_deg: r.longitude,
                altitude_m: r.altitude,
                velocity_north_mps,
                velocity_east_mps,
                // A GNSS ground track carries no vertical component. Zero is the reference a
                // level run should be scored against, and it is what the existing integration
                // tests have always compared to.
                velocity_vertical_mps: 0.0,
                attitude: usable_attitude(r),
            }
        })
        .collect()
}

/// The record's attitude, or `None` when its quaternion cannot supply one.
///
/// [`TestDataRecord::attitude`] substitutes the identity for a non-finite or zero-norm
/// quaternion, so the same test has to be made here before the result can be called truth.
fn usable_attitude(record: &TestDataRecord) -> Option<Rotation3<f64>> {
    let finite = [record.qw, record.qx, record.qy, record.qz]
        .iter()
        .all(|c| c.is_finite());
    let norm_squared = record.qw * record.qw
        + record.qx * record.qx
        + record.qy * record.qy
        + record.qz * record.qz;
    (finite && norm_squared > f64::EPSILON * f64::EPSILON).then(|| record.attitude())
}

/// One gated accuracy metric.
///
/// Marked `#[non_exhaustive]` so that adding a metric is not a breaking change for a
/// downstream exhaustive `match`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricId {
    /// Root-mean-square great-circle position error, metres.
    HorizontalRmse,
    /// Median radial position error, metres.
    ///
    /// The empirical 50th percentile of the radial error, which is the usual non-parametric
    /// estimator of CEP. It is *not* the Rayleigh-parametric CEP, so do not assert a
    /// distributional relationship between this and [`Self::HorizontalRmse`].
    HorizontalCep50,
    /// 95th-percentile radial position error, metres. See [`Self::HorizontalCep50`].
    HorizontalCep95,
    /// Largest single-sample radial position error, metres.
    HorizontalMax,
    /// Root-mean-square altitude error, metres.
    VerticalRmse,
    /// Mean **signed** altitude error, metres.
    ///
    /// Reported beside [`Self::VerticalRmse`] because the vertical channel's characteristic
    /// failure is a slow one-sided drift, which a squared error reports only after it is
    /// large and which a signed mean catches while it is small.
    VerticalBias,
    /// Root-mean-square horizontal velocity error, m/s, north and east combined.
    VelocityHorizontalRmse,
    /// Root-mean-square vertical velocity error, m/s.
    VelocityVerticalRmse,
    /// Root-mean-square roll error, degrees. Observable through gravity.
    RollRmse,
    /// Root-mean-square pitch error, degrees. Observable through gravity.
    PitchRmse,
    /// Root-mean-square yaw error, degrees.
    ///
    /// Only meaningful where the run carries a heading aid. Under position-only aiding yaw is
    /// unobservable and this tends to 103.9 degrees, the RMS of an error uniform on the
    /// circle.
    YawRmse,
    /// Root-mean-square geodesic attitude error, degrees.
    ///
    /// The rotation angle of `R_est^-1 R_truth`: the single rotation carrying one attitude
    /// onto the other, and the only attitude error that does not depend on a choice of Euler
    /// sequence.
    AttitudeGeodesicRmse,
    /// Mean normalized position error squared, dimensionless. See the module documentation --
    /// this is not the NEES, and 3.0 is the consistent value.
    ///
    /// Kept beside [`Self::NeesPosition`] rather than replaced by it, so the baseline this
    /// metric has accumulated stays comparable across #376.
    NpesPosition,
    /// Mean normalized estimation error squared over the 3x3 position block, dimensionless.
    ///
    /// The real statistic: the mean of $e^\top P^{-1} e$ with $e$ the position error in the
    /// states' own units (radians, radians, metres) and $P$ the filter's position block,
    /// off-diagonals included. Consistent at 3.0, the block's degrees of freedom.
    ///
    /// [`Self::NpesPosition`] is the same quantity computed as though $P$ were diagonal, which
    /// is what this crate could measure before the off-diagonals reached
    /// [`NavigationResult`](crate::sim::NavigationResult). That form is **optimistic**: it
    /// equals the NEES only when the position states are genuinely uncorrelated, and after a
    /// GNSS update they are not. Where the two disagree, this one is right.
    NeesPosition,
    /// Fraction of horizontal channel-samples inside plus or minus three sigma.
    ///
    /// Latitude and longitude are counted as separate channel-samples, so a run of `N` aligned
    /// samples contributes up to `2N` here.
    Containment3SigmaHorizontal,
    /// Fraction of altitude samples inside plus or minus three sigma.
    Containment3SigmaVertical,
}

/// Whether a metric improves by decreasing, or by approaching a particular value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricDirection {
    /// Smaller is strictly better, and the value is non-negative. Equivalent to
    /// `TowardTarget { target: 0.0 }`, spelled separately because it reads better in a report.
    LowerIsBetter,
    /// Better means closer to `target`, from either side.
    TowardTarget {
        /// The value a correct implementation would produce.
        target: f64,
    },
}

impl MetricId {
    /// Every metric, in report order.
    ///
    /// A slice rather than a fixed-size array so that adding a metric does not change a type.
    pub const ALL: &'static [Self] = &[
        Self::HorizontalRmse,
        Self::HorizontalCep50,
        Self::HorizontalCep95,
        Self::HorizontalMax,
        Self::VerticalRmse,
        Self::VerticalBias,
        Self::VelocityHorizontalRmse,
        Self::VelocityVerticalRmse,
        Self::RollRmse,
        Self::PitchRmse,
        Self::YawRmse,
        Self::AttitudeGeodesicRmse,
        Self::NpesPosition,
        Self::NeesPosition,
        Self::Containment3SigmaHorizontal,
        Self::Containment3SigmaVertical,
    ];

    /// Stable key used in the baseline file. Never change one in place -- a renamed key reads
    /// as one metric removed and another added, which is a deliberately loud failure but a
    /// confusing one.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::HorizontalRmse => "horizontal_rmse_m",
            Self::HorizontalCep50 => "horizontal_cep50_m",
            Self::HorizontalCep95 => "horizontal_cep95_m",
            Self::HorizontalMax => "horizontal_max_m",
            Self::VerticalRmse => "vertical_rmse_m",
            Self::VerticalBias => "vertical_bias_m",
            Self::VelocityHorizontalRmse => "velocity_horizontal_rmse_mps",
            Self::VelocityVerticalRmse => "velocity_vertical_rmse_mps",
            Self::RollRmse => "roll_rmse_deg",
            Self::PitchRmse => "pitch_rmse_deg",
            Self::YawRmse => "yaw_rmse_deg",
            Self::AttitudeGeodesicRmse => "attitude_geodesic_rmse_deg",
            Self::NpesPosition => "npes_position",
            Self::NeesPosition => "nees_position",
            Self::Containment3SigmaHorizontal => "containment_3sigma_horizontal",
            Self::Containment3SigmaVertical => "containment_3sigma_vertical",
        }
    }

    /// Unit symbol for a printed table.
    #[must_use]
    pub const fn unit(self) -> &'static str {
        match self {
            Self::HorizontalRmse
            | Self::HorizontalCep50
            | Self::HorizontalCep95
            | Self::HorizontalMax
            | Self::VerticalRmse
            | Self::VerticalBias => "m",
            Self::VelocityHorizontalRmse | Self::VelocityVerticalRmse => "m/s",
            Self::RollRmse | Self::PitchRmse | Self::YawRmse | Self::AttitudeGeodesicRmse => "deg",
            Self::NpesPosition | Self::NeesPosition => "1",
            Self::Containment3SigmaHorizontal | Self::Containment3SigmaVertical => "fraction",
        }
    }

    /// How this metric improves.
    ///
    /// Deliberately a property of the code rather than of the baseline file: direction is a
    /// fact about the metric, and a hand-editable copy of it could disagree with this one,
    /// whose failure mode is a gate that silently runs backwards.
    #[must_use]
    pub const fn direction(self) -> MetricDirection {
        match self {
            Self::VerticalBias => MetricDirection::TowardTarget { target: 0.0 },
            Self::NpesPosition | Self::NeesPosition => MetricDirection::TowardTarget {
                target: POSITION_DEGREES_OF_FREEDOM,
            },
            Self::Containment3SigmaHorizontal | Self::Containment3SigmaVertical => {
                MetricDirection::TowardTarget {
                    target: THREE_SIGMA_CONTAINMENT,
                }
            }
            _ => MetricDirection::LowerIsBetter,
        }
    }

    /// Absolute slack added to a relative tolerance band, in this metric's own units.
    ///
    /// Without it a metric whose baseline sits near its target is gated on nothing but
    /// floating-point noise: three-sigma containment measured at 1.000 against a target of
    /// 0.9973 is 0.0027 away, and a bare ten-percent band on that distance is plus or minus
    /// 0.00027 -- tighter than the difference one sample makes on a 5,000-sample run.
    #[must_use]
    pub const fn noise_floor(self) -> f64 {
        // Grouped by value rather than by unit: metres and degrees share a floor because
        // 0.02 is negligible in both, as do m/s and a fraction at 0.005.
        match self.unit().as_bytes() {
            b"m" | b"deg" => 0.02,
            b"m/s" | b"fraction" => 0.005,
            _ => 0.05,
        }
    }
}

/// Knobs that change what is compared, as opposed to what is computed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricOptions {
    /// Largest timestamp gap, in seconds, at which an estimate may be matched to a truth
    /// sample. Zero -- the default -- requires the two to fall on the same millisecond.
    pub max_match_gap_s: f64,
    /// Aligned pairs to discard from the front, to exclude an initialisation transient.
    pub warmup_samples: usize,
}

/// Every metric for one run, plus the sample count they were computed over.
///
/// A metric that cannot be computed for this run -- the two containment metrics and
/// `npes_position` on a dead-reckoning solution, whose covariance columns are all `NaN` -- is
/// `None`, not `NaN`. The distinction has to live in the type: `serde_json` writes `f64::NAN`
/// as `null` and then refuses to read `null` back into an `f64`, so a `NaN` here would produce
/// a baseline file that cannot be loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Number of aligned estimate/truth pairs the metrics were computed over, after warmup.
    pub sample_count: usize,
    /// Channel-samples dropped as non-finite while reducing those pairs, summed over metrics.
    ///
    /// Zero on any healthy run. It is reported because a dropped sample does not lower
    /// [`Self::sample_count`] but does leave the metric it was dropped from computed over
    /// fewer values -- and dropping the worst samples makes an RMSE, a percentile and a
    /// containment fraction all look better. A gate that recorded only `sample_count` could
    /// therefore be passed by a change that produced non-finite errors rather than by one
    /// that was correct, so a caller gating on these metrics should compare this too.
    ///
    /// This is separate from an absent covariance, which is not a discard: a `NaN` variance
    /// makes the three consistency metrics `None` and is counted nowhere.
    pub discarded_channel_samples: usize,
    /// See [`MetricId::HorizontalRmse`].
    pub horizontal_rmse_m: Option<f64>,
    /// See [`MetricId::HorizontalCep50`].
    pub horizontal_cep50_m: Option<f64>,
    /// See [`MetricId::HorizontalCep95`].
    pub horizontal_cep95_m: Option<f64>,
    /// See [`MetricId::HorizontalMax`].
    pub horizontal_max_m: Option<f64>,
    /// See [`MetricId::VerticalRmse`].
    pub vertical_rmse_m: Option<f64>,
    /// See [`MetricId::VerticalBias`].
    pub vertical_bias_m: Option<f64>,
    /// See [`MetricId::VelocityHorizontalRmse`].
    pub velocity_horizontal_rmse_mps: Option<f64>,
    /// See [`MetricId::VelocityVerticalRmse`].
    pub velocity_vertical_rmse_mps: Option<f64>,
    /// See [`MetricId::RollRmse`].
    pub roll_rmse_deg: Option<f64>,
    /// See [`MetricId::PitchRmse`].
    pub pitch_rmse_deg: Option<f64>,
    /// See [`MetricId::YawRmse`].
    pub yaw_rmse_deg: Option<f64>,
    /// See [`MetricId::AttitudeGeodesicRmse`].
    pub attitude_geodesic_rmse_deg: Option<f64>,
    /// See [`MetricId::NpesPosition`].
    pub npes_position: Option<f64>,
    /// See [`MetricId::NeesPosition`].
    pub nees_position: Option<f64>,
    /// See [`MetricId::Containment3SigmaHorizontal`].
    pub containment_3sigma_horizontal: Option<f64>,
    /// See [`MetricId::Containment3SigmaVertical`].
    pub containment_3sigma_vertical: Option<f64>,
}

impl AccuracyMetrics {
    /// One metric by identity.
    #[must_use]
    pub const fn get(&self, id: MetricId) -> Option<f64> {
        match id {
            MetricId::HorizontalRmse => self.horizontal_rmse_m,
            MetricId::HorizontalCep50 => self.horizontal_cep50_m,
            MetricId::HorizontalCep95 => self.horizontal_cep95_m,
            MetricId::HorizontalMax => self.horizontal_max_m,
            MetricId::VerticalRmse => self.vertical_rmse_m,
            MetricId::VerticalBias => self.vertical_bias_m,
            MetricId::VelocityHorizontalRmse => self.velocity_horizontal_rmse_mps,
            MetricId::VelocityVerticalRmse => self.velocity_vertical_rmse_mps,
            MetricId::RollRmse => self.roll_rmse_deg,
            MetricId::PitchRmse => self.pitch_rmse_deg,
            MetricId::YawRmse => self.yaw_rmse_deg,
            MetricId::AttitudeGeodesicRmse => self.attitude_geodesic_rmse_deg,
            MetricId::NpesPosition => self.npes_position,
            MetricId::NeesPosition => self.nees_position,
            MetricId::Containment3SigmaHorizontal => self.containment_3sigma_horizontal,
            MetricId::Containment3SigmaVertical => self.containment_3sigma_vertical,
        }
    }

    /// Every metric paired with its identity, in [`MetricId::ALL`] order.
    ///
    /// This is what lets a baseline reader, a table printer and a gate share one traversal
    /// instead of each spelling out fifteen field accesses.
    pub fn iter(&self) -> impl Iterator<Item = (MetricId, Option<f64>)> + '_ {
        MetricId::ALL.iter().map(move |&id| (id, self.get(id)))
    }
}

/// Score a navigation solution against a truth series.
///
/// Estimates are matched to truth samples by timestamp; see [`MetricOptions::max_match_gap_s`]
/// for the matching rule. Unmatched estimates and unmatched truth samples are ignored, so
/// [`AccuracyMetrics::sample_count`] is the number of pairs actually scored and should be
/// checked by any caller gating on the result.
///
/// # Errors
///
/// Returns [`StrapdownError::InvalidConfiguration`] when:
///
/// * `estimates` or `truth` is empty;
/// * `options.max_match_gap_s` is negative or not finite;
/// * no estimate matched any truth sample -- the message names the first and last timestamp on
///   each side, because the overwhelmingly likely cause is two series on different epochs
///   (`generate_synthetic` pins its start to 2025-01-01T00:00:00Z regardless of what log is
///   also in scope);
/// * `options.warmup_samples` is at least the number of aligned pairs, which would leave
///   nothing to score.
pub fn evaluate(
    estimates: &[NavigationResult],
    truth: &[TruthSample],
    options: MetricOptions,
) -> Result<AccuracyMetrics, StrapdownError> {
    if estimates.is_empty() {
        return Err(StrapdownError::InvalidConfiguration {
            field: "estimates",
            reason: "cannot score an empty navigation solution".to_string(),
        });
    }
    if truth.is_empty() {
        return Err(StrapdownError::InvalidConfiguration {
            field: "truth",
            reason: "cannot score against an empty truth series".to_string(),
        });
    }
    if !options.max_match_gap_s.is_finite() || options.max_match_gap_s < 0.0 {
        return Err(StrapdownError::InvalidConfiguration {
            field: "max_match_gap_s",
            reason: format!(
                "must be finite and non-negative; got {}",
                options.max_match_gap_s
            ),
        });
    }

    let pairs = align(estimates, truth, options.max_match_gap_s);
    if pairs.is_empty() {
        return Err(StrapdownError::InvalidConfiguration {
            field: "truth",
            reason: epoch_mismatch_reason(estimates, truth, options.max_match_gap_s),
        });
    }
    if options.warmup_samples >= pairs.len() {
        return Err(StrapdownError::InvalidConfiguration {
            field: "warmup_samples",
            reason: format!(
                "discards all {} aligned samples; got {}",
                pairs.len(),
                options.warmup_samples
            ),
        });
    }
    let scored = &pairs[options.warmup_samples..];

    Ok(reduce(scored))
}

/// One estimate matched to its truth sample.
type Pair<'a> = (&'a NavigationResult, &'a TruthSample);

/// Match every estimate to a truth sample, in `O((n + m) log m)`.
///
/// The truth timestamps are copied out, sorted and binary-searched once per estimate, which
/// replaces the linear scan per estimate the test targets used to do -- on this crate's
/// 5,366-row log that is roughly 68,000 comparisons against 29 million.
///
/// Timestamps are compared at **millisecond** resolution on both sides, because that is the
/// resolution [`sim::run_closed_loop`](crate::sim::run_closed_loop) labels its rows at: it
/// rounds an elapsed-seconds float to whole milliseconds. Comparing at any finer resolution
/// would match nothing. Where several truth samples share a millisecond the first in sorted
/// order wins.
fn align<'a>(
    estimates: &'a [NavigationResult],
    truth: &'a [TruthSample],
    max_gap_s: f64,
) -> Vec<Pair<'a>> {
    let mut keys: Vec<(i64, usize)> = truth
        .iter()
        .enumerate()
        .map(|(i, t)| (t.timestamp.timestamp_millis(), i))
        .collect();
    keys.sort_unstable();

    let max_gap_ms = (max_gap_s * 1000.0).floor() as i64;
    let mut pairs = Vec::with_capacity(estimates.len().min(truth.len()));

    for estimate in estimates {
        let target = estimate.timestamp.timestamp_millis();
        // First entry at or after `target`. `partition_point` rather than `binary_search`
        // because the latter may return any one of a run of equal keys.
        let at_or_after = keys.partition_point(|&(ms, _)| ms < target);

        let after = keys
            .get(at_or_after)
            .map(|&(ms, idx)| ((ms - target).abs(), idx));
        let before = at_or_after
            .checked_sub(1)
            .and_then(|i| keys.get(i))
            .map(|&(ms, idx)| ((ms - target).abs(), idx));
        // Ties go to the earlier sample, which is the "first in sorted order" the doc promises.
        let best = match (before, after) {
            (Some(b), Some(a)) => Some(if b.0 <= a.0 { b } else { a }),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        };

        if let Some((distance, idx)) = best
            && distance <= max_gap_ms
            && let Some(sample) = truth.get(idx)
        {
            pairs.push((estimate, sample));
        }
    }
    pairs
}

/// The error message for a run that aligned nothing, naming both series' extents.
fn epoch_mismatch_reason(
    estimates: &[NavigationResult],
    truth: &[TruthSample],
    max_gap_s: f64,
) -> String {
    let est_first = estimates.first().map(|e| e.timestamp);
    let est_last = estimates.last().map(|e| e.timestamp);
    let truth_first = truth.first().map(|t| t.timestamp);
    let truth_last = truth.last().map(|t| t.timestamp);
    format!(
        "no estimate matched any truth sample within {max_gap_s} s. \
         Estimates span {est_first:?} to {est_last:?}; truth spans {truth_first:?} to \
         {truth_last:?}. The usual cause is two series on different epochs -- \
         `generate_synthetic` pins its start to 2025-01-01T00:00:00Z."
    )
}

/// Reduce aligned pairs to the metric set. Infallible by construction: `pairs` is non-empty.
fn reduce(pairs: &[Pair<'_>]) -> AccuracyMetrics {
    let mut horizontal = Vec::with_capacity(pairs.len());
    let mut altitude_signed = Vec::with_capacity(pairs.len());
    let mut velocity_horizontal_sq = Vec::with_capacity(pairs.len());
    let mut velocity_vertical = Vec::with_capacity(pairs.len());
    let mut roll = Vec::new();
    let mut pitch = Vec::new();
    let mut yaw = Vec::new();
    let mut geodesic = Vec::new();
    let mut npes = Vec::new();
    let mut nees = Vec::new();
    let mut horizontal_containment = Containment::default();
    let mut vertical_containment = Containment::default();
    let mut discarded = 0usize;

    for &(estimate, sample) in pairs {
        let distance = haversine_distance(
            estimate.latitude.to_radians(),
            estimate.longitude.to_radians(),
            sample.latitude_deg.to_radians(),
            sample.longitude_deg.to_radians(),
        );
        push_or_count(&mut horizontal, distance, &mut discarded);
        push_or_count(
            &mut altitude_signed,
            estimate.altitude - sample.altitude_m,
            &mut discarded,
        );

        let north = estimate.velocity_north - sample.velocity_north_mps;
        let east = estimate.velocity_east - sample.velocity_east_mps;
        push_or_count(
            &mut velocity_horizontal_sq,
            north * north + east * east,
            &mut discarded,
        );
        push_or_count(
            &mut velocity_vertical,
            estimate.velocity_vertical - sample.velocity_vertical_mps,
            &mut discarded,
        );

        if let Some(reference) = sample.attitude {
            let (reference_roll, reference_pitch, reference_yaw) = reference.euler_angles();
            push_or_count(
                &mut roll,
                wrap_to_pi(estimate.roll - reference_roll),
                &mut discarded,
            );
            push_or_count(
                &mut pitch,
                wrap_to_pi(estimate.pitch - reference_pitch),
                &mut discarded,
            );
            push_or_count(
                &mut yaw,
                wrap_to_pi(estimate.yaw - reference_yaw),
                &mut discarded,
            );
            let estimated =
                Rotation3::from_euler_angles(estimate.roll, estimate.pitch, estimate.yaw);
            push_or_count(
                &mut geodesic,
                (estimated.inverse() * reference).angle(),
                &mut discarded,
            );
        }

        // Position error in **radians** against a variance in rad^2. The state's latitude is
        // stored in degrees while its covariance is the raw filter diagonal, so the conversion
        // has to go this way round; converting the variance to degrees instead would give the
        // same number here and the wrong one the moment anyone reuses it in metres.
        let error_lat_rad = (estimate.latitude - sample.latitude_deg).to_radians();
        let error_lon_rad = (estimate.longitude - sample.longitude_deg).to_radians();
        let error_alt_m = estimate.altitude - sample.altitude_m;

        horizontal_containment.observe(error_lat_rad, estimate.latitude_cov);
        horizontal_containment.observe(error_lon_rad, estimate.longitude_cov);
        vertical_containment.observe(error_alt_m, estimate.altitude_cov);

        if let (Some(lat), Some(lon), Some(alt)) = (
            normalized_square(error_lat_rad, estimate.latitude_cov),
            normalized_square(error_lon_rad, estimate.longitude_cov),
            normalized_square(error_alt_m, estimate.altitude_cov),
        ) {
            push_or_count(&mut npes, lat + lon + alt, &mut discarded);
        }

        // The real statistic, alongside the diagonal-only form above (#376). Deliberately not
        // counted into `discarded`: that counter means "a channel this run should have scored
        // and could not", and a row whose off-diagonals were never recorded -- every row
        // written before #376, and every row from a constructor with no covariance to report
        // -- is not a discarded sample. It is a row this metric does not apply to.
        if let Some(value) = normalized_error_squared(
            [error_lat_rad, error_lon_rad, error_alt_m],
            position_block(estimate),
        ) {
            push_finite(&mut nees, value);
        }
    }

    AccuracyMetrics {
        sample_count: pairs.len(),
        discarded_channel_samples: discarded,
        horizontal_rmse_m: root_mean_square(&horizontal),
        horizontal_cep50_m: percentile(&horizontal, 0.50),
        horizontal_cep95_m: percentile(&horizontal, 0.95),
        horizontal_max_m: percentile(&horizontal, 1.0),
        vertical_rmse_m: root_mean_square(&altitude_signed),
        vertical_bias_m: mean(&altitude_signed),
        // Already squared per sample, so this is the square root of the mean rather than the
        // root mean *square*: taking the RMS of a squared quantity would be a fourth power.
        velocity_horizontal_rmse_mps: mean(&velocity_horizontal_sq).map(f64::sqrt),
        velocity_vertical_rmse_mps: root_mean_square(&velocity_vertical),
        roll_rmse_deg: root_mean_square(&roll).map(f64::to_degrees),
        pitch_rmse_deg: root_mean_square(&pitch).map(f64::to_degrees),
        yaw_rmse_deg: root_mean_square(&yaw).map(f64::to_degrees),
        attitude_geodesic_rmse_deg: root_mean_square(&geodesic).map(f64::to_degrees),
        npes_position: mean(&npes),
        nees_position: mean(&nees),
        containment_3sigma_horizontal: horizontal_containment.fraction(),
        containment_3sigma_vertical: vertical_containment.fraction(),
    }
}

/// Running count of channel-samples inside plus or minus three sigma.
#[derive(Debug, Default)]
struct Containment {
    inside: usize,
    total: usize,
}

impl Containment {
    /// Record one channel-sample, ignoring it when its variance is unusable.
    ///
    /// A non-finite or non-positive variance is not a miss, it is an absence of information:
    /// `dead_reckoning` emits `NaN` for every covariance column, and counting those as
    /// failures would report a dead-reckoning run as zero-percent contained rather than as
    /// unmeasured.
    fn observe(&mut self, error: f64, variance: f64) {
        if !variance.is_finite() || variance <= 0.0 || !error.is_finite() {
            return;
        }
        self.total += 1;
        if error.abs() <= 3.0 * variance.sqrt() {
            self.inside += 1;
        }
    }

    /// The contained fraction, or `None` when no sample carried a usable variance.
    fn fraction(&self) -> Option<f64> {
        (self.total > 0).then(|| self.inside as f64 / self.total as f64)
    }
}

/// The filter's 3x3 position covariance block, in the states' own units.
///
/// Rows and columns are (latitude, longitude, altitude), so the units are mixed: rad^2, rad^2,
/// m^2 on the diagonal and rad*m where altitude meets an angle. The error vector scored against
/// it has to be `(rad, rad, m)` to match -- which is why [`evaluate`] converts the position
/// error to radians rather than converting the covariance to degrees.
const fn position_block(estimate: &NavigationResult) -> Matrix3<f64> {
    let (lat_lon, lat_alt, lon_alt) = (
        estimate.latitude_longitude_cov,
        estimate.latitude_altitude_cov,
        estimate.longitude_altitude_cov,
    );
    Matrix3::new(
        estimate.latitude_cov,
        lat_lon,
        lat_alt,
        lat_lon,
        estimate.longitude_cov,
        lon_alt,
        lat_alt,
        lon_alt,
        estimate.altitude_cov,
    )
}

/// $e^\top P^{-1} e$, or `None` when the block cannot support the statistic.
///
/// `None` rather than a large number in every degenerate case, because each of them means "this
/// row cannot answer the question" rather than "this filter is badly wrong":
///
/// * any entry non-finite -- a row written before #376 recorded off-diagonals, or by a
///   constructor that has no covariance to report and writes `NaN`;
/// * a singular or near-singular block, which `try_inverse` declines;
/// * a negative result, which a covariance that is not positive definite can produce and which
///   is not a squared anything.
fn normalized_error_squared(error: [f64; 3], block: Matrix3<f64>) -> Option<f64> {
    if !error.iter().all(|v| v.is_finite()) || !block.iter().all(|v| v.is_finite()) {
        return None;
    }
    let inverse = block.try_inverse()?;
    let error = Vector3::new(error[0], error[1], error[2]);
    let value = (error.transpose() * inverse * error)[(0, 0)];
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// `error^2 / variance` for one channel, or `None` when the variance is unusable.
fn normalized_square(error: f64, variance: f64) -> Option<f64> {
    (variance.is_finite() && variance > 0.0 && error.is_finite()).then(|| error * error / variance)
}

/// [`push_finite`], counting a refusal into `discarded`.
fn push_or_count(into: &mut Vec<f64>, value: f64, discarded: &mut usize) {
    if !push_finite(into, value) {
        *discarded += 1;
    }
}

/// Push `value` only when it is finite, so one bad sample cannot `NaN` a whole metric.
///
/// Returns whether it pushed. The caller counts the refusals into
/// [`AccuracyMetrics::discarded_channel_samples`]: dropping a sample makes the metric it was
/// dropped from look better, and a gate that could not see that would be one a change could
/// pass by producing garbage rather than by being correct.
fn push_finite(into: &mut Vec<f64>, value: f64) -> bool {
    if value.is_finite() {
        into.push(value);
        return true;
    }
    false
}

/// Arithmetic mean, or `None` for an empty slice.
///
/// `None` rather than zero on purpose: a zero would be indistinguishable from a perfect run
/// and could be blessed as a baseline for a scenario that produced nothing at all.
fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Root mean square of a slice, or `None` when it is empty.
///
/// Public because the test targets each grew their own copy of this, which is what
/// [`crate::metrics`] exists to stop (#368). `None` rather than zero for the same reason
/// [`mean`] returns `None`: a zero is indistinguishable from a perfect run.
///
/// ```
/// use strapdown::metrics::root_mean_square;
///
/// assert_eq!(root_mean_square(&[3.0, 4.0]), Some(12.5_f64.sqrt()));
/// assert_eq!(root_mean_square(&[]), None);
/// ```
pub fn root_mean_square(values: &[f64]) -> Option<f64> {
    mean(&values.iter().map(|v| v * v).collect::<Vec<_>>()).map(f64::sqrt)
}

/// Nearest-rank percentile of `values` for `p` in `[0, 1]`, or `None` for an empty slice.
///
/// The rank is $\lceil p N \rceil$, clamped to $[1, N]$, and the value returned is the one at
/// that rank in ascending order. This is the standard nearest-rank definition: it always
/// returns an element that is actually in the sample, never an interpolation between two, so
/// `horizontal_cep50_m` is a radial error some epoch genuinely had.
///
/// It is written out because the obvious-looking alternative is a different statistic and the
/// difference is not visible from a call site. Rounding $(N-1) p$ -- the index form of the
/// linear-interpolation convention, and what this used to do -- picks the sixth of ten sorted
/// values for `p = 0.5` where nearest-rank picks the fifth.
///
/// Sorted with [`f64::total_cmp`], which is a total order, so there is no `partial_cmp` to
/// unwrap and no ordering that depends on where a `NaN` happened to sit. `p = 1.0` is the
/// maximum and `p = 0.0` the minimum.
fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let count = sorted.len();
    let rank = (p * count as f64).ceil().max(1.0) as usize;
    sorted.get(rank.min(count) - 1).copied()
}

/// Metrics keyed by their stable baseline key, for a caller writing a file or a table.
///
/// A convenience over [`AccuracyMetrics::iter`]; the ordering is the map's, not
/// [`MetricId::ALL`]'s.
#[must_use]
pub fn keyed(metrics: &AccuracyMetrics) -> BTreeMap<&'static str, Option<f64>> {
    metrics
        .iter()
        .map(|(id, value)| (id.key(), value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// A navigation row at `t0 + seconds`, otherwise at the origin with unit covariance.
    fn estimate_at(seconds: i64) -> NavigationResult {
        NavigationResult {
            timestamp: epoch() + chrono::Duration::seconds(seconds),
            ..NavigationResult::new()
        }
    }

    /// The epoch `generate_synthetic` pins itself to, so the tests share one clock.
    fn epoch() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
            .single()
            .unwrap_or_else(Utc::now)
    }

    /// A truth sample at `t0 + seconds`, at the origin and level.
    fn truth_at(seconds: i64) -> TruthSample {
        TruthSample {
            timestamp: epoch() + chrono::Duration::seconds(seconds),
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            altitude_m: 0.0,
            velocity_north_mps: 0.0,
            velocity_east_mps: 0.0,
            velocity_vertical_mps: 0.0,
            attitude: Some(Rotation3::identity()),
        }
    }

    #[test]
    fn metric_keys_are_unique_and_cover_every_variant() {
        let keys: std::collections::BTreeSet<_> = MetricId::ALL.iter().map(|m| m.key()).collect();
        assert_eq!(keys.len(), MetricId::ALL.len());
        // `AccuracyMetrics::get` must answer for every metric, or `iter` silently drops one.
        let metrics = AccuracyMetrics::default();
        for id in MetricId::ALL {
            assert!(
                metrics.get(*id).is_none(),
                "{} defaulted to a value",
                id.key()
            );
        }
        assert_eq!(metrics.iter().count(), MetricId::ALL.len());
    }

    #[test]
    fn metric_ids_round_trip_through_serde() {
        for id in MetricId::ALL {
            let json = serde_json::to_string(id).expect("serialize");
            let back: MetricId = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*id, back);
        }
    }

    /// The units trap this module exists to settle: position is degrees, covariance is rad^2.
    ///
    /// A latitude error of exactly one sigma must normalize to exactly 1.0 per channel, so the
    /// three-channel sum is 3.0 -- the value a consistent filter scores.
    #[test]
    fn a_one_sigma_error_normalizes_to_one_per_channel() {
        let sigma_deg = 1e-5_f64;
        let sigma_rad = sigma_deg.to_radians();
        let sigma_alt_m = 2.0;

        let mut estimate = estimate_at(0);
        estimate.latitude = sigma_deg;
        estimate.longitude = sigma_deg;
        estimate.altitude = sigma_alt_m;
        estimate.latitude_cov = sigma_rad * sigma_rad;
        estimate.longitude_cov = sigma_rad * sigma_rad;
        estimate.altitude_cov = sigma_alt_m * sigma_alt_m;

        let metrics = evaluate(&[estimate], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        let npes = metrics.npes_position.expect("covariance is usable");
        assert!(
            (npes - 3.0).abs() < 1e-9,
            "expected 3.0 for a one-sigma error in each of three channels, got {npes}"
        );
        // One sigma is inside three sigma, so containment is total.
        assert_eq!(metrics.containment_3sigma_horizontal, Some(1.0));
        assert_eq!(metrics.containment_3sigma_vertical, Some(1.0));
    }

    /// With an uncorrelated position block the two consistency metrics must agree exactly.
    ///
    /// That is the condition under which `npes_position` is the NEES rather than a stand-in
    /// for it, so anything else here would mean one of the two is computing the wrong thing.
    #[test]
    fn nees_equals_npes_when_the_position_block_is_diagonal() {
        let sigma_deg = 1e-5_f64;
        let sigma_rad = sigma_deg.to_radians();

        let mut estimate = estimate_at(0);
        estimate.latitude = sigma_deg;
        estimate.longitude = sigma_deg;
        estimate.altitude = 2.0;
        estimate.latitude_cov = sigma_rad * sigma_rad;
        estimate.longitude_cov = sigma_rad * sigma_rad;
        estimate.altitude_cov = 4.0;
        estimate.latitude_longitude_cov = 0.0;
        estimate.latitude_altitude_cov = 0.0;
        estimate.longitude_altitude_cov = 0.0;

        let metrics = evaluate(&[estimate], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        let npes = metrics.npes_position.expect("npes");
        let nees = metrics.nees_position.expect("nees");
        assert!(
            (npes - nees).abs() < 1e-9,
            "diagonal block: npes {npes} and nees {nees} should agree"
        );
        assert!((nees - POSITION_DEGREES_OF_FREEDOM).abs() < 1e-9);
    }

    /// Correlated position states make `npes_position` **optimistic**, and the NEES is what
    /// says so.
    ///
    /// This is the whole reason #376 exists. With latitude and longitude correlated at 0.9 and
    /// an error along the correlated direction, the diagonal-only form divides each component
    /// by its own variance and reports a comfortable number; inverting the real block shows the
    /// error is far less likely than that. A filter is not consistent because the metric that
    /// cannot see its correlations says so.
    #[test]
    fn nees_exceeds_npes_when_the_position_states_are_correlated() {
        let sigma_deg = 1e-5_f64;
        let sigma_rad = sigma_deg.to_radians();
        let variance = sigma_rad * sigma_rad;
        let correlation = 0.9;

        let mut estimate = estimate_at(0);
        // A one-sigma error in latitude and minus one sigma in longitude: against a block that
        // says the two move *together*, that is a very unlikely place to be.
        estimate.latitude = sigma_deg;
        estimate.longitude = -sigma_deg;
        estimate.altitude = 0.0;
        estimate.latitude_cov = variance;
        estimate.longitude_cov = variance;
        estimate.altitude_cov = 1.0;
        estimate.latitude_longitude_cov = correlation * variance;
        estimate.latitude_altitude_cov = 0.0;
        estimate.longitude_altitude_cov = 0.0;

        let metrics = evaluate(&[estimate], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        let npes = metrics.npes_position.expect("npes");
        let nees = metrics.nees_position.expect("nees");

        // The diagonal form sees two one-sigma errors and no altitude error: exactly 2.0.
        assert!((npes - 2.0).abs() < 1e-9, "npes {npes}");
        // The real block: e^T P^-1 e for an anti-correlated error is 2/(1 - rho) = 20.
        let expected = 2.0 / (1.0 - correlation);
        assert!(
            (nees - expected).abs() < 1e-6,
            "nees {nees} should be {expected} for rho = {correlation}"
        );
        assert!(
            nees > npes,
            "npes {npes} must understate the real {nees}: that is what makes it optimistic"
        );
    }

    /// A row with no recorded off-diagonals yields no NEES, and does not count as a discard.
    ///
    /// Every row written before #376 is such a row, as is every row from a constructor with no
    /// covariance to report. `None` is the honest answer -- "this metric does not apply here"
    /// -- and it must not inflate `discarded_channel_samples`, which means something else:
    /// a channel this run should have scored and could not.
    #[test]
    fn nees_is_absent_rather_than_discarded_when_the_block_is_unrecorded() {
        let sigma_rad = 1e-5_f64.to_radians();
        let mut estimate = estimate_at(0);
        estimate.latitude = 1e-5;
        estimate.latitude_cov = sigma_rad * sigma_rad;
        estimate.longitude_cov = sigma_rad * sigma_rad;
        estimate.altitude_cov = 1.0;
        estimate.latitude_longitude_cov = f64::NAN;
        estimate.latitude_altitude_cov = f64::NAN;
        estimate.longitude_altitude_cov = f64::NAN;

        let metrics = evaluate(&[estimate], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        assert!(metrics.nees_position.is_none(), "NaN block must not score");
        assert!(metrics.npes_position.is_some(), "npes is unaffected");
        assert_eq!(
            metrics.discarded_channel_samples, 0,
            "an inapplicable metric is not a discarded sample"
        );
    }

    /// A singular position block declines rather than producing an infinity.
    #[test]
    fn nees_declines_a_singular_position_block() {
        let mut estimate = estimate_at(0);
        estimate.latitude = 1e-5;
        estimate.latitude_cov = 0.0;
        estimate.longitude_cov = 0.0;
        estimate.altitude_cov = 0.0;
        estimate.latitude_longitude_cov = 0.0;
        estimate.latitude_altitude_cov = 0.0;
        estimate.longitude_altitude_cov = 0.0;

        let metrics = evaluate(&[estimate], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        assert!(metrics.nees_position.is_none());
    }

    #[test]
    fn a_four_sigma_error_falls_outside_the_three_sigma_band() {
        let sigma_rad = 1e-5_f64.to_radians();
        let mut estimate = estimate_at(0);
        estimate.latitude = 4.0 * 1e-5;
        estimate.latitude_cov = sigma_rad * sigma_rad;
        estimate.longitude_cov = sigma_rad * sigma_rad;
        estimate.altitude_cov = 1.0;

        let metrics = evaluate(&[estimate], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        // Two horizontal channel-samples, one of which is out of band.
        assert_eq!(metrics.containment_3sigma_horizontal, Some(0.5));
    }

    /// Attitude error is modulo a full turn: +179 against -179 is 2 degrees, not 358.
    #[test]
    fn yaw_error_wraps_across_the_branch_cut() {
        let mut estimate = estimate_at(0);
        estimate.yaw = 179.0_f64.to_radians();

        let mut sample = truth_at(0);
        sample.attitude = Some(Rotation3::from_euler_angles(
            0.0,
            0.0,
            -179.0_f64.to_radians(),
        ));

        let metrics =
            evaluate(&[estimate], &[sample], MetricOptions::default()).expect("one aligned pair");
        let yaw = metrics.yaw_rmse_deg.expect("attitude present");
        assert!(
            (yaw - 2.0).abs() < 1e-6,
            "expected 2 deg across the branch cut, got {yaw}"
        );
    }

    /// Dead reckoning emits `NaN` for every covariance column; that is unmeasured, not failed.
    #[test]
    fn an_all_nan_covariance_leaves_the_consistency_metrics_unmeasured() {
        let mut estimate = estimate_at(0);
        estimate.latitude_cov = f64::NAN;
        estimate.longitude_cov = f64::NAN;
        estimate.altitude_cov = f64::NAN;

        let metrics = evaluate(&[estimate], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        assert_eq!(metrics.npes_position, None);
        assert_eq!(metrics.containment_3sigma_horizontal, None);
        assert_eq!(metrics.containment_3sigma_vertical, None);
        // The accuracy metrics are unaffected: a missing covariance is not a missing position.
        assert_eq!(metrics.horizontal_rmse_m, Some(0.0));
        assert_eq!(metrics.sample_count, 1);
    }

    #[test]
    fn empty_input_is_an_error_rather_than_a_zero_metric() {
        let err = evaluate(&[], &[truth_at(0)], MetricOptions::default()).unwrap_err();
        assert!(matches!(
            err,
            StrapdownError::InvalidConfiguration {
                field: "estimates",
                ..
            }
        ));
        let err = evaluate(&[estimate_at(0)], &[], MetricOptions::default()).unwrap_err();
        assert!(matches!(
            err,
            StrapdownError::InvalidConfiguration { field: "truth", .. }
        ));
    }

    #[test]
    fn a_negative_match_gap_is_rejected() {
        let options = MetricOptions {
            max_match_gap_s: -1.0,
            warmup_samples: 0,
        };
        let err = evaluate(&[estimate_at(0)], &[truth_at(0)], options).unwrap_err();
        assert!(matches!(
            err,
            StrapdownError::InvalidConfiguration {
                field: "max_match_gap_s",
                ..
            }
        ));
    }

    /// Two series on different epochs is the likeliest misuse, so the message has to say so.
    #[test]
    fn zero_overlap_names_both_epochs() {
        let far_away = TruthSample {
            timestamp: epoch() + chrono::Duration::days(365),
            ..truth_at(0)
        };
        let err = evaluate(&[estimate_at(0)], &[far_away], MetricOptions::default()).unwrap_err();
        let StrapdownError::InvalidConfiguration { reason, .. } = err else {
            panic!("expected an InvalidConfiguration");
        };
        assert!(reason.contains("no estimate matched"), "{reason}");
        assert!(reason.contains("2025-01-01"), "{reason}");
    }

    #[test]
    fn warmup_cannot_discard_every_sample() {
        let options = MetricOptions {
            max_match_gap_s: 0.0,
            warmup_samples: 1,
        };
        let err = evaluate(&[estimate_at(0)], &[truth_at(0)], options).unwrap_err();
        assert!(matches!(
            err,
            StrapdownError::InvalidConfiguration {
                field: "warmup_samples",
                ..
            }
        ));
    }

    #[test]
    fn warmup_drops_the_leading_samples() {
        let estimates: Vec<_> = (0..4).map(estimate_at).collect();
        let truth: Vec<_> = (0..4).map(truth_at).collect();
        let options = MetricOptions {
            max_match_gap_s: 0.0,
            warmup_samples: 2,
        };
        let metrics = evaluate(&estimates, &truth, options).expect("two samples survive");
        assert_eq!(metrics.sample_count, 2);
    }

    /// Only the estimates with a truth partner are scored; the rest are ignored, not zeroed.
    #[test]
    fn unmatched_estimates_are_dropped_from_the_sample_count() {
        let estimates: Vec<_> = (0..10).map(estimate_at).collect();
        let truth: Vec<_> = [0, 2, 4].into_iter().map(truth_at).collect();
        let metrics =
            evaluate(&estimates, &truth, MetricOptions::default()).expect("three aligned pairs");
        assert_eq!(metrics.sample_count, 3);
    }

    #[test]
    fn a_match_gap_admits_neighbouring_timestamps() {
        let mut estimate = estimate_at(0);
        estimate.timestamp = epoch() + chrono::Duration::milliseconds(400);
        let options = MetricOptions {
            max_match_gap_s: 0.5,
            warmup_samples: 0,
        };
        assert_eq!(
            evaluate(&[estimate.clone()], &[truth_at(0)], options)
                .expect("within the gap")
                .sample_count,
            1
        );
        // The same pair is not a match at the default exact-millisecond rule.
        assert!(evaluate(&[estimate], &[truth_at(0)], MetricOptions::default()).is_err());
    }

    #[test]
    fn percentiles_handle_degenerate_slices() {
        assert_eq!(percentile(&[], 0.5), None);
        assert_eq!(percentile(&[7.0], 0.0), Some(7.0));
        assert_eq!(percentile(&[7.0], 0.5), Some(7.0));
        assert_eq!(percentile(&[7.0], 1.0), Some(7.0));
        assert_eq!(percentile(&[1.0, 2.0], 1.0), Some(2.0));
        assert_eq!(percentile(&[3.0, 3.0, 3.0], 0.95), Some(3.0));
        // Out of range is `None` rather than a clamped answer.
        assert_eq!(percentile(&[1.0], 1.5), None);
    }

    /// The rank is `ceil(p N)` clamped to `[1, N]`, not the rounded `(N - 1) p` of the
    /// interpolating convention. On ten values those differ by one position at the median,
    /// which is exactly the sort of silent disagreement a named convention exists to prevent.
    #[test]
    fn percentiles_use_the_nearest_rank_convention() {
        let values: Vec<f64> = (0..10).map(f64::from).collect();
        assert_eq!(percentile(&values, 0.0), Some(0.0));
        assert_eq!(percentile(&values, 0.5), Some(4.0));
        assert_eq!(percentile(&values, 0.95), Some(9.0));
        assert_eq!(percentile(&values, 1.0), Some(9.0));
        // Every answer is an element of the sample, never an interpolation between two.
        for p in [0.0, 0.1, 0.33, 0.5, 0.9, 0.95, 1.0] {
            let picked = percentile(&values, p).expect("non-empty");
            assert!(values.contains(&picked), "p={p} produced {picked}");
        }
    }

    /// An unusable quaternion is an absent attitude, not a level-and-north-facing one.
    #[test]
    fn a_record_without_a_usable_quaternion_supplies_no_truth_attitude() {
        let base = TestDataRecord {
            time: epoch(),
            latitude: 40.0,
            longitude: -75.0,
            altitude: 100.0,
            qw: 1.0,
            ..TestDataRecord::default()
        };
        assert!(
            truth_from_records(std::slice::from_ref(&base))[0]
                .attitude
                .is_some()
        );

        let all_zero = TestDataRecord {
            qw: 0.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            ..base
        };
        assert_eq!(truth_from_records(&[all_zero])[0].attitude, None);

        let not_finite = TestDataRecord {
            qw: f64::NAN,
            ..base
        };
        assert_eq!(truth_from_records(&[not_finite])[0].attitude, None);
    }

    /// A sample dropped as non-finite is counted, because dropping it flatters the metric.
    #[test]
    fn non_finite_channel_samples_are_counted_not_hidden() {
        let clean = evaluate(&[estimate_at(0)], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        assert_eq!(clean.discarded_channel_samples, 0);

        let mut broken = estimate_at(0);
        broken.altitude = f64::NAN;
        let scored = evaluate(&[broken], &[truth_at(0)], MetricOptions::default())
            .expect("one aligned pair");
        // The pair still aligned, so the count is unchanged -- which is the whole reason the
        // discards have to be reported separately.
        assert_eq!(scored.sample_count, 1);
        assert!(
            scored.discarded_channel_samples > 0,
            "a NaN altitude must be visible somewhere"
        );
        assert_eq!(scored.vertical_rmse_m, None);
    }

    #[test]
    fn mean_and_rms_are_none_on_an_empty_slice() {
        assert_eq!(mean(&[]), None);
        assert_eq!(root_mean_square(&[]), None);
        assert_eq!(root_mean_square(&[3.0, -4.0]), Some(12.5_f64.sqrt()));
        // The signed mean is what separates a bias from a spread.
        assert_eq!(mean(&[3.0, -3.0]), Some(0.0));
    }

    #[test]
    fn truth_from_records_drops_records_without_a_fix() {
        let good = TestDataRecord {
            time: epoch(),
            latitude: 40.0,
            longitude: -75.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 90.0,
            ..TestDataRecord::default()
        };

        let mut bad = good.clone();
        bad.latitude = f64::NAN;

        let truth = truth_from_records(&[good, bad]);
        assert_eq!(truth.len(), 1);
        // Bearing is degrees on the record; 90 deg is due east.
        assert!(truth[0].velocity_north_mps.abs() < 1e-9);
        assert!((truth[0].velocity_east_mps - 10.0).abs() < 1e-9);
    }

    #[test]
    fn truth_from_trajectory_drops_non_finite_rows() {
        let good = estimate_at(0);
        let mut bad = estimate_at(1);
        bad.altitude = f64::INFINITY;
        assert_eq!(truth_from_trajectory(&[good, bad]).len(), 1);
    }

    #[test]
    fn keyed_exposes_every_metric_by_its_baseline_key() {
        let keyed = keyed(&AccuracyMetrics::default());
        assert_eq!(keyed.len(), MetricId::ALL.len());
        assert!(keyed.contains_key("horizontal_rmse_m"));
        assert!(keyed.contains_key("containment_3sigma_vertical"));
    }
}
