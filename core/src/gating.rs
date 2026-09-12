//! Innovation gating for measurement outlier rejection.
//!
//! A Kalman-family update trusts its measurement unconditionally: the correction
//! $K\nu$ is applied whatever the innovation $\nu$ happens to be. That is the right
//! behaviour for a measurement whose error really is zero-mean Gaussian with
//! covariance $R$, and the wrong behaviour for a GNSS fix corrupted by multipath,
//! spoofing or a momentarily degraded constellation, where a single fix can be
//! hundreds of metres off and drags the whole state with it.
//!
//! The standard defence is a statistical consistency test on the innovation itself.
//! Under the filter's own hypothesis the innovation is distributed
//! $\nu \sim \mathcal{N}(0, S)$ with $S = HPH^\top + R$, so the **normalized
//! innovation squared** (NIS), also called the squared Mahalanobis distance,
//!
//! $$
//! d^2 = \nu^\top S^{-1} \nu
//! $$
//!
//! is distributed $\chi^2$ with as many degrees of freedom as the measurement has
//! components. Rejecting an update whose NIS exceeds the $\chi^2$ quantile for a
//! chosen confidence level discards the tail the filter's own model says should
//! almost never occur, while accepting ordinary noisy fixes.
//!
//! # What this module provides
//!
//! - [`normalized_innovation_squared`] -- the statistic itself.
//! - [`InnovationGate`] -- the accept/reject policy, either a $\chi^2$ quantile
//!   evaluated at the measurement's own degrees of freedom or a fixed threshold.
//! - [`UpdateOutcome`] -- what every [`NavigationFilter::update`] now returns, so a
//!   caller can log rejections and feed a real NIS to
//!   [`sim::health::HealthMonitor`](crate::sim::health::HealthMonitor).
//! - [`chi_squared_cdf`] and [`chi_squared_quantile`] -- the distribution functions
//!   the gate is built on, public because a test or a report generator needs them.
//!
//! # Gating is not a substitute for health monitoring
//!
//! A gate rejects *individual* measurements. A filter whose state has genuinely
//! diverged produces a large NIS on every fix, and gating each one in turn
//! silently degrades the run to dead reckoning -- it looks healthy from the
//! outside because no error is ever returned. The consecutive-exceedance counter in
//! [`HealthLimits`](crate::sim::health::HealthLimits) is the circuit breaker for
//! that case, and it is why the NIS is reported out of `update` rather than merely
//! consumed inside it.
//!
//! # References
//!
//! - Bar-Shalom, Y., Li, X.-R., Kirubarajan, T., *Estimation with Applications to
//!   Tracking and Navigation*, Section 5.4 (filter consistency, NIS).
//! - Groves, P. D., *Principles of GNSS, Inertial, and Multisensor Integrated
//!   Navigation Systems*, 2nd ed., Section 17.3 (integrity monitoring).
//!
//! [`NavigationFilter::update`]: crate::NavigationFilter::update

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::StrapdownError;
use crate::linalg::{robust_spd_solve, symmetrize};

/// Default confidence level for [`InnovationGate::ChiSquared`].
///
/// At 3 degrees of freedom (a GNSS position fix) this is a threshold of ~16.3,
/// so roughly one valid fix in a thousand is rejected. That is deliberately far
/// out in the tail: the cost of rejecting a good fix is one missed correction,
/// whereas the cost of accepting a multipath fix is a state error that the filter
/// then has to unlearn.
pub const DEFAULT_GATE_CONFIDENCE: f64 = 0.999;

/// Outcome of a single measurement update.
///
/// Returned by [`NavigationFilter::update`](crate::NavigationFilter::update) so
/// callers can distinguish "the correction was applied" from "the measurement was
/// gated out", and can log or monitor the statistic that decided it. An update
/// that is gated out leaves the state and covariance untouched; it is not an
/// error, because the filter behaved exactly as configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdateOutcome {
    /// Normalized innovation squared, $\nu^\top S^{-1} \nu$.
    pub nis: f64,
    /// Degrees of freedom of the test, i.e. the measurement dimension.
    pub dof: usize,
    /// Whether the correction was applied to the state.
    pub accepted: bool,
}

impl UpdateOutcome {
    /// An update whose correction was applied.
    #[must_use]
    pub const fn accepted(nis: f64, dof: usize) -> Self {
        Self {
            nis,
            dof,
            accepted: true,
        }
    }

    /// An update the gate rejected; the state is unchanged.
    #[must_use]
    pub const fn rejected(nis: f64, dof: usize) -> Self {
        Self {
            nis,
            dof,
            accepted: false,
        }
    }
}

/// Accept/reject policy for a measurement update.
///
/// The variants differ only in where the threshold comes from. Both are compared
/// against the same statistic, and the comparison is inclusive: a NIS exactly at
/// the threshold is accepted.
///
/// # Example
///
/// ```rust
/// use strapdown::gating::InnovationGate;
///
/// // Reject the worst 0.1% of fixes the filter's own model predicts.
/// let gate = InnovationGate::chi_squared(0.999).unwrap();
///
/// // A 3-DOF position fix is tested against chi^2(3) at 0.999.
/// assert!(gate.accepts(10.0, 3));
/// assert!(!gate.accepts(50.0, 3));
///
/// // The same gate is looser for a 6-DOF position+velocity fix, because the
/// // statistic itself is larger for more components.
/// assert!(gate.threshold(6) > gate.threshold(3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InnovationGate {
    /// Threshold is the `confidence` quantile of $\chi^2$ evaluated at the
    /// measurement's own degrees of freedom.
    ///
    /// This is the variant to prefer: a filter mixing 1-DOF barometric altitude,
    /// 3-DOF position and 6-DOF position+velocity aiding needs a different
    /// threshold for each, and deriving it from the measurement dimension is the
    /// only way one configured number stays meaningful across all of them.
    ChiSquared {
        /// Probability mass retained, in $(0, 1)$.
        confidence: f64,
    },
    /// Fixed threshold, applied regardless of the measurement dimension.
    ///
    /// Useful for reproducing a published configuration or for a deployment with a
    /// single measurement type, where the DOF never varies.
    Fixed {
        /// Maximum accepted NIS.
        threshold: f64,
    },
}

impl Default for InnovationGate {
    fn default() -> Self {
        Self::ChiSquared {
            confidence: DEFAULT_GATE_CONFIDENCE,
        }
    }
}

impl InnovationGate {
    /// Build a $\chi^2$ gate at `confidence`.
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `confidence` is not a finite value strictly
    /// inside $(0, 1)$. A confidence of 1 would accept everything and a confidence of
    /// 0 would reject everything; both are almost certainly a units mistake (a
    /// percentage rather than a probability) and are worth refusing loudly.
    pub fn chi_squared(confidence: f64) -> Result<Self, StrapdownError> {
        if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
            return Err(StrapdownError::OutOfRange {
                what: "innovation gate confidence",
                value: confidence,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(Self::ChiSquared { confidence })
    }

    /// Build a gate with a fixed NIS threshold.
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `threshold` is not finite and positive.
    pub fn fixed(threshold: f64) -> Result<Self, StrapdownError> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(StrapdownError::OutOfRange {
                what: "innovation gate threshold",
                value: threshold,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        Ok(Self::Fixed { threshold })
    }

    /// The NIS above which a measurement of `dof` components is rejected.
    ///
    /// Out-of-range configurations degrade to "accept everything"
    /// (`f64::INFINITY`) rather than panicking, so a gate built by deserializing a
    /// hand-edited config cannot take a run down mid-flight. The constructors
    /// above reject those values up front, which is where the complaint belongs.
    #[must_use]
    pub fn threshold(&self, dof: usize) -> f64 {
        match *self {
            Self::ChiSquared { confidence } => chi_squared_quantile(confidence, dof),
            Self::Fixed { threshold } => {
                if threshold.is_finite() && threshold > 0.0 {
                    threshold
                } else {
                    f64::INFINITY
                }
            }
        }
    }

    /// Whether a measurement with this `nis` and `dof` should be applied.
    ///
    /// A non-finite NIS is always rejected: it means the innovation covariance was
    /// unusable, and applying a correction derived from it would put NaN into the
    /// state.
    #[must_use]
    pub fn accepts(&self, nis: f64, dof: usize) -> bool {
        nis.is_finite() && nis >= 0.0 && nis <= self.threshold(dof)
    }
}

/// Normalized innovation squared, $d^2 = \nu^\top S^{-1} \nu$.
///
/// `innovation_covariance` is symmetrized before the solve, because $S$ is formed
/// as `H * P * H^T + R` and accumulates asymmetry of order the rounding error,
/// which a Cholesky factorization is entitled to object to.
///
/// # Errors
/// - [`StrapdownError::DimensionMismatch`] if `innovation_covariance` is not square
///   with side equal to the length of `innovation`.
/// - [`StrapdownError::SingularMatrix`] if $S$ cannot be factored or inverted. A
///   caller that wants to continue should treat this the same way it treats a
///   failed gate: skip the measurement.
///
/// # Example
///
/// ```rust
/// use nalgebra::{DMatrix, DVector};
/// use strapdown::gating::normalized_innovation_squared;
///
/// // A 2 sigma innovation in each of two independent components: d^2 = 4 + 4.
/// let innovation = DVector::from_vec(vec![2.0, 6.0]);
/// let s = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 9.0]));
/// let nis = normalized_innovation_squared(&innovation, &s).unwrap();
/// assert!((nis - 8.0).abs() < 1e-12);
/// ```
pub fn normalized_innovation_squared(
    innovation: &DVector<f64>,
    innovation_covariance: &DMatrix<f64>,
) -> Result<f64, StrapdownError> {
    let dim = innovation.len();
    if innovation_covariance.nrows() != dim || innovation_covariance.ncols() != dim {
        return Err(StrapdownError::DimensionMismatch {
            what: "innovation covariance",
            expected: dim,
            got: innovation_covariance
                .nrows()
                .max(innovation_covariance.ncols()),
        });
    }
    // Solve S y = nu rather than forming S^-1: one triangular solve instead of a
    // full inverse, and it keeps the conditioning of the result tied to S itself.
    let rhs = DMatrix::from_columns(std::slice::from_ref(innovation));
    let solved = robust_spd_solve(&symmetrize(innovation_covariance), &rhs)?;
    Ok(innovation.dot(&solved.column(0)))
}

/// Cumulative distribution function of the $\chi^2$ distribution.
///
/// Returns $P(X \le x)$ for $X \sim \chi^2_k$. Evaluates to 0 for non-positive `x`
/// and for `dof == 0`, which has no density.
///
/// # Example
///
/// ```rust
/// use strapdown::gating::chi_squared_cdf;
///
/// // The median of chi^2(1) is ~0.4549.
/// assert!((chi_squared_cdf(0.4549, 1) - 0.5).abs() < 1e-4);
/// // chi^2(2) is Exponential(1/2), so its CDF is 1 - exp(-x/2) exactly.
/// assert!((chi_squared_cdf(3.0, 2) - (1.0 - (-1.5f64).exp())).abs() < 1e-12);
/// ```
#[must_use]
pub fn chi_squared_cdf(x: f64, dof: usize) -> f64 {
    if dof == 0 || !x.is_finite() || x <= 0.0 {
        return if x.is_infinite() && x > 0.0 { 1.0 } else { 0.0 };
    }
    regularized_lower_gamma(dof as f64 / 2.0, x / 2.0)
}

/// Inverse CDF (quantile function) of the $\chi^2$ distribution.
///
/// Returns the smallest $x$ with $P(X \le x) \ge p$ for $X \sim \chi^2_k$.
/// Returns 0 for `p <= 0` or `dof == 0`, and `f64::INFINITY` for `p >= 1` or a
/// non-finite `p` -- an infinite threshold accepts every measurement, which is the
/// safe direction to fail for a gate.
///
/// The root is bracketed by doubling and then refined by bisection against
/// [`chi_squared_cdf`]. That is slower than a closed-form approximation and exact
/// to within the CDF's own accuracy, which matters more here: the threshold is
/// compared against a statistic that decides whether a fix is used at all, and
/// this runs once per measurement update (order 1 Hz) against a state update that
/// costs more.
///
/// # Example
///
/// ```rust
/// use strapdown::gating::chi_squared_quantile;
///
/// // Standard table values.
/// assert!((chi_squared_quantile(0.95, 3) - 7.8147).abs() < 1e-3);
/// assert!((chi_squared_quantile(0.999, 3) - 16.2662).abs() < 1e-3);
/// ```
#[must_use]
pub fn chi_squared_quantile(p: f64, dof: usize) -> f64 {
    if dof == 0 || (p.is_finite() && p <= 0.0) {
        return 0.0;
    }
    if !p.is_finite() || p >= 1.0 {
        return f64::INFINITY;
    }

    // Bracket: grow the upper bound until the CDF passes p. The mean of chi^2(k)
    // is k, so starting there needs only a handful of doublings even at p = 1-1e-12.
    let mut high = (dof as f64).max(1.0);
    let mut bracketed = false;
    for _ in 0..MAX_BRACKET_DOUBLINGS {
        if chi_squared_cdf(high, dof) >= p {
            bracketed = true;
            break;
        }
        high *= 2.0;
    }
    if !bracketed {
        return f64::INFINITY;
    }
    let mut low = 0.0;

    // Bisection. Each step halves the interval, so BISECTION_ITERATIONS steps take
    // an interval of width `high` down to `high * 2^-100`, i.e. below the f64
    // resolution of any threshold this is asked for; the relative-width test
    // normally stops it long before that.
    for _ in 0..BISECTION_ITERATIONS {
        let mid = 0.5 * (low + high);
        if mid <= low || mid >= high {
            break;
        }
        if chi_squared_cdf(mid, dof) < p {
            low = mid;
        } else {
            high = mid;
        }
        if high - low <= QUANTILE_RELATIVE_TOLERANCE * high {
            break;
        }
    }
    high
}

/// Maximum doublings used to bracket a quantile before giving up.
///
/// Each doubling multiplies the bound by 2 starting from the distribution mean, so
/// 64 of them reaches ~1e19 times the mean -- far past any finite quantile.
const MAX_BRACKET_DOUBLINGS: usize = 64;
/// Bisection steps used to refine a bracketed quantile.
const BISECTION_ITERATIONS: usize = 100;
/// Relative interval width at which bisection stops early.
const QUANTILE_RELATIVE_TOLERANCE: f64 = 1e-12;
/// Convergence tolerance for the incomplete-gamma series and continued fraction.
const GAMMA_TOLERANCE: f64 = 1e-15;
/// Iteration cap for the incomplete-gamma series and continued fraction.
///
/// Both converge in tens of terms over the domain reached here; the cap exists so
/// a pathological argument cannot spin rather than because it is expected to bind.
const GAMMA_MAX_ITERATIONS: usize = 1000;
/// Floor used by the modified Lentz algorithm to step over a zero denominator.
const LENTZ_TINY: f64 = 1e-300;

/// Lanczos parameter `g` paired with [`LANCZOS_COEFFICIENTS`].
const LANCZOS_G: f64 = 7.0;
/// Lanczos series coefficients for `g = 7`, giving ~15 significant digits.
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural logarithm of the gamma function, via the Lanczos approximation.
///
/// Only ever called here with `x = dof / 2 >= 0.5`, but the reflection formula is
/// kept so the function is correct as written rather than correct by caller
/// convention.
fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x).
        return PI.ln() - (PI * x).sin().abs().ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut series = LANCZOS_COEFFICIENTS[0];
    for (i, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        series += coefficient / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// Regularized lower incomplete gamma function $P(a, x)$.
///
/// Uses the ascending series where it converges quickly (`x < a + 1`) and the
/// continued fraction for the complement elsewhere, the standard split: the series
/// converges slowly in the far tail and the continued fraction converges slowly
/// near the origin.
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        lower_gamma_series(a, x)
    } else {
        (1.0 - upper_gamma_continued_fraction(a, x)).clamp(0.0, 1.0)
    }
}

/// Ascending series for $P(a, x)$, valid for `x < a + 1`.
fn lower_gamma_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut denominator = a;
    for _ in 0..GAMMA_MAX_ITERATIONS {
        denominator += 1.0;
        term *= x / denominator;
        sum += term;
        if term.abs() < sum.abs() * GAMMA_TOLERANCE {
            break;
        }
    }
    (sum * (a * x.ln() - x - ln_gamma(a)).exp()).clamp(0.0, 1.0)
}

/// Modified-Lentz continued fraction for $Q(a, x) = 1 - P(a, x)$, valid for
/// `x >= a + 1`.
fn upper_gamma_continued_fraction(shape: f64, x: f64) -> f64 {
    // Modified Lentz: `b_term` is the continued fraction's b_i, `numerator` its a_i, and
    // `c_term`/`d_term` the two running ratios. Spelled out rather than kept as the
    // published single letters, which the project's naming convention rules out.
    let mut b_term = x + 1.0 - shape;
    let mut c_term = 1.0 / LENTZ_TINY;
    let mut d_term = if b_term.abs() < LENTZ_TINY {
        1.0 / LENTZ_TINY
    } else {
        1.0 / b_term
    };
    let mut fraction = d_term;
    for i in 1..=GAMMA_MAX_ITERATIONS {
        let numerator = -(i as f64) * (i as f64 - shape);
        b_term += 2.0;
        d_term = numerator * d_term + b_term;
        if d_term.abs() < LENTZ_TINY {
            d_term = LENTZ_TINY;
        }
        c_term = b_term + numerator / c_term;
        if c_term.abs() < LENTZ_TINY {
            c_term = LENTZ_TINY;
        }
        d_term = 1.0 / d_term;
        let delta = d_term * c_term;
        fraction *= delta;
        if (delta - 1.0).abs() < GAMMA_TOLERANCE {
            break;
        }
    }
    (fraction * (shape * x.ln() - x - ln_gamma(shape)).exp()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    /// Textbook chi-squared critical values, `(dof, probability, quantile)`.
    ///
    /// Taken from standard statistical tables rather than regenerated from this
    /// module -- a self-consistency check would pass just as happily on a wrong
    /// implementation.
    const CHI_SQUARED_TABLE: [(usize, f64, f64); 12] = [
        (1, 0.950, 3.841),
        (1, 0.990, 6.635),
        (1, 0.999, 10.828),
        (2, 0.950, 5.991),
        (2, 0.990, 9.210),
        (3, 0.950, 7.815),
        (3, 0.990, 11.345),
        (3, 0.999, 16.266),
        (5, 0.950, 11.070),
        (6, 0.950, 12.592),
        (6, 0.999, 22.458),
        (10, 0.950, 18.307),
    ];

    #[test]
    fn quantile_matches_published_tables() {
        for (dof, probability, expected) in CHI_SQUARED_TABLE {
            let actual = chi_squared_quantile(probability, dof);
            assert!(
                (actual - expected).abs() < 1e-3,
                "chi2_quantile({probability}, {dof}) = {actual}, table says {expected}"
            );
        }
    }

    #[test]
    fn cdf_matches_published_tables() {
        for (dof, probability, quantile) in CHI_SQUARED_TABLE {
            let actual = chi_squared_cdf(quantile, dof);
            assert!(
                (actual - probability).abs() < 1e-4,
                "chi2_cdf({quantile}, {dof}) = {actual}, table says {probability}"
            );
        }
    }

    #[test]
    fn cdf_matches_the_closed_form_for_two_degrees_of_freedom() {
        // chi^2(2) is Exponential with mean 2, so its CDF is 1 - exp(-x/2) exactly.
        // This is the one case where an independent closed form exists, so it checks
        // the incomplete-gamma machinery rather than a table lookup.
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0] {
            assert_approx_eq!(chi_squared_cdf(x, 2), 1.0 - (-x / 2.0).exp(), 1e-12);
        }
    }

    #[test]
    fn cdf_and_quantile_are_inverses() {
        // Round-tripping exercises both branches of the incomplete gamma: the series
        // below x = a + 1 and the continued fraction above it.
        for dof in [1, 2, 3, 5, 6, 9, 15] {
            for probability in [0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 0.999_99] {
                let x = chi_squared_quantile(probability, dof);
                assert!(
                    (chi_squared_cdf(x, dof) - probability).abs() < 1e-9,
                    "round trip failed at dof {dof}, p {probability}"
                );
            }
        }
    }

    #[test]
    fn cdf_is_monotone_and_bounded() {
        let mut previous = 0.0;
        for step in 0..200 {
            let value = chi_squared_cdf(f64::from(step) * 0.25, 3);
            assert!((0.0..=1.0).contains(&value), "CDF left [0, 1]: {value}");
            assert!(value >= previous, "CDF decreased at step {step}");
            previous = value;
        }
    }

    #[test]
    fn cdf_handles_degenerate_arguments() {
        assert_approx_eq!(chi_squared_cdf(-1.0, 3), 0.0, 1e-15);
        assert_approx_eq!(chi_squared_cdf(0.0, 3), 0.0, 1e-15);
        assert_approx_eq!(chi_squared_cdf(f64::INFINITY, 3), 1.0, 1e-15);
        assert_approx_eq!(chi_squared_cdf(f64::NAN, 3), 0.0, 1e-15);
        // Zero degrees of freedom has no density.
        assert_approx_eq!(chi_squared_cdf(5.0, 0), 0.0, 1e-15);
    }

    #[test]
    fn quantile_fails_open_rather_than_closed() {
        // An unusable request must produce a threshold that accepts everything, never
        // one that rejects everything: a gate that silently discards every fix
        // degrades the run to dead reckoning without reporting anything.
        assert!(chi_squared_quantile(1.0, 3).is_infinite());
        assert!(chi_squared_quantile(1.5, 3).is_infinite());
        assert!(chi_squared_quantile(f64::NAN, 3).is_infinite());
        assert_approx_eq!(chi_squared_quantile(0.0, 3), 0.0, 1e-15);
        assert_approx_eq!(chi_squared_quantile(0.95, 0), 0.0, 1e-15);
    }

    #[test]
    fn nis_of_a_whitened_innovation_is_its_squared_norm() {
        // With S = I the Mahalanobis distance degenerates to the Euclidean one.
        let innovation = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let s = DMatrix::identity(3, 3);
        assert_approx_eq!(
            normalized_innovation_squared(&innovation, &s).unwrap(),
            14.0,
            1e-12
        );
    }

    #[test]
    fn nis_scales_with_the_covariance() {
        // Each component enters as (nu_i / sigma_i)^2, so a 3 sigma error in each of
        // three independent components is 27 regardless of the sigmas themselves.
        let sigmas = [0.5, 2.0, 100.0];
        let innovation = DVector::from_vec(sigmas.iter().map(|s| 3.0 * s).collect());
        let s = DMatrix::from_diagonal(&DVector::from_vec(
            sigmas.iter().map(|s| s * s).collect::<Vec<_>>(),
        ));
        assert_approx_eq!(
            normalized_innovation_squared(&innovation, &s).unwrap(),
            27.0,
            1e-9
        );
    }

    #[test]
    fn nis_accounts_for_correlation() {
        // Two strongly correlated components with a *common-mode* innovation is a
        // likely event; the same innovation in opposite directions is not. A
        // diagonal-only treatment would score these identically.
        let s = DMatrix::from_row_slice(2, 2, &[1.0, 0.9, 0.9, 1.0]);
        let common_mode = DVector::from_vec(vec![1.0, 1.0]);
        let differential = DVector::from_vec(vec![1.0, -1.0]);
        let common = normalized_innovation_squared(&common_mode, &s).unwrap();
        let differential = normalized_innovation_squared(&differential, &s).unwrap();
        assert!(
            differential > 10.0 * common,
            "correlation ignored: common {common}, differential {differential}"
        );
    }

    #[test]
    fn nis_rejects_a_mismatched_covariance() {
        let innovation = DVector::from_vec(vec![1.0, 2.0]);
        let s = DMatrix::identity(3, 3);
        let error = normalized_innovation_squared(&innovation, &s).unwrap_err();
        assert!(matches!(
            error,
            StrapdownError::DimensionMismatch { expected: 2, .. }
        ));
    }

    #[test]
    fn chi_squared_gate_threshold_follows_the_measurement_dimension() {
        let gate = InnovationGate::chi_squared(0.999).unwrap();
        assert_approx_eq!(gate.threshold(1), 10.828, 1e-3);
        assert_approx_eq!(gate.threshold(3), 16.266, 1e-3);
        assert_approx_eq!(gate.threshold(6), 22.458, 1e-3);
        // The whole point of the variant: one configured number, three thresholds.
        assert!(gate.threshold(1) < gate.threshold(3));
        assert!(gate.threshold(3) < gate.threshold(6));
    }

    #[test]
    fn fixed_gate_ignores_the_measurement_dimension() {
        let gate = InnovationGate::fixed(25.0).unwrap();
        for dof in 1..10 {
            assert_approx_eq!(gate.threshold(dof), 25.0, 1e-15);
        }
        assert!(gate.accepts(24.9, 3));
        assert!(!gate.accepts(25.1, 3));
    }

    #[test]
    fn gate_boundary_is_inclusive() {
        let gate = InnovationGate::fixed(25.0).unwrap();
        assert!(gate.accepts(25.0, 3));
    }

    #[test]
    fn gate_rejects_non_finite_and_negative_statistics() {
        // A NaN NIS means the innovation covariance was unusable. Applying a
        // correction derived from it would put NaN into the navigation state, so this
        // has to reject even though every `<=` comparison against NaN is false
        // anyway -- the explicit check is what makes that intentional.
        let gate = InnovationGate::chi_squared(0.999).unwrap();
        assert!(!gate.accepts(f64::NAN, 3));
        assert!(!gate.accepts(f64::INFINITY, 3));
        assert!(!gate.accepts(-1.0, 3));
    }

    #[test]
    fn constructors_reject_out_of_range_configuration() {
        // 99.9 rather than 0.999 is the mistake worth catching loudly: as a
        // confidence it would silently mean "accept everything".
        for bad in [0.0, 1.0, 99.9, -0.5, f64::NAN] {
            assert!(
                InnovationGate::chi_squared(bad).is_err(),
                "accepted confidence {bad}"
            );
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                InnovationGate::fixed(bad).is_err(),
                "accepted threshold {bad}"
            );
        }
    }

    #[test]
    fn a_hand_edited_gate_fails_open() {
        // Public fields let a deserialized config bypass the constructors. That must
        // degrade to "accept everything" rather than take a run down mid-flight.
        let gate = InnovationGate::Fixed {
            threshold: f64::NAN,
        };
        assert!(gate.threshold(3).is_infinite());
        assert!(gate.accepts(1e9, 3));
    }

    #[test]
    fn default_gate_is_the_documented_confidence() {
        assert_eq!(
            InnovationGate::default(),
            InnovationGate::ChiSquared {
                confidence: DEFAULT_GATE_CONFIDENCE
            }
        );
    }

    #[test]
    fn gate_round_trips_through_yaml() {
        // The gate is meant to be set from a scenario file, so the serde shape is
        // part of its contract, not an implementation detail.
        let gate = InnovationGate::chi_squared(0.997).unwrap();
        let yaml = serde_yaml::to_string(&gate).unwrap();
        assert!(yaml.contains("chi_squared"), "unexpected encoding: {yaml}");
        assert_eq!(serde_yaml::from_str::<InnovationGate>(&yaml).unwrap(), gate);

        let parsed: InnovationGate = serde_yaml::from_str("!fixed\nthreshold: 25.0\n")
            .unwrap_or_else(|_| serde_yaml::from_str("fixed:\n  threshold: 25.0\n").unwrap());
        assert_eq!(parsed, InnovationGate::Fixed { threshold: 25.0 });
    }

    #[test]
    fn update_outcome_constructors_record_the_decision() {
        let accepted = UpdateOutcome::accepted(4.2, 3);
        assert!(accepted.accepted);
        assert_approx_eq!(accepted.nis, 4.2, 1e-15);
        assert_eq!(accepted.dof, 3);

        let rejected = UpdateOutcome::rejected(400.0, 3);
        assert!(!rejected.accepted);
        assert_approx_eq!(rejected.nis, 400.0, 1e-15);
    }

    #[test]
    fn a_well_tuned_filter_passes_its_own_gate_almost_always() {
        // The statistical claim the gate rests on: if innovations really are
        // N(0, S), a 0.999 gate rejects about one in a thousand. Draw from a known
        // S, score them, and count. Deterministic seed -- this is an assertion about
        // the distribution, not about a particular random stream.
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(20_260_912);
        let standard = Normal::new(0.0, 1.0).unwrap();
        let sigmas = [3.0, 3.0, 7.0];
        let s = DMatrix::from_diagonal(&DVector::from_vec(
            sigmas.iter().map(|v| v * v).collect::<Vec<_>>(),
        ));
        let gate = InnovationGate::chi_squared(0.999).unwrap();

        let trials = 20_000;
        let mut rejected = 0;
        for _ in 0..trials {
            let innovation = DVector::from_vec(
                sigmas
                    .iter()
                    .map(|sigma| sigma * standard.sample(&mut rng))
                    .collect(),
            );
            let nis = normalized_innovation_squared(&innovation, &s).unwrap();
            if !gate.accepts(nis, 3) {
                rejected += 1;
            }
        }
        let rate = f64::from(rejected) / f64::from(trials);
        // Expected 0.001; the binomial standard error at n = 20000 is ~2.2e-4, so
        // 0.003 is a comfortable multiple of it and still catches a gate that is an
        // order of magnitude too tight.
        assert!(
            rate < 0.003,
            "rejected {rate} of valid measurements, expected ~0.001"
        );
    }

    #[test]
    fn the_gate_catches_a_gross_outlier() {
        // The case the gate exists for: a fix hundreds of metres off when the filter
        // expects metres.
        let s = DMatrix::from_diagonal(&DVector::from_vec(vec![25.0, 25.0, 100.0]));
        let gate = InnovationGate::chi_squared(0.999).unwrap();

        let ordinary = DVector::from_vec(vec![4.0, -3.0, 12.0]);
        let multipath = DVector::from_vec(vec![250.0, -80.0, 30.0]);

        assert!(gate.accepts(normalized_innovation_squared(&ordinary, &s).unwrap(), 3));
        assert!(!gate.accepts(normalized_innovation_squared(&multipath, &s).unwrap(), 3));
    }
}
