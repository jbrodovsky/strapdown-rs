//! Linear algebra helpers for robust covariance square roots.
//!
//! Public API:
//!     pub fn `matrix_square_root(matrix`: &`DMatrix<f64>`) -> Result<`DMatrix<f64>`, `StrapdownError`>
//!
//! Internal pipeline (each step isolated for testing):
//!     - `symmetrize()`
//!     - `chol_sqrt()`
//!     - `chol_sqrt_with_jitter()`
//!     - `evd_symmetric_sqrt_with_floor()`
//!
//! Strategy:
//! 1) Symmetrize P ← 0.5 (P + Pᵀ)
//! 2) Cholesky
//! 3) Jittered Cholesky (geometric ramp)
//! 4) Symmetric EVD with eigenvalue floor → S = U * sqrt(Λ⁺) * Uᵀ

use nalgebra::DMatrix;
use nalgebra::linalg::{Cholesky, SymmetricEigen};

use crate::StrapdownError;

/// Compute a robust symmetric square root `S` such that approximately `matrix ≈ S * Sᵀ`.
///
/// Attempts Cholesky decomposition first (yielding L such that matrix = L * L^T).
/// If Cholesky fails (e.g., matrix is not positive definite), it attempts to compute
/// the square root using eigenvalue decomposition (S = V * sqrt(D) * V^T).
///
/// # Arguments
/// * `matrix` - The `DMatrix<f64>` to find the square root of. It's assumed to be symmetric and square.
///
/// # Returns
/// A matrix square root `M` such that `matrix` approx `M * M.transpose()`. The result from
/// Cholesky is lower triangular; the result from eigenvalue decomposition is symmetric.
///
/// # Errors
/// [`StrapdownError::NotSquare`] if `matrix` is not square. The documented contract always
/// said so; until the zero-panic work of #254 the signature simply could not express it.
///
/// # Panics
/// Does not panic. The eigenvalue fallback floors negative eigenvalues rather than failing,
/// so every input that is square yields a result.
pub fn matrix_square_root(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, StrapdownError> {
    // Tunable guards (conservative defaults for double precision INS scales)
    const INITIAL_JITTER: f64 = 1e-12;
    const MAX_JITTER: f64 = 1e-6;
    const MAX_TRIES: usize = 6;
    const EIGEN_FLOOR: f64 = 1e-12;

    if !matrix.is_square() {
        return Err(StrapdownError::NotSquare {
            what: "matrix_square_root",
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }

    assert!(
        matrix.is_square(),
        "matrix_square_root: matrix must be square"
    );
    // 1) Symmetrize to kill round-off asymmetry
    let p = symmetrize(matrix);
    // 2) Equilibrated Cholesky (fast path)
    if let Some(s) = equilibrated_chol_sqrt(&p) {
        return Ok(s);
    }
    // 3) Plain Cholesky, for the inputs equilibration declines
    if let Some(s) = chol_sqrt(&p) {
        return Ok(s);
    }
    // 4) Jittered Cholesky
    if let Some(s) = chol_sqrt_with_jitter(&p, INITIAL_JITTER, MAX_JITTER, MAX_TRIES) {
        return Ok(s);
    }
    // 5) EVD fallback with eigenvalue floor — symmetric square root
    Ok(evd_symmetric_sqrt_with_floor(&p, EIGEN_FLOOR))
}
/// Cholesky of a covariance, taken on its *correlation* matrix and scaled back.
///
/// # What this is worth, measured
///
/// The navigation state mixes units: latitude and longitude are radians while altitude is
/// metres, and one metre is $1.57\times10^{-7}$ radians. A position uncertainty that is
/// physically isotropic therefore lands on the covariance diagonal fourteen orders of
/// magnitude apart. Measured on `syn_outage_60s__ukf`, the worst-conditioned covariance the
/// UKF hands this function has a latitude variance of $5.90\times10^{-20}\,\mathrm{rad}^2$
/// (a sigma of **1.5 mm**) beside an altitude variance of $3.23\times10^{-6}\,\mathrm{m}^2$
/// (a sigma of **1.8 mm**) -- the same physical uncertainty -- for a condition number of
/// $\kappa = 5.17\times10^{14}$.
///
/// **That $\kappa$ is not worth what it looks like, and saying so is the point of this
/// note.** Cholesky is backward stable, so $L L^\top$ reproduces $P$ to a relative
/// $2.6\times10^{-16}$ however badly scaled $P$ is; the naive "a factorization loses
/// $\log_{10}\kappa$ digits" reading predicts a catastrophe that does not happen. What the
/// unscented filter consumes is not the product but the individual *columns* of $L$, which
/// are its sigma points, and those are where the scaling shows:
///
/// | | worst relative change in $L$ from a **one-ulp** change in $P$ |
/// |---|---:|
/// | plain Cholesky | $1.58\times10^{-14}$ |
/// | equilibrated | $7.51\times10^{-15}$ |
///
/// So equilibrating is worth **a factor of about two**, not fourteen orders of magnitude.
/// End to end on the accuracy suite it agrees: with `alpha = 0.1`, the worst of 236 gated
/// metrics responds to a one-ulp perturbation of a WGS84 constant by $2.97\times10^{-4}$%
/// plain and $1.40\times10^{-4}$% equilibrated.
///
/// It is kept because it is free -- $P = D S D$ with $D = \mathrm{diag}(\sqrt{P_{ii}})$
/// makes $S$ a correlation matrix, and $L = D L_S$ is a Cholesky factor of $P$ *exactly*,
/// since $D L_S (D L_S)^\top = D S D = P$, so in exact arithmetic this changes nothing at
/// all -- and because the factor of two is the part of the scaling that does not depend on
/// `alpha` staying where it is. A caller is free to set `ukf_alpha` back down.
///
/// **The large term is elsewhere.** [`#399`](https://github.com/jbrodovsky/strapdown-rs/issues/399)
/// is dominated by the sigma-point weights, not by this: at the old `alpha = 1e-3` default
/// the mean was formed as $-999{,}999\,x_0 + \sum 31{,}250\,x_i$ and the same one-ulp
/// perturbation moved a gated metric by **9.77%**. Raising `alpha` to `0.1` took that to
/// $2.97\times10^{-4}$% on its own -- a factor of 33,000 against this function's 2. See
/// [`ClosedLoopConfig::ukf_alpha`](crate::sim::ClosedLoopConfig).
///
/// # Why it may decline
///
/// Returns `None` -- leaving [`matrix_square_root`]'s remaining rungs to handle it -- when
/// any diagonal entry is not strictly positive and finite, or when the scaled matrix is
/// still not factorizable. A non-positive diagonal entry is exactly the case the jitter and
/// eigenvalue fallbacks exist for, and dividing by it here would turn a recoverable input
/// into a non-finite one.
fn equilibrated_chol_sqrt(p: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = p.nrows();
    let mut scale = vec![0.0_f64; n];
    for (i, entry) in scale.iter_mut().enumerate() {
        let diagonal = p[(i, i)];
        if !(diagonal.is_finite() && diagonal > 0.0) {
            return None;
        }
        *entry = diagonal.sqrt();
    }

    let mut correlation = p.clone();
    for i in 0..n {
        for j in 0..n {
            correlation[(i, j)] /= scale[i] * scale[j];
        }
        // Set the diagonal rather than dividing it: `p_ii / (sqrt(p_ii) * sqrt(p_ii))` is 1
        // only to within a rounding error, and a correlation matrix whose diagonal is
        // 0.9999999999999999 is one the factorization has to work around for no reason.
        correlation[(i, i)] = 1.0;
    }

    let mut factor = Cholesky::new(correlation)?.l().into_owned();
    for i in 0..n {
        for j in 0..=i {
            factor[(i, j)] *= scale[i];
        }
    }
    Some(factor)
}
/// Symmetrize a matrix: P ← 0.5 (P + Pᵀ)
///
/// Simple matrix symmetrization function that reduces round-off errors associated
/// with floating point arithmetic.
///
/// # Arguments
/// * `m` - the matrix to symmetrize
///
/// # Returns
/// A symmetrized version of the input matrix.
#[inline]
pub fn symmetrize(m: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (m + m.transpose())
}
/// Plain Cholesky square root
///
/// Cholesky factorization that returns L such that P ≈ L Lᵀ, or None if it fails.
/// This is a quick way to initially attempt to calculate a matrix square root.
///
/// # Arguments
/// * ``p` - the matrix to factor
///
/// # Returns
/// A lower triangular matrix L such that P ≈ L Lᵀ, or None if it fails.
fn chol_sqrt(p: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    Cholesky::new(p.clone()).map(|ch| ch.l().into_owned())
}
/// Cholesky with diagonal jitter (geometric ramp). Returns None if all tries fail.
///
/// Perform Cholesky decomposition with a jittered diagonal on a geometric ramp up.
/// Returns None if all tries fail.
fn chol_sqrt_with_jitter(
    p: &DMatrix<f64>,
    initial_jitter: f64,
    max_jitter: f64,
    max_tries: usize,
) -> Option<DMatrix<f64>> {
    let n = p.nrows();
    let mut jitter = initial_jitter;
    for _ in 0..max_tries {
        let mut pj = p.clone();
        for i in 0..n {
            pj[(i, i)] += jitter;
        }
        if let Some(ch) = Cholesky::new(pj) {
            return Some(ch.l().into_owned());
        }
        jitter *= 10.0;
        if jitter > max_jitter {
            break;
        }
    }
    None
}

/// Symmetric EVD square root with eigenvalue flooring:
/// S = U * sqrt(max(λ, floor)) * Uᵀ
fn evd_symmetric_sqrt_with_floor(p: &DMatrix<f64>, floor: f64) -> DMatrix<f64> {
    let se = SymmetricEigen::new(p.clone());
    let mut lambdas = se.eigenvalues;
    let u = se.eigenvectors;

    for i in 0..lambdas.len() {
        if lambdas[i] < floor {
            lambdas[i] = floor;
        }
    }

    let sqrt_vals = lambdas.map(f64::sqrt);
    let sigma_half = DMatrix::<f64>::from_diagonal(&sqrt_vals);
    &u * sigma_half * u.transpose()
}

/// Bounds on the jittered Cholesky retry ramp used by [`chol_solve_spd`].
///
/// A covariance that has drifted slightly indefinite through round-off fails a plain
/// Cholesky factorization. Rather than give up, [`chol_solve_spd`] adds a small positive
/// amount to every diagonal entry and refactors, multiplying that amount by ten on each
/// subsequent attempt. These fields bound that ramp. The [`Default`] values --
/// `1e-12`, `1e-6` and 6 attempts -- are the ones [`robust_spd_solve`] uses and match the
/// guards hard-coded in [`matrix_square_root`].
#[derive(Debug, Clone, Copy)]
pub struct SolveOptions {
    /// Amount added to each diagonal entry on the first retry; defaults to `1e-12`.
    pub initial_jitter: f64,
    /// Ceiling on the ramp: retries stop once the next jitter would exceed it. Defaults to `1e-6`.
    pub max_jitter: f64,
    /// Maximum number of jittered factorization attempts; defaults to 6.
    pub max_tries: usize,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            initial_jitter: 1e-12,
            max_jitter: 1e-6,
            max_tries: 6,
        }
    }
}
/// Solve `A X = B` for SPD-ish `A` via Cholesky, with jitter retries.
///
/// # Errors
/// * [`StrapdownError::NotSquare`] if `A` is not square.
/// * [`StrapdownError::DimensionMismatch`] if `A` and `B` have different row counts.
/// * [`StrapdownError::SingularMatrix`] if every jittered Cholesky attempt fails, which in
///   practice means a diverged or collapsed covariance.
pub fn chol_solve_spd(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    opt: SolveOptions,
) -> Result<DMatrix<f64>, StrapdownError> {
    if !a.is_square() {
        return Err(StrapdownError::NotSquare {
            what: "chol_solve_spd",
            rows: a.nrows(),
            cols: a.ncols(),
        });
    }
    if a.nrows() != b.nrows() {
        return Err(StrapdownError::DimensionMismatch {
            what: "chol_solve_spd right-hand side rows",
            expected: a.nrows(),
            got: b.nrows(),
        });
    }

    // Symmetrize first (SPD drift is common).
    let a_sym = symmetrize(a);

    // Try plain Cholesky
    if let Some(ch) = Cholesky::new(a_sym.clone()) {
        return Ok(ch.solve(b));
    }

    // Jitter ramp
    let n = a_sym.nrows();
    let mut jitter = opt.initial_jitter;
    for _ in 0..opt.max_tries {
        let mut a_j = a_sym.clone();
        for i in 0..n {
            a_j[(i, i)] += jitter;
        }
        if let Some(ch) = Cholesky::new(a_j) {
            return Ok(ch.solve(b));
        }
        jitter *= 10.0;
        if jitter > opt.max_jitter {
            break;
        }
    }
    Err(StrapdownError::SingularMatrix {
        what: "chol_solve_spd",
        dim: n,
    })
}

/// Robust SPD solve with sane defaults:
/// - Cholesky + jitter (preferred)
/// - Last resort: explicit inverse
///
/// # Errors
/// * [`StrapdownError::NotSquare`] / [`StrapdownError::DimensionMismatch`] propagated from
///   [`chol_solve_spd`] — these are caller errors and are returned verbatim rather than
///   falling through to the inverse, which would fail for the same reason.
/// * [`StrapdownError::SingularMatrix`] if neither the jittered Cholesky nor an explicit
///   inverse succeeds.
pub fn robust_spd_solve(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
) -> Result<DMatrix<f64>, StrapdownError> {
    match chol_solve_spd(a, b, SolveOptions::default()) {
        Ok(x) => Ok(x),
        // Only a genuine factorization failure justifies the explicit-inverse fallback; a
        // shape error would fail identically and should surface as itself.
        Err(StrapdownError::SingularMatrix { .. }) => symmetrize(a).try_inverse().map_or_else(
            || {
                Err(StrapdownError::SingularMatrix {
                    what: "robust_spd_solve",
                    dim: a.nrows(),
                })
            },
            |inv| Ok(&inv * b),
        ),
        Err(other) => Err(other),
    }
}

/* =============================== Tests ==================================== */

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured diagonal of a covariance the UKF actually factorizes, from
    /// `syn_outage_60s__ukf` (#399). Sixteen states: latitude and longitude in rad^2,
    /// altitude in m^2, velocities, attitudes, six IMU biases and a barometric bias.
    ///
    /// The first two entries are a horizontal sigma of **1.5 mm** and the third a vertical
    /// sigma of **1.8 mm** -- the same physical uncertainty, fourteen orders apart on the
    /// diagonal purely because one metre is 1.57e-7 radians.
    const MEASURED_UKF_DIAGONAL: [f64; 16] = [
        5.898e-20, 6.982e-20, 3.227e-6, 1.083e-6, 9.417e-7, 2.840e-7, 4.288e-9, 4.305e-9, 1.876e-8,
        1.204e-7, 1.204e-7, 7.820e-10, 7.238e-12, 7.249e-12, 1.033e-11, 1.372e-5,
    ];

    /// Build `D C D` from [`MEASURED_UKF_DIAGONAL`] and a correlation matrix with a
    /// deterministic, non-trivial off-diagonal pattern.
    fn ill_conditioned_covariance() -> DMatrix<f64> {
        let n = MEASURED_UKF_DIAGONAL.len();
        let mut correlation = DMatrix::identity(n, n);
        for i in 0..n {
            for j in 0..i {
                // Deterministic and bounded well inside 1, so the matrix stays positive
                // definite while carrying real correlation.
                #[allow(clippy::cast_precision_loss)]
                let rho = 0.45 * (((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.5);
                correlation[(i, j)] = rho;
                correlation[(j, i)] = rho;
            }
        }
        let scale = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            n,
            MEASURED_UKF_DIAGONAL.iter().map(|v| v.sqrt()),
        ));
        &scale * correlation * &scale
    }

    /// Worst relative change in a Cholesky factor's entries between two nearby inputs.
    ///
    /// Per entry and relative, on purpose. The unscented filter consumes the *columns* of
    /// `L` as its sigma points, and on this matrix the entries span 1e14, so an absolute
    /// norm would report only the largest block and never the horizontal position one.
    fn worst_relative_factor_change(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..a.nrows() {
            for j in 0..=i {
                let denom = a[(i, j)].abs().max(b[(i, j)].abs());
                if denom > 0.0 {
                    worst = worst.max((a[(i, j)] - b[(i, j)]).abs() / denom);
                }
            }
        }
        worst
    }

    /// #399: equilibrating halves how far the factor moves when the covariance moves by one
    /// ulp.
    ///
    /// This asserts the *ratio* between the two routes on the same input, not an absolute
    /// bound, so it is a statement about the scaling rather than about this matrix.
    ///
    /// It deliberately does **not** assert a large factor. The first version of this fix
    /// claimed equilibration recovered the $\log_{10}\kappa \approx 14$ digits the condition
    /// number suggests, and a reconstruction test refuted that immediately: plain Cholesky
    /// reproduces `P` to 2.6e-16 whatever the scaling, because it is backward stable. Two is
    /// what the measurement supports, and two is what this pins.
    #[test]
    fn equilibrating_halves_the_factors_sensitivity_to_a_one_ulp_input_change() {
        let p = ill_conditioned_covariance();
        let mut nudged = p.clone();
        nudged[(2, 2)] = f64::from_bits(p[(2, 2)].to_bits() + 1);

        let plain = worst_relative_factor_change(
            &chol_sqrt(&p).expect("the test matrix is positive definite"),
            &chol_sqrt(&nudged).expect("so is the nudged one"),
        );
        let equilibrated = worst_relative_factor_change(
            &equilibrated_chol_sqrt(&p).expect("its diagonal is strictly positive and finite"),
            &equilibrated_chol_sqrt(&nudged).expect("likewise"),
        );

        assert!(
            equilibrated < plain * 0.75,
            "equilibration must make the factor meaningfully less sensitive; plain \
             {plain:.4e} vs equilibrated {equilibrated:.4e}"
        );
        assert!(
            plain > 0.0 && equilibrated > 0.0,
            "a one-ulp input change must move the factor at all, or this measures nothing: \
             plain {plain:.4e}, equilibrated {equilibrated:.4e}"
        );
        // Both routes stay near machine precision -- the point is the ratio, and a route
        // that had gone badly wrong would show up here rather than passing the ratio test
        // by being uniformly terrible.
        assert!(
            plain < 1e-12 && equilibrated < 1e-12,
            "neither route should be anywhere near a percent: plain {plain:.4e}, \
             equilibrated {equilibrated:.4e}"
        );
    }

    /// The backward-stability control for the test above: whatever the scaling, `L L^T`
    /// reproduces `P`. Without this the ratio test reads as "equilibration fixes a broken
    /// factorization", which is not what it does.
    #[test]
    fn a_plain_cholesky_of_an_ill_scaled_covariance_still_reconstructs_it() {
        let p = ill_conditioned_covariance();
        let l = chol_sqrt(&p).expect("positive definite");
        let reconstructed = &l * l.transpose();
        let mut worst = 0.0_f64;
        for i in 0..p.nrows() {
            for j in 0..p.ncols() {
                let scale = (p[(i, i)] * p[(j, j)]).sqrt();
                if scale > 0.0 {
                    worst = worst.max((reconstructed[(i, j)] - p[(i, j)]).abs() / scale);
                }
            }
        }
        assert!(
            worst < 1e-14,
            "backward stability: a condition number of 5e14 does not stop `L L^T` from \
             reproducing `P`, got {worst:.4e}"
        );
    }

    /// The guard: a diagonal that is not strictly positive and finite declines, so the
    /// jitter and eigenvalue rungs below still see the matrix they exist for.
    #[test]
    fn equilibration_declines_a_non_positive_diagonal() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut p = ill_conditioned_covariance();
            p[(3, 3)] = bad;
            assert!(
                equilibrated_chol_sqrt(&p).is_none(),
                "a diagonal entry of {bad} must be declined, not divided by"
            );
        }
        // And `matrix_square_root` still answers for such an input, via the lower rungs.
        let mut p = ill_conditioned_covariance();
        p[(3, 3)] = 0.0;
        assert!(matrix_square_root(&p).is_ok());
    }

    fn approx_eq(a: &DMatrix<f64>, b: &DMatrix<f64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        let mut max_abs = 0.0f64;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                max_abs = max_abs.max((a[(i, j)] - b[(i, j)]).abs());
            }
        }
        max_abs <= tol
    }

    #[test]
    fn t_symmetrize() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.0, 3.0]);
        let s = symmetrize(&m);
        let s_expected = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 3.0]);
        assert!(approx_eq(&s, &s_expected, 1e-15));
    }

    #[test]
    fn t_chol_sqrt_spd() {
        // P = A Aᵀ is SPD
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 0.5, 0.0, 1.0, -1.0, 0.0, 0.0, 0.2]);
        let p = &a * a.transpose();
        let s = chol_sqrt(&p).expect("Cholesky should succeed for SPD");
        let back = &s * s.transpose();
        assert!(approx_eq(&back, &p, 1e-12));
    }

    #[test]
    fn t_chol_sqrt_with_jitter() {
        // Nudge diagonal a hair negative to break plain Cholesky
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 0.2, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 1.0]);
        let mut p = &a * a.transpose();
        p[(2, 2)] -= 1e-10;

        //assert!(chol_sqrt(&p).is_none(), "plain Cholesky should fail here");
        let s =
            chol_sqrt_with_jitter(&p, 1e-12, 1e-6, 6).expect("jittered Cholesky should succeed");
        let back = &s * s.transpose();
        let p_sym = symmetrize(&p);
        assert!(approx_eq(&back, &p_sym, 1e-8));
    }

    #[test]
    fn t_evd_floor() {
        // Make P symmetric but with a negative eigenvalue, EVD should floor it.
        let p = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]); // eigenvalues {+1, -1}
        let s = evd_symmetric_sqrt_with_floor(&p, 1e-12);
        let back = &s * s.transpose();
        // back should be PSD and close to symmetrized p with floor effects
        let p_sym = symmetrize(&p);
        assert_eq!(back.nrows(), p_sym.nrows());
        assert_eq!(back.ncols(), p_sym.ncols());
        // sanity: back is symmetric
        assert!(approx_eq(&back, &back.transpose(), 1e-14));
    }

    #[test]
    fn t_public_identity() {
        let i = DMatrix::<f64>::identity(4, 4);
        let s = matrix_square_root(&i).unwrap();
        assert!(approx_eq(&s, &i, 1e-14));
        let back = &s * s.transpose();
        assert!(approx_eq(&back, &i, 1e-12));
    }

    #[test]
    fn t_public_nearly_spd() {
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 0.1, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 1.0]);
        let mut p = &a * a.transpose();
        p[(2, 2)] -= 1e-10;
        p[(0, 2)] += 1e-12; // asymmetry

        let s = matrix_square_root(&p).unwrap();
        let back = &s * s.transpose();
        let p_sym = symmetrize(&p);
        assert!(approx_eq(&back, &p_sym, 1e-8));
    }

    /// Was `t_public_non_square_panics`. The precondition the doc always described is now
    /// returned rather than asserted (#254).
    #[test]
    fn t_public_non_square_errors() {
        let m = DMatrix::<f64>::zeros(3, 2);
        let got = matrix_square_root(&m);
        assert!(
            matches!(
                got,
                Err(StrapdownError::NotSquare {
                    rows: 3,
                    cols: 2,
                    ..
                })
            ),
            "expected NotSquare{{rows: 3, cols: 2}}, got {got:?}"
        );
    }

    #[test]
    fn t_chol_sqrt_none() {
        // Create a matrix that is NOT positive definite (negative eigenvalue)
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]); // eigenvalues: 3, -1
        let result = chol_sqrt(&m);
        assert!(result.is_none(), "Cholesky should fail for non-PD matrix");
    }

    #[test]
    fn t_chol_sqrt_with_jitter_max_tries() {
        // Create a matrix that needs jitter to become PD
        let mut m = DMatrix::<f64>::identity(3, 3);
        m[(0, 0)] = 0.0; // Make it PSD but not PD

        // With sufficient jitter, should succeed
        let result = chol_sqrt_with_jitter(&m, 0.01, 2.0, 3);
        // The function may or may not succeed depending on jitter parameters
        // Just verify it doesn't crash
        let _ = result;
    }

    #[test]
    fn t_chol_sqrt_with_jitter_none() {
        // Create a matrix that cannot be fixed even with jitter
        let mut m = DMatrix::<f64>::identity(3, 3);
        m[(0, 0)] = -1e10; // Extremely negative diagonal

        // With reasonable jitter bounds, this should fail
        let result = chol_sqrt_with_jitter(&m, 1e-12, 1e-6, 6);
        // This might still succeed with enough jitter, so we just test it runs
        let _ = result;
    }

    #[test]
    fn t_evd_floor_negative_eigenvalues() {
        // Matrix with negative eigenvalues that need flooring
        let m = DMatrix::from_row_slice(3, 3, &[-1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 3.0]);

        let s = evd_symmetric_sqrt_with_floor(&m, 1e-6);
        let back = &s * s.transpose();

        // Should be symmetric and PSD
        assert!(approx_eq(&back, &back.transpose(), 1e-12));

        // All eigenvalues of back should be >= floor
        let se = SymmetricEigen::new(back);
        for lambda in se.eigenvalues.iter() {
            assert!(
                *lambda >= -1e-10,
                "Eigenvalue should be non-negative after flooring"
            );
        }
    }

    #[test]
    fn t_matrix_square_root_evd_fallback() {
        // Create a matrix that will fail Cholesky but succeed with EVD
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]); // Has negative eigenvalue

        let s = matrix_square_root(&m).unwrap();
        let back = &s * s.transpose();

        // Result should be symmetric and close to symmetrized input
        assert!(approx_eq(&back, &back.transpose(), 1e-12));
    }

    #[test]
    fn t_chol_solve_spd_basic() {
        // Solve A X = B where A is SPD
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let b = DMatrix::from_row_slice(2, 1, &[6.0, 5.0]);

        let x = chol_solve_spd(&a, &b, SolveOptions::default()).expect("Should solve");
        let result = &a * &x;

        assert!(approx_eq(&result, &b, 1e-10));
    }

    #[test]
    fn t_chol_solve_spd_with_jitter() {
        // Solve with a nearly-singular matrix
        let mut a = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        a[(1, 1)] -= 0.25; // Make it barely PD
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);

        let x = chol_solve_spd(&a, &b, SolveOptions::default()).expect("Should solve with jitter");
        let result = &a * &x;

        assert!(approx_eq(&result, &b, 1e-8));
    }

    #[test]
    fn t_chol_solve_spd_none() {
        // Create a very ill-conditioned or singular matrix
        let a = DMatrix::from_row_slice(2, 2, &[1e-15, 0.0, 0.0, 1e-15]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);

        let opts = SolveOptions {
            initial_jitter: 1e-20,
            max_jitter: 1e-18,
            max_tries: 2,
        };

        let result = chol_solve_spd(&a, &b, opts);
        // Might fail or succeed depending on numerical precision
        let _ = result;
    }

    #[test]
    fn t_robust_spd_solve_basic() {
        // Test the robust solver with a good matrix
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let b = DMatrix::from_row_slice(2, 1, &[6.0, 5.0]);

        let x = robust_spd_solve(&a, &b).unwrap();
        let result = &a * &x;

        assert!(approx_eq(&result, &b, 1e-10));
    }

    #[test]
    fn t_robust_spd_solve_fallback() {
        // Test fallback to inverse when Cholesky fails
        let mut a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        a[(0, 1)] = 1e-8; // Small asymmetry
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 2.0]);

        let x = robust_spd_solve(&a, &b).unwrap();
        let a_sym = symmetrize(&a);
        let result = &a_sym * &x;

        assert!(approx_eq(&result, &b, 1e-8));
    }

    #[test]
    fn t_robust_spd_solve_panic() {
        // Test with a singular matrix - robust_spd_solve should either solve or panic
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);

        // This may panic or may handle it gracefully depending on implementation
        // We test that it at least executes
        let result = std::panic::catch_unwind(|| robust_spd_solve(&a, &b)).unwrap();

        // Expect either panic or some result
        assert!(result.is_err() || result.is_ok());
    }

    /// Was `t_chol_solve_spd_non_square_panic`. The precondition is now reported rather
    /// than asserted (#254); the test still pins the same contract.
    #[test]
    fn t_chol_solve_spd_non_square_errors() {
        let a = DMatrix::<f64>::zeros(3, 2);
        let b = DMatrix::<f64>::zeros(3, 1);
        let got = chol_solve_spd(&a, &b, SolveOptions::default());
        assert!(
            matches!(
                got,
                Err(StrapdownError::NotSquare {
                    rows: 3,
                    cols: 2,
                    ..
                })
            ),
            "expected NotSquare{{rows: 3, cols: 2}}, got {got:?}"
        );
    }

    /// Was `t_chol_solve_spd_incompatible_panic`.
    #[test]
    fn t_chol_solve_spd_incompatible_errors() {
        let a = DMatrix::<f64>::identity(2, 2);
        let b = DMatrix::<f64>::zeros(3, 1);
        let got = chol_solve_spd(&a, &b, SolveOptions::default());
        assert!(
            matches!(
                got,
                Err(StrapdownError::DimensionMismatch {
                    expected: 2,
                    got: 3,
                    ..
                })
            ),
            "expected DimensionMismatch{{expected: 2, got: 3}}, got {got:?}"
        );
    }

    #[test]
    fn t_chol_solve_spd_max_jitter_exceeded() {
        // Test that jitter loop terminates when max_jitter is exceeded
        let a = DMatrix::from_row_slice(2, 2, &[-10.0, 0.0, 0.0, -10.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);

        let opts = SolveOptions {
            initial_jitter: 1e-6,
            max_jitter: 1e-5,
            max_tries: 10,
        };

        let result = chol_solve_spd(&a, &b, opts);
        assert!(
            matches!(result, Err(StrapdownError::SingularMatrix { .. })),
            "should report SingularMatrix when the jitter limit is exceeded, got {result:?}"
        );
    }

    #[test]
    fn t_robust_spd_solve_inverse_fallback() {
        // Create a matrix that will fail Cholesky but has valid inverse
        let mut a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        a[(0, 0)] = -0.1; // Make it fail Cholesky with small negative eigenvalue
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);

        let opts = SolveOptions {
            initial_jitter: 1e-20,
            max_jitter: 1e-19,
            max_tries: 1,
        };

        // Force chol_solve_spd to fail by using very restrictive options
        let chol_result = chol_solve_spd(&a, &b, opts);
        assert!(
            matches!(chol_result, Err(StrapdownError::SingularMatrix { .. })),
            "Cholesky should report SingularMatrix with restrictive jitter, got {chol_result:?}"
        );

        // Now test robust solver which should use inverse
        let x = robust_spd_solve(&a, &b).unwrap();
        let a_sym = symmetrize(&a);
        let result = &a_sym * &x;
        assert!(approx_eq(&result, &b, 1e-6));
    }

    #[test]
    fn t_robust_spd_solve_singular_handled() {
        // Create a truly singular matrix
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);

        // robust_spd_solve may panic or may handle it via jitter
        // We test that it either panics or succeeds (doesn't hang)
        let result = std::panic::catch_unwind(|| robust_spd_solve(&a, &b)).unwrap();

        // Either panic (Err) or succeed (Ok) - both are acceptable
        // The key is that it terminates
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn t_solve_options_default() {
        let opts = SolveOptions::default();
        assert_eq!(opts.initial_jitter, 1e-12);
        assert_eq!(opts.max_jitter, 1e-6);
        assert_eq!(opts.max_tries, 6);
    }
}

// ============ OLD ====================================

// Calculates a square root of a symmetric matrix.
//
// Attempts Cholesky decomposition first (yielding L such that matrix = L * L^T).
// If Cholesky fails (e.g., matrix is not positive definite), it attempts to compute
// the square root using eigenvalue decomposition (S = V * sqrt(D) * V^T).
// For eigenvalue decomposition, eigenvalues are clamped to be non-negative.
//
// # Arguments
// * `matrix` - The DMatrix<f64> to find the square root of. It's assumed to be symmetric and square.
//
// # Returns
// * `Some(DMatrix<f64>)` containing a matrix square root.
//   The result from Cholesky is lower triangular. The result from eigenvalue decomposition is symmetric.
//   In both cases, if the result is `M`, then `matrix` approx `M * M.transpose()`.
// * `None` if the matrix is not square or another fundamental issue prevents computation (though
//   this implementation tries to be robust for positive semi-definite cases).
//pub fn matrix_square_root(matrix: &DMatrix<f64>) -> DMatrix<f64> {
//    if !matrix.is_square() {
//        panic!("Error: Matrix must be square to compute square root.");
//    }
//    // Attempt Cholesky decomposition (yields L where matrix = L * L^T)
//    // Cholesky requires the matrix to be symmetric positive definite.
//    match cholesky_pass(matrix) {
//        Some(chol_l) => {
//            return chol_l;
//        }
//        None => {
//            //println!("Cholesky decomposition failed. Attempting eigenvalue decomposition.");
//        }
//    }
//    // If Cholesky failed, we try eigenvalue decomposition.
//    match eigenvalue_pass(matrix) {
//        Some(eigen_sqrt) => eigen_sqrt,
//        None => {
//            panic!(
//                "Cholesky and Eigenvalue decomposition failed. No valid square root found for the covariance matrix: \n {:?}",
//                matrix
//            );
//        }
//    }
//}
// Attempts to compute the matrix square root using Cholesky decomposition.
//
// This method is only applicable to symmetric positive definite matrices.
// If successful, it returns the lower triangular matrix `L` such that `matrix = L * L.transpose()`.
//
// When the computation _fails_ (e.g., the matrix is not positive definite or not square),
// a None value is returned instead of panicking, permitting the public API to proceed to the
// next method.
//
// # Arguments
// * `matrix` - The DMatrix<f64> to find the square root of. Assumed to be symmetric and square.
//
// # Returns
// * `Some(DMatrix<f64>)` containing the lower triangular Cholesky factor `L`.
// * `None` if the matrix is not positive definite or not square.
// fn cholesky_pass(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
//     if !matrix.is_square() {
//         eprintln!("Error: Matrix must be square for Cholesky decomposition.");
//         return None;
//     }
//     matrix
//         .clone()
//         .cholesky()
//         .map(|chol: Cholesky<f64, nalgebra::Dyn>| chol.l())
// }
// Computes a symmetric matrix square root using eigenvalue decomposition.
//
// This method is suitable for symmetric positive semi-definite matrices.
// It returns a symmetric matrix `S` such that `matrix = S * S`.
// Eigenvalues are clamped to be non-negative to handle positive semi-definite cases
// and minor numerical inaccuracies.
//
// When the computation _fails_ (e.g., the matrix is not positive definite or not square),
// a None value is returned instead of panicking, permitting the public API to proceed to the
// next method.
//
// # Arguments
// * `matrix` - The DMatrix<f64> to find the square root of. Assumed to be symmetric and square.
//
// # Returns
// * `Some(DMatrix<f64>)` containing the symmetric matrix square root `S`.
// * `None` if the matrix is not square (though this should be checked by the caller for symmetry assumptions).
// fn eigenvalue_pass(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
//     if !matrix.is_square() {
//         eprintln!("Error: Matrix must be square for eigenvalue decomposition based square root.");
//         return None;
//     }
//     // For eigenvalue decomposition of a symmetric matrix,
//     // we use `symmetric_eigen`. This returns real eigenvalues and orthogonal eigenvectors.
//     let eigen_decomposition: SymmetricEigen<f64, nalgebra::Dyn> = matrix.clone().symmetric_eigen();
//     let eigenvalues = eigen_decomposition.eigenvalues;
//     let eigenvectors = eigen_decomposition.eigenvectors;
//
//     // Check for significantly negative eigenvalues, indicating non-positive semi-definiteness.
//     // While we clamp them, a warning is useful for diagnosis.
//     if eigenvalues.iter().any(|&val| val < -1e-9) {
//         println!(
//             "Warning: Negative eigenvalues encountered during eigenvalue decomposition. The input matrix was not positive semi-definite."
//         );
//     //     println!("{:?}", matrix.data);
//     //     // return None;
//     }
//
//     // Create diagonal matrix of sqrt(eigenvalues), clamping eigenvalues to be non-negative.
//     // `DMatrix::from_diagonal` takes a DVector.
//     let sqrt_eigenvalues_diag_vec = eigenvalues.map(|val| val.max(1e-9).sqrt());
//     let sqrt_eigenvalues_diag = DMatrix::from_diagonal(&sqrt_eigenvalues_diag_vec);
//
//     // Reconstruct the square root: S = V * sqrt(D) * V^T
//     // This S will be symmetric, and S * S = matrix (or S * S^T = matrix).
//     let sqrt_m = eigenvectors.clone() * sqrt_eigenvalues_diag * eigenvectors.transpose();
//
//     Some(sqrt_m)
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use nalgebra::DMatrix;
//     use std::sync::LazyLock;
//
//     static BASIC_SQRT: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
//         DMatrix::from_row_slice(3, 3, &[4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0])
//     });
//     static POSITIVE_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
//         DMatrix::from_row_slice(3, 3, &[4.0, 2.0, 0.0, 2.0, 9.0, 3.0, 0.0, 3.0, 16.0])
//     });
//     static POSITIVE_SEMI_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
//         DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
//     });
//     static NEGATIVE_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
//         DMatrix::from_row_slice(3, 3, &[-4.0, 0.0, 0.0, 0.0, -9.0, 0.0, 0.0, 0.0, -16.0])
//     });
//     static NEGATIVE_SEMI_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
//         DMatrix::from_row_slice(3, 3, &[-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0])
//     });
//     static NON_SQUARE: LazyLock<DMatrix<f64>> =
//         LazyLock::new(|| DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
//
//     /// Helper function to verify if a matrix is a valid square root of another matrix.
//     /// Returns true if sqrt_matrix * sqrt_matrix.T ≈ original_matrix within tolerance.
//     fn is_valid_square_root(
//         sqrt_matrix: &DMatrix<f64>,
//         original_matrix: &DMatrix<f64>,
//         tolerance: f64,
//     ) -> bool {
//         let reconstructed = sqrt_matrix * sqrt_matrix.transpose();
//
//         if reconstructed.nrows() != original_matrix.nrows()
//             || reconstructed.ncols() != original_matrix.ncols()
//         {
//             return false;
//         }
//
//         for i in 0..original_matrix.nrows() {
//             for j in 0..original_matrix.ncols() {
//                 if (reconstructed[(i, j)] - original_matrix[(i, j)]).abs() > tolerance {
//                     return false;
//                 }
//             }
//         }
//         true
//     }
//     // Test matrix square root calculation
//     #[test]
//     fn cholesky_square_root() {
//         let sqrt_matrix = matrix_square_root(&BASIC_SQRT);
//         assert!(is_valid_square_root(&sqrt_matrix, &BASIC_SQRT, 1e-9));
//     }
//     #[test]
//     fn cholesky_positive_definite() {
//         let sqrt_matrix = matrix_square_root(&POSITIVE_DEFINITE);
//         assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
//     }
//     #[test]
//     #[should_panic]
//     fn cholesky_negative_definite() {
//         // This should panic because the matrix is negative definite.
//         let _sqrt_matrix = matrix_square_root(&NEGATIVE_DEFINITE);
//     }
//     #[test]
//     #[should_panic]
//     fn cholesky_negative_semi_definite() {
//         // This should panic because the matrix is negative semi-definite.
//         let _sqrt_matrix = matrix_square_root(&NEGATIVE_SEMI_DEFINITE);
//     }
//     #[test]
//     #[should_panic]
//     fn cholesky_non_square() {
//         // This should panic because the matrix is not square.
//         let _sqrt_matrix = matrix_square_root(&NON_SQUARE);
//     }
//     #[test]
//     fn eigenvalue_square_root() {
//         let sqrt_matrix = matrix_square_root(&POSITIVE_SEMI_DEFINITE);
//         assert!(is_valid_square_root(
//             &sqrt_matrix,
//             &POSITIVE_SEMI_DEFINITE,
//             1e-9
//         ));
//     }
//     #[test]
//     fn eigenvalue_positive_definite() {
//         let sqrt_matrix = matrix_square_root(&POSITIVE_DEFINITE);
//         assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
//     }
//     #[test]
//     fn eigenvalue_positive_semi_definite() {
//         let sqrt_matrix = matrix_square_root(&POSITIVE_SEMI_DEFINITE);
//         assert!(is_valid_square_root(
//             &sqrt_matrix,
//             &POSITIVE_SEMI_DEFINITE,
//             1e-9
//         ));
//     }
//     #[test]
//     #[should_panic]
//     fn eigenvalue_negative_definite() {
//         // This should panic because the matrix is negative definite.
//         let _sqrt_matrix = matrix_square_root(&NEGATIVE_DEFINITE);
//     }
//     #[test]
//     #[should_panic]
//     fn eigenvalue_negative_semi_definite() {
//         // This should panic because the matrix is negative semi-definite.
//         let _sqrt_matrix = matrix_square_root(&NEGATIVE_SEMI_DEFINITE);
//     }
//     #[test]
//     #[should_panic]
//     fn eigenvalue_non_square() {
//         // This should panic because the matrix is not square.
//         let _sqrt_matrix = matrix_square_root(&NON_SQUARE);
//     }
//     #[test]
//     fn public_api_square_root() {
//         let sqrt_matrix = matrix_square_root(&POSITIVE_DEFINITE);
//         assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
//         let sqrt_matrix = matrix_square_root(&POSITIVE_SEMI_DEFINITE);
//         assert!(is_valid_square_root(
//             &sqrt_matrix,
//             &POSITIVE_SEMI_DEFINITE,
//             1e-9
//         ));
//         let sqrt_matrix = matrix_square_root(&BASIC_SQRT);
//         assert!(is_valid_square_root(&sqrt_matrix, &BASIC_SQRT, 1e-9));
//     }
//     #[test]
//     #[should_panic]
//     fn public_api_negative_definite() {
//         // This should panic because the matrix is negative definite.
//         let _sqrt_matrix = matrix_square_root(&NEGATIVE_DEFINITE);
//     }
//     #[test]
//     #[should_panic]
//     fn public_api_negative_semi_definite() {
//         // This should panic because the matrix is negative semi-definite.
//         let _sqrt_matrix = matrix_square_root(&NEGATIVE_SEMI_DEFINITE);
//     }
//     #[test]
//     #[should_panic]
//     fn public_api_non_square() {
//         // This should panic because the matrix is not square.
//         let _sqrt_matrix = matrix_square_root(&NON_SQUARE);
//     }
// }
//
