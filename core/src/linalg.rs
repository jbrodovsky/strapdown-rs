/// Basic linear algebra utilities for matrix square root computation.
/// 
/// This module provides linear algebra utilities for matrix square root calculations in an attempt to
/// provide similar functionality to the `scipy.linalg.sqrtm` function in Python. That said, due to 
/// the applied nature of this crate, the problem is simplified to only handle square, positive 
/// definite, and positive semi-definite matrices. The inclusion of this module is to provide an implementation
/// of matrix square root calculations for the Unscented Kalman Filter (UKF) algorithm, which requires
/// the computation of the square root of a covariance matrix. Covariance matrices are symmetric and positive 
/// semi-definite, making them suitable for this approach.
/// 
/// Two methods are implemented:
/// 1. **Cholesky Decomposition**: Computes the square root of a symmetric positive definite matrix.
/// 2. **Eigenvalue Decomposition**: Computes the square root of a symmetric positive semi-definite matrix.
/// 
/// These calculations are intended to be used through the `matrix_square_root` function, which will
/// attempt to compute the square root using Cholesky decomposition first, then eigenvalue decomposition. If both
/// methods fail, then the method will panic.
use nalgebra::DMatrix;
use nalgebra::linalg::{Cholesky, SymmetricEigen};

/// Calculates a square root of a symmetric matrix.
///
/// Attempts Cholesky decomposition first (yielding L such that matrix = L * L^T).
/// If Cholesky fails (e.g., matrix is not positive definite), it attempts to compute
/// the square root using eigenvalue decomposition (S = V * sqrt(D) * V^T).
/// For eigenvalue decomposition, eigenvalues are clamped to be non-negative.
///
/// # Arguments
/// * `matrix` - The DMatrix<f64> to find the square root of. It's assumed to be symmetric and square.
///
/// # Returns
/// * `Some(DMatrix<f64>)` containing a matrix square root.
///   The result from Cholesky is lower triangular. The result from eigenvalue decomposition is symmetric.
///   In both cases, if the result is `M`, then `matrix` approx `M * M.transpose()`.
/// * `None` if the matrix is not square or another fundamental issue prevents computation (though
///   this implementation tries to be robust for positive semi-definite cases).
pub fn matrix_square_root(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    if !matrix.is_square() {
        panic!("Error: Matrix must be square to compute square root.");
    }
    // Attempt Cholesky decomposition (yields L where matrix = L * L^T)
    // Cholesky requires the matrix to be symmetric positive definite.
    match cholesky_pass(matrix) {
        Some(chol_l) => {
            return chol_l;
        },
        None => { 
            //println!("Cholesky decomposition failed. Attempting eigenvalue decomposition.");
        }
    }
    // If Cholesky failed, we try eigenvalue decomposition.
    match eigenvalue_pass(matrix) {
        Some(eigen_sqrt) => {
            eigen_sqrt
        },
        None => {
            panic!("Cholesky and Eigenvalue decomposition failed. No valid square root found.");
        }
    }
}
/// Attempts to compute the matrix square root using Cholesky decomposition.
///
/// This method is only applicable to symmetric positive definite matrices.
/// If successful, it returns the lower triangular matrix `L` such that `matrix = L * L.transpose()`.
/// 
/// When the computation _fails_ (e.g., the matrix is not positive definite or not square),
/// a None value is returned instead of panicking, permitting the public API to proceed to the
/// next method.
///
/// # Arguments
/// * `matrix` - The DMatrix<f64> to find the square root of. Assumed to be symmetric and square.
///
/// # Returns
/// * `Some(DMatrix<f64>)` containing the lower triangular Cholesky factor `L`.
/// * `None` if the matrix is not positive definite or not square.
fn cholesky_pass(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if !matrix.is_square() {
        eprintln!("Error: Matrix must be square for Cholesky decomposition.");
        return None;
    }
    matrix.clone().cholesky().map(|chol: Cholesky<f64, nalgebra::Dyn>| chol.l())
}
/// Computes a symmetric matrix square root using eigenvalue decomposition.
///
/// This method is suitable for symmetric positive semi-definite matrices.
/// It returns a symmetric matrix `S` such that `matrix = S * S`.
/// Eigenvalues are clamped to be non-negative to handle positive semi-definite cases
/// and minor numerical inaccuracies.
/// 
/// When the computation _fails_ (e.g., the matrix is not positive definite or not square),
/// a None value is returned instead of panicking, permitting the public API to proceed to the
/// next method.
/// 
/// # Arguments
/// * `matrix` - The DMatrix<f64> to find the square root of. Assumed to be symmetric and square.
///
/// # Returns
/// * `Some(DMatrix<f64>)` containing the symmetric matrix square root `S`.
/// * `None` if the matrix is not square (though this should be checked by the caller for symmetry assumptions).
fn eigenvalue_pass(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if !matrix.is_square() {
        eprintln!("Error: Matrix must be square for eigenvalue decomposition based square root.");
        return None;
    }
    // For eigenvalue decomposition of a symmetric matrix,
    // we use `symmetric_eigen`. This returns real eigenvalues and orthogonal eigenvectors.
    let eigen_decomposition: SymmetricEigen<f64, nalgebra::Dyn> = matrix.clone().symmetric_eigen();
    let eigenvalues = eigen_decomposition.eigenvalues;
    let eigenvectors = eigen_decomposition.eigenvectors;

    // Check for significantly negative eigenvalues, indicating non-positive semi-definiteness.
    // While we clamp them, a warning is useful for diagnosis.
    if eigenvalues.iter().any(|&val| val < -1e-9) {
        eprintln!("Warning: Negative eigenvalues encountered during eigenvalue decomposition. The input matrix was not positive semi-definite.");
        return None;
    }

    // Create diagonal matrix of sqrt(eigenvalues), clamping eigenvalues to be non-negative.
    // `DMatrix::from_diagonal` takes a DVector.
    let sqrt_eigenvalues_diag_vec = eigenvalues.map(|val| val.max(0.0).sqrt());
    let sqrt_eigenvalues_diag = DMatrix::from_diagonal(&sqrt_eigenvalues_diag_vec);

    // Reconstruct the square root: S = V * sqrt(D) * V^T
    // This S will be symmetric, and S * S = matrix (or S * S^T = matrix).
    let sqrt_m = eigenvectors.clone() * sqrt_eigenvalues_diag * eigenvectors.transpose();

    Some(sqrt_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;
    use nalgebra::DMatrix;

    static BASIC_SQRT: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            4.0, 0.0, 0.0,
            0.0, 9.0, 0.0,
            0.0, 0.0, 16.0]
        )
    });
    static POSITIVE_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            4.0, 2.0, 0.0,
            2.0, 9.0, 3.0,
            0.0, 3.0, 16.0]
        )
    });
    static POSITIVE_SEMI_DEFINIE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0]
        )
    });
    static NEGATIVE_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            -4.0, 0.0, 0.0,
            0.0, -9.0, 0.0,
            0.0, 0.0, -16.0]
        )
    });
    static NEGATIVE_SEMI_DEFINIE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            -1.0, 0.0, -1.0,
            0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0]
        )
    });
    static NON_SQUARE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(2,3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0]
        )
    });

    /// Helper function to verify if a matrix is a valid square root of another matrix.
    /// Returns true if sqrt_matrix * sqrt_matrix.T ≈ original_matrix within tolerance.
    fn is_valid_square_root(sqrt_matrix: &DMatrix<f64>, original_matrix: &DMatrix<f64>, tolerance: f64) -> bool {
        let reconstructed = sqrt_matrix * sqrt_matrix.transpose();
        
        if reconstructed.nrows() != original_matrix.nrows() || 
           reconstructed.ncols() != original_matrix.ncols() {
            return false;
        }
        
        for i in 0..original_matrix.nrows() {
            for j in 0..original_matrix.ncols() {
                if (reconstructed[(i, j)] - original_matrix[(i, j)]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
    // Test matrix square root calculation
    #[test]
    fn cholesky_square_root() {
        let sqrt_matrix = matrix_square_root(&BASIC_SQRT);
        assert!(is_valid_square_root(&sqrt_matrix, &BASIC_SQRT, 1e-9));
    }
    #[test]
    fn cholesky_positive_definite() {
        let sqrt_matrix = matrix_square_root(&POSITIVE_DEFINITE);
        assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
    }
    #[test]
    #[should_panic]
    fn cholesky_negative_definite() {
        // This should panic because the matrix is negative definite.
        let _sqrt_matrix = matrix_square_root(&NEGATIVE_DEFINITE);
    }
    #[test]
    #[should_panic]
    fn cholesky_negative_semi_definite() {
        // This should panic because the matrix is negative semi-definite.
        let _sqrt_matrix = matrix_square_root(&NEGATIVE_SEMI_DEFINIE);
    }
    #[test]
    #[should_panic]
    fn cholesky_non_square() {
        // This should panic because the matrix is not square.
        let _sqrt_matrix = matrix_square_root(&NON_SQUARE);
    }
    #[test]
    fn eigenvalue_square_root() {
        let sqrt_matrix = matrix_square_root(&POSITIVE_SEMI_DEFINIE);
        assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_SEMI_DEFINIE, 1e-9));
    }
    #[test]
    fn eigenvalue_positive_definite() {
        let sqrt_matrix = matrix_square_root(&POSITIVE_DEFINITE);
        assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
    }
    #[test]
    fn eigenvalue_positive_semi_definite() {
        let sqrt_matrix = matrix_square_root(&POSITIVE_SEMI_DEFINIE);
        assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_SEMI_DEFINIE, 1e-9));
    }
    #[test]
    #[should_panic]
    fn eigenvalue_negative_definite() {
        // This should panic because the matrix is negative definite.
        let _sqrt_matrix = matrix_square_root(&NEGATIVE_DEFINITE);
    }
    #[test]
    #[should_panic]
    fn eigenvalue_negative_semi_definite() {
        // This should panic because the matrix is negative semi-definite.
        let _sqrt_matrix = matrix_square_root(&NEGATIVE_SEMI_DEFINIE);
    }
    #[test]
    #[should_panic]
    fn eigenvalue_non_square() {
        // This should panic because the matrix is not square.
        let _sqrt_matrix = matrix_square_root(&NON_SQUARE);
    }
    #[test]
    fn public_api_square_root() {
        let sqrt_matrix = matrix_square_root(&POSITIVE_DEFINITE);
        assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
        let sqrt_matrix = matrix_square_root(&POSITIVE_SEMI_DEFINIE);
        assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_SEMI_DEFINIE, 1e-9));
        let sqrt_matrix = matrix_square_root(&BASIC_SQRT);
        assert!(is_valid_square_root(&sqrt_matrix, &BASIC_SQRT, 1e-9));
    }
    #[test]
    #[should_panic]
    fn public_api_negative_definite() {
        // This should panic because the matrix is negative definite.
        let _sqrt_matrix = matrix_square_root(&NEGATIVE_DEFINITE);
    }
    #[test]
    #[should_panic]
    fn public_api_negative_semi_definite() {
        // This should panic because the matrix is negative semi-definite.
        let _sqrt_matrix = matrix_square_root(&NEGATIVE_SEMI_DEFINIE);
    }
    #[test]
    #[should_panic]
    fn public_api_non_square() {
        // This should panic because the matrix is not square.
        let _sqrt_matrix = matrix_square_root(&NON_SQUARE);
    }
}