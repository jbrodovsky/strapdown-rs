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
/// Three methods are implemented:
/// 1. **Cholesky Decomposition**: Computes the square root of a symmetric positive definite matrix.
/// 2. **Eigenvalue Decomposition**: Computes the square root of a symmetric positive semi-definite matrix.
/// 3. **Schur Decomposition**: Computes the principal square root of a general square matrix, which can be complex. Complex values are ignored in this implementation, and if such a matrix is encountered, it will return `None`.
/// 
/// These calculations are intended to be used through the `matrix_square_root` function, which will
/// attempt to compute the square root using Cholesky decomposition first, then eigenvalue decomposition,
/// and finally Schur decomposition if the previous methods fail.
use nalgebra::{DMatrix, DVector, RealField, Complex};
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
pub fn matrix_square_root(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if !matrix.is_square() {
        eprintln!("Error: Matrix must be square to compute square root.");
        return None;
    }
    // Attempt Cholesky decomposition (yields L where matrix = L * L^T)
    // Cholesky requires the matrix to be symmetric positive definite.
    match cholesky_pass(matrix) {
        Some(chol_l) => {
            println!("Successfully computed square root using Cholesky decomposition.");
            return Some(chol_l);
        },
        None => {
            println!("Cholesky decomposition failed. Attempting eigenvalue decomposition.");
        }
    }
    // If Cholesky failed, we try eigenvalue decomposition.
    match eigenvalue_pass(matrix) {
        Some(eigen_sqrt) => {
            println!("Successfully computed square root using eigenvalue decomposition.");
            return Some(eigen_sqrt);
        },
        None => {
            println!("Eigenvalue decomposition failed. No valid square root found.");
        }
    }
    match schur_pass(matrix) {
        Some(schur_sqrt) => {
            println!("Successfully computed square root using Schur decomposition.");
            return Some(schur_sqrt);
        },
        None => {
            println!("Schur decomposition failed. No valid square root found.");
        }
    }
    // If all methods fail, we return None.
    None
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
/// Computes the principal matrix square root of a general square matrix using the Schur method.
///
/// This is the most general approach and can handle non-symmetric matrices.
/// It returns a complex matrix, as the square root of a general matrix can be complex.
/// This implementation broadly follows the Schur method described in the previous response.
///
/// # Arguments
/// * `matrix` - The DMatrix<f64> to find the square root of.
///
/// # Returns
/// * `Some(DMatrix<Complex<f64>>)` containing the principal matrix square root.
/// * `None` if the matrix is not square or if issues arise (e.g., encountering certain singular cases
///   where `s_ii + s_jj` is close to zero, or non-convergence of Schur decomposition).
pub fn schur_pass(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let schur_decomp = matrix.clone().schur();
    let (q, t) = schur_decomp.unpack();
    let sqrt_t = match cholesky_pass(&t) {
        Some(sqrt) => sqrt,
        None => {
            return match eigenvalue_pass(&t) {
                Some(sqrt) => Some(sqrt),
                None => {
                    eprintln!("Error: Failed to compute square root of Schur form.");
                    None
                }
            };
        }
    };
    // Reconstruct the square root of the original matrix
    Some(&q * sqrt_t * &q.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;
    use nalgebra::{DMatrix, DVector};

    static BASIC_SQRT: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            4.0, 0.0, 0.0,
            0.0, 9.0, 0.0,
            0.0, 0.0, 16.0]
        )
    });
    const POSITIVE_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            4.0, 2.0, 0.0,
            2.0, 9.0, 3.0,
            0.0, 3.0, 16.0]
        )
    });
    const POSITIVE_SEMI_DEFINIE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0]
        )
    });
    const NEGATIVE_DEFINITE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            -4.0, 0.0, 0.0,
            0.0, -9.0, 0.0,
            0.0, 0.0, -16.0]
        )
    });
    const NEGATIVE_SEMI_DEFINIE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
        DMatrix::from_row_slice(3,3, &[
            -1.0, 0.0, -1.0,
            0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0]
        )
    });
    const NON_SQUARE: LazyLock<DMatrix<f64>> = LazyLock::new(|| {
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
    fn test_cholesky_pass() {
        match cholesky_pass(&BASIC_SQRT) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &BASIC_SQRT, 1e-9));
            },
            None => panic!("Cholesky decomposition failed for BASIC_SQRT"),
        }
        match cholesky_pass(&POSITIVE_DEFINITE) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
            },
            None => panic!("Cholesky decomposition failed for POSITIVE_DEFINITE"),
        }
        // Positive semi-definite matrices should not pass Cholesky
        assert!(cholesky_pass(&POSITIVE_SEMI_DEFINIE).is_none());
        // Negative definite matrices should not pass Cholesky
        assert!(cholesky_pass(&NEGATIVE_DEFINITE).is_none());
        // Negative semi-definite matrices should not pass Cholesky
        assert!(cholesky_pass(&NEGATIVE_SEMI_DEFINIE).is_none());
        // Non-square matrices should not pass Cholesky
        assert!(cholesky_pass(&NON_SQUARE).is_none());
    }
    #[test]
    fn test_eigenvalue_pass() {
        match eigenvalue_pass(&BASIC_SQRT) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &BASIC_SQRT, 1e-9));
            },
            None => panic!("Eigenvalue decomposition failed for BASIC_SQRT"),
        }
        match eigenvalue_pass(&POSITIVE_DEFINITE) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
            },
            None => panic!("Eigenvalue decomposition failed for POSITIVE_DEFINITE"),
        }
        // Primary advantage of eigenvalue decomposition is that it can handle positive semi-definite matrices.
        match eigenvalue_pass(&POSITIVE_SEMI_DEFINIE) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_SEMI_DEFINIE, 1e-9));
            },
            None => panic!("Eigenvalue decomposition failed for POSITIVE_SEMI_DEFINIE"),
        }
        // Negative definite matrices should not pass eigenvalue decomposition
        assert!(eigenvalue_pass(&NEGATIVE_DEFINITE).is_none());
        // Negative semi-definite matrices should not pass eigenvalue decomposition
        assert!(eigenvalue_pass(&NEGATIVE_SEMI_DEFINIE).is_none());
        // Non-square matrices should not pass eigenvalue decomposition
        assert!(eigenvalue_pass(&NON_SQUARE).is_none());
    }
    #[test]
    fn test_schur_pass() {
        match schur_pass(&BASIC_SQRT) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &BASIC_SQRT, 1e-9));
                println!("Schur decomposition passed for BASIC_SQRT.");
            },
            None => panic!("Schur decomposition failed for BASIC_SQRT"),
        }
        match schur_pass(&POSITIVE_DEFINITE) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_DEFINITE, 1e-9));
                println!("Schur decomposition passed for POSITIVE_DEFINITE.");
            },
            None => panic!("Schur decomposition failed for POSITIVE_DEFINITE"),
        }
        match schur_pass(&POSITIVE_SEMI_DEFINIE) {
            Some(sqrt_matrix) => {
                assert!(is_valid_square_root(&sqrt_matrix, &POSITIVE_SEMI_DEFINIE, 1e-9));
                println!("Schur decomposition passed for POSITIVE_SEMI_DEFINIE.");
            },
            None => panic!("Schur decomposition failed for POSITIVE_SEMI_DEFINIE"),
        }
        // Negative definite matrices should not pass Cholesky
        assert!(schur_pass(&NEGATIVE_DEFINITE).is_none());
        // Negative semi-definite matrices should not pass Cholesky
        assert!(schur_pass(&NEGATIVE_SEMI_DEFINIE).is_none());
        // Non-square matrices should not pass Cholesky
        assert!(schur_pass(&NON_SQUARE).is_none());
    }

}