import numpy as np
import scipy.linalg
from eigenpairflow.correction import ogita_aishima_refinement


def _residual_norm(A: np.ndarray, X: np.ndarray) -> float:
    """固有値分解の残差ノルムを計算する。"""
    D = np.diag(np.diag(X.T @ A @ X))
    return np.linalg.norm(A @ X - X @ D)


def test_ogita_aishima_refinement_simple():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2

    # Exact eigenvectors for constructing a perturbed initial guess
    _, exact_eigvecs = scipy.linalg.eigh(A)
    X_hat = exact_eigvecs + rng.standard_normal((n, n)) * 0.1

    X_refined, D_refined = ogita_aishima_refinement(A, X_hat)

    initial_residual = _residual_norm(A, X_hat)
    refined_residual = np.linalg.norm(A @ X_refined - X_refined @ D_refined)

    assert refined_residual < initial_residual


def test_ogita_aishima_refinement_clustered():
    n = 5
    A = np.diag([1.0, 1.0 + 1e-9, 3.0, 4.0, 5.0]).astype(float)
    # deterministic tiny coupling for the first two eigenvalues
    A[0, 1] = A[1, 0] = 1e-12

    # Exact eigenvectors and a perturbed initial guess
    _, exact_eigvecs = scipy.linalg.eigh(A)
    rng = np.random.default_rng(0)
    X_hat = exact_eigvecs + rng.standard_normal((n, n)) * 0.1

    X_refined, D_refined = ogita_aishima_refinement(A, X_hat)

    initial_residual = _residual_norm(A, X_hat)
    refined_residual = np.linalg.norm(A @ X_refined - X_refined @ D_refined)

    assert refined_residual < initial_residual
