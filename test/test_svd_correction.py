import numpy as np
from eigenpairflow.correction import ogita_svd_refinement


def test_svd_refinement_simple():
    # Create a simple random matrix
    np.random.seed(42)
    A = np.random.randn(5, 5)

    # True SVD
    U_true, S_true, Vt_true = np.linalg.svd(A)
    V_true = Vt_true.T

    # Create a perturbed guess
    # Perturb U and V by small rotation
    theta = 0.1
    c, s = np.cos(theta), np.sin(theta)
    Rot = np.eye(5)
    Rot[0, 0] = c
    Rot[0, 1] = -s
    Rot[1, 0] = s
    Rot[1, 1] = c

    U_guess = U_true @ Rot
    V_guess = V_true @ Rot
    S_guess = S_true + np.random.randn(5) * 0.1

    # Run refinement
    # Note: Since we only implement the function now, this test will fail to import initially.
    # But this is the target behavior.
    U_ref, S_ref, V_ref = ogita_svd_refinement(
        A, U_guess, S_guess, V_guess, max_iter=20, tol=1e-10
    )

    # Check reconstruction
    A_rec = U_ref @ np.diag(S_ref) @ V_ref.T
    rec_error = np.linalg.norm(A - A_rec)
    print(f"Reconstruction error: {rec_error}")
    assert rec_error < 1e-12

    # Check orthogonality
    assert np.linalg.norm(U_ref.T @ U_ref - np.eye(5)) < 1e-12
    assert np.linalg.norm(V_ref.T @ V_ref - np.eye(5)) < 1e-12

    # Check convergence to true values (up to sign/phase)
    # Singular values should be close
    print(f"S_true: {S_true}")
    print(f"S_ref: {S_ref}")
    assert np.allclose(np.sort(S_ref), np.sort(S_true), atol=1e-10)


def test_svd_refinement_degenerate():
    # Matrix with close singular values
    # Sigma = diag(1.0, 1.0 + 1e-8, 0.5, ...)
    U_true = np.eye(4)
    V_true = np.eye(4)
    S_true = np.array([1.0, 1.0 + 1e-9, 0.5, 0.1])

    A = U_true @ np.diag(S_true) @ V_true.T

    # Guess
    U_guess = U_true.copy()
    V_guess = V_true.copy()
    S_guess = S_true.copy() + 1e-5

    # Run refinement
    U_ref, S_ref, V_ref = ogita_svd_refinement(A, U_guess, S_guess, V_guess, tol=1e-8)

    # Check reconstruction
    A_rec = U_ref @ np.diag(S_ref) @ V_ref.T
    rec_error = np.linalg.norm(A - A_rec)
    print(f"Degenerate Reconstruction error: {rec_error}")
    assert rec_error < 1e-12

    # Check if close singular values are preserved
    diff = np.abs(S_ref[0] - S_ref[1])
    print(f"Singular value diff: {diff}")
    assert diff < 2e-9  # Should remain close
