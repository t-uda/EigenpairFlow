import numpy as np
import scipy.linalg
from eigenpairflow.correction import ogita_aishima_refinement

def test_ogita_aishima_refinement_simple():
    # Create a random symmetric matrix
    n = 5
    A = np.random.rand(n, n)
    A = (A + A.T) / 2

    # Get the exact solution
    exact_eigvals, exact_eigvecs = scipy.linalg.eigh(A)

    # Create a perturbed initial guess
    X_hat = exact_eigvecs + np.random.rand(n, n) * 0.1

    # Refine the solution
    X_refined, D_refined = ogita_aishima_refinement(A, X_hat)

    # Check if the refined solution is closer to the exact solution
    initial_error = np.linalg.norm(X_hat - exact_eigvecs)
    refined_error = np.linalg.norm(X_refined - exact_eigvecs)

    assert refined_error < initial_error

def test_ogita_aishima_refinement_clustered():
    # Create a symmetric matrix with clustered eigenvalues
    n = 5
    A = np.diag([1.0, 1.0 + 1e-9, 3.0, 4.0, 5.0])

    # Add some off-diagonal noise
    noise = np.random.rand(n, n) * 1e-12
    noise = (noise + noise.T) / 2
    A = A + noise

    # Get the exact solution
    exact_eigvals, exact_eigvecs = scipy.linalg.eigh(A)

    # Create a perturbed initial guess
    X_hat = exact_eigvecs + np.random.rand(n, n) * 0.1

    # Refine the solution
    X_refined, D_refined = ogita_aishima_refinement(A, X_hat)

    # Check if the refined solution is closer to the exact solution
    initial_error = np.linalg.norm(X_hat - exact_eigvecs)
    refined_error = np.linalg.norm(X_refined - exact_eigvecs)

    assert refined_error < initial_error
