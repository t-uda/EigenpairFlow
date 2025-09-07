import numpy as np
from eigenpairflow.correction import match_decompositions


def test_match_decompositions_reorders_and_aligns_signs():
    # predicted decomposition with swapped order
    predicted_eigvals = np.array([2.0, 1.0])
    predicted_eigvecs = np.eye(2)

    # exact decomposition in ascending order with a sign flip on the first vector
    exact_eigvals = np.array([1.0, 2.0])
    exact_eigvecs = np.array([[0.0, -1.0], [1.0, 0.0]])

    matched_vals, matched_vecs = match_decompositions(
        predicted_eigvals, predicted_eigvecs, exact_eigvals, exact_eigvecs
    )

    assert np.allclose(matched_vals, predicted_eigvals)
    assert np.allclose(matched_vecs, predicted_eigvecs)
