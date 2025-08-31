import numpy as np
from eigenpairflow.magnitude import calculate_magnitudes


def test_calculate_magnitudes():
    """
    Tests the calculate_magnitudes function.
    """
    # Test case 1: Simple, non-zero eigenvalues
    Qs_1 = [np.eye(2)]
    Lambdas_1 = [np.diag([2.0, 4.0])]
    D_1 = np.zeros((2, 2))  # D is needed to get the shape for np.ones
    magnitudes_1 = calculate_magnitudes(Qs_1, Lambdas_1, D_1)
    # Expected: (1/2 + 1/4) = 0.75
    np.testing.assert_allclose(magnitudes_1, [0.75])

    # Test case 2: Eigenvectors are not identity
    Qs_2 = [
        np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
    ]
    Lambdas_2 = [np.diag([1.0, 2.0])]
    D_2 = np.zeros((2, 2))
    # v = Q.T @ [1, 1] = [sqrt(2), 0]
    # v.T @ inv(L) @ v = [sqrt(2), 0] @ [[1, 0], [0, 0.5]] @ [sqrt(2), 0].T = 2
    magnitudes_2 = calculate_magnitudes(Qs_2, Lambdas_2, D_2)
    np.testing.assert_allclose(magnitudes_2, [2.0])

    # Test case 3: With a zero eigenvalue, expecting infinity
    Qs_3 = [np.eye(2)]
    Lambdas_3 = [np.diag([0.0, 2.0])]
    D_3 = np.zeros((2, 2))
    magnitudes_3 = calculate_magnitudes(Qs_3, Lambdas_3, D_3)
    assert len(magnitudes_3) == 1
    assert np.isinf(magnitudes_3[0])
