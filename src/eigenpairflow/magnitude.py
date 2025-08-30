import numpy as np

def get_zero_indices(Lambdas):
    """
    Identifies indices of eigenvalues that cross zero.

    Args:
        Lambdas (list of np.ndarray): A list of diagonal eigenvalue matrices.

    Returns:
        list: A list of integer indices for eigenvalues that cross zero.
    """
    eigenvalues_traces = np.array([np.diag(L) for L in Lambdas])
    zero_indices = []
    for i in range(eigenvalues_traces.shape[1]):
        lambda_i = eigenvalues_traces[:, i]
        if np.amin(lambda_i) < 0.0 < np.amax(lambda_i):
            zero_indices.append(i)
    return zero_indices


def calculate_magnitudes(Qs, Lambdas, D):
    """
    Calculate the magnitudes for each time step.

    Args:
        Qs (list of np.ndarray): List of eigenvector matrices.
        Lambdas (list of np.ndarray): List of eigenvalue matrices.
        D (np.ndarray): The distance matrix.

    Returns:
        np.ndarray: An array of magnitude values for each time step.
    """
    magnitudes = []
    for i in range(len(Qs)):
        Q_t = Qs[i]
        Lambda_t = Lambdas[i]

        with np.errstate(divide='ignore'):
            inv_diag_lambda = 1.0 / np.diag(Lambda_t)

        Lambda_inverse = np.diag(inv_diag_lambda)

        v = Q_t.T @ np.ones(D.shape[0])
        mag = v.T @ Lambda_inverse @ v
        magnitudes.append(mag)
    return np.array(magnitudes)


def calculate_pseudo_magnitudes(Qs, Lambdas, D, zero_indices):
    """
    Calculate the pseudo-magnitudes for each time step.

    Args:
        Qs (list of np.ndarray): List of eigenvector matrices.
        Lambdas (list of np.ndarray): List of eigenvalue matrices.
        D (np.ndarray): The distance matrix.
        zero_indices (list): List of indices of eigenvalues that cross zero.

    Returns:
        np.ndarray: An array of pseudo-magnitude values for each time step.
    """
    pseudo_magnitudes = []
    for i in range(len(Qs)):
        Q_t = Qs[i]
        Lambda_t = Lambdas[i]

        with np.errstate(divide='ignore'):
            inv_diag_lambda = 1.0 / np.diag(Lambda_t)

        pseudo_Lambda_inverse_diag = inv_diag_lambda
        if zero_indices:
            pseudo_Lambda_inverse_diag[zero_indices] = 0

        pseudo_Lambda_inverse = np.diag(pseudo_Lambda_inverse_diag)

        v = Q_t.T @ np.ones(D.shape[0])
        pseudo_mag = v.T @ pseudo_Lambda_inverse @ v
        pseudo_magnitudes.append(pseudo_mag)
    return np.array(pseudo_magnitudes)
