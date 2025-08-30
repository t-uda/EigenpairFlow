import numpy as np

def calculate_magnitudes_and_pseudo(Lambdas, t_eval):
    """
    Calculates magnitudes and pseudo-magnitudes from eigenvalue data.

    The magnitude M_i(t) is defined as M_i(t) = - (1/t) * log(lambda_i(t)),
    where lambda_i(t) is the i-th eigenvalue at time t.

    Pseudo-magnitudes are the indices of eigenvalues that cross zero.

    Args:
        Lambdas (list of np.ndarray): List of diagonal eigenvalue matrices.
        t_eval (np.ndarray): Array of time points.

    Returns:
        tuple[np.ndarray, list[int]]:
            - magnitudes: A 2D array where magnitudes[i, j] is the magnitude of the j-th
                          eigenvalue at time t_eval[i].
            - pseudo_magnitudes: A list of indices of eigenvalues that cross zero.
    """
    eigenvalues_traces = np.array([np.diag(L) for L in Lambdas])

    # Calculate magnitudes
    # We need to handle t=0 and lambda <= 0 for the log
    # Add a small epsilon to t_eval to avoid division by zero
    t_broadcast = t_eval[:, np.newaxis]

    # Magnitudes are not defined for lambda <= 0, they will be nan.
    with np.errstate(divide='ignore', invalid='ignore'):
        magnitudes = - (1 / t_broadcast) * np.log(eigenvalues_traces)

    # Calculate pseudo-magnitudes (indices of eigenvalues that cross zero)
    pseudo_magnitudes = []
    for i in range(eigenvalues_traces.shape[1]):
        lambda_i = eigenvalues_traces[:, i]
        # Check if the sign changes
        if np.any(lambda_i > 0) and np.any(lambda_i < 0):
            pseudo_magnitudes.append(i)

    return magnitudes, pseudo_magnitudes
