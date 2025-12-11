import os
import numpy as np
import pytest

from eigenpairflow.tracking import singularpairtrack
from eigenpairflow.types import SVTrackingResults


# --- Test Helper Functions ---


def create_smooth_matrix_func(n=3, return_components=False):
    """Creates a function that returns a smoothly evolving matrix for SVD tracking.
    Can also return the true, continuous components of the SVD.
    """

    def A_func(t):
        current_scale = 1.0 + 0.5 * np.sin(t / 3.0)
        rotation_speed = 1.0
        theta = t * rotation_speed

        U_base = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        V_base = np.array(
            [
                [np.cos(-theta / 2.0), -np.sin(-theta / 2.0)],
                [np.sin(-theta / 2.0), np.cos(-theta / 2.0)],
            ]
        )

        if n > 2:
            U_full = np.eye(n)
            U_full[:2, :2] = U_base
            V_full = np.eye(n)
            V_full[:2, :2] = V_base
            S_diag_vals = np.array(
                [current_scale * 3.0, current_scale * 1.5] + [1.0] * (n - 2)
            )
            S_full = np.diag(S_diag_vals)
        else:
            U_full = U_base
            V_full = V_base
            S_diag_vals = np.array([current_scale * 3.0, current_scale * 1.5])
            S_full = np.diag(S_diag_vals)

        A_matrix = U_full @ S_full @ V_full.T

        if return_components:
            return A_matrix, U_full, S_diag_vals, V_full
        return A_matrix

    return A_func


def create_degenerate_matrix_func(n=2):
    """Creates a function that returns a matrix where singular values become close."""

    def A_func(t):
        sv1 = 2.0 + 1.0 * np.sin(t / 2.0)
        sv2 = 1.0 + 1.0 * np.cos(t / 2.0)

        theta = t * np.pi / 5.0
        U_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        V_mat = np.array(
            [
                [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                [np.sin(theta / 2.0), np.cos(theta / 2.0)],
            ]
        )

        if n > 2:
            U_full = np.eye(n)
            U_full[:2, :2] = U_mat
            V_full = np.eye(n)
            V_full[:2, :2] = V_mat
            S_diag_vals = np.array([sv1, sv2] + [1.0] * (n - 2))
            S_full = np.diag(S_diag_vals)
        else:
            U_full = U_mat
            V_full = V_mat
            S_full = np.diag([sv1, sv2])

        A_matrix = U_full @ S_full @ V_full.T
        return A_matrix

    return A_func


# --- Test Cases ---


@pytest.mark.parametrize("n", [2, 3])
def test_singularpairtrack_basic(n):
    """Tests basic functionality and continuity of singularpairtrack."""
    t_eval = np.linspace(0, 1, 100)
    A_func_with_components = create_smooth_matrix_func(n=n, return_components=True)
    A_func = create_smooth_matrix_func(n=n, return_components=False)

    # Generate continuous ground truth directly from the constructor
    true_Us, true_Ss, true_Vs = [], [], []
    for t in t_eval:
        _, U, S, V = A_func_with_components(t)
        true_Us.append(U)
        true_Ss.append(S)
        true_Vs.append(V)

    # Run the tracking function, providing the initial true decomposition
    results = singularpairtrack(
        A_func, t_eval, U0=true_Us[0], S0=true_Ss[0], V0=true_Vs[0]
    )

    # Assertions
    assert isinstance(results, SVTrackingResults)
    assert results.success
    assert results.message == "Ogita-SVD Tracking"
    assert np.allclose(results.t_eval, t_eval)
    assert len(results.Us) == len(t_eval)
    assert len(results.Ss) == len(t_eval)
    assert len(results.Vs) == len(t_eval)

    # Check shapes of the results
    for i in range(len(t_eval)):
        assert results.Us[i].shape == (n, n)
        assert results.Ss[i].shape == (n,)
        assert results.Vs[i].shape == (n, n)

    # Check reconstruction error
    assert results.norm_errors is not None
    assert len(results.norm_errors) == len(t_eval)
    assert np.all(results.norm_errors < 1e-6)

    # Check if the tracked SVD is close to the true continuous SVD
    for i in range(len(t_eval)):
        # Check singular values - allow for reordering (compare sorted values)
        assert np.allclose(np.sort(results.Ss[i]), np.sort(true_Ss[i]))

        # Check singular vectors, allowing for sign flips
        # The dot product of corresponding vectors should be close to +/- 1
        dot_U = np.abs(np.sum(results.Us[i] * true_Us[i], axis=0))
        dot_V = np.abs(np.sum(results.Vs[i] * true_Vs[i], axis=0))
        assert np.allclose(dot_U, 1.0)
        assert np.allclose(dot_V, 1.0)


def test_singularpairtrack_degenerate_values():
    """Tests singularpairtrack when singular values become close."""
    t_eval = np.linspace(0, 2 * np.pi, 10)
    A_func = create_degenerate_matrix_func(n=2)

    results = singularpairtrack(A_func, t_eval, tol=1e-8)

    assert isinstance(results, SVTrackingResults)
    assert results.success
    assert len(results.t_eval) == len(t_eval)
    assert len(results.Us) == len(t_eval)
    assert len(results.Ss) == len(t_eval)
    assert len(results.Vs) == len(t_eval)
    assert results.norm_errors is not None

    # Check that the norm errors are small, indicating successful tracking.
    assert np.all(results.norm_errors < 1e-5)  # norm_errors is now a np.array

    # Check for a point where singular values are close, i.e., at t=0
    zero_time_idx = 0
    sv1_at_zero = results.Ss[zero_time_idx][0]
    sv2_at_zero = results.Ss[zero_time_idx][1]
    assert abs(sv1_at_zero - sv2_at_zero) < 1e-5  # They should be very close at t=0


def test_singularpairtrack_empty_t_eval():
    """Tests singularpairtrack with an empty t_eval array."""
    t_eval = np.array([])
    A_func = create_smooth_matrix_func(n=2)
    results = singularpairtrack(A_func, t_eval)

    assert isinstance(results, SVTrackingResults)
    assert not results.success
    assert results.message == "No time points provided."
    assert results.t_eval.shape == (0,)
    assert results.Us is None
    assert results.Ss is None
    assert results.Vs is None
    assert results.norm_errors is None


def test_svd_results_methods():
    """Tests the __str__, save, and load methods of SVTrackingResults."""
    t_eval = np.array([0.0, 1.0])
    Us = [np.eye(2), np.eye(2) * -1]
    Ss = [np.array([2.0, 1.0]), np.array([1.5, 0.5])]
    Vs = [np.eye(2), np.eye(2)]
    norm_errors = np.array([1e-7, 1e-8])  # This is now a numpy array
    success = True
    message = "Test message"

    results = SVTrackingResults(
        t_eval=t_eval,
        Us=Us,
        Ss=Ss,
        Vs=Vs,
        norm_errors=norm_errors,
        success=success,
        message=message,
    )

    # Test __str__ method
    str_output = str(results)
    assert "SVTrackingResults Summary:" in str_output
    assert "success: True" in str_output
    assert "message: Test message" in str_output
    assert "Us: list of 2 np.ndarray(s), first shape: (2, 2)" in str_output
    # This assertion checks the formatting for Ss, which is a list of arrays.
    assert (
        "Ss: list of 2 np.ndarray(s), first shape: (2,)" in str_output
    )  # Corrected expectation for Ss format
    assert "Vs: list of 2 np.ndarray(s), first shape: (2, 2)" in str_output
    # This assertion checks the formatting for norm_errors, which is now expected to be a np.ndarray.
    assert "norm_errors: np.ndarray with shape (2,)" in str_output

    # Test save and load
    filepath = "./test_svd_results.pkl"
    results.save(filepath)

    loaded_results = SVTrackingResults.load(filepath)

    assert isinstance(loaded_results, SVTrackingResults)
    assert np.allclose(loaded_results.t_eval, results.t_eval)
    assert np.allclose(loaded_results.Ss[0], results.Ss[0])
    assert np.allclose(loaded_results.norm_errors, results.norm_errors)
    assert loaded_results.success == results.success
    assert loaded_results.message == results.message

    # Clean up the file
    os.remove(filepath)


def test_singularpairtrack_failure():
    """Tests singularpairtrack when the underlying SVD refinement fails."""

    def failing_A_func(t):
        if t > 5:
            raise ValueError("Simulated error in matrix generation")
        return np.eye(2) * (1 + t)

    t_eval = np.linspace(0, 10, 5)
    results = singularpairtrack(failing_A_func, t_eval)

    assert isinstance(results, SVTrackingResults)
    assert not results.success
    assert "Simulated error" in results.message
    assert results.Us is None
    assert results.Ss is None
    assert results.Vs is None
    assert results.norm_errors is None
