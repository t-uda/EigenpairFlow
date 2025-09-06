import numpy as np
import os
import tempfile
import networkx as nx
from eigenpairflow import eigenpairtrack, EigenTrackingResults
from eigenpairflow.graph import create_n_partite_graph


def test_eigen_tracking_results_str_representation():
    """
    Tests the __str__ method of the EigenTrackingResults class.
    """
    # Create a dummy successful result object
    results = EigenTrackingResults(
        t_eval=np.array([0.1, 0.2]),
        Qs=[np.eye(2), np.eye(2)],
        Lambdas=[np.diag([1, 2]), np.diag([1, 2])],
        norm_errors=np.array([0.0, 0.0]),
        success=True,
        message="Completed successfully.",
    )

    s = str(results)
    assert "success: True" in s
    assert "norm_errors: np.ndarray with shape (2,)" in s

    # Create a dummy failed result object
    failed_results = EigenTrackingResults(
        t_eval=None,
        Qs=None,
        Lambdas=None,
        norm_errors=None,
        success=False,
        message="Graph is disconnected.",
    )
    s_failed = str(failed_results)
    assert "EigenTracking failed: Graph is disconnected." in s_failed


def test_eigen_tracking_results_serialization():
    """
    Tests the save and load methods of the EigenTrackingResults class.
    """
    # 1. Create a sample graph and define the matrix function
    partitions = [2, 2]
    lengths = {(0, 1): 1.0}
    G = create_n_partite_graph(partitions, lengths)
    D = np.array(nx.floyd_warshall_numpy(G, weight="length"))

    def A_func(t):
        return np.exp(-t * D)

    def dA_func(t):
        return -D * np.exp(-t * D)

    # 2. Run the tracking
    t_start, t_end = 1.0, 0.1
    t_eval = np.linspace(t_start, t_end, 10)
    results = eigenpairtrack(
        A_func,
        dA_func,
        (t_start, t_end),
        t_eval,
        matrix_type="symmetric",
        method="eigh",
        correction_method=None,
    )

    assert results.success
    assert results.norm_errors is not None

    # 3. Use a temporary file for saving and loading
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "results.joblib")
        results.save(filepath)

        assert os.path.exists(filepath)

        loaded_results = EigenTrackingResults.load(filepath)

        assert isinstance(loaded_results, EigenTrackingResults)
        assert results.success == loaded_results.success
        np.testing.assert_array_equal(results.t_eval, loaded_results.t_eval)
        np.testing.assert_array_equal(results.norm_errors, loaded_results.norm_errors)
        for i in range(len(results.Qs)):
            np.testing.assert_array_equal(results.Qs[i], loaded_results.Qs[i])
            np.testing.assert_array_equal(results.Lambdas[i], loaded_results.Lambdas[i])
