import numpy as np
import os
import tempfile
from eigenpairflow import track_and_analyze_eigenvalue_decomposition, create_n_partite_graph, EigenTrackingResults

def test_eigen_tracking_results_str_representation():
    """
    Tests the __str__ method of the EigenTrackingResults class.
    """
    # Create a dummy successful result object
    results = EigenTrackingResults(
        t_eval=np.array([0.1, 0.2]),
        Qs=[np.eye(2), np.eye(2)],
        Lambdas=[np.diag([1, 2]), np.diag([1, 2])],
        errors=np.array([0.0, 0.0]),
        zero_indices=[],
        success=True,
        message="Completed successfully.",
        state=0,
        errors_before_correction=None
    )

    s = str(results)
    assert "success: True" in s
    assert "message: Completed successfully." in s
    assert "t_eval: np.ndarray with shape (2,)" in s
    assert "Qs: list of 2 np.ndarray(s), first shape: (2, 2)" in s
    assert "Lambdas: list of 2 np.ndarray(s), first shape: (2, 2)" in s

    # Create a dummy failed result object
    failed_results = EigenTrackingResults(
        t_eval=None, Qs=None, Lambdas=None, errors=None, zero_indices=None,
        success=False, message="Graph is disconnected.", state=None,
        errors_before_correction=None
    )
    s_failed = str(failed_results)
    assert "EigenTracking failed: Graph is disconnected." in s_failed


def test_eigen_tracking_results_serialization():
    """
    Tests the save and load methods of the EigenTrackingResults class.
    """
    # Define partition sizes and edge lengths for a sample graph
    partitions = [2, 2]
    lengths = {(0, 1): 1.0}

    # Create the n-partite graph
    G = create_n_partite_graph(partitions, lengths)

    # Run the analysis function
    results, _ = track_and_analyze_eigenvalue_decomposition(G, apply_correction=False)

    assert results.success

    # Use a temporary file for saving and loading
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "results.joblib")
        results.save(filepath)

        # Check that the file was created
        assert os.path.exists(filepath)

        # Load the results back
        loaded_results = EigenTrackingResults.load(filepath)

        # Compare the original and loaded results
        assert isinstance(loaded_results, EigenTrackingResults)
        assert results.success == loaded_results.success
        assert results.message == loaded_results.message

        # Compare numpy arrays
        np.testing.assert_array_equal(results.t_eval, loaded_results.t_eval)
        for i in range(len(results.Qs)):
            np.testing.assert_array_equal(results.Qs[i], loaded_results.Qs[i])
            np.testing.assert_array_equal(results.Lambdas[i], loaded_results.Lambdas[i])
        np.testing.assert_array_equal(results.errors, loaded_results.errors)
        assert results.zero_indices == loaded_results.zero_indices
