from eigenpairflow import (
    track_and_analyze_eigenvalue_decomposition,
    create_n_partite_graph,
)


def test_track_and_analyze_eigenvalue_decomposition_runs():
    """
    Tests that the main analysis function runs without errors on a sample graph.
    """
    # Define partition sizes and edge lengths for a sample graph
    partitions = [3, 2, 1, 1, 4, 2]
    lengths = {(0, 1): 3, (1, 2): 1.5, (2, 3): 0.1, (3, 4): 1.0, (4, 5): 3.5}

    # Create the n-partite graph
    G = create_n_partite_graph(partitions, lengths)

    # Run the analysis function
    results, _ = track_and_analyze_eigenvalue_decomposition(G, apply_correction=True)

    # Assert that the function ran successfully
    assert results.success
    assert results.t_eval is not None
    assert results.Qs is not None
    assert results.Lambdas is not None
    assert results.errors is not None
