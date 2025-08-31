import numpy as np
import networkx as nx
from eigenpairflow import eigenpairtrack
from eigenpairflow.graph import create_n_partite_graph


def test_track_runs_with_graph_example():
    """
    Tests that the main tracking function runs without errors using the graph example.
    """
    # 1. Create the graph and distance matrix
    partitions = [3, 2, 1, 1, 4, 2]
    lengths = {(0, 1): 3, (1, 2): 1.5, (2, 3): 0.1, (3, 4): 1.0, (4, 5): 3.5}
    G = create_n_partite_graph(partitions, lengths)
    D = np.array(nx.floyd_warshall_numpy(G, weight="length"))

    # 2. Define the matrix functions and time span
    def A_func(t):
        return np.exp(-t * D)

    def dA_func(t):
        return -D * np.exp(-t * D)

    t_start, t_end = 4.0, 1.0e-2
    t_eval = np.geomspace(t_start, t_end, 100)  # Reduced points for faster test

    # 3. Run the tracking function
    results = eigenpairtrack(
        A_func,
        dA_func,
        (t_start, t_end),
        t_eval,
        matrix_type="symmetric",
        method="eigh",
        apply_correction=True,
    )

    # 4. Assert that the function ran successfully
    assert results.success
    assert results.t_eval is not None
    assert results.Qs is not None
    assert results.Lambdas is not None
    assert results.errors is not None
