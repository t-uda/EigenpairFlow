import numpy as np
import networkx as nx
import sympy
from eigenpairflow.main import track_and_analyze_eigenvalue_decomposition

def get_analytical_eigenpairs_bipartite(m, n):
    """
    Computes the analytical eigenvalues and eigenvectors for the similarity matrix
    of a complete bipartite graph K_{m,n}.
    """
    t = sympy.Symbol('t', real=True)
    total_nodes = m + n
    A = sympy.zeros(total_nodes, total_nodes)
    x = sympy.exp(-t)
    y = sympy.exp(-2 * t)

    for i in range(m):
        for j in range(m):
            A[i, j] = 1 if i == j else y
    for i in range(m, total_nodes):
        for j in range(m, total_nodes):
            A[i, j] = 1 if i == j else y
    for i in range(m):
        for j in range(m, total_nodes):
            A[i, j] = x
            A[j, i] = x

    eigensystem = A.eigenvects()
    eigen_pairs = []
    for val, mult, vecs in eigensystem:
        ortho_vecs = sympy.GramSchmidt(vecs, True)
        for i in range(mult):
            eigen_pairs.append((val, ortho_vecs[i]))

    try:
        eigen_pairs.sort(key=lambda pair: float(pair[0].subs(t, 1.0).evalf()))
    except (AttributeError, TypeError):
        pass

    eigvals = [p[0] for p in eigen_pairs]
    eigvecs_matrix = sympy.Matrix.hstack(*[p[1] for p in eigen_pairs])

    lambdify_eigvals = sympy.lambdify(t, eigvals, 'numpy')
    lambdify_eigvecs = sympy.lambdify(t, eigvecs_matrix, 'numpy')

    return lambdify_eigvals, lambdify_eigvecs

def compare_eigenspaces(lambda_num, q_num, lambda_ana, q_ana, tol=1e-6):
    """
    Compares eigenvalues and eigenspaces, handling multiplicities.
    """
    # First, check if the eigenvalues are close
    np.testing.assert_allclose(np.sort(lambda_num), np.sort(lambda_ana), rtol=tol, atol=tol)

    # Group eigenvectors by clustering eigenvalues
    unique_lambdas, inverse_indices = np.unique(np.round(lambda_ana, int(-np.log10(tol))), return_inverse=True)

    for i, unique_val in enumerate(unique_lambdas):
        # Find indices of eigenvectors belonging to this eigenspace
        num_indices = np.where(np.abs(lambda_num - unique_val) < tol)[0]
        ana_indices = np.where(inverse_indices == i)[0]

        if len(num_indices) == 0 or len(ana_indices) == 0:
            continue

        # Get the bases for the eigenspace
        Q_num_k = q_num[:, num_indices]
        Q_ana_k = q_ana[:, ana_indices]

        # Compare the projection matrices
        P_num_k = Q_num_k @ Q_num_k.T
        P_ana_k = Q_ana_k @ Q_ana_k.T

        np.testing.assert_allclose(P_num_k, P_ana_k, rtol=tol, atol=tol)


def run_analytical_test(m, n):
    """Helper function to run the analytical comparison test for K_{m,n}."""
    G = nx.complete_bipartite_graph(m, n)
    for i, j in G.edges():
        G.edges[i, j]['length'] = 1.0

    results_numerical = track_and_analyze_eigenvalue_decomposition(G, apply_correction=False)
    assert results_numerical.success

    analytical_eigvals_func, analytical_eigvecs_func = get_analytical_eigenpairs_bipartite(m, n)

    for i in range(0, len(results_numerical.t_eval), 2000):
        t = results_numerical.t_eval[i]

        q_num = results_numerical.Qs[i]
        lambda_num_diag = np.diag(results_numerical.Lambdas[i])

        lambda_ana_diag = np.array(analytical_eigvals_func(t))
        q_ana = analytical_eigvecs_func(t)

        compare_eigenspaces(lambda_num_diag, q_num, lambda_ana_diag, q_ana, tol=1e-5)


def test_k3_2_analytical_comparison():
    """
    Tests the eigenvalue tracking against the analytical solution for K_{3,2}.
    """
    run_analytical_test(3, 2)


def test_k4_2_analytical_comparison():
    """
    Tests the eigenvalue tracking against the analytical solution for K_{4,2}.
    """
    run_analytical_test(4, 2)
