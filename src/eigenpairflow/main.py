import numpy as np
import scipy.linalg
import scipy.integrate
import networkx as nx
from scipy.optimize import linear_sum_assignment
from .types import EigenTrackingResults

def solve_symmetric_ode_system_linsolve(Lambda, F):
    """
    実対称行列の発展方程式 F = dLambda + [H, Lambda] を、
    H と dLambda の独立成分に関する連立一次方程式として厳密に解く。

    Solves the evolution equation for symmetric matrices F = dLambda + [H, Lambda]
    as a system of linear equations for the independent components of H and dLambda.

    Args:
        Lambda (np.ndarray): n x n の対角固有値行列。
        F (np.ndarray): n x n の対称行列 (Q^T * dA * Q)。

    Returns:
        tuple[np.ndarray, np.ndarray]: (H, dLambda_diag) のタプル。
                                       H は歪対称行列。
                                       dLambda_diag は固有値の変化率（対角成分のみのベクトル）。
    """
    n = Lambda.shape[0]
    num_h_unknowns = n * (n - 1) // 2
    num_dlambda_unknowns = n
    total_unknowns = num_h_unknowns + num_dlambda_unknowns

    M = np.zeros((total_unknowns, total_unknowns))
    b = np.zeros(total_unknowns)
    lambdas = np.diag(Lambda)

    h_indices_r, h_indices_c = np.triu_indices(n, k=1)
    b[:num_dlambda_unknowns] = np.diag(F)
    b[num_dlambda_unknowns:] = F[h_indices_r, h_indices_c]

    M_diag = np.ones(total_unknowns)
    lambda_diffs = lambdas[h_indices_c] - lambdas[h_indices_r]
    M_diag[num_dlambda_unknowns:] = lambda_diffs
    np.fill_diagonal(M, M_diag)

    x = np.linalg.lstsq(M, b, rcond=None)[0]
    dLambda_diag = x[:num_dlambda_unknowns]
    H_upper_vals = x[num_dlambda_unknowns:]

    H = np.zeros((n, n))
    H[h_indices_r, h_indices_c] = H_upper_vals
    H[h_indices_c, h_indices_r] = -H_upper_vals

    return H, dLambda_diag

def symmetric_ode_derivative(t, y, n, dA_func):
    """
    solve_ivp に渡すための微分方程式の右辺 f(t, y) を定義する。
    y は [Q.flatten(), diag(Lambda)] を連結したベクトル。

    Defines the right-hand side of the ODE for solve_ivp.
    y is a flattened vector of [Q, diag(Lambda)].
    """
    Q = y[:n*n].reshape((n, n))
    lambdas = y[n*n:]
    Lambda = np.diag(lambdas)

    dA = dA_func(t)
    F = Q.T @ dA @ Q

    H, dLambda_diag = solve_symmetric_ode_system_linsolve(Lambda, F)
    dQ = Q @ H

    return np.concatenate([dQ.flatten(), dLambda_diag])

def track_eigen_decomposition(A_func, dA_func, t_span, t_eval, rtol=1e-5, atol=1e-8):
    """
    実対称行列関数 A(t) の固有値分解を、常微分方程式を解くことで追跡する。

    Tracks the eigenvalue decomposition of a symmetric matrix function A(t) by solving an ODE.
    """
    t0 = t_span[0]
    A0 = A_func(t0)
    n = A0.shape[0]

    lambdas0, Q0 = scipy.linalg.eigh(A0)
    sort_indices = np.argsort(lambdas0)
    lambdas0 = lambdas0[sort_indices]
    Q0 = Q0[:, sort_indices]

    y0 = np.concatenate([Q0.flatten(), lambdas0])

    sol = scipy.integrate.solve_ivp(
        symmetric_ode_derivative, t_span, y0, method='DOP853', t_eval=t_eval,
        args=(n, dA_func), rtol=rtol, atol=atol, dense_output=True
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    Qs = [sol_y[:n*n].reshape((n, n)) for sol_y in sol.y.T]
    Lambdas = [np.diag(sol_y[n*n:]) for sol_y in sol.y.T]

    return Qs, Lambdas, sol

def match_decompositions(predicted_eigvals, predicted_eigvecs, exact_eigvals, exact_eigvecs):
    """
    正確な対角化分解を、予測された対角化分解にマッチさせる。

    Matches an exact eigendecomposition to a predicted one using the Hungarian algorithm.
    """
    cost_matrix = np.abs(predicted_eigvals[:, np.newaxis] - exact_eigvals[np.newaxis, :])
    pred_indices, exact_indices = linear_sum_assignment(cost_matrix)

    matched_eigvals = exact_eigvals[exact_indices]
    matched_eigvecs = exact_eigvecs[:, exact_indices]

    for i in range(predicted_eigvecs.shape[1]):
        if np.dot(predicted_eigvecs[:, i], matched_eigvecs[:, i]) < 0.0:
            matched_eigvecs[:, i] *= -1.0

    return matched_eigvals, matched_eigvecs

def correct_trajectory(A_func, t_eval, Qs_ode, Lambdas_ode):
    """
    ODEソルバーで追跡した固有値分解を、各時刻で正確に計算した分解に事後補正する。

    Corrects the tracked eigendecomposition using exact calculations at each time step.
    """
    corrected_Qs = []
    corrected_Lambdas = []
    for i, t in enumerate(t_eval):
        A_t = A_func(t)
        exact_eigvals, exact_eigvecs = scipy.linalg.eigh(A_t)

        predicted_eigvals = np.diag(Lambdas_ode[i])
        predicted_eigvecs = Qs_ode[i]

        matched_eigvals, matched_eigvecs = match_decompositions(
            predicted_eigvals, predicted_eigvecs, exact_eigvals, exact_eigvecs
        )

        corrected_Qs.append(matched_eigvecs)
        corrected_Lambdas.append(np.diag(matched_eigvals))

    return corrected_Qs, corrected_Lambdas

def create_n_partite_graph(partition_sizes, edge_lengths_dict):
    """
    n-partiteグラフを生成します。

    Creates an n-partite graph.
    """
    G = nx.Graph()
    node_id = 1
    partition_nodes = []
    for i, size in enumerate(partition_sizes):
        nodes_in_partition = list(range(node_id, node_id + size))
        G.add_nodes_from(nodes_in_partition, type=f'p{i}')
        partition_nodes.append(nodes_in_partition)
        node_id += size

    for (p1_idx, p2_idx), length in edge_lengths_dict.items():
        if p1_idx < len(partition_sizes) and p2_idx < len(partition_sizes) and p1_idx != p2_idx:
            for u in partition_nodes[p1_idx]:
                for v in partition_nodes[p2_idx]:
                    G.add_edge(u, v, length=length, weight=1/length)
    return G

def track_and_analyze_eigenvalue_decomposition(G, t_start=4.0, t_end=1.0e-2, num_t=1000, apply_correction=True):
    """
    グラフの距離行列に対して固有値追跡と解析を実行します。

    Performs eigenvalue tracking and analysis for a graph's distance matrix.
    """
    try:
        D = np.array(nx.floyd_warshall_numpy(G, weight='length'))
    except nx.NetworkXNoPath:
         return EigenTrackingResults(
            t_eval=None, Qs=None, Lambdas=None, magnitudes=None,
            pseudo_magnitudes=None, errors=None, zero_indices=None,
            success=False, message="Graph is disconnected.", state=None,
            errors_before_correction=None
        )

    A_func = lambda t: np.exp(-t * D)
    dA_func = lambda t: -D * np.exp(-t * D)
    t_eval = np.geomspace(t_start, t_end, num_t)

    try:
        Qs_ode, Lambdas_ode, sol = track_eigen_decomposition(
            A_func, dA_func, (t_start, t_end), t_eval, rtol=1e-13, atol=1e-12
        )
        success, message, state = sol.success, sol.message, sol.status
    except RuntimeError as e:
        success, message, state = False, f"Tracking failed: {e}", None

    if not success:
         return EigenTrackingResults(
            t_eval=sol.t if 'sol' in locals() else None, Qs=None, Lambdas=None, magnitudes=None,
            pseudo_magnitudes=None, errors=None, zero_indices=None,
            success=success, message=message, state=state,
            errors_before_correction=None
        )

    eigenvalues_traces = np.array([np.diag(L) for L in Lambdas_ode])
    zero_indices = [i for i, l in enumerate(eigenvalues_traces.T) if np.amin(l) < 0.0 < np.amax(l)]

    magnitudes, pseudo_magnitudes, errors = [], [], []
    for i, t in enumerate(sol.t):
        Q_t, Lambda_t = Qs_ode[i], Lambdas_ode[i]
        Lambda_inverse = np.linalg.inv(Lambda_t)
        v = Q_t.T @ np.ones(D.shape[0])
        magnitudes.append(v.T @ Lambda_inverse @ v)

        pseudo_Lambda_inverse = Lambda_inverse.copy()
        if zero_indices:
            pseudo_Lambda_inverse[zero_indices, zero_indices] = 0
        pseudo_magnitudes.append(v.T @ pseudo_Lambda_inverse @ v)

        A_t = A_func(t)
        reconstructed_A = Q_t @ Lambda_t @ Q_t.T
        errors.append(np.linalg.norm(A_t - reconstructed_A, 'fro'))

    results_data = {
        't_eval': sol.t, 'Qs': Qs_ode, 'Lambdas': Lambdas_ode,
        'magnitudes': magnitudes, 'pseudo_magnitudes': pseudo_magnitudes,
        'errors': errors, 'zero_indices': zero_indices, 'success': success,
        'message': message, 'state': state, 'errors_before_correction': None
    }

    if apply_correction:
        try:
            corrected_Qs, corrected_Lambdas = correct_trajectory(A_func, sol.t, Qs_ode, Lambdas_ode)
            corrected_magnitudes, corrected_pseudo_magnitudes, corrected_errors = [], [], []
            for i, t in enumerate(sol.t):
                Q_t, Lambda_t = corrected_Qs[i], corrected_Lambdas[i]
                Lambda_inverse = np.linalg.inv(Lambda_t)
                v = Q_t.T @ np.ones(D.shape[0])
                corrected_magnitudes.append(v.T @ Lambda_inverse @ v)

                pseudo_Lambda_inverse = Lambda_inverse.copy()
                if zero_indices:
                    pseudo_Lambda_inverse[zero_indices, zero_indices] = 0
                corrected_pseudo_magnitudes.append(v.T @ pseudo_Lambda_inverse @ v)

                A_t = A_func(t)
                reconstructed_A = Q_t @ Lambda_t @ Q_t.T
                corrected_errors.append(np.linalg.norm(A_t - reconstructed_A, 'fro'))

            results_data.update({
                'Qs': corrected_Qs, 'Lambdas': corrected_Lambdas, 'magnitudes': corrected_magnitudes,
                'pseudo_magnitudes': corrected_pseudo_magnitudes, 'errors': corrected_errors,
                'errors_before_correction': errors
            })
        except Exception as e:
            print(f"Correction failed: {e}")

    return EigenTrackingResults(**results_data)
