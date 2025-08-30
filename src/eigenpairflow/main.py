import numpy as np
import scipy.linalg
import scipy.integrate
import networkx as nx
from collections import namedtuple
import joblib
from scipy.optimize import linear_sum_assignment

from .types import EigenTrackingResults

def solve_symmetric_ode_system_linsolve(Lambda, F):
    """
    実対称行列の発展方程式 F = dLambda + [H, Lambda] を、
    H と dLambda の独立成分に関する連立一次方程式として厳密に解く。

    Args:
        Lambda (np.ndarray): n x n の対角固有値行列。
        F (np.ndarray): n x n の対称行列 (Q^T * dA * Q)。

    Returns:
        tuple[np.ndarray, np.ndarray]: (H, dLambda_diag) のタプル。
                                       H は歪対称行列。
                                       dLambda_diag は固有値の変化率（対角成分のみのベクトル）。
    """
    n = Lambda.shape[0]

    # 1. 未知数の数を定義
    num_h_unknowns = n * (n - 1) // 2
    num_dlambda_unknowns = n
    total_unknowns = num_h_unknowns + num_dlambda_unknowns

    # 2. 連立一次方程式 M*x = b を構成
    M = np.zeros((total_unknowns, total_unknowns))
    b = np.zeros(total_unknowns)

    lambdas = np.diag(Lambda)

    # Numpyのインデックス機能を使い、手動ループを避ける
    # eq_indices は上三角部分(対角含む)のインデックス (r, c) r<=c
    # h_indices は厳密に上三角部分のインデックス (r, c) r<c
    eq_indices_r, eq_indices_c = np.triu_indices(n)
    h_indices_r, h_indices_c = np.triu_indices(n, k=1)

    # b ベクトルを構成 (Fの上三角成分)
    b[:num_dlambda_unknowns] = np.diag(F) # 対角成分 (n個)
    b[num_dlambda_unknowns:] = F[h_indices_r, h_indices_c] # 非対角成分 (n(n-1)/2個)

    # 係数行列 M を構成 (Mは対角行列になる)
    # dLambdaに対応するブロック (係数は常に1)
    M_diag = np.ones(total_unknowns)

    # Hに対応するブロック (係数は lambda_j - lambda_i)
    lambda_diffs = lambdas[h_indices_c] - lambdas[h_indices_r]
    M_diag[num_dlambda_unknowns:] = lambda_diffs

    np.fill_diagonal(M, M_diag)

    # 3. 連立一次方程式を解く
    # lstsq は悪条件（固有値の接近）に対して頑健
    # x, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
    x = np.linalg.lstsq(M, b, rcond=None)[0]

    # 4. 解ベクトル x を H と dLambda に再構成
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
    """
    # 1. 状態ベクトルを行列 Q と対角行列 Lambda に復元
    Q = y[:n*n].reshape((n, n))
    lambdas = y[n*n:]
    Lambda = np.diag(lambdas)

    # 2. dA/dt と F を計算
    dA = dA_func(t)
    F = Q.T @ dA @ Q

    # 3. H と dLambda を計算
    H, dLambda_diag = solve_symmetric_ode_system_linsolve(Lambda, F)

    # 4. dQ/dt を計算
    dQ = Q @ H

    # 5. 結果を平坦化して単一のベクトルとして返す
    dydt = np.concatenate([dQ.flatten(), dLambda_diag])
    return dydt

def track_eigen_decomposition(A_func, dA_func, t_span, t_eval, rtol=1e-5, atol=1e-8):
    """
    実対称行列関数 A(t) の固有値分解を、常微分方程式を解くことで追跡する。

    Args:
        A_func (callable): 時刻 t を受け取り、行列 A(t) を返す関数。
        dA_func (callable): 時刻 t を受け取り、行列の微分 dA/dt を返す関数。
        t_span (tuple): 計算を開始・終了する時刻 (t_start, t_end)。
        t_eval (np.ndarray): 結果を評価する時刻の配列。
        rtol (float): solve_ivp用の相対許容誤差。
        atol (float): solve_ivp用の絶対許容誤差。

    Returns:
        tuple: (list[np.ndarray], list[np.ndarray], object)
               - Qs: 各評価時刻における直交固有ベクトル行列 Q のリスト。
               - Lambdas: 各評価時刻における対角固有値行列 Lambda のリスト。
               - sol: scipy.integrate.solve_ivp から返された結果オブジェクト。
    """
    t0 = t_span[0]
    A0 = A_func(t0)
    n = A0.shape[0]

    # 初期条件: eighは実対称行列用に最適化されており、固有値と直交行列を返す
    lambdas0, Q0 = scipy.linalg.eigh(A0)

    # 固有値と固有ベクトルをソートして、追跡中の順序を一定に保つ
    sort_indices = np.argsort(lambdas0)
    lambdas0 = lambdas0[sort_indices]
    Q0 = Q0[:, sort_indices]

    # 初期状態ベクトル y0 を作成
    y0 = np.concatenate([Q0.flatten(), lambdas0])

    # 高精度ソルバーの呼び出し (DOP853は高次のRunge-Kutta法)
    sol = scipy.integrate.solve_ivp(
        symmetric_ode_derivative,
        t_span,
        y0,
        method='DOP853',
        t_eval=t_eval,
        args=(n, dA_func),
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed. {sol.message=}")

    # 結果をリストに復元
    Qs = [sol_y[:n*n].reshape((n, n)) for sol_y in sol.y.T]
    Lambdas = [np.diag(sol_y[n*n:]) for sol_y in sol.y.T]

    return Qs, Lambdas, sol

def match_decompositions(predicted_eigvals, predicted_eigvecs,
                         exact_eigvals, exact_eigvecs):
    """
    正確な対角化分解を、予測された対角化分解にマッチさせる。

    この関数は、ハンガリー法を用いて固有値の最適な対応付け（並べ替え）を
    見つけ出し、その後、対応する固有ベクトル間の内積を計算して符号を揃える。
    これにより、`eigh`のような関数の出力順序や符号の任意性に起因する不連続性を
    解消し、時間積分の連続性を維持する。

    Args:
        predicted_eigvals (np.ndarray): 1D配列。予測された固有値。
        predicted_eigvecs (np.ndarray): 2D配列。予測された固有ベクトル（列ベクトル）。
        exact_eigvals (np.ndarray): 1D配列。正確に計算された固有値。
        exact_eigvecs (np.ndarray): 2D配列。正確に計算された固有ベクトル（列ベクトル）。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - matched_eigvals: `predicted_eigvals` の順序に並べ替えられた `exact_eigvals`。
            - matched_eigvecs: `predicted_eigvecs` の順序と符号に合わせられた `exact_eigvecs`。
    """
    # --- ステップ1: 固有値の最適な対応付け（並べ替え）を見つける ---
    # コスト行列を計算する。C[i, j]は i 番目の予測値と j 番目の正確な値の差。
    # このコストを最小化するようなペアリングを見つけることが目的。
    cost_matrix = np.abs(predicted_eigvals[:, np.newaxis] - exact_eigvals[np.newaxis, :])

    # ハンガリー法（線形和割り当て問題）を解き、最適なペアリングを見つける
    # pred_indices[i] は、exact_indices[i] に対応する
    pred_indices, exact_indices = linear_sum_assignment(cost_matrix)

    # 並べ替えられた正確な値を格納する配列を準備
    matched_eigvals = np.zeros_like(exact_eigvals)
    matched_eigvecs = np.zeros_like(exact_eigvecs)

    # 見つかった対応付けに従って、正確な値を並べ替える
    # pred_indices は通常 0, 1, 2, ... となるので、exact_indices が置換を表す
    matched_eigvals = exact_eigvals[exact_indices]
    matched_eigvecs = exact_eigvecs[:, exact_indices]

    # --- ステップ2: 対応する固有ベクトルの符号を合わせる ---
    # 各々の対応するベクトル対の内積を計算する
    # 内積が負の場合、ベクトルは反対方向を向いているため、一方の符号を反転させる
    for i in range(predicted_eigvecs.shape[1]):
        dot_product = np.dot(predicted_eigvecs[:, i], matched_eigvecs[:, i])
        if dot_product < 0.0:
            matched_eigvecs[:, i] *= -1.0

    return matched_eigvals, matched_eigvecs

def correct_trajectory(A_func, t_eval, Qs_ode, Lambdas_ode):
    """
    ODEソルバーで追跡した固有値分解を、各時刻で正確に計算した分解に事後補正する。
    ODE結果をガイドとして、正確な分解を並べ替え・符号合わせする。

    Args:
        A_func (callable): 時刻 t を受け取り、真の行列 A(t) を返す関数。
        t_eval (np.ndarray): 評価時刻の1D配列。
        Qs_ode (list of np.ndarray): ODEソルバーから得られた固有ベクトル行列のリスト。
        Lambdas_ode (list of np.ndarray): ODEソルバーから得られた対角固有値行列のリスト。

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            - corrected_Qs: 補正された固有ベクトル行列のリスト。
            - corrected_Lambdas: 補正された対角固有値行列のリスト。
    """
    corrected_Qs = []
    corrected_Lambdas = []

    for i, t in enumerate(t_eval):
        # 1. その時刻における真の行列 A_t を計算
        A_t = A_func(t)

        # 2. A_t の正確な固有値分解を計算
        exact_eigvals, exact_eigvecs = scipy.linalg.eigh(A_t)

        # 3. match_decompositions を用いて、正確な分解をODE結果にマッチさせる
        # ODE結果を「予測」、正確な計算を「正確」として渡す
        # Lambdas_ode は対角行列なので、固有値を取り出す
        predicted_eigvals = np.diag(Lambdas_ode[i])
        predicted_eigvecs = Qs_ode[i]

        matched_eigvals, matched_eigvecs = match_decompositions(
            predicted_eigvals,
            predicted_eigvecs,
            exact_eigvals,
            exact_eigvecs
        )

        # 4. 補正結果をリストに追加
        corrected_Qs.append(matched_eigvecs)
        corrected_Lambdas.append(np.diag(matched_eigvals)) # 対角行列に戻す

    return corrected_Qs, corrected_Lambdas

def create_n_partite_graph(partition_sizes, edge_lengths_dict):
    """
    Creates an n-partite graph based on partition sizes and edge lengths between partitions.

    Args:
        partition_sizes (list): A list of integers representing the number of nodes in each partition.
        edge_lengths_dict (dict): A dictionary where keys are tuples of partition indices (i, j)
                                  and values are the lengths of edges between nodes in partition i and partition j.

    Returns:
        nx.Graph: The constructed n-partite graph.
    """
    G = nx.Graph()
    node_id = 1
    partition_nodes = []

    # Add nodes to each partition
    for i, size in enumerate(partition_sizes):
        nodes_in_partition = list(range(node_id, node_id + size))
        G.add_nodes_from(nodes_in_partition, type=f'p{i}')
        partition_nodes.append(nodes_in_partition)
        node_id += size

    # Add edges between partitions with specified lengths
    for (p1_idx, p2_idx), length in edge_lengths_dict.items():
        if p1_idx < len(partition_sizes) and p2_idx < len(partition_sizes) and p1_idx != p2_idx:
            for u in partition_nodes[p1_idx]:
                for v in partition_nodes[p2_idx]:
                    G.add_edge(u, v, length=length, weight=1/length)

    return G

def track_and_analyze_eigenvalue_decomposition(G, apply_correction=True):
    """
    グラフの距離行列から構成される行列 A(t) = exp(-tD) の固有値分解を追跡・分析する。

    この関数は、一連の処理をまとめて実行する。
    1. グラフ G からフロイド・ワーシャル法で距離行列 D を計算する。
    2. パラメータ t に依存する行列 A(t) = exp(-tD) とその微分 dA/dt を定義する。
    3. 常微分方程式を解くことで、A(t) の固有値と固有ベクトルの軌跡を追跡する (`track_eigen_decomposition`)。
    4. (オプション) 各時刻で A(t) を厳密に対角化し、その結果を使ってODEソルバーからの軌跡を補正する (`correct_trajectory`)。
       これにより、数値誤差の累積を防ぎ、精度を向上させる。
    5. 追跡された固有対を用いて、関連する物理量（マグニチュード、擬マグニチュード）および再構成誤差を計算する。
    6. 全ての結果を `EigenTrackingResults` オブジェクトにまとめて返す。

    Args:
        G (nx.Graph): 解析対象の重み付きグラフ。辺には 'length' 属性が必要。
        apply_correction (bool): 軌跡の事後補正を適用するかどうか。デフォルトは True。

    Returns:
        EigenTrackingResults: 追跡と分析の結果を格納した名前付きタプル。
                          成功したかどうか、メッセージ、各時刻の固有対、計算された物理量などが含まれる。

    --- English ---
    Tracks and analyzes the eigenvalue decomposition of the matrix A(t) = exp(-tD) derived from a graph's distance matrix.

    This function performs a complete workflow:
    1. Computes the distance matrix D from graph G using the Floyd-Warshall algorithm.
    2. Defines the parameter-dependent matrix A(t) = exp(-tD) and its derivative dA/dt.
    3. Tracks the eigenvalue and eigenvector trajectories of A(t) by solving an ordinary differential equation (`track_eigen_decomposition`).
    4. (Optional) Applies a post-hoc correction to the trajectory from the ODE solver by using exact diagonalization of A(t) at each time step (`correct_trajectory`). This mitigates the accumulation of numerical errors and improves accuracy.
    5. Computes relevant physical quantities (magnitude, pseudo-magnitude) and reconstruction error using the tracked eigenpairs.
    6. Returns all results consolidated into an `EigenTrackingResults` object.

    Args:
        G (nx.Graph): The weighted input graph. Edges must have a 'length' attribute.
        apply_correction (bool): Whether to apply the post-hoc trajectory correction. Defaults to True.

    Returns:
        EigenTrackingResults: A named tuple containing the results of the tracking and analysis,
                          including success status, messages, eigenpairs at each time step, and computed quantities.
    """
    # 1. Compute the distance matrix D from the input graph G
    try:
        D = np.array(nx.floyd_warshall_numpy(G, weight='length'))
    except nx.NetworkXNoPath:
         # Handle disconnected graphs if necessary, or let it propagate
         return EigenTrackingResults(
            t_eval=None, Qs=None, Lambdas=None, magnitudes=None,
            pseudo_magnitudes=None, errors=None, zero_indices=None,
            success=False, message="Graph is disconnected.", state=None,
            errors_before_correction=None
        )


    # 2. Define the matrix functions A(t) and dA/dt
    def A_func(t):
        return np.exp(-t * D)

    def dA_func(t):
        return -D * A_func(t)

    # 3. Define time span and evaluation points
    t_start, t_end = 4.0, 1.0e-2 # Example time span
    t_eval = np.geomspace(t_start, t_end, 10000)

    # 4. Call the track_eigen_decomposition function
    Qs_ode, Lambdas_ode, sol = None, None, None
    success = False
    message = "Tracking failed."
    state = None

    try:
        Qs_ode, Lambdas_ode, sol = track_eigen_decomposition(
            A_func, dA_func, (t_start, t_end), t_eval, rtol=1e-13, atol=1e-12
        )
        success = sol.success
        message = sol.message
        state = sol.status
    except RuntimeError as e:
        message = f"Tracking failed: {e}"

    # If tracking failed, return early
    if not success:
         return EigenTrackingResults(
            t_eval=sol.t if sol else None, Qs=None, Lambdas=None, magnitudes=None,
            pseudo_magnitudes=None, errors=None, zero_indices=None,
            success=success, message=message, state=state,
            errors_before_correction=None
        )


    # 6. Extract the original eigenvalue traces
    eigenvalues_traces = np.array([np.diag(L) for L in Lambdas_ode])

    # 7. Identify indices of eigenvalues that cross zero
    zero_indices = []
    for i in range(eigenvalues_traces.shape[1]):
        lambda_i = eigenvalues_traces[:, i]
        if np.amin(lambda_i) < 0.0 < np.amax(lambda_i):
            zero_indices.append(i)

    # 8. Calculate original magnitudes, pseudo-magnitudes, and reconstruction errors
    original_magnitudes = []
    original_pseudo_magnitudes = []
    errors_before_correction = []

    for i, t in enumerate(sol.t):
        Q_t = Qs_ode[i]
        Lambda_t = Lambdas_ode[i]

        Lambda_inverse = np.linalg.inv(Lambda_t)

        v = Q_t.T @ np.ones(D.shape[0])

        mag = v.T @ Lambda_inverse @ v
        original_magnitudes.append(mag)

        pseudo_Lambda_inverse = Lambda_inverse.copy()
        if zero_indices:
            pseudo_Lambda_inverse[zero_indices, zero_indices] = 0

        pseudo_mag = v.T @ pseudo_Lambda_inverse @ v
        original_pseudo_magnitudes.append(pseudo_mag)

        A_t = A_func(t)
        reconstructed_A = Q_t @ Lambda_t @ Q_t.T
        error = np.linalg.norm(A_t - reconstructed_A, 'fro')
        errors_before_correction.append(error)

    # 9. Initialize variables for corrected results
    corrected_Qs = None
    corrected_Lambdas = None
    corrected_magnitudes = None
    corrected_pseudo_magnitudes = None
    errors_after_correction = None

    # 10. Apply correction if requested
    if apply_correction:
        try:
            corrected_Qs, corrected_Lambdas = correct_trajectory(
                A_func, sol.t, Qs_ode, Lambdas_ode
            )

            # Calculate corrected magnitudes and pseudo-magnitudes
            corrected_magnitudes = []
            corrected_pseudo_magnitudes = []
            errors_after_correction = []

            for i, t in enumerate(sol.t):
                Q_t = corrected_Qs[i]
                Lambda_t = corrected_Lambdas[i]

                Lambda_inverse = np.linalg.inv(Lambda_t)

                v = Q_t.T @ np.ones(D.shape[0])

                mag = v.T @ Lambda_inverse @ v
                corrected_magnitudes.append(mag)

                pseudo_Lambda_inverse = Lambda_inverse.copy()
                if zero_indices:
                    pseudo_Lambda_inverse[zero_indices, zero_indices] = 0

                pseudo_mag = v.T @ pseudo_Lambda_inverse @ v
                corrected_pseudo_magnitudes.append(pseudo_mag)

                # Calculate corrected reconstruction error
                A_t = A_func(t)
                reconstructed_A = Q_t @ Lambda_t @ Q_t.T
                error = np.linalg.norm(A_t - reconstructed_A, 'fro')
                errors_after_correction.append(error)

        except Exception as e:
            print(f"Correction failed: {e}")
            # Proceed with original results if correction fails
            apply_correction = False # Revert to original results


    # 11. Select results based on apply_correction flag
    if apply_correction:
        final_Qs = corrected_Qs
        final_Lambdas = corrected_Lambdas
        final_magnitudes = corrected_magnitudes
        final_pseudo_magnitudes = corrected_pseudo_magnitudes
        final_errors = errors_after_correction
        final_errors_before_correction = errors_before_correction
    else:
        final_Qs = Qs_ode
        final_Lambdas = Lambdas_ode
        final_magnitudes = original_magnitudes
        final_pseudo_magnitudes = original_pseudo_magnitudes
        final_errors = errors_before_correction
        final_errors_before_correction = None # Set to None if correction wasn't applied


    # 12. Populate the EigenTrackingResults namedtuple
    results = EigenTrackingResults(
        t_eval=sol.t,
        Qs=final_Qs,
        Lambdas=final_Lambdas,
        magnitudes=final_magnitudes,
        pseudo_magnitudes=final_pseudo_magnitudes,
        errors=final_errors,
        zero_indices=zero_indices,
        success=success,
        message=message,
        state=state,
        errors_before_correction=final_errors_before_correction # Include only if correction applied
    )

    # 13. Return the namedtuple
    return results
