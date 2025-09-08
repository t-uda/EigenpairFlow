import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment


def _find_clusters(eigvals, delta):
    """固有値の差が ``delta`` 未満である連続区間を検出する。"""
    sorted_idx = np.argsort(eigvals)
    clusters: list[list[int]] = []
    current = [sorted_idx[0]] if sorted_idx.size > 0 else []
    for idx in sorted_idx[1:]:
        if abs(eigvals[idx] - eigvals[current[-1]]) < delta:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    if current:
        clusters.append(current)
    return clusters


def match_decompositions(
    predicted_eigvals, predicted_eigvecs, exact_eigvals, exact_eigvecs
):
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
    cost_matrix = np.abs(
        predicted_eigvals[:, np.newaxis] - exact_eigvals[np.newaxis, :]
    )

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


def correct_trajectory(A_func, t_eval, Qs_ode, Lambdas_ode, method="matching"):
    """
    Corrects the eigenvalue decomposition trajectory from an ODE solver.

    Depending on the specified method, this function either matches the ODE
    results to a freshly computed decomposition or uses the Ogita-Aishima
    iterative refinement method.

    Args:
        A_func (callable): Function that returns the true matrix A(t).
        t_eval (np.ndarray): Array of time points for evaluation.
        Qs_ode (list of np.ndarray): List of eigenvector matrices from the ODE solver.
        Lambdas_ode (list of np.ndarray): List of diagonal eigenvalue matrices from the ODE solver.
        method (str): The correction method to use.
                      'matching': Re-calculates the decomposition and matches it.
                      'ogita_aishima': Refines the ODE result using the Ogita-Aishima method.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            - corrected_Qs: List of corrected eigenvector matrices.
            - corrected_Lambdas: List of corrected diagonal eigenvalue matrices.
    """
    corrected_Qs = []
    corrected_Lambdas = []

    for i, t in enumerate(t_eval):
        A_t = A_func(t)

        if method == "matching":
            exact_eigvals, exact_eigvecs = scipy.linalg.eigh(A_t)
            predicted_eigvals = np.diag(Lambdas_ode[i])
            predicted_eigvecs = Qs_ode[i]
            matched_eigvals, matched_eigvecs = match_decompositions(
                predicted_eigvals, predicted_eigvecs, exact_eigvals, exact_eigvecs
            )
            corrected_Qs.append(matched_eigvecs)
            corrected_Lambdas.append(np.diag(matched_eigvals))

        elif method == "ogita_aishima":
            X_hat = Qs_ode[i]
            X_refined, D_refined = ogita_aishima_refinement(A_t, X_hat)
            corrected_Qs.append(X_refined)
            corrected_Lambdas.append(D_refined)

        else:
            raise ValueError(f"Unknown correction method: {method}")

    return corrected_Qs, corrected_Lambdas


def ogita_aishima_refinement(A, X_hat, max_iter=10, tol=1e-12, rho=1.0):
    """
    荻田・相島の方法で近似固有ベクトルを改良する。
    固有値が近接している場合には小さな部分空間で再対角化を行う。

    Args:
        A (np.ndarray): 対称行列。
        X_hat (np.ndarray): 初期固有ベクトル近似。
        max_iter (int): 最大反復回数。
        tol (float): 収束判定の許容誤差。
        rho (float): 固有値クラスタ判定に用いる係数。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X_new: 改良された固有ベクトル。
            - D_new: 改良された固有値を対角に持つ行列。
    """
    n = A.shape[0]
    Id = np.eye(n)
    X_new = X_hat.copy()

    for _ in range(max_iter):
        # 1. Calculate residual matrices
        R = Id - X_new.T @ X_new
        S = X_new.T @ A @ X_new

        # 2. Calculate approximate eigenvalues
        r_ii = np.diag(R)
        s_ii = np.diag(S)
        lambda_tilde = s_ii / (1 - r_ii)

        # 3. Calculate the correction matrix E_tilde (vectorized)
        s_off_diag = S - np.diag(s_ii)
        delta = rho * np.max(np.abs(s_off_diag))

        # Differences between all pairs of eigenvalues
        lambda_diffs = lambda_tilde[np.newaxis, :] - lambda_tilde[:, np.newaxis]

        # Mask for distinct eigenvalues (avoiding the diagonal)
        distinct_mask = np.abs(lambda_diffs) > delta
        np.fill_diagonal(distinct_mask, False)

        # Numerator for the distinct case update: s_ij + lambda_j * r_ij
        numerator = S + R * lambda_tilde[np.newaxis, :]

        # Denominator, with a safe value where the mask is false to avoid division by zero
        denominator = np.where(distinct_mask, lambda_diffs, 1.0)

        # Calculate E_tilde using np.where for conditional logic
        E_tilde = np.where(distinct_mask, numerator / denominator, R / 2.0)

        # Set diagonal elements
        np.fill_diagonal(E_tilde, r_ii / 2.0)

        # 4. Update the solution
        X_new = X_new @ (Id + E_tilde)

        # 5. Additional refinement for clustered eigenvalues
        clusters = _find_clusters(lambda_tilde, delta)
        for cluster in clusters:
            if len(cluster) > 1:
                X_block = X_new[:, cluster]
                X_block, _ = np.linalg.qr(X_block)
                B = X_block.T @ A @ X_block
                w, V = np.linalg.eigh(B)
                X_new[:, cluster] = X_block @ V

        # 6. Check for convergence
        if np.linalg.norm(E_tilde) < tol:
            break

    # Final calculation of eigenvalues and sorting
    S_final = X_new.T @ A @ X_new
    final_eigvals = np.diag(S_final)
    sort_indices = np.argsort(final_eigvals)
    D_new = np.diag(final_eigvals[sort_indices])
    X_new = X_new[:, sort_indices]
    dots = np.sum(X_hat[:, sort_indices] * X_new, axis=0)
    signs = np.where(dots >= 0, 1.0, -1.0)
    X_new = X_new * signs

    return X_new, D_new
