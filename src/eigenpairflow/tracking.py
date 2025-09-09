import numpy as np
import scipy.linalg
import scipy.integrate
from types import SimpleNamespace

from .ode import symmetric_ode_derivative
from .correction import correct_trajectory, ogita_aishima_refinement
from .types import EigenTrackingResults


def _track_ogita_aishima(A_func, t_eval, *, max_iter=10, tol=1.0e-12, rho=1.0):
    A_t = A_func(t_eval[0])
    eigvals, eigvecs = scipy.linalg.eigh(A_t)
    sort_indices = np.argsort(eigvals)
    eigvals = eigvals[sort_indices]
    eigvecs = eigvecs[:, sort_indices]
    eigvals_list = [np.diag(eigvals)]
    eigvecs_list = [eigvecs]
    y0 = np.concatenate([eigvecs.flatten(), eigvals])
    y = np.zeros((len(t_eval), len(y0)))
    y[0] = y0
    for i, t in enumerate(t_eval[1:], start=1):
        A_t = A_func(t)
        eigvecs, D = ogita_aishima_refinement(
            A_t, eigvecs, max_iter=max_iter, tol=tol, rho=rho
        )
        eigvals_list.append(D)
        eigvecs_list.append(eigvecs)
        y[i] = np.concatenate([eigvecs.flatten(), np.diag(D)])

    sol = SimpleNamespace(t=t_eval, y=y, success=True, message="Ogita-Aishima")
    return eigvecs_list, eigvals_list, sol


def _track_symmetric_eigh(
    A_func,
    dA_func,
    t_span,
    t_eval,
    solver_method="Euler",
    rtol=1e-13,
    atol=1e-12,
    dense_output=False,
):
    """
    (内部関数) 対称行列族 A(t) の固有分解を指定された ODE 解法で追跡する。

    Args:
        A_func (callable): 時刻 t に対して行列 A(t) を返す関数。
        dA_func (callable): 時刻 t に対して微分 dA/dt を返す関数。
        t_span (tuple): 追跡する時間区間 (t_start, t_end)。
        t_eval (np.ndarray): 結果を保存する時刻の配列。
        solver_method (str): ODE の解法。 ``'Euler'`` を指定すると前進 Euler 法を用いる。
        rtol (float): ``solve_ivp`` 利用時の相対誤差許容値。
        atol (float): ``solve_ivp`` 利用時の絶対誤差許容値。
        dense_output (bool): ``solve_ivp`` で密な出力を生成するかどうか。

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], object]:
            各時刻の固有ベクトル・固有値行列と、
            ``solve_ivp`` の戻り値または同等のオブジェクト。
    """
    t0 = t_span[0]
    A0 = A_func(t0)
    n = A0.shape[0]

    lambdas0, Q0 = scipy.linalg.eigh(A0)
    sort_indices = np.argsort(lambdas0)
    lambdas0 = lambdas0[sort_indices]
    Q0 = Q0[:, sort_indices]

    y0 = np.concatenate([Q0.flatten(), lambdas0])

    if solver_method.lower() == "euler":
        t_values = np.array(t_eval)
        if not np.isclose(t_values[0], t0):
            raise ValueError("t_eval must start with t_span[0] for Euler solver")
        num_steps = t_values.size
        y = np.zeros((y0.size, num_steps))
        y[:, 0] = y0
        for i in range(num_steps - 1):
            t_i = t_values[i]
            dt = t_values[i + 1] - t_i
            dydt = symmetric_ode_derivative(t_i, y[:, i], n, dA_func)
            y[:, i + 1] = y[:, i] + dt * dydt
        sol = SimpleNamespace(
            t=t_values, y=y, success=True, message="Euler integration"
        )
    else:
        sol = scipy.integrate.solve_ivp(
            symmetric_ode_derivative,
            t_span,
            y0,
            method=solver_method,
            t_eval=t_eval,
            args=(n, dA_func),
            rtol=rtol,
            atol=atol,
            dense_output=dense_output,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed. {sol.message=}")

    Qs = [sol_y[: n * n].reshape((n, n)) for sol_y in sol.y.T]
    Lambdas = [np.diag(sol_y[n * n :]) for sol_y in sol.y.T]

    return Qs, Lambdas, sol


def eigenpairtrack(
    A_func,
    dA_func,
    t_span,
    t_eval,
    matrix_type="symmetric",
    method="eigh",
    solver_method="ogita_aishima",
    correction_method=None,
    rtol=1e-13,
    atol=1e-12,
    dense_output=False,
):
    """
    パラメータに依存する行列 A(t) の固有分解を追跡する。

    行列の種類や時間積分の解法に応じて内部の計算を振り分ける高水準インターフェース。

    Args:
        A_func (callable): 時刻 t に対して行列 A(t) を返す関数。
        dA_func (callable): 時刻 t に対して微分 dA/dt を返す関数。
        t_span (tuple): 追跡する時間区間 (t_start, t_end)。
        t_eval (np.ndarray): 結果を保存する時刻の配列。
        matrix_type (str): 行列の種類。現在は `"symmetric"` のみ対応。
        method (str): 固有分解の手法。現在は `"eigh"` のみ対応。
        solver_method (str): 追跡の手法。 `'ogita_aishima'` がデフォルト。
            `solve_ivp` の `method` を指定可能。
        correction_method (str or None): 補正手法。 `'none'` がデフォルト。
            `solver_method` で ODE ソルバー指定の場合の後処理方法を指定。
            詳細は `correct_trajectory` 関数を参照。
        rtol (float): `solve_ivp` 利用時の相対誤差許容値。
        atol (float): `solve_ivp` 利用時の絶対誤差許容値。
        dense_output (bool): `solve_ivp` で密な出力を生成するかどうか。

    Returns:
        EigenTrackingResults: 追跡と解析結果を含むオブジェクト。
    """
    if matrix_type != "symmetric" or method != "eigh":
        raise NotImplementedError(
            f"Tracking for matrix_type='{matrix_type}' and method='{method}' is not yet implemented."
        )

    # --- Dispatch to the appropriate low-level tracker ---
    try:
        if solver_method == "ogita_aishima":
            Qs, Lambdas, sol = _track_ogita_aishima(A_func, t_eval)
        else:
            Qs, Lambdas, sol = _track_symmetric_eigh(
                A_func,
                dA_func,
                t_span,
                t_eval,
                solver_method=solver_method,
                rtol=rtol,
                atol=atol,
                dense_output=dense_output,
            )
        success = sol.success
        message = sol.message
    except RuntimeError as e:
        success = False
        message = f"Tracking failed: {e}"
        sol = None

    if not success:
        return EigenTrackingResults(
            t_eval=t_eval,
            Qs=None,
            Lambdas=None,
            norm_errors=None,
            success=success,
            message=message,
        )

    # --- Post-processing (correction and error calculation) ---
    if correction_method:
        try:
            Qs, Lambdas = correct_trajectory(
                A_func, sol.t, Qs, Lambdas, method=correction_method
            )
        except Exception as e:
            success = False
            message += f" | Correction failed with method '{correction_method}': {e}"

    # Calculate the final norm errors
    norm_errors = []
    for i, t in enumerate(sol.t):
        A_t = A_func(t)
        reconstructed_A = Qs[i] @ Lambdas[i] @ Qs[i].T
        error = np.linalg.norm(A_t - reconstructed_A, "fro")
        norm_errors.append(error)

    return EigenTrackingResults(
        t_eval=sol.t,
        Qs=Qs,
        Lambdas=Lambdas,
        norm_errors=norm_errors,
        success=success,
        message=message,
    )
