import matplotlib.pyplot as plt
import numpy as np

from .types import EigenTrackingResults


def plot_eigenvalue_trajectories(results: EigenTrackingResults, ax=None):
    """
    追跡された各固有値の軌跡を、パラメータ t の関数としてプロットする。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if results.Lambdas is not None and results.t_eval is not None:
        eigenvalue_traces = np.array([np.diag(L) for L in results.Lambdas])
        for i in range(eigenvalue_traces.shape[1]):
            ax.plot(results.t_eval, eigenvalue_traces[:, i], label=f"λ_{i+1}(t)")
        ax.set_title("Eigenvalue Trajectories")
        ax.set_xlabel("Parameter t")
        ax.set_xscale("log")
        ax.set_ylabel("Eigenvalues")
        ax.legend()
        ax.grid(True)

    if show_plot:
        plt.show()

    return ax


def plot_reconstruction_error(results: EigenTrackingResults, ax=None):
    """
    再構成誤差 ||A(t) - Q(t)Λ(t)Q(t)^T||_F の時間発展をプロットする。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if results.norm_errors is not None and results.t_eval is not None:
        ax.semilogy(
            results.t_eval,
            results.norm_errors,
            label="Reconstruction Norm Error",
            color="crimson",
        )
        ax.set_title("Reconstruction Error")
        ax.set_xlabel("Parameter t")
        ax.set_xscale("log")
        ax.set_ylabel(r"$||A(t) - Q(t)\Lambda(t)Q(t)^T||_F$ (log scale)")
        ax.legend()
        ax.grid(True)

    if show_plot:
        plt.show()

    return ax


def plot_tracking_results(results: EigenTrackingResults, axes=None):
    """
    固有値追跡の結果（軌跡、誤差）を一つの図にまとめてプロットする。
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False

    plot_eigenvalue_trajectories(results, ax=axes[0])
    plot_reconstruction_error(results, ax=axes[1])

    plt.tight_layout()

    if show_plot:
        plt.show()

    return axes


def plot_eigenvectors_rotation(
    result: EigenTrackingResults, ax=None
):
    """
    内積から得られる「角度」の累積値によって固有ベクトルの角度の推移を描画する。

    Args:
        result (EigenTrackingResults): 固有値追跡の結果。
        ax (matplotlib.axes.Axes, optional): 描画先。指定しなければ新たに生成する。

    Returns:
        matplotlib.axes.Axes: 描画に使用した Axes。
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if result.Qs is not None and result.t_eval is not None:
        t = result.t_eval
        t_index = np.arange(len(t) - 1)
        # [t_index, row_index, column_index]
        eigenvecs = np.array(result.Qs)
        cosines = np.sum(eigenvecs[t_index + 1] * eigenvecs[t_index], axis=1)
        thetas = np.arccos(np.where(cosines > 1.0, 1.0, cosines))
        dt = (t[t_index + 1] - t[t_index])[:, None]
        ax.plot(t[1:], np.add.accumulate(thetas * dt), lw=1, label=np.arange(thetas.shape[1]))
        ax.set_title("Eigenvector Rotation Dynamics")
        ax.set_xlabel("Parameter t")
        ax.set_ylabel("Accumulated Angle θ")

    if show_plot:
        plt.show()

    return ax
