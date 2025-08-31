import matplotlib.pyplot as plt
import numpy as np

from .types import EigenTrackingResults


def plot_eigenvalue_trajectories(results: EigenTrackingResults, ax=None):
    """
    追跡された各固有値の軌跡を、パラメータ t の関数としてプロットする。

    横軸にパラメータ t（対数スケール）、縦軸に固有値 λ_i(t) をとり、
    各固有値が t の変化に伴いどのように変動するかを視覚化する。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if results.Lambdas is not None and results.t_eval is not None:
        eigenvalues_traces = np.array([np.diag(L) for L in results.Lambdas])
        for i in range(eigenvalues_traces.shape[1]):
            ax.plot(results.t_eval, eigenvalues_traces[:, i], label=f"λ_{i+1}(t)")
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

    横軸にパラメータ t（対数スケール）、縦軸にフロベニウスノルムで計算した
    再構成誤差（対数スケール）をとり、追跡された固有値分解の精度を評価する。
    補正が適用された場合、補正前後の誤差を比較して表示する。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if results.errors is not None and results.t_eval is not None:
        ax.semilogy(
            results.t_eval,
            results.errors,
            label="Reconstruction Error",
            color="crimson",
        )
        if results.errors_before_correction is not None:
            ax.semilogy(
                results.t_eval,
                results.errors_before_correction,
                label="Original ODE Error",
                linestyle="--",
                color="darkblue",
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
