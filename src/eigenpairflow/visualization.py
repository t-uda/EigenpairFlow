import matplotlib.pyplot as plt
import numpy as np

from eigenpairflow.types import EigenTrackingResults

def plot_eigenvalue_trajectories(results: EigenTrackingResults, ax=None):
    """
    追跡された各固有値の軌跡を、パラメータ t の関数としてプロットする。

    横軸にパラメータ t（対数スケール）、縦軸に固有値 λ_i(t) をとり、
    各固有値が t の変化に伴いどのように変動するかを視覚化する。

    --- English ---
    Plots the trajectory of each tracked eigenvalue as a function of the parameter t.

    Visualizes how each eigenvalue λ_i(t) varies as t changes, with parameter t
    on the x-axis (log scale) and the eigenvalue on the y-axis.

    Args:
        results (EigenTrackingResults): The namedtuple containing the tracking results.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on.
                                             If None, a new figure and axes are created.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if results.Lambdas is not None and results.t_eval is not None:
        eigenvalues_traces = np.array([np.diag(L) for L in results.Lambdas])
        for i in range(eigenvalues_traces.shape[1]):
            ax.plot(results.t_eval, eigenvalues_traces[:, i], label=f'λ_{i+1}(t)')
        ax.set_title('Eigenvalue Trajectories')
        ax.set_xlabel('Parameter t')
        ax.set_xscale('log')
        ax.set_ylabel('Eigenvalues')
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

    --- English ---
    Plots the evolution of the reconstruction error ||A(t) - Q(t)Λ(t)Q(t)^T||_F.

    Evaluates the accuracy of the tracked eigenvalue decomposition by plotting the
    reconstruction error, calculated using the Frobenius norm, on the y-axis (log scale)
    against the parameter t on the x-axis (log scale). If correction was applied,
    it compares the error before and after correction.

    Args:
        results (EigenTrackingResults): The namedtuple containing the tracking results.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on.
                                             If None, a new figure and axes are created.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if results.errors is not None and results.t_eval is not None:
        ax.semilogy(results.t_eval, results.errors, label='Reconstruction Error', color='crimson')
        if results.errors_before_correction is not None:
             ax.semilogy(results.t_eval, results.errors_before_correction, label='Original ODE Error', linestyle='--', color='darkblue')

        ax.set_title('Reconstruction Error')
        ax.set_xlabel('Parameter t')
        ax.set_xscale('log')
        ax.set_ylabel(r'$||A(t) - Q(t)\Lambda(t)Q(t)^T||_F$ (log scale)')
        ax.legend()
        ax.grid(True)

    if show_plot:
        plt.show()

    return ax

def plot_magnitudes(t_eval, magnitudes, pseudo_magnitudes, ax=None):
    """
    マグニチュードと擬似マグニチュードの軌跡をプロットする。

    Args:
        t_eval (np.ndarray): 評価時刻。
        magnitudes (list[float]): マグニチュードのリスト。
        pseudo_magnitudes (list[float]): 擬似マグニチュードのリスト。
        ax (matplotlib.axes.Axes, optional): プロット用の軸オブジェクト。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if magnitudes is not None and pseudo_magnitudes is not None and t_eval is not None:
        ax.plot(t_eval, magnitudes, color='darkred', label='Magnitude')
        ax.plot(t_eval, pseudo_magnitudes, color='darkgreen', label='Pseudo-Magnitude')

        ax.set_title('Magnitude vs Pseudo-Magnitude')
        ax.set_xlabel('Parameter t')
        ax.set_xscale('log')
        ax.set_ylabel('Value')
        # Set a reasonable y-axis limit
        y_min = -1
        if pseudo_magnitudes:
            y_max = np.amax(pseudo_magnitudes) + 2
            ax.set_ylim(y_min, y_max)

        ax.legend()
        ax.grid(True)

    if show_plot:
        plt.show()

    return ax

def plot_eigen_tracking_results(results: EigenTrackingResults, magnitudes, pseudo_magnitudes, axes=None):
    """
    固有値追跡の全結果（軌跡、誤差、マグニチュード）を一つの図にまとめてプロットする。

    Args:
        results (EigenTrackingResults): 追跡結果。
        magnitudes (list[float]): マグニチュードのリスト。
        pseudo_magnitudes (list[float]): 擬似マグニチュードのリスト。
        axes (np.ndarray, optional): プロット用の軸オブジェクトの配列。
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        show_plot = True
    else:
        show_plot = False

    plot_eigenvalue_trajectories(results, ax=axes[0])
    plot_reconstruction_error(results, ax=axes[1])
    plot_magnitudes(results.t_eval, magnitudes, pseudo_magnitudes, ax=axes[2])

    plt.tight_layout()

    if show_plot:
        plt.show()

    return axes
