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

def plot_magnitudes(t_eval, magnitudes, pseudo_magnitudes_indices=None, ax=None):
    """
    Plots the magnitude trajectories for each eigenvalue.

    Highlights the trajectories of eigenvalues that cross zero.

    Args:
        t_eval (np.ndarray): Array of time points.
        magnitudes (np.ndarray): 2D array of magnitudes (n_timesteps x n_eigenvalues).
        pseudo_magnitudes_indices (list[int], optional): List of indices of eigenvalues
                                                         that cross zero. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on.
                                             If None, a new figure and axes are created.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    if magnitudes is not None and t_eval is not None:
        num_eigenvalues = magnitudes.shape[1]
        for i in range(num_eigenvalues):
            label = f'Magnitude of λ_{i+1}'
            style = {'linewidth': 2}
            if pseudo_magnitudes_indices and i in pseudo_magnitudes_indices:
                label += ' (zero-crossing)'
                style['linestyle'] = '--'
                style['color'] = 'red'
                ax.plot(t_eval, magnitudes[:, i], label=label, **style)
            else:
                style['alpha'] = 0.7
                ax.plot(t_eval, magnitudes[:, i], **style)


        ax.set_title('Magnitude Trajectories')
        ax.set_xlabel('Parameter t')
        ax.set_xscale('log')
        ax.set_ylabel('Magnitude M(t) = -log(λ(t))/t')
        ax.legend()
        ax.grid(True)

    if show_plot:
        plt.show()

    return ax

def plot_eigen_tracking_results(results: EigenTrackingResults, magnitudes, pseudo_magnitudes_indices, axes=None):
    """
    固有値追跡の全結果（軌跡、誤差、マグニチュード）を一つの図にまとめてプロットする。

    Args:
        results (EigenTrackingResults): 追跡結果。
        magnitudes (np.ndarray): マグニチュードの2D配列。
        pseudo_magnitudes_indices (list[int]): 擬似マグニチュードのインデックスのリスト。
        axes (np.ndarray, optional): プロット用の軸オブジェクトの配列。
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        show_plot = True
    else:
        show_plot = False

    plot_eigenvalue_trajectories(results, ax=axes[0])
    plot_reconstruction_error(results, ax=axes[1])
    plot_magnitudes(results.t_eval, magnitudes, pseudo_magnitudes_indices, ax=axes[2])

    plt.tight_layout()

    if show_plot:
        plt.show()

    return axes
