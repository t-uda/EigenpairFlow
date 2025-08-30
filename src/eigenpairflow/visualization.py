import matplotlib.pyplot as plt
import numpy as np

from eigenpairflow.types import EigenTrackingResults

def plot_eigenvalue_trajectories(results: EigenTrackingResults, ax=None):
    """
    Plots the eigenvalue trajectories.

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
    Plots the reconstruction error.

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

def plot_magnitudes(results: EigenTrackingResults, ax=None):
    """
    Plots the magnitude and pseudo-magnitude.

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

    if results.magnitudes is not None and results.pseudo_magnitudes is not None and results.t_eval is not None:
        ax.plot(results.t_eval, results.magnitudes, color='darkred', label='Magnitude')
        ax.plot(results.t_eval, results.pseudo_magnitudes, color='darkgreen', label='Pseudo-Magnitude')

        ax.set_title('Magnitude vs Pseudo-Magnitude')
        ax.set_xlabel('Parameter t')
        ax.set_xscale('log')
        ax.set_ylabel('Value')
        # Set a reasonable y-axis limit
        y_min = -1
        if results.pseudo_magnitudes:
            y_max = np.amax(results.pseudo_magnitudes) + 2
            ax.set_ylim(y_min, y_max)

        ax.legend()
        ax.grid(True)

    if show_plot:
        plt.show()

    return ax

def plot_eigen_tracking_results(results: EigenTrackingResults, axes=None):
    """
    Plots all eigenpair tracking results on a set of axes.

    Args:
        results (EigenTrackingResults): The namedtuple containing the tracking results.
        axes (np.ndarray, optional): A numpy array of matplotlib axes objects
                                     (e.g., from plt.subplots(1, 3)).
                                     If None, a new figure and axes are created.

    Returns:
        np.ndarray: A numpy array of the used axes objects.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        show_plot = True
    else:
        fig = axes[0].get_figure()
        show_plot = False

    plot_eigenvalue_trajectories(results, ax=axes[0])
    plot_reconstruction_error(results, ax=axes[1])
    plot_magnitudes(results, ax=axes[2])

    plt.tight_layout()

    if show_plot:
        plt.show()

    return axes
