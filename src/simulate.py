import numpy as np

from src.processes import simulate_gaussian_process


def simulate_model(
    n_samples=10,
    n_periods=100,
    n_points=1,
    beta=(0, 1),
    kernel="WhiteKernel",
    **kernel_kwargs
):
    """Simulate points-of-impact model.

    Args:
        n_samples (int): Number of samples.
        n_periods (int): Number of periods for each sample.
        n_points (int): Number of points-of-impact.
        beta (list-like): Beta coefficients for linear model. len(beta) = n_points + 1.
        kernel (str): Kernel to be used in simulation of gaussian process.
        **kernel_kwargs: Keyword arguments passed to kernel.

    Returns:
        y (np.ndarray): Outcomes. Has shape (n_samples,).
        X (np.ndarray): Features. Has shape (n_periods, n_samples).
        poi_location (np.ndarray): Location of points-of-impact. Has shape (n_points,).

    """
    if len(beta) != n_points + 1:
        raise ValueError("Length of argument beta must be equal to n_points + 1.")
    beta = np.array(beta)

    X = simulate_gaussian_process(n_samples, n_periods, kernel, kernel_kwargs)
    poi_location = _compute_poi_location(n_points, n_periods)

    X_poi = X[poi_location, :].T
    y = beta[0] + X_poi @ beta[1:]

    return y, X, poi_location


def _compute_poi_location(n_points, n_periods):
    location = np.array(n_periods / (2 ** np.arange(1, n_points + 1)) - 1, dtype=int)
    location = np.sort(location)
    return location
