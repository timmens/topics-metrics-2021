import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_process(process):
    _ = plt.plot(np.linspace(0, 1, len(process)), process)


def simulate_process(n_sim, n_periods, method, a=0, b=1, seed=None, **kwargs):
    if seed is not None:
        np.random.seed(seed)

    dt = (b - a) / (n_periods - 1)
    if method == "brownian_motion":
        process = _simulate_brownian_motion(dt, n_periods, n_sim)
    elif method == "ornstein_uhlenbeck":
        process = _simulate_ornstein_uhlenbeck(dt, n_periods, n_sim, **kwargs)
    elif method == "polynomial":
        process = _simulate_polynomial(a, b, n_periods, n_sim, **kwargs)
    elif method == "gaussian_process":
        process = _simulate_gaussian_process(n_periods, n_sim, **kwargs)
    elif method == "white-noise":
        process = _simulate_white_noise(n_periods, n_sim, **kwargs)
    elif method == "constant-normal":
        process = _simulate_constant_normal(n_periods, n_sim, **kwargs)
    elif method == "poisson":
        process = _simulate_poisson(dt, n_periods, n_sim, **kwargs)
    elif method == "levy":
        process = _simulate_levy(dt, n_periods, n_sim, **kwargs)
    else:
        raise ValueError(
            f"Method {method} not implemted. See source code for implemented processes"
        )
    return process


def _simulate_polynomial(a, b, n_periods, n_sim, order=5, mu=0, sigma=1):
    coefficients = np.random.normal(mu, sigma, (n_sim, order + 1))
    grid = np.linspace(a, b, n_periods)
    process = np.empty((n_periods, n_sim))
    for k in range(n_sim):
        process[:, k] = np.polyval(coefficients[k], grid)
    return process


def _simulate_brownian_motion(dt, n_periods, n_sim):
    innovations = np.random.normal(0, np.sqrt(dt), (n_periods, n_sim))
    process = np.cumsum(innovations, axis=0)
    return process


def _simulate_ornstein_uhlenbeck(dt, n_periods, n_sim, theta=5, sigmaou=3.5):
    mu = np.exp(-theta * dt)
    sigma = np.sqrt((1 - mu ** 2) * sigmaou ** 2 / (2 * theta))
    innovation = np.random.normal(0, 1, (n_periods - 1, n_sim))
    process = np.zeros((n_periods, n_sim))
    for j in range(1, n_periods):
        process[j] = process[j - 1] * mu + sigma * innovation[j - 1]
    return process


def _simulate_white_noise(n_periods, n_sim, sigma=1):
    process = np.random.normal(0, sigma, (n_periods, n_sim))
    return process


def _simulate_constant_normal(n_periods, n_sim, sigma=1):
    innovation = np.random.normal(0, sigma, n_sim)
    process = np.tile(innovation, (n_periods, 1))
    return process


def _simulate_poisson(dt, n_periods, n_sim, rate=1):
    innovations = np.random.poisson(dt * rate, (n_periods, n_sim))
    process = innovations
    process[0, :] = 0
    process = process.cumsum(axis=0)
    return process


def _simulate_levy(dt, n_periods, n_sim, increment_simulator):
    process = np.zeros((n_periods, n_sim))
    innovations = increment_simulator(dt, n_periods - 1, n_sim)
    process[1:, :] = innovations
    process = process.cumsum(axis=0)
    return process
