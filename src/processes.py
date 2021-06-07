import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as sklearn_kernels


def plot_process(process):
    """Plot process."""
    _ = plt.plot(np.linspace(0, 1, len(process)), process)


def simulate_gaussian_process(n_sim, n_periods, kernel, seed=None, kernel_kwargs=None):
    """Simulate gaussian process using specified kernel.

    Args:
        n_sim (int): Number of simulated processes.
        n_periods (int): Number of time realizations of each process.
            Realizations points are given by an equidistant grid over [0, 1].
        kernel (str): Kernel. Available kernels are in ["WhiteKernel", "RBF",
        "Matern", "BrownianMotion"].
        seed (int): Seed. Default None, which does not set a seed.
        **kernel_kwargs: Keyword arguments passed to the specified kernel.

    Returns:
        process (np.ndarray): Simulated process of shape (n_periods, n_sim).

    """
    if kernel == "BrownianMotion":
        process = _simulate_brownian_motion(1 / (n_periods - 1), n_periods, n_sim)
    else:
        grid = np.linspace(0, 1, n_periods)
        cov = get_kernel(kernel, kernel_kwargs)(grid)
        process = np.random.multivariate_normal(np.zeros(n_periods), cov, size=n_sim).T
    return process


def get_kernel(kernel, kernel_kwargs=None):
    """Return kernel function.

    Args:
        kernel (str): Kernel. Available kernels are in ["WhiteKernel", "RBF",
        "Matern", "BrownianMotion"].
        **kernel_kwargs: Keyword arguments passed to the specified kernel.

    Returns:
        kernel (callable): Kernel function of two arguments.

    """
    kernel_kwargs = _add_defaults_to_kwargs(kernel, kernel_kwargs)
    if kernel == "BrownianMotion":

        def _kernel(x, y):
            return kernel_kwargs["sigma"] ** 2 * np.minimum(x, y)

    else:
        kernel = getattr(sklearn_kernels, kernel)(**kernel_kwargs)

        def _kernel(X, Y=None):  # noqa: N803
            if Y is None:
                return kernel(X.reshape(-1, 1))
            else:
                Y = np.atleast_2d(Y)
                if len(Y) > 1:
                    raise ValueError("Second argument has to be a scalar.")
                return kernel(X.reshape(-1, 1), Y).flatten()

    return _kernel


def _add_defaults_to_kwargs(kernel, kwargs):
    kwargs = {} if kwargs is None else kwargs
    if kernel == "Matern":
        if "nu" not in kwargs:
            kwargs["nu"] = 0.5
        if "length_scale" not in kwargs:
            kwargs["length_scale"] = 0.1
    elif kernel == "RBF":
        if "length_scale" not in kwargs:
            kwargs["length_scale"] = 0.1
    elif kernel == "BrownianMotion":
        if "sigma" not in kwargs:
            kwargs["sigma"] = np.sqrt(2)
    return kwargs


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
