import numpy as np
import sklearn.gaussian_process.kernels as sklearn_kernels


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
    seed = 0 if seed is None else seed
    np.random.seed(seed)
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

        def _kernel(X):  # noqa: N803
            x, y = np.meshgrid(X, X)
            return kernel_kwargs["sigma"] ** 2 * np.minimum(x, y)

    elif kernel == "SelfSimilar":

        def _kernel(X):  # noqa: N803
            sigma = kernel_kwargs["sigma"]
            kappa = kernel_kwargs["kappa"]
            x, y = np.meshgrid(X, X)
            return sigma * (x ** kappa + y ** kappa - np.abs(x - y) ** kappa)

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
    elif kernel == "SelfSimilar":
        if "sigma" not in kwargs:
            kwargs["sigma"] = 3.0
        if "kappa" not in kwargs:
            kwargs["kappa"] = 3.0
    return kwargs
