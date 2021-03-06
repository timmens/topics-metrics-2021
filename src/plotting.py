from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.processes import get_kernel


def plot_process(process):
    """Plot process."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.linspace(0, 1, len(process)), process)

    for pos in ["left", "bottom", "right", "top"]:
        ax.spines[pos].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def create_barplot(order, n_points, df_kwargs, df_result, threshold=0):
    """Create barplot."""

    def _compute_increment(l, width):  # noqa: E741
        increment_dict = {0.5: -width / 2, 1.5: 0, 2.5: width / 2}
        return increment_dict[l]

    def _get_label(l):  # noqa: E741
        label = fr"$\nu = {l:.1f}$"
        return label

    def _get_color(l):  # noqa: E741
        color_dict = {
            0.5: "tab:blue",
            1.5: "tab:green",
            2.5: "goldenrod",
        }
        return color_dict[l]

    df_kwargs = df_kwargs.query("order == @order & n_points == @n_points")

    df_kwargs = df_kwargs.set_index(["nu"], append=True)
    df_kwargs = df_kwargs.sort_index(level="nu")

    dfs = [df_result[k] for k in df_kwargs.index.get_level_values(0)]

    x_grid = dfs[0].index
    bar_width = 2

    plt.rc("font", **{"family": "serif", "serif": ["Times"]})
    plt.rc("text", usetex=True)

    fig, ax = plt.subplots()

    fig.set_size_inches(6.3, 2.5, forward=True)

    ylimmax = 0.3
    if n_points == 1:
        ylimmax = 0.7
    elif n_points == 2:
        ylimmax = 0.4
    ax.set_ylim((0, ylimmax))

    for k, l in df_kwargs.index:

        increment = _compute_increment(l, bar_width)
        label = _get_label(l)
        color = _get_color(l)

        counts = df_result[k]["count"]

        freq = counts / counts.sum()

        if threshold > 0:
            too_small = counts <= threshold
            freq[too_small] = 0

        ax.bar(
            x_grid + increment,
            freq,
            bar_width,
            label=label,
            color=color,
            alpha=0.7,
            zorder=2,
        )

    _, ylim_max = ax.get_ylim()

    for poi in df_kwargs.poi.values[0]:
        ax.axvline(poi, color="tab:red", linewidth=1.5, zorder=1)
        ax.annotate(
            f"{poi}",
            (poi - 1.5, 0),
            (poi - 1.5, -0.11 * ylim_max),
            fontsize=12,
            color="tab:red",
        )

    ax.legend(fontsize=12, loc="upper right", frameon=False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set(ylabel="Frequency", xlabel="Period")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

    return fig, ax


palette = {
    "WhiteKernel": "#04539C",
    "BrownianMotion": "#8A9384",
    "RBF": "#FBBB06",
    "Matern": "#411010",
}


def create_cross_covariance_plot(
    path,
    kernels=("WhiteKernel", "BrownianMotion", "RBF", "Matern"),
    locations=(1 / 4, 2 / 4, 3 / 4),
    betas=(1, 3, -2),
    n_grid_points=200,
):
    """Plot cross covariance along time for different kernels.

    Args:
        path (str or pathlib.Path): Write path.
        kernels (str or list[str]): Kernel ids.
        locations (list[int]): List of point-of-impacts locations.
        betas (list[float] or np.ndarray): Slope coefficients and intercept
            corresponding to locations. Has to have one more item than locations.
        n_grid_points (int): Number of grid points used for plotting.

    Returns:
        None

    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        }
    )

    # create data
    data = _create_plot_data(
        _compute_cross_var_path, kernels, locations, betas, n_grid_points
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_bounds(-2, 3)
    ax.spines["bottom"].set_bounds(0, 1)

    ax.spines["left"].set_position(("data", -0.02))
    ax.spines["bottom"].set_position(("data", -2.2))

    ax.set_ylim(-2, 3)
    ax.set_xlim(0, 1)

    ax.xaxis.set_ticks([0, 1 / 4, 2 / 4, 3 / 4, 1])
    ax.yaxis.set_ticks(list(range(-2, 4, 1)))

    ax.set_xticklabels(
        [r"$0$", r"$\tau_1 = 1/4$", r"$\tau_2 = 1/2$", r"$\tau_3 = 3/4$", r"$1$"],
        size="x-large",
    )
    ax.set_yticklabels([r"$-2$", r"$-1$", r"$0$", r"$1$", r"$2$", r"$3$"], size="large")

    # make ticks inward facing
    ax.xaxis.set_tick_params(direction="in")
    ax.yaxis.set_tick_params(direction="in")

    # plot data
    for kernel in data.columns.drop("x"):
        ax.plot(data["x"], data[kernel], zorder=1, label=kernel, color=palette[kernel])

    # plot vertical lines for locations
    for loc, beta in zip(locations, betas):
        ax.axvline(loc, color="black", alpha=0.4, linestyle="--", linewidth=0.9)
        if "WhiteKernel" in kernels:
            _x = [loc, loc]
            _y = [0, beta]
            ax.plot(_x, _y, color=palette["WhiteKernel"])

    ax.text(
        0, 2.9, r"$\mathbb{E}\left[Y_i X_i(t)\right]$", va="center", size="xx-large"
    )
    ax.text(0.985, -2.1, r"$t$", va="center", size="xx-large")

    ax.legend(
        [r"$White \,\, Kernel$", r"$Brownian \,\, Motion$", r"$RBF$", r"$Matern$"],
        frameon=False,
        prop={"size": 14},
    )
    plt.savefig(path, bbox_inches="tight")
    return None


def create_second_centered_difference_plot(
    path,
    locations=(1 / 4, 2 / 4, 3 / 4),
    betas=(1, 3, -2),
    n_grid_points=200,
    kernels=("WhiteKernel", "BrownianMotion", "Matern", "RBF"),
    delta=0.01,
):
    """Create second centered-difference plot.

    Args:
        to be done

    Returns:
        None

    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        }
    )

    # create data
    func = partial(_second_central_difference, delta=delta)
    data = _create_plot_data(func, kernels, locations, betas, n_grid_points)

    # rescale data
    for col in data.columns.drop(["WhiteKernel", "x"]):
        data = data.assign(**{col: data[col] / data[col].abs().max()})

    # colors
    palette = {
        "WhiteKernel": "#04539C",
        "BrownianMotion": "#8A9384",
        "RBF": "#FBBB06",
        "Matern": "#411010",
    }

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_bounds(-1, 1)
    ax.spines["bottom"].set_bounds(0, 1)

    ax.spines["left"].set_position(("data", -0.02))
    ax.spines["bottom"].set_position(("data", -1.05))

    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)

    ax.xaxis.set_ticks([0, 1 / 4, 2 / 4, 3 / 4, 1])

    ax.set_xticklabels(
        [r"$0$", r"$\tau_1 = 1/4$", r"$\tau_2 = 1/2$", r"$\tau_3 = 3/4$", r"$1$"],
        size="x-large",
    )
    ax.set_yticklabels([])

    # make ticks inward facing
    ax.xaxis.set_tick_params(direction="in")
    ax.yaxis.set_tick_params(direction="in")

    # plot data
    for kernel in data.columns.drop("x"):
        ax.plot(data["x"], data[kernel], zorder=1, label=kernel, color=palette[kernel])

    # plot vertical lines for locations
    for loc in locations:
        ax.axvline(loc, color="black", alpha=0.4, linestyle="--", linewidth=0.9)

    ax.text(0, 0.9, r"$c^2(t, \delta)$", va="center", size="xx-large")
    ax.text(0.985, -0.1, r"$t$", va="center", size="xx-large")

    ax.legend(
        [r"$White \,\, Kernel$", r"$Brownian \,\, Motion$", r"$RBF$", r"$Matern$"],
        frameon=False,
        prop={"size": 14},
    )
    plt.savefig(path, bbox_inches="tight")
    return None


def _second_central_difference(grid, kernel, locations, beta, delta):
    func = _get_cross_covariance(get_kernel(kernel), locations, beta)
    return func(grid) - (func(grid + delta) + func(grid - delta)) / 2


def _get_cross_covariance(kernel, locations, betas):
    if len(locations) != len(betas):
        raise ValueError("Length of beta has to equal number of locations.")

    def func(t):
        return sum(beta * kernel(t, loc) for (beta, loc) in zip(betas, locations))

    return func


def _compute_cross_var_path(grid, kernel, locations, betas):
    func = _get_cross_covariance(get_kernel(kernel), locations, betas)
    return func(grid)


def _create_plot_data(func, kernels, locations, betas, n_grid_points):
    kernels = kernels if isinstance(kernels, (list, tuple)) else [kernels]
    grid = np.linspace(0, 1, n_grid_points)
    grid = np.sort(np.append(grid, locations))
    values = {kernel: func(grid, kernel, locations, betas) for kernel in kernels}
    data = pd.DataFrame(values)
    data = data.assign(**{"x": grid})
    return data
