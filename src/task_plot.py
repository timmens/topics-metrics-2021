import pytask

from src.config import BLD
from src.config import SRC
from src.plotting import create_cross_covariance_plot
from src.plotting import plot_process
from src.simulate import simulate_gaussian_process


@pytask.mark.depends_on(SRC / "plotting.py")
@pytask.mark.produces(BLD / "figures" / "cross-covariance.png")
def task_write_specs(produces):  # noqa: D103
    create_cross_covariance_plot(produces)


length_scales = [0.1, 0.05, 0.01]


@pytask.mark.parametrize(
    "produces, length_scale",
    [
        (BLD / "figures" / f"process_scale{length_scale}.png", length_scale)
        for length_scale in length_scales
    ],
)
def task_process_plots(produces, length_scale):  # noqa: D103
    process = simulate_gaussian_process(
        n_sim=3,
        n_periods=100,
        kernel="RBF",
        kernel_kwargs={"length_scale": length_scale},
    )
    fig, ax = plot_process(process)
    fig.savefig(produces, bbox_inches="tight")
