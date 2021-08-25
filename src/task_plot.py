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


nu_list = [0.5, 1.5, 2.5]


@pytask.mark.parametrize(
    "produces, nu",
    [(BLD / "figures" / f"process_scale{nu}.png", nu) for nu in nu_list],
)
def task_process_plots(produces, nu):  # noqa: D103
    process = simulate_gaussian_process(
        n_sim=3,
        n_periods=100,
        kernel="Matern",
        kernel_kwargs={"nu": nu},
    )
    fig, ax = plot_process(process)
    fig.savefig(produces, bbox_inches="tight")
