import pytask

from src.config import BLD
from src.config import SRC
from src.plotting import create_cross_covariance_plot


@pytask.mark.depends_on(SRC / "plotting.py")
@pytask.mark.produces(BLD / "figures" / "cross-covariance.png")
def task_write_specs(produces):  # noqa: D103
    create_cross_covariance_plot(produces)
