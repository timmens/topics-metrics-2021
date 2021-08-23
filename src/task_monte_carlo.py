import itertools
import os

import pandas as pd
import pytask
import yaml

from src.config import BLD
from src.config import SRC
from src.plotting import create_bar_plot
from src.shared import clean_df_kwargs
from src.shared import clean_df_result
from src.shared import config_to_kwargs_df

os.environ["MKL_THREADING_LAYER"] = "GNU"  # os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


config = yaml.safe_load(open(SRC / "config_monte_carlo.yaml", "r"))
df = config_to_kwargs_df(config)


@pytask.mark.depends_on(SRC / "config_monte_carlo.yaml")
@pytask.mark.produces(BLD / "monte_carlo" / "kwargs.csv")
def task_write_specs(produces):  # noqa: D103
    df.to_csv(produces, index=False)


produces = [BLD / "monte_carlo" / f"result{k}.csv" for k in range(len(df))]


@pytask.mark.r([SRC, BLD])
@pytask.mark.depends_on([SRC / "monte_carlo.R", BLD / "monte_carlo" / "kwargs.csv"])
@pytask.mark.produces(produces)
def task_monte_carlo():  # noqa: D103
    pass


order_list = df["order"].unique()
n_points_list = df["n_points"].unique()

figure_parametrize = [
    (
        BLD / "figures" / "monte_carlo" / f"barplot_{order}_{n_points}.png",
        order,
        n_points,
    )
    for order, n_points in itertools.product(order_list, n_points_list)
]


@pytask.mark.depends_on([BLD / "monte_carlo" / "kwargs.csv"] + produces)
@pytask.mark.parametrize("produces, order, n_points", figure_parametrize)
def task_plot_monte_carlo_results(depends_on, produces, order, n_points):  # noqa: D103
    depends_on = list(depends_on.values())

    df_kwargs = pd.read_csv(depends_on[0])
    df_result = [pd.read_csv(f) for f in depends_on[1:]]

    df_kwargs = clean_df_kwargs(df_kwargs)
    df_result = clean_df_result(df_result)

    fig, ax = create_bar_plot(order, n_points, df_kwargs, df_result)
    fig.savefig(produces, bbox_inches="tight")
