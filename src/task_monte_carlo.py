import os

import pytask
import yaml

from src.config import BLD
from src.config import SRC
from src.shared import config_to_kwargs_df

os.environ["MKL_THREADING_LAYER"] = "GNU"  # os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


config = yaml.safe_load(open(SRC / "config_monte_carlo.yaml", "r"))
df = config_to_kwargs_df(config)


@pytask.mark.depends_on(SRC / "config_monte_carlo.yaml")
@pytask.mark.produces(BLD / "monte_carlo" / "kwargs.csv")
def task_write_specs(produces):  # noqa: D103
    df.to_csv(produces, index=False)


@pytask.mark.r([SRC, BLD])
@pytask.mark.depends_on([SRC / "monte_carlo.R", BLD / "monte_carlo" / "kwargs.csv"])
@pytask.mark.produces([BLD / "monte_carlo" / f"result{k}.csv" for k in range(len(df))])
def task_monte_carlo():  # noqa: D103
    pass
