import itertools

import numpy as np
import pandas as pd

from src.simulate import _compute_poi_location


def config_to_kwargs_df(config):
    """Parse config dict to data frame with kwargs as rows.

    Args:
        config (dict): Dictionary of argument lists for each argument.

    Returns:
        df (pd.DataFrame): Data frame with kwargs as entry.

    """
    c = config.copy()
    beta = c.pop("beta")
    _ = c.pop("n_sim")

    data = list(itertools.product(*c.values()))
    df = pd.DataFrame(data, columns=c.keys())

    mapper = {v: k for k, v in dict(enumerate(c["n_points"])).items()}
    df = df.assign(
        **{"beta": df.n_points.replace(mapper).apply(lambda i: tuple(beta[i]))}
    )
    return df


def clean_df_kwargs(df_kwargs):
    """Clean data frame with kwargs."""
    df_kwargs = df_kwargs.assign(
        poi=df_kwargs.n_points.apply(
            lambda n_points: _compute_poi_location(n_points, n_periods=100)
        )
    )
    df_kwargs = df_kwargs.assign(
        **{"nu": df_kwargs.kernel_kwargs.str.extract(r"(\d+\.?\d*)")[0].astype(float)}
    )
    df_kwargs = df_kwargs.drop(["kernel_kwargs", "kernel", "beta"], axis=1)
    return df_kwargs


def clean_df_result(df_result):
    """Clean data frame with results."""
    df_base = pd.DataFrame(
        data=np.c_[np.arange(1, 101), np.zeros(100, dtype=int)],
        columns=["locations", "count"],
    ).set_index("locations")

    df_result = [df.dropna() for df in df_result]
    df_result = [df.set_index("locations") for df in df_result]
    df_result = [_update_base_df(df, df_base) for df in df_result]
    return df_result


def _update_base_df(df, df_base):
    df_new = df_base.copy()
    df_new.loc[df.index] = df
    return df_new
