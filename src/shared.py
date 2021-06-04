import itertools

import pandas as pd


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
