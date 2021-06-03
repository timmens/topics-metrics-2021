import pytask

from src.config import BLD
from src.config import SRC

depends_on = SRC / "data" / "emotion_rating.csv"
produces = BLD / "fitted_models" / "result-replication.csv"


@pytask.mark.r([depends_on, produces])
@pytask.mark.depends_on(["replicate.R", depends_on])
@pytask.mark.produces(produces)
def task_replicate_paper():  # noqa: D103
    pass
