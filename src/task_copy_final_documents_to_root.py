import shutil

import pytask

from src.config import BLD
from src.config import ROOT


@pytask.mark.depends_on(
    [BLD / "manuscript" / "manuscript.pdf", BLD / "presentation" / "presentation.pdf"]
)
@pytask.mark.produces([ROOT / "manuscript.pdf", ROOT / "presentation.pdf"])
def task_write_specs(depends_on, produces):  # noqa: D103
    depends_on = list(depends_on.values())
    produces = list(produces.values())

    for origin, destination in zip(depends_on, produces):
        shutil.copy(origin, destination)
