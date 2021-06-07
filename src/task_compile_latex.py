import pytask

from src.config import BLD
from src.config import SRC


bld_path = BLD / "presentation" / "presentation.pdf"

dependencies = [
    SRC / "presentation" / "main.tex",
    SRC / "presentation" / "files" / "drawing.pdf",
    SRC / "presentation" / "files" / "item_label.png",
    SRC / "presentation" / "files" / "paper-titlepage.pdf",
    SRC / "presentation" / "files" / "video-snapshot.pdf",
    SRC / "presentation" / "files" / "video-snapshot.png",
    BLD / "figures" / "cross-covariance.png",
]


@pytask.mark.latex()
@pytask.mark.depends_on(dependencies)
@pytask.mark.produces(bld_path)
def task_compile_latex():  # noqa: D103
    pass
