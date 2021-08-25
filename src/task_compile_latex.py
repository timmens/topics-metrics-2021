import pytask

from src.config import BLD
from src.config import SRC


bld_path_presentation = BLD / "presentation" / "presentation.pdf"
bld_path_manuscript = BLD / "manuscript" / "manuscript.pdf"

dependencies_presentation = [
    SRC / "presentation" / "main.tex",
    SRC / "presentation" / "files" / "drawing.pdf",
    SRC / "presentation" / "files" / "item_label.png",
    SRC / "presentation" / "files" / "paper-titlepage.pdf",
    SRC / "presentation" / "files" / "video-snapshot.pdf",
    SRC / "presentation" / "files" / "video-snapshot.png",
    BLD / "figures" / "cross-covariance.png",
]


dependencies_manuscript = [
    SRC / "manuscript" / "main.tex",
    SRC / "manuscript" / "preamble.tex",
    SRC / "manuscript" / "sections" / "introduction.tex",
    SRC / "manuscript" / "sections" / "review.tex",
    SRC / "manuscript" / "sections" / "extension.tex",
    SRC / "manuscript" / "sections" / "monte_carlo.tex",
    BLD / "figures" / "process_scale0.5.png",
    BLD / "figures" / "process_scale1.5.png",
    BLD / "figures" / "process_scale2.5.png",
]


@pytask.mark.latex()
@pytask.mark.depends_on(dependencies_presentation)
@pytask.mark.produces(bld_path_presentation)
def task_compile_presentation():  # noqa: D103
    pass


@pytask.mark.latex()
@pytask.mark.depends_on(dependencies_manuscript)
@pytask.mark.produces(bld_path_manuscript)
def task_compile_manuscript():  # noqa: D103
    pass
