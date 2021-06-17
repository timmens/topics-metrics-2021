from functools import partial

import numpy as np
import pandas as pd
from bokeh.io import curdoc  # noqa: F401
from bokeh.layouts import column
from bokeh.layouts import row
from bokeh.models import CheckboxButtonGroup
from bokeh.models import ColumnDataSource
from bokeh.models import Label
from bokeh.models import Span
from bokeh.models.widgets import Slider
from bokeh.plotting import figure

from src.plotting import _create_plot_data
from src.plotting import _second_central_difference
from src.simulate import simulate_model


palette = {
    "BrownianMotion": "#8A9384",
    "RBF": "#FBBB06",
    "Matern": "#04539C",
}


def _create_data_bokeh(delta, kernels, locations, betas, n_grid_points):
    func = partial(_second_central_difference, delta=delta)
    data = _create_plot_data(func, kernels, locations, betas, n_grid_points)
    for col in data.columns.drop(["x"]):
        data = data.assign(**{col: data[col] / data[col].abs().max()})
    return data


def _transform_output_to_dict(delta, func):
    data = func(delta)
    d = {
        "x": data["x"],
        "matern": data["Matern"],
        "rbf": data["RBF"],
        "bm": data["BrownianMotion"],
    }
    return d


n_grid_points = 200
start = 0.001
end = 0.4
initial_value = 0.01
kernels = ["BrownianMotion", "RBF", "Matern"]
locations = [1 / 4, 2 / 4, 3 / 4]
betas = [1, 3, -2]

data = {
    kernel: simulate_model(
        n_samples=500,
        n_periods=int(end / start),
        n_points=len(locations),
        beta=[0] + betas,
        locations=locations,
        kernel=kernel,
        seed=1,
    )
    for kernel in kernels
}


def _estimate_second_central_difference(delta, data, start, end):
    n_neighbors = int(delta / start)

    grid = np.linspace(0, 1, int(end / start))
    df = pd.DataFrame({"x": grid})
    for kernel, (y, X, _) in data.items():
        result = np.tile(np.nan, len(X))
        for i in range(n_neighbors, len(X) - n_neighbors):
            Z = X[i, :] - (X[i + n_neighbors, :] + X[i - n_neighbors, :]) / 2
            result[i] = (Z * y).mean()
        if np.isnan(result).all():
            df[kernel] = np.nan
        else:
            df[kernel] = result / np.nanmax(np.abs(result))
    return df


estimate_second_central_difference = partial(
    _estimate_second_central_difference, **{"data": data, "start": start, "end": end}
)

create_data = partial(
    _create_data_bokeh,
    **{
        "kernels": kernels,
        "locations": locations,
        "betas": betas,
        "n_grid_points": n_grid_points,
    },
)
get_data_dict = partial(_transform_output_to_dict, func=create_data)
get_estimates_dict = partial(
    _transform_output_to_dict, func=estimate_second_central_difference
)

source = ColumnDataSource(data=get_data_dict(initial_value))
estimates = ColumnDataSource(data=get_estimates_dict(initial_value))

# initialize figure
p = figure(
    plot_width=900,
    plot_height=500,
    tools="wheel_zoom",
    x_range=[0, 1],
    y_range=[-3, 3],
)

# plot line for each kernel
line_list = []
estimate_line_list = []
for kernel in kernels:
    kernel_id = "bm" if kernel == "BrownianMotion" else kernel.lower()
    line_list += [
        p.line(
            "x",
            kernel_id,
            source=source,
            line_width=3,
            line_alpha=0.6,
            legend_label=kernel,
            line_color=palette[kernel],
        )
    ]
    estimate_line_list += [
        p.line(
            "x",
            kernel_id,
            source=estimates,
            line_width=2,
            line_alpha=0.4,
            legend_label=kernel,
            line_color=palette[kernel],
        )
    ]


def update_selection(attr, old, new):
    """Update checkbox selection of kernel."""
    selected_kernels = list(checkbox.active)
    selected = list(checkbox2.active)
    for line_id, line in enumerate(line_list):
        line.visible = line_id in selected_kernels and 0 in selected
    for line_id, estimate_line in enumerate(estimate_line_list):
        estimate_line.visible = line_id in selected_kernels and 1 in selected


def update_estimate_selection(attr, old, new):
    """Update checkbox selection of population / estimation points."""
    selected_kernels = list(checkbox.active)
    selected = list(checkbox2.active)
    for line_id, line in enumerate(line_list):
        line.visible = line_id in selected_kernels and 0 in selected
    for line_id, estimate_line in enumerate(estimate_line_list):
        estimate_line.visible = line_id in selected_kernels and 1 in selected


checkbox = CheckboxButtonGroup(labels=kernels, active=[])
checkbox.on_change("active", update_selection)

checkbox2 = CheckboxButtonGroup(labels=["Population", "Estimate"], active=[])
checkbox2.on_change("active", update_estimate_selection)


def update_data_and_label(attr, old, new):
    """Update data of lines and set new text label."""
    delta = slider.value
    source.data = get_data_dict(delta)
    estimates.data = get_estimates_dict(delta)
    annotation.text = f"δ = {delta:.3f}"


slider = Slider(title="delta", value=initial_value, start=start, end=end, step=start)
slider.on_change("value", update_data_and_label)

annotation = Label(x=0.05, y=2.5, text="δ = 0.01", text_font_size="22pt")
p.add_layout(annotation)

p.legend.location = "top_right"
p.legend.label_text_font = "Palatino"
p.legend.label_text_font_style = "italic"
p.legend.label_text_color = "black"
p.legend.label_text_font_size = "17pt"

p.legend.border_line_width = 0
p.legend.background_fill_alpha = 0

p.xaxis.axis_line_width = 2

p.xgrid.visible = False
p.ygrid.visible = False

p.xaxis.ticker = [0, 1 / 4, 2 / 4, 3 / 4, 4 / 4]
p.xaxis.major_label_overrides = {
    1 / 4: "τ₁ = 1/4",
    2 / 4: "τ₂ = 1/2",
    3 / 4: "τ₃ = 3/4",
}
p.xaxis.major_label_text_font_size = "16pt"
p.xaxis.major_label_text_font = "Palatino"
p.yaxis.ticker = []

for loc in locations:
    loc_span = Span(
        location=loc,
        dimension="height",
        line_color="black",
        line_dash="dashed",
        line_width=0.5,
    )
    p.add_layout(loc_span)

p.toolbar.logo = None
p.toolbar_location = None

layout = column(row(slider, checkbox, checkbox2), p)
# curdoc().add_root(layout)  # noqa: E800
