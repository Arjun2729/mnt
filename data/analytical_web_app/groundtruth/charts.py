"""Chart construction driven by the semantic layer.

Because column roles are known, the app can propose charts that make sense for
the data rather than offering every chart type for every column.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .semantic import DatasetSpec

from .motion import CHART_TRANSITION

PALETTE = ["#0B6A72", "#C2691B", "#3E6BA8", "#6B8F3E", "#8A4F8C", "#B4474C", "#4E7F8C", "#9A7B33"]
TEMPLATE = "plotly_white"


@dataclass
class ChartSuggestion:
    chart_type: str
    x: str
    y: str | None
    color: str | None
    rationale: str


def suggest(spec: DatasetSpec, limit: int = 6) -> list[ChartSuggestion]:
    """Propose charts from column roles."""
    out: list[ChartSuggestion] = []
    measures, dimensions = spec.measures, spec.dimensions

    if spec.time_column and measures:
        out.append(ChartSuggestion("Line", spec.time_column, measures[0],
                                   dimensions[0] if dimensions else None,
                                   f"{measures[0]} over time at {spec.time_grain or 'period'} grain"))
    for dimension in dimensions[:2]:
        if measures:
            out.append(ChartSuggestion("Bar", dimension, measures[0], None,
                                       f"Compare {measures[0]} across {dimension}"))
    if len(measures) >= 2:
        out.append(ChartSuggestion("Scatter", measures[0], measures[1],
                                   dimensions[0] if dimensions else None,
                                   f"Relationship between {measures[0]} and {measures[1]}"))
    if measures:
        out.append(ChartSuggestion("Histogram", measures[0], None, None, f"Distribution of {measures[0]}"))
    if dimensions and measures:
        out.append(ChartSuggestion("Box", dimensions[0], measures[0], None,
                                   f"Spread of {measures[0]} within each {dimensions[0]}"))
    if len(measures) >= 2:
        out.append(ChartSuggestion("Correlation heatmap", "", None, None, "All numeric pairs at once"))
    return out[:limit]


def _style(figure: go.Figure, title: str = "") -> go.Figure:
    # The title sits in the container's top margin and the legend just above the
    # plotting area; without separate reserved space the two overlap.
    # A title dict carrying text=None renders the string "undefined", so the
    # whole key has to be omitted when there is no title.
    title_layout = (
        dict(
            text=title,
            x=0, xanchor="left", y=0.97, yanchor="top",
            font=dict(family="IBM Plex Sans Condensed, IBM Plex Sans, sans-serif", size=17),
        )
        if title
        else None
    )
    figure.update_layout(
        template=TEMPLATE,
        colorway=PALETTE,
        # Applies when the figure's data changes — switching aggregation morphs
        # the marks rather than snapping to the new values.
        transition=CHART_TRANSITION,
        margin=dict(l=12, r=12, t=86 if title else 52, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="left", x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="IBM Plex Sans, -apple-system, Segoe UI, sans-serif", size=13),
        hoverlabel=dict(font_size=12),
    )
    if title_layout:
        figure.update_layout(title=title_layout)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zerolinecolor="rgba(128,128,128,0.3)")
    return figure


def build(
    frame: pd.DataFrame,
    chart_type: str,
    x: str = "",
    y: str = "",
    color: str = "",
    aggregation: str = "sum",
    title: str = "",
) -> go.Figure:
    """Build one chart. Aggregation happens here so the caller passes raw rows."""
    color_arg = color or None

    if chart_type == "Correlation heatmap":
        numeric = frame.select_dtypes("number")
        if numeric.shape[1] < 2:
            raise ValueError("Need at least two numeric columns.")
        matrix = numeric.corr()
        figure = go.Figure(
            go.Heatmap(
                z=matrix.values, x=list(matrix.columns), y=list(matrix.index),
                zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                text=matrix.round(2).values, texttemplate="%{text}", hoverongaps=False,
            )
        )
        return _style(figure, title or "Correlation matrix")

    if chart_type == "Histogram":
        return _style(px.histogram(frame, x=x, color=color_arg, marginal="box", nbins=40), title)

    if chart_type == "Box":
        return _style(px.box(frame, x=x or None, y=y, color=color_arg, points="outliers"), title)

    if chart_type == "Scatter":
        return _style(px.scatter(frame, x=x, y=y, color=color_arg, opacity=0.75,
                                 trendline="ols" if color_arg is None else None), title)

    if chart_type in ("Bar", "Pie"):
        group_keys = [k for k in (x, color) if k]
        if y:
            plot = frame.groupby(group_keys, dropna=False, as_index=False)[y].agg(aggregation)
            value_column = y
        else:
            plot = frame.groupby(group_keys, dropna=False, as_index=False).size().rename(columns={"size": "count"})
            value_column = "count"
        plot = plot.sort_values(value_column, ascending=False).head(60)
        if chart_type == "Pie":
            return _style(px.pie(plot.head(12), names=x, values=value_column, hole=0.42), title)
        return _style(px.bar(plot, x=x, y=value_column, color=color_arg, barmode="group"), title)

    if chart_type in ("Line", "Area"):
        group_keys = [k for k in (x, color) if k]
        plot = frame.groupby(group_keys, dropna=False, as_index=False)[y].agg(aggregation) if y else frame
        plot = plot.sort_values(x)
        builder = px.line if chart_type == "Line" else px.area
        figure = builder(plot, x=x, y=y, color=color_arg, markers=(chart_type == "Line"))
        return _style(figure, title)

    raise ValueError(f"Unknown chart type: {chart_type}")


def forecast_figure(forecast_result, title: str = "Forecast") -> go.Figure:
    """History, forecast mean and the prediction interval as a band."""
    figure = go.Figure()
    history, mean, lower, upper = (
        forecast_result.history, forecast_result.mean, forecast_result.lower, forecast_result.upper,
    )
    figure.add_trace(go.Scatter(
        x=list(upper.index) + list(lower.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself", fillcolor="rgba(11,106,114,0.14)", line=dict(width=0),
        name="Prediction interval", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(x=history.index, y=history.values, name="Observed",
                                line=dict(color=PALETTE[0], width=2)))
    figure.add_trace(go.Scatter(x=mean.index, y=mean.values, name="Forecast",
                                line=dict(color=PALETTE[1], width=2, dash="dash")))
    return _style(figure, title)


def decomposition_figure(decomposition, title: str = "Decomposition") -> go.Figure:
    from plotly.subplots import make_subplots

    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    for row, (series, color) in enumerate(
        [(decomposition.observed, PALETTE[0]), (decomposition.trend, PALETTE[1]),
         (decomposition.seasonal, PALETTE[2]), (decomposition.residual, PALETTE[5])], start=1
    ):
        figure.add_trace(go.Scatter(x=series.index, y=series.values, line=dict(color=color, width=1.6),
                                    showlegend=False, name=""), row=row, col=1)
    figure.update_layout(height=680)
    return _style(figure, title)


# ---------------------------------------------------------------- cross-filtering


def _round_bounds(low: float, high: float) -> tuple[float, float]:
    """Trim float noise from a drag selection so the filter reads sensibly."""
    span = abs(high - low)
    if span == 0:
        return low, high
    digits = max(0, 3 - int(f"{span:e}".split("e")[1]))
    return round(low, digits), round(high, digits)


def selection_to_conditions(
    selection: dict | None,
    chart_type: str,
    x: str,
    y: str,
    color: str,
    x_is_measure: bool,
    y_is_measure: bool,
) -> list[dict]:
    """Translate a Plotly selection event into filter-tree conditions.

    Clicking categories yields an `in` condition over the picked values; dragging
    a box over numeric axes yields `between` conditions. Returns [] when the
    selection cannot be expressed as a filter, which is the honest outcome for
    aggregated marks whose axes are computed rather than stored.
    """
    if not selection:
        return []

    conditions: list[dict] = []

    # A box or lasso drag over numeric axes becomes range conditions.
    for box in selection.get("box") or []:
        if x and x_is_measure and box.get("x") and len(box["x"]) == 2:
            low, high = _round_bounds(min(box["x"]), max(box["x"]))
            conditions.append({"column": x, "operator": "between", "value": [low, high]})
        if y and y_is_measure and box.get("y") and len(box["y"]) == 2:
            low, high = _round_bounds(min(box["y"]), max(box["y"]))
            conditions.append({"column": y, "operator": "between", "value": [low, high]})
    if conditions:
        return conditions

    points = selection.get("points") or []
    if not points:
        return []

    # Clicked marks: categorical axes filter by value, numeric axes by range.
    if x and not x_is_measure:
        values = sorted({str(p["x"]) for p in points if p.get("x") is not None})
        if values:
            conditions.append({"column": x, "operator": "in", "value": values})
    elif x and x_is_measure:
        xs = [p["x"] for p in points if isinstance(p.get("x"), (int, float))]
        if xs:
            low, high = _round_bounds(min(xs), max(xs))
            conditions.append({"column": x, "operator": "between", "value": [low, high]})

    # For a coloured series, the legend group identifies the category clicked.
    if color:
        groups = sorted({
            str(p[key]) for p in points
            for key in ("legendgroup", "label")
            if p.get(key) not in (None, "")
        })
        if groups:
            conditions.append({"column": color, "operator": "in", "value": groups})

    return conditions


def describe_selection(conditions: list[dict]) -> str:
    parts = []
    for condition in conditions:
        value = condition["value"]
        if condition["operator"] == "between":
            parts.append(f"{condition['column']} {value[0]:,g}–{value[1]:,g}")
        else:
            shown = ", ".join(map(str, value[:3])) + ("…" if len(value) > 3 else "")
            parts.append(f"{condition['column']} = {shown}")
    return " · ".join(parts)


def add_time_controls(figure: go.Figure, grain: str = "month") -> go.Figure:
    """Attach a range slider and quick-range buttons to a temporal x-axis."""
    steps = {
        "hour": [(24, "h", "24h"), (168, "h", "7d"), (720, "h", "30d")],
        "day": [(7, "day", "7d"), (30, "day", "30d"), (90, "day", "90d"), (365, "day", "1y")],
        "week": [(4, "week", "4w"), (13, "week", "13w"), (52, "week", "1y")],
        "month": [(3, "month", "3m"), (6, "month", "6m"), (12, "month", "1y"), (24, "month", "2y")],
        "quarter": [(4, "month", "1y"), (8, "month", "2y")],
        "year": [(5, "year", "5y")],
    }.get(grain, [(6, "month", "6m"), (12, "month", "1y")])

    figure.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07),
        rangeselector=dict(
            buttons=[dict(count=n, label=label, step=step, stepmode="backward") for n, step, label in steps]
            + [dict(step="all", label="All")],
            bgcolor="rgba(127,127,127,0.10)",
            activecolor="rgba(11,106,114,0.28)",
            borderwidth=0,
            font=dict(size=11),
            x=0, xanchor="left", y=1.06, yanchor="bottom",
        ),
    )
    figure.update_layout(margin=dict(t=112))
    return figure
