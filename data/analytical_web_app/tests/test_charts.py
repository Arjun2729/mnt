"""Chart construction across every type and the degenerate inputs users hit."""
import numpy as np
import pandas as pd
import pytest

from groundtruth import charts


@pytest.mark.parametrize("chart_type,kwargs", [
    ("Bar", {"x": "region", "y": "revenue"}),
    ("Bar", {"x": "region"}),                                  # count, no measure
    ("Line", {"x": "date", "y": "revenue"}),
    ("Line", {"x": "date", "y": "revenue", "color": "channel"}),
    ("Area", {"x": "date", "y": "revenue"}),
    ("Scatter", {"x": "cost", "y": "revenue"}),
    ("Scatter", {"x": "cost", "y": "revenue", "color": "region"}),
    ("Histogram", {"x": "revenue"}),
    ("Box", {"x": "channel", "y": "revenue"}),
    ("Pie", {"x": "region", "y": "revenue"}),
    ("Correlation heatmap", {}),
])
def test_every_chart_type_builds(frame, chart_type, kwargs):
    figure = charts.build(frame, chart_type, **kwargs)
    assert figure.data, f"{chart_type} produced no traces"


@pytest.mark.parametrize("aggregation", ["sum", "mean", "median", "min", "max"])
def test_bar_aggregations(frame, aggregation):
    figure = charts.build(frame, "Bar", x="region", y="revenue", aggregation=aggregation)
    assert len(figure.data[0].y) == frame["region"].nunique()


def test_unknown_chart_type_is_refused(frame):
    with pytest.raises(ValueError, match="Unknown chart type"):
        charts.build(frame, "Sankey", x="region")


def test_heatmap_needs_two_numeric_columns():
    with pytest.raises(ValueError, match="at least two numeric"):
        charts.build(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), "Correlation heatmap")


def test_bar_caps_high_cardinality():
    frame = pd.DataFrame({"k": [f"k{i}" for i in range(500)], "v": range(500)})
    figure = charts.build(frame, "Bar", x="k", y="v")
    assert len(figure.data[0].x) <= 60


def test_pie_caps_slices():
    frame = pd.DataFrame({"k": [f"k{i}" for i in range(40)], "v": range(40)})
    figure = charts.build(frame, "Pie", x="k", y="v")
    assert len(figure.data[0].labels) <= 12


def test_titles_are_applied_and_omitted(frame):
    assert charts.build(frame, "Bar", x="region", y="revenue", title="T").layout.title.text == "T"
    assert charts.build(frame, "Bar", x="region", y="revenue").layout.title.text is None


def test_charts_carry_the_palette(frame):
    assert charts.build(frame, "Bar", x="region", y="revenue").layout.colorway == tuple(charts.PALETTE)


# ---------------- suggestions ----------------


def test_suggestions_match_the_columns(store, spec):
    suggestions = charts.suggest(spec)
    assert suggestions
    assert "Line" in {s.chart_type for s in suggestions}   # a time column exists
    for suggestion in suggestions:
        if suggestion.x:
            assert suggestion.x in {c.name for c in spec.columns}


def test_suggestions_respect_the_limit(store, spec):
    assert len(charts.suggest(spec, limit=3)) == 3
    assert len(charts.suggest(spec)) <= 6


def test_heatmap_is_offered_once_the_limit_allows(store, spec):
    """It is last in the ranking, so it only appears when more are requested."""
    assert "Correlation heatmap" in {s.chart_type for s in charts.suggest(spec, limit=10)}


def test_suggestions_survive_a_dimensionless_dataset(store):
    from groundtruth.semantic import profile

    store.register_frame("nums", pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2}))
    suggestions = charts.suggest(profile(store, "nums"))
    assert all(s.chart_type != "Box" for s in suggestions)


# ---------------- forecast and decomposition figures ----------------


def test_forecast_figure_has_history_forecast_and_band(frame):
    from groundtruth import timeseries as ts

    series = ts.aggregate(frame, "date", "revenue", "month", "sum")
    figure = charts.forecast_figure(ts.forecast(series, 6, "month"), "F")
    names = [t.name for t in figure.data]
    assert "Observed" in names and "Forecast" in names and "Prediction interval" in names


def test_decomposition_figure_has_four_panels(frame):
    from groundtruth import timeseries as ts

    series = ts.aggregate(frame, "date", "revenue", "month", "sum")
    figure = charts.decomposition_figure(ts.decompose(series, "month"), "D")
    assert len(figure.data) == 4
