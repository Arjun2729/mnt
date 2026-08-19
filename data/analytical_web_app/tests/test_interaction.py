"""Cross-filtering and the streamed agent loop — the interactive layer."""
import pandas as pd
import pytest

from groundtruth.charts import add_time_controls, build, describe_selection, selection_to_conditions
from groundtruth.filters import Condition, Group, compile_tree
from groundtruth.semantic import profile_with_distributions


# ---------------- chart selection -> filter conditions ----------------


def test_clicking_categories_builds_an_in_condition():
    selection = {"points": [{"x": "East"}, {"x": "West"}]}
    conditions = selection_to_conditions(selection, "Bar", "region", "revenue", "", False, True)
    assert conditions == [{"column": "region", "operator": "in", "value": ["East", "West"]}]


def test_clicked_duplicates_collapse():
    selection = {"points": [{"x": "East"}, {"x": "East"}]}
    conditions = selection_to_conditions(selection, "Bar", "region", "revenue", "", False, True)
    assert conditions[0]["value"] == ["East"]


def test_box_drag_over_measures_builds_ranges():
    selection = {"box": [{"x": [10.0, 20.0], "y": [100.0, 200.0]}]}
    conditions = selection_to_conditions(selection, "Scatter", "cost", "revenue", "", True, True)
    assert {c["column"] for c in conditions} == {"cost", "revenue"}
    assert all(c["operator"] == "between" for c in conditions)


def test_box_ignores_categorical_axes():
    selection = {"box": [{"x": [0, 1], "y": [100.0, 200.0]}]}
    conditions = selection_to_conditions(selection, "Bar", "region", "revenue", "", False, True)
    assert [c["column"] for c in conditions] == ["revenue"]


def test_box_takes_precedence_over_points():
    selection = {"box": [{"x": [1.0, 2.0]}], "points": [{"x": 1.5}]}
    conditions = selection_to_conditions(selection, "Scatter", "cost", "revenue", "", True, False)
    assert len(conditions) == 1
    assert conditions[0]["operator"] == "between"


def test_colour_series_adds_its_own_condition():
    selection = {"points": [{"x": "2024-03-01", "legendgroup": "Paid"}]}
    conditions = selection_to_conditions(selection, "Line", "date", "revenue", "channel", False, True)
    assert {"column": "channel", "operator": "in", "value": ["Paid"]} in conditions


def test_empty_selection_yields_nothing():
    assert selection_to_conditions(None, "Bar", "a", "b", "", False, True) == []
    assert selection_to_conditions({}, "Bar", "a", "b", "", False, True) == []
    assert selection_to_conditions({"points": []}, "Bar", "a", "b", "", False, True) == []


def test_selection_compiles_to_working_sql(store):
    """A click must survive the whole path: selection -> condition -> SQL -> rows."""
    selection = {"points": [{"x": "East"}, {"x": "North"}]}
    conditions = selection_to_conditions(selection, "Bar", "region", "revenue", "", False, True)
    tree = Group("AND", [Condition(c["column"], c["operator"], c["value"]) for c in conditions])
    where, params = compile_tree(tree)
    matched = store.count("sample", where, params)
    assert 0 < matched < store.count("sample")
    assert matched == store.count("sample", '"region" IN (?, ?)', ["East", "North"])


def test_description_is_human_readable():
    conditions = [
        {"column": "region", "operator": "in", "value": ["East", "West", "North", "South"]},
        {"column": "revenue", "operator": "between", "value": [1000, 2000]},
    ]
    text = describe_selection(conditions)
    assert "region" in text and "…" in text and "1,000–2,000" in text


# ---------------- chart affordances ----------------


def test_time_controls_attach_a_slider(frame):
    figure = build(frame, "Line", x="date", y="revenue", title="t")
    add_time_controls(figure, "month")
    assert figure.layout.xaxis.rangeslider.visible is True
    labels = [b.label for b in figure.layout.xaxis.rangeselector.buttons]
    assert "All" in labels


def test_untitled_chart_has_no_title(frame):
    """A title dict holding text=None renders the string 'undefined'."""
    assert build(frame, "Bar", x="region", y="revenue").layout.title.text is None


# ---------------- inline distributions ----------------


def test_distributions_are_normalised(store, spec):
    shaped = profile_with_distributions(store, spec)
    assert set(shaped["column"]) == {c.name for c in spec.columns}
    for shape in shaped["shape"]:
        if shape:
            assert max(shape) == pytest.approx(1.0)
            assert min(shape) >= 0


def test_measures_get_a_histogram(store, spec):
    shaped = profile_with_distributions(store, spec, bins=16).set_index("column")
    assert len(shaped.loc["revenue", "shape"]) == 16


def test_completeness_reflects_missingness(store, spec):
    shaped = profile_with_distributions(store, spec).set_index("column")
    assert shaped.loc["conversion_rate", "complete"] < 1.0
    assert shaped.loc["revenue", "complete"] == 1.0
