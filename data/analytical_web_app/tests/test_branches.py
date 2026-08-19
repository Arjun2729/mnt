"""The remaining partial branches — conditions never yet evaluated both ways.

Each test here exercises the *false* side of a condition whose true side the rest
of the suite already covers.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from groundtruth import charts, insights, ml, semantic, timeseries
from groundtruth.agent import _merge_fragment
from groundtruth.report import Report, render_excel, render_html
from groundtruth.semantic import DatasetSpec, ColumnSpec, profile, profile_with_distributions
from groundtruth.store import Store


# ---------------- agent ----------------


def test_empty_fragment_fields_are_not_copied():
    """Streamed deltas carry blank placeholders that must not overwrite real values."""
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

    pending: dict = {}
    _merge_fragment(pending, ChoiceDeltaToolCall.construct(
        index=0, id="c1", type="function", function={"name": "run_sql", "arguments": "{}"}))
    _merge_fragment(pending, ChoiceDeltaToolCall.construct(
        index=0, id="", type="", function={"arguments": ""}))
    assert pending[0]["id"] == "c1"
    assert pending[0]["type"] == "function"


# ---------------- chart suggestions ----------------


def _spec(columns: list[ColumnSpec], time_column=None) -> DatasetSpec:
    return DatasetSpec(table="t", rows=100, columns=columns, time_column=time_column)


def _column(name, role, **kwargs) -> ColumnSpec:
    return ColumnSpec(name=name, sql_type="X", role=role, distinct=5, missing=0, missing_pct=0.0, **kwargs)


def test_dimensions_without_measures_suggest_nothing_numeric():
    spec = _spec([_column("a", semantic.ROLE_DIMENSION), _column("b", semantic.ROLE_DIMENSION)])
    assert charts.suggest(spec) == []


def test_a_single_measure_suggests_no_scatter_or_heatmap():
    spec = _spec([_column("a", semantic.ROLE_DIMENSION), _column("m", semantic.ROLE_MEASURE)])
    kinds = {s.chart_type for s in charts.suggest(spec, limit=10)}
    assert "Scatter" not in kinds and "Correlation heatmap" not in kinds
    assert "Bar" in kinds and "Histogram" in kinds


def test_no_columns_at_all_suggests_nothing():
    assert charts.suggest(_spec([])) == []


# ---------------- chart selections ----------------


def test_categorical_points_without_values_yield_nothing():
    selection = {"points": [{"x": None}, {"y": 3}]}
    assert charts.selection_to_conditions(
        selection, "Bar", "region", "revenue", "", x_is_measure=False, y_is_measure=True) == []


def test_points_with_no_x_column_configured():
    selection = {"points": [{"x": "East"}]}
    assert charts.selection_to_conditions(
        selection, "Bar", "", "revenue", "", x_is_measure=False, y_is_measure=True) == []


def test_colour_without_a_legend_group_yields_nothing():
    selection = {"points": [{"x": "East"}]}
    conditions = charts.selection_to_conditions(
        selection, "Bar", "region", "revenue", "channel", x_is_measure=False, y_is_measure=False)
    assert [c["column"] for c in conditions] == ["region"]


# ---------------- insights ----------------


def test_perfectly_linear_series_has_no_notable_jump():
    """Constant deltas have zero spread, so no period stands out."""
    store = Store()
    store.register_frame("t", pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "v": np.arange(12.0) * 10,
    }))
    found = insights.scan(store, "t", profile(store, "t"))
    assert all(f.kind != "movement" for f in found)
    store.close()


# ---------------- ml ----------------


def test_leakage_check_needs_enough_paired_rows():
    frame = pd.DataFrame({"x": [1.0, 2.0, None, None], "target": [1.0, 2.0, 3.0, 4.0]})
    assert ml.detect_leakage(frame, "target", ["x"], ml.REGRESSION) == []


def test_all_categorical_features_build_a_pipeline():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.choice(list("xyz"), 120),
        "b": rng.choice(list("pq"), 120),
        "target": rng.normal(size=120),
    })
    result = ml.train(frame, "target", ["a", "b"])
    assert result.problem_type == ml.REGRESSION


def test_multiclass_reports_no_auc():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "x": rng.normal(size=150),
        "target": rng.choice(["a", "b", "c"], 150),
    })
    result = ml.train(frame, "target", ["x"])
    assert "roc_auc" not in result.metrics
    assert "accuracy" in result.metrics


# ---------------- report ----------------


def test_removing_an_invalid_index_is_ignored():
    report = Report("T")
    report.add_text("one")
    report.remove(9)
    report.remove(-4)
    assert len(report.blocks) == 1


def test_moving_beyond_the_ends_is_ignored():
    report = Report("T")
    report.add_text("a")
    report.add_text("b")
    report.move(0, -1)
    report.move(1, 1)
    report.move(9, 1)
    assert [b.body for b in report.blocks] == ["a", "b"]


def test_blank_paragraphs_are_dropped():
    report = Report("T")
    report.add_text("first\n\n\n\nsecond")
    html = render_html(report, standalone=False)
    assert html.count("<p>") == 2


def test_a_chart_block_with_no_figure_is_skipped():
    report = Report("T")
    report.add_chart(None, "Empty", note="still noted")
    html = render_html(report, standalone=False)
    assert "still noted" in html
    assert "plotly" not in html.lower()


def test_duplicate_metric_titles_are_written_once():
    report = Report("T")
    report.add_metrics({"a": 1}, "Same")
    report.add_metrics({"b": 2}, "Same")
    assert len(render_excel(report)) > 0


# ---------------- semantic ----------------


def test_measure_whose_bounds_collapse_under_float_conversion():
    """Two distinct integers can round to the same float, leaving no range to bin."""
    store = Store()
    store.con.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1000000000000000001), (1000000000000000002), "
        "(1000000000000000003)) AS v(big)"
    )
    spec = profile(store, "t")
    column = spec.column("big")
    assert column.distinct == 3          # three distinct, so it is a measure not a flag
    assert float(column.min_value) == float(column.max_value)
    shaped = profile_with_distributions(store, spec).set_index("column")
    assert shaped.loc["big", "shape"] == []
    store.close()


def test_text_columns_get_no_shape():
    store = Store()
    store.register_frame("t", pd.DataFrame({"note": [f"unique text {i}" for i in range(80)]}))
    spec = profile(store, "t")
    assert spec.column("note").role == semantic.ROLE_TEXT
    shaped = profile_with_distributions(store, spec).set_index("column")
    assert shaped.loc["note", "shape"] == []
    store.close()


# ---------------- store ----------------


def test_readonly_queries_can_opt_out_of_the_row_cap(store):
    """Internal callers that need the whole result pass limit=None."""
    uncapped = store.sql_readonly("SELECT * FROM sample", limit=None)
    assert len(uncapped) == 288


# ---------------- timeseries ----------------


def test_noisy_series_uses_the_robust_scale():
    """A varied residual gives a usable MAD, so the fallback is not needed."""
    rng = np.random.default_rng(3)
    index = pd.date_range("2024-01-01", periods=40, freq="MS")
    values = 100 + rng.normal(0, 5, 40)
    values[20] = 500.0
    found = timeseries.anomalies(pd.Series(values, index=index), "month", sensitivity=3.0)
    assert len(found) >= 1
    assert 500.0 in set(found["value"])
