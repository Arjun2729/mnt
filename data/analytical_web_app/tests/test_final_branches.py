"""The last defensive branches: corrupt state, degenerate inputs, guard rails."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from groundtruth import charts, connectors, insights, llm, ml, security, semantic, stats, timeseries
from groundtruth.alerts import AlertStore, Rule, evaluate, evaluations_frame
from groundtruth.report import Report, render_html
from groundtruth.semantic import profile
from groundtruth.store import Store

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------- alerts ----------------


def test_threshold_rule_describes_itself():
    assert Rule("r", "revenue", "sum", ">=", 1234.5).describe() == "sum(revenue) >= 1,234"


def test_corrupt_state_file_is_ignored(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not json")
    store = AlertStore(path)
    assert store.states == {} and store.history == []


def test_unreadable_state_file_is_ignored(tmp_path):
    """A directory where a file is expected must not crash startup."""
    path = tmp_path / "state.json"
    path.mkdir()
    assert AlertStore(path).states == {}


def test_naive_timestamps_in_state_are_handled(tmp_path):
    """State written by an older build has no timezone; cooldown must still work."""
    store = AlertStore(tmp_path / "s.json")
    rule = Rule("r", "metric", "mean", ">", 1, cooldown_minutes=60)
    state = store.get("r")
    state.state = "firing"
    state.last_notified = "2026-01-01T00:00:00"          # naive
    result = evaluate(rule, pd.DataFrame({"metric": [0]}), store, T0 + timedelta(minutes=5))
    assert result.transition == "suppressed"


def test_unparseable_timestamp_does_not_suppress(tmp_path):
    store = AlertStore(tmp_path / "s.json")
    rule = Rule("r", "metric", "mean", ">", 1, cooldown_minutes=60)
    state = store.get("r")
    state.state = "firing"
    state.last_notified = "not a timestamp"
    result = evaluate(rule, pd.DataFrame({"metric": [0]}), store, T0)
    assert result.transition == "recovered"


def test_evaluations_render_as_a_table(tmp_path):
    store = AlertStore(tmp_path / "s.json")
    rule = Rule("r", "metric", "mean", ">", 1, cooldown_minutes=0)
    table = evaluations_frame([evaluate(rule, pd.DataFrame({"metric": [5]}), store, T0)])
    assert list(table.columns) == ["rule", "condition", "value", "state", "transition", "notify"]


# ---------------- charts ----------------


def test_zero_width_selection_is_returned_unchanged():
    assert charts._round_bounds(5.0, 5.0) == (5.0, 5.0)


def test_clicking_numeric_points_yields_a_range():
    selection = {"points": [{"x": 10.4}, {"x": 20.9}]}
    conditions = charts.selection_to_conditions(
        selection, "Scatter", "cost", "revenue", "", x_is_measure=True, y_is_measure=True)
    assert conditions[0]["operator"] == "between"
    assert conditions[0]["column"] == "cost"


def test_numeric_points_without_values_yield_nothing():
    selection = {"points": [{"x": None}]}
    assert charts.selection_to_conditions(
        selection, "Scatter", "cost", "revenue", "", x_is_measure=True, y_is_measure=True) == []


# ---------------- connectors ----------------


def test_repeated_column_names_are_numbered():
    store = Store()
    frame = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "a"])
    result = connectors._finish(store, "d", frame, "test", "")
    assert sorted(result.dataset.columns) == ["a", "a_1", "a_2"]
    store.close()


def test_ragged_dict_payload_falls_back_to_one_record(monkeypatch):
    """A dict of unequal-length lists is not a table; it becomes a single record."""
    store = Store()
    monkeypatch.setattr(security, "_resolves_to_private_address", lambda host: False)

    class _R:
        def json(self):
            return {"a": [1, 2, 3], "b": [1]}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(connectors.requests, "get", lambda *a, **k: _R())
    result = connectors.load_api(store, "https://api.test/x", allowed_hosts=["api.test"])
    assert result.dataset.rows == 1
    store.close()


# ---------------- insights ----------------


def test_segment_scale_falls_back_to_standard_deviation():
    """Three identical group means make the MAD exactly zero. The scale must fall
    back to the standard deviation rather than skipping the comparison."""
    store = Store()
    ordinary = [100.0] * 20 + [102.0] * 20          # mean 101, repeated for a, b and c
    store.register_frame("t", pd.DataFrame({
        "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40 + ["d"] * 40,
        "v": ordinary * 3 + [400.0] * 40,
    }))
    spec = profile(store, "t")
    assert spec.measures == ["v"], "v must be a measure for the segment scan to run"
    segments = [f for f in insights.scan(store, "t", spec) if f.kind == "segment"]
    assert segments and "d" in segments[0].headline
    store.close()


def test_high_cardinality_dimension_is_skipped():
    store = Store()
    frame = pd.DataFrame({"code": [f"c{i}" for i in range(60)] * 2, "v": np.arange(120.0)})
    store.register_frame("t", frame)
    spec = profile(store, "t")
    spec.columns[0].role = semantic.ROLE_DIMENSION      # force the branch
    assert all(f.kind != "segment" for f in insights.scan(store, "t", spec))
    store.close()


def test_short_series_produces_no_trend():
    store = Store()
    frame = pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=3, freq="MS"),
        "v": [1.0, 2.0, 3.0],
    })
    store.register_frame("t", frame)
    assert all(f.kind not in ("trend", "movement") for f in insights.scan(store, "t", profile(store, "t")))
    store.close()


def test_zero_iqr_measure_is_skipped():
    store = Store()
    store.register_frame("t", pd.DataFrame({"v": [5.0] * 40, "w": np.arange(40.0)}))
    assert all("v" not in f.headline for f in insights.scan(store, "t", profile(store, "t")) if f.kind == "outlier")
    store.close()


# ---------------- ml, stats, security, report, semantic, timeseries ----------------


def test_every_candidate_failing_is_fatal(frame, monkeypatch):
    def always_fails(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(ml, "cross_val_score", always_fails)
    with pytest.raises(RuntimeError):
        ml.train(frame, "revenue", ["cost", "orders"])


def test_auc_failure_is_swallowed(monkeypatch):
    """A probability failure must not lose the metrics that did compute."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    frame = pd.DataFrame({"x": x, "target": np.where(x > 0, "a", "b")})
    monkeypatch.setattr(ml, "roc_auc_score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no auc")))
    result = ml.train(frame, "target", ["x"])
    assert "accuracy" in result.metrics and "roc_auc" not in result.metrics


def test_correlation_scan_survives_a_failing_pair(frame, monkeypatch):
    real = stats.correlation
    calls = {"n": 0}

    def flaky(df, x, y, method="pearson", alpha=0.05):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated")
        return real(df, x, y, method, alpha)

    monkeypatch.setattr(stats, "correlation", flaky)
    assert not stats.correlation_scan(frame, ["revenue", "cost", "orders"]).empty


@pytest.mark.parametrize("statement", [
    "CREATE TABLE t AS SELECT 1", "INSERT INTO t SELECT 1", "COPY (SELECT 1) TO 'x.csv'",
    "DELETE FROM t", "UPDATE t SET a = 1", "ATTACH 'x.db'", "INSTALL httpfs", "LOAD httpfs",
])
def test_write_statements_are_blocked_by_the_parser_alone(statement):
    """A keyword-prefix check used to follow the parser check. It was unreachable:
    DuckDB types every one of these itself, so the parser rejects them first."""
    with pytest.raises(security.SecurityError, match="read-only"):
        security.assert_read_only(statement)


def test_report_notes_are_rendered():
    report = Report("T")
    report.add_table(pd.DataFrame({"a": [1]}), "Table", note="Sourced from the filtered view.")
    assert "Sourced from the filtered view." in render_html(report, standalone=False)


def test_columns_with_no_values_are_skipped_by_coercion():
    store = Store()
    store.register_frame("t", pd.DataFrame({"a": pd.Series([], dtype="object")}))
    assert semantic.coerce_types(store, "t") == []
    store.close()


def test_medium_cardinality_text_is_text():
    store = Store()
    values = [f"note {i // 2}" for i in range(200)]      # 100 distinct of 200 rows
    store.register_frame("t", pd.DataFrame({"note": values}))
    assert profile(store, "t").column("note").role == semantic.ROLE_TEXT
    store.close()


def test_multi_year_gaps_read_as_yearly():
    store = Store()
    stamps = pd.to_datetime([f"{year}-01-01" for year in range(1900, 1960, 5)])
    store.register_frame("t", pd.DataFrame({"when": stamps}))
    assert profile(store, "t").time_grain == "year"
    store.close()


def test_distribution_of_an_empty_dataset():
    store = Store()
    store.register_frame("t", pd.DataFrame({"a": [1.0]}))
    spec = profile(store, "t")
    spec.rows = 0                                        # force the zero-row branch
    assert not semantic.profile_with_distributions(store, spec).empty
    store.close()


def test_changepoint_search_stops_at_the_cap():
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    blocks = [rng.normal(level, 0.5, 25) for level in (0, 50, 0, 50, 0, 50, 0, 50)]
    found = timeseries.detect_changepoints(pd.Series(np.concatenate(blocks), index=index), max_points=3)
    assert len(found) <= 3


def test_suggest_replacement_needs_a_versioned_id():
    assert llm.suggest_replacement("deprecated, please use something") is None


# ---------------- the last branches ----------------


def test_usage_only_chunks_are_skipped(store, spec):
    """Some providers emit trailing chunks carrying no choices at all."""
    from groundtruth.agent import ToolBox, ask_stream

    class _Client:
        def __init__(self):
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            class _Empty:
                choices = []

            class _Delta:
                content, tool_calls = "hello", None

            class _Choice:
                delta = _Delta()

            class _Chunk:
                choices = [_Choice()]

            return iter([_Empty(), _Chunk(), _Empty()])

    answer = None
    for kind, payload in ask_stream(_Client(), "m", "q", ToolBox(store, "sample", spec)):
        if kind == "done":
            answer = payload
    assert answer.text == "hello"


def test_streamed_loop_stops_at_the_round_limit(store, spec):
    from groundtruth.agent import ToolBox, ask_stream
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

    class _Looping:
        def __init__(self):
            self.chat = self
            self.rounds = 0

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.rounds += 1

            class _Delta:
                content = None
                tool_calls = [ChoiceDeltaToolCall.construct(
                    index=0, id=f"c{self.rounds}", type="function",
                    function={"name": "describe_columns", "arguments": "{}"})]

            class _Choice:
                delta = _Delta()

            class _Chunk:
                choices = [_Choice()]

            return iter([_Chunk()])

    client = _Looping()
    answer = None
    for kind, payload in ask_stream(client, "m", "q", ToolBox(store, "sample", spec), max_rounds=3):
        if kind == "done":
            answer = payload
    assert answer.rounds == 3
    assert "tool-call limit" in answer.text
    assert client.rounds == 3


def test_identical_group_means_yield_no_segment_finding():
    """Every group the same: both the MAD and the standard deviation are zero, so
    there is no scale to judge against and the comparison must be abandoned."""
    store = Store()
    store.register_frame("t", pd.DataFrame({
        "g": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
        "v": [0.0, 1.0, 2.0] * 30,          # three distinct values, identical group means
    }))
    spec = profile(store, "t")
    assert spec.measures == ["v"]
    assert all(f.kind != "segment" for f in insights.scan(store, "t", spec))
    store.close()


def test_measure_with_no_spread_is_skipped_for_outliers():
    """A zero interquartile range has no outlier boundary to compute."""
    store = Store()
    values = [10.0] * 98 + [11.0, 12.0]          # q1 == q3 == 10
    store.register_frame("t", pd.DataFrame({"v": values}))
    found = insights.scan(store, "t", profile(store, "t"))
    assert all(f.kind != "outlier" for f in found)
    store.close()


def test_retirement_message_without_a_usable_id():
    assert llm.suggest_replacement("this model is no longer available") is None


def test_all_candidates_failing_is_reported(frame, monkeypatch):
    """The baseline still fits, so the failure must be the explicit message."""
    real = ml.cross_val_score

    def only_candidates_fail(pipeline, X, y, **kwargs):
        model = pipeline.named_steps["model"]
        if type(model).__name__.startswith("Dummy"):
            return real(pipeline, X, y, **kwargs)
        raise RuntimeError("simulated candidate failure")

    monkeypatch.setattr(ml, "cross_val_score", only_candidates_fail)
    with pytest.raises(ValueError, match="No candidate model could be fitted"):
        ml.train(frame, "revenue", ["cost", "orders"])


def test_all_null_text_column_is_skipped_by_coercion():
    """A VARCHAR column with no values has nothing to infer a type from."""
    store = Store()
    store.con.execute("CREATE TABLE t AS SELECT CAST(NULL AS VARCHAR) AS created_at FROM range(20)")
    assert store.schema("t")["data_type"].iloc[0] == "VARCHAR"
    assert semantic.coerce_types(store, "t") == []
    store.close()


def test_profile_frame_renders(spec):
    table = semantic.profile_frame(spec)
    assert list(table.columns) == ["column", "role", "type", "distinct", "missing", "missing_%", "mean"]
    assert len(table) == len(spec.columns)


def test_changepoint_search_returns_on_a_short_window():
    tiny = pd.Series([1.0] * 8, index=pd.date_range("2024-01-01", periods=8, freq="D"))
    assert timeseries.detect_changepoints(tiny, min_segment=4) == []


def test_changepoint_search_stops_once_the_budget_is_spent():
    """After the cap is reached the recursion must unwind rather than keep splitting."""
    index = pd.date_range("2020-01-01", periods=90, freq="D")
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 30), rng.normal(40, 1, 30), rng.normal(80, 1, 30)])
    assert len(timeseries.detect_changepoints(pd.Series(values, index=index), max_points=1)) == 1
