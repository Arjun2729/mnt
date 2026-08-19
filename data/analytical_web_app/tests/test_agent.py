"""The agent's tools — exercised without a model, which is the point of the split."""
import pandas as pd

from groundtruth.agent import TOOL_SCHEMAS, ToolBox, ask


def toolbox(store, spec):
    return ToolBox(store, "sample", spec)


def test_every_tool_has_a_schema():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"run_sql", "describe_columns", "make_chart", "run_stat_test"}


def test_run_sql_returns_rows(store, spec):
    call = toolbox(store, spec).dispatch(
        "run_sql", {"query": "SELECT channel, AVG(revenue) a FROM sample GROUP BY 1"}
    )
    assert not call.error
    assert len(call.dataframe) == 3


def test_run_sql_refuses_writes(store, spec):
    call = toolbox(store, spec).dispatch("run_sql", {"query": "DROP TABLE sample"})
    assert "read-only" in call.error


def test_sql_errors_come_back_for_self_correction(store, spec):
    """A bad column must return an error the model can read, not raise."""
    call = toolbox(store, spec).dispatch("run_sql", {"query": "SELECT nope FROM sample"})
    assert call.error
    assert "nope" in call.error


def test_describe_columns_exposes_schema_without_rows(store, spec):
    call = toolbox(store, spec).dispatch("describe_columns", {})
    assert "revenue" in call.result_summary
    assert "sample_rows" not in call.result_summary


def test_make_chart_builds_a_figure(store, spec):
    call = toolbox(store, spec).dispatch("make_chart", {
        "query": "SELECT region, SUM(revenue) rev FROM sample GROUP BY 1",
        "chart_type": "bar", "x": "region", "y": "rev", "title": "t",
    })
    assert not call.error
    assert call.figure is not None


def test_make_chart_reports_a_missing_column(store, spec):
    call = toolbox(store, spec).dispatch("make_chart", {
        "query": "SELECT region FROM sample", "chart_type": "bar",
        "x": "region", "y": "absent", "title": "t",
    })
    assert "absent" in call.error


def test_stat_tool_returns_an_effect_size(store, spec):
    call = toolbox(store, spec).dispatch(
        "run_stat_test", {"test": "correlation", "x": "revenue", "y": "orders"}
    )
    assert not call.error
    assert "r" in call.dataframe.columns


def test_unknown_tool_is_reported(store, spec):
    assert "Unknown tool" in toolbox(store, spec).dispatch("nope", {}).error


class _FakeClient:
    """Minimal stand-in for the OpenAI client, so the loop is testable offline."""

    class _Function:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

    class _Call:
        def __init__(self, name, arguments):
            self.id, self.type = "call_1", "function"
            self.function = _FakeClient._Function(name, arguments)

    class _Message:
        def __init__(self, content=None, tool_calls=None):
            self.content, self.tool_calls = content, tool_calls

    class _Choice:
        def __init__(self, message):
            self.message = message

    class _Response:
        def __init__(self, message):
            self.choices = [_FakeClient._Choice(message)]

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        message = self.scripted[min(self.calls, len(self.scripted) - 1)]
        self.calls += 1
        return _FakeClient._Response(message)


def test_tool_loop_executes_then_answers(store, spec):
    scripted = [
        _FakeClient._Message(tool_calls=[_FakeClient._Call("run_sql", '{"query": "SELECT COUNT(*) n FROM sample"}')]),
        _FakeClient._Message(content="There are 288 rows."),
    ]
    answer = ask(_FakeClient(scripted), "fake-model", "how many rows?", toolbox(store, spec))
    assert answer.text == "There are 288 rows."
    assert answer.queries == ["SELECT COUNT(*) n FROM sample"]
    assert answer.rounds == 2


def test_tool_loop_stops_at_the_round_limit(store, spec):
    looping = [_FakeClient._Message(tool_calls=[_FakeClient._Call("run_sql", '{"query": "SELECT 1"}')])]
    answer = ask(_FakeClient(looping), "fake-model", "loop forever", toolbox(store, spec), max_rounds=3)
    assert answer.rounds == 3
    assert "tool-call limit" in answer.text
