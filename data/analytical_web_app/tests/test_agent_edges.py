"""Agent tool behaviour on empty results, big results and bad input."""
import json

import pandas as pd
import pytest

from groundtruth.agent import AgentAnswer, ToolBox, ToolCall, _assistant_message, _dump, ask, ask_stream


def toolbox(store, spec):
    return ToolBox(store, "sample", spec)


def test_empty_result_is_reported(store, spec):
    call = toolbox(store, spec).dispatch("run_sql", {"query": "SELECT * FROM sample WHERE 1=0"})
    assert "no rows" in call.result_summary
    assert call.dataframe is not None and call.dataframe.empty


def test_large_results_are_truncated(store, spec):
    """The model gets a bounded excerpt, not an unbounded dump."""
    box = ToolBox(store, "sample", spec, row_cap=5000)
    call = box.dispatch("run_sql", {"query": "SELECT * FROM sample, range(40) AS r(i)"})
    assert "rows total, first 50 shown" in call.result_summary
    assert len(call.result_summary) < 14000


def test_chi_square_through_the_tool(store, spec):
    call = toolbox(store, spec).dispatch(
        "run_stat_test", {"test": "chi_square", "x": "region", "y": "channel"})
    assert not call.error
    assert "cramers_v" in call.dataframe.columns


def test_unknown_stat_test_is_reported(store, spec):
    call = toolbox(store, spec).dispatch("run_stat_test", {"test": "anova", "x": "revenue", "y": "region"})
    assert "Unknown test" in call.error


def test_chart_on_an_empty_result_is_reported(store, spec):
    call = toolbox(store, spec).dispatch("make_chart", {
        "query": "SELECT region, 1 AS v FROM sample WHERE 1=0",
        "chart_type": "bar", "x": "region", "y": "v", "title": "t"})
    assert "no rows" in call.error


def test_pie_chart_through_the_tool(store, spec):
    call = toolbox(store, spec).dispatch("make_chart", {
        "query": "SELECT region, SUM(revenue) v FROM sample GROUP BY 1",
        "chart_type": "pie", "x": "region", "y": "v", "title": "t"})
    assert not call.error and call.figure is not None


def test_answer_exposes_its_figures():
    figure = object()
    answer = AgentAnswer("text", [ToolCall("make_chart", {}, "", figure=figure)])
    assert answer.figures == [figure]


def test_dump_handles_every_shape():
    assert _dump(None) == {}
    assert _dump({"a": 1, "b": None}) == {"a": 1}
    assert _dump(42) == {}

    class _Plain:
        def __init__(self):
            self.a, self.b, self._hidden = 1, None, "x"

    assert _dump(_Plain()) == {"a": 1}


def test_assistant_message_always_carries_a_role():
    assert _assistant_message({"content": "hi"})["role"] == "assistant"


# ---------------- malformed streamed arguments ----------------


class _BadArgsClient:
    """Emits tool-call arguments that are not valid JSON."""

    def __init__(self, streaming: bool):
        self.streaming, self.round = streaming, 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.round += 1
        if self.streaming:
            return self._stream()
        return self._whole()

    def _stream(self):
        from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

        class _D:
            def __init__(s, content=None, tool_calls=None):
                s.content, s.tool_calls = content, tool_calls

        class _C:
            def __init__(s, d):
                s.delta = d

        class _Ch:
            def __init__(s, d):
                s.choices = [_C(d)]

        if self.round == 1:
            return iter([_Ch(_D(tool_calls=[ChoiceDeltaToolCall.construct(
                index=0, id="c1", type="function",
                function={"name": "describe_columns", "arguments": "{not json"})]))])
        return iter([_Ch(_D(content="done"))])

    def _whole(self):
        class _Fn:
            name, arguments = "describe_columns", "{not json"

        class _TC:
            id, type, function = "c1", "function", _Fn()

            def model_dump(self, exclude_none=True):
                return {"id": "c1", "type": "function",
                        "function": {"name": "describe_columns", "arguments": "{not json"}}

        class _Msg:
            content, tool_calls = None, [_TC()]

            def model_dump(self, exclude_none=True):
                return {"role": "assistant", "tool_calls": [_TC().model_dump()]}

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp() if self.round == 1 else type(
            "R", (), {"choices": [type("C", (), {"message": type(
                "M", (), {"content": "done", "tool_calls": None})()})()]})()


def test_unparseable_streamed_arguments_become_an_empty_call(store, spec):
    answer = None
    for kind, payload in ask_stream(_BadArgsClient(streaming=True), "m", "q", toolbox(store, spec)):
        if kind == "done":
            answer = payload
    assert answer.text == "done"
    assert answer.tool_calls and not answer.tool_calls[0].error


def test_unparseable_arguments_in_the_blocking_loop(store, spec):
    answer = ask(_BadArgsClient(streaming=False), "m", "q", toolbox(store, spec))
    assert answer.text == "done"


def test_on_tool_callback_fires(store, spec):
    seen = []
    ask(_BadArgsClient(streaming=False), "m", "q", toolbox(store, spec), on_tool=seen.append)
    assert len(seen) == 1
