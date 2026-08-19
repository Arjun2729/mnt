"""The agent's tools — exercised without a model, which is the point of the split."""
import pandas as pd
import pytest

from groundtruth.agent import TOOL_SCHEMAS, ToolBox, ask, ask_stream


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


# ---------------- vendor fields must survive the loop ----------------
#
# Gemini 3.x attaches a thought_signature to every functionCall and rejects the
# next request if it is missing. Rebuilding the assistant message field by field
# dropped it, producing a 400 on the second round of any tool conversation.


THOUGHT_SIGNATURE = "Cs8BAVKV5xQ"


def _fragment_with_signature():
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

    return ChoiceDeltaToolCall.construct(
        index=0, id="call_1", type="function",
        function={"name": "describe_columns", "arguments": "{}"},
        extra_content={"google": {"thought_signature": THOUGHT_SIGNATURE}},
    )


class _StrictProvider:
    """Rejects a follow-up whose tool calls lost their thought_signature."""

    def __init__(self):
        self.round = 0
        self.sent: list[list[dict]] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.round += 1
        self.sent.append(kwargs["messages"])
        if self.round > 1:
            for message in kwargs["messages"]:
                for call in message.get("tool_calls") or []:
                    signature = (call.get("extra_content") or {}).get("google", {}).get("thought_signature")
                    if not signature:
                        raise RuntimeError(
                            "Error code: 400 - Function call is missing a thought_signature "
                            "in functionCall parts."
                        )

        class _Delta:
            def __init__(self, content=None, tool_calls=None):
                self.content, self.tool_calls = content, tool_calls

        class _Choice:
            def __init__(self, delta):
                self.delta = delta

        class _Chunk:
            def __init__(self, delta):
                self.choices = [_Choice(delta)]

        if self.round == 1:
            return iter([_Chunk(_Delta(tool_calls=[_fragment_with_signature()]))])
        return iter([_Chunk(_Delta(content="The dataset has 7 columns."))])

    def echoed_tool_calls(self) -> list[dict]:
        return [
            call
            for turn in self.sent
            for message in turn
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        ]


def test_thought_signature_survives_the_streamed_loop(store, spec):
    provider = _StrictProvider()
    answer = None
    for kind, payload in ask_stream(provider, "gemini-3.6-flash", "how many columns?", toolbox(store, spec)):
        if kind == "done":
            answer = payload

    assert answer is not None
    assert answer.text == "The dataset has 7 columns."
    assert provider.round == 2, "the follow-up round must have been accepted"

    echoed = provider.echoed_tool_calls()
    assert echoed, "the assistant turn must be echoed back with its tool calls"
    assert echoed[0]["extra_content"]["google"]["thought_signature"] == THOUGHT_SIGNATURE


def test_streamed_arguments_still_concatenate_across_fragments(store, spec):
    """Preserving extra fields must not break the piecewise arguments."""
    from groundtruth.agent import _finalise_slot, _merge_fragment
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

    pending: dict = {}
    _merge_fragment(pending, ChoiceDeltaToolCall.construct(
        index=0, id="c1", type="function", function={"name": "run_sql", "arguments": '{"query":'}))
    _merge_fragment(pending, ChoiceDeltaToolCall.construct(
        index=0, function={"arguments": ' "SELECT 1"}'}))

    call = _finalise_slot(pending[0])
    assert call["function"]["name"] == "run_sql"
    assert call["function"]["arguments"] == '{"query": "SELECT 1"}'
    assert call["id"] == "c1"


def test_empty_arguments_default_to_an_object(store, spec):
    from groundtruth.agent import _finalise_slot

    call = _finalise_slot({"id": "c1", "function": {"name": "describe_columns", "arguments": ""}})
    assert call["function"]["arguments"] == "{}"
    assert call["type"] == "function"


def test_output_only_fields_are_not_sent_back():
    """Some providers reject their own output fields on input."""
    from groundtruth.agent import _assistant_message

    class _Message:
        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": "hi", "annotations": [], "reasoning_content": "x"}

    payload = _assistant_message(_Message())
    assert "annotations" not in payload and "reasoning_content" not in payload
    assert payload["content"] == "hi"


# ---------------- rate-limit resilience ----------------


class _RateLimited:
    """Fails with 429 a fixed number of times, then succeeds."""

    def __init__(self, failures: int):
        self.failures, self.calls = failures, 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(
                "Error code: 429 - RESOURCE_EXHAUSTED. Quota exceeded, limit: 5. "
                "Please retry in 2s."
            )
        return "response"


def test_rate_limits_are_waited_out():
    from groundtruth.agent import _create_with_retry

    client = _RateLimited(failures=2)
    waits: list[float] = []
    result = _create_with_retry(
        client, on_wait=lambda d, a, t: waits.append(d), sleep=lambda d: None,
        model="m", messages=[],
    )
    assert result == "response"
    assert client.calls == 3
    assert waits == [3.0, 3.0]  # the provider's 2s plus a second of headroom


def test_retries_are_bounded():
    from groundtruth.agent import _create_with_retry

    client = _RateLimited(failures=99)
    with pytest.raises(RuntimeError, match="429"):
        _create_with_retry(client, retries=2, sleep=lambda d: None, model="m", messages=[])
    assert client.calls == 3  # the first attempt plus two retries


def test_non_rate_limit_errors_are_not_retried():
    from groundtruth.agent import _create_with_retry

    class _Broken:
        calls = 0
        chat = property(lambda s: s)
        completions = property(lambda s: s)

        def create(self, **kwargs):
            type(self).calls += 1
            raise RuntimeError("Error code: 401 - invalid api key")

    client = _Broken()
    with pytest.raises(RuntimeError, match="401"):
        _create_with_retry(client, sleep=lambda d: None, model="m", messages=[])
    assert _Broken.calls == 1


def test_tool_rounds_are_capped_to_limit_request_spend():
    """Each round costs one metered request, so the ceiling matters on a free tier."""
    from groundtruth.agent import MAX_TOOL_ROUNDS

    assert MAX_TOOL_ROUNDS <= 6


# ---------------- malformed tool arguments ----------------
#
# A live model called run_stat_test with no arguments at all, which raised a raw
# TypeError. The model cannot correct itself from that; it needs to be told what
# was expected.


def test_missing_arguments_are_reported_not_raised(store, spec):
    call = toolbox(store, spec).dispatch("run_stat_test", {})
    assert call.error
    assert "test" in call.error and "x" in call.error and "y" in call.error


def test_partially_supplied_arguments_name_only_what_is_missing(store, spec):
    call = toolbox(store, spec).dispatch("run_stat_test", {"test": "correlation"})
    assert "x, y" in call.error
    assert "Missing required" in call.error


def test_unexpected_arguments_are_reported(store, spec):
    call = toolbox(store, spec).dispatch("run_sql", {"query": "SELECT 1", "nonsense": True})
    assert "Bad arguments" in call.error


def test_valid_arguments_still_work(store, spec):
    call = toolbox(store, spec).dispatch(
        "run_stat_test", {"test": "compare_groups", "x": "revenue", "y": "channel"}
    )
    assert not call.error


def test_required_arguments_come_from_the_schema():
    from groundtruth.agent import _required_arguments

    assert _required_arguments("run_stat_test") == ["test", "x", "y"]
    assert _required_arguments("describe_columns") == []
    assert _required_arguments("unknown_tool") == []
