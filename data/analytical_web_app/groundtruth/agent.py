"""L4 — the analyst agent.

The model is given tools, not a sample of rows. It answers by executing read-only
SQL against the current filtered view, so every figure it reports is computed
rather than estimated, and the query that produced it is returned alongside the
answer for anyone who wants to check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .security import SecurityError
from .semantic import DatasetSpec
from .store import Store

MAX_TOOL_ROUNDS = 6  # each round is one metered request
MAX_ROWS_RETURNED = 200

SYSTEM_PROMPT = """You are a careful data analyst working over a SQL table in DuckDB.

Rules:
- Answer by querying. Never estimate a number you could compute with run_sql.
- The table you may query is named in the schema. Use exact column names.
- DuckDB SQL dialect. Read-only SELECT statements only.
- If a question is ambiguous, state the interpretation you chose, then answer it.
- Distinguish what the data shows from what you infer. Do not claim causation from correlation.
- If the data cannot answer the question, say so plainly and say what would be needed.
- Prefer a few well-chosen aggregate queries over many small ones. Each call costs
  rate-limit budget, so combine what you need into as few queries as possible.
- Batch independent lookups into one SQL statement rather than issuing several.
- When a result is best seen as a picture, call make_chart after computing it.
- Be concise. Lead with the answer, then the supporting numbers."""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a read-only DuckDB SELECT against the filtered dataset and return rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single SELECT statement."},
                    "purpose": {"type": "string", "description": "One short line on what this answers."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_columns",
            "description": "Return roles, types, distinct counts and example values for the dataset's columns.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "Render a chart from a SQL query. Use after establishing the numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SELECT producing the plotted columns."},
                    "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "area", "pie"]},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "color": {"type": "string", "description": "Optional grouping column."},
                    "title": {"type": "string"},
                },
                "required": ["query", "chart_type", "x", "y", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_stat_test",
            "description": "Run a significance test with an effect size: correlation, chi_square, or compare_groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "enum": ["correlation", "chi_square", "compare_groups"]},
                    "x": {"type": "string", "description": "First column (numeric for correlation, the value column for compare_groups)."},
                    "y": {"type": "string", "description": "Second column (the grouping column for compare_groups)."},
                    "method": {"type": "string", "enum": ["pearson", "spearman"], "description": "Correlation only."},
                },
                "required": ["test", "x", "y"],
            },
        },
    },
]



def _required_arguments(name: str) -> list[str]:
    """The required parameters a tool declares, straight from its schema."""
    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return list(schema["function"].get("parameters", {}).get("required", []))
    return []


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result_summary: str
    dataframe: pd.DataFrame | None = None
    figure: Any = None
    error: str = ""


@dataclass
class AgentAnswer:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    rounds: int = 0

    @property
    def queries(self) -> list[str]:
        return [c.arguments.get("query", "") for c in self.tool_calls if c.arguments.get("query")]

    @property
    def figures(self) -> list[Any]:
        return [c.figure for c in self.tool_calls if c.figure is not None]


class ToolBox:
    """Executes the agent's tool calls against the store. Provider-independent."""

    def __init__(self, store: Store, view: str, spec: DatasetSpec, row_cap: int = MAX_ROWS_RETURNED) -> None:
        self.store = store
        self.view = view
        self.spec = spec
        self.row_cap = row_cap

    def describe_columns(self) -> tuple[str, None]:
        payload = self.spec.to_prompt_json()
        payload["table"] = self.view
        return json.dumps(payload, default=str), None

    def run_sql(self, query: str, purpose: str = "") -> tuple[str, pd.DataFrame]:
        frame = self.store.sql_readonly(query, limit=self.row_cap)
        if frame.empty:
            return "Query returned no rows.", frame
        text = frame.to_csv(index=False)
        if len(text) > 12000:
            text = frame.head(50).to_csv(index=False) + f"\n... {len(frame)} rows total, first 50 shown"
        return text, frame

    def run_stat_test(self, test: str, x: str, y: str, method: str = "pearson") -> tuple[str, pd.DataFrame]:
        from . import stats as stats_module

        frame = self.store.materialize(self.view)[sorted({x, y})]

        if test == "correlation":
            result = stats_module.correlation(frame, x, y, method)
        elif test == "chi_square":
            result = stats_module.chi_square(frame, x, y)
        elif test == "compare_groups":
            result = stats_module.compare_groups(frame, x, y)
        else:
            raise ValueError(f"Unknown test: {test}")
        row = result.as_row()
        return json.dumps(row, default=str), pd.DataFrame([row])

    def make_chart(self, query: str, chart_type: str, x: str, y: str, title: str, color: str = "") -> tuple[str, pd.DataFrame, Any]:
        import plotly.express as px

        frame = self.store.sql_readonly(query, limit=5000)
        if frame.empty:
            raise ValueError("Chart query returned no rows.")
        for column in (x, y, *([color] if color else [])):
            if column not in frame.columns:
                raise ValueError(f"Column {column!r} is not in the query result: {list(frame.columns)}")
        builders = {"bar": px.bar, "line": px.line, "scatter": px.scatter, "area": px.area, "pie": px.pie}
        if chart_type == "pie":
            figure = px.pie(frame, names=x, values=y, title=title)
        else:
            figure = builders[chart_type](frame, x=x, y=y, color=color or None, title=title)
        figure.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        return f"Chart rendered: {chart_type} of {y} by {x} ({len(frame)} rows).", frame, figure

    def dispatch(self, name: str, arguments: dict) -> ToolCall:
        # Models sometimes call a tool with missing or misspelled arguments. Turning
        # that into a readable message lets the model correct itself; a raw TypeError
        # tells it nothing about what was expected.
        required = _required_arguments(name)
        missing = [key for key in required if key not in arguments]
        if missing:
            return ToolCall(
                name, arguments, "",
                error=f"Missing required argument(s): {', '.join(missing)}. "
                      f"{name} requires: {', '.join(required)}.",
            )
        try:
            if name == "run_sql":
                summary, frame = self.run_sql(**arguments)
                return ToolCall(name, arguments, summary, frame)
            if name == "describe_columns":
                summary, _ = self.describe_columns()
                return ToolCall(name, arguments, summary)
            if name == "run_stat_test":
                summary, frame = self.run_stat_test(**arguments)
                return ToolCall(name, arguments, summary, frame)
            if name == "make_chart":
                summary, frame, figure = self.make_chart(**arguments)
                return ToolCall(name, arguments, summary, frame, figure)
            return ToolCall(name, arguments, "", error=f"Unknown tool: {name}")
        except TypeError as exc:
            return ToolCall(
                name, arguments, "",
                error=f"Bad arguments for {name}: {exc}. Expected: {', '.join(_required_arguments(name)) or 'none'}.",
            )
        except (SecurityError, ValueError) as exc:
            return ToolCall(name, arguments, "", error=str(exc))
        except Exception as exc:  # surfaced back to the model so it can correct itself
            return ToolCall(name, arguments, "", error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- message plumbing
#
# Providers attach fields to tool calls that the conversation must carry back
# unchanged. Gemini 3.x returns a `thought_signature` on each functionCall and
# rejects the next request if it is missing. Rebuilding messages field by field
# silently drops anything not in the OpenAI schema, so everything is preserved
# by dumping the provider's own objects instead.


def _dump(obj) -> dict:
    """Full dict for an SDK model, including vendor-specific extra fields."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if v is not None}
    # Plain objects (and test doubles) expose their fields on __dict__.
    attributes = getattr(obj, "__dict__", None)
    if attributes is not None:
        return {k: v for k, v in attributes.items() if not k.startswith("_") and v is not None}
    return {}


def _assistant_message(message) -> dict:
    """The assistant turn as the provider produced it, safe to send back."""
    payload = _dump(message)
    payload["role"] = "assistant"
    # Fields that are outputs only; some providers reject them on input.
    for key in ("annotations", "audio", "function_call", "reasoning_content"):
        payload.pop(key, None)
    return payload


def _merge_fragment(pending: dict[int, dict], fragment) -> None:
    """Accumulate one streamed tool-call delta, preserving unknown keys."""
    dump = _dump(fragment)
    index = dump.pop("index", 0)
    slot = pending.setdefault(index, {"function": {"name": "", "arguments": ""}})

    function = dump.pop("function", None) or {}
    target = slot.setdefault("function", {"name": "", "arguments": ""})
    # Arguments arrive in pieces; every other field is sent once.
    target["arguments"] = (target.get("arguments") or "") + (function.get("arguments") or "")
    for key, value in function.items():
        if key != "arguments" and value not in (None, ""):
            target[key] = value

    for key, value in dump.items():
        if value not in (None, ""):
            slot[key] = value


def _finalise_slot(slot: dict) -> dict:
    call = dict(slot)
    call.setdefault("type", "function")
    function = dict(call.get("function") or {})
    function["arguments"] = function.get("arguments") or "{}"
    call["function"] = function
    return call


def _create_with_retry(client, on_wait=None, retries: int = 2, sleep=None, **kwargs):
    """Call the provider, waiting out rate limits rather than failing on them.

    Free tiers meter per minute and this loop spends one request per tool round,
    so a 429 mid-conversation is ordinary rather than exceptional. The provider
    states how long to wait; we honour it instead of guessing.
    """
    import time as _time

    from . import llm

    pause = sleep or _time.sleep
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            # A daily quota reports a short retryDelay too, but waiting cannot help.
            if attempt >= retries or not llm.is_retryable_rate_limit(exc):
                raise
            delay = llm.parse_retry_delay(exc)
            attempt += 1
            if on_wait:
                on_wait(delay, attempt, retries)
            pause(delay)


def ask(
    client: Any,
    model: str,
    question: str,
    toolbox: ToolBox,
    history: list[dict] | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
    on_tool: Callable[[ToolCall], None] | None = None,
    on_wait: Callable[[float, int, int], None] | None = None,
) -> AgentAnswer:
    """Run the tool loop against an OpenAI-compatible chat completions client."""
    schema_hint = json.dumps(toolbox.spec.to_prompt_json() | {"table": toolbox.view}, default=str)
    messages: list[dict] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nSCHEMA:\n{schema_hint}"},
        *(history or []),
        {"role": "user", "content": question},
    ]

    calls: list[ToolCall] = []
    for round_index in range(max_rounds):
        response = _create_with_retry(
            client, on_wait=on_wait,
            model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0,
        )
        message = response.choices[0].message
        if not getattr(message, "tool_calls", None):
            return AgentAnswer(message.content or "", calls, round_index + 1)

        # Echo the provider's own message back rather than rebuilding it. Some
        # providers attach fields the loop must return unchanged — Gemini 3.x
        # rejects a follow-up whose functionCall lost its thought_signature.
        messages.append(_assistant_message(message))
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            executed = toolbox.dispatch(call.function.name, arguments)
            calls.append(executed)
            if on_tool:
                on_tool(executed)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": executed.error and f"ERROR: {executed.error}" or executed.result_summary,
                }
            )

    return AgentAnswer(
        "I reached the tool-call limit before finishing. Here is what I established so far — "
        "try narrowing the question.",
        calls,
        max_rounds,
    )


def ask_stream(
    client: Any,
    model: str,
    question: str,
    toolbox: ToolBox,
    history: list[dict] | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
    on_wait: Callable[[float, int, int], None] | None = None,
):
    """Streaming variant of `ask`, yielding events as they happen.

    Yields ("tool", ToolCall) as each tool finishes and ("text", chunk) as the
    final answer arrives token by token, then ("done", AgentAnswer). Tool calls
    are reassembled from streamed deltas, so the loop still works while the
    response is being streamed rather than waiting for a complete message.
    """
    schema_hint = json.dumps(toolbox.spec.to_prompt_json() | {"table": toolbox.view}, default=str)
    messages: list[dict] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nSCHEMA:\n{schema_hint}"},
        *(history or []),
        {"role": "user", "content": question},
    ]
    calls: list[ToolCall] = []

    for round_index in range(max_rounds):
        stream = _create_with_retry(
            client, on_wait=on_wait,
            model=model, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", temperature=0, stream=True,
        )

        text_parts: list[str] = []
        # Tool calls stream as fragments keyed by index; reassemble them here.
        pending: dict[int, dict] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
                yield "text", delta.content
            for fragment in getattr(delta, "tool_calls", None) or []:
                _merge_fragment(pending, fragment)

        if not pending:
            answer = AgentAnswer("".join(text_parts), calls, round_index + 1)
            yield "done", answer
            return

        assembled = [_finalise_slot(slot) for slot in pending.values()]
        messages.append({
            "role": "assistant",
            "content": "".join(text_parts) or None,
            "tool_calls": assembled,
        })

        for call in assembled:
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            executed = toolbox.dispatch(function.get("name", ""), arguments)
            calls.append(executed)
            yield "tool", executed
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": f"ERROR: {executed.error}" if executed.error else executed.result_summary,
            })

    yield "done", AgentAnswer(
        "I reached the tool-call limit before finishing. Try narrowing the question.", calls, max_rounds
    )
