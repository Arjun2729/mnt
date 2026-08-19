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

MAX_TOOL_ROUNDS = 8
MAX_ROWS_RETURNED = 200

SYSTEM_PROMPT = """You are a careful data analyst working over a SQL table in DuckDB.

Rules:
- Answer by querying. Never estimate a number you could compute with run_sql.
- The table you may query is named in the schema. Use exact column names.
- DuckDB SQL dialect. Read-only SELECT statements only.
- If a question is ambiguous, state the interpretation you chose, then answer it.
- Distinguish what the data shows from what you infer. Do not claim causation from correlation.
- If the data cannot answer the question, say so plainly and say what would be needed.
- Prefer a few well-chosen aggregate queries over many small ones.
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
        except (SecurityError, ValueError) as exc:
            return ToolCall(name, arguments, "", error=str(exc))
        except Exception as exc:  # surfaced back to the model so it can correct itself
            return ToolCall(name, arguments, "", error=f"{type(exc).__name__}: {exc}")


def ask(
    client: Any,
    model: str,
    question: str,
    toolbox: ToolBox,
    history: list[dict] | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
    on_tool: Callable[[ToolCall], None] | None = None,
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
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0
        )
        message = response.choices[0].message
        if not getattr(message, "tool_calls", None):
            return AgentAnswer(message.content or "", calls, round_index + 1)

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in message.tool_calls
                ],
            }
        )
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
        stream = client.chat.completions.create(
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
                slot = pending.setdefault(
                    fragment.index, {"id": "", "name": "", "arguments": ""}
                )
                if fragment.id:
                    slot["id"] = fragment.id
                if fragment.function and fragment.function.name:
                    slot["name"] = fragment.function.name
                if fragment.function and fragment.function.arguments:
                    slot["arguments"] += fragment.function.arguments

        if not pending:
            answer = AgentAnswer("".join(text_parts), calls, round_index + 1)
            yield "done", answer
            return

        messages.append({
            "role": "assistant",
            "content": "".join(text_parts) or None,
            "tool_calls": [
                {"id": slot["id"], "type": "function",
                 "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"}}
                for slot in pending.values()
            ],
        })

        for slot in pending.values():
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            executed = toolbox.dispatch(slot["name"], arguments)
            calls.append(executed)
            yield "tool", executed
            messages.append({
                "role": "tool",
                "tool_call_id": slot["id"],
                "content": f"ERROR: {executed.error}" if executed.error else executed.result_summary,
            })

    yield "done", AgentAnswer(
        "I reached the tool-call limit before finishing. Try narrowing the question.", calls, max_rounds
    )
