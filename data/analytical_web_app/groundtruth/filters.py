"""L3 — the filter tree.

Filters are a nested tree of AND/OR groups, compiled to a parameterised SQL
WHERE clause. Because the compiler emits SQL rather than chained DataFrame
masks, arbitrary nesting is free and the same filter means the same thing to the
grid, the charts, the models and the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .store import quote_ident

Combinator = Literal["AND", "OR"]

# operator -> (sql template, number of bound values)
OPERATORS: dict[str, tuple[str, int]] = {
    "=": ("{col} = ?", 1),
    "!=": ("{col} IS DISTINCT FROM ?", 1),
    ">": ("{col} > ?", 1),
    ">=": ("{col} >= ?", 1),
    "<": ("{col} < ?", 1),
    "<=": ("{col} <= ?", 1),
    "between": ("{col} BETWEEN ? AND ?", 2),
    "in": ("__IN__", -1),
    "not in": ("__NOT_IN__", -1),
    "contains": ("CAST({col} AS VARCHAR) ILIKE ?", 1),
    "starts with": ("CAST({col} AS VARCHAR) ILIKE ?", 1),
    "ends with": ("CAST({col} AS VARCHAR) ILIKE ?", 1),
    "is null": ("{col} IS NULL", 0),
    "is not null": ("{col} IS NOT NULL", 0),
}


@dataclass
class Condition:
    column: str
    operator: str
    value: Any = None

    def compile(self) -> tuple[str, list[Any]]:
        if self.operator not in OPERATORS:
            raise ValueError(f"Unknown operator: {self.operator}")
        col = quote_ident(self.column)
        template, arity = OPERATORS[self.operator]

        if self.operator in ("in", "not in"):
            values = list(self.value or [])
            if not values:
                # An empty set matches nothing; say so explicitly rather than emitting invalid SQL.
                return ("FALSE" if self.operator == "in" else "TRUE"), []
            placeholders = ", ".join("?" for _ in values)
            negate = "NOT " if self.operator == "not in" else ""
            return f"{col} {negate}IN ({placeholders})", values

        if arity == 0:
            return template.format(col=col), []
        if arity == 2:
            lo, hi = self.value
            return template.format(col=col), [lo, hi]

        value = self.value
        if self.operator == "contains":
            value = f"%{value}%"
        elif self.operator == "starts with":
            value = f"{value}%"
        elif self.operator == "ends with":
            value = f"%{value}"
        return template.format(col=col), [value]


@dataclass
class Group:
    combinator: Combinator = "AND"
    children: list["Group | Condition"] = field(default_factory=list)
    negate: bool = False

    def compile(self) -> tuple[str, list[Any]]:
        if not self.children:
            return "", []
        parts: list[str] = []
        params: list[Any] = []
        for child in self.children:
            sql, child_params = child.compile()
            if not sql:
                continue
            parts.append(f"({sql})")
            params.extend(child_params)
        if not parts:
            return "", []
        joined = f" {self.combinator} ".join(parts)
        if self.negate:
            joined = f"NOT ({joined})"
        return joined, params


def compile_tree(node: Group | Condition | None) -> tuple[str, list[Any]]:
    """Compile a filter tree into (where_clause, params). Empty tree -> no clause."""
    if node is None:
        return "", []
    return node.compile()


# ---------- serialisation, so filter sets can be saved and replayed ----------


def to_dict(node: Group | Condition) -> dict:
    if isinstance(node, Condition):
        return {"kind": "condition", "column": node.column, "operator": node.operator, "value": node.value}
    return {
        "kind": "group",
        "combinator": node.combinator,
        "negate": node.negate,
        "children": [to_dict(c) for c in node.children],
    }


def from_dict(payload: dict) -> Group | Condition:
    if payload.get("kind") == "condition":
        return Condition(payload["column"], payload["operator"], payload.get("value"))
    return Group(
        combinator=payload.get("combinator", "AND"),
        negate=bool(payload.get("negate", False)),
        children=[from_dict(c) for c in payload.get("children", [])],
    )


def describe(node: Group | Condition, depth: int = 0) -> str:
    """Human-readable rendering of the tree, for the provenance log and reports."""
    pad = "  " * depth
    if isinstance(node, Condition):
        if node.operator in ("is null", "is not null"):
            return f"{pad}{node.column} {node.operator}"
        return f"{pad}{node.column} {node.operator} {node.value!r}"
    if not node.children:
        return f"{pad}(no filters)"
    head = f"{pad}{'NOT ' if node.negate else ''}{node.combinator}"
    body = "\n".join(describe(c, depth + 1) for c in node.children)
    return f"{head}\n{body}"


def describe_inline(node: Group | Condition | None) -> str:
    """One-line rendering for the header summary.

    `describe` indents the tree for the lineage log; this collapses it to
    something readable in a single line, dropping the scaffolding of groups that
    hold only one child.
    """
    if node is None:
        return "no filters"
    if isinstance(node, Condition):
        if node.operator in ("is null", "is not null"):
            return f"{node.column} {node.operator}"
        if node.operator == "in" and isinstance(node.value, (list, tuple)):
            shown = ", ".join(map(str, node.value[:4]))
            more = f" +{len(node.value) - 4}" if len(node.value) > 4 else ""
            return f"{node.column} in {shown}{more}"
        if node.operator == "between" and isinstance(node.value, (list, tuple)):
            return f"{node.column} {node.value[0]}–{node.value[1]}"
        return f"{node.column} {node.operator} {node.value}"

    parts = [describe_inline(child) for child in node.children]
    parts = [p for p in parts if p and p != "no filters"]
    if not parts:
        return "no filters"
    if len(parts) == 1:
        return f"NOT ({parts[0]})" if node.negate else parts[0]
    joined = f" {node.combinator} ".join(f"({p})" if " AND " in p or " OR " in p else p for p in parts)
    return f"NOT ({joined})" if node.negate else joined
