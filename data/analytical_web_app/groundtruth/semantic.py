"""L2 — the semantic layer.

Classifies every column by the role it plays in analysis rather than by its
storage type. This one artifact is what makes the layers above it smart: chart
suggestions, agent grounding, ML target validation and forecast eligibility all
read from here instead of re-deriving column roles on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from .store import Store, quote_ident

# Roles are about analytical intent, not dtype.
ROLE_TIME = "time"
ROLE_MEASURE = "measure"
ROLE_DIMENSION = "dimension"
ROLE_IDENTIFIER = "identifier"
ROLE_TEXT = "text"
ROLE_BOOLEAN = "boolean"

_TIME_TOKENS = ("date", "time", "timestamp", "created", "updated", "period", "month", "day", "year")
_ID_TOKENS = ("id", "uuid", "guid", "key", "code", "sku", "ref")


@dataclass
class ColumnSpec:
    name: str
    sql_type: str
    role: str
    distinct: int
    missing: int
    missing_pct: float
    min_value: Any = None
    max_value: Any = None
    mean_value: float | None = None
    top_values: list[tuple[str, int]] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.role == ROLE_MEASURE

    @property
    def is_categorical(self) -> bool:
        return self.role in (ROLE_DIMENSION, ROLE_BOOLEAN)


@dataclass
class DatasetSpec:
    table: str
    rows: int
    columns: list[ColumnSpec]
    time_column: str | None = None
    time_grain: str | None = None
    key_candidates: list[str] = field(default_factory=list)

    def by_role(self, *roles: str) -> list[str]:
        return [c.name for c in self.columns if c.role in roles]

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def measures(self) -> list[str]:
        return self.by_role(ROLE_MEASURE)

    @property
    def dimensions(self) -> list[str]:
        return self.by_role(ROLE_DIMENSION, ROLE_BOOLEAN)

    def to_prompt_json(self) -> dict:
        """Compact form handed to the agent — schema only, never row data."""
        return {
            "table": self.table,
            "rows": self.rows,
            "time_column": self.time_column,
            "time_grain": self.time_grain,
            "columns": [
                {
                    "name": c.name,
                    "type": c.sql_type,
                    "role": c.role,
                    "distinct": c.distinct,
                    "missing_pct": c.missing_pct,
                    **({"min": str(c.min_value), "max": str(c.max_value)} if c.min_value is not None else {}),
                    **({"examples": [v for v, _ in c.top_values[:5]]} if c.top_values else {}),
                }
                for c in self.columns
            ],
        }


_NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "DECIMAL", "REAL", "NUMERIC", "UBIGINT", "UINTEGER")
_TEMPORAL_TYPES = ("DATE", "TIMESTAMP", "TIME")


def coerce_types(store: Store, table: str) -> list[str]:
    """Promote text columns that are really dates or numbers. Returns changed columns."""
    schema = store.schema(table)
    changed: list[str] = []
    for _, row in schema.iterrows():
        name, sql_type = row["column_name"], str(row["data_type"]).upper()
        if not sql_type.startswith("VARCHAR"):
            continue
        col = quote_ident(name)
        total = store.sql(f"SELECT COUNT({col}) AS n FROM {quote_ident(table)}")["n"].iloc[0]
        if not total:
            continue

        looks_temporal = any(token in name.lower() for token in _TIME_TOKENS)
        if looks_temporal:
            ok = store.sql(
                f"SELECT COUNT(TRY_CAST({col} AS TIMESTAMP)) AS n FROM {quote_ident(table)}"
            )["n"].iloc[0]
            if ok / total >= 0.8:
                store.con.execute(
                    f"ALTER TABLE {quote_ident(table)} ALTER {col} TYPE TIMESTAMP USING TRY_CAST({col} AS TIMESTAMP)"
                )
                changed.append(name)
                continue

        ok = store.sql(f"SELECT COUNT(TRY_CAST({col} AS DOUBLE)) AS n FROM {quote_ident(table)}")["n"].iloc[0]
        if ok / total >= 0.95:
            store.con.execute(
                f"ALTER TABLE {quote_ident(table)} ALTER {col} TYPE DOUBLE USING TRY_CAST({col} AS DOUBLE)"
            )
            changed.append(name)
    return changed


def _infer_role(name: str, sql_type: str, distinct: int, rows: int) -> str:
    upper = sql_type.upper()
    lower = name.lower()

    if any(upper.startswith(t) for t in _TEMPORAL_TYPES):
        return ROLE_TIME
    if upper.startswith("BOOLEAN"):
        return ROLE_BOOLEAN

    if any(upper.startswith(t) for t in _NUMERIC_TYPES):
        # A numeric column that is unique per row is an identifier, not a measure.
        if rows and distinct >= rows * 0.98 and any(lower.endswith(t) or lower == t for t in _ID_TOKENS):
            return ROLE_IDENTIFIER
        # Low-cardinality integers behave as categories (a rating, a flag, a year).
        if distinct <= 2:
            return ROLE_BOOLEAN
        return ROLE_MEASURE

    # Text from here down.
    if rows and distinct >= rows * 0.9:
        return ROLE_IDENTIFIER if any(t in lower for t in _ID_TOKENS) else ROLE_TEXT
    if distinct <= max(50, rows * 0.05):
        return ROLE_DIMENSION
    return ROLE_TEXT


def _detect_grain(store: Store, table: str, column: str) -> str | None:
    col = quote_ident(column)
    gaps = store.sql(
        f"""
        WITH ordered AS (
            SELECT DISTINCT {col} AS ts FROM {quote_ident(table)} WHERE {col} IS NOT NULL
        ), deltas AS (
            SELECT date_diff('second', LAG(ts) OVER (ORDER BY ts), ts) AS d FROM ordered
        )
        SELECT median(d) AS med FROM deltas WHERE d IS NOT NULL
        """
    )
    if gaps.empty or pd.isna(gaps["med"].iloc[0]):
        return None
    seconds = float(gaps["med"].iloc[0])
    for label, size in (("hour", 3600), ("day", 86400), ("week", 604800), ("month", 2419200), ("quarter", 7776000), ("year", 30758400)):
        if seconds <= size * 1.35:
            return label
    return "year"


def profile(store: Store, table: str, top_n: int = 10) -> DatasetSpec:
    """Build the semantic spec for a registered table."""
    rows = store.count(table)
    schema = store.schema(table)
    columns: list[ColumnSpec] = []

    for _, row in schema.iterrows():
        name, sql_type = row["column_name"], str(row["data_type"])
        col = quote_ident(name)
        agg = store.sql(
            f"SELECT COUNT(DISTINCT {col}) AS distinct_n, COUNT(*) - COUNT({col}) AS missing FROM {quote_ident(table)}"
        )
        distinct = int(agg["distinct_n"].iloc[0])
        missing = int(agg["missing"].iloc[0])
        role = _infer_role(name, sql_type, distinct, rows)

        min_value = max_value = mean_value = None
        top_values: list[tuple[str, int]] = []

        if role in (ROLE_MEASURE, ROLE_TIME):
            stats = store.sql(f"SELECT MIN({col}) AS lo, MAX({col}) AS hi FROM {quote_ident(table)}")
            min_value, max_value = stats["lo"].iloc[0], stats["hi"].iloc[0]
            if role == ROLE_MEASURE:
                mean_value = float(store.sql(f"SELECT AVG({col}) AS m FROM {quote_ident(table)}")["m"].iloc[0] or 0.0)
        if role in (ROLE_DIMENSION, ROLE_BOOLEAN):
            counts = store.sql(
                f"SELECT CAST({col} AS VARCHAR) AS v, COUNT(*) AS n FROM {quote_ident(table)} "
                f"GROUP BY 1 ORDER BY n DESC LIMIT {int(top_n)}"
            )
            top_values = [(str(v), int(n)) for v, n in zip(counts["v"], counts["n"])]

        columns.append(
            ColumnSpec(
                name=name,
                sql_type=sql_type,
                role=role,
                distinct=distinct,
                missing=missing,
                missing_pct=round(missing / rows * 100, 2) if rows else 0.0,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
                top_values=top_values,
            )
        )

    time_cols = [c.name for c in columns if c.role == ROLE_TIME]
    time_column = time_cols[0] if time_cols else None
    grain = _detect_grain(store, table, time_column) if time_column else None
    # Continuous measures are unique by coincidence, not because they identify a row.
    keys = [
        c.name
        for c in columns
        if rows and c.distinct == rows and c.missing == 0 and c.role in (ROLE_IDENTIFIER, ROLE_DIMENSION, ROLE_TIME, ROLE_TEXT)
    ]

    return DatasetSpec(table=table, rows=rows, columns=columns, time_column=time_column, time_grain=grain, key_candidates=keys)


def profile_frame(spec: DatasetSpec) -> pd.DataFrame:
    """The spec as a table, for display."""
    return pd.DataFrame(
        [
            {
                "column": c.name,
                "role": c.role,
                "type": c.sql_type,
                "distinct": c.distinct,
                "missing": c.missing,
                "missing_%": c.missing_pct,
                "mean": round(c.mean_value, 3) if c.mean_value is not None else None,
            }
            for c in spec.columns
        ]
    )


def profile_with_distributions(store: Store, spec: DatasetSpec, bins: int = 16) -> pd.DataFrame:
    """The profile plus a per-column shape, for rendering as inline sparklines.

    Measures get a histogram, dimensions get their top category frequencies, and
    time columns get counts per period. Everything is normalised to 0-1 so the
    bars are comparable across rows of very different magnitudes.
    """
    rows = []
    for column in spec.columns:
        col = quote_ident(column.name)
        shape: list[float] = []

        try:
            if column.role == ROLE_MEASURE and column.distinct > 1:
                low, high = float(column.min_value), float(column.max_value)
                if high > low:
                    width = (high - low) / bins
                    counts = store.sql(
                        f"SELECT LEAST(CAST(({col} - {low}) / {width} AS INTEGER), {bins - 1}) AS b, "
                        f"COUNT(*) AS n FROM {quote_ident(spec.table)} "
                        f"WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY 1"
                    )
                    buckets = dict(zip(counts["b"].astype(int), counts["n"].astype(float)))
                    shape = [buckets.get(i, 0.0) for i in range(bins)]
            elif column.role in (ROLE_DIMENSION, ROLE_BOOLEAN) and column.top_values:
                shape = [float(n) for _, n in column.top_values]
            elif column.role == ROLE_TIME:
                counts = store.sql(
                    f"SELECT COUNT(*) AS n FROM {quote_ident(spec.table)} "
                    f"WHERE {col} IS NOT NULL GROUP BY date_trunc('{spec.time_grain or 'month'}', {col}) "
                    f"ORDER BY date_trunc('{spec.time_grain or 'month'}', {col})"
                )
                shape = [float(n) for n in counts["n"]][:60]
        except Exception:
            shape = []

        peak = max(shape) if shape else 0.0
        if peak:
            shape = [round(v / peak, 4) for v in shape]

        rows.append({
            "column": column.name,
            "role": column.role,
            "shape": shape,
            "type": column.sql_type,
            "distinct": column.distinct,
            "complete": round(1 - (column.missing / spec.rows if spec.rows else 0), 4),
            "mean": round(column.mean_value, 3) if column.mean_value is not None else None,
        })
    return pd.DataFrame(rows)
