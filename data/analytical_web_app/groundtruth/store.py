"""L1 — the DuckDB substrate.

Every dataset lands here as a table. Nothing above this layer holds its own copy
of the data; the UI, the agent, the statistics and the models all read through
this one engine, which is what lets a filter set mean the same thing everywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd

from .security import assert_read_only

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    """Quote an identifier for safe interpolation (DuckDB uses doubled quotes)."""
    return '"' + str(name).replace('"', '""') + '"'


def safe_table_name(name: str) -> str:
    cleaned = re.sub(r"\W+", "_", str(name)).strip("_").lower() or "dataset"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned[:60]


@dataclass
class Dataset:
    """A table registered in the store, plus how it got there."""

    name: str
    table: str
    source_kind: str
    source_detail: str
    rows: int
    columns: list[str] = field(default_factory=list)


class Store:
    """Owns the DuckDB connection and the dataset registry."""

    def __init__(self, path: str = ":memory:") -> None:
        self.con = duckdb.connect(path)
        self.con.execute("SET TimeZone='UTC'")
        self.datasets: dict[str, Dataset] = {}

    # ---------- registration ----------

    def register_frame(self, name: str, df: pd.DataFrame, source_kind: str = "frame", source_detail: str = "") -> Dataset:
        table = safe_table_name(name)
        # Column names have to survive as SQL identifiers.
        frame = df.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        self.con.register("_incoming", frame)
        self.con.execute(f"CREATE OR REPLACE TABLE {quote_ident(table)} AS SELECT * FROM _incoming")
        self.con.unregister("_incoming")
        dataset = Dataset(
            name=name,
            table=table,
            source_kind=source_kind,
            source_detail=source_detail,
            rows=int(self.con.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[0]),
            columns=list(frame.columns),
        )
        self.datasets[name] = dataset
        return dataset

    def register_file(self, name: str, path: str, reader_sql: str) -> Dataset:
        """Register a file DuckDB can scan natively, without loading it first."""
        table = safe_table_name(name)
        self.con.execute(f"CREATE OR REPLACE VIEW {quote_ident(table)} AS {reader_sql}")
        rows = int(self.con.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[0])
        columns = [d[0] for d in self.con.execute(f"SELECT * FROM {quote_ident(table)} LIMIT 0").description]
        dataset = Dataset(name, table, "file", path, rows, columns)
        self.datasets[name] = dataset
        return dataset

    def drop(self, name: str) -> None:
        dataset = self.datasets.pop(name, None)
        if dataset:
            self.con.execute(f"DROP TABLE IF EXISTS {quote_ident(dataset.table)}")
            self.con.execute(f"DROP VIEW IF EXISTS {quote_ident(dataset.table)}")

    # ---------- reading ----------

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run a trusted internal query."""
        return self.con.execute(query, params or []).fetch_df()

    def sql_readonly(self, query: str, params: list[Any] | None = None, limit: int | None = 5000) -> pd.DataFrame:
        """Run an untrusted query (agent- or user-authored) with a hard row cap."""
        checked = assert_read_only(query)
        if limit is not None:
            checked = f"SELECT * FROM ({checked}) AS _q LIMIT {int(limit)}"
        return self.con.execute(checked, params or []).fetch_df()

    def schema(self, table: str) -> pd.DataFrame:
        return self.sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        )

    def count(self, table: str, where: str = "", params: list[Any] | None = None) -> int:
        clause = f" WHERE {where}" if where else ""
        return int(self.con.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}{clause}", params or []).fetchone()[0])

    def page(
        self,
        table: str,
        where: str = "",
        params: list[Any] | None = None,
        offset: int = 0,
        limit: int = 200,
        order_by: str = "",
    ) -> pd.DataFrame:
        clause = f" WHERE {where}" if where else ""
        order = f" ORDER BY {order_by}" if order_by else ""
        return self.sql(
            f"SELECT * FROM {quote_ident(table)}{clause}{order} LIMIT {int(limit)} OFFSET {int(offset)}",
            params or [],
        )

    def materialize(self, table: str, where: str = "", params: list[Any] | None = None, limit: int | None = None) -> pd.DataFrame:
        """Pull a filtered slice into pandas — for modelling, which needs real arrays."""
        clause = f" WHERE {where}" if where else ""
        cap = f" LIMIT {int(limit)}" if limit else ""
        return self.sql(f"SELECT * FROM {quote_ident(table)}{clause}{cap}", params or [])

    def create_filtered_view(self, view: str, table: str, where: str = "", params: list[Any] | None = None) -> str:
        """Expose the current filtered slice as a view the agent can query by name."""
        clause = f" WHERE {where}" if where else ""
        if params:
            # Views cannot carry bind parameters, so inline the already-validated literals.
            resolved = self.con.execute(
                f"SELECT * FROM {quote_ident(table)}{clause} LIMIT 0", params
            )  # validates types before we materialize
            del resolved
            self.con.execute(f"CREATE OR REPLACE TABLE {quote_ident(view)} AS SELECT * FROM {quote_ident(table)}{clause}", params)
        else:
            self.con.execute(f"CREATE OR REPLACE VIEW {quote_ident(view)} AS SELECT * FROM {quote_ident(table)}")
        return view

    def close(self) -> None:
        self.con.close()
