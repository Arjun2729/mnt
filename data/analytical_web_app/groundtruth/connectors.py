"""L0 — sources.

One interface, three source families. Every connector lands its data in the
store as a registered table rather than returning a DataFrame, so nothing above
L1 ever holds its own private copy.
"""
from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sqlalchemy import create_engine, text

from .security import SecurityError, assert_read_only, validate_api_url
from .semantic import coerce_types
from .store import Dataset, Store

SUPPORTED_UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet"]


@dataclass
class LoadResult:
    dataset: Dataset
    coerced_columns: list[str]
    note: str = ""


def _finish(store: Store, name: str, frame: pd.DataFrame, kind: str, detail: str, note: str = "") -> LoadResult:
    if frame.empty:
        raise ValueError("The source returned no rows.")
    # Deduplicate column names, which Excel and JSON sources both produce.
    seen: dict[str, int] = {}
    columns: list[str] = []
    for column in frame.columns:
        base = str(column).strip() or "column"
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        columns.append(base)
    frame.columns = columns

    dataset = store.register_frame(name, frame, kind, detail)
    coerced = coerce_types(store, dataset.table)
    dataset.rows = store.count(dataset.table)
    return LoadResult(dataset, coerced, note)


# ---------- files ----------


def load_upload(store: Store, uploaded_file: Any, name: str | None = None) -> LoadResult:
    """Read a Streamlit UploadedFile (or any file-like with .name and .getvalue())."""
    filename = getattr(uploaded_file, "name", "upload")
    label = name or Path(filename).stem
    lower = filename.lower()
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else Path(filename).read_bytes()

    if lower.endswith(".parquet"):
        # Parquet goes through DuckDB directly — no pandas round-trip.
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
            handle.write(raw)
            path = handle.name
        dataset = store.register_file(label, path, f"SELECT * FROM read_parquet('{path}')")
        # Registered as a view, so no type coercion pass — Parquet already carries its types.
        return LoadResult(dataset, [], "Scanned in place by DuckDB.")

    if lower.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(io.BytesIO(raw))
    elif lower.endswith((".jsonl", ".ndjson")):
        frame = pd.read_json(io.BytesIO(raw), lines=True)
    elif lower.endswith(".json"):
        try:
            payload = json.loads(raw)
            frame = pd.json_normalize(payload) if isinstance(payload, list) else pd.json_normalize([payload])
        except (json.JSONDecodeError, ValueError):
            frame = pd.read_json(io.BytesIO(raw), lines=True)
    elif lower.endswith((".tsv", ".txt")):
        frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    elif lower.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(raw))
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    return _finish(store, label, frame, "upload", filename)


def load_path(store: Store, path: str, name: str | None = None) -> LoadResult:
    """Load a file from disk. CSV and Parquet are scanned by DuckDB natively."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")
    label = name or file_path.stem
    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        dataset = store.register_file(label, str(file_path), f"SELECT * FROM read_parquet('{file_path}')")
        return LoadResult(dataset, [], "Scanned in place by DuckDB.")
    if suffix in (".csv", ".tsv", ".txt"):
        frame = pd.read_csv(file_path, sep=None, engine="python")
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(file_path)
    elif suffix in (".json", ".jsonl", ".ndjson"):
        try:
            frame = pd.read_json(file_path)
        except ValueError:
            frame = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return _finish(store, label, frame, "file", str(file_path))


# ---------- databases ----------


def load_database(store: Store, connection_url: str, query: str, name: str = "query") -> LoadResult:
    """Run a validated read-only query through SQLAlchemy and land the result."""
    checked = assert_read_only(query)
    engine = create_engine(connection_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(text(checked), connection)
    finally:
        engine.dispose()
    scheme = connection_url.split("://", 1)[0]
    return _finish(
        store, name, frame, "database", f"{scheme} — {checked[:120]}",
        "Use a read-only database credential; the SQL guard is a second line of defence, not the first.",
    )


# ---------- APIs ----------


def extract_json_path(payload: Any, path: str) -> Any:
    current = payload
    if not path.strip():
        return current
    for part in path.split("."):
        part = part.strip()
        if not part:
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Cannot index a list with {part!r}") from exc
        elif isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Key {part!r} not found. Available: {', '.join(list(current)[:8])}")
            current = current[part]
        else:
            raise ValueError(f"Cannot descend into {part!r} — reached a {type(current).__name__}")
    return current


def load_api(
    store: Store,
    url: str,
    headers: dict[str, str] | None = None,
    json_path: str = "",
    name: str = "api",
    allowed_hosts: list[str] | None = None,
    timeout: int = 20,
) -> LoadResult:
    """Fetch JSON from an allowlisted host. Hosts must be permitted explicitly."""
    validate_api_url(url, allowed_hosts)
    response = requests.get(url, headers=headers or {}, timeout=timeout)
    response.raise_for_status()
    payload = extract_json_path(response.json(), json_path)

    if isinstance(payload, list):
        frame = pd.json_normalize(payload)
    elif isinstance(payload, dict):
        try:
            frame = pd.DataFrame(payload)
        except ValueError:
            frame = pd.json_normalize([payload])
    else:
        frame = pd.DataFrame({"value": [payload]})

    return _finish(store, name, frame, "api", url)
