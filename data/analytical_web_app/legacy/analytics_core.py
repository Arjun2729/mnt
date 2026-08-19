from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sqlalchemy import create_engine, text


SUPPORTED_UPLOAD_TYPES = ["csv", "xlsx", "xls", "json", "parquet"]


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load a Streamlit UploadedFile into a pandas DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    if name.endswith(".json"):
        raw = uploaded_file.getvalue()
        try:
            return pd.read_json(io.BytesIO(raw))
        except ValueError:
            return pd.read_json(io.BytesIO(raw), lines=True)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def load_database(connection_url: str, query: str) -> pd.DataFrame:
    """Run a read-only SQL query through SQLAlchemy."""
    cleaned = query.strip().lstrip("(")
    if not cleaned.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT/CTE queries are allowed in this MVP.")
    engine = create_engine(connection_url, pool_pre_ping=True)
    with engine.connect() as connection:
        return pd.read_sql_query(text(query), connection)


def _extract_json_path(payload: Any, path: str) -> Any:
    current = payload
    if not path.strip():
        return current
    for part in path.split("."):
        part = part.strip()
        if not part:
            continue
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(f"Cannot descend into {part!r}")
    return current


def load_api(url: str, headers: dict[str, str] | None = None, json_path: str = "") -> pd.DataFrame:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("API URL must use http:// or https://")
    response = requests.get(url, headers=headers or {}, timeout=20)
    response.raise_for_status()
    payload = _extract_json_path(response.json(), json_path)
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        # A dict of equal-length arrays is naturally tabular; otherwise one record.
        try:
            return pd.DataFrame(payload)
        except ValueError:
            return pd.json_normalize([payload])
    return pd.DataFrame({"value": [payload]})


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Conservatively detect date-like object columns without mutating ambiguous text."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        sample = out[col].dropna().astype(str).head(100)
        if sample.empty:
            continue
        date_hint = any(token in col.lower() for token in ("date", "time", "timestamp", "created", "updated"))
        if not date_hint:
            continue
        converted = pd.to_datetime(out[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            out[col] = converted
    return out


def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_%": round(float(s.isna().mean() * 100), 2),
                "unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    return numeric.corr(numeric_only=True) if numeric.shape[1] >= 2 else pd.DataFrame()


@dataclass
class AssociationResult:
    method: str
    statistic: float
    p_value: float
    n: int


def numeric_association(df: pd.DataFrame, x: str, y: str, method: str) -> AssociationResult:
    pair = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        raise ValueError("Need at least 3 complete observations.")
    if method == "Pearson":
        statistic, p_value = stats.pearsonr(pair[x], pair[y])
    elif method == "Spearman":
        statistic, p_value = stats.spearmanr(pair[x], pair[y])
    else:
        raise ValueError(f"Unsupported association method: {method}")
    return AssociationResult(method, float(statistic), float(p_value), len(pair))


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered Data")
        dataset_profile(df).to_excel(writer, index=False, sheet_name="Profile")
        corr = correlation_table(df)
        if not corr.empty:
            corr.to_excel(writer, sheet_name="Correlation")
    return buffer.getvalue()


def build_html_report(df: pd.DataFrame, title: str = "Analytical Report") -> str:
    profile = dataset_profile(df)
    numeric = df.describe(include=[np.number]).T if not df.select_dtypes(include=np.number).empty else pd.DataFrame()
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 18px;color:#17202a}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ddd;padding:7px;text-align:left}th{background:#f4f6f7}h1,h2{margin-top:28px}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p><strong>Rows:</strong> {len(df):,} &nbsp; <strong>Columns:</strong> {df.shape[1]:,} &nbsp; <strong>Missing cells:</strong> {int(df.isna().sum().sum()):,}</p>",
        "<h2>Column profile</h2>",
        profile.to_html(index=False, escape=True),
    ]
    if not numeric.empty:
        html += ["<h2>Numeric summary</h2>", numeric.to_html(escape=True)]
    html += ["<h2>Preview</h2>", df.head(100).to_html(index=False, escape=True), "</body></html>"]
    return "".join(html)


def build_ai_context(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Create a bounded textual context for natural-language analysis."""
    profile = dataset_profile(df)
    numeric = df.describe(include=[np.number]).round(4).to_dict() if not df.select_dtypes(include=np.number).empty else {}
    categorical = {}
    for col in df.select_dtypes(exclude=np.number).columns[:20]:
        categorical[col] = df[col].astype(str).value_counts(dropna=False).head(10).to_dict()
    sample = df.head(max_rows).replace({np.nan: None}).to_dict(orient="records")
    payload = {
        "shape": {"rows": len(df), "columns": df.shape[1]},
        "column_profile": profile.to_dict(orient="records"),
        "numeric_summary": numeric,
        "top_categorical_values": categorical,
        "sample_rows": sample,
    }
    return json.dumps(payload, default=str, ensure_ascii=False)
