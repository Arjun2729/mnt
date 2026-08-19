"""Sources land in the store; the JSON path walker reports its own failures."""
import json
import sqlite3

import pandas as pd
import pytest

from groundtruth import connectors
from groundtruth.security import SecurityError
from groundtruth.store import Store


def test_csv_lands_and_types_are_promoted(sample_csv):
    store = Store()
    result = connectors.load_path(store, str(sample_csv), "s")
    assert result.dataset.rows == 288
    assert "date" in result.coerced_columns


def test_parquet_is_scanned_in_place(tmp_path, frame):
    path = tmp_path / "d.parquet"
    frame.to_parquet(path)
    store = Store()
    result = connectors.load_path(store, str(path), "p")
    assert result.dataset.rows == len(frame)
    assert "DuckDB" in result.note


def test_database_query_lands(tmp_path, frame):
    db = tmp_path / "t.db"
    connection = sqlite3.connect(db)
    frame.to_sql("t", connection, index=False)
    connection.close()
    store = Store()
    result = connectors.load_database(store, f"sqlite:///{db}", "SELECT region, SUM(revenue) rev FROM t GROUP BY region")
    assert result.dataset.rows == 4


def test_database_rejects_writes(tmp_path, frame):
    db = tmp_path / "t.db"
    connection = sqlite3.connect(db)
    frame.to_sql("t", connection, index=False)
    connection.close()
    with pytest.raises(SecurityError):
        connectors.load_database(Store(), f"sqlite:///{db}", "DROP TABLE t")


def test_missing_file_is_reported():
    with pytest.raises(ValueError, match="not found"):
        connectors.load_path(Store(), "/nope/missing.csv")


def test_unsupported_extension_is_reported(tmp_path):
    path = tmp_path / "x.docx"
    path.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported"):
        connectors.load_path(Store(), str(path))


def test_empty_source_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n")
    with pytest.raises(ValueError, match="no rows"):
        connectors.load_path(Store(), str(path))


def test_duplicate_columns_are_disambiguated(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text("a,a,b\n1,2,3\n4,5,6\n")
    store = Store()
    result = connectors.load_path(store, str(path), "d")
    assert len(set(result.dataset.columns)) == 3


@pytest.mark.parametrize(
    "path,expected",
    [("data.items", [{"x": 1}]), ("data", {"items": [{"x": 1}]}), ("", {"data": {"items": [{"x": 1}]}})],
)
def test_json_path_walks(path, expected):
    payload = {"data": {"items": [{"x": 1}]}}
    assert connectors.extract_json_path(payload, path) == expected


def test_json_path_names_the_available_keys():
    with pytest.raises(ValueError, match="Available"):
        connectors.extract_json_path({"data": {"items": []}}, "data.missing")


def test_json_path_cannot_descend_into_a_scalar():
    with pytest.raises(ValueError, match="reached a"):
        connectors.extract_json_path({"a": 5}, "a.b")


def test_api_requires_an_allowlist():
    with pytest.raises(SecurityError, match="allowlist"):
        connectors.load_api(Store(), "https://example.com/x")
