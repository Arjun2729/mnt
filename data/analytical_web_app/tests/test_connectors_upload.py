"""The upload path — how most people actually load data, and previously untested."""
import io
import json

import pandas as pd
import pytest

from groundtruth import connectors
from groundtruth.store import Store


class FakeUpload:
    """Stands in for a Streamlit UploadedFile: a name plus getvalue()."""

    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture
def store():
    s = Store()
    yield s
    s.close()


def test_csv_upload(store, frame):
    upload = FakeUpload("sales.csv", frame.to_csv(index=False).encode())
    result = connectors.load_upload(store, upload)
    assert result.dataset.rows == len(frame)
    assert result.dataset.name == "sales"


def test_excel_upload(store, frame):
    buffer = io.BytesIO()
    frame.head(20).to_excel(buffer, index=False)
    result = connectors.load_upload(store, FakeUpload("book.xlsx", buffer.getvalue()))
    assert result.dataset.rows == 20


def test_json_array_upload(store):
    payload = json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).encode()
    result = connectors.load_upload(store, FakeUpload("d.json", payload))
    assert result.dataset.rows == 2
    assert set(result.dataset.columns) == {"a", "b"}


def test_json_single_object_upload(store):
    result = connectors.load_upload(store, FakeUpload("one.json", json.dumps({"a": 1}).encode()))
    assert result.dataset.rows == 1


def test_nested_json_is_flattened(store):
    payload = json.dumps([{"id": 1, "user": {"name": "ana", "city": "NY"}}]).encode()
    result = connectors.load_upload(store, FakeUpload("nested.json", payload))
    assert "user.name" in result.dataset.columns


def test_jsonl_upload(store):
    payload = b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'
    result = connectors.load_upload(store, FakeUpload("d.jsonl", payload))
    assert result.dataset.rows == 3


def test_malformed_json_falls_back_to_lines(store):
    """A .json file that is really JSONL should still load."""
    result = connectors.load_upload(store, FakeUpload("odd.json", b'{"a": 1}\n{"a": 2}\n'))
    assert result.dataset.rows == 2


def test_tsv_upload(store):
    result = connectors.load_upload(store, FakeUpload("d.tsv", b"a\tb\n1\tx\n2\ty\n"))
    assert result.dataset.rows == 2
    assert set(result.dataset.columns) == {"a", "b"}


def test_parquet_upload_is_scanned_in_place(store, frame, tmp_path):
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    result = connectors.load_upload(store, FakeUpload("d.parquet", buffer.getvalue()))
    assert result.dataset.rows == len(frame)
    assert "DuckDB" in result.note


def test_upload_promotes_date_columns(store, frame):
    upload = FakeUpload("sales.csv", frame.to_csv(index=False).encode())
    assert "date" in connectors.load_upload(store, upload).coerced_columns


def test_unsupported_upload_is_refused(store):
    with pytest.raises(ValueError, match="Unsupported"):
        connectors.load_upload(store, FakeUpload("notes.docx", b"nope"))


def test_empty_upload_is_refused(store):
    with pytest.raises(ValueError, match="no rows"):
        connectors.load_upload(store, FakeUpload("empty.csv", b"a,b\n"))


def test_explicit_name_overrides_the_filename(store, frame):
    upload = FakeUpload("ugly_name_v2_FINAL.csv", frame.to_csv(index=False).encode())
    assert connectors.load_upload(store, upload, name="Q3 sales").dataset.name == "Q3 sales"


def test_duplicate_columns_survive_upload(store):
    result = connectors.load_upload(store, FakeUpload("d.csv", b"a,a,b\n1,2,3\n"))
    assert len(set(result.dataset.columns)) == 3
