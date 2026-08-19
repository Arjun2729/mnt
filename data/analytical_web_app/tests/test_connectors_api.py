"""The API connector, with the network stubbed at the requests boundary."""
import json

import pandas as pd
import pytest

from groundtruth import connectors, security
from groundtruth.store import Store


@pytest.fixture
def allowed(monkeypatch):
    """Allowlist a host and treat it as public."""
    monkeypatch.setattr(security, "_resolves_to_private_address", lambda host: False)
    return ["api.test"]


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


@pytest.fixture
def fake_get(monkeypatch):
    captured = {}

    def install(payload, status=200):
        def get(url, headers=None, timeout=None):
            captured.update({"url": url, "headers": headers, "timeout": timeout})
            return _Response(payload, status)

        monkeypatch.setattr(connectors.requests, "get", get)
        return captured

    return install


@pytest.fixture
def store():
    s = Store()
    yield s
    s.close()


def test_list_payload(store, allowed, fake_get):
    fake_get([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    result = connectors.load_api(store, "https://api.test/rows", allowed_hosts=allowed)
    assert result.dataset.rows == 2


def test_dot_path_extraction(store, allowed, fake_get):
    fake_get({"data": {"items": [{"a": 1}, {"a": 2}, {"a": 3}]}})
    result = connectors.load_api(store, "https://api.test/x", json_path="data.items", allowed_hosts=allowed)
    assert result.dataset.rows == 3


def test_columnar_dict_payload(store, allowed, fake_get):
    fake_get({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert connectors.load_api(store, "https://api.test/x", allowed_hosts=allowed).dataset.rows == 3


def test_single_record_dict_payload(store, allowed, fake_get):
    fake_get({"a": 1, "nested": {"b": 2}})
    result = connectors.load_api(store, "https://api.test/x", allowed_hosts=allowed)
    assert result.dataset.rows == 1


def test_scalar_payload_becomes_one_value(store, allowed, fake_get):
    fake_get(42)
    result = connectors.load_api(store, "https://api.test/x", allowed_hosts=allowed)
    assert result.dataset.columns == ["value"]


def test_headers_are_forwarded(store, allowed, fake_get):
    captured = fake_get([{"a": 1}])
    connectors.load_api(store, "https://api.test/x", headers={"Authorization": "Bearer t"},
                        allowed_hosts=allowed)
    assert captured["headers"]["Authorization"] == "Bearer t"


def test_http_errors_propagate(store, allowed, fake_get):
    fake_get([{"a": 1}], status=503)
    with pytest.raises(RuntimeError, match="503"):
        connectors.load_api(store, "https://api.test/x", allowed_hosts=allowed)


def test_index_into_a_list(store):
    assert connectors.extract_json_path({"rows": [{"a": 1}, {"a": 2}]}, "rows.1") == {"a": 2}


def test_bad_list_index_is_reported():
    with pytest.raises(ValueError, match="Cannot index a list"):
        connectors.extract_json_path({"rows": [{"a": 1}]}, "rows.nine")


def test_out_of_range_index_is_reported():
    with pytest.raises(ValueError, match="Cannot index a list"):
        connectors.extract_json_path({"rows": [{"a": 1}]}, "rows.7")


def test_empty_path_segments_are_skipped():
    assert connectors.extract_json_path({"a": {"b": 1}}, "a..b") == 1


def test_excel_from_disk(store, frame, tmp_path):
    path = tmp_path / "book.xlsx"
    frame.head(15).to_excel(path, index=False)
    assert connectors.load_path(store, str(path)).dataset.rows == 15


def test_json_from_disk(store, tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    assert connectors.load_path(store, str(path)).dataset.rows == 2


def test_jsonl_from_disk(store, tmp_path):
    path = tmp_path / "d.json"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    assert connectors.load_path(store, str(path)).dataset.rows == 3
