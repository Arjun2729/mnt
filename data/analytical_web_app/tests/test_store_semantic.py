"""The engine and the semantic layer above it."""
import pytest

from groundtruth.security import SecurityError
from groundtruth.semantic import ROLE_DIMENSION, ROLE_MEASURE, ROLE_TIME, profile


def test_dataset_registers(store):
    assert store.count("sample") == 288
    assert "revenue" in store.datasets["sample"].columns


def test_text_dates_are_promoted(spec):
    assert spec.time_column == "date"
    assert spec.column("date").role == ROLE_TIME


def test_grain_is_detected(spec):
    assert spec.time_grain == "month"


def test_roles_separate_measures_from_dimensions(spec):
    assert set(spec.measures) == {"revenue", "cost", "orders", "conversion_rate"}
    assert set(spec.dimensions) == {"region", "channel"}
    assert spec.column("region").role == ROLE_DIMENSION
    assert spec.column("revenue").role == ROLE_MEASURE


def test_unique_floats_are_not_treated_as_keys(spec):
    # revenue and cost happen to be unique, but they are quantities, not identifiers.
    assert "revenue" not in spec.key_candidates
    assert "cost" not in spec.key_candidates


def test_missingness_is_measured(spec):
    assert spec.column("conversion_rate").missing == 5


def test_paging_does_not_load_everything(store):
    page = store.page("sample", limit=10, offset=20)
    assert len(page) == 10


def test_readonly_query_is_capped(store):
    assert len(store.sql_readonly("SELECT * FROM sample", limit=5)) == 5


def test_readonly_query_rejects_writes(store):
    with pytest.raises(SecurityError):
        store.sql_readonly("DROP TABLE sample")


def test_prompt_json_carries_no_row_data(spec):
    payload = spec.to_prompt_json()
    assert "columns" in payload and payload["rows"] == 288
    assert "sample_rows" not in payload
