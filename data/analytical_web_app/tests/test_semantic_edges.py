"""Role inference across the column shapes the sample data does not contain."""
import numpy as np
import pandas as pd
import pytest

from groundtruth.semantic import (
    ROLE_BOOLEAN,
    ROLE_IDENTIFIER,
    ROLE_MEASURE,
    ROLE_TEXT,
    ROLE_TIME,
    coerce_types,
    profile,
    profile_with_distributions,
)
from groundtruth.store import Store


@pytest.fixture
def blank():
    """An empty store — named so it does not shadow conftest's sample store."""
    store = Store()
    yield store
    store.close()


def role_of(store: Store, frame: pd.DataFrame, column: str) -> str:
    store.register_frame("t", frame)
    coerce_types(store, "t")
    return profile(store, "t").column(column).role


# ---------------- roles ----------------


def test_boolean_column(blank):
    assert role_of(blank, pd.DataFrame({"flag": [True, False] * 30}), "flag") == ROLE_BOOLEAN


def test_two_valued_numeric_reads_as_boolean(blank):
    assert role_of(blank, pd.DataFrame({"active": [0, 1] * 30}), "active") == ROLE_BOOLEAN


def test_unique_text_id_column(blank):
    frame = pd.DataFrame({"order_id": [f"id-{i}" for i in range(60)]})
    assert role_of(blank, frame, "order_id") == ROLE_IDENTIFIER


def test_unique_numeric_id_column(blank):
    frame = pd.DataFrame({"user_id": range(100), "v": np.random.default_rng(0).normal(size=100)})
    assert role_of(blank, frame, "user_id") == ROLE_IDENTIFIER


def test_free_text_column(blank):
    frame = pd.DataFrame({"comment": [f"a unique sentence number {i}" for i in range(80)]})
    assert role_of(blank, frame, "comment") == ROLE_TEXT


def test_role_predicates(spec):
    assert spec.column("revenue").is_numeric
    assert not spec.column("revenue").is_categorical
    assert spec.column("region").is_categorical
    assert not spec.column("region").is_numeric


# ---------------- type coercion ----------------


def test_numeric_strings_are_promoted(blank):
    blank.register_frame("t", pd.DataFrame({"amount": [f"{i}.5" for i in range(60)]}))
    assert "amount" in coerce_types(blank, "t")
    assert profile(blank, "t").column("amount").role == ROLE_MEASURE


def test_date_named_text_that_is_not_a_date_is_left_alone(blank):
    blank.register_frame("t", pd.DataFrame({"date_note": ["not a date at all"] * 40}))
    coerce_types(blank, "t")
    assert profile(blank, "t").column("date_note").role != ROLE_TIME


def test_all_null_column_is_skipped(blank):
    blank.register_frame("t", pd.DataFrame({"created_at": [None] * 20}))
    assert coerce_types(blank, "t") == []


# ---------------- time grain ----------------


@pytest.mark.parametrize("freq,periods,expected", [
    ("D", 60, "day"),
    ("h", 72, "hour"),
    ("W", 30, "week"),
    ("QS", 16, "quarter"),
    ("YS", 12, "year"),
])
def test_grain_detection(blank, freq, periods, expected):
    blank.register_frame("t", pd.DataFrame({"when": pd.date_range("2020-01-01", periods=periods, freq=freq)}))
    assert profile(blank, "t").time_grain == expected


def test_single_timestamp_has_no_grain(blank):
    blank.register_frame("t", pd.DataFrame({"when": pd.to_datetime(["2024-01-01"])}))
    assert profile(blank, "t").time_grain is None


# ---------------- distributions ----------------


def test_constant_column_gets_a_single_bar(blank):
    """One distinct value reads as boolean, so its shape is one full bar."""
    blank.register_frame("t", pd.DataFrame({"same": [1.0] * 40}))
    spec = profile(blank, "t")
    assert spec.column("same").distinct == 1
    shaped = profile_with_distributions(blank, spec).set_index("column")
    assert shaped.loc["same", "shape"] == [1.0]


def test_time_column_shape_counts_periods(blank):
    blank.register_frame("t", pd.DataFrame({"when": pd.date_range("2024-01-01", periods=24, freq="MS")}))
    spec = profile(blank, "t")
    shaped = profile_with_distributions(blank, spec).set_index("column")
    assert len(shaped.loc["when", "shape"]) == 24


def test_distribution_survives_a_failing_column(blank, monkeypatch):
    """One shape failing must not lose the whole profile."""
    blank.register_frame("t", pd.DataFrame({"a": np.arange(50.0)}))
    spec = profile(blank, "t")
    real_sql = blank.sql

    def flaky(query, params=None):
        if "LEAST" in query:
            raise RuntimeError("simulated failure")
        return real_sql(query, params)

    monkeypatch.setattr(blank, "sql", flaky)
    shaped = profile_with_distributions(blank, spec)
    assert list(shaped["shape"]) == [[]]
