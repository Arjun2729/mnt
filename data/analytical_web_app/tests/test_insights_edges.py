"""Insight kinds that the clean sample data never triggers."""
import numpy as np
import pandas as pd
import pytest

from groundtruth.insights import Insight, insights_frame, scan
from groundtruth.semantic import profile
from groundtruth.store import Store


@pytest.fixture
def blank():
    store = Store()
    yield store
    store.close()


def findings(store, frame, table="t"):
    store.register_frame(table, frame)
    return scan(store, table, profile(store, table))


def test_heavy_missingness_is_flagged(blank):
    frame = pd.DataFrame({
        "mostly_empty": [1.0] + [None] * 79,
        "value": np.random.default_rng(0).normal(size=80),
    })
    found = findings(blank, frame)
    assert any(f.kind == "quality" and "empty" in f.headline for f in found)
    assert any(f.severity == "warning" for f in found)


def test_constant_column_is_flagged(blank):
    frame = pd.DataFrame({"same": ["x"] * 60, "value": range(60)})
    assert any("single value" in f.headline for f in findings(blank, frame))


def test_duplicate_key_is_flagged(blank):
    """A column that looks like a key but repeats is worth saying out loud."""
    frame = pd.DataFrame({"order_id": [f"k{i}" for i in range(50)] + ["k0"], "v": range(51)})
    blank.register_frame("t", frame)
    spec = profile(blank, "t")
    spec.key_candidates = ["order_id"]      # repeats, so profile would not offer it
    assert any("Duplicate" in f.headline for f in scan(blank, "t", spec))


def test_segment_outlier_is_found(blank):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "region": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
        "revenue": np.concatenate([rng.normal(100, 5, 40), rng.normal(105, 5, 40), rng.normal(400, 5, 40)]),
    })
    found = findings(blank, frame)
    assert any(f.kind == "segment" for f in found)


def test_outlier_detection(blank):
    values = list(np.random.default_rng(2).normal(100, 1, 200)) + [10_000.0] * 5
    assert any(f.kind == "outlier" for f in findings(blank, pd.DataFrame({"v": values})))


def test_trend_is_detected(blank):
    frame = pd.DataFrame({
        "when": pd.date_range("2022-01-01", periods=36, freq="MS"),
        "revenue": np.linspace(100, 500, 36),
    })
    assert any(f.kind == "trend" for f in findings(blank, frame))


def test_no_measures_yields_no_correlations(blank):
    frame = pd.DataFrame({"a": list("xyz") * 10, "b": list("pqr") * 10})
    assert all(f.kind != "correlation" for f in findings(blank, frame))


def test_scan_respects_its_cap(blank, store, spec):
    assert len(scan(store, "sample", spec, max_insights=2)) <= 2


def test_findings_render_as_a_table():
    found = [Insight("trend", "head", "detail", "notable")]
    table = insights_frame(found)
    assert list(table.columns) == ["severity", "kind", "finding", "detail"]
    assert table.iloc[0]["finding"] == "head"


def test_empty_findings_render():
    assert insights_frame([]).empty


def test_segment_detection_is_not_fooled_by_the_outlier_it_seeks():
    """With few groups a big outlier inflates the standard deviation enough to
    hide itself, so the scale has to be robust."""
    rng = np.random.default_rng(1)
    store = Store()
    frame = pd.DataFrame({
        "region": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
        "revenue": np.concatenate([rng.normal(100, 5, 40), rng.normal(105, 5, 40), rng.normal(400, 5, 40)]),
    })
    store.register_frame("t", frame)
    segments = [f for f in scan(store, "t", profile(store, "t")) if f.kind == "segment"]
    assert segments and "c" in segments[0].headline
    store.close()


def test_similar_groups_produce_no_segment_finding():
    """A robust scale makes small spreads score high, so a magnitude floor is needed."""
    rng = np.random.default_rng(1)
    store = Store()
    frame = pd.DataFrame({
        "region": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
        "revenue": rng.normal(100, 5, 120),
    })
    store.register_frame("t", frame)
    assert not [f for f in scan(store, "t", profile(store, "t")) if f.kind == "segment"]
    store.close()
