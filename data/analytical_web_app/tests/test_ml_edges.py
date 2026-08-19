"""Modelling paths reached only by awkward data."""
import numpy as np
import pandas as pd
import pytest

from groundtruth import ml


def test_no_features_is_refused(frame):
    with pytest.raises(ValueError, match="at least one feature"):
        ml.train(frame, "revenue", [])


def test_identifier_feature_is_flagged_as_leakage():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "row_key": [f"k{i}" for i in range(80)],
        "x": rng.normal(size=80),
        "target": rng.normal(size=80),
    })
    result = ml.train(frame, "target", ["row_key", "x"])
    assert any("row_key" in w and "identifier" in w for w in result.leakage_warnings)


def test_categorical_relabelling_is_flagged():
    labels = ["up", "down"] * 40
    frame = pd.DataFrame({
        "alias": ["U" if v == "up" else "D" for v in labels],
        "noise": np.random.default_rng(1).normal(size=80),
        "target": labels,
    })
    result = ml.train(frame, "target", ["alias", "noise"])
    assert any("alias" in w for w in result.leakage_warnings)


def test_singleton_classes_are_dropped_with_a_note():
    rng = np.random.default_rng(2)
    labels = ["a"] * 40 + ["b"] * 40 + ["rare"]
    frame = pd.DataFrame({"x": rng.normal(size=81), "target": labels})
    result = ml.train(frame, "target", ["x"])
    assert any("single example" in note for note in result.notes)


def test_single_class_target_is_refused():
    frame = pd.DataFrame({"x": range(60), "target": ["only"] * 60})
    with pytest.raises(ValueError, match="two classes"):
        ml.train(frame, "target", ["x"])


def test_importance_can_be_skipped(frame):
    result = ml.train(frame, "revenue", ["cost", "orders"], compute_importance=False)
    assert result.importance.empty
    assert list(result.importance.columns) == ["feature", "importance", "std", "informative"]


def test_integer_label_column_reads_as_classification():
    frame = pd.DataFrame({"x": range(200), "target": [1, 2, 3, 4] * 50})
    assert ml.infer_problem_type(frame["target"]) == ml.CLASSIFICATION


def test_high_cardinality_integer_reads_as_regression():
    assert ml.infer_problem_type(pd.Series(range(500))) == ml.REGRESSION


def test_regression_reports_residuals(frame):
    result = ml.train(frame, "revenue", ["cost", "orders"])
    assert "residual" in result.predictions.columns


def test_binary_classification_reports_auc():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    frame = pd.DataFrame({"x": x, "target": np.where(x + rng.normal(scale=0.4, size=200) > 0, "yes", "no")})
    result = ml.train(frame, "target", ["x"])
    assert "roc_auc" in result.metrics


def test_a_failing_candidate_is_noted_not_fatal(frame, monkeypatch):
    """One model family failing must not abort the leaderboard."""
    real = ml.cross_val_score
    calls = {"n": 0}

    def flaky(pipeline, X, y, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:      # the first candidate after the baseline
            raise RuntimeError("simulated fit failure")
        return real(pipeline, X, y, **kwargs)

    monkeypatch.setattr(ml, "cross_val_score", flaky)
    result = ml.train(frame, "revenue", ["cost", "orders"])
    assert any("failed to fit" in note for note in result.notes)
    assert result.leaderboard, "the surviving candidates must still be ranked"


def test_leaderboard_frame_shape(frame):
    result = ml.train(frame, "revenue", ["cost", "orders"])
    table = ml.leaderboard_frame(result)
    assert set(table.columns) == {"model", f"cv_{result.scoring}", "std", "beats_baseline"}


def test_relabelling_check_does_not_fire_on_independent_columns():
    """Regression guard: the check previously never ran at all because it tested
    `dtype == object`, which modern pandas no longer uses for strings."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "region": rng.choice(list("NSEW"), 200),
        "target": rng.choice(["up", "down"], 200),
    })
    assert ml.detect_leakage(frame, "target", ["region"], ml.CLASSIFICATION) == []


def test_relabelling_check_fires_on_a_perfect_mapping():
    labels = ["up", "down"] * 60
    frame = pd.DataFrame({"alias": ["U" if v == "up" else "D" for v in labels], "target": labels})
    warnings_found = ml.detect_leakage(frame, "target", ["alias"], ml.CLASSIFICATION)
    assert any("alias" in w for w in warnings_found)
