"""Modelling: baselines, cross-validation, unbiased importance, leakage."""
import numpy as np
import pandas as pd
import pytest

from groundtruth import ml


@pytest.fixture
def signal_frame():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    return pd.DataFrame({
        "x": x,
        "noise": noise,
        "cat": rng.choice(["a", "b", "c"], n),
        "target": 3 * x + rng.normal(scale=0.4, size=n),
    })


def test_problem_type_inference():
    assert ml.infer_problem_type(pd.Series(np.linspace(0, 100, 200))) == ml.REGRESSION
    assert ml.infer_problem_type(pd.Series(["a", "b"] * 50)) == ml.CLASSIFICATION
    assert ml.infer_problem_type(pd.Series([0, 1] * 50)) == ml.CLASSIFICATION


def test_regression_beats_baseline(signal_frame):
    result = ml.train(signal_frame, "target", ["x", "noise", "cat"])
    assert result.problem_type == ml.REGRESSION
    assert result.metrics["r2"] > result.baseline_metrics["r2"]
    assert any(entry.beats_baseline for entry in result.leaderboard)


def test_leaderboard_ranks_several_models(signal_frame):
    result = ml.train(signal_frame, "target", ["x", "cat"])
    assert len(result.leaderboard) >= 2
    scores = [entry.cv_mean for entry in result.leaderboard]
    assert scores == sorted(scores, reverse=True)


def test_permutation_importance_ranks_signal_over_noise(signal_frame):
    result = ml.train(signal_frame, "target", ["x", "noise"])
    ranked = result.importance.set_index("feature")["importance"]
    assert ranked["x"] > ranked["noise"]


def test_pure_noise_does_not_beat_the_baseline():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150), "target": rng.normal(size=150)})
    result = ml.train(df, "target", ["a", "b"])
    assert result.metrics["r2"] < 0.3
    assert any("baseline" in note for note in result.notes) or result.metrics["r2"] <= result.baseline_metrics["r2"] + 0.2


def test_leakage_is_detected():
    rng = np.random.default_rng(2)
    target = rng.normal(size=120)
    df = pd.DataFrame({"copy_of_target": target * 2, "other": rng.normal(size=120), "target": target})
    result = ml.train(df, "target", ["copy_of_target", "other"])
    assert any("copy_of_target" in warning for warning in result.leakage_warnings)


def test_target_cannot_be_a_feature(signal_frame):
    with pytest.raises(ValueError, match="cannot also be a feature"):
        ml.train(signal_frame, "target", ["x", "target"])


def test_too_few_rows_is_refused():
    df = pd.DataFrame({"x": range(10), "target": range(10)})
    with pytest.raises(ValueError, match="at least 30 rows"):
        ml.train(df, "target", ["x"])


def test_classification_reports_class_metrics():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "label": np.where(x + rng.normal(scale=0.3, size=n) > 0, "up", "down")})
    result = ml.train(df, "label", ["x"])
    assert result.problem_type == ml.CLASSIFICATION
    assert "accuracy" in result.metrics
    assert result.metrics["accuracy"] > result.baseline_metrics["accuracy"]
