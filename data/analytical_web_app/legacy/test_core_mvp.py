import pandas as pd

from analytics_core import build_ai_context, dataset_profile, numeric_association
from ml_core import train_random_forest


def test_profile_and_association():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "g": ["a", "a", "b", "b"]})
    profile = dataset_profile(df)
    assert set(profile["column"]) == {"x", "y", "g"}
    result = numeric_association(df, "x", "y", "Pearson")
    assert result.statistic > 0.99


def test_ai_context_is_bounded():
    df = pd.DataFrame({"x": range(100), "label": ["a"] * 100})
    context = build_ai_context(df, max_rows=5)
    assert '"rows": 100' in context


def test_ml_regression():
    df = pd.DataFrame({
        "x": range(60),
        "cat": ["a", "b"] * 30,
        "target": [v * 2.0 for v in range(60)],
    })
    result = train_random_forest(df, "target", ["x", "cat"], "Regression", test_size=0.2)
    assert result.problem_type == "Regression"
    assert "mae" in result.metrics
    assert not result.feature_importance.empty


def test_shipped_sample_supports_every_tab():
    """The README tells users to start with sample_data.csv, so it must be
    large and varied enough for the ML tab (>=20 rows) and not degenerate."""
    from pathlib import Path

    from analytics_core import coerce_dates, correlation_table

    df = coerce_dates(pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data.csv"))
    assert len(df) >= 20
    assert correlation_table(df).shape[0] >= 2
    assert df.select_dtypes(exclude="number").shape[1] >= 1

    regression = train_random_forest(df, "revenue", ["region", "channel", "cost", "orders"], "Auto")
    assert regression.problem_type == "Regression"
    classification = train_random_forest(df, "channel", ["revenue", "cost", "orders"], "Auto")
    assert classification.problem_type == "Classification"
