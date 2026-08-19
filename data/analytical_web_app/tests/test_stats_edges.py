"""Statistical branches the sample data does not reach."""
import numpy as np
import pandas as pd
import pytest

from groundtruth import stats


def test_spearman_path(frame):
    result = stats.correlation(frame, "revenue", "orders", method="spearman")
    assert result.name == "Spearman correlation"


def test_perfect_correlation_has_a_degenerate_interval():
    frame = pd.DataFrame({"a": range(20), "b": [v * 2 for v in range(20)]})
    result = stats.correlation(frame, "a", "b")
    assert result.ci == (result.statistic, result.statistic)


def test_unknown_correlation_method_falls_back_to_spearman(frame):
    assert stats.correlation(frame, "revenue", "cost", method="kendall").name == "Spearman correlation"


def test_chi_square_needs_two_categories_each():
    frame = pd.DataFrame({"a": ["x"] * 20, "b": ["y"] * 20})
    with pytest.raises(ValueError, match="at least two categories"):
        stats.chi_square(frame, "a", "b")


def test_chi_square_warns_on_sparse_cells():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.choice([f"a{i}" for i in range(12)], 30),
        "b": rng.choice([f"b{i}" for i in range(12)], 30),
    })
    assert stats.chi_square(frame, "a", "b").detail["warning"]


def test_group_comparison_needs_two_usable_groups():
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="at least two groups"):
        stats.compare_groups(frame, "v", "g")


def test_parametric_two_group_path_is_welch():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "v": np.concatenate([rng.normal(0, 1, 60), rng.normal(1, 1, 60)]),
        "g": ["a"] * 60 + ["b"] * 60,
    })
    assert stats.compare_groups(frame, "v", "g", parametric=True).name == "Welch's t-test"


def test_nonparametric_two_group_path_is_mann_whitney():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({
        "v": np.concatenate([rng.exponential(1, 60), rng.exponential(2, 60)]),
        "g": ["a"] * 60 + ["b"] * 60,
    })
    assert stats.compare_groups(frame, "v", "g", parametric=False).name == "Mann-Whitney U"


def test_anova_path_reports_eta_squared():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "v": np.concatenate([rng.normal(m, 1, 40) for m in (0, 1, 2)]),
        "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
    })
    result = stats.compare_groups(frame, "v", "g", parametric=True)
    assert result.name == "One-way ANOVA"
    assert result.effect_name == "eta_squared"
    assert 0 <= result.effect_size <= 1


def test_kruskal_path_reports_epsilon_squared():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "v": np.concatenate([rng.exponential(s, 40) for s in (1, 2, 3)]),
        "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
    })
    result = stats.compare_groups(frame, "v", "g", parametric=False)
    assert result.name == "Kruskal-Wallis H"
    assert result.effect_name == "epsilon_squared"


def test_normality_needs_three_observations():
    with pytest.raises(ValueError, match="at least 3"):
        stats.normality(pd.Series([1.0, 2.0]))


def test_normality_subsamples_large_inputs():
    big = pd.Series(np.random.default_rng(5).normal(size=20000))
    assert stats.normality(big).detail["tested_on"] == 5000


def test_empty_adjustment():
    assert stats.adjust_p_values([]) == []


def test_correlation_scan_of_a_single_column_is_empty(frame):
    assert stats.correlation_scan(frame, ["revenue"]).empty


def test_correlation_scan_skips_undefined_pairs(frame):
    """A constant column has no defined correlation and must not enter the table."""
    scan = stats.correlation_scan(frame.assign(constant=1.0), ["revenue", "cost", "constant"])
    assert "constant" not in set(scan["x"]) | set(scan["y"])


def test_undefined_pairs_do_not_invalidate_the_others(frame):
    """A NaN p-value would propagate through the FDR adjustment and mark every
    other pair insignificant — the whole scan would silently go blank."""
    clean = stats.correlation_scan(frame, ["revenue", "cost", "orders"])
    padded = stats.correlation_scan(frame.assign(constant=1.0), ["revenue", "cost", "orders", "constant"])
    assert padded["significant"].all()
    assert padded["p_adjusted"].notna().all()
    assert len(padded) == len(clean)


def test_as_row_handles_a_missing_effect():
    result = stats.TestResult("t", 1.0, 0.5, 10)
    assert result.as_row()["effect"] is None
