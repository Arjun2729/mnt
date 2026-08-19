"""Every test reports an effect size; p-values alone are not enough."""
import numpy as np
import pandas as pd
import pytest

from groundtruth import stats


def test_correlation_has_interval(frame):
    result = stats.correlation(frame, "revenue", "orders")
    assert result.effect_name == "r"
    assert result.ci is not None
    assert result.ci[0] < result.statistic < result.ci[1]


def test_correlation_needs_enough_data():
    tiny = pd.DataFrame({"a": [1, 2], "b": [1, 2]})
    with pytest.raises(ValueError):
        stats.correlation(tiny, "a", "b")


def test_chi_square_reports_cramers_v(frame):
    result = stats.chi_square(frame, "region", "channel")
    assert result.effect_name == "cramers_v"
    assert 0 <= result.effect_size <= 1


def test_chi_square_finds_a_real_association():
    # y is fully determined by x, so association must be strong.
    x = ["a"] * 50 + ["b"] * 50
    df = pd.DataFrame({"x": x, "y": ["p" if v == "a" else "q" for v in x]})
    assert stats.chi_square(df, "x", "y").effect_size > 0.9


def test_two_groups_use_a_two_sample_test(frame):
    two = frame[frame["channel"].isin(["Paid", "Organic"])]
    result = stats.compare_groups(two, "revenue", "channel")
    assert result.effect_name == "cohens_d"
    assert result.ci is not None
    assert result.name in ("Welch's t-test", "Mann-Whitney U")


def test_many_groups_use_an_omnibus_test(frame):
    result = stats.compare_groups(frame, "revenue", "region")
    assert result.effect_name in ("eta_squared", "epsilon_squared")
    assert result.name in ("One-way ANOVA", "Kruskal-Wallis H")


def test_normality_flags_a_skewed_column():
    skewed = pd.Series(np.random.default_rng(0).exponential(size=400))
    assert stats.normality(skewed).p_value < 0.05


def test_benjamini_hochberg_is_monotone_and_inflates():
    raw = [0.001, 0.01, 0.04, 0.2]
    adjusted = stats.adjust_p_values(raw)
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert adjusted == sorted(adjusted)


def test_bonferroni_is_harsher_than_fdr():
    raw = [0.01, 0.02, 0.03]
    assert stats.adjust_p_values(raw, "bonferroni")[0] >= stats.adjust_p_values(raw)[0]


def test_correlation_scan_corrects_for_multiplicity(frame):
    scan = stats.correlation_scan(frame, ["revenue", "cost", "orders", "conversion_rate"])
    assert "p_adjusted" in scan
    assert (scan["p_adjusted"] >= scan["p_value"]).all()
    # Ranked by absolute strength.
    assert scan["r"].abs().is_monotonic_decreasing
