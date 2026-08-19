"""L3 — statistics.

Every test reports an effect size alongside its p-value, and an interval
wherever one is defined. A p-value on its own says only "probably not zero",
which is rarely the question anyone actually has.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    n: int
    effect_name: str = ""
    effect_size: float | None = None
    ci: tuple[float, float] | None = None
    interpretation: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "test": self.name,
            "statistic": round(self.statistic, 4),
            "p_value": self.p_value,
            "n": self.n,
            self.effect_name or "effect": round(self.effect_size, 4) if self.effect_size is not None else None,
            "ci_low": round(self.ci[0], 4) if self.ci else None,
            "ci_high": round(self.ci[1], 4) if self.ci else None,
            "reading": self.interpretation,
        }


def _label(value: float, thresholds: tuple[float, float, float], labels=("negligible", "small", "moderate", "large")) -> str:
    magnitude = abs(value)
    for threshold, label in zip(thresholds, labels):
        if magnitude < threshold:
            return label
    return labels[-1]


# ---------- numeric vs numeric ----------


def correlation(df: pd.DataFrame, x: str, y: str, method: str = "pearson", alpha: float = 0.05) -> TestResult:
    pair = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    if n < 4:
        raise ValueError("Need at least 4 complete observations.")

    if method.lower() == "pearson":
        r, p = sps.pearsonr(pair[x], pair[y])
        name = "Pearson correlation"
    else:
        r, p = sps.spearmanr(pair[x], pair[y])
        name = "Spearman correlation"

    # Fisher z transform gives an interval for both r and rho.
    if abs(r) < 1:
        z = math.atanh(r)
        se = 1 / math.sqrt(n - 3) if n > 3 else float("nan")
        crit = sps.norm.ppf(1 - alpha / 2)
        ci = (math.tanh(z - crit * se), math.tanh(z + crit * se))
    else:
        ci = (r, r)

    return TestResult(
        name=name,
        statistic=float(r),
        p_value=float(p),
        n=n,
        effect_name="r",
        effect_size=float(r),
        ci=ci,
        interpretation=f"{_label(r, (0.1, 0.3, 0.5))} {'positive' if r >= 0 else 'negative'} association",
        detail={"r_squared": round(float(r) ** 2, 4)},
    )


# ---------- categorical vs categorical ----------


def chi_square(df: pd.DataFrame, x: str, y: str) -> TestResult:
    table = pd.crosstab(df[x], df[y])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both columns need at least two categories.")
    chi2, p, dof, expected = sps.chi2_contingency(table)
    n = int(table.values.sum())

    # Cramer's V, bias-corrected (Bergsma 2013).
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    v = math.sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0

    low_expected = float((expected < 5).mean())
    return TestResult(
        name="Chi-square test of independence",
        statistic=float(chi2),
        p_value=float(p),
        n=n,
        effect_name="cramers_v",
        effect_size=float(v),
        interpretation=f"{_label(v, (0.1, 0.3, 0.5))} association",
        detail={
            "dof": int(dof),
            "shape": f"{r}x{k}",
            "cells_expected_below_5_pct": round(low_expected * 100, 1),
            "warning": "Chi-square is unreliable when many expected counts are below 5."
            if low_expected > 0.2
            else "",
        },
    )


# ---------- numeric vs categorical ----------


def compare_groups(df: pd.DataFrame, value: str, group: str, parametric: bool | None = None) -> TestResult:
    data = df[[value, group]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = [g[value].to_numpy() for _, g in data.groupby(group) if len(g) >= 2]
    labels = [str(name) for name, g in data.groupby(group) if len(g) >= 2]
    if len(groups) < 2:
        raise ValueError("Need at least two groups with 2+ observations each.")
    n = int(sum(len(g) for g in groups))

    # Choose the test family from a normality check unless the caller insists.
    if parametric is None:
        residuals = np.concatenate([g - g.mean() for g in groups])
        sample = residuals if len(residuals) <= 5000 else np.random.default_rng(0).choice(residuals, 5000, replace=False)
        parametric = bool(sps.shapiro(sample).pvalue > 0.05) if 3 <= len(sample) <= 5000 else True

    if len(groups) == 2:
        a, b = groups
        if parametric:
            statistic, p = sps.ttest_ind(a, b, equal_var=False)
            name = "Welch's t-test"
        else:
            statistic, p = sps.mannwhitneyu(a, b, alternative="two-sided")
            name = "Mann-Whitney U"
        pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
        d = (a.mean() - b.mean()) / pooled if pooled else 0.0
        se = math.sqrt((len(a) + len(b)) / (len(a) * len(b)) + d**2 / (2 * (len(a) + len(b))))
        crit = sps.norm.ppf(0.975)
        return TestResult(
            name=name,
            statistic=float(statistic),
            p_value=float(p),
            n=n,
            effect_name="cohens_d",
            effect_size=float(d),
            ci=(float(d - crit * se), float(d + crit * se)),
            interpretation=f"{_label(d, (0.2, 0.5, 0.8))} difference between {labels[0]} and {labels[1]}",
            detail={"groups": labels, "means": [round(float(g.mean()), 4) for g in groups], "normal_residuals": parametric},
        )

    if parametric:
        statistic, p = sps.f_oneway(*groups)
        name = "One-way ANOVA"
        grand = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = sum(((np.concatenate(groups) - grand) ** 2))
        effect = ss_between / ss_total if ss_total else 0.0
        effect_name, thresholds = "eta_squared", (0.01, 0.06, 0.14)
    else:
        statistic, p = sps.kruskal(*groups)
        name = "Kruskal-Wallis H"
        k = len(groups)
        effect = (statistic - k + 1) / (n - k) if n > k else 0.0
        effect_name, thresholds = "epsilon_squared", (0.01, 0.06, 0.14)

    return TestResult(
        name=name,
        statistic=float(statistic),
        p_value=float(p),
        n=n,
        effect_name=effect_name,
        effect_size=float(effect),
        interpretation=f"{_label(effect, thresholds)} share of variance explained by {group}",
        detail={"groups": labels, "means": [round(float(g.mean()), 4) for g in groups], "normal_residuals": parametric},
    )


# ---------- normality ----------


def normality(series: pd.Series) -> TestResult:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(clean) < 3:
        raise ValueError("Need at least 3 observations.")
    sample = clean if len(clean) <= 5000 else np.random.default_rng(0).choice(clean, 5000, replace=False)
    statistic, p = sps.shapiro(sample)
    return TestResult(
        name="Shapiro-Wilk normality",
        statistic=float(statistic),
        p_value=float(p),
        n=len(clean),
        effect_name="skew",
        effect_size=float(sps.skew(clean)),
        interpretation="consistent with normal" if p > 0.05 else "departs from normal",
        detail={"kurtosis": round(float(sps.kurtosis(clean)), 4), "tested_on": len(sample)},
    )


# ---------- multiple comparisons ----------


def adjust_p_values(p_values: list[float], method: str = "fdr_bh") -> list[float]:
    """Benjamini-Hochberg by default: scanning many pairs inflates false positives."""
    p_array = np.asarray(p_values, dtype=float)
    n = len(p_array)
    if n == 0:
        return []
    if method == "bonferroni":
        return list(np.minimum(p_array * n, 1.0))
    order = np.argsort(p_array)
    ranked = p_array[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return list(out)


def correlation_scan(df: pd.DataFrame, columns: list[str], method: str = "pearson", alpha: float = 0.05) -> pd.DataFrame:
    """All numeric pairs, ranked by strength, with FDR-corrected p-values."""
    rows = []
    for i, x in enumerate(columns):
        for y in columns[i + 1 :]:
            try:
                with warnings.catch_warnings():
                    # A constant column warns rather than raising; it is filtered below.
                    warnings.simplefilter("ignore")
                    result = correlation(df, x, y, method)
            except Exception:
                continue
            # An undefined correlation must not enter the table: its NaN p-value would
            # propagate through the FDR adjustment and mark every other pair
            # insignificant.
            if pd.isna(result.statistic) or pd.isna(result.p_value):
                continue
            rows.append({"x": x, "y": y, "r": result.statistic, "p_value": result.p_value, "n": result.n,
                         "ci_low": result.ci[0] if result.ci else None, "ci_high": result.ci[1] if result.ci else None})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["p_adjusted"] = adjust_p_values(frame["p_value"].tolist())
    frame["significant"] = frame["p_adjusted"] < alpha
    return frame.reindex(frame["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
