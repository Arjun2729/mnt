"""L4 — the proactive scan.

Runs on load, before anyone asks a question. Everything here is mechanical
statistics rather than a model call: it finds the things a competent analyst
would notice in the first five minutes, so the user starts from findings rather
than from a blank prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .semantic import DatasetSpec
from .store import Store, quote_ident


@dataclass
class Insight:
    kind: str
    headline: str
    detail: str
    severity: str = "info"  # info | notable | warning
    evidence_sql: str = ""
    score: float = 0.0


def _q(name: str) -> str:
    return quote_ident(name)


def scan(store: Store, table: str, spec: DatasetSpec, max_insights: int = 12) -> list[Insight]:
    findings: list[Insight] = []
    findings += _data_quality(store, table, spec)
    findings += _correlations(store, table, spec)
    findings += _segment_outliers(store, table, spec)
    findings += _time_movement(store, table, spec)
    findings += _distribution_outliers(store, table, spec)
    findings.sort(key=lambda i: i.score, reverse=True)
    return findings[:max_insights]


def _data_quality(store: Store, table: str, spec: DatasetSpec) -> list[Insight]:
    out: list[Insight] = []
    for column in spec.columns:
        if column.missing_pct >= 20:
            out.append(
                Insight(
                    "quality",
                    f"{column.name} is {column.missing_pct:.0f}% empty",
                    f"{column.missing:,} of {spec.rows:,} rows have no value. Filters and models on this column "
                    f"silently drop those rows.",
                    "warning",
                    f"SELECT COUNT(*) FROM {_q(table)} WHERE {_q(column.name)} IS NULL",
                    score=column.missing_pct / 10,
                )
            )
        if column.distinct == 1 and spec.rows > 1:
            out.append(
                Insight(
                    "quality",
                    f"{column.name} holds a single value",
                    "A constant column cannot explain variation in anything. Consider excluding it.",
                    "info",
                    f"SELECT DISTINCT {_q(column.name)} FROM {_q(table)}",
                    score=1.5,
                )
            )
    if spec.key_candidates:
        duplicate_check = store.sql(
            f"SELECT COUNT(*) - COUNT(DISTINCT {_q(spec.key_candidates[0])}) AS dupes FROM {_q(table)}"
        )["dupes"].iloc[0]
        if duplicate_check:
            out.append(Insight("quality", f"Duplicate values in {spec.key_candidates[0]}", f"{int(duplicate_check):,} repeats.", "warning", score=3))
    return out


def _correlations(store: Store, table: str, spec: DatasetSpec) -> list[Insight]:
    measures = spec.measures
    if len(measures) < 2:
        return []
    frame = store.sql(f"SELECT {', '.join(_q(m) for m in measures)} FROM {_q(table)}")
    matrix = frame.corr(numeric_only=True)
    out: list[Insight] = []
    seen: set[frozenset] = set()
    for x in matrix.columns:
        for y in matrix.columns:
            if x == y or frozenset((x, y)) in seen:
                continue
            seen.add(frozenset((x, y)))
            r = matrix.loc[x, y]
            if pd.isna(r) or abs(r) < 0.6:
                continue
            direction = "rise together" if r > 0 else "move in opposite directions"
            out.append(
                Insight(
                    "correlation",
                    f"{x} and {y} {direction} (r = {r:.2f})",
                    f"{abs(r) ** 2 * 100:.0f}% of the variation in one is shared with the other. "
                    f"Shared movement is not evidence that one causes the other.",
                    "notable" if abs(r) >= 0.8 else "info",
                    f"SELECT corr({_q(x)}, {_q(y)}) FROM {_q(table)}",
                    score=abs(r) * 4,
                )
            )
    return out


def _segment_outliers(store: Store, table: str, spec: DatasetSpec) -> list[Insight]:
    """Group means that sit far from the overall mean."""
    out: list[Insight] = []
    for dimension in spec.dimensions[:4]:
        column = spec.column(dimension)
        if not column or column.distinct > 40 or column.distinct < 2:
            continue
        for measure in spec.measures[:3]:
            frame = store.sql(
                f"SELECT {_q(dimension)} AS grp, AVG({_q(measure)}) AS avg_value, COUNT(*) AS n "
                f"FROM {_q(table)} GROUP BY 1 HAVING COUNT(*) >= 3"
            )
            if len(frame) < 3:
                continue
            # Median and MAD rather than mean and standard deviation: with only a
            # handful of groups an outlier inflates the very spread used to judge
            # it, and the detector misses exactly what it exists to find.
            overall = float(frame["avg_value"].median())
            deviations = (frame["avg_value"] - overall).abs()
            spread = float(1.4826 * deviations.median())
            if not spread or pd.isna(spread):
                spread = float(frame["avg_value"].std())
            if not spread or pd.isna(spread) or overall == 0:
                continue
            frame["z"] = (frame["avg_value"] - overall) / spread
            extreme = frame.reindex(frame["z"].abs().sort_values(ascending=False).index).iloc[0]
            delta = (extreme["avg_value"] - overall) / abs(overall) * 100
            # A robust scale makes tiny spreads produce large z-scores, so also
            # require the gap to be material. Statistically unusual is not the
            # same as worth reading.
            if abs(extreme["z"]) < 1.3 or abs(delta) < 10:
                continue
            out.append(
                Insight(
                    "segment",
                    f"{dimension} = {extreme['grp']} averages {delta:+.0f}% on {measure}",
                    f"Mean {measure} of {extreme['avg_value']:,.2f} against {overall:,.2f} across all "
                    f"{dimension} values, from {int(extreme['n']):,} rows.",
                    "notable" if abs(extreme["z"]) > 1.8 else "info",
                    f"SELECT {_q(dimension)}, AVG({_q(measure)}) FROM {_q(table)} GROUP BY 1 ORDER BY 2 DESC",
                    score=abs(extreme["z"]) * 2,
                )
            )
    return out


def _time_movement(store: Store, table: str, spec: DatasetSpec) -> list[Insight]:
    if not spec.time_column or not spec.measures:
        return []
    grain = spec.time_grain or "month"
    out: list[Insight] = []
    for measure in spec.measures[:3]:
        frame = store.sql(
            f"SELECT date_trunc('{grain}', {_q(spec.time_column)}) AS period, SUM({_q(measure)}) AS total "
            f"FROM {_q(table)} WHERE {_q(spec.time_column)} IS NOT NULL GROUP BY 1 ORDER BY 1"
        )
        if len(frame) < 4:
            continue
        values = frame["total"].to_numpy(dtype=float)

        # Overall drift, measured as a share of the mean level.
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        level = float(np.mean(values))
        if level and abs(slope * len(values)) / abs(level) > 0.25:
            direction = "rising" if slope > 0 else "falling"
            out.append(
                Insight(
                    "trend",
                    f"{measure} is {direction} across the period",
                    f"About {slope * len(values) / abs(level) * 100:+.0f}% total drift over {len(values)} {grain}s "
                    f"({values[0]:,.0f} to {values[-1]:,.0f}).",
                    "notable",
                    f"SELECT date_trunc('{grain}', {_q(spec.time_column)}), SUM({_q(measure)}) FROM {_q(table)} GROUP BY 1 ORDER BY 1",
                    score=min(abs(slope * len(values)) / abs(level) * 6, 9),
                )
            )

        # Largest single-period jump.
        deltas = np.diff(values)
        if len(deltas) and np.std(deltas):
            index = int(np.argmax(np.abs(deltas)))
            z = abs(deltas[index]) / np.std(deltas)
            if z > 2.2:
                period = pd.to_datetime(frame["period"].iloc[index + 1])
                pct = deltas[index] / abs(values[index]) * 100 if values[index] else 0
                out.append(
                    Insight(
                        "movement",
                        f"{measure} moved {pct:+.0f}% into {period.date()}",
                        f"The largest single-{grain} change in the series, {z:.1f}x the typical move.",
                        "notable",
                        f"SELECT date_trunc('{grain}', {_q(spec.time_column)}), SUM({_q(measure)}) FROM {_q(table)} GROUP BY 1 ORDER BY 1",
                        score=z * 1.6,
                    )
                )
    return out


def _distribution_outliers(store: Store, table: str, spec: DatasetSpec) -> list[Insight]:
    out: list[Insight] = []
    for measure in spec.measures[:5]:
        stats = store.sql(
            f"SELECT quantile_cont({_q(measure)}, 0.25) AS q1, quantile_cont({_q(measure)}, 0.75) AS q3 FROM {_q(table)}"
        )
        q1, q3 = float(stats["q1"].iloc[0] or 0), float(stats["q3"].iloc[0] or 0)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        count = store.count(table, f"{_q(measure)} < ? OR {_q(measure)} > ?", [low, high])
        if count and count / max(spec.rows, 1) > 0.005:
            out.append(
                Insight(
                    "outlier",
                    f"{count:,} extreme values in {measure}",
                    f"{count / spec.rows * 100:.1f}% of rows sit beyond 3x the interquartile range "
                    f"({low:,.2f} to {high:,.2f}). These will pull means and regression fits.",
                    "warning" if count / spec.rows > 0.05 else "info",
                    f"SELECT * FROM {_q(table)} WHERE {_q(measure)} < {low} OR {_q(measure)} > {high}",
                    score=min(count / spec.rows * 30, 6),
                )
            )
    return out


def insights_frame(findings: list[Insight]) -> pd.DataFrame:
    return pd.DataFrame([{"severity": f.severity, "kind": f.kind, "finding": f.headline, "detail": f.detail} for f in findings])
