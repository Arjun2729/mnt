"""L3 — time series.

Decomposition, changepoints and forecasting with prediction intervals. The
forecast intervals matter beyond the chart: the alert layer uses them to raise
anomalies without anyone having to pick a threshold by hand.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

_GRAIN_FREQ = {"hour": "h", "day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
_GRAIN_SEASON = {"hour": 24, "day": 7, "week": 52, "month": 12, "quarter": 4, "year": 1}


@dataclass
class Decomposition:
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    seasonal_strength: float
    trend_strength: float
    caveat: str = ""


@dataclass
class Forecast:
    history: pd.Series
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    model: str
    in_sample_mae: float


def aggregate(df: pd.DataFrame, time_column: str, value_column: str, grain: str = "month", how: str = "sum") -> pd.Series:
    """Collapse to one observation per period — the shape every model here expects."""
    frame = df[[time_column, value_column]].copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.dropna(subset=[time_column])
    if frame.empty:
        raise ValueError("No valid timestamps in the selected column.")
    freq = _GRAIN_FREQ.get(grain, "MS")
    series = frame.set_index(time_column)[value_column].resample(freq).agg(how)
    return series.astype(float)


def decompose(series: pd.Series, grain: str = "month") -> Decomposition:
    from statsmodels.tsa.seasonal import STL

    period = _GRAIN_SEASON.get(grain, 12)
    clean = series.interpolate().dropna()
    if period < 2 or len(clean) < 2 * period:
        raise ValueError(f"Need at least {2 * max(period, 2)} periods to separate seasonality (have {len(clean)}).")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = STL(clean, period=period, robust=True).fit()

    residual_var = float(np.var(result.resid))
    # Strength measures from Hyndman & Athanasopoulos, capped at 0.
    seasonal_strength = max(0.0, 1 - residual_var / max(float(np.var(result.resid + result.seasonal)), 1e-12))
    trend_strength = max(0.0, 1 - residual_var / max(float(np.var(result.resid + result.trend)), 1e-12))
    # Two cycles is the mathematical minimum; the split is unstable until roughly three.
    caveat = (
        f"Only {len(clean) / period:.1f} seasonal cycles available — strength estimates are unstable below 3."
        if len(clean) < 3 * period
        else ""
    )
    return Decomposition(clean, result.trend, result.seasonal, result.resid, seasonal_strength, trend_strength, caveat)


def detect_changepoints(series: pd.Series, min_segment: int = 4, max_points: int = 5, threshold: float = 2.5) -> list[pd.Timestamp]:
    """Binary segmentation on mean shift, scored against within-segment noise."""
    values = series.interpolate().dropna()
    if len(values) < min_segment * 2:
        return []

    found: list[int] = []

    def search(lo: int, hi: int) -> None:
        if len(found) >= max_points or hi - lo < min_segment * 2:
            return
        window = values.iloc[lo:hi].to_numpy()
        best_score, best_index = 0.0, -1
        for i in range(min_segment, len(window) - min_segment):
            left, right = window[:i], window[i:]
            spread = np.sqrt((left.var() + right.var()) / 2)
            if spread <= 0:
                continue
            score = abs(left.mean() - right.mean()) / spread
            if score > best_score:
                best_score, best_index = score, i
        if best_index > 0 and best_score >= threshold:
            absolute = lo + best_index
            found.append(absolute)
            search(lo, absolute)
            search(absolute, hi)

    search(0, len(values))
    return [values.index[i] for i in sorted(found)]


def forecast(series: pd.Series, periods: int = 6, grain: str = "month", alpha: float = 0.05) -> Forecast:
    """Seasonal model where there is enough history for one, damped trend otherwise."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    clean = series.interpolate().dropna()
    if len(clean) < 6:
        raise ValueError(f"Need at least 6 periods of history to forecast (have {len(clean)}).")

    period = _GRAIN_SEASON.get(grain, 12)
    seasonal = period >= 2 and len(clean) >= 2 * period + 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            order, seasonal_order = (1, 1, 1), ((1, 1, 1, period) if seasonal else (0, 0, 0, 0))
            fitted = SARIMAX(
                clean, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            prediction = fitted.get_forecast(steps=periods)
            mean = prediction.predicted_mean
            interval = prediction.conf_int(alpha=alpha)
            lower, upper = interval.iloc[:, 0], interval.iloc[:, 1]
            in_sample = float(np.mean(np.abs(fitted.resid[period:]))) if len(fitted.resid) > period else float("nan")
            name = f"SARIMAX{order}x{seasonal_order}" if seasonal else f"ARIMA{order}"
        except Exception:
            model = ExponentialSmoothing(
                clean, trend="add", damped_trend=True,
                seasonal="add" if seasonal else None,
                seasonal_periods=period if seasonal else None,
            ).fit()
            mean = model.forecast(periods)
            residual_sd = float(np.std(model.resid))
            from scipy import stats as sps

            crit = sps.norm.ppf(1 - alpha / 2)
            # Widen with horizon: uncertainty compounds the further out you look.
            spread = crit * residual_sd * np.sqrt(np.arange(1, periods + 1))
            lower, upper = mean - spread, mean + spread
            in_sample = float(np.mean(np.abs(model.resid)))
            name = "Holt-Winters (damped)"

    return Forecast(clean, mean, pd.Series(lower.values, index=mean.index), pd.Series(upper.values, index=mean.index), name, in_sample)


def anomalies(series: pd.Series, grain: str = "month", sensitivity: float = 3.0) -> pd.DataFrame:
    """Points whose residual against a robust rolling level is unusually large."""
    clean = series.interpolate().dropna()
    period = _GRAIN_SEASON.get(grain, 12)
    window = max(3, min(period, max(3, len(clean) // 4)))
    level = clean.rolling(window, center=True, min_periods=2).median()
    residual = clean - level
    # MAD is the robust default, but it collapses to zero on a near-constant
    # baseline — where a single spike is exactly what we most want to catch.
    scale = 1.4826 * (residual - residual.median()).abs().median()
    if not scale or np.isnan(scale):
        scale = float(residual.std())
    if not scale or np.isnan(scale):
        return pd.DataFrame(columns=["timestamp", "value", "expected", "z_score"])
    z = (residual / scale).abs()
    flagged = z[z >= sensitivity]
    return pd.DataFrame(
        {
            "timestamp": flagged.index,
            "value": clean.loc[flagged.index].values,
            "expected": level.loc[flagged.index].values,
            "z_score": flagged.values.round(2),
        }
    ).reset_index(drop=True)
