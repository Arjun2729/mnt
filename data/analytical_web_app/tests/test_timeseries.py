"""Decomposition, changepoints, forecasting and anomalies."""
import numpy as np
import pandas as pd
import pytest

from groundtruth import timeseries as ts


@pytest.fixture
def seasonal_series():
    index = pd.date_range("2020-01-01", periods=60, freq="MS")
    trend = np.linspace(100, 200, 60)
    season = 20 * np.sin(2 * np.pi * np.arange(60) / 12)
    return pd.Series(trend + season, index=index)


def test_aggregate_collapses_to_one_row_per_period(frame):
    series = ts.aggregate(frame, "date", "revenue", "month", "sum")
    assert len(series) == 24
    assert series.index.is_monotonic_increasing


def test_decompose_separates_trend_and_season(seasonal_series):
    result = ts.decompose(seasonal_series, "month")
    assert result.seasonal_strength > 0.5
    assert result.trend_strength > 0.5
    assert not result.caveat  # five cycles is plenty


def test_decompose_warns_on_short_history(frame):
    series = ts.aggregate(frame, "date", "revenue", "month")
    assert "cycles" in ts.decompose(series, "month").caveat


def test_decompose_refuses_impossible_history():
    short = pd.Series(range(5), index=pd.date_range("2024-01-01", periods=5, freq="MS"))
    with pytest.raises(ValueError, match="at least"):
        ts.decompose(short, "month")


def test_forecast_produces_ordered_intervals(seasonal_series):
    result = ts.forecast(seasonal_series, periods=6, grain="month")
    assert len(result.mean) == 6
    assert (result.lower <= result.mean).all()
    assert (result.mean <= result.upper).all()


def test_forecast_intervals_widen_with_horizon(seasonal_series):
    result = ts.forecast(seasonal_series, periods=12, grain="month")
    width = result.upper - result.lower
    assert width.iloc[-1] > width.iloc[0]


def test_forecast_needs_history():
    with pytest.raises(ValueError, match="at least 6"):
        ts.forecast(pd.Series(range(3), index=pd.date_range("2024-01-01", periods=3, freq="MS")))


def test_changepoint_is_found_at_a_level_shift():
    index = pd.date_range("2020-01-01", periods=40, freq="MS")
    values = np.concatenate([np.full(20, 10.0), np.full(20, 50.0)])
    found = ts.detect_changepoints(pd.Series(values, index=index))
    assert found
    assert abs((found[0] - index[20]).days) <= 62


def test_no_changepoint_in_flat_noise():
    index = pd.date_range("2020-01-01", periods=40, freq="MS")
    values = np.random.default_rng(0).normal(10, 0.5, 40)
    assert ts.detect_changepoints(pd.Series(values, index=index)) == []


def test_anomaly_is_flagged():
    index = pd.date_range("2020-01-01", periods=30, freq="MS")
    values = np.full(30, 10.0)
    values[15] = 500.0
    found = ts.anomalies(pd.Series(values, index=index), "month", sensitivity=3.0)
    assert len(found) >= 1
    assert (found["value"] == 500.0).any()
