"""Time-series branches: bad input, fallbacks, degenerate series."""
import numpy as np
import pandas as pd
import pytest

from groundtruth import timeseries as ts


def test_unparseable_timestamps_are_refused():
    frame = pd.DataFrame({"when": ["not a date", "nor this"], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="No valid timestamps"):
        ts.aggregate(frame, "when", "v", "month")


@pytest.mark.parametrize("how", ["sum", "mean", "median", "max", "min"])
def test_aggregations(frame, how):
    assert len(ts.aggregate(frame, "date", "revenue", "month", how)) == 24


def test_changepoints_need_enough_points():
    short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    assert ts.detect_changepoints(short) == []


def test_changepoints_on_a_flat_series():
    flat = pd.Series([5.0] * 40, index=pd.date_range("2024-01-01", periods=40, freq="MS"))
    assert ts.detect_changepoints(flat) == []


def test_a_perfectly_clean_step_is_detected():
    """Flat segments have zero within-segment variance. Treating that as
    unscoreable made the strongest possible changepoint undetectable."""
    index = pd.date_range("2020-01-01", periods=90, freq="D")
    values = np.concatenate([np.full(45, 10.0), np.full(45, 60.0)])
    found = ts.detect_changepoints(pd.Series(values, index=index))
    assert found
    assert abs((found[0] - index[45]).days) <= 2


def test_a_noisy_step_is_detected():
    index = pd.date_range("2020-01-01", periods=90, freq="D")
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(10, 1, 45), rng.normal(50, 1, 45)])
    assert ts.detect_changepoints(pd.Series(values, index=index))


def test_symmetric_double_steps_are_a_known_limit():
    """Binary segmentation scores the top-level split against two inhomogeneous
    halves, so an up-and-back-down pattern is missed. Documented, not fixed."""
    index = pd.date_range("2020-01-01", periods=90, freq="D")
    values = np.concatenate([np.full(30, 10.0), np.full(30, 60.0), np.full(30, 10.0)])
    assert ts.detect_changepoints(pd.Series(values, index=index)) == []


def test_forecast_falls_back_when_the_state_space_model_fails(monkeypatch):
    """SARIMAX can fail to converge; the exponential-smoothing path must cover it."""
    import statsmodels.tsa.statespace.sarimax as sarimax_module

    class _Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated convergence failure")

    monkeypatch.setattr(sarimax_module, "SARIMAX", _Broken)
    index = pd.date_range("2020-01-01", periods=48, freq="MS")
    series = pd.Series(np.linspace(100, 200, 48) + 10 * np.sin(np.arange(48)), index=index)
    result = ts.forecast(series, periods=6, grain="month")
    assert "Holt-Winters" in result.model
    assert (result.lower <= result.mean).all() and (result.mean <= result.upper).all()


def test_fallback_intervals_widen_with_horizon(monkeypatch):
    import statsmodels.tsa.statespace.sarimax as sarimax_module

    class _Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("nope")

    monkeypatch.setattr(sarimax_module, "SARIMAX", _Broken)
    index = pd.date_range("2020-01-01", periods=48, freq="MS")
    series = pd.Series(np.linspace(100, 200, 48), index=index)
    result = ts.forecast(series, periods=12, grain="month")
    width = result.upper - result.lower
    assert width.iloc[-1] > width.iloc[0]


def test_anomalies_on_a_perfectly_flat_series():
    """Zero variance means nothing can be anomalous."""
    flat = pd.Series([7.0] * 30, index=pd.date_range("2024-01-01", periods=30, freq="MS"))
    assert ts.anomalies(flat, "month").empty


def test_anomaly_columns_are_stable():
    flat = pd.Series([7.0] * 30, index=pd.date_range("2024-01-01", periods=30, freq="MS"))
    assert list(ts.anomalies(flat, "month").columns) == ["timestamp", "value", "expected", "z_score"]


def test_unknown_grain_falls_back_to_month(frame):
    assert len(ts.aggregate(frame, "date", "revenue", "fortnight")) == 24
