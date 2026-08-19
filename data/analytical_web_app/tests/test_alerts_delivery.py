"""Webhook delivery and anomaly rules — the alert paths that reach outside."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from groundtruth.alerts import AlertStore, Rule, evaluate, evaluate_all, notify

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return AlertStore(tmp_path / "state.json")


class _Recorder:
    """Captures webhook posts instead of making them."""

    def __init__(self, fail: bool = False):
        self.posts: list[dict] = []
        self.fail = fail

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "body": json})

        class _Response:
            def raise_for_status(inner):
                if self.fail:
                    raise RuntimeError("502 Bad Gateway")

        return _Response()


def test_only_transitions_are_delivered(store, monkeypatch):
    import groundtruth.alerts as alerts

    recorder = _Recorder()
    monkeypatch.setitem(__import__("sys").modules, "requests", recorder)

    rule = Rule("r", "metric", "mean", ">", 10, cooldown_minutes=0)
    frame = pd.DataFrame({"metric": [50] * 5})

    first = [evaluate(rule, frame, store, T0)]
    assert notify(first, "https://hook.test/x") == 1

    second = [evaluate(rule, frame, store, T0.replace(hour=1))]
    assert notify(second, "https://hook.test/x") == 0, "a still-firing rule must not re-notify"
    assert len(recorder.posts) == 1


def test_payload_carries_the_state(store, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(__import__("sys").modules, "requests", recorder)

    rule = Rule("Revenue", "metric", "mean", ">", 10, cooldown_minutes=0)
    notify([evaluate(rule, pd.DataFrame({"metric": [99]}), store, T0)], "https://hook.test/x")

    body = recorder.posts[0]["body"]
    assert body["rule"] == "Revenue"
    assert body["state"] == "firing"
    assert body["transition"] == "fired"
    assert "Revenue" in body["text"]


def test_delivery_failure_propagates(store, monkeypatch):
    recorder = _Recorder(fail=True)
    monkeypatch.setitem(__import__("sys").modules, "requests", recorder)

    rule = Rule("r", "metric", "mean", ">", 1, cooldown_minutes=0)
    with pytest.raises(RuntimeError, match="502"):
        notify([evaluate(rule, pd.DataFrame({"metric": [5]}), store, T0)], "https://hook.test/x")


# ---------------- anomaly rules ----------------


def _seasonal_frame(spike: float | None = None) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=36, freq="MS")
    values = 100 + 10 * np.sin(2 * np.pi * np.arange(36) / 12)
    if spike is not None:
        values[-1] = spike
    return pd.DataFrame({"when": index, "metric": values})


def test_anomaly_rule_stays_quiet_on_normal_data(store):
    rule = Rule("anom", "metric", kind="anomaly", sensitivity=3.0, cooldown_minutes=0)
    result = evaluate(rule, _seasonal_frame(), store, T0, time_column="when", grain="month")
    assert result.new_state == "ok"
    assert result.expected is not None and result.expected[0] < result.expected[1]


def test_anomaly_rule_fires_on_a_spike(store):
    rule = Rule("anom", "metric", kind="anomaly", sensitivity=3.0, cooldown_minutes=0)
    result = evaluate(rule, _seasonal_frame(spike=100000.0), store, T0, time_column="when", grain="month")
    assert result.new_state == "firing"
    assert result.transition == "fired"


def test_anomaly_rule_needs_a_time_column(store):
    rule = Rule("anom", "metric", kind="anomaly")
    with pytest.raises(ValueError, match="time column"):
        evaluate(rule, _seasonal_frame(), store, T0)


def test_anomaly_rule_needs_enough_history(store):
    rule = Rule("anom", "metric", kind="anomaly")
    short = pd.DataFrame({
        "when": pd.date_range("2026-01-01", periods=4, freq="MS"),
        "metric": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="at least 8 periods"):
        evaluate(rule, short, store, T0, time_column="when", grain="month")


def test_anomaly_describes_itself_without_a_threshold():
    assert "forecast" in Rule("a", "metric", kind="anomaly").describe()


# ---------------- aggregation coverage ----------------


@pytest.mark.parametrize("aggregation,expected", [
    ("mean", 3.0), ("sum", 15.0), ("min", 1.0), ("max", 5.0),
    ("count", 5.0), ("median", 3.0),
])
def test_every_aggregation_computes(store, aggregation, expected):
    rule = Rule("r", "metric", aggregation, ">", -1, cooldown_minutes=0)
    frame = pd.DataFrame({"metric": [1, 2, 3, 4, 5]})
    assert evaluate(rule, frame, store, T0).value == pytest.approx(expected)


def test_non_numeric_column_is_refused(store):
    rule = Rule("r", "metric", "mean", ">", 1)
    with pytest.raises(ValueError, match="No numeric values"):
        evaluate(rule, pd.DataFrame({"metric": ["a", "b"]}), store, T0)
