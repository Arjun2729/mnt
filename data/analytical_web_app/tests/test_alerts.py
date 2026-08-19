"""Alerts are a state machine: notify on transitions, not on every run."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from groundtruth.alerts import AlertStore, Rule, evaluate, evaluate_all, rules_from_json, rules_to_json

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return AlertStore(tmp_path / "state.json")


def frame(value: float) -> pd.DataFrame:
    return pd.DataFrame({"metric": [value] * 10})


def test_first_breach_fires(store):
    rule = Rule("r", "metric", "mean", ">", 100, cooldown_minutes=0)
    assert evaluate(rule, frame(150), store, T0).transition == "fired"


def test_continued_breach_does_not_renotify(store):
    """The MVP's central bug: a breached rule notified on every single run."""
    rule = Rule("r", "metric", "mean", ">", 100, cooldown_minutes=0)
    first = evaluate(rule, frame(150), store, T0)
    second = evaluate(rule, frame(150), store, T0 + timedelta(hours=1))
    assert first.should_notify is True
    assert second.transition == "still_firing"
    assert second.should_notify is False


def test_recovery_notifies_once(store):
    rule = Rule("r", "metric", "mean", ">", 100, cooldown_minutes=0)
    evaluate(rule, frame(150), store, T0)
    recovered = evaluate(rule, frame(50), store, T0 + timedelta(hours=1))
    assert recovered.transition == "recovered"
    assert recovered.should_notify is True
    assert evaluate(rule, frame(50), store, T0 + timedelta(hours=2)).transition == "still_ok"


def test_cooldown_suppresses_flapping(store):
    rule = Rule("r", "metric", "mean", ">", 100, cooldown_minutes=60)
    evaluate(rule, frame(150), store, T0)
    flap = evaluate(rule, frame(50), store, T0 + timedelta(minutes=5))
    assert flap.transition == "suppressed"
    assert flap.should_notify is False


def test_cooldown_expires(store):
    rule = Rule("r", "metric", "mean", ">", 100, cooldown_minutes=30)
    evaluate(rule, frame(150), store, T0)
    later = evaluate(rule, frame(50), store, T0 + timedelta(hours=2))
    assert later.transition == "recovered"


def test_state_survives_a_restart(tmp_path):
    path = tmp_path / "state.json"
    rule = Rule("r", "metric", "mean", ">", 100, cooldown_minutes=0)
    first_store = AlertStore(path)
    evaluate(rule, frame(150), first_store, T0)
    first_store.save()

    # A fresh process reading the same state file must not re-fire.
    second_store = AlertStore(path)
    assert evaluate(rule, frame(150), second_store, T0 + timedelta(hours=1)).transition == "still_firing"


def test_history_accumulates(store):
    rule = Rule("r", "metric", "mean", ">", 100, cooldown_minutes=0)
    evaluate(rule, frame(150), store, T0)
    evaluate(rule, frame(50), store, T0 + timedelta(hours=1))
    assert len(store.history_frame()) == 2


def test_disabled_rules_are_skipped(store):
    rules = [Rule("on", "metric", enabled=True), Rule("off", "metric", enabled=False)]
    assert len(evaluate_all(rules, frame(1), store, T0)) == 1


def test_broken_rule_does_not_abort_the_batch(store):
    rules = [Rule("missing", "nope", "mean", ">", 1), Rule("fine", "metric", "mean", ">", 0, cooldown_minutes=0)]
    results = evaluate_all(rules, frame(5), store, T0)
    assert len(results) == 2
    assert "could not be evaluated" in results[0].message


def test_rules_roundtrip():
    rules = [Rule("a", "metric", "sum", "<", 5), Rule("b", "metric", kind="anomaly", sensitivity=2.5)]
    restored = rules_from_json(rules_to_json(rules))
    assert [r.name for r in restored] == ["a", "b"]
    assert restored[1].kind == "anomaly"


def test_unknown_json_fields_are_ignored():
    restored = rules_from_json('[{"name":"a","column":"metric","from_the_future":true}]')
    assert restored[0].name == "a"
