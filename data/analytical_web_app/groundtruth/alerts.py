"""L5 — alerts as a state machine.

The MVP re-evaluated rules on every run and notified whenever a rule was true,
so a metric that crossed its threshold once notified every run forever. Here a
rule has a state, and a notification is emitted on the *transition* into firing.
Recovery is reported too, and a cooldown stops a metric oscillating around its
threshold from becoming a pager storm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

State = Literal["ok", "firing"]

AGGREGATIONS = ("mean", "sum", "min", "max", "count", "median", "std")
OPERATORS = (">", ">=", "<", "<=", "==", "!=")


@dataclass
class Rule:
    name: str
    column: str
    aggregation: str = "mean"
    operator: str = ">"
    threshold: float = 0.0
    kind: str = "threshold"  # threshold | anomaly
    sensitivity: float = 3.0  # anomaly rules only
    cooldown_minutes: int = 60
    enabled: bool = True
    filter_tree: dict | None = None

    def describe(self) -> str:
        if self.kind == "anomaly":
            return f"{self.column} deviates beyond {self.sensitivity}σ of its forecast"
        return f"{self.aggregation}({self.column}) {self.operator} {self.threshold:,.4g}"


@dataclass
class Evaluation:
    rule: Rule
    value: float
    breached: bool
    previous_state: State
    new_state: State
    transition: str  # fired | recovered | still_firing | still_ok | suppressed
    message: str = ""
    expected: tuple[float, float] | None = None

    @property
    def should_notify(self) -> bool:
        return self.transition in ("fired", "recovered")


@dataclass
class RuleState:
    state: State = "ok"
    since: str = ""
    last_notified: str = ""
    last_value: float | None = None
    fire_count: int = 0


class AlertStore:
    """Persists rule state between runs — the thing that makes transitions knowable."""

    def __init__(self, path: str | Path = "alert_state.json") -> None:
        self.path = Path(path)
        self.states: dict[str, RuleState] = {}
        self.history: list[dict] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self.states = {k: RuleState(**v) for k, v in payload.get("states", {}).items()}
        self.history = payload.get("history", [])

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {"states": {k: asdict(v) for k, v in self.states.items()}, "history": self.history[-500:]},
                indent=2,
                default=str,
            )
        )

    def get(self, rule_name: str) -> RuleState:
        return self.states.setdefault(rule_name, RuleState())

    def record(self, evaluation: Evaluation, now: datetime) -> None:
        state = self.get(evaluation.rule.name)
        if evaluation.new_state != state.state:
            state.since = now.isoformat(timespec="seconds")
        state.state = evaluation.new_state
        state.last_value = evaluation.value
        if evaluation.transition == "fired":
            state.fire_count += 1
        if evaluation.should_notify:
            state.last_notified = now.isoformat(timespec="seconds")
        self.history.append(
            {
                "at": now.isoformat(timespec="seconds"),
                "rule": evaluation.rule.name,
                "value": evaluation.value,
                "state": evaluation.new_state,
                "transition": evaluation.transition,
            }
        )

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def _aggregate(series: pd.Series, aggregation: str) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        raise ValueError("No numeric values available for this rule.")
    if aggregation == "count":
        return float(len(numeric))
    return float(getattr(numeric, aggregation)())


def _compare(value: float, operator: str, threshold: float) -> bool:
    return {
        ">": value > threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        "<=": value <= threshold,
        "==": value == threshold,
        "!=": value != threshold,
    }[operator]


def evaluate(
    rule: Rule,
    df: pd.DataFrame,
    store: AlertStore,
    now: datetime | None = None,
    time_column: str | None = None,
    grain: str = "month",
) -> Evaluation:
    """Evaluate one rule and resolve the transition against its stored state."""
    now = now or datetime.now(timezone.utc)
    previous = store.get(rule.name)
    expected: tuple[float, float] | None = None

    if rule.kind == "anomaly":
        if not time_column:
            raise ValueError("Anomaly rules need a time column.")
        from .timeseries import aggregate as ts_aggregate, forecast

        series = ts_aggregate(df, time_column, rule.column, grain, "sum")
        if len(series) < 8:
            raise ValueError(f"Need at least 8 periods for an anomaly rule (have {len(series)}).")
        # Fit on all but the last point, then judge that point against the interval.
        prediction = forecast(series.iloc[:-1], periods=1, grain=grain, alpha=0.003 if rule.sensitivity >= 3 else 0.05)
        value = float(series.iloc[-1])
        low, high = float(prediction.lower.iloc[0]), float(prediction.upper.iloc[0])
        expected = (low, high)
        breached = value < low or value > high
        detail = f"{value:,.4g} against an expected {low:,.4g} to {high:,.4g}"
    else:
        value = _aggregate(df[rule.column], rule.aggregation)
        breached = _compare(value, rule.operator, rule.threshold)
        detail = f"{rule.aggregation}({rule.column}) = {value:,.4g}, threshold {rule.operator} {rule.threshold:,.4g}"

    new_state: State = "firing" if breached else "ok"

    # Transition resolution — the whole point of keeping state.
    if new_state == previous.state:
        transition = "still_firing" if new_state == "firing" else "still_ok"
    elif new_state == "firing":
        transition = "fired"
    else:
        transition = "recovered"

    # A rule that flaps around its threshold should not notify on every crossing.
    if transition in ("fired", "recovered") and previous.last_notified and rule.cooldown_minutes > 0:
        try:
            last = datetime.fromisoformat(previous.last_notified)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if now - last < timedelta(minutes=rule.cooldown_minutes):
                transition = "suppressed"
        except ValueError:
            pass

    messages = {
        "fired": f"ALERT — {rule.name} started firing. {detail}.",
        "recovered": f"RESOLVED — {rule.name} returned to normal. {detail}.",
        "still_firing": f"{rule.name} remains firing (no new notification). {detail}.",
        "still_ok": f"{rule.name} is within bounds. {detail}.",
        "suppressed": f"{rule.name} changed state but is inside its {rule.cooldown_minutes}m cooldown. {detail}.",
    }

    evaluation = Evaluation(rule, value, breached, previous.state, new_state, transition, messages[transition], expected)
    store.record(evaluation, now)
    return evaluation


def evaluate_all(
    rules: list[Rule],
    df: pd.DataFrame,
    store: AlertStore,
    now: datetime | None = None,
    time_column: str | None = None,
    grain: str = "month",
) -> list[Evaluation]:
    results: list[Evaluation] = []
    for rule in rules:
        if not rule.enabled:
            continue
        try:
            results.append(evaluate(rule, df, store, now, time_column, grain))
        except Exception as exc:
            results.append(
                Evaluation(rule, float("nan"), False, store.get(rule.name).state, store.get(rule.name).state,
                           "still_ok", f"{rule.name} could not be evaluated: {exc}")
            )
    return results


def notify(evaluations: list[Evaluation], webhook_url: str, timeout: int = 20) -> int:
    """POST only the transitions. Returns the number of notifications sent."""
    import requests

    sent = 0
    for evaluation in evaluations:
        if not evaluation.should_notify:
            continue
        response = requests.post(
            webhook_url,
            json={
                "text": evaluation.message,
                "rule": evaluation.rule.name,
                "state": evaluation.new_state,
                "value": evaluation.value,
                "transition": evaluation.transition,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        sent += 1
    return sent


def rules_to_json(rules: list[Rule]) -> str:
    return json.dumps([asdict(r) for r in rules], indent=2)


def rules_from_json(payload: str) -> list[Rule]:
    data = json.loads(payload)
    known = {f for f in Rule.__dataclass_fields__}
    return [Rule(**{k: v for k, v in item.items() if k in known}) for item in data]


def evaluations_frame(evaluations: list[Evaluation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule": e.rule.name,
                "condition": e.rule.describe(),
                "value": round(e.value, 4) if pd.notna(e.value) else None,
                "state": e.new_state,
                "transition": e.transition,
                "notify": e.should_notify,
            }
            for e in evaluations
        ]
    )
