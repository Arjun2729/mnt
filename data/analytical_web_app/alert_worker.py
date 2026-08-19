"""Unattended alert runner.

Fetches a dataset, evaluates the rules in alerts.json against it, and notifies
only on transitions — the state that makes that possible lives in a JSON file
between runs, so a metric that stays breached does not re-notify every cycle.

    python alert_worker.py

Exit codes: 0 nothing changed · 10 something fired · 2 configuration error.

Environment:
    ALERT_SOURCE            file | api | database        (default: file)
    ALERT_DATA_PATH         path, when source=file
    ALERT_DATA_URL          url, when source=api
    ALERT_DATA_JSON_PATH    dot path into the JSON body
    GT_ALLOWED_API_HOSTS    comma-separated allowlist, required for source=api
    ALERT_DB_URL            SQLAlchemy url, when source=database
    ALERT_DB_QUERY          read-only SQL, when source=database
    ALERT_RULES_FILE        default alerts.json
    ALERT_STATE_FILE        default alert_state.json
    ALERT_WEBHOOK_URL       optional; receives one POST per transition
    ALERT_TIME_COLUMN       required for anomaly rules
    ALERT_TIME_GRAIN        default month
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from groundtruth.alerts import AlertStore, evaluate_all, notify, rules_from_json
from groundtruth.config import load_env
from groundtruth.connectors import load_api, load_database, load_path
from groundtruth.store import Store


load_env()


def load_dataset(store: Store):
    source = os.getenv("ALERT_SOURCE", "file").lower()

    if source == "file":
        path = os.getenv("ALERT_DATA_PATH", "")
        if not path:
            raise SystemExit("ALERT_DATA_PATH is required when ALERT_SOURCE=file")
        return load_path(store, path, "alert_data")

    if source == "api":
        url = os.getenv("ALERT_DATA_URL", "")
        if not url:
            raise SystemExit("ALERT_DATA_URL is required when ALERT_SOURCE=api")
        return load_api(store, url, json_path=os.getenv("ALERT_DATA_JSON_PATH", ""), name="alert_data")

    if source == "database":
        url, query = os.getenv("ALERT_DB_URL", ""), os.getenv("ALERT_DB_QUERY", "")
        if not (url and query):
            raise SystemExit("ALERT_DB_URL and ALERT_DB_QUERY are required when ALERT_SOURCE=database")
        return load_database(store, url, query, "alert_data")

    raise SystemExit(f"Unknown ALERT_SOURCE: {source}")


def main() -> int:
    rules_file = Path(os.getenv("ALERT_RULES_FILE", "alerts.json"))
    if not rules_file.exists():
        print(f"Rules file not found: {rules_file}", file=sys.stderr)
        return 2

    try:
        rules = rules_from_json(rules_file.read_text())
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"Could not read {rules_file}: {exc}", file=sys.stderr)
        return 2
    if not rules:
        print("No rules defined.", file=sys.stderr)
        return 2

    store = Store()
    try:
        result = load_dataset(store)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Could not load data: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    frame = store.materialize(result.dataset.table)
    alert_store = AlertStore(os.getenv("ALERT_STATE_FILE", "alert_state.json"))

    evaluations = evaluate_all(
        rules, frame, alert_store,
        time_column=os.getenv("ALERT_TIME_COLUMN") or None,
        grain=os.getenv("ALERT_TIME_GRAIN", "month"),
    )
    alert_store.save()

    for evaluation in evaluations:
        marker = {"fired": "FIRED", "recovered": "RECOVERED", "suppressed": "SUPPRESSED"}.get(evaluation.transition, "     ")
        print(f"{marker:10} {evaluation.rule.name}: value={evaluation.value:,.4g} state={evaluation.new_state}")

    webhook = os.getenv("ALERT_WEBHOOK_URL", "")
    if webhook:
        try:
            sent = notify(evaluations, webhook)
            print(f"Sent {sent} notification(s).")
        except Exception as exc:
            print(f"Webhook delivery failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    transitions = sum(1 for e in evaluations if e.transition == "fired")
    print(f"{len(evaluations)} rule(s) evaluated, {transitions} newly firing.")
    return 10 if transitions else 0


if __name__ == "__main__":
    raise SystemExit(main())
