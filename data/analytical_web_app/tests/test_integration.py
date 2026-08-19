"""One pass through every layer, in the order a user would touch them."""
import pandas as pd

from groundtruth import connectors, insights, ml, stats, timeseries
from groundtruth.agent import ToolBox
from groundtruth.alerts import AlertStore, Rule, evaluate_all
from groundtruth.filters import Condition, Group, compile_tree
from groundtruth.provenance import Provenance
from groundtruth.report import Report, render_html
from groundtruth.semantic import profile
from groundtruth.store import Store


def test_full_workflow(sample_csv, tmp_path):
    lineage = Provenance()
    report = Report("End to end")

    # L0/L1 — land the data.
    store = Store()
    loaded = connectors.load_path(store, str(sample_csv), "sales")
    lineage.record("load", "Loaded sample", path=str(sample_csv), rows=loaded.dataset.rows)
    assert loaded.dataset.rows == 288

    # L2 — roles and grain.
    spec = profile(store, "sales")
    assert spec.time_column == "date" and spec.time_grain == "month"

    # L3 — a nested filter, compiled to SQL.
    tree = Group("AND", [
        Condition("region", "in", ["North", "East"]),
        Group("OR", [Condition("revenue", ">", 150000), Condition("channel", "=", "Paid")]),
    ])
    where, params = compile_tree(tree)
    matching = store.count("sales", where, params)
    lineage.record("filter", "Applied nested filter", where=where, params=params)
    assert 0 < matching < 288

    frame = store.materialize("sales", where, params)

    # L3 — statistics with an effect size.
    correlation = stats.correlation(frame, "revenue", "orders")
    assert correlation.ci is not None
    lineage.record("stat_test", "Correlation", test=correlation.name, result=str(correlation.p_value))

    # L3 — a model that beats its baseline.
    model = ml.train(frame, "revenue", ["region", "channel", "cost", "orders"])
    assert model.metrics["r2"] > model.baseline_metrics["r2"]
    lineage.record("model", "Trained", target="revenue", best_model=model.best_model,
                   metrics=model.metrics, features=["region", "channel", "cost", "orders"])

    # L3 — forecast the filtered slice.
    series = timeseries.aggregate(frame, "date", "revenue", "month", "sum")
    forecast = timeseries.forecast(series, 3, "month")
    assert (forecast.lower <= forecast.mean).all()

    # L4 — the agent's tools run against the same rows.
    store.con.execute(f'CREATE OR REPLACE TABLE "filtered" AS SELECT * FROM "sales" WHERE {where}', params)
    toolbox = ToolBox(store, "filtered", profile(store, "filtered"))
    call = toolbox.dispatch("run_sql", {"query": "SELECT COUNT(*) AS n FROM filtered"})
    assert int(call.dataframe["n"].iloc[0]) == matching

    # L4 — the proactive scan.
    found = insights.scan(store, "filtered", profile(store, "filtered"))
    assert isinstance(found, list)

    # L5 — alerts fire once, then hold.
    alert_store = AlertStore(tmp_path / "state.json")
    rules = [Rule("Revenue", "revenue", "mean", ">", 100, cooldown_minutes=0)]
    first = evaluate_all(rules, frame, alert_store)
    second = evaluate_all(rules, frame, alert_store)
    assert first[0].transition == "fired"
    assert second[0].transition == "still_firing"

    # L5 — the report carries a chart, and the lineage compiles.
    import plotly.express as px

    report.add_metrics(model.metrics, "Model")
    report.add_table(model.importance, "Importance")
    report.add_chart(px.line(x=series.index, y=series.values), "Revenue")
    html = render_html(report, standalone=False)
    assert "plotly" in html.lower()
    compile(lineage.to_script("sales"), "<lineage>", "exec")
    assert len(lineage.events) == 4
