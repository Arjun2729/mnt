"""Groundtruth — an analytical workspace.

Every tab reads through the same DuckDB view, so a filter set means the same
thing to the grid, the charts, the models, the agent and the alerts.
"""
from __future__ import annotations

import html as _html
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from groundtruth import charts, connectors, insights, llm, ml, stats, theme, timeseries
from groundtruth.config import load_env
from groundtruth.agent import ToolBox, ask_stream
from groundtruth.alerts import (
    AGGREGATIONS,
    OPERATORS,
    AlertStore,
    Rule,
    evaluate_all,
    evaluations_frame,
    rules_from_json,
    rules_to_json,
)
from groundtruth.filters import Condition, Group, compile_tree, describe, describe_inline
from groundtruth.provenance import Provenance
from groundtruth.report import Report, render_excel, render_html
from groundtruth.security import SecurityError
from groundtruth.semantic import profile, profile_frame, profile_with_distributions
from groundtruth.store import Store

# Credentials from .env before any provider default is resolved.
load_env()

st.set_page_config(page_title="Groundtruth", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
theme.inject()

FILTERED_VIEW = "filtered_view"


def html_escape(text: str) -> str:
    return _html.escape(str(text))


# ---------------------------------------------------------------- session state


def session():
    if "store" not in st.session_state:
        st.session_state.store = Store()
        st.session_state.provenance = Provenance()
        st.session_state.report = Report("Analysis Report")
        st.session_state.alert_store = AlertStore(os.getenv("GT_ALERT_STATE", "alert_state.json"))
        st.session_state.rules = []
        st.session_state.active = None
        st.session_state.spec = None
        st.session_state.groups = [{"combinator": "AND", "conditions": []}]
        st.session_state.root_combinator = "AND"
        st.session_state.chat = []
        st.session_state.saved_views = {}
        st.session_state.insights = []
    return st.session_state


S = session()


def refresh_spec() -> None:
    if S.active:
        S.spec = profile(S.store, S.store.datasets[S.active].table)


def build_tree() -> Group:
    """Assemble the UI's two-level filter state into a real filter tree."""
    children = []
    for group in S.groups:
        conditions = [Condition(c["column"], c["operator"], c["value"]) for c in group["conditions"]]
        if conditions:
            children.append(Group(group["combinator"], conditions))
    return Group(S.root_combinator, children)


def current_where() -> tuple[str, list]:
    return compile_tree(build_tree())


def filtered_frame(limit: int | None = None) -> pd.DataFrame:
    where, params = current_where()
    return S.store.materialize(S.store.datasets[S.active].table, where, params, limit)


def sync_view() -> str:
    """Expose the filtered slice under a stable name for the agent and the scan."""
    where, params = current_where()
    return S.store.create_filtered_view(FILTERED_VIEW, S.store.datasets[S.active].table, where, params)


def pin(kind: str, **kwargs) -> None:
    getattr(S.report, f"add_{kind}")(**kwargs)
    st.toast(f"Pinned to report ({len(S.report.blocks)} blocks)")


def set_model(provider_name: str, model: str) -> None:
    """Change the model field.

    Streamlit forbids writing to a widget's state once that widget has been
    instantiated during the current run, so this only ever runs as a callback —
    callbacks fire before the rerun builds any widgets.
    """
    st.session_state[f"model_{provider_name}"] = model


def use_suggested_model(provider_name: str) -> None:
    picked = st.session_state.get(f"pick_{provider_name}")
    if picked:
        set_model(provider_name, picked)


def apply_cross_filter(conditions: list[dict], origin: str) -> None:
    """Push a chart selection into the filter tree so every tab follows it."""
    # Fill the first empty group rather than leaving it stranded beside a new one.
    empty = next((g for g in S.groups if not g["conditions"]), None)
    if empty is not None:
        empty["conditions"] = list(conditions)
    else:
        S.groups.append({"combinator": "AND", "conditions": list(conditions)})
    S.provenance.record(
        "filter", f"Cross-filtered from {origin}: {charts.describe_selection(conditions)}",
        where=current_where()[0],
    )
    st.toast(f"Filtered to {charts.describe_selection(conditions)}")


# ---------------------------------------------------------------- sidebar: sources


with st.sidebar:
    st.markdown(
        '<div class="gt-mark" style="font-size:1.15rem;margin-bottom:.15rem">◆ Groundtruth</div>'
        '<div style="font-family:var(--gt-mono);font-size:.66rem;letter-spacing:.09em;'
        'text-transform:uppercase;color:var(--gt-muted);margin-bottom:.9rem">'
        'Traceable analytics</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Data source", expanded=not S.active):
        kind = st.radio("Source", ["Upload", "File path", "Database", "API"], horizontal=True, label_visibility="collapsed")

        try:
            if kind == "Upload":
                uploaded = st.file_uploader("CSV · Excel · JSON · Parquet", type=connectors.SUPPORTED_UPLOAD_TYPES)
                if uploaded and st.button("Load file", type="primary", width="stretch"):
                    result = connectors.load_upload(S.store, uploaded)
                    S.active = result.dataset.name
                    S.provenance.record("load", f"Loaded {uploaded.name} ({result.dataset.rows:,} rows)",
                                        path=uploaded.name, rows=result.dataset.rows)
                    refresh_spec()
                    st.rerun()

            elif kind == "File path":
                path = st.text_input("Path on disk", value="sample_data.csv")
                if st.button("Load path", type="primary", width="stretch"):
                    result = connectors.load_path(S.store, path)
                    S.active = result.dataset.name
                    S.provenance.record("load", f"Loaded {path} ({result.dataset.rows:,} rows)",
                                        path=path, rows=result.dataset.rows)
                    refresh_spec()
                    st.rerun()

            elif kind == "Database":
                url = st.text_input("SQLAlchemy URL", placeholder="sqlite:///analytics.db")
                query = st.text_area("Read-only SQL", value="SELECT * FROM your_table LIMIT 100000", height=110)
                st.caption("Parsed and rejected unless it is a single SELECT. Use a read-only credential too.")
                if st.button("Run query", type="primary", width="stretch"):
                    result = connectors.load_database(S.store, url, query)
                    S.active = result.dataset.name
                    S.provenance.record("load", f"Database query ({result.dataset.rows:,} rows)",
                                        detail=result.dataset.source_detail, rows=result.dataset.rows)
                    refresh_spec()
                    st.rerun()

            else:
                url = st.text_input("JSON API URL", placeholder="https://api.example.com/metrics")
                json_path = st.text_input("JSON path", placeholder="data.items")
                headers_raw = st.text_area("Headers (JSON)", value="{}", height=68)
                allowed = os.getenv("GT_ALLOWED_API_HOSTS", "")
                st.caption(f"Allowlisted hosts: `{allowed or 'none — set GT_ALLOWED_API_HOSTS'}`")
                if st.button("Fetch", type="primary", width="stretch"):
                    result = connectors.load_api(
                        S.store, url, json.loads(headers_raw or "{}"), json_path,
                        name=f"api_{len(S.store.datasets)}",
                    )
                    S.active = result.dataset.name
                    S.provenance.record("load", f"API {url} ({result.dataset.rows:,} rows)",
                                        detail=url, rows=result.dataset.rows)
                    refresh_spec()
                    st.rerun()

        except (SecurityError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    if S.store.datasets:
        names = list(S.store.datasets)
        chosen = st.selectbox("Active dataset", names, index=names.index(S.active) if S.active in names else 0)
        if chosen != S.active:
            S.active = chosen
            S.groups = [{"combinator": "AND", "conditions": []}]
            S.insights = []
            refresh_spec()
            st.rerun()

if not S.active:
    theme.hero()
    columns = st.columns(3, gap="large")
    with columns[0]:
        theme.feature("01 / ASK", "The analyst computes",
                      "It runs real SQL against your data and shows the query behind every number, "
                      "rather than estimating from a sample of rows.")
    with columns[1]:
        theme.feature("02 / FILTER", "Nesting costs nothing",
                      "Conditions compile to parameterised SQL, so (A AND B) OR C is as cheap as a "
                      "single clause — and means the same thing in every tab.")
    with columns[2]:
        theme.feature("03 / TRACE", "Keep the lineage",
                      "Every load, filter, question and model is logged, and exports as a Python "
                      "script that reproduces the session.")
    st.stop()

if S.spec is None:
    refresh_spec()

spec = S.spec
dataset = S.store.datasets[S.active]


# ---------------------------------------------------------------- sidebar: filters


with st.sidebar:
    st.markdown("---")
    theme.label("Filters")
    st.session_state.setdefault("root_comb", S.root_combinator)
    S.root_combinator = st.segmented_control("Join groups with", ["AND", "OR"], key="root_comb") or "AND"

    for group_index, group in enumerate(S.groups):
        with st.container(border=True):
            head, remove = st.columns([3, 1])
            head.markdown(f"**Group {group_index + 1}**")
            if len(S.groups) > 1 and remove.button("✕", key=f"delgrp_{group_index}", help="Remove group"):
                S.groups.pop(group_index)
                st.rerun()
            st.session_state.setdefault(f"comb_{group_index}", group["combinator"])
            group["combinator"] = st.segmented_control(
                "Within group", ["AND", "OR"], key=f"comb_{group_index}",
                label_visibility="collapsed",
            ) or group["combinator"]

            for condition_index, condition in enumerate(group["conditions"]):
                text = f"`{condition['column']}` {condition['operator']} `{condition['value']}`"
                row, drop = st.columns([5, 1])
                row.markdown(text)
                if drop.button("✕", key=f"delcond_{group_index}_{condition_index}"):
                    group["conditions"].pop(condition_index)
                    st.rerun()

            with st.popover("Add condition", width="stretch"):
                column = st.selectbox("Column", [c.name for c in spec.columns], key=f"col_{group_index}")
                column_spec = spec.column(column)
                if column_spec.role == "measure":
                    operators = [">", ">=", "<", "<=", "=", "!=", "between", "is null", "is not null"]
                elif column_spec.role == "time":
                    operators = [">=", "<=", "between", "is null", "is not null"]
                else:
                    operators = ["=", "!=", "in", "not in", "contains", "starts with", "ends with", "is null", "is not null"]
                operator = st.selectbox("Operator", operators, key=f"op_{group_index}")

                value = None
                if operator in ("is null", "is not null"):
                    st.caption("No value needed.")
                elif operator in ("in", "not in"):
                    options = [v for v, _ in column_spec.top_values] or []
                    value = st.multiselect("Values", options, key=f"val_{group_index}")
                elif operator == "between":
                    low = st.text_input("From", key=f"lo_{group_index}")
                    high = st.text_input("To", key=f"hi_{group_index}")
                    value = [low, high]
                elif column_spec.role == "measure":
                    value = st.number_input("Value", value=float(column_spec.mean_value or 0.0), key=f"val_{group_index}")
                else:
                    value = st.text_input("Value", key=f"val_{group_index}")

                if st.button("Add", key=f"add_{group_index}", type="primary"):
                    if column_spec.role == "measure" and operator == "between":
                        value = [float(value[0] or 0), float(value[1] or 0)]
                    group["conditions"].append({"column": column, "operator": operator, "value": value})
                    where_clause, _ = current_where()
                    S.provenance.record("filter", f"Added {column} {operator} {value}", where=where_clause)
                    st.rerun()

    if st.button("Add group", width="stretch"):
        S.groups.append({"combinator": "AND", "conditions": []})
        st.rerun()

    where, params = current_where()
    total = S.store.count(dataset.table)
    matching = S.store.count(dataset.table, where, params)
    st.metric("Rows matching", f"{matching:,}", delta=f"{matching - total:,}" if where else None)

    if where:
        with st.expander("Generated SQL"):
            st.code(f"WHERE {where}\n-- params: {params}", language="sql")
        if st.button("Clear filters", width="stretch"):
            S.groups = [{"combinator": "AND", "conditions": []}]
            st.rerun()

    st.markdown("---")
    name_for_view = st.text_input("Save this filter as", placeholder="Q4 paid channels")
    if st.button("Save view", width="stretch", disabled=not name_for_view):
        S.saved_views[name_for_view] = json.loads(json.dumps(S.groups))
        S.provenance.record("view", f"Saved view {name_for_view!r}")
        st.rerun()
    if S.saved_views:
        pick = st.selectbox("Saved views", ["—", *S.saved_views])
        if pick != "—" and st.button("Apply view", width="stretch"):
            S.groups = json.loads(json.dumps(S.saved_views[pick]))
            st.rerun()


# ---------------------------------------------------------------- header


where, params = current_where()
matching = S.store.count(dataset.table, where, params)

total_rows = S.store.count(dataset.table)

theme.masthead(dataset.name, dataset.source_kind)

a, b, c, d, e = st.columns(5)
a.metric("Rows", f"{matching:,}", delta=f"{matching - total_rows:,} filtered" if where else None)
b.metric("Columns", f"{len(spec.columns):,}")
c.metric("Measures", f"{len(spec.measures):,}")
d.metric("Dimensions", f"{len(spec.dimensions):,}")
e.metric("Time grain", spec.time_grain or "none")

if where:
    badge, clear = st.columns([6, 1], vertical_alignment="center")
    with badge:
        theme.live_badge("cross-filter active")
        theme.filter_summary(describe_inline(build_tree()), matching, total_rows)
    if clear.button("Clear", width="stretch", help="Remove every filter"):
        S.groups = [{"combinator": "AND", "conditions": []}]
        st.rerun()
else:
    st.write("")

# A compact trend strip: one sparkline per measure over the detected time grain.
if spec.time_column and spec.measures:
    with st.expander("Trend at a glance", expanded=False):
        try:
            trend_frame = filtered_frame(limit=200_000)
            grain = spec.time_grain or "month"
            rows = []
            for measure in spec.measures[:6]:
                series = timeseries.aggregate(trend_frame, spec.time_column, measure, grain, "sum")
                if len(series) < 2:
                    continue
                values = series.tolist()
                first, last = values[0], values[-1]
                rows.append({
                    "measure": measure,
                    "trend": values,
                    "latest": last,
                    "change": (last - first) / abs(first) if first else 0.0,
                })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows), width="stretch", hide_index=True,
                    column_config={
                        "measure": st.column_config.TextColumn("Measure", width="medium"),
                        "trend": st.column_config.LineChartColumn(
                            f"Per {grain}", width="large",
                            help=f"Total per {grain} across the filtered rows.",
                        ),
                        "latest": st.column_config.NumberColumn("Latest", format="localized"),
                        "change": st.column_config.NumberColumn("Change", format="percent"),
                    },
                )
            else:
                st.caption("Not enough periods in the current selection to draw a trend.")
        except Exception as exc:
            st.caption(f"Trend unavailable — {exc}")

tabs = st.tabs(
    ["Findings", "Data", "Explore", "Visualize", "Analyst", "Statistics", "Predict", "Time series", "Report", "Alerts", "Lineage"]
)


# ---------------------------------------------------------------- findings


with tabs[0]:
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        theme.label("What stands out")
        st.caption("Runs automatically over the filtered data — no question required.")
    if right.button("Scan now", type="primary", width="stretch"):
        sync_view()
        with st.spinner("Scanning…"):
            scan_spec = profile(S.store, FILTERED_VIEW)
            S.insights = insights.scan(S.store, FILTERED_VIEW, scan_spec)
        S.provenance.record("scan", f"Insight scan surfaced {len(S.insights)} findings")

    if not S.insights:
        st.info("Run a scan to surface trends, segment outliers, correlations and data-quality problems.")
    else:
        counts = {"warning": 0, "notable": 0, "info": 0}
        for item in S.insights:
            counts[item.severity] = counts.get(item.severity, 0) + 1
        summary = st.columns(3)
        summary[0].metric("Needs attention", counts.get("warning", 0))
        summary[1].metric("Notable", counts.get("notable", 0))
        summary[2].metric("Informational", counts.get("info", 0))
        st.write("")

        for index, item in enumerate(S.insights):
            with st.container(border=True):
                theme.finding(item.headline, item.detail, item.kind, item.severity)
                controls = st.columns([1, 1, 8], gap="small")
                show_evidence = controls[0].button("Evidence", key=f"ev_{index}", width="stretch") if item.evidence_sql else False
                if controls[1].button("Pin", key=f"pin_{index}", width="stretch"):
                    pin("text", text=f"{item.headline}\n\n{item.detail}", title="Finding")
                if show_evidence:
                    st.code(item.evidence_sql, language="sql")
                    st.dataframe(S.store.sql_readonly(item.evidence_sql, limit=50), width="stretch")


# ---------------------------------------------------------------- data


with tabs[1]:

    @st.fragment
    def data_panel() -> None:
        controls = st.columns([2, 2, 4])
        page_size = controls[0].select_slider("Rows per page", [50, 100, 250, 500, 1000], value=250)
        pages = max(1, -(-matching // page_size))
        page = controls[1].number_input("Page", 1, pages, 1, help=f"{pages:,} pages") if pages > 1 else 1
        sort_column = controls[2].selectbox("Sort by", ["(none)", *[c.name for c in spec.columns]])
        order = f'"{sort_column}" DESC' if sort_column != "(none)" else ""

        st.dataframe(
            S.store.page(dataset.table, where, params,
                         offset=(page - 1) * page_size, limit=page_size, order_by=order),
            width="stretch", height=540, hide_index=True,
        )
        st.caption(
            f"Page {page:,} of {pages:,} · {matching:,} rows matched · "
            "read through DuckDB, never held in memory"
        )

    data_panel()


# ---------------------------------------------------------------- explore


with tabs[2]:
    theme.label("Column profile")
    st.caption("Roles are inferred once at load and drive chart suggestions, agent grounding and model validation.")
    sync_view()
    view_spec = profile(S.store, FILTERED_VIEW)
    shaped = profile_with_distributions(S.store, view_spec)
    st.dataframe(
        shaped,
        width="stretch",
        hide_index=True,
        column_config={
            "column": st.column_config.TextColumn("Column", width="medium"),
            "role": st.column_config.TextColumn("Role", width="small"),
            "shape": st.column_config.BarChartColumn(
                "Distribution", help="Histogram for measures, category frequencies for dimensions, "
                                     "records per period for time columns.",
                y_min=0, y_max=1, width="medium",
            ),
            "type": st.column_config.TextColumn("SQL type", width="small"),
            "distinct": st.column_config.NumberColumn("Distinct", format="%d"),
            "complete": st.column_config.ProgressColumn("Complete", min_value=0, max_value=1, format="percent"),
            "mean": st.column_config.NumberColumn("Mean", format="%.3f"),
        },
    )
    frame = profile_frame(view_spec)
    if st.button("Pin profile to report"):
        pin("table", frame=frame, title="Column profile")

    if view_spec.key_candidates:
        st.success(f"Candidate keys: {', '.join(view_spec.key_candidates)}")

    theme.label("Derived column")
    st.caption("A SQL expression evaluated over the filtered view — the most common analytical need the MVP had no answer for.")
    new_name = st.text_input("Name", placeholder="margin")
    expression = st.text_input("Expression", placeholder="revenue - cost")
    if st.button("Add column", disabled=not (new_name and expression)):
        try:
            table = dataset.table
            S.store.con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{new_name}" DOUBLE')
            S.store.con.execute(f'UPDATE "{table}" SET "{new_name}" = {expression}')
            S.provenance.record("transform", f"Added column {new_name} = {expression}",
                                sql=f'ALTER TABLE "{table}" ADD COLUMN "{new_name}" DOUBLE; '
                                    f'UPDATE "{table}" SET "{new_name}" = {expression};')
            refresh_spec()
            st.success(f"Added {new_name}")
            st.rerun()
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    numeric_columns = view_spec.measures
    if len(numeric_columns) >= 2:
        theme.label("Correlation scan")
        st.caption("All numeric pairs, ranked, with Benjamini-Hochberg correction for multiple comparisons.")
        data = filtered_frame()
        scan_frame = stats.correlation_scan(data, numeric_columns)
        if not scan_frame.empty:
            st.dataframe(scan_frame, width="stretch")
            if st.button("Pin correlation scan"):
                pin("table", frame=scan_frame, title="Correlation scan")


# ---------------------------------------------------------------- visualize


with tabs[3]:

    @st.fragment
    def visualize_panel() -> None:
        """A fragment: changing chart controls reruns only this panel."""
        suggestions = charts.suggest(spec)

        # Open on a chart that suits the data rather than on empty selectors.
        if suggestions and not st.session_state.get("chart_seeded"):
            first = suggestions[0]
            st.session_state.setdefault("chart_type", first.chart_type)
            st.session_state.setdefault("chart_x", first.x)
            st.session_state.setdefault("chart_y", first.y or "")
            st.session_state.setdefault("chart_color", first.color or "")
            st.session_state["chart_seeded"] = True

        if suggestions:
            theme.label("Suggested from column roles")
            columns = st.columns(min(3, len(suggestions)))
            for index, suggestion in enumerate(suggestions):
                if columns[index % len(columns)].button(
                    f"{suggestion.chart_type}: {suggestion.rationale}", key=f"sug_{index}", width="stretch"
                ):
                    st.session_state.update({
                        "chart_type": suggestion.chart_type, "chart_x": suggestion.x,
                        "chart_y": suggestion.y or "", "chart_color": suggestion.color or "",
                    })
                    st.rerun(scope="fragment")

        all_columns = [c.name for c in spec.columns]
        # The key is seeded above, so passing `default` as well would conflict.
        st.session_state.setdefault("chart_type", "Bar")
        chart_type = st.pills(
            "Chart", ["Bar", "Line", "Area", "Scatter", "Histogram", "Box", "Pie", "Correlation heatmap"],
            key="chart_type",
        ) or "Bar"

        controls = st.columns(4)
        x = controls[0].selectbox("X", ["", *all_columns], key="chart_x")
        y = controls[1].selectbox("Y", ["", *spec.measures], key="chart_y")
        color = controls[2].selectbox("Colour", ["", *spec.dimensions], key="chart_color")
        aggregation = controls[3].selectbox("Aggregate", ["sum", "mean", "median", "min", "max", "count"])

        try:
            data = filtered_frame(limit=200_000)
            if chart_type == "Correlation heatmap":
                chart_title = "Correlation matrix"
            elif chart_type == "Histogram":
                chart_title = f"Distribution of {x}" if x else "Distribution"
            else:
                measure = y or "count"
                chart_title = f"{measure} by {x}" if x else measure
                if color:
                    chart_title += f", split by {color}"

            figure = charts.build(data, chart_type, x, y, color, aggregation, title=chart_title)

            # Temporal x-axes get a scrubber and quick ranges.
            if x and spec.column(x) and spec.column(x).role == "time" and chart_type in ("Line", "Area", "Scatter"):
                charts.add_time_controls(figure, spec.time_grain or "month")

            selectable = chart_type not in ("Correlation heatmap", "Pie")
            event = st.plotly_chart(
                figure, width="stretch", key="viz_chart",
                on_select="rerun" if selectable else "ignore",
                selection_mode=["points", "box"] if selectable else None,
            )

            conditions = []
            if selectable and event and getattr(event, "selection", None):
                conditions = charts.selection_to_conditions(
                    dict(event.selection), chart_type, x, y, color,
                    x_is_measure=bool(x and spec.column(x) and spec.column(x).role == "measure"),
                    y_is_measure=bool(y and spec.column(y) and spec.column(y).role == "measure"),
                )

            actions = st.columns([2, 2, 4])
            if conditions:
                theme.hint(
                    f"Selected <b>{html_escape(charts.describe_selection(conditions))}</b> — "
                    "apply it and every tab follows."
                )
                if actions[0].button("Filter to selection", type="primary", width="stretch"):
                    apply_cross_filter(conditions, f"{chart_type} chart")
                    st.rerun()
            elif selectable:
                theme.hint("Click marks or drag a box on the chart to filter the whole workspace.")

            if actions[1].button("Pin chart to report", width="stretch"):
                pin("chart", figure=figure, title=chart_title)
                S.provenance.record("chart", f"{chart_type} of {y or 'count'} by {x}",
                                    spec=f"{chart_type} x={x} y={y} color={color} agg={aggregation}")
        except Exception as exc:
            st.info(f"Pick columns for this chart type — {exc}")

    visualize_panel()


# ---------------------------------------------------------------- analyst


with tabs[4]:
    theme.label("Analyst")
    st.caption("Given SQL, chart and statistics tools over the filtered view. It computes answers rather than estimating them.")

    # Stay out of the way when the environment already supplies a key — there is
    # nothing to configure, and an open panel pushes the conversation off-screen.
    already_configured = bool(llm.resolve_key(llm.resolve_provider()))
    with st.expander("Model provider", expanded=not S.chat and not already_configured):
        names = list(llm.PROVIDERS)
        # Honour LLM_PROVIDER from the environment on first render; the widget's own
        # state takes over once the user has chosen.
        default_provider = st.session_state.get("provider_name") or llm.resolve_provider().name
        provider_name = st.selectbox(
            "Provider", names,
            index=names.index(default_provider) if default_provider in names else 0,
            format_func=lambda n: f"{n} — free" if llm.PROVIDERS[n].free else n,
            key="provider_name",
        )
        provider = llm.PROVIDERS[provider_name]
        if provider.note:
            (st.success if provider.free else st.info)(provider.note)
        advice = llm.RATE_LIMIT_ADVICE.get(provider_name)
        if advice:
            st.caption(f"**Rate limits.** {advice}")
        st.caption(
            "The analyst spends one request per tool round, so a single question can cost "
            "several requests against a per-minute quota."
        )
        if provider.signup_url:
            st.caption(f"Get a key: {provider.signup_url}")

        setup = st.columns([2, 2])
        base_url = setup[0].text_input(
            "Base URL", value=provider.base_url or "",
            help="The OpenAI-compatible endpoint. Leave as-is unless self-hosting.",
            disabled=provider_name not in ("Custom (OpenAI-compatible)", "Ollama (local)"),
        )
        model_key = f"model_{provider_name}"
        # Only the provider named in the environment inherits LLM_MODEL; the others
        # start from their own default rather than another vendor's model id.
        from_env = provider_name == llm.resolve_provider().name
        st.session_state.setdefault(model_key, llm.resolve_model(provider, use_env=from_env))
        model_name = setup[1].text_input(
            "Model", key=model_key,
            help="Model ids change often — use 'List models' to see what your key can reach.",
        )
        if provider.suggested_models:
            st.pills(
                "Quick pick", provider.suggested_models, key=f"pick_{provider_name}",
                on_change=use_suggested_model, args=(provider_name,),
                help="Verified to support tool calling. Lite models have the most generous free quotas.",
            )

        api_key = st.text_input(
            f"{provider_name} API key", type="password", value=llm.resolve_key(provider),
            help=f"Or set {provider.key_env} in your environment. "
                 f"Keys are per-provider — a key from another provider will be rejected.",
            disabled=not provider.needs_key,
        )

        checks = st.columns(4)
        if checks[0].button("Test connection", width="stretch"):
            ok, message = llm.check_connection(provider, api_key, model_name, base_url)
            (st.success if ok else st.error)(message)
            if not ok:
                replacement = llm.suggest_replacement(message)
                if replacement:
                    st.button(
                        f"Use {replacement} instead", type="primary", key="use_suggested_connect",
                        on_click=set_model, args=(provider_name, replacement),
                    )
        if checks[1].button("Check tool calling", width="stretch"):
            ok, message = llm.supports_tools(provider, api_key, model_name, base_url)
            (st.success if ok else st.warning)(message)
        if checks[2].button("List models", width="stretch"):
            try:
                st.code("\n".join(llm.list_models(provider, api_key, base_url)) or "none returned")
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
        if checks[3].button("Clear chat", width="stretch"):
            S.chat = []
            st.rerun()

    for turn in S.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            for query in turn.get("queries", []):
                with st.expander("Query run"):
                    st.code(query, language="sql")
            for figure in turn.get("figures", []):
                st.plotly_chart(figure, width="stretch")

    question = st.chat_input("Ask about the filtered data…")
    if question:
        if provider.needs_key and not api_key:
            st.error(
                f"Add a {provider_name} API key to use the analyst — "
                f"free from {provider.signup_url}"
            )
        else:
            S.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    view = sync_view()
                    view_spec = profile(S.store, view)
                    toolbox = ToolBox(S.store, view, view_spec)
                    history = [{"role": t["role"], "content": t["content"]} for t in S.chat[:-1]][-8:]

                    status = st.status("Thinking…", expanded=True)
                    # The generator runs inside st.write_stream, so the finished
                    # answer comes back through a container rather than a closure.
                    captured: dict = {}

                    def event_stream():
                        """Drive the agent, surfacing tool calls live and streaming the answer."""
                        client = llm.make_client(provider, api_key, base_url)
                        for kind, payload in ask_stream(
                            client, model_name, question, toolbox, history
                        ):
                            if kind == "tool":
                                if payload.error:
                                    status.write(f"⚠︎ `{payload.name}` — {payload.error[:140]}")
                                else:
                                    detail = payload.arguments.get("purpose") or payload.arguments.get("query", "")
                                    status.write(f"✓ `{payload.name}` — {str(detail)[:110]}")
                            elif kind == "text":
                                yield payload
                            else:
                                captured["answer"] = payload

                    st.write_stream(event_stream())
                    answer = captured.get("answer")
                    status.update(
                        label=f"Answered using {len(answer.tool_calls)} tool call(s)"
                        if answer else "Done",
                        state="complete", expanded=False,
                    )

                    for query in (answer.queries if answer else []):
                        with st.expander("Query run"):
                            st.code(query, language="sql")
                    for figure in (answer.figures if answer else []):
                        st.plotly_chart(figure, width="stretch")

                    if answer:
                        S.chat.append({
                            "role": "assistant", "content": answer.text,
                            "queries": answer.queries, "figures": answer.figures,
                        })
                        S.provenance.record("question", f"Asked: {question}",
                                            question=question, queries=answer.queries)
                        if st.button("Pin answer to report"):
                            pin("text", text=answer.text, title=question)
                except Exception as exc:
                    if llm.is_rate_limited(exc):
                        quota = llm.parse_quota_limit(exc)
                        scope = llm.parse_quota_scope(exc)
                        if scope == "day":
                            st.error(
                                f"Daily free-tier quota exhausted on `{model_name}`"
                                + (f" — {quota} requests per day." if quota else ".")
                                + " Waiting will not help; it resets on the provider's daily cycle."
                            )
                            st.info(
                                "**Switch provider.** A daily budget this small is a few questions "
                                "at most, because each question costs one request per tool round. "
                                "**Groq**'s free tier is far larger, and **Ollama** runs locally "
                                "with no quota at all."
                            )
                        else:
                            wait = llm.parse_retry_delay(exc)
                            st.error(
                                f"Rate limited on `{model_name}`"
                                + (f" — {quota} requests/minute." if quota else ".")
                                + f" Retry in about {wait:.0f}s."
                            )
                            st.info(
                                "**Options:** wait it out · pick a lighter model above · "
                                "switch to **Groq** · or run **Ollama** locally."
                            )
                        st.stop()
                    replacement = llm.suggest_replacement(exc)
                    if replacement:
                        st.error(
                            f"`{model_name}` has been retired by {provider_name}. "
                            f"It suggests **{replacement}**."
                        )
                        st.button(
                            f"Switch to {replacement}", type="primary", key="use_suggested_answer",
                            on_click=set_model, args=(provider_name, replacement),
                        )
                    else:
                        st.error(f"{type(exc).__name__}: {exc}")
                        st.caption(
                            "If this is a model-name problem, use **List models** above to see "
                            "what your key can reach."
                        )


# ---------------------------------------------------------------- statistics


with tabs[5]:
    theme.label("Significance tests")
    st.caption("Every test reports an effect size, and an interval wherever one is defined.")
    data = filtered_frame()

    test_name = st.selectbox(
        "Test",
        ["Correlation", "Compare groups (t-test / ANOVA)", "Chi-square (categorical association)", "Normality"],
    )
    try:
        if test_name == "Correlation":
            columns = st.columns(3)
            x = columns[0].selectbox("X", spec.measures, key="corr_x")
            y = columns[1].selectbox("Y", spec.measures, index=min(1, len(spec.measures) - 1), key="corr_y")
            method = columns[2].selectbox("Method", ["pearson", "spearman"])
            result = stats.correlation(data, x, y, method) if st.button("Run", type="primary") else None
        elif test_name == "Compare groups (t-test / ANOVA)":
            columns = st.columns(2)
            value = columns[0].selectbox("Measure", spec.measures, key="grp_val")
            group = columns[1].selectbox("Group by", spec.dimensions, key="grp_by")
            result = stats.compare_groups(data, value, group) if st.button("Run", type="primary") else None
        elif test_name == "Chi-square (categorical association)":
            columns = st.columns(2)
            x = columns[0].selectbox("First", spec.dimensions, key="chi_x")
            y = columns[1].selectbox("Second", spec.dimensions, index=min(1, len(spec.dimensions) - 1), key="chi_y")
            result = stats.chi_square(data, x, y) if st.button("Run", type="primary") else None
        else:
            column = st.selectbox("Column", spec.measures, key="norm_col")
            result = stats.normality(data[column]) if st.button("Run", type="primary") else None

        if result:
            st.markdown(f"**{result.name}**")
            cells = st.columns(4)
            cells[0].metric("Statistic", f"{result.statistic:,.4f}")
            cells[1].metric("p-value", f"{result.p_value:.4g}")
            cells[2].metric(result.effect_name or "effect",
                            f"{result.effect_size:,.4f}" if result.effect_size is not None else "—")
            cells[3].metric("n", f"{result.n:,}")
            if result.ci:
                st.caption(f"95% CI for the effect: {result.ci[0]:,.4f} to {result.ci[1]:,.4f}")
            st.info(f"Reading: {result.interpretation}")
            if result.detail.get("warning"):
                st.warning(result.detail["warning"])
            with st.expander("Detail"):
                st.json(result.detail)
            S.provenance.record("stat_test", f"{result.name} on this selection",
                                test=result.name, result=f"p={result.p_value:.4g}, effect={result.effect_size}")
            if st.button("Pin result"):
                pin("table", frame=pd.DataFrame([result.as_row()]), title=result.name)
    except Exception as exc:
        st.error(str(exc))


# ---------------------------------------------------------------- predict


with tabs[6]:
    theme.label("Predictive model")
    st.caption("Candidate models compete on cross-validated scores against a naive baseline. Importances are permutation-based.")
    data = filtered_frame()
    all_columns = [c.name for c in spec.columns]

    controls = st.columns([2, 1, 1])
    # Default to a measure: the first column is frequently a timestamp, which makes a nonsense target.
    default_target = spec.measures[0] if spec.measures else all_columns[0]
    target = controls[0].selectbox("Target", all_columns, index=all_columns.index(default_target))
    problem = controls[1].selectbox("Type", ["Auto", "Classification", "Regression"])
    test_size = controls[2].slider("Test share", 0.1, 0.4, 0.2, 0.05)
    candidates = [c for c in all_columns if c != target]
    features = st.multiselect("Features", candidates, default=candidates[: min(8, len(candidates))])

    if st.button("Train", type="primary"):
        try:
            with st.spinner("Cross-validating candidates…"):
                result = ml.train(data, target, features, problem, test_size=test_size)

            for warning in result.leakage_warnings:
                st.warning(f"Possible leakage — {warning}")
            for note in result.notes:
                st.info(note)

            st.success(f"{result.problem_type} · best model: {result.best_model}")
            metric_columns = st.columns(len(result.metrics))
            for column, (name, value) in zip(metric_columns, result.metrics.items()):
                baseline = result.baseline_metrics.get(name)
                delta = f"{value - baseline:+.4f} vs baseline" if baseline is not None else None
                column.metric(name.upper(), f"{value:,.4f}", delta=delta)

            theme.label("Leaderboard")
            leaderboard = ml.leaderboard_frame(result)
            st.dataframe(leaderboard, width="stretch")

            theme.label("Permutation importance")
            st.caption("Measured by shuffling each feature on held-out data — unbiased by cardinality, unlike impurity importance.")
            st.dataframe(result.importance, width="stretch")
            st.plotly_chart(
                charts.build(result.importance.head(15), "Bar", x="feature", y="importance", aggregation="sum"),
                width="stretch",
            )

            theme.label("Predictions")
            st.dataframe(result.predictions.head(200), width="stretch")

            S.provenance.record("model", f"Trained {result.problem_type.lower()} on {target}",
                                target=target, features=features, best_model=result.best_model,
                                metrics={k: round(v, 4) for k, v in result.metrics.items()})
            S.report.add_heading(f"Model — {target}")
            S.report.add_metrics(result.metrics, f"{result.best_model} ({result.problem_type})")
            S.report.add_table(leaderboard, "Model leaderboard")
            S.report.add_table(result.importance, "Permutation importance")
            st.toast("Model summary pinned to report")
        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------------- time series


with tabs[7]:
    if not spec.time_column:
        st.info("No time column detected in this dataset.")
    else:
        theme.label("Time series")
        controls = st.columns(4)
        time_column = controls[0].selectbox("Time", [c.name for c in spec.columns if c.role == "time"])
        measure = controls[1].selectbox("Measure", spec.measures, key="ts_measure")
        grain = controls[2].selectbox(
            "Grain", ["hour", "day", "week", "month", "quarter", "year"],
            index=["hour", "day", "week", "month", "quarter", "year"].index(spec.time_grain or "month"),
        )
        how = controls[3].selectbox("Aggregate", ["sum", "mean", "median", "max", "min"])

        try:
            data = filtered_frame()
            series = timeseries.aggregate(data, time_column, measure, grain, how)
            st.caption(f"{len(series):,} periods · {series.index.min().date()} to {series.index.max().date()}")

            horizon = st.slider("Forecast periods", 1, 24, 6)
            forecast_tab, decompose_tab, anomaly_tab = st.tabs(["Forecast", "Decomposition", "Anomalies"])

            with forecast_tab:
                result = timeseries.forecast(series, horizon, grain)
                figure = charts.forecast_figure(result, f"{measure} — {result.model}")
                st.plotly_chart(figure, width="stretch")
                st.caption(f"Model {result.model} · in-sample MAE {result.in_sample_mae:,.2f}")
                table = pd.DataFrame({
                    "period": result.mean.index, "forecast": result.mean.values.round(2),
                    "low": result.lower.values.round(2), "high": result.upper.values.round(2),
                })
                st.dataframe(table, width="stretch")
                if st.button("Pin forecast"):
                    pin("chart", figure=figure, title=f"{measure} forecast")
                    S.report.add_table(table, "Forecast values")

            with decompose_tab:
                try:
                    decomposition = timeseries.decompose(series, grain)
                    if decomposition.caveat:
                        st.warning(decomposition.caveat)
                    left, right = st.columns(2)
                    left.metric("Seasonal strength", f"{decomposition.seasonal_strength:.3f}")
                    right.metric("Trend strength", f"{decomposition.trend_strength:.3f}")
                    figure = charts.decomposition_figure(decomposition, f"{measure} decomposition")
                    st.plotly_chart(figure, width="stretch")
                    if st.button("Pin decomposition"):
                        pin("chart", figure=figure, title=f"{measure} decomposition")
                except ValueError as exc:
                    st.info(str(exc))

            with anomaly_tab:
                sensitivity = st.slider("Sensitivity (σ)", 1.5, 5.0, 3.0, 0.5)
                found = timeseries.anomalies(series, grain, sensitivity)
                changepoints = timeseries.detect_changepoints(series)
                if found.empty:
                    st.success("No anomalous periods at this sensitivity.")
                else:
                    st.dataframe(found, width="stretch")
                st.caption(
                    f"Changepoints: {', '.join(str(c.date()) for c in changepoints) if changepoints else 'none detected'}"
                )
        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------------- report


with tabs[8]:
    theme.label("Report")
    S.report.title = st.text_input("Title", value=S.report.title)
    S.report.subtitle = st.text_input(
        "Subtitle", value=S.report.subtitle or f"{dataset.name} · {matching:,} rows · {datetime.now().date()}"
    )

    if not S.report.blocks:
        st.info("Pin charts, tables and findings from the other tabs to build a report.")
    else:
        for index, block in enumerate(S.report.blocks):
            with st.container(border=True):
                columns = st.columns([6, 1, 1, 1])
                columns[0].markdown(f"**{index + 1}. {block.kind}** — {block.title or '(untitled)'}")
                if columns[1].button("↑", key=f"up_{index}", disabled=index == 0):
                    S.report.move(index, -1)
                    st.rerun()
                if columns[2].button("↓", key=f"down_{index}", disabled=index == len(S.report.blocks) - 1):
                    S.report.move(index, 1)
                    st.rerun()
                if columns[3].button("✕", key=f"rm_{index}"):
                    S.report.remove(index)
                    st.rerun()

        theme.label("Export")
        standalone = st.checkbox(
            "Self-contained HTML (embeds Plotly, ~4.5MB)", value=True,
            help="Uncheck for a small file that loads Plotly from a CDN and needs internet.",
        )
        html = render_html(S.report, standalone=standalone)
        columns = st.columns(3)
        columns[0].download_button("HTML report", html.encode(), "report.html", "text/html", width="stretch")
        columns[1].download_button("Excel workbook", render_excel(S.report), "report.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   width="stretch")
        columns[2].download_button("Filtered CSV", filtered_frame().to_csv(index=False).encode(),
                                   "filtered.csv", "text/csv", width="stretch")
        with st.expander("Preview"):
            preview_path = Path(tempfile.gettempdir()) / "groundtruth_report_preview.html"
            preview_path.write_text(render_html(S.report, standalone=True), encoding="utf-8")
            st.iframe(preview_path, height=620)


# ---------------------------------------------------------------- alerts


with tabs[9]:
    theme.label("Alerts")
    st.caption("Rules hold state, so a notification fires on the transition into breach — not on every run while it stays breached.")

    with st.expander("New rule", expanded=not S.rules):
        rule_kind = st.radio("Kind", ["threshold", "anomaly"], horizontal=True)
        columns = st.columns(2)
        rule_name = columns[0].text_input("Name", value=f"Rule {len(S.rules) + 1}")
        rule_column = columns[1].selectbox("Column", spec.measures)

        if rule_kind == "threshold":
            controls = st.columns(3)
            aggregation = controls[0].selectbox("Aggregate", AGGREGATIONS)
            operator = controls[1].selectbox("Operator", OPERATORS)
            threshold = controls[2].number_input("Threshold", value=0.0)
            sensitivity = 3.0
        else:
            aggregation, operator, threshold = "mean", ">", 0.0
            sensitivity = st.slider("Sensitivity (σ)", 2.0, 4.0, 3.0, 0.5)
            st.caption("Fires when the latest period falls outside the forecast's prediction interval.")

        cooldown = st.number_input("Cooldown (minutes)", 0, 1440, 60,
                                   help="Suppresses repeat notifications while a metric flaps around its threshold.")
        if st.button("Add rule", type="primary"):
            S.rules.append(Rule(rule_name, rule_column, aggregation, operator, threshold,
                                rule_kind, sensitivity, int(cooldown)))
            S.provenance.record("alert", f"Created rule {rule_name!r}")
            st.rerun()

    if S.rules:
        for index, rule in enumerate(S.rules):
            with st.container(border=True):
                columns = st.columns([5, 1, 1])
                state = S.alert_store.get(rule.name)
                columns[0].markdown(
                    f"**{rule.name}** &nbsp;{theme.pill(state.state)}  \n"
                    f'<span style="font-family:var(--gt-mono);font-size:.75rem;color:var(--gt-muted)">'
                    f"{rule.describe()} · fired {state.fire_count}x</span>",
                    unsafe_allow_html=True,
                )
                rule.enabled = columns[1].toggle("On", value=rule.enabled, key=f"en_{index}")
                if columns[2].button("✕", key=f"delrule_{index}"):
                    S.rules.pop(index)
                    st.rerun()

        columns = st.columns([1, 1, 2])
        if columns[0].button("Evaluate now", type="primary", width="stretch"):
            data = filtered_frame()
            evaluations = evaluate_all(S.rules, data, S.alert_store, time_column=spec.time_column,
                                       grain=spec.time_grain or "month")
            S.alert_store.save()
            st.dataframe(evaluations_frame(evaluations), width="stretch")
            for evaluation in evaluations:
                if evaluation.should_notify:
                    st.error(evaluation.message) if evaluation.new_state == "firing" else st.success(evaluation.message)
                else:
                    st.caption(evaluation.message)
            S.provenance.record("alert", f"Evaluated {len(evaluations)} rules")

        columns[1].download_button("Export rules", rules_to_json(S.rules), "alerts.json",
                                   "application/json", width="stretch")
        uploaded_rules = columns[2].file_uploader("Import rules", type=["json"], label_visibility="collapsed")
        if uploaded_rules:
            S.rules = rules_from_json(uploaded_rules.getvalue().decode())
            st.rerun()

        history = S.alert_store.history_frame()
        if not history.empty:
            with st.expander(f"History ({len(history)} events)"):
                st.dataframe(history.tail(100), width="stretch")

    st.caption("Schedule `python alert_worker.py` to run these rules unattended. Exit code 10 means something fired.")


# ---------------------------------------------------------------- lineage


with tabs[10]:
    theme.label("Lineage")
    st.caption("Every action in this session, exportable as a script that reproduces it.")
    frame = S.provenance.frame()
    if frame.empty:
        st.info("No steps recorded yet.")
    else:
        st.dataframe(frame, width="stretch", height=380)
        script = S.provenance.to_script(dataset.table)
        columns = st.columns(3)
        columns[0].download_button("Python script", script, "reproduce.py", "text/x-python", width="stretch")
        columns[1].download_button("Lineage JSON", S.provenance.to_json(), "lineage.json",
                                   "application/json", width="stretch")
        columns[2].download_button("Markdown", S.provenance.to_markdown(), "lineage.md",
                                   "text/markdown", width="stretch")
        with st.expander("Preview script"):
            st.code(script, language="python")
