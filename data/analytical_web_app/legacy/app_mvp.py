"""The original MVP Streamlit app, preserved.

Superseded by the Groundtruth app at ../app.py. Kept runnable for reference:

    streamlit run legacy/app_mvp.py

It depends on legacy/analytics_core.py and legacy/ml_core.py.
"""
from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytics_core import (
    SUPPORTED_UPLOAD_TYPES,
    build_ai_context,
    build_html_report,
    coerce_dates,
    correlation_table,
    dataset_profile,
    load_api,
    load_database,
    load_uploaded_file,
    numeric_association,
    to_excel_bytes,
)
from ml_core import train_random_forest


st.set_page_config(page_title="Analytical Web Application", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def cached_api(url: str, headers_json: str, json_path: str) -> pd.DataFrame:
    headers = json.loads(headers_json) if headers_json.strip() else {}
    return coerce_dates(load_api(url, headers=headers, json_path=json_path))


@st.cache_data(show_spinner=False)
def cached_database(connection_url: str, query: str) -> pd.DataFrame:
    return coerce_dates(load_database(connection_url, query))


def set_dataset(df: pd.DataFrame, source_name: str) -> None:
    st.session_state["dataset"] = coerce_dates(df)
    st.session_state["source_name"] = source_name


def get_dataset() -> pd.DataFrame | None:
    return st.session_state.get("dataset")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    with st.sidebar:
        st.header("🔍 Filters")
        filter_cols = st.multiselect("Columns to filter", df.columns.tolist(), key="filter_cols")
        for col in filter_cols:
            s = result[col]
            st.caption(col)
            if pd.api.types.is_numeric_dtype(s):
                clean = s.replace([np.inf, -np.inf], np.nan).dropna()
                if clean.empty:
                    continue
                lo, hi = float(clean.min()), float(clean.max())
                if lo == hi:
                    st.write(f"Constant: {lo}")
                    continue
                selected = st.slider(
                    f"Range — {col}", min_value=lo, max_value=hi, value=(lo, hi), key=f"num_{col}"
                )
                result = result[result[col].between(selected[0], selected[1], inclusive="both")]
            elif pd.api.types.is_datetime64_any_dtype(s):
                clean = s.dropna()
                if clean.empty:
                    continue
                start, end = clean.min().date(), clean.max().date()
                selected = st.date_input(f"Date range — {col}", value=(start, end), key=f"date_{col}")
                if isinstance(selected, tuple) and len(selected) == 2:
                    result = result[result[col].dt.date.between(selected[0], selected[1])]
            else:
                values = s.dropna().astype(str)
                unique = sorted(values.unique().tolist())
                if len(unique) <= 100:
                    chosen = st.multiselect(f"Values — {col}", unique, default=unique, key=f"cat_{col}")
                    result = result[result[col].astype(str).isin(chosen)]
                else:
                    needle = st.text_input(f"Contains — {col}", key=f"txt_{col}")
                    if needle:
                        result = result[result[col].astype(str).str.contains(needle, case=False, na=False)]
        st.metric("Rows after filters", f"{len(result):,}")
    return result


def chart_builder(df: pd.DataFrame) -> None:
    chart_type = st.selectbox("Chart type", ["Histogram", "Bar", "Line", "Scatter", "Box", "Pie", "Correlation heatmap"])
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    fig = None
    if chart_type == "Histogram":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        x = st.selectbox("X", numeric_cols)
        color = st.selectbox("Color/group", ["None"] + [c for c in all_cols if c != x])
        fig = px.histogram(df, x=x, color=None if color == "None" else color, marginal="box")
    elif chart_type == "Bar":
        x = st.selectbox("Category", all_cols)
        aggregation = st.selectbox("Aggregation", ["count", "sum", "mean", "median"])
        if aggregation == "count":
            plot_df = df.groupby(x, dropna=False).size().reset_index(name="value").sort_values("value", ascending=False).head(50)
        else:
            if not numeric_cols:
                st.info("No numeric columns available.")
                return
            y = st.selectbox("Numeric value", numeric_cols)
            plot_df = df.groupby(x, dropna=False)[y].agg(aggregation).reset_index(name="value").sort_values("value", ascending=False).head(50)
        fig = px.bar(plot_df, x=x, y="value")
    elif chart_type == "Line":
        x = st.selectbox("X", all_cols)
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        y = st.selectbox("Y", numeric_cols)
        color = st.selectbox("Series", ["None"] + [c for c in all_cols if c not in {x, y}])
        plot_df = df.sort_values(x)
        fig = px.line(plot_df, x=x, y=y, color=None if color == "None" else color)
    elif chart_type == "Scatter":
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric columns.")
            return
        x = st.selectbox("X", numeric_cols, index=0)
        y = st.selectbox("Y", numeric_cols, index=1)
        color = st.selectbox("Color", ["None"] + [c for c in all_cols if c not in {x, y}])
        fig = px.scatter(df, x=x, y=y, color=None if color == "None" else color, trendline=None)
    elif chart_type == "Box":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        y = st.selectbox("Numeric value", numeric_cols)
        x = st.selectbox("Group", ["None"] + [c for c in all_cols if c != y])
        fig = px.box(df, x=None if x == "None" else x, y=y, points="outliers")
    elif chart_type == "Pie":
        x = st.selectbox("Category", all_cols)
        plot_df = df.groupby(x, dropna=False).size().reset_index(name="count").sort_values("count", ascending=False).head(12)
        fig = px.pie(plot_df, names=x, values="count")
    else:
        corr = correlation_table(df)
        if corr.empty:
            st.info("Need at least two numeric columns.")
            return
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1))
    st.plotly_chart(fig, use_container_width=True)


def ai_analyst(df: pd.DataFrame) -> None:
    st.caption("The AI sees column metadata, summaries, top categories, and up to 30 sample rows—not your entire dataset.")
    key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5.6"))
    question = st.text_area("Ask a question about the filtered data", placeholder="What changed most, and which segments deserve attention?")
    if st.button("Ask AI", type="primary", disabled=not question.strip()):
        if not key:
            st.error("Add an OpenAI API key first.")
            return
        try:
            from openai import OpenAI

            context = build_ai_context(df)
            client = OpenAI(api_key=key)
            response = client.responses.create(
                model=model,
                reasoning={"effort": "low"},
                instructions=(
                    "You are a careful data analyst. Answer only from the supplied dataset context. "
                    "Separate observations from hypotheses. Do not claim to have inspected rows not supplied. "
                    "When useful, show calculations conceptually and mention limitations caused by sampling."
                ),
                input=f"DATASET CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}",
            )
            st.markdown(response.output_text)
        except Exception as exc:
            st.exception(exc)


def statistics_and_ml(df: pd.DataFrame) -> None:
    stats_tab, ml_tab = st.tabs(["Statistics", "Predictive ML"])
    with stats_tab:
        st.subheader("Descriptive statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            st.subheader("Pairwise association")
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("Variable X", numeric_cols, key="stats_x")
            y = c2.selectbox("Variable Y", numeric_cols, index=1, key="stats_y")
            method = c3.selectbox("Method", ["Pearson", "Spearman"])
            if st.button("Run test"):
                try:
                    res = numeric_association(df, x, y, method)
                    a, b, c = st.columns(3)
                    a.metric("Statistic", f"{res.statistic:.4f}")
                    b.metric("p-value", f"{res.p_value:.4g}")
                    c.metric("N", res.n)
                except Exception as exc:
                    st.error(str(exc))

    with ml_tab:
        if df.shape[1] < 2:
            st.info("Need at least two columns.")
            return
        target = st.selectbox("Target", df.columns.tolist())
        candidates = [c for c in df.columns if c != target]
        features = st.multiselect("Features", candidates, default=candidates[: min(8, len(candidates))])
        problem_type = st.selectbox("Problem type", ["Auto", "Classification", "Regression"])
        test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
        st.caption("Model: Random Forest with median imputation for numeric features and one-hot encoding for categorical features.")
        if st.button("Train model", type="primary"):
            try:
                with st.spinner("Training..."):
                    result = train_random_forest(df, target, features, problem_type, test_size=test_size)
                st.success(f"Trained {result.problem_type.lower()} model")
                cols = st.columns(len(result.metrics))
                for col, (name, value) in zip(cols, result.metrics.items()):
                    col.metric(name.upper(), f"{value:.4f}")
                st.subheader("Top feature importances")
                st.dataframe(result.feature_importance.head(30), use_container_width=True)
                st.subheader("Test predictions")
                st.dataframe(result.predictions.head(200), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))


def export_and_alerts(df: pd.DataFrame) -> None:
    st.subheader("📤 Export")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "filtered_data.csv", "text/csv")
    c2.download_button(
        "Download Excel", to_excel_bytes(df), "analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    report = build_html_report(df)
    c3.download_button("Download HTML report", report.encode("utf-8"), "report.html", "text/html")

    st.subheader("🔔 Alert rule builder")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("Alerts require at least one numeric column.")
        return
    name = st.text_input("Rule name", value="KPI threshold")
    col1, col2, col3, col4 = st.columns(4)
    column = col1.selectbox("Column", numeric_cols)
    aggregation = col2.selectbox("Aggregation", ["mean", "sum", "min", "max", "count"])
    operator = col3.selectbox("Operator", [">", ">=", "<", "<=", "=="])
    threshold = col4.number_input("Threshold", value=0.0)

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    current = getattr(series, aggregation)() if not series.empty else np.nan
    fired = {
        ">": current > threshold,
        ">=": current >= threshold,
        "<": current < threshold,
        "<=": current <= threshold,
        "==": current == threshold,
    }[operator]
    st.metric("Current aggregated value", f"{current:,.4f}" if pd.notna(current) else "N/A", delta="ALERT FIRED" if fired else "Not fired")

    rule = [{"name": name, "column": column, "aggregation": aggregation, "operator": operator, "threshold": threshold}]
    st.download_button("Download alerts.json", json.dumps(rule, indent=2), "alerts.json", "application/json")
    st.caption("For automation, run alert_worker.py on a schedule (cron/GitHub Actions/etc.) with ALERT_DATA_URL and optional ALERT_WEBHOOK_URL.")


st.title("📊 Analytical Web Application")
st.caption("Multi-source ingestion • exploratory analysis • interactive visualization • AI analytics • statistics/ML • exports & alerts")

with st.sidebar:
    st.header("📂 Data source")
    source_type = st.radio("Source", ["Upload", "Database", "API"], horizontal=True)

    if source_type == "Upload":
        uploaded = st.file_uploader("CSV / Excel / JSON / Parquet", type=SUPPORTED_UPLOAD_TYPES)
        if uploaded is not None:
            try:
                set_dataset(load_uploaded_file(uploaded), uploaded.name)
            except Exception as exc:
                st.error(str(exc))

    elif source_type == "Database":
        connection_url = st.text_input("SQLAlchemy connection URL", placeholder="sqlite:///example.db")
        query = st.text_area("Read-only SQL", value="SELECT * FROM your_table LIMIT 10000")
        if st.button("Load database"):
            try:
                set_dataset(cached_database(connection_url, query), "Database query")
            except Exception as exc:
                st.error(str(exc))

    else:
        api_url = st.text_input("JSON API URL", placeholder="https://api.example.com/data")
        headers_json = st.text_area("Headers (JSON, optional)", value="{}")
        json_path = st.text_input("JSON path (optional)", placeholder="data.items")
        if st.button("Load API"):
            try:
                set_dataset(cached_api(api_url, headers_json, json_path), api_url)
            except Exception as exc:
                st.error(str(exc))

raw_df = get_dataset()
if raw_df is None:
    st.info("Load a dataset from the sidebar to begin. A sample file is included in the project folder.")
    st.stop()

filtered_df = apply_filters(raw_df)
source_name = st.session_state.get("source_name", "Unknown")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(filtered_df):,}")
m2.metric("Columns", f"{filtered_df.shape[1]:,}")
m3.metric("Missing cells", f"{int(filtered_df.isna().sum().sum()):,}")
m4.metric("Source", source_name[:28])

data_tab, explore_tab, viz_tab, ai_tab, sm_tab, export_tab = st.tabs(
    ["Data", "Explore", "Visualize", "AI Analyst", "Statistics / ML", "Export / Alerts"]
)

with data_tab:
    st.dataframe(filtered_df, use_container_width=True, height=520)

with explore_tab:
    st.subheader("Column profile")
    st.dataframe(dataset_profile(filtered_df), use_container_width=True)
    st.subheader("Numeric correlation")
    corr = correlation_table(filtered_df)
    if corr.empty:
        st.info("Need at least two numeric columns.")
    else:
        st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), use_container_width=True)

with viz_tab:
    chart_builder(filtered_df)

with ai_tab:
    ai_analyst(filtered_df)

with sm_tab:
    statistics_and_ml(filtered_df)

with export_tab:
    export_and_alerts(filtered_df)
