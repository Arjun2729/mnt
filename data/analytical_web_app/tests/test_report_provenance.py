"""Delivery: reports that carry their charts, and a lineage that re-runs."""
import pandas as pd
import plotly.express as px
import pytest

from groundtruth.provenance import Provenance
from groundtruth.report import Report, render_excel, render_html


@pytest.fixture
def report(frame):
    aggregated = frame.groupby("region", as_index=False)["revenue"].sum()
    r = Report("Test Report", "subtitle")
    r.add_heading("Section")
    r.add_text("Some prose.")
    r.add_metrics({"Rows": 288, "Revenue": 40219919.4})
    r.add_table(aggregated, "By region")
    r.add_chart(px.bar(aggregated, x="region", y="revenue"), "Chart")
    r.add_code("SELECT 1", "Query")
    return r


def test_charts_reach_the_report(report):
    """The MVP built seven chart types and shipped none of them into its report."""
    html = render_html(report, standalone=True)
    assert "plotly" in html.lower()
    assert "Plotly.newPlot" in html or "data-plotly" in html or "plotly-graph-div" in html


def test_standalone_is_offline_and_cdn_is_small(report):
    standalone = render_html(report, standalone=True)
    cdn = render_html(report, standalone=False)
    assert len(standalone) > 1_000_000     # bundle inlined
    assert len(cdn) < 100_000              # bundle linked
    assert "cdn.plot.ly" in cdn


def test_content_is_escaped():
    r = Report("T")
    r.add_text("<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in render_html(r)
    assert "&lt;script&gt;" in render_html(r)


def test_tables_are_truncated_with_a_note(frame):
    r = Report("T")
    r.add_table(frame, "Big")
    html = render_html(r, max_table_rows=10)
    assert "Showing 10" in html


def test_blocks_reorder_and_remove(report):
    first = report.blocks[0].title
    report.move(0, 1)
    assert report.blocks[1].title == first
    count = len(report.blocks)
    report.remove(0)
    assert len(report.blocks) == count - 1


def test_excel_gets_a_sheet_per_table(report):
    assert len(render_excel(report)) > 4000


def test_excel_handles_a_report_with_no_tables():
    assert len(render_excel(Report("Empty"))) > 0


def test_duplicate_sheet_names_are_disambiguated(frame):
    r = Report("T")
    r.add_table(frame.head(2), "Same")
    r.add_table(frame.head(3), "Same")
    assert len(render_excel(r)) > 0


# ---------------- provenance ----------------


def test_events_are_ordered():
    p = Provenance()
    p.record("load", "one")
    p.record("filter", "two")
    assert list(p.frame()["step"]) == [1, 2]


def test_script_export_is_valid_python():
    p = Provenance()
    p.record("load", "Loaded s.csv", path="s.csv", rows=10)
    p.record("filter", "Filtered", where='"a" > ?', params=[1])
    p.record("query", "Aggregated", sql="SELECT 1")
    compile(p.to_script("dataset"), "<script>", "exec")


def test_script_carries_the_filter_and_query():
    p = Provenance()
    p.record("filter", "Filtered", where='"region" = ?', params=["North"])
    p.record("query", "Ran", sql="SELECT COUNT(*) FROM dataset")
    script = p.to_script("dataset")
    assert '"region" = ?' in script
    assert "SELECT COUNT(*) FROM dataset" in script


def test_json_and_markdown_export():
    p = Provenance()
    p.record("model", "Trained", target="y")
    assert "Trained" in p.to_json()
    assert "Trained" in p.to_markdown()
