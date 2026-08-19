"""L5 — the report builder.

Blocks are pinned from anywhere in the app and exported as one self-contained
HTML file. Charts are embedded as live Plotly figures rather than dropped, which
was the MVP's disconnect: it built seven chart types and shipped none of them.
"""
from __future__ import annotations

import html
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class Block:
    kind: str  # heading | text | table | chart | metrics | code
    title: str = ""
    body: Any = None
    note: str = ""


@dataclass
class Report:
    title: str = "Analysis Report"
    subtitle: str = ""
    blocks: list[Block] = field(default_factory=list)

    def add(self, block: Block) -> "Report":
        self.blocks.append(block)
        return self

    def add_heading(self, text: str) -> "Report":
        return self.add(Block("heading", title=text))

    def add_text(self, text: str, title: str = "") -> "Report":
        return self.add(Block("text", title=title, body=text))

    def add_table(self, frame: pd.DataFrame, title: str = "", note: str = "") -> "Report":
        return self.add(Block("table", title=title, body=frame.copy(), note=note))

    def add_chart(self, figure: Any, title: str = "", note: str = "") -> "Report":
        return self.add(Block("chart", title=title, body=figure, note=note))

    def add_metrics(self, metrics: dict[str, Any], title: str = "") -> "Report":
        return self.add(Block("metrics", title=title, body=dict(metrics)))

    def add_code(self, code: str, title: str = "") -> "Report":
        return self.add(Block("code", title=title, body=code))

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.blocks):
            self.blocks.pop(index)

    def move(self, index: int, delta: int) -> None:
        target = index + delta
        if 0 <= index < len(self.blocks) and 0 <= target < len(self.blocks):
            self.blocks[index], self.blocks[target] = self.blocks[target], self.blocks[index]


_CSS = """
:root{--ink:#12191f;--muted:#5b6874;--rule:#dde3e9;--bg:#ffffff;--soft:#f4f7f9;--accent:#0b6a72}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 96px}
header{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:36px}
h1{font-size:2rem;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);margin:0;font-size:.95rem}
h2{font-size:1.25rem;margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
h3{font-size:1rem;margin:30px 0 10px;color:var(--ink)}
p{color:#3d4a56}
.note{color:var(--muted);font-size:.85rem;margin:8px 0 0;font-style:italic}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:5px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:.87rem}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
th{background:var(--soft);font-weight:600;color:var(--muted);
  font-size:.72rem;letter-spacing:.06em;text-transform:uppercase}
tbody tr:last-child td{border-bottom:0}
td{font-variant-numeric:tabular-nums}
.metrics{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.metric{border:1px solid var(--rule);border-left:3px solid var(--accent);border-radius:5px;
  padding:12px 18px;min-width:150px;background:var(--soft)}
.metric .k{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.metric .v{font-size:1.4rem;font-weight:600;font-variant-numeric:tabular-nums;margin-top:3px}
pre{background:var(--soft);border:1px solid var(--rule);border-radius:5px;padding:14px;
  overflow-x:auto;font-size:.82rem;line-height:1.5}
.chart{margin:16px 0}
footer{margin-top:64px;border-top:1px solid var(--rule);padding-top:14px;
  color:var(--muted);font-size:.78rem}
@media print{.wrap{max-width:none;padding:0}h2{page-break-after:avoid}.chart{page-break-inside:avoid}}
"""


def _format_frame(frame: pd.DataFrame, max_rows: int) -> str:
    shown = frame.head(max_rows)
    table = shown.to_html(index=False, escape=True, border=0, float_format=lambda v: f"{v:,.4g}")
    suffix = ""
    if len(frame) > max_rows:
        suffix = f'<p class="note">Showing {max_rows:,} of {len(frame):,} rows.</p>'
    return f'<div class="scroll">{table}</div>{suffix}'


def render_html(report: Report, max_table_rows: int = 100, standalone: bool = True) -> str:
    """Render the report to one HTML file.

    standalone=True inlines the Plotly bundle once, so the file opens with no
    network at all — at roughly 4.5MB. standalone=False links the CDN instead,
    giving a file in the tens of KB that needs internet to draw its charts.
    """
    parts: list[str] = []
    plotly_included = False

    for block in report.blocks:
        title = html.escape(block.title) if block.title else ""

        if block.kind == "heading":
            parts.append(f"<h2>{title}</h2>")
            continue
        if title:
            parts.append(f"<h3>{title}</h3>")

        if block.kind == "text":
            for paragraph in str(block.body).split("\n\n"):
                if paragraph.strip():
                    parts.append(f"<p>{html.escape(paragraph.strip())}</p>")

        elif block.kind == "table" and isinstance(block.body, pd.DataFrame):
            parts.append(_format_frame(block.body, max_table_rows))

        elif block.kind == "metrics" and isinstance(block.body, dict):
            cells = "".join(
                f'<div class="metric"><div class="k">{html.escape(str(k))}</div>'
                f'<div class="v">{html.escape(f"{v:,.4g}" if isinstance(v, (int, float)) else str(v))}</div></div>'
                for k, v in block.body.items()
            )
            parts.append(f'<div class="metrics">{cells}</div>')

        elif block.kind == "code":
            parts.append(f"<pre><code>{html.escape(str(block.body))}</code></pre>")

        elif block.kind == "chart" and block.body is not None:
            # Inline the plotly bundle on the first chart only; later charts reuse it.
            if standalone:
                include = not plotly_included
            else:
                include = "cdn" if not plotly_included else False
            fragment = block.body.to_html(
                full_html=False,
                include_plotlyjs=include,
                default_height=420,
                config={"displaylogo": False, "responsive": True},
            )
            plotly_included = True
            parts.append(f'<div class="chart">{fragment}</div>')

        if block.note:
            parts.append(f'<p class="note">{html.escape(block.note)}</p>')

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle = f'<p class="sub">{html.escape(report.subtitle)}</p>' if report.subtitle else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(report.title)}</title><style>{_CSS}</style></head><body><div class='wrap'>"
        f"<header><h1>{html.escape(report.title)}</h1>{subtitle}</header>"
        + "".join(parts)
        + f"<footer>Generated by Groundtruth on {generated} · {len(report.blocks)} blocks</footer>"
        "</div></body></html>"
    )


def render_excel(report: Report) -> bytes:
    """Every tabular block as its own worksheet."""
    buffer = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        wrote_any = False
        for index, block in enumerate(report.blocks):
            if block.kind == "table" and isinstance(block.body, pd.DataFrame):
                name = (block.title or f"Table {index + 1}")[:28]
                candidate, suffix = name, 2
                while candidate.lower() in used:
                    candidate, suffix = f"{name[:26]}_{suffix}", suffix + 1
                used.add(candidate.lower())
                block.body.to_excel(writer, sheet_name=candidate, index=False)
                wrote_any = True
            elif block.kind == "metrics" and isinstance(block.body, dict):
                name = (block.title or f"Metrics {index + 1}")[:28]
                if name.lower() not in used:
                    used.add(name.lower())
                    pd.DataFrame([block.body]).to_excel(writer, sheet_name=name, index=False)
                    wrote_any = True
        if not wrote_any:
            pd.DataFrame({"note": ["This report contains no tabular blocks."]}).to_excel(
                writer, sheet_name="Report", index=False
            )
    return buffer.getvalue()
