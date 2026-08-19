"""Presentation layer for the Streamlit UI.

The palette, radii and semantic colours live in .streamlit/config.toml, which is
the supported path and survives Streamlit upgrades. This module adds what config
cannot express — the webfonts, the density of the layout, and a handful of small
composite components — plus the masthead and stat tiles the app opens with.

Selectors here are limited to Streamlit's stable `data-testid` and `data-baseweb`
hooks. Anything more specific would break on the next release.
"""
from __future__ import annotations

import html

import streamlit as st

FONT_LINK = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=IBM+Plex+Mono:wght@400;500;600&\
family=IBM+Plex+Sans+Condensed:wght@500;600;700&\
family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
"""

CSS = """
<style>
:root {
  --gt-accent: #0B6A72;
  --gt-rule: #D5DEE5;
  --gt-muted: #5C6A76;
  --gt-surface: #FFFFFF;
  --gt-gap: #A65B10;
  --gt-have: #2F6B4F;
  --gt-crit: #A8353B;
  --gt-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --gt-cond: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --gt-accent: #4EC5CD;
    --gt-rule: #25313C;
    --gt-muted: #8695A2;
    --gt-surface: #161F29;
    --gt-gap: #D9944A;
    --gt-have: #6FBF8F;
    --gt-crit: #E0797F;
  }
}

/* ---------- density ---------- */

.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1560px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 0.6rem; }
[data-testid="stSidebarUserContent"] { padding-top: 1.1rem; }
[data-testid="stVerticalBlockBorderWrapper"] > div { gap: 0.55rem; }

/* Tighten the runs of stacked widgets Streamlit spaces generously by default. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }

/* ---------- masthead ---------- */

.gt-masthead {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.55rem 1.1rem;
  border-bottom: 2px solid currentColor; padding-bottom: 0.7rem; margin-bottom: 1.15rem;
}
.gt-mark {
  font-family: var(--gt-cond); font-size: 1.42rem; font-weight: 700;
  letter-spacing: -0.015em; line-height: 1; color: var(--gt-accent);
}
.gt-dataset {
  font-family: var(--gt-cond); font-size: 1.42rem; font-weight: 600;
  letter-spacing: -0.01em; line-height: 1;
}
.gt-source {
  font-family: var(--gt-mono); font-size: 0.68rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--gt-muted); margin-left: auto;
  border: 1px solid var(--gt-rule); border-radius: 3px; padding: 0.18rem 0.5rem;
}

/* ---------- filter summary ---------- */

.gt-filter {
  font-family: var(--gt-mono); font-size: 0.73rem; line-height: 1.7;
  color: var(--gt-muted); border-left: 2px solid var(--gt-accent);
  padding: 0.3rem 0 0.3rem 0.7rem; margin: -0.4rem 0 1rem;
}
.gt-filter b { color: var(--gt-accent); font-weight: 500; }

/* ---------- metrics as tiles ---------- */

[data-testid="stMetric"] {
  background: var(--gt-surface);
  border: 1px solid var(--gt-rule);
  border-left: 3px solid var(--gt-accent);
  border-radius: 4px;
  padding: 0.7rem 0.9rem 0.75rem;
}
[data-testid="stMetricLabel"] p {
  font-family: var(--gt-mono); font-size: 0.66rem !important;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--gt-muted);
}
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
[data-testid="stMetricDelta"] { font-size: 0.76rem; }

/* ---------- tabs ---------- */

.stTabs [data-baseweb="tab-list"] {
  gap: 0.15rem; border-bottom: 1px solid var(--gt-rule); padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--gt-cond); font-size: 0.9rem; font-weight: 600;
  letter-spacing: 0.005em; padding: 0.5rem 0.85rem; border-radius: 4px 4px 0 0;
}
.stTabs [aria-selected="true"] { color: var(--gt-accent); }

/* ---------- section labels ---------- */

.gt-label {
  font-family: var(--gt-mono); font-size: 0.67rem; font-weight: 500;
  letter-spacing: 0.11em; text-transform: uppercase; color: var(--gt-muted);
  display: flex; align-items: center; gap: 0.6rem; margin: 0.3rem 0 0.7rem;
}
.gt-label::after { content: ""; flex: 1; height: 1px; background: var(--gt-rule); }

/* ---------- finding cards ---------- */

.gt-finding { display: flex; gap: 0.7rem; align-items: flex-start; }
.gt-sev {
  flex: none; width: 3px; align-self: stretch; border-radius: 2px; margin-top: 0.15rem;
}
.gt-sev-warning { background: var(--gt-crit); }
.gt-sev-notable { background: var(--gt-gap); }
.gt-sev-info    { background: var(--gt-accent); }
.gt-finding-body { min-width: 0; }
.gt-finding-head { font-weight: 600; line-height: 1.35; margin-bottom: 0.18rem; }
.gt-finding-kind {
  font-family: var(--gt-mono); font-size: 0.62rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--gt-muted); margin-right: 0.5rem;
}
.gt-finding-detail { font-size: 0.84rem; color: var(--gt-muted); line-height: 1.55; }

/* ---------- state pills ---------- */

.gt-pill {
  display: inline-block; font-family: var(--gt-mono); font-size: 0.66rem;
  letter-spacing: 0.07em; text-transform: uppercase; padding: 0.1rem 0.45rem;
  border-radius: 3px; border: 1px solid currentColor; white-space: nowrap;
}
.gt-pill-firing { color: var(--gt-crit); }
.gt-pill-ok     { color: var(--gt-have); }

/* ---------- dataframes and code ---------- */

[data-testid="stDataFrame"] { border-radius: 4px; }
[data-testid="stCodeBlock"] code { font-size: 0.8rem; line-height: 1.55; }
[data-testid="stExpander"] summary { font-size: 0.87rem; }
[data-testid="stCaptionContainer"] p { line-height: 1.55; }

/* ---------- empty-state hero ---------- */

.gt-hero-title {
  font-family: var(--gt-cond); font-size: clamp(2rem, 4.5vw, 3rem); font-weight: 700;
  letter-spacing: -0.02em; line-height: 1.03; margin: 0 0 0.55rem;
}
.gt-hero-sub { font-size: 1.02rem; color: var(--gt-muted); max-width: 60ch; line-height: 1.55; margin: 0; }
.gt-hero-rule { height: 2px; background: currentColor; margin: 1.6rem 0 1.4rem; opacity: 0.9; }
.gt-feature-num {
  font-family: var(--gt-mono); font-size: 0.66rem; letter-spacing: 0.1em;
  color: var(--gt-accent); display: block; margin-bottom: 0.3rem;
}
.gt-feature-head { font-family: var(--gt-cond); font-weight: 600; font-size: 1rem; margin-bottom: 0.2rem; }
.gt-feature-body { font-size: 0.86rem; color: var(--gt-muted); line-height: 1.55; }


/* ---------- motion ---------- */

@keyframes gt-rise {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: none; }
}
@keyframes gt-sweep {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}
@keyframes gt-pulse {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.45; }
}

/* Metric tiles and finding cards arrive in a short stagger. */
[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"] {
  animation: gt-rise 340ms cubic-bezier(.22,.61,.36,1) both;
}
[data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] { animation-delay: 0ms; }
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { animation-delay: 45ms; }
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { animation-delay: 90ms; }
[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] { animation-delay: 135ms; }
[data-testid="stHorizontalBlock"] > div:nth-child(5) [data-testid="stMetric"] { animation-delay: 180ms; }

/* Tiles lift and warm their accent rule on hover. */
[data-testid="stMetric"] {
  transition: transform 160ms ease, box-shadow 160ms ease, border-left-color 160ms ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px -10px rgba(0,0,0,0.32);
  border-left-color: var(--gt-gap);
}

[data-testid="stVerticalBlockBorderWrapper"] { transition: border-color 160ms ease, box-shadow 160ms ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: color-mix(in srgb, var(--gt-accent) 45%, transparent);
  box-shadow: 0 4px 16px -12px rgba(0,0,0,0.4);
}

/* The rule beside a section label draws itself in. */
.gt-label::after { transform-origin: left; animation: gt-sweep 520ms cubic-bezier(.22,.61,.36,1) both; }

/* Buttons and tabs respond to the pointer. */
.stButton button, .stDownloadButton button { transition: transform 120ms ease, box-shadow 120ms ease; }
.stButton button:hover, .stDownloadButton button:hover { transform: translateY(-1px); }
.stButton button:active, .stDownloadButton button:active { transform: translateY(0); }
.stTabs [data-baseweb="tab"] { transition: color 140ms ease, background 140ms ease; }
.stTabs [data-baseweb="tab"]:hover { background: color-mix(in srgb, var(--gt-accent) 9%, transparent); }

/* A live cross-filter reads as active, not decorative. */
.gt-live {
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-family: var(--gt-mono); font-size: 0.66rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--gt-accent);
}
.gt-live::before {
  content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: currentColor; animation: gt-pulse 1.9s ease-in-out infinite;
}

/* Selection hint under an interactive chart. */
.gt-hint {
  font-family: var(--gt-mono); font-size: 0.68rem; letter-spacing: 0.05em;
  color: var(--gt-muted); border: 1px dashed var(--gt-rule);
  border-radius: 4px; padding: 0.4rem 0.65rem; margin-top: 0.4rem;
}
.gt-hint b { color: var(--gt-accent); font-weight: 500; }

@media (prefers-reduced-motion: reduce) {
  [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"], .gt-label::after {
    animation: none !important;
  }
  [data-testid="stMetric"]:hover, .stButton button:hover { transform: none !important; }
}

</style>
"""


def inject() -> None:
    """Load the webfonts and stylesheet. Call once, immediately after set_page_config."""
    st.markdown(FONT_LINK + CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- components


def masthead(dataset_name: str, source_kind: str) -> None:
    st.markdown(
        f'<div class="gt-masthead">'
        f'<span class="gt-mark">◆ Groundtruth</span>'
        f'<span class="gt-dataset">{html.escape(dataset_name)}</span>'
        f'<span class="gt-source">{html.escape(source_kind)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def label(text: str) -> None:
    """A ruled section label — quieter than a heading, louder than a caption."""
    st.markdown(f'<div class="gt-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def filter_summary(description: str, matching: int, total: int) -> None:
    share = f"{matching / total * 100:.0f}%" if total else "0%"
    body = html.escape(description.replace("\n", " · ").replace("  ", " "))
    st.markdown(
        f'<div class="gt-filter"><b>{matching:,}</b> of {total:,} rows ({share}) &nbsp;·&nbsp; {body}</div>',
        unsafe_allow_html=True,
    )


def finding(headline: str, detail: str, kind: str, severity: str) -> None:
    st.markdown(
        f'<div class="gt-finding">'
        f'<div class="gt-sev gt-sev-{html.escape(severity)}"></div>'
        f'<div class="gt-finding-body">'
        f'<div class="gt-finding-head">'
        f'<span class="gt-finding-kind">{html.escape(kind)}</span>{html.escape(headline)}</div>'
        f'<div class="gt-finding-detail">{html.escape(detail)}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def pill(state: str) -> str:
    """Inline HTML for an alert state badge."""
    variant = "firing" if state == "firing" else "ok"
    return f'<span class="gt-pill gt-pill-{variant}">{html.escape(state)}</span>'


def hero() -> None:
    st.markdown(
        '<div class="gt-hero-title">Every number traceable<br>to the query that produced it.</div>'
        '<p class="gt-hero-sub">An analytical workspace on DuckDB. Load a dataset from the sidebar to begin — '
        "<code>sample_data.csv</code> is included and exercises every tab.</p>"
        '<div class="gt-hero-rule"></div>',
        unsafe_allow_html=True,
    )


def feature(number: str, head: str, body: str) -> None:
    st.markdown(
        f'<span class="gt-feature-num">{html.escape(number)}</span>'
        f'<div class="gt-feature-head">{html.escape(head)}</div>'
        f'<div class="gt-feature-body">{html.escape(body)}</div>',
        unsafe_allow_html=True,
    )


def live_badge(text: str) -> None:
    """A pulsing indicator for state that is actively driving the rest of the app."""
    st.markdown(f'<span class="gt-live">{html.escape(text)}</span>', unsafe_allow_html=True)


def hint(text_html: str) -> None:
    """A dashed affordance strip — used to advertise chart selection."""
    st.markdown(f'<div class="gt-hint">{text_html}</div>', unsafe_allow_html=True)
