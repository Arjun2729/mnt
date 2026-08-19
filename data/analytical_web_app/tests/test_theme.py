"""Presentation helpers: escaping and structure, without a running app."""
import re

import pytest
import streamlit as st

from groundtruth import theme


@pytest.fixture
def captured(monkeypatch):
    """Collect markdown emitted by the theme helpers."""
    written: list[str] = []
    monkeypatch.setattr(st, "markdown", lambda body, **kwargs: written.append(body))
    return written


def test_inject_loads_fonts_and_styles(captured):
    theme.inject()
    body = captured[0]
    assert "fonts.googleapis.com" in body
    assert "<style>" in body and "</style>" in body


def test_masthead_shows_the_dataset(captured):
    theme.masthead("Q3 sales", "file")
    assert "Q3 sales" in captured[0] and "file" in captured[0]


def test_masthead_escapes_markup(captured):
    theme.masthead("<script>alert(1)</script>", "file")
    assert "<script>" not in captured[0]
    assert "&lt;script&gt;" in captured[0]


def test_label_escapes(captured):
    theme.label("A & B")
    assert "&amp;" in captured[0]


def test_filter_summary_reports_the_share(captured):
    theme.filter_summary("region in East", 72, 288)
    body = captured[0]
    assert "72" in body and "288" in body and "25%" in body


def test_filter_summary_handles_an_empty_dataset(captured):
    theme.filter_summary("none", 0, 0)
    assert "0%" in captured[0]


@pytest.mark.parametrize("severity", ["warning", "notable", "info"])
def test_finding_carries_its_severity(captured, severity):
    theme.finding("head", "detail", "trend", severity)
    assert f"gt-sev-{severity}" in captured[0]


def test_finding_escapes_content(captured):
    theme.finding("<b>h</b>", "<i>d</i>", "kind", "info")
    assert "<b>h</b>" not in captured[0] and "&lt;b&gt;" in captured[0]


def test_pill_variants():
    assert "gt-pill-firing" in theme.pill("firing")
    assert "gt-pill-ok" in theme.pill("ok")


def test_pill_escapes():
    assert "<script>" not in theme.pill("<script>")


def test_hero_and_feature(captured):
    theme.hero()
    theme.feature("01 / ASK", "Head", "Body text")
    assert "traceable" in captured[0].lower()
    assert "Head" in captured[1] and "Body text" in captured[1]


def test_feature_escapes(captured):
    theme.feature("01", "<h1>", "<p>")
    assert "<h1>" not in captured[0]
    assert "&lt;h1&gt;" in captured[0]


def test_live_badge(captured):
    theme.live_badge("cross-filter active")
    assert "gt-live" in captured[0] and "cross-filter active" in captured[0]


def test_hint_allows_intentional_markup(captured):
    """The hint takes pre-escaped HTML so callers can emphasise a value."""
    theme.hint("Selected <b>region = East</b>")
    assert "<b>region = East</b>" in captured[0]


def test_stylesheet_defines_both_themes():
    assert "prefers-color-scheme: dark" in theme.CSS
    assert "prefers-reduced-motion" in theme.CSS


def test_every_custom_class_used_by_helpers_is_defined(captured):
    """A helper emitting a class the stylesheet lacks would render unstyled."""
    theme.masthead("d", "s")
    theme.label("l")
    theme.filter_summary("f", 1, 2)
    theme.finding("h", "d", "k", "info")
    theme.hero()
    theme.feature("n", "h", "b")
    theme.live_badge("x")
    theme.hint("y")
    emitted = " ".join(captured) + theme.pill("ok")
    used = set(re.findall(r'class="(gt-[\w-]+)"', emitted))
    for name in used:
        assert f".{name}" in theme.CSS, f"{name} is emitted but never styled"


def test_home_link_targets_the_start_screen(captured):
    """The wordmark is a real link, so it creates browser history."""
    theme.home_link()
    assert '?view=home' in captured[0]
    assert 'target="_self"' in captured[0]
    assert "gt-home" in captured[0]


def test_methodology_note_states_the_rule(captured):
    theme.methodology_note("Rule: |r| >= 0.6")
    assert "gt-rule" in captured[0]
    assert "|r| &gt;= 0.6" in captured[0]      # escaped, not injected
