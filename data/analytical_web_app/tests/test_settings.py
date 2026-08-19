"""Methodology thresholds: documented, adjustable, and actually load-bearing."""
import pandas as pd
import pytest

from groundtruth import insights, ml
from groundtruth.semantic import profile
from groundtruth.settings import BY_KEY, DEFAULTS, GROUPS, KNOBS, Settings
from groundtruth.store import Store


def test_every_knob_is_documented():
    for knob in KNOBS:
        assert knob.rationale.strip(), f"{knob.key} has no rationale"
        assert knob.source.strip(), f"{knob.key} does not say where its value came from"
        assert knob.minimum <= knob.default <= knob.maximum


def test_every_knob_maps_to_a_setting():
    settings = Settings()
    for knob in KNOBS:
        assert hasattr(settings, knob.key), f"{knob.key} has no field"
        assert getattr(settings, knob.key) == knob.default


def test_no_orphan_settings():
    """A field with no knob would be an invisible constant again."""
    for name in Settings().to_dict():
        assert name in BY_KEY, f"{name} is adjustable but undocumented"


def test_groups_are_populated():
    assert GROUPS
    for group in GROUPS:
        assert any(k.group == group for k in KNOBS)


def test_conventions_are_labelled_as_such():
    """Values taken from convention should say so rather than imply derivation."""
    alpha = BY_KEY["alpha"]
    assert alpha.is_convention
    assert "arbitrary" in alpha.source.lower()


def test_changes_from_default_are_reportable():
    settings = Settings(correlation_floor=0.9, alpha=0.01)
    changed = settings.changed_from_default()
    assert changed["correlation_floor"] == (0.6, 0.9)
    assert changed["alpha"] == (0.05, 0.01)
    assert "segment_z" not in changed


def test_defaults_report_no_changes():
    assert Settings().changed_from_default() == {}


def test_describe_names_the_rule_and_its_source():
    text = Settings().describe("outlier_iqr")
    assert "IQR" in text and "Tukey" in text


def test_describe_of_an_unknown_key():
    assert Settings().describe("nope") == ""


# ---------------- the knobs must actually change results ----------------


@pytest.fixture
def loaded(store, spec):
    return store, spec


def test_correlation_floor_filters_findings(store, spec):
    lenient = [f for f in insights.scan(store, "sample", spec, settings=Settings(correlation_floor=0.6))
               if f.kind == "correlation"]
    strict = [f for f in insights.scan(store, "sample", spec, settings=Settings(correlation_floor=0.9))
              if f.kind == "correlation"]
    assert len(strict) < len(lenient), "raising the floor must drop weaker correlations"


def test_every_finding_states_its_rule(store, spec):
    for finding in insights.scan(store, "sample", spec):
        assert finding.rule, f"{finding.headline} does not say which rule fired"


def test_impossible_thresholds_silence_findings(store, spec):
    silent = insights.scan(store, "sample", spec, settings=Settings(
        correlation_floor=0.999, segment_z=99, trend_drift=1e6,
        movement_z=99, outlier_iqr=99, missing_warn=99.9))
    assert silent == []


def test_missing_warning_threshold_is_load_bearing():
    frame = pd.DataFrame({"mostly": [1.0] * 70 + [None] * 30, "v": range(100)})
    store = Store()
    store.register_frame("t", frame)
    spec = profile(store, "t")
    lenient = insights.scan(store, "t", spec, settings=Settings(missing_warn=20))
    strict = insights.scan(store, "t", spec, settings=Settings(missing_warn=50))
    assert any("empty" in f.headline for f in lenient)
    assert not any("empty" in f.headline for f in strict)
    store.close()


def test_leakage_cutoff_is_adjustable():
    import numpy as np

    rng = np.random.default_rng(0)
    target = rng.normal(size=120)
    # r lands near 0.96 — above a loose cutoff, below the default.
    frame = pd.DataFrame({"near": target + rng.normal(scale=0.3, size=120), "target": target})
    observed = frame.corr().iloc[0, 1]
    assert 0.9 < observed < 0.995, f"fixture correlation {observed:.4f} is outside the tested band"

    strict = ml.detect_leakage(frame, "target", ["near"], ml.REGRESSION, correlation_cutoff=0.995)
    loose = ml.detect_leakage(frame, "target", ["near"], ml.REGRESSION, correlation_cutoff=0.9)
    assert not strict, "the default cutoff should not flag a merely strong predictor"
    assert loose, "a loose cutoff should flag it"


def test_forest_size_is_adjustable(frame):
    result = ml.train(frame, "revenue", ["cost", "orders"], forest_trees=50, cv_folds=3)
    assert result.leaderboard


def test_session_defaults_cover_every_key_the_app_reads():
    """A session opened before a feature shipped must not break on its new state.

    The app previously guarded initialisation on a single key, so anything added
    later was skipped for sessions already running.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text())

    defaults_dict = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "session":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict) and child.keys:
                    defaults_dict = child
                    break
    assert defaults_dict is not None, "session() no longer declares its defaults as a dict"
    declared = {k.value for k in defaults_dict.keys if isinstance(k, ast.Constant)}

    # Every S.<attr> the app reads must be declared.
    read = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "S":
            read.add(node.attr)
    undeclared = read - declared
    assert not undeclared, f"app reads S.{undeclared} but session() never initialises it"
