"""Spring easing: the curves must be physically real, not decorative."""
import re

import pytest

from groundtruth import motion
from groundtruth.motion import BY_NAME, SPRINGS, Spring


def test_every_spring_is_named_and_distinct():
    assert len({s.name for s in SPRINGS}) == len(SPRINGS)
    assert set(BY_NAME) == {s.name for s in SPRINGS}


def test_underdamped_springs_actually_overshoot():
    """Overshoot is what separates spring motion from ordinary easing."""
    gentle = BY_NAME["gentle"]
    assert gentle.zeta < 1
    peak = max(gentle.position(t / 1000) for t in range(0, 1500, 5))
    assert peak > 1.0, "an underdamped spring must exceed its target"


def test_value_bearing_motion_does_not_overshoot():
    """A number that bounces past its own figure reads as sloppy."""
    settle = BY_NAME["settle"]
    assert settle.zeta >= 1
    peak = max(settle.position(t / 1000) for t in range(0, 1500, 5))
    assert peak <= 1.0001


def test_springs_start_at_rest_and_converge():
    for spring in SPRINGS:
        assert abs(spring.position(0)) < 1e-9, f"{spring.name} does not start at rest"
        assert abs(spring.position(spring.settling_time() * 2) - 1) < 0.01


@pytest.mark.parametrize("zeta_case", [
    Spring("critical", stiffness=100, damping=20),          # zeta == 1 exactly
    Spring("over", stiffness=100, damping=60),              # zeta > 1
    Spring("under", stiffness=100, damping=4),              # zeta << 1
])
def test_all_damping_regimes_solve(zeta_case):
    assert abs(zeta_case.position(0)) < 1e-9
    assert abs(zeta_case.position(zeta_case.settling_time() * 2) - 1) < 0.02


def test_critically_damped_branch_is_exercised():
    critical = Spring("c", stiffness=100, damping=20)
    assert abs(critical.zeta - 1) < 1e-9
    assert 0 < critical.position(0.1) < 1


def test_linear_easing_is_well_formed():
    for spring in SPRINGS:
        css = spring.to_linear()
        assert css.startswith("linear(") and css.endswith(")")
        stops = css[len("linear("):-1].split(", ")
        assert len(stops) > 10, "too few stops to describe a spring"
        assert stops[0] == "0" and stops[-1] == "1", "must run from rest to target"
        for stop in stops:
            float(stop)


def test_durations_are_sane():
    for spring in SPRINGS:
        assert 120 <= spring.duration_ms() <= 2000


def test_settling_time_is_bounded_for_a_barely_damped_spring():
    """A spring that rings for ages must still yield a finite duration."""
    ringing = Spring("ringing", stiffness=400, damping=1)
    assert ringing.settling_time() <= 6.0


def test_tokens_cover_every_spring():
    tokens = motion.easing_tokens()
    for spring in SPRINGS:
        assert f"--spring-{spring.name}:" in tokens
        assert f"--spring-{spring.name}-ms:" in tokens


def test_css_substitutes_its_tokens():
    css = motion.css()
    assert "__TOKENS__" not in css
    assert css.count("linear(") >= len(SPRINGS)


def test_motion_respects_the_reduced_motion_preference():
    css = motion.css()
    assert "prefers-reduced-motion" in css
    guard = css[css.index("prefers-reduced-motion"):]
    for selector in ("stMetric", "stPlotlyChart", "stVerticalBlockBorderWrapper"):
        assert selector in guard, f"{selector} keeps animating when motion is reduced"


def test_scroll_reveal_is_feature_detected():
    """Browsers without scroll timelines must still show the content."""
    assert "@supports (animation-timeline: view())" in motion.css()


def test_chart_transition_is_configured():
    assert motion.CHART_TRANSITION["duration"] > 0
    assert motion.CHART_TRANSITION["easing"]


def test_charts_carry_the_transition():
    import pandas as pd

    from groundtruth import charts

    frame = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})
    figure = charts.build(frame, "Bar", x="a", y="b")
    assert figure.layout.transition.duration == motion.CHART_TRANSITION["duration"]
