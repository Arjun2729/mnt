"""Spring motion for the UI.

Streamlit renders no React tree, so a JavaScript animation library has nothing to
attach to — and its frontend strips `<script>` from injected HTML, so a physics
loop cannot run either. Everything here is therefore CSS.

That is less of a compromise than it sounds. CSS `linear()` takes an arbitrary
list of easing stops, so a real damped-spring solution can be sampled and emitted
directly. The curves below are integrated from the same stiffness/damping/mass
parameterisation animation libraries use, rather than eyeballed with
cubic-bezier — including the overshoot, which is what makes spring motion read as
physical rather than merely eased.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Spring:
    """A damped harmonic oscillator, in the usual animation parameterisation."""

    name: str
    stiffness: float
    damping: float
    mass: float = 1.0

    @property
    def omega(self) -> float:
        """Undamped angular frequency."""
        return math.sqrt(self.stiffness / self.mass)

    @property
    def zeta(self) -> float:
        """Damping ratio. Below 1 overshoots, at 1 settles without overshoot."""
        return self.damping / (2 * math.sqrt(self.stiffness * self.mass))

    def position(self, t: float) -> float:
        """Displacement at time t for a unit step, starting at rest."""
        zeta, omega = self.zeta, self.omega
        if zeta < 1:                                    # underdamped — overshoots
            damped = omega * math.sqrt(1 - zeta * zeta)
            envelope = math.exp(-zeta * omega * t)
            return 1 - envelope * (math.cos(damped * t) + (zeta * omega / damped) * math.sin(damped * t))
        if abs(zeta - 1) < 1e-9:                        # critically damped
            return 1 - math.exp(-omega * t) * (1 + omega * t)
        # Overdamped — two real roots, no oscillation.
        root = omega * math.sqrt(zeta * zeta - 1)
        a, b = -zeta * omega + root, -zeta * omega - root
        return 1 - (a * math.exp(b * t) - b * math.exp(a * t)) / (a - b)

    def settling_time(self, tolerance: float = 0.001, ceiling: float = 6.0) -> float:
        """When the oscillation stays inside the tolerance band for good."""
        step, t, last_outside = 0.005, 0.0, 0.0
        while t < ceiling:
            if abs(self.position(t) - 1) > tolerance:
                last_outside = t
            t += step
        return min(last_outside + step, ceiling)

    def duration_ms(self) -> int:
        return max(120, int(round(self.settling_time() * 1000)))

    def to_linear(self, samples: int = 34) -> str:
        """Emit the solution as a CSS `linear()` easing function."""
        duration = self.settling_time()
        points = []
        for i in range(samples + 1):
            t = duration * i / samples
            points.append(f"{self.position(t):.4f}".rstrip("0").rstrip("."))
        points[-1] = "1"
        return f"linear({', '.join(points)})"


# Named springs, chosen for what each is used on rather than for variety.
SPRINGS: tuple[Spring, ...] = (
    # A little overshoot: tiles and cards arriving feel physical.
    Spring("gentle", stiffness=170, damping=22),
    # Snappier, for things responding directly to a pointer.
    Spring("snappy", stiffness=320, damping=26),
    # No overshoot, for anything showing a value — a number that bounces past
    # its own figure reads as sloppy rather than lively.
    Spring("settle", stiffness=210, damping=30),
)

BY_NAME = {spring.name: spring for spring in SPRINGS}


def easing_tokens() -> str:
    """CSS custom properties: one easing function and duration per spring."""
    lines = []
    for spring in SPRINGS:
        lines.append(f"  --spring-{spring.name}: {spring.to_linear()};")
        lines.append(f"  --spring-{spring.name}-ms: {spring.duration_ms()}ms;")
    return "\n".join(lines)


CSS = """
/* ---------- spring motion ----------
   Easing functions are sampled from a damped-spring solution, so the overshoot
   is the real thing rather than a cubic-bezier that looks a bit like it. */

:root {
__TOKENS__
  --stagger: 55ms;
}

@keyframes gt-enter {
  from { opacity: 0; transform: translateY(10px) scale(0.985); }
  to   { opacity: 1; transform: none; }
}
@keyframes gt-enter-left {
  from { opacity: 0; transform: translateX(-10px); }
  to   { opacity: 1; transform: none; }
}
@keyframes gt-draw { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
@keyframes gt-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

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

/* Metric tiles arrive on a spring, in sequence across the row. */
[data-testid="stMetric"] {
  animation: gt-enter var(--spring-gentle-ms) var(--spring-gentle) both;
}
[data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] { animation-delay: 0ms; }
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { animation-delay: var(--stagger); }
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { animation-delay: calc(var(--stagger) * 2); }
[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] { animation-delay: calc(var(--stagger) * 3); }
[data-testid="stHorizontalBlock"] > div:nth-child(5) [data-testid="stMetric"] { animation-delay: calc(var(--stagger) * 4); }

/* Bordered containers — findings, alert rules, filter groups. */
[data-testid="stVerticalBlockBorderWrapper"] {
  animation: gt-enter var(--spring-gentle-ms) var(--spring-gentle) both;
}

/* Charts fade up, then the plotting area wipes in behind them. */
[data-testid="stPlotlyChart"] {
  animation: gt-enter var(--spring-settle-ms) var(--spring-settle) both;
}
[data-testid="stPlotlyChart"] .cartesianlayer {
  animation: gt-draw 700ms var(--spring-settle) both 120ms;
}

/* Dataframes and the section rule. */
[data-testid="stDataFrame"] { animation: gt-enter var(--spring-settle-ms) var(--spring-settle) both; }
.gt-label::after { transform-origin: left; animation: gt-enter-left 520ms var(--spring-gentle) both; }

/* Pointer response uses the snappier spring. */
.stButton button, .stDownloadButton button, [data-testid="stMetric"] {
  transition: transform var(--spring-snappy-ms) var(--spring-snappy),
              box-shadow 200ms ease, border-color 200ms ease;
}
.stButton button:hover, .stDownloadButton button:hover { transform: translateY(-2px); }
.stButton button:active, .stDownloadButton button:active { transform: translateY(0) scale(0.98); }
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 24px -14px rgba(0,0,0,0.45);
}

/* Long tabs reveal as they scroll into view. No JS: this is a scroll-driven
   timeline, and browsers without it simply show the content. */
@supports (animation-timeline: view()) {
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stPlotlyChart"] {
    animation: gt-enter var(--spring-gentle-ms) var(--spring-gentle) both;
    animation-timeline: view();
    animation-range: entry 0% entry 42%;
  }
}

@media (prefers-reduced-motion: reduce) {
  [data-testid="stMetric"],
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stPlotlyChart"],
  [data-testid="stPlotlyChart"] .cartesianlayer,
  [data-testid="stDataFrame"],
  .gt-label::after {
    animation: none !important;
  }
  .stButton button:hover, .stDownloadButton button:hover, [data-testid="stMetric"]:hover {
    transform: none !important;
  }
  .gt-live::before { animation: none !important; }
}
"""


def css() -> str:
    return CSS.replace("__TOKENS__", easing_tokens())


# Plotly's own transition, applied when a figure's data changes rather than on
# first paint — so changing an aggregation morphs the bars instead of snapping.
CHART_TRANSITION = {"duration": 520, "easing": "cubic-in-out", "ordering": "traces first"}
