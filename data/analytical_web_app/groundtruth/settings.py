"""Analysis methodology — every threshold this app applies, in one place.

Any analytics tool needs defaults. The problem is not that they exist, it is that
buried constants let a tool look more adaptive than it is: a "finding" is really
a rule firing, and if the rule is invisible the user cannot judge the finding.

So every cutoff lives here, each with the reason it holds that value and where it
came from. They are adjustable at runtime, and the UI shows which rule produced
each result. When a number is a convention rather than a derivation, the note
says so.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass(frozen=True)
class Knob:
    """One tunable threshold, with its justification."""

    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    group: str
    rationale: str
    source: str = ""

    @property
    def is_convention(self) -> bool:
        return "convention" in self.source.lower() or "textbook" in self.source.lower()


KNOBS: tuple[Knob, ...] = (
    # ---- what counts as a finding -------------------------------------------------
    Knob("correlation_floor", "Correlation worth reporting", 0.6, 0.1, 0.95, 0.05, "Findings",
         "Below this, |r| explains under 36% of shared variation and tends to be noise in "
         "small samples. Raise it to see only strong relationships.",
         "Convention. No principled cutoff exists; it trades false positives against misses."),
    Knob("segment_z", "Segment deviation (robust z)", 1.3, 0.5, 4.0, 0.1, "Findings",
         "How far a group's average must sit from the median group, measured in median "
         "absolute deviations. A robust scale is used so one extreme group cannot hide itself.",
         "Convention, chosen low because segment counts are usually small."),
    Knob("segment_min_delta", "Segment minimum difference (%)", 10.0, 0.0, 100.0, 1.0, "Findings",
         "A robust scale makes tiny spreads produce large z-scores, so a gap must also be "
         "materially large. Statistically unusual is not the same as worth reading.",
         "Judgement. Guards against the robust scale over-reporting."),
    Knob("trend_drift", "Trend drift to report (%)", 25.0, 5.0, 200.0, 5.0, "Findings",
         "Total movement across the whole window, as a share of the mean level, before a "
         "series is called rising or falling.",
         "Judgement. Not a significance test — no p-value is computed for the slope."),
    Knob("movement_z", "Period-change surprise (z)", 2.2, 1.0, 5.0, 0.1, "Findings",
         "How unusual a single period-to-period change must be, against the spread of all "
         "changes in the series.",
         "Convention, near the 2-sigma mark."),
    Knob("missing_warn", "Missing data warning (%)", 20.0, 1.0, 90.0, 5.0, "Findings",
         "Column emptiness that gets flagged, because filters and models silently drop "
         "those rows.",
         "Judgement."),

    # ---- outliers -----------------------------------------------------------------
    Knob("outlier_iqr", "Outlier distance (× IQR)", 3.0, 1.0, 6.0, 0.5, "Outliers",
         "Distance beyond the quartiles before a value is called extreme. Tukey's original "
         "proposal used 1.5 for 'outlier' and 3.0 for 'far out'; the stricter figure is used "
         "to keep the finding list short.",
         "Textbook (Tukey, 1977)."),
    Knob("outlier_min_share", "Minimum share of rows (%)", 0.5, 0.0, 20.0, 0.1, "Outliers",
         "How much of the data must be extreme before it is worth mentioning at all.",
         "Judgement."),

    # ---- statistics ---------------------------------------------------------------
    Knob("alpha", "Significance level (alpha)", 0.05, 0.001, 0.2, 0.005, "Statistics",
         "The false-positive rate tolerated per test. Also sets the width of confidence "
         "intervals and of forecast prediction bands.",
         "Convention (Fisher). Arbitrary, and widely criticised as such."),
    Knob("normality_alpha", "Normality test cutoff", 0.05, 0.001, 0.2, 0.005, "Statistics",
         "Shapiro-Wilk p below this sends group comparison to the rank-based test instead "
         "of the parametric one. You can override the choice per test in the UI.",
         "Convention. Note that this test grows more sensitive with sample size."),

    # ---- time series --------------------------------------------------------------
    Knob("changepoint_threshold", "Changepoint strength", 2.5, 1.0, 6.0, 0.25, "Time series",
         "Mean shift between segments, in pooled standard deviations, before a break is "
         "declared.",
         "Judgement. Binary segmentation has no exact test."),
    Knob("changepoint_min_segment", "Minimum segment length", 4, 2, 30, 1, "Time series",
         "Shortest run of periods that can count as its own regime.",
         "Judgement."),
    Knob("changepoint_max", "Maximum changepoints", 5, 1, 20, 1, "Time series",
         "Cap on how many breaks are reported, to keep the search bounded.",
         "Judgement."),
    Knob("anomaly_sensitivity", "Anomaly sensitivity (sigma)", 3.0, 1.5, 6.0, 0.5, "Time series",
         "Deviation from a robust rolling level before a period is flagged.",
         "Convention (3-sigma)."),

    # ---- modelling ----------------------------------------------------------------
    Knob("cv_folds", "Cross-validation folds", 5, 2, 10, 1, "Modelling",
         "How many splits the model leaderboard is scored over. More folds give a steadier "
         "estimate and take longer.",
         "Convention (5 or 10 fold)."),
    Knob("leakage_correlation", "Leakage warning (|r|)", 0.995, 0.9, 1.0, 0.005, "Modelling",
         "Correlation with the target above which a feature is assumed to be derived from "
         "it rather than predictive of it.",
         "Judgement. A heuristic, not a proof of leakage."),
    Knob("forest_trees", "Random forest size", 300, 50, 1000, 50, "Modelling",
         "Trees per forest. More is steadier and slower; accuracy plateaus early.",
         "Convention."),
)

BY_KEY = {knob.key: knob for knob in KNOBS}
GROUPS = tuple(dict.fromkeys(knob.group for knob in KNOBS))


@dataclass
class Settings:
    """Live methodology values. Defaults mirror KNOBS."""

    correlation_floor: float = 0.6
    segment_z: float = 1.3
    segment_min_delta: float = 10.0
    trend_drift: float = 25.0
    movement_z: float = 2.2
    missing_warn: float = 20.0
    outlier_iqr: float = 3.0
    outlier_min_share: float = 0.5
    alpha: float = 0.05
    normality_alpha: float = 0.05
    changepoint_threshold: float = 2.5
    changepoint_min_segment: int = 4
    changepoint_max: int = 5
    anomaly_sensitivity: float = 3.0
    cv_folds: int = 5
    leakage_correlation: float = 0.995
    forest_trees: int = 300

    def to_dict(self) -> dict:
        return asdict(self)

    def changed_from_default(self) -> dict[str, tuple[float, float]]:
        """Anything the user has moved, so a report can declare it."""
        defaults = Settings()
        return {
            f.name: (getattr(defaults, f.name), getattr(self, f.name))
            for f in fields(self)
            if getattr(defaults, f.name) != getattr(self, f.name)
        }

    def describe(self, key: str) -> str:
        knob = BY_KEY.get(key)
        if not knob:
            return ""
        return f"{knob.label} = {getattr(self, key)} — {knob.rationale} ({knob.source})"


DEFAULTS = Settings()
