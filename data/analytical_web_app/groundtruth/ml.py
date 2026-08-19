"""L3 — predictive modelling.

Differences from a single-split random forest: several model families compete on
cross-validated scores, a dummy baseline sets the floor, importances come from
permutation rather than impurity, and features that leak the target are caught
before they flatter the metrics.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSIFICATION = "Classification"
REGRESSION = "Regression"


@dataclass
class LeaderboardEntry:
    model: str
    cv_mean: float
    cv_std: float
    beats_baseline: bool


@dataclass
class ModelResult:
    problem_type: str
    best_model: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    leaderboard: list[LeaderboardEntry]
    importance: pd.DataFrame
    predictions: pd.DataFrame
    leakage_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scoring: str = ""
    pipeline: object | None = None


def infer_problem_type(target: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(target):
        return CLASSIFICATION
    distinct = target.nunique(dropna=True)
    if distinct <= 2:
        return CLASSIFICATION
    # An integer column with few levels relative to its length is a label, not a quantity.
    if pd.api.types.is_integer_dtype(target) and distinct <= max(10, len(target) * 0.01):
        return CLASSIFICATION
    return REGRESSION


def detect_leakage(df: pd.DataFrame, target: str, features: list[str], problem_type: str) -> list[str]:
    """Flag features that all but contain the answer."""
    warnings_found: list[str] = []
    y = df[target]
    for feature in features:
        column = df[feature]
        if column.nunique(dropna=True) >= len(df) * 0.98 and not pd.api.types.is_numeric_dtype(column):
            warnings_found.append(f"{feature}: unique per row — an identifier, not a predictor")
            continue
        if problem_type == REGRESSION and pd.api.types.is_numeric_dtype(column):
            pair = pd.concat([column, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) > 3:
                r = pair.corr().iloc[0, 1]
                if pd.notna(r) and abs(r) > 0.995:
                    warnings_found.append(f"{feature}: correlates {r:.4f} with the target — likely derived from it")
        if problem_type == CLASSIFICATION and column.dtype == object:
            crosstab = pd.crosstab(column, y)
            if len(crosstab) > 1 and (crosstab.max(axis=1) == crosstab.sum(axis=1)).all():
                warnings_found.append(f"{feature}: each value maps to exactly one class — likely a relabelling of the target")
    return warnings_found


def _build_preprocessor(X: pd.DataFrame, scale: bool = False) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    steps = []
    if numeric:
        pipeline = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            pipeline.append(("scaler", StandardScaler()))
        steps.append(("num", Pipeline(pipeline), numeric))
    if categorical:
        steps.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


def _candidates(problem_type: str, seed: int) -> dict[str, object]:
    if problem_type == CLASSIFICATION:
        return {
            "Random forest": RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample", n_jobs=-1),
            "Gradient boosting": HistGradientBoostingClassifier(random_state=seed),
            "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
        }
    return {
        "Random forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingRegressor(random_state=seed),
        "Ridge regression": Ridge(random_state=seed),
    }


def train(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    problem_type: str = "Auto",
    test_size: float = 0.2,
    cv_folds: int = 5,
    seed: int = 42,
    compute_importance: bool = True,
) -> ModelResult:
    if not features:
        raise ValueError("Select at least one feature.")
    if target in features:
        raise ValueError("The target cannot also be a feature.")

    data = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    if len(data) < 30:
        raise ValueError(f"Need at least 30 rows with a non-null target (have {len(data)}).")

    resolved = infer_problem_type(data[target]) if problem_type == "Auto" else problem_type
    notes: list[str] = []
    X, y = data[features], data[target]

    if resolved == CLASSIFICATION:
        counts = y.value_counts()
        if (counts < 2).any():
            keep = counts[counts >= 2].index
            dropped = int((~y.isin(keep)).sum())
            data = data[y.isin(keep)]
            X, y = data[features], data[target]
            notes.append(f"Dropped {dropped} row(s) in classes with a single example.")
        if y.nunique() < 2:
            raise ValueError("The target needs at least two classes with 2+ examples each.")

    leakage = detect_leakage(data, target, features, resolved)

    stratify = y if resolved == CLASSIFICATION and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)

    # Cross-validated leaderboard on the training split only — the test split stays untouched.
    scoring = "f1_weighted" if resolved == CLASSIFICATION else "r2"
    n_splits = int(min(cv_folds, len(X_train) // 5)) or 2
    if resolved == CLASSIFICATION:
        n_splits = int(max(2, min(n_splits, y_train.value_counts().min())))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)

    dummy = DummyClassifier(strategy="most_frequent") if resolved == CLASSIFICATION else DummyRegressor(strategy="mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline_scores = cross_val_score(
            Pipeline([("preprocess", _build_preprocessor(X_train)), ("model", dummy)]),
            X_train, y_train, cv=splitter, scoring=scoring, n_jobs=1,
        )
    baseline_cv = float(np.mean(baseline_scores))

    leaderboard: list[LeaderboardEntry] = []
    best_name, best_score, best_pipeline = "", -np.inf, None
    for name, estimator in _candidates(resolved, seed).items():
        needs_scaling = name in ("Logistic regression", "Ridge regression")
        pipeline = Pipeline([("preprocess", _build_preprocessor(X_train, scale=needs_scaling)), ("model", estimator)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipeline, X_train, y_train, cv=splitter, scoring=scoring, n_jobs=1)
        except Exception as exc:
            notes.append(f"{name} failed to fit: {str(exc)[:80]}")
            continue
        mean_score = float(np.mean(scores))
        leaderboard.append(LeaderboardEntry(name, mean_score, float(np.std(scores)), mean_score > baseline_cv))
        if mean_score > best_score:
            best_name, best_score, best_pipeline = name, mean_score, pipeline

    if best_pipeline is None:
        raise ValueError("No candidate model could be fitted on this data.")
    leaderboard.sort(key=lambda e: e.cv_mean, reverse=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_pipeline.fit(X_train, y_train)
        predicted = best_pipeline.predict(X_test)

        baseline_pipeline = Pipeline([("preprocess", _build_preprocessor(X_train)), ("model", dummy)]).fit(X_train, y_train)
        baseline_predicted = baseline_pipeline.predict(X_test)

    def score_set(truth, prediction) -> dict[str, float]:
        if resolved == CLASSIFICATION:
            out = {
                "accuracy": float(accuracy_score(truth, prediction)),
                "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
            }
            if truth.nunique() == 2:
                try:
                    classes = list(best_pipeline.named_steps["model"].classes_)
                    proba = best_pipeline.predict_proba(X_test)[:, classes.index(sorted(classes)[1])]
                    out["roc_auc"] = float(roc_auc_score((truth == sorted(classes)[1]).astype(int), proba))
                except Exception:
                    pass
            return out
        return {
            "r2": float(r2_score(truth, prediction)),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(root_mean_squared_error(truth, prediction)),
        }

    metrics = score_set(y_test, predicted)
    baseline_metrics = score_set(y_test, baseline_predicted)

    # Permutation importance: unlike impurity, it is not biased toward high-cardinality columns.
    if compute_importance:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = permutation_importance(
                best_pipeline, X_test, y_test, n_repeats=10, random_state=seed, scoring=scoring, n_jobs=1
            )
        importance = pd.DataFrame(
            {"feature": features, "importance": perm.importances_mean, "std": perm.importances_std}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        importance["informative"] = importance["importance"] > importance["std"]
    else:
        importance = pd.DataFrame(columns=["feature", "importance", "std", "informative"])

    predictions = X_test.copy()
    predictions[f"actual_{target}"] = y_test
    predictions[f"predicted_{target}"] = predicted
    if resolved == REGRESSION:
        predictions["residual"] = predictions[f"actual_{target}"] - predictions[f"predicted_{target}"]

    if not any(entry.beats_baseline for entry in leaderboard):
        notes.append("No model beat the naive baseline — these features carry little signal for this target.")

    return ModelResult(
        problem_type=resolved,
        best_model=best_name,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        leaderboard=leaderboard,
        importance=importance,
        predictions=predictions,
        leakage_warnings=leakage,
        notes=notes,
        scoring=scoring,
        pipeline=best_pipeline,
    )


def leaderboard_frame(result: ModelResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": e.model,
                f"cv_{result.scoring}": round(e.cv_mean, 4),
                "std": round(e.cv_std, 4),
                "beats_baseline": e.beats_baseline,
            }
            for e in result.leaderboard
        ]
    )
