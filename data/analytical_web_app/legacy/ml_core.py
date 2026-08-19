from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelResult:
    problem_type: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def infer_problem_type(target: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(target):
        return "Classification"
    unique = target.nunique(dropna=True)
    if unique <= max(20, int(len(target) * 0.05)):
        return "Classification"
    return "Regression"


def train_random_forest(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    problem_type: str = "Auto",
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    if not features:
        raise ValueError("Select at least one feature.")
    data = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    if len(data) < 20:
        raise ValueError("Need at least 20 rows with a non-null target.")

    resolved_type = infer_problem_type(data[target]) if problem_type == "Auto" else problem_type
    X = data[features]
    y = data[target]

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in features if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    if resolved_type == "Classification":
        model = RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced_subsample", n_jobs=-1)
        stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    else:
        model = RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1)
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    predicted = pipe.predict(X_test)

    if resolved_type == "Classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        }
    else:
        metrics = {
            "r2": float(r2_score(y_test, predicted)),
            "mae": float(mean_absolute_error(y_test, predicted)),
        }

    pred_df = X_test.copy()
    pred_df[f"actual_{target}"] = y_test
    pred_df[f"predicted_{target}"] = predicted

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    return ModelResult(resolved_type, metrics, pred_df, fi)
