"""A transparent segment baseline and two reproducible candidate classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class SmoothedSegmentBaseline:
    """Smoothed historical subscriptions by prior outcome and occupation."""

    def __init__(self, strength: float = 40.0):
        self.strength = strength

    def fit(self, x: pd.DataFrame, y):
        y = np.asarray(y, dtype=int)
        self.global_rate_ = float(y.mean())
        d = pd.DataFrame({
            "segment": x["job"].astype(str).to_numpy() + "|" + x["poutcome"].astype(str).to_numpy(),
            "y": y,
        })
        g = d.groupby("segment")["y"].agg(["sum", "count"])
        self.rates_ = ((g["sum"] + self.strength * self.global_rate_)
                       / (g["count"] + self.strength)).to_dict()
        return self

    def predict_proba(self, x: pd.DataFrame):
        seg = x["job"].astype(str) + "|" + x["poutcome"].astype(str)
        p = seg.map(self.rates_).fillna(self.global_rate_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ])
    return {
        "logistic_regression": Pipeline([
            ("preprocess", preprocessing),
            ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", preprocessing),
            ("classifier", RandomForestClassifier(
                n_estimators=120, min_samples_leaf=15, max_features="sqrt",
                random_state=random_state, n_jobs=-1,
            )),
        ]),
    }
