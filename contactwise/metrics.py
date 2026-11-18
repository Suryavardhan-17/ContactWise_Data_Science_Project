"""Ranking, calibration, and explicitly *non-causal* scenario metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_k_indices(scores, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain NaN or Inf")
    if k < 1:
        raise ValueError("Call budget must be >= 1")
    k = min(k, len(scores))
    return np.argsort(-scores, kind="stable")[:k]


def precision_at_k(y_true, scores, k: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) != len(scores) or len(y_true) == 0:
        raise ValueError("Labels/scores must have matching nonzero length")
    return float(y_true[top_k_indices(scores, k)].mean())


def summarize(y_true, scores, k: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    if len(np.unique(y_true)) < 2:
        raise ValueError("At least one class from each outcome required")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("Predicted probabilities must lie in [0, 1]")
    base = float(y_true.mean())
    p_at_k = precision_at_k(y_true, scores, k)
    return {
        "n": int(len(y_true)),
        "positive_rate": base,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "precision_at_k": p_at_k,
        "lift_at_k": float(p_at_k / base) if base else 0.0,
        "observed_positives_at_k": int(y_true[top_k_indices(scores, k)].sum()),
        "effective_k": int(min(k, len(y_true))),
    }


def scenario_value(y_true, scores, k: int, value_per_subscription: float,
                   cost_per_call: float) -> dict[str, float]:
    """Historical outcome-based *proxy*, not a causal incremental ROI estimate."""
    if value_per_subscription < 0 or cost_per_call < 0:
        raise ValueError("Costs and values must be non-negative")
    yy = np.asarray(y_true)[top_k_indices(scores, k)]
    return {
        "observed_subscriptions_in_selected_records": int(yy.sum()),
        "selected_records": len(yy),
        "assumed_value_per_subscription": value_per_subscription,
        "assumed_cost_per_call": cost_per_call,
        "historical_outcome_scenario_proxy": float(
            yy.sum() * value_per_subscription - len(yy) * cost_per_call
        ),
    }
