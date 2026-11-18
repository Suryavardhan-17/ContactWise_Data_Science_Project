"""Reproduce analysis, fit models, select on past validation, audit future holdout.

Usage:
    python -m contactwise.download
    python -m contactwise.run --budget 500 --cost 5 --value 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import joblib

from .config import OUTPUT_DIR, RAW_CSV, FEATURES
from .data import chronological_slices, load_data, target_as_binary, transform_features
from .metrics import scenario_value, summarize, top_k_indices
from .models import SmoothedSegmentBaseline, build_candidates
from .sql_analysis import run_queries
from .statistics import export_descriptive_statistics, export_subgroup_audit


def render_figures(y_test, p_test, base_test, budget: int, split_stats: dict,
                   output_dir: Path) -> None:
    names = list(split_stats)
    rates = [split_stats[k]["positive_rate"] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(names, rates)
    ax.set_ylabel("Historical subscription rate (%)")
    ax.set_title("Campaign outcome drift across row-order partitions")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "time_proxy_drift.png", dpi=170)
    plt.close(fig)

    sorted_idx = np.argsort(-p_test, kind="stable")
    selected = np.cumsum(np.asarray(y_test)[sorted_idx])
    n = len(y_test)
    xs = np.arange(1, n + 1)
    k = min(budget, n)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs, selected, label="Selected model (observed positives)")
    ax.plot(xs, np.cumsum(np.asarray(y_test)[np.argsort(-base_test, kind="stable")]),
            label="Segment baseline")
    ax.plot(xs, xs * np.asarray(y_test).mean(), linestyle="--", label="Expected random ranking")
    ax.axvline(k, color="black", linestyle=":", alpha=0.7, label=f"Top-{k} budget")
    ax.set_xlabel("Number of contacted records in ranking")
    ax.set_ylabel("Cumulative observed subscriptions")
    ax.set_title("Historical top-K ranking (not causal uplift)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "ranking_gain.png", dpi=170)
    plt.close(fig)

    bins = pd.DataFrame({"probability": np.asarray(p_test), "observed": np.asarray(y_test)})
    bins["bin"] = pd.qcut(bins["probability"].rank(method="first"), q=10, labels=False)
    reliability = bins.groupby("bin").agg(predicted=("probability", "mean"),
                                           observed=("observed", "mean"), n=("observed", "size"))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(reliability["predicted"], reliability["observed"], marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed subscription frequency")
    ax.set_title("Out-of-time calibration (10 equal-count bins)")
    ax.set_xlim(0, max(0.4, float(reliability[["predicted", "observed"]].max().max()) * 1.08))
    ax.set_ylim(0, max(0.4, float(reliability[["predicted", "observed"]].max().max()) * 1.08))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=170)
    plt.close(fig)


def build_report(summary: dict) -> str:
    test = summary["test"]
    winner = test["champion"]
    baseline = test["segment_baseline"]
    k = winner["effective_k"]
    drift = ", ".join(
        f"{name} {stats['positive_rate']:.1%} (n={stats['n']:,})"
        for name, stats in summary["partitions"].items()
    )
    scenario = summary["scenario"]
    return f"""# ContactWise — Generated Executive Summary

> **Retrospective evaluation, not a causal experiment.** This file is produced by
> actual code execution. Figures are outcomes from the downloaded historical UCI data.

## Decision question

Given a fixed budget of **{k} contacts per evaluation cohort**, which customer records
should be prioritized *before* a call? The dataset does not contain unique customer
IDs or sufficiently granular dates to estimate an actual 500-calls-per-day policy.

## Validation protocol

- Ordered source rows: train / independent calibration / model selection / final test.
- Model picked **only on selection data**, using top-{k} precision and AP for ties.
- Final test evaluated once after choosing the model.
- Call duration, current campaign call count and current contact details excluded.
- Source records may include multiple contacts from a client; no client IDs are available.

## Dataset shift

{drift}

Different historical periods have potentially very different subscription rates.
This is why random cross-validation alone would be an optimistic deployment proxy.

## Final held-out test (future row-order block)

| Metric | Selected model | Segment-rate baseline |
| --- | ---: | ---: |
| ROC-AUC | {winner['roc_auc']:.3f} | {baseline['roc_auc']:.3f} |
| Average precision | {winner['average_precision']:.3f} | {baseline['average_precision']:.3f} |
| Brier score (lower better) | {winner['brier_score']:.3f} | {baseline['brier_score']:.3f} |
| Precision@{k} | {winner['precision_at_k']:.3f} | {baseline['precision_at_k']:.3f} |
| Lift@{k} vs cohort base rate | {winner['lift_at_k']:.2f}x | {baseline['lift_at_k']:.2f}x |
| Observed positives in selected {k} | {winner['observed_positives_at_k']} | {baseline['observed_positives_at_k']} |

Selected classifier: **{summary['selected_model']}**.

## Scenario analysis (illustrative, NOT revenue impact)

Assumed value per recorded subscription: **{scenario['assumed_value_per_subscription']:.2f} units**.
Assumed cost per contact: **{scenario['assumed_cost_per_call']:.2f} units**.
Historical outcome-based scenario proxy: **{scenario['historical_outcome_scenario_proxy']:.2f} units**.

This proxy **does not** establish incremental sales, causal retention, or true ROI:
customers were already contacted in the historical dataset, and there is no untreated
control cohort. The score is a sensitivity aid only.

## Recommendation and next test

Review the top-K ranking and evaluate stability under newer data and different cost
assumptions. Before operational use, obtain a complete pre-contact universe,
customer identifiers, observed contact costs, and randomized control data to estimate
incremental campaign effects. Check privacy, fairness, and contact permissions.

## Artifacts

- `model_selection.csv`: historical selection-set evidence for model choice.
- `test_ranked.csv`: held-out ranking with **model probabilities and labels**, no raw customer attributes.
- `scenario_sensitivity.csv`: illustrative cost-value calculations.
- `segment_uncertainty.csv`: descriptive rates and Wilson intervals.
- `age_subgroup_audit.csv`: descriptive group-level calibration gaps, not a fairness certification.
- `*.png`: drift, ranking, calibration plots.
- `champion_model.joblib`: fitted classifier (load only from trusted files).
"""


def run(input_csv: Path, output_dir: Path, budget: int, cost: float, value: float) -> dict:
    raw = load_data(input_csv)
    x, y = transform_features(raw), target_as_binary(raw)
    slices = chronological_slices(len(raw))
    pieces = {name: (x.iloc[sl], y.iloc[sl]) for name, sl in slices.items()}
    partitions = {}
    for name, (_, yi) in pieces.items():
        if yi.nunique() < 2:
            raise ValueError(f"Split '{name}' has only one class; can't evaluate reliably")
        partitions[name] = {"n": len(yi), "positive_rate": float(yi.mean())}
    output_dir.mkdir(parents=True, exist_ok=True)
    run_queries(raw, output_dir)
    export_descriptive_statistics(raw, output_dir)
    x_train, y_train = pieces["train"]
    x_cal, y_cal = pieces["calibration"]
    x_select, y_select = pieces["selection"]
    x_test, y_test = pieces["test"]

    # Baseline is fit only on the oldest train rows.
    baseline = SmoothedSegmentBaseline().fit(x_train, y_train)
    base_select = baseline.predict_proba(x_select)[:, 1]
    candidate_rows = [{"model": "segment_baseline", **summarize(y_select, base_select, budget)}]
    candidates = {"segment_baseline": baseline}
    for name, pipeline in build_candidates().items():
        print(f"Training {name} ...")
        pipeline.fit(x_train, y_train)
        # Keep calibration separate from training, selection, and final test.
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
        calibrated.fit(x_cal, y_cal)
        p_select = calibrated.predict_proba(x_select)[:, 1]
        candidate_rows.append({"model": name, **summarize(y_select, p_select, budget)})
        candidates[name] = calibrated

    selection_df = pd.DataFrame(candidate_rows).sort_values(
        ["precision_at_k", "average_precision", "model"],
        ascending=[False, False, True], kind="stable",
    )
    selection_df.to_csv(output_dir / "model_selection.csv", index=False)
    champion_name = str(selection_df.iloc[0]["model"])
    champion = candidates[champion_name]
    p_test = champion.predict_proba(x_test)[:, 1]
    base_test = baseline.predict_proba(x_test)[:, 1]
    test_scores = {
        "champion": summarize(y_test, p_test, budget),
        "segment_baseline": summarize(y_test, base_test, budget),
    }
    test_df = pd.DataFrame({
        "original_source_row": np.arange(slices["test"].start, slices["test"].stop),
        "observed_label": np.asarray(y_test),
        "predicted_subscription_probability": p_test,
    }).sort_values("predicted_subscription_probability", ascending=False, kind="stable")
    test_df["rank"] = np.arange(1, len(test_df) + 1)
    test_df["within_contact_budget"] = test_df["rank"] <= budget
    test_df.to_csv(output_dir / "test_ranked.csv", index=False)
    export_subgroup_audit(y_test, p_test, raw.iloc[slices["test"]]["age"], output_dir)

    scenarios = []
    for c in sorted(set([2.0, 5.0, 10.0, float(cost)])):
        for v in sorted(set([25.0, 50.0, 100.0, 150.0, float(value)])):
            scenarios.append(scenario_value(y_test, p_test, budget, v, c))
    pd.DataFrame(scenarios).to_csv(output_dir / "scenario_sensitivity.csv", index=False)
    summary = {
        "source_rows": len(raw), "features": FEATURES,
        "selected_model": champion_name, "contact_budget": budget,
        "selection": selection_df.to_dict(orient="records"),
        "partitions": partitions, "test": test_scores,
        "scenario": scenario_value(y_test, p_test, budget, value, cost),
        "notes": [
            "No incremental/causal effect is estimated.",
            "Campaign row order is a time proxy, not exact daily timing.",
            "No customer identifiers are provided: repeat contacts may cross splits.",
            "No actual cost or subscription value data; scenario figures are assumptions.",
            "No real prospective deployment or customer targeting occurred.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "executive_summary.md").write_text(build_report(summary), encoding="utf-8")
    joblib.dump(champion, output_dir / "champion_model.joblib")
    render_figures(y_test, p_test, base_test, budget, partitions, output_dir)
    print(f"\nChampion selected on validation: {champion_name}")
    print(f"TEST precision@{min(budget, len(y_test))}: {test_scores['champion']['precision_at_k']:.3f}")
    print(f"TEST average precision: {test_scores['champion']['average_precision']:.3f}")
    print(f"TEST Brier score: {test_scores['champion']['brier_score']:.3f}")
    print(f"Results saved under {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_CSV,
                        help="Path to bank-additional-full.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Directory for generated reports")
    parser.add_argument("--budget", type=int, default=500,
                        help="Top K records in evaluation cohort (not per calendar day)")
    parser.add_argument("--cost", type=float, default=5,
                        help="Assumed cost per call, arbitrary units")
    parser.add_argument("--value", type=float, default=100,
                        help="Assumed value per observed historical positive, arbitrary units")
    args = parser.parse_args()
    if args.budget < 1 or args.cost < 0 or args.value < 0:
        parser.error("Budget >= 1 and nonnegative cost/value are required")
    run(args.input, args.output, args.budget, args.cost, args.value)


if __name__ == "__main__":
    main()
