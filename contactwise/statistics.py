"""Descriptive statistical evidence; not a randomized causal comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    """95% Wilson binomial confidence interval; assumes independent rows."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Expected n > 0 and 0 <= successes <= n")
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return float(center - margin), float(center + margin)


def export_descriptive_statistics(raw: pd.DataFrame, output_dir: Path) -> None:
    df = raw[["job", "y"]].copy()
    df["positive"] = (df["y"] == "yes").astype(int)
    summary = df.groupby("job", dropna=False)["positive"].agg(["sum", "count"]).reset_index()
    summary = summary.rename(columns={"sum": "observed_subscriptions", "count": "records"})
    summary["subscription_rate"] = summary["observed_subscriptions"] / summary["records"]
    intervals = [wilson_interval(int(k), int(n)) for k,n in
                 zip(summary["observed_subscriptions"], summary["records"])]
    summary["wilson_lower_95"] = [a for a,b in intervals]
    summary["wilson_upper_95"] = [b for a,b in intervals]
    summary = summary.sort_values("records", ascending=False)
    summary.to_csv(output_dir / "segment_uncertainty.csv", index=False)

    contingency = pd.crosstab(df["job"], df["positive"])
    chi2, pvalue, dof, expected = chi2_contingency(contingency)
    checks = {
        "comparison": "Unadjusted job category vs historical subscription label",
        "chi_square": float(chi2), "p_value": float(pvalue),
        "degrees_of_freedom": int(dof), "smallest_expected_count": float(expected.min()),
        "caution": (
            "Association is not causation. Repeated individuals are not uniquely identifiable; "
            "independence may be violated, and multiple testing is not controlled. "
            "Use this as exploratory descriptive evidence only."
        ),
    }
    (output_dir / "statistical_checks.json").write_text(json.dumps(checks, indent=2))


def export_subgroup_audit(y_test, p_test, ages, output_dir: Path) -> None:
    """Diagnostic by age only; not an eligibility or fairness certification."""
    df = pd.DataFrame({"observed": np.asarray(y_test),
                       "score": np.asarray(p_test), "age": np.asarray(ages)})
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 65, np.inf],
                             labels=["18–29", "30–49", "50–64", "65+"],
                             right=False, include_lowest=True)
    groups = df.groupby("age_group", observed=True).agg(
        records=("observed", "size"), observed_rate=("observed", "mean"),
        mean_prediction=("score", "mean"),
    ).reset_index()
    # Reporting expected calibration differences with group sizes; small groups are unstable.
    groups["calibration_gap"] = groups["mean_prediction"] - groups["observed_rate"]
    groups["caution"] = np.where(groups["records"] < 100, "small subgroup; unstable", "descriptive only")
    groups.to_csv(output_dir / "age_subgroup_audit.csv", index=False)
