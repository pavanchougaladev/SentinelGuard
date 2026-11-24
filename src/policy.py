"""Cost-sensitive threshold policy engine and probability calibration.

A fraud model outputs probabilities; a *policy* turns them into decisions.
Those decisions are a business trade-off: a lower cutoff catches more fraud but
floods the analyst queue, a higher cutoff keeps the queue clean but lets fraud
through. This module simulates several explicit policies on the held-out split
so the trade-off is inspectable instead of hidden inside one hard-coded number.

Policies produced:
- ``cost_optimized``   minimises false-positive review cost + missed-fraud cost
- ``balanced_f1``      maximises F1
- ``high_recall``      smallest cutoff reaching the recall target
- ``high_precision``   highest recall while keeping the precision target
- ``review_capacity``  highest recall within the analyst review capacity

The artifacts (JSON/CSV/Markdown) are decision support, not approved rules.

Usage (recompute for models that are already trained):
    python -m src.policy --data data/transactions.csv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Cost assumptions. A review costs analyst minutes; a missed fraud costs the
# lost value plus chargeback handling. Both are deliberately conservative and
# easy to override so the trade-off stays explicit.
REVIEW_COST = 1.25
FRAUD_HANDLING_COST = 15.0

# Default guards for the capacity- and target-driven policies.
REVIEW_CAPACITY = 0.02
TARGET_RECALL = 0.95
TARGET_PRECISION = 0.90

POLICY_ORDER = (
    "cost_optimized",
    "balanced_f1",
    "high_recall",
    "high_precision",
    "review_capacity",
)

POLICY_PURPOSE = {
    "cost_optimized": "Minimises review cost plus missed-fraud cost",
    "balanced_f1": "Balances precision and recall (F1)",
    "high_recall": f"Catches at least {TARGET_RECALL:.0%} of fraud",
    "high_precision": f"Keeps precision at or above {TARGET_PRECISION:.0%}",
    "review_capacity": f"Stays within {REVIEW_CAPACITY:.0%} of traffic flagged",
}


def _candidate_thresholds(scores: np.ndarray, n: int = 250) -> np.ndarray:
    """Score quantiles as candidate cutoffs (dense where the scores are)."""
    if len(scores) == 0:
        return np.array([0.5])
    quantiles = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, n)))
    return np.clip(quantiles, 1e-3, 1.0 - 1e-3)


def threshold_table(
    y: Iterable[int],
    scores: Iterable[float],
    amounts: Iterable[float],
    review_cost: float = REVIEW_COST,
    fraud_handling_cost: float = FRAUD_HANDLING_COST,
) -> pd.DataFrame:
    """Every candidate cutoff with its confusion counts, rates and cost."""
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    amount_arr = np.asarray(amounts, dtype=float)
    if len(y_arr) != len(score_arr) or len(y_arr) != len(amount_arr):
        raise ValueError("y, scores and amounts must have the same length")

    positive = y_arr == 1
    total = max(len(y_arr), 1)
    total_fraud = max(int(positive.sum()), 1)
    total_legit = max(int((~positive).sum()), 1)

    rows: list[dict[str, Any]] = []
    for threshold in _candidate_thresholds(score_arr):
        flagged = score_arr >= threshold
        tp = int(np.sum(flagged & positive))
        fp = int(np.sum(flagged & ~positive))
        fn = int(total_fraud - tp) if positive.any() else 0
        tn = int(total - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_fraud
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        missed_value = float(amount_arr[~flagged & positive].sum())

        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "fpr": round(fp / total_legit, 6),
                "flagged_rate": round((tp + fp) / total, 6),
                "alerts": tp + fp,
                "missed_value": round(missed_value, 2),
                "cost": round(
                    fp * review_cost + fn * fraud_handling_cost + missed_value, 2
                ),
            }
        )
    return pd.DataFrame(rows)


def _as_row(series: pd.Series) -> dict[str, Any]:
    """Row dict with count columns kept integral across mixed dtypes."""
    row = series.to_dict()
    for key in ("tp", "fp", "fn", "tn", "alerts"):
        if key in row:
            row[key] = int(row[key])
    return row


def select_policies(
    table: pd.DataFrame,
    review_capacity: float = REVIEW_CAPACITY,
    target_recall: float = TARGET_RECALL,
    target_precision: float = TARGET_PRECISION,
) -> list[dict[str, Any]]:
    """Pick one row of ``table`` per policy, in ``POLICY_ORDER``."""
    if table.empty:
        return []

    purposes = {
        "cost_optimized": POLICY_PURPOSE["cost_optimized"],
        "balanced_f1": POLICY_PURPOSE["balanced_f1"],
        "high_recall": f"Catches at least {target_recall:.0%} of fraud",
        "high_precision": f"Keeps precision at or above {target_precision:.0%}",
        "review_capacity": f"Stays within {review_capacity:.0%} of traffic flagged",
    }

    def pick(frame: pd.DataFrame, rule: str) -> dict[str, Any]:
        row = _as_row(frame.iloc[0])
        row["policy"] = rule
        row["purpose"] = purposes[rule]
        return row

    selected: list[dict[str, Any]] = []

    # Cheapest expected decision cost.
    selected.append(pick(table.loc[[table["cost"].idxmin()]], "cost_optimized"))

    # Best plain F1.
    selected.append(pick(table.loc[[table["f1"].idxmax()]], "balanced_f1"))

    # Highest precision among cutoffs that still reach the recall target;
    # fall back to the best recall available.
    recall_ok = table[table["recall"] >= target_recall]
    if not recall_ok.empty:
        selected.append(pick(recall_ok.loc[[recall_ok["precision"].idxmax()]], "high_recall"))
    else:
        selected.append(pick(table.loc[[table["recall"].idxmax()]], "high_recall"))

    # Highest recall while precision stays at target; fall back to best precision.
    precision_ok = table[table["precision"] >= target_precision]
    if not precision_ok.empty:
        selected.append(
            pick(precision_ok.loc[[precision_ok["recall"].idxmax()]], "high_precision")
        )
    else:
        selected.append(pick(table.loc[[table["precision"].idxmax()]], "high_precision"))

    # Highest recall within the analyst review budget.
    capacity_ok = table[table["flagged_rate"] <= review_capacity]
    if not capacity_ok.empty:
        selected.append(
            pick(capacity_ok.loc[[capacity_ok["recall"].idxmax()]], "review_capacity")
        )
    else:
        selected.append(pick(table.loc[[table["flagged_rate"].idxmin()]], "review_capacity"))

    return selected


def calibration_report(
    y: Iterable[int], scores: Iterable[float], bins: int = 10
) -> dict[str, Any]:
    """Reliability curve plus the Brier score for the held-out split.

    A calibrated score means "of all transactions scored 20%, about 20% are
    fraud"; that matters when the score is also a policy input.
    """
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(score_arr, edges[1:-1], right=False), 0, bins - 1)

    curve: list[dict[str, Any]] = []
    for bin_index in range(bins):
        in_bin = index == bin_index
        count = int(in_bin.sum())
        curve.append(
            {
                "bin_low": round(float(edges[bin_index]), 4),
                "bin_high": round(float(edges[bin_index + 1]), 4),
                "mean_predicted": round(float(score_arr[in_bin].mean()), 6)
                if count
                else None,
                "fraction_positive": round(float(y_arr[in_bin].mean()), 6)
                if count
                else None,
                "count": count,
            }
        )

    brier = float(np.mean((score_arr - y_arr) ** 2)) if len(y_arr) else 0.0
    return {"bins": bins, "brier": round(brier, 6), "curve": curve}


def baseline_report(y: Iterable[int], scores: Iterable[float]) -> dict[str, Any]:
    """Reference strategies so the model is compared with something trivial."""
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    total = max(len(y_arr), 1)
    prevalence = float(y_arr.mean()) if len(y_arr) else 0.0
    flagged_rate = float(np.mean(score_arr >= np.median(score_arr))) if len(y_arr) else 0.0

    return {
        "prevalence": round(prevalence, 6),
        "majority": {
            "description": "Never alert",
            "precision": 0.0,
            "recall": 0.0,
            "flagged_rate": 0.0,
        },
        "flag_all": {
            "description": "Alert on every transaction",
            "precision": round(prevalence, 6),
            "recall": 1.0,
            "flagged_rate": 1.0,
        },
        "random": {
            "description": "Random alerts at the model's median-score rate",
            "precision": round(prevalence, 6),
            "recall": round(flagged_rate, 6),
            "flagged_rate": round(flagged_rate, 6),
        },
        "n": int(total),
    }


def build_policy_artifacts(
    y: Iterable[int],
    scores: Iterable[float],
    amounts: Iterable[float],
    suffix: str = "",
    output_dir: str = "models",
    operating_threshold: float | None = None,
    review_capacity: float = REVIEW_CAPACITY,
    target_recall: float = TARGET_RECALL,
    target_precision: float = TARGET_PRECISION,
    review_cost: float = REVIEW_COST,
    fraud_handling_cost: float = FRAUD_HANDLING_COST,
) -> dict[str, Any]:
    """Score every policy and write JSON, CSV and Markdown artifacts."""
    table = threshold_table(y, scores, amounts, review_cost, fraud_handling_cost)
    policies = select_policies(
        table, review_capacity, target_recall, target_precision
    )

    operating: dict[str, Any] | None = None
    if operating_threshold is not None and not table.empty:
        nearest = (table["threshold"] - operating_threshold).abs().idxmin()
        operating = _as_row(table.loc[nearest])
        operating["threshold"] = float(operating_threshold)
        operating["policy"] = "operating"
        operating["purpose"] = "Cutoff currently used by the deployed model"

    payload = {
        "assumptions": {
            "review_cost": review_cost,
            "fraud_handling_cost": fraud_handling_cost,
            "review_capacity": review_capacity,
            "target_recall": target_recall,
            "target_precision": target_precision,
        },
        "operating": operating,
        "policies": policies,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"threshold_policy{suffix}.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    pd.DataFrame(policies).to_csv(
        os.path.join(output_dir, f"threshold_policy{suffix}.csv"), index=False
    )
    with open(os.path.join(output_dir, f"threshold_policy{suffix}.md"), "w") as fh:
        fh.write(_policy_markdown(payload))

    return payload


def _policy_markdown(payload: dict[str, Any]) -> str:
    assumptions = payload["assumptions"]
    lines = [
        "# Threshold policy simulation",
        "",
        "Decision-support artifacts for analyst review, not approved rules.",
        "",
        f"- Review cost per alert: ${assumptions['review_cost']:,.2f}",
        f"- Handling cost per missed fraud: ${assumptions['fraud_handling_cost']:,.2f}",
        f"- Review capacity: {assumptions['review_capacity']:.2%} of traffic",
        f"- Recall target: {assumptions['target_recall']:.0%}",
        f"- Precision target: {assumptions['target_precision']:.0%}",
        "",
        "| Policy | Purpose | Threshold | Precision | Recall | FPR | Flagged | Alerts | Expected cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    rows = [payload["operating"]] if payload.get("operating") else []
    rows.extend(payload["policies"])
    for row in rows:
        lines.append(
            "| {policy} | {purpose} | {threshold:.3f} | {precision:.1%} | "
            "{recall:.1%} | {fpr:.3%} | {flagged_rate:.2%} | {alerts:,} | "
            "${cost:,.0f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone CLI: recompute artifacts for an already trained bundle
# ---------------------------------------------------------------------------


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.features import build_features

    return build_features(df)


def compute_for_bundle(
    data_path: str,
    bundle_path: str,
    suffix: str = "",
    output_dir: str = "models",
) -> dict[str, Any]:
    """Rebuild the held-out split, score it, and refresh policy artifacts."""
    import joblib

    bundle = joblib.load(bundle_path)
    model, iso = bundle["model"], bundle["iso"]
    base_cols, cols = bundle["base_cols"], bundle["cols"]

    df = pd.read_csv(data_path)
    time_col = "Time" if "Time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col is not None:
        df = df.sort_values(time_col).reset_index(drop=True)

    split = int(len(df) * 0.85)
    test = df.iloc[split:].reset_index(drop=True)
    target_col = "Class" if "Class" in test.columns else "is_fraud"
    if target_col not in test.columns:
        raise ValueError(f"{data_path} has no Class/is_fraud label column")

    X = _build_features(test)
    for name in base_cols:
        if name not in X.columns:
            X[name] = 0.0
    X["anomaly"] = -iso.score_samples(X[base_cols])
    X = X[cols]

    y = test[target_col].astype(int).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    amount_col = "Amount" if "Amount" in test.columns else "amount"
    amounts = test[amount_col].astype(float).to_numpy()

    payload = build_policy_artifacts(
        y,
        scores,
        amounts,
        suffix=suffix,
        output_dir=output_dir,
        operating_threshold=float(bundle["threshold"]),
    )

    # Merge the probability diagnostics into the metrics artifact so the
    # dashboard can show them without rescoring.
    metrics_path = os.path.join(output_dir, f"metrics{suffix}.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        calibration = calibration_report(y, scores)
        metrics["brier"] = calibration["brier"]
        metrics["calibration"] = calibration
        metrics["baselines"] = baseline_report(y, scores)
        metrics["policy_summary"] = {
            row["policy"]: row["threshold"] for row in payload["policies"]
        }
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

    return payload


def main(a: argparse.Namespace) -> None:
    for data_path, bundle_path, suffix in a.jobs:
        if not (os.path.exists(data_path) and os.path.exists(bundle_path)):
            print(f"skipping {bundle_path}: missing data or bundle")
            continue
        payload = compute_for_bundle(data_path, bundle_path, suffix, a.output_dir)
        print(f"\n{suffix or 'kaggle'} policy thresholds:")
        for row in payload["policies"]:
            print(
                f"  {row['policy']:<16} thr={row['threshold']:.3f} "
                f"precision={row['precision']:.1%} recall={row['recall']:.1%} "
                f"flagged={row['flagged_rate']:.2%} cost=${row['cost']:,.0f}"
            )
        print(f"wrote threshold_policy{suffix}.json/.csv/.md to {a.output_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--bundle", default="models/bundle.joblib")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--output-dir", default="models")
    a = ap.parse_args()
    main(
        argparse.Namespace(
            output_dir=a.output_dir,
            jobs=[(a.data, a.bundle, a.suffix)],
        )
    )
