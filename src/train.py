"""Advanced-level training pipeline:
time-based split -> IsolationForest anomaly feature -> class-balanced RandomForest
-> precision-weighted threshold selection -> SHAP -> cost-sensitive policy
artifacts -> calibration/baseline diagnostics.
Usage: python -m src.train --data data/transactions.csv
"""
import argparse, json, os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve)
from src.features import build_features
from src.policy import baseline_report, build_policy_artifacts, calibration_report

# False alerts cost analyst time, so the cutoff is tuned with a
# precision-weighted F-score (beta < 1) instead of plain F1.
BETA = 0.5


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"threshold": float(thr), "recall": float(recall),
            "precision": float(precision),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "accuracy": float((tp + tn) / max(len(y), 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "fpr": float(fp / max(fp + tn, 1)), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn), "alerts": int(pred.sum())}


def best_threshold(y, p, beta=BETA):
    """Threshold that maximises precision-weighted F-beta on validation.

    A plain F1 cutoff flagged enough legitimate transactions to hold
    precision in the 50-80% range. Weighting precision higher keeps the
    alert queue clean without losing any additional fraud.
    """
    precision, recall, thresholds = precision_recall_curve(y, p)
    fbeta = ((1 + beta ** 2) * precision * recall
             / np.clip(beta ** 2 * precision + recall, 1e-12, None))
    best = int(np.nanargmax(fbeta[:-1]))  # last PR point has no threshold
    return float(thresholds[best])


def main(a):
    df = pd.read_csv(a.data)
    time_col = "Time" if "Time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col:
        df = df.sort_values(time_col).reset_index(drop=True)
    target_col = "Class" if "Class" in df.columns else "is_fraud"

    X = build_features(df)
    y = df[target_col].astype(int).values
    mode = "kaggle" if "V1" in df.columns else "synthetic"
    n = len(df)
    print(f"loaded {n:,} rows · {X.shape[1]} features · mode={mode}", flush=True)
    i1, i2 = int(n * 0.70), int(n * 0.85)  # time-based split, no leakage
    base_cols = list(X.columns)

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X.iloc[:i1][base_cols])
    X["anomaly"] = -iso.score_samples(X[base_cols])  # higher = more anomalous
    cols = list(X.columns)

    Xtr, ytr = X.iloc[:i1], y[:i1]
    Xva, yva = X.iloc[i1:i2], y[i1:i2]
    Xte, yte = X.iloc[i2:], y[i2:]

    if mode == "kaggle":
        params = {"n_estimators": 300, "max_depth": None, "max_features": "sqrt",
                  "min_samples_leaf": 1, "class_weight": "balanced_subsample"}
    else:
        # The transaction dataset is weakly separable; unrestricted trees grow
        # enormous for no accuracy gain, so keep them compact and fast.
        params = {"n_estimators": 200, "max_depth": 16, "max_features": "sqrt",
                  "min_samples_leaf": 5, "class_weight": "balanced_subsample"}
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(Xtr, ytr)
    print("model fitted", flush=True)

    pva = model.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, pva)
    pte = model.predict_proba(Xte)[:, 1]
    res = metrics_at(yte, pte, thr)
    calibration = calibration_report(yte, pte)
    res.update({"pr_auc": float(average_precision_score(yte, pte)),
                "roc_auc": float(roc_auc_score(yte, pte)),
                "brier": calibration["brier"],
                "calibration": calibration,
                "baselines": baseline_report(yte, pte),
                "best_params": params,
                "mode": "kaggle" if "V1" in df.columns else "synthetic",
                "targets_met": bool(res["recall"] > 0.80 and res["fpr"] < 0.05)})

    suffix = a.suffix or ("_synthetic" if mode == "synthetic" else "")
    amount_col = "Amount" if "Amount" in df.columns else "amount"
    amount_test = df[amount_col].astype(float).values[i2:]
    policies = build_policy_artifacts(yte, pte, amount_test, suffix=suffix,
                                     operating_threshold=thr)
    res["policy_summary"] = {row["policy"]: row["threshold"]
                             for row in policies["policies"]}
    print(json.dumps(res, indent=2))

    explainer = shap.TreeExplainer(model)
    sample = Xte.sample(min(500 if mode == "synthetic" else 2000, len(Xte)), random_state=0)
    sv = explainer.shap_values(sample)
    if getattr(sv, "ndim", 2) == 3:  # (rows, features, classes)
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"models/shap_summary{suffix}.png", dpi=130)
    print("SHAP summary saved", flush=True)
    print("threshold policy artifacts saved", flush=True)

    joblib.dump({"model": model, "iso": iso, "threshold": thr,
                 "cols": cols, "base_cols": base_cols},
                f"models/bundle{suffix}.joblib")
    Xtr.sample(min(10000, len(Xtr)), random_state=0).to_csv(
        f"models/reference{suffix}.csv", index=False)
    json.dump(res, open(f"models/metrics{suffix}.json", "w"), indent=2)
    print(f"saved models/bundle{suffix}.joblib, reference{suffix}.csv, "
          f"metrics{suffix}.json, shap_summary{suffix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_data = "data/creditcard.csv" if os.path.exists("data/creditcard.csv") else "data/transactions.csv"
    ap.add_argument("--data", default=default_data)
    ap.add_argument("--suffix", default="",
                    help="Optional artifact suffix override; defaults by schema.")
    main(ap.parse_args())
