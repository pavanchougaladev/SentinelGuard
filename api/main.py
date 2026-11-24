"""FastAPI real-time scoring service with SHAP reasons, Prometheus metrics and drift check.

Two models are deployed side by side: the Kaggle PCA model (V1-V28) and the
transaction model (timestamp/amount/category fields). Requests are routed to
the model that matches their schema.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json, time, os
from datetime import datetime
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.features import build_features, TYPES, MCC, COUNTRIES
from src.drift import drift_report
from src.reason_codes import format_reason_codes, reason_codes

BUNDLE_PATHS = {
    "kaggle": ("models/bundle.joblib", "models/reference.csv"),
    "synthetic": ("models/bundle_synthetic.joblib", "models/reference_synthetic.csv"),
}
BUNDLES: dict[str, dict] = {}
for _mode, (_bundle_path, _reference_path) in BUNDLE_PATHS.items():
    if not os.path.exists(_bundle_path):
        continue
    _bundle = joblib.load(_bundle_path)
    BUNDLES[_mode] = {
        "model": _bundle["model"],
        "iso": _bundle["iso"],
        "threshold": _bundle["threshold"],
        "cols": _bundle["cols"],
        "base_cols": _bundle["base_cols"],
        "explainer": shap.TreeExplainer(_bundle["model"]),
        "ref": (
            pd.read_csv(_reference_path)
            if os.path.exists(_reference_path)
            else pd.DataFrame()
        ),
    }
if not BUNDLES:
    raise RuntimeError("No model bundles found in models/")

PRIMARY = "kaggle" if "kaggle" in BUNDLES else next(iter(BUNDLES))
THR = BUNDLES[PRIMARY]["threshold"]  # kept for health-check compatibility
LOG_PATH = "models/predictions.jsonl"
V_FIELDS = tuple(f"V{i}" for i in range(1, 29))

PRED = Counter("fraud_predictions_total", "Predictions", ["decision"])
LAT = Histogram("fraud_inference_seconds", "Inference latency",
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1))

app = FastAPI(
    title="SentinelGuard",
    version="1.1",
    root_path=os.getenv("API_ROOT_PATH", ""),
)

TxType = Literal["purchase", "transfer", "withdrawal", "online"]
Mcc = Literal["grocery", "fuel", "electronics", "travel", "dining",
              "gambling", "crypto", "utilities", "clothing", "other"]
Country = Literal["IN", "US", "GB", "AE", "SG", "NG", "RU", "CN"]


class Tx(BaseModel):
    model_config = {"extra": "allow"}
    amount: float = Field(gt=0, default=100.0)
    tx_type: Optional[TxType] = "purchase"
    merchant_category: Optional[Mcc] = "grocery"
    country: Optional[Country] = "IN"
    device_risk: Optional[float] = Field(ge=0, le=1, default=0.2)
    ip_risk: Optional[float] = Field(ge=0, le=1, default=0.2)
    account_age_days: Optional[int] = Field(ge=0, default=365)
    tx_count_24h: Optional[int] = Field(ge=0, default=1)
    timestamp: Optional[datetime] = None


class BatchTx(Tx):
    """Batch transaction; zero amounts are allowed for historical uploads."""

    amount: float = Field(ge=0, default=100.0)


class BatchRequest(BaseModel):
    transactions: list[BatchTx] = Field(min_length=1, max_length=5000)


def schema_of(tx: Tx) -> str:
    """A PCA field means the Kaggle schema; otherwise it is a transaction."""
    data = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
    return "kaggle" if any(data.get(name) is not None for name in V_FIELDS) else "synthetic"


def ctx_for(mode: str) -> dict:
    ctx = BUNDLES.get(mode)
    if ctx is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model is deployed for the {mode} schema.",
        )
    return ctx


def tx_frame(txs: list[Tx]) -> pd.DataFrame:
    rows = []
    for tx in txs:
        d = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
        if d.get("timestamp") is None:
            d["timestamp"] = datetime.utcnow()
        if "amount" in d and "Amount" not in d:
            d["Amount"] = d["amount"]
        elif "Amount" in d and "amount" not in d:
            d["amount"] = d["Amount"]
        rows.append(d)
    return pd.DataFrame(rows)


def featurize_frame(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    X = build_features(df)
    for name in ctx["base_cols"]:
        if name not in X.columns:
            X[name] = 0.0
    X["anomaly"] = -ctx["iso"].score_samples(X[ctx["base_cols"]])
    return X[ctx["cols"]]


def featurize(tx: Tx, ctx: dict) -> pd.DataFrame:
    return featurize_frame(tx_frame([tx]), ctx)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": THR,
        "models": {
            mode: {"threshold": float(ctx["threshold"])}
            for mode, ctx in BUNDLES.items()
        },
    }


@app.post("/predict")
def predict(tx: Tx):
    t0 = time.perf_counter()
    mode = schema_of(tx)
    ctx = ctx_for(mode)
    X = featurize(tx, ctx)
    p = float(ctx["model"].predict_proba(X)[0, 1])
    sv = ctx["explainer"].shap_values(X)
    if getattr(sv, "ndim", 2) == 3:  # (rows, features, classes)
        sv = sv[:, :, 1]
    sv = sv[0]
    top = np.argsort(-np.abs(sv))[:4]
    reasons = [{"feature": ctx["cols"][i], "value": float(X.iloc[0, i]),
                "shap": round(float(sv[i]), 4),
                "direction": "raises risk" if sv[i] > 0 else "lowers risk"} for i in top]
    decision = "BLOCK/REVIEW" if p >= ctx["threshold"] else "APPROVE"
    ms = (time.perf_counter() - t0) * 1000
    PRED.labels(decision).inc()
    LAT.observe(ms / 1000)
    codes = reason_codes(reasons, mode)
    row = X.iloc[0].to_dict()
    row.update({"ts": time.time(), "score": p, "decision": decision, "schema": mode})
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    return {"fraud_score": round(p, 4), "decision": decision,
            "threshold": float(ctx["threshold"]), "schema": mode,
            "reasons": reasons, "reason_codes": codes,
            "reason_code_summary": format_reason_codes(codes),
            "latency_ms": round(ms, 2)}


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    """Score up to 5,000 transactions, routing each row to its schema's model.

    Batch scoring is intended for uploaded files and backtesting, so it does
    not append to the live prediction log.
    """
    t0 = time.perf_counter()
    txs = list(batch.transactions)
    modes = [schema_of(tx) for tx in txs]
    df = tx_frame(txs)
    scores = np.zeros(len(df), dtype=float)
    decisions = np.empty(len(df), dtype=object)
    thresholds: dict[str, float] = {}
    for mode in dict.fromkeys(modes):
        ctx = ctx_for(mode)
        index = np.flatnonzero(np.array(modes) == mode)
        X = featurize_frame(df.iloc[index].reset_index(drop=True), ctx)
        mode_scores = ctx["model"].predict_proba(X)[:, 1]
        scores[index] = mode_scores
        decisions[index] = np.where(
            mode_scores >= ctx["threshold"], "BLOCK/REVIEW", "APPROVE"
        )
        thresholds[mode] = float(ctx["threshold"])

    approved = int(np.sum(decisions == "APPROVE"))
    PRED.labels("APPROVE").inc(approved)
    PRED.labels("BLOCK/REVIEW").inc(int(len(decisions) - approved))
    ms = (time.perf_counter() - t0) * 1000
    LAT.observe(ms / 1000)
    unique = list(dict.fromkeys(modes))
    return {
        "threshold": thresholds[unique[0]],
        "thresholds": thresholds,
        "schema": unique[0] if len(unique) == 1 else "mixed",
        "count": int(len(df)),
        "scores": [round(float(s), 6) for s in scores],
        "decisions": decisions.tolist(),
        "latency_ms": round(ms, 2),
    }


@app.get("/policy")
def policy(mode: Optional[str] = None):
    """Cost-sensitive threshold policies generated by the training pipeline."""
    mode = mode if mode in BUNDLES else PRIMARY
    suffix = "_synthetic" if mode == "synthetic" else ""
    path = f"models/threshold_policy{suffix}.json"
    if not os.path.exists(path):
        return {"message": f"no threshold policy artifact for the {mode} model",
                "mode": mode}
    with open(path) as f:
        payload = json.load(f)
    payload["mode"] = mode
    return payload


@app.get("/drift")
def drift(last_n: int = 500, mode: Optional[str] = None):
    mode = mode if mode in BUNDLES else PRIMARY
    ctx = BUNDLES[mode]
    if not os.path.exists(LOG_PATH):
        return {"message": "no predictions logged yet"}
    cur = pd.read_json(LOG_PATH, lines=True)
    if "schema" in cur.columns:
        cur = cur[cur["schema"] == mode]
    elif mode == "kaggle" and "V1" in cur.columns:
        cur = cur[cur["V1"].notna()]
    elif mode == "synthetic" and "V1" in cur.columns:
        cur = cur[cur["V1"].isna()]
    cur = cur.tail(last_n)
    if len(cur) < 50:
        return {"message": f"need >=50 logged predictions, have {len(cur)}"}

    ref = ctx["ref"]
    if ref.empty:
        return {"message": f"reference sample missing for the {mode} model"}
    cols = [c for c in ["log_amount", "device_risk", "ip_risk", "account_age_days",
            "tx_count_24h", "anomaly", "V1", "V2", "V3", "V4"]
            if c in ref.columns and c in cur.columns]
    if not cols:
        return {"message": f"no shared drift features for the {mode} model"}
    rep = drift_report(ref, cur, cols)
    rep["_retrain_recommended"] = any(v["status"] == "ALERT" for v in rep.values())
    return rep


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
