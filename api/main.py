"""FastAPI real-time scoring service.

Two supervised models are deployed side by side: the Kaggle PCA model
(V1-V28) and the transaction model (timestamp/amount/category fields). On top
of the batch model the service exposes:

- ``/agents/analyze``: transaction, identity, and mule-account agents;
- ``/cheque/analyze``: AWS Bedrock VLM cheque analysis with image forensics;
- ``/online/*``: River online learning, adaptive threshold, and ADWIN drift;
- ``/federated/status``: the latest secure-aggregation training report.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json, time, os
from datetime import datetime
from typing import Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.agents import AgentContext, DEFAULT_WEIGHTS, FraudAgentOrchestrator, agent_catalog
from src.cheque_vlm import (
    MAX_IMAGE_BYTES,
    ChequeExpected,
    analyze_cheque,
    bedrock_configuration,
)
from src.features import build_features, TYPES, MCC, COUNTRIES
from src.drift import drift_report
from src.federated import load_report as load_federated_report
from src.online_learning import OnlineFraudLearner
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
FEDERATED_REPORT_PATH = os.getenv(
    "FEDERATED_REPORT_PATH", "models/federated_report.json"
)
V_FIELDS = tuple(f"V{i}" for i in range(1, 29))

PRED = Counter("fraud_predictions_total", "Predictions", ["decision"])
LAT = Histogram("fraud_inference_seconds", "Inference latency",
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1))
AGENT_RUNS = Counter(
    "fraud_agent_runs_total", "Multi-agent decisions", ["decision", "primary_type"]
)
CHEQUE_ANALYSES = Counter(
    "cheque_analyses_total", "Cheque VLM analyses", ["backend", "decision"]
)
ONLINE_UPDATES = Counter(
    "online_learning_updates_total", "River online updates", ["label"]
)
ONLINE_DRIFT = Counter("online_drift_events_total", "ADWIN drift events")
ONLINE_SAMPLES = Gauge("online_learning_samples", "Samples seen by the online learner")

ORCHESTRATOR = FraudAgentOrchestrator()
ONLINE = OnlineFraudLearner()

app = FastAPI(
    title="SentinelGuard",
    version="1.2",
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


class IdentitySignals(BaseModel):
    kyc_status: Literal["verified", "pending", "failed", "review", "unknown"] = "verified"
    identity_mismatch: bool = False
    device_accounts_7d: int = Field(default=1, ge=0, le=10_000)
    email_accounts_7d: int = Field(default=1, ge=0, le=10_000)
    phone_accounts_7d: int = Field(default=1, ge=0, le=10_000)
    ip_country_mismatch: bool = False
    sanctions_hit: bool = False
    pep_hit: bool = False
    synthetic_identity_score: float = Field(default=0.0, ge=0, le=1)
    document_tamper_score: float = Field(default=0.0, ge=0, le=1)
    account_age_days: Optional[int] = Field(default=None, ge=0)


class MuleSignals(BaseModel):
    inbound_count_24h: int = Field(default=0, ge=0, le=100_000)
    outbound_count_24h: int = Field(default=0, ge=0, le=100_000)
    unique_senders_7d: int = Field(default=0, ge=0, le=100_000)
    unique_receivers_7d: int = Field(default=0, ge=0, le=100_000)
    pass_through_ratio: float = Field(default=0.0, ge=0, le=1)
    dormant_days_before_activity: int = Field(default=0, ge=0, le=100_000)
    round_amount_ratio: float = Field(default=0.0, ge=0, le=1)
    shared_device_accounts: int = Field(default=1, ge=0, le=10_000)
    beneficiary_risk: float = Field(default=0.0, ge=0, le=1)
    avg_outbound_amount: float = Field(default=0.0, ge=0)
    near_threshold_ratio: float = Field(default=0.0, ge=0, le=1)
    account_age_days: Optional[int] = Field(default=None, ge=0)
    tx_count_24h: Optional[int] = Field(default=None, ge=0)


class ChequeSignals(BaseModel):
    risk_score: float = Field(default=0.0, ge=0, le=1)
    tamper_indicators: list[str] = Field(default_factory=list, max_length=50)
    signature_consistent: Optional[bool] = None
    amount_mismatch: bool = False


class AgentRequest(BaseModel):
    transaction: Tx = Field(default_factory=Tx)
    identity: IdentitySignals = Field(default_factory=IdentitySignals)
    mule: MuleSignals = Field(default_factory=MuleSignals)
    cheque: Optional[ChequeSignals] = None
    include_model_score: bool = True
    include_online_score: bool = True
    threshold: float = Field(default=0.55, ge=0, le=1)


class OnlinePredictRequest(BaseModel):
    transaction: Tx


class OnlineLearnRequest(BaseModel):
    transaction: Tx
    label: Literal[0, 1]
    weight: Optional[float] = Field(default=None, gt=0, le=1000)


class OnlineBootstrapRequest(BaseModel):
    rows: int = Field(default=3000, ge=100, le=50_000)
    reset: bool = True


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


def explain(ctx: dict, X: pd.DataFrame, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return top SHAP drivers and stable analyst reason codes."""
    sv = ctx["explainer"].shap_values(X)
    if getattr(sv, "ndim", 2) == 3:  # (rows, features, classes)
        sv = sv[:, :, 1]
    sv = sv[0]
    top = np.argsort(-np.abs(sv))[:4]
    reasons = [
        {
            "feature": ctx["cols"][i],
            "value": float(X.iloc[0, i]),
            "shap": round(float(sv[i]), 4),
            "direction": "raises risk" if sv[i] > 0 else "lowers risk",
        }
        for i in top
    ]
    return reasons, reason_codes(reasons, mode)


def tx_dict(tx: Tx) -> dict[str, Any]:
    return tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()


@app.get("/health")
def health():
    federation = load_federated_report(FEDERATED_REPORT_PATH)
    return {
        "status": "ok",
        "threshold": THR,
        "models": {
            mode: {"threshold": float(ctx["threshold"])}
            for mode, ctx in BUNDLES.items()
        },
        "capabilities": {
            "federated_learning": {
                "report_available": federation is not None,
                "last_run": federation.get("generated_at") if federation else None,
                "secure_aggregation": (
                    federation.get("privacy", {}).get("secure_aggregation")
                    if federation
                    else None
                ),
            },
            "online_learning": {
                "available": ONLINE.available,
                "river_version": ONLINE.status().get("river_version"),
                "samples_seen": ONLINE.samples_seen,
                "warm": ONLINE.warm,
            },
            "cheque_vlm": bedrock_configuration(),
            "agents": [item["name"] for item in agent_catalog()],
        },
    }


@app.post("/predict")
def predict(tx: Tx, include_online: bool = False):
    t0 = time.perf_counter()
    mode = schema_of(tx)
    ctx = ctx_for(mode)
    X = featurize(tx, ctx)
    p = float(ctx["model"].predict_proba(X)[0, 1])
    reasons, codes = explain(ctx, X, mode)
    decision = "BLOCK/REVIEW" if p >= ctx["threshold"] else "APPROVE"
    ms = (time.perf_counter() - t0) * 1000
    PRED.labels(decision).inc()
    LAT.observe(ms / 1000)
    row = X.iloc[0].to_dict()
    row.update({"ts": time.time(), "score": p, "decision": decision, "schema": mode})
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    payload: dict[str, Any] = {
        "fraud_score": round(p, 4),
        "decision": decision,
        "threshold": float(ctx["threshold"]),
        "schema": mode,
        "reasons": reasons,
        "reason_codes": codes,
        "reason_code_summary": format_reason_codes(codes),
        "latency_ms": round(ms, 2),
    }
    if include_online and mode == "synthetic" and ONLINE.available:
        try:
            online = ONLINE.predict(tx_dict(tx))
            payload["online_score"] = online["online_score"]
            payload["online_threshold"] = online["threshold"]
            payload["online_warm"] = online["warm"]
        except Exception:
            payload["online_score"] = None
    return payload


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


@app.get("/agents/catalog")
def agents_catalog():
    """Specialist agents, their mandates, and the orchestrator's fusion weights."""
    return {
        "threshold": ORCHESTRATOR.threshold,
        "weights": DEFAULT_WEIGHTS,
        "agents": agent_catalog(),
    }


@app.post("/agents/analyze")
def agents_analyze(req: AgentRequest):
    """Run the transaction, identity, and mule-account agents on one case.

    The batch model score and its SHAP reason codes are optional evidence; the
    agents still produce a decision from rules when the model is unavailable or
    the caller sets ``include_model_score`` to false. The online River score is
    fused in only after the learner is warm.
    """
    t0 = time.perf_counter()
    tx = req.transaction
    data = tx_dict(tx)
    mode = schema_of(tx)
    model_score = None
    model_reasons: list[dict[str, Any]] = []
    if req.include_model_score and mode in BUNDLES:
        try:
            ctx = BUNDLES[mode]
            X = featurize(tx, ctx)
            model_score = float(ctx["model"].predict_proba(X)[0, 1])
            _, model_reasons = explain(ctx, X, mode)
        except Exception:
            model_score = None
            model_reasons = []

    online_score = None
    if req.include_online_score and ONLINE.available and ONLINE.warm:
        try:
            online_score = float(ONLINE.predict(data)["online_score"])
        except Exception:
            online_score = None

    context = AgentContext(
        transaction=data,
        model_score=model_score,
        model_reasons=model_reasons,
        identity=req.identity.model_dump(exclude_none=False),
        mule=req.mule.model_dump(exclude_none=False),
        cheque=req.cheque.model_dump() if req.cheque is not None else {},
        online_score=online_score,
        threshold=req.threshold,
    )
    result = ORCHESTRATOR.analyze(context, threshold=req.threshold)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    result["schema"] = mode
    AGENT_RUNS.labels(result["decision"], result["primary_fraud_type"]).inc()
    return result


@app.post("/cheque/analyze")
def cheque_analyze(
    file: UploadFile = File(...),
    expected_amount: Optional[float] = Form(default=None),
    expected_payee: Optional[str] = Form(default=None),
    account_last4: Optional[str] = Form(default=None),
    expected_date: Optional[str] = Form(default=None),
    backend: Optional[str] = Form(default=None),
):
    """Analyze a cheque image with the Bedrock VLM plus deterministic forensics.

    Only JPEG, PNG, WEBP, and GIF are accepted. The raw image is never written
    to disk; the audit log stores the SHA-256 digest and the decision.
    """
    if file.size is not None and file.size > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Cheque image is larger than the "
                f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
            ),
        )
    raw = file.file.read()
    expected = ChequeExpected(
        amount=expected_amount,
        payee=expected_payee,
        account_last4=account_last4,
        date=expected_date,
    )
    try:
        result = analyze_cheque(
            raw,
            content_type=file.content_type,
            expected=expected,
            backend=backend,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    CHEQUE_ANALYSES.labels(result["backend"], result["decision"]).inc()
    return result


@app.get("/online/status")
def online_status():
    return ONLINE.status()


@app.post("/online/predict")
def online_predict(req: OnlinePredictRequest):
    if schema_of(req.transaction) != "synthetic":
        raise HTTPException(
            status_code=400,
            detail="The online learner supports the transaction schema, not V1-V28 rows.",
        )
    if not ONLINE.available:
        raise HTTPException(
            status_code=503,
            detail="River is not installed; online learning is unavailable.",
        )
    return ONLINE.predict(tx_dict(req.transaction))


@app.post("/online/learn")
def online_learn(req: OnlineLearnRequest):
    if schema_of(req.transaction) != "synthetic":
        raise HTTPException(
            status_code=400,
            detail="The online learner supports the transaction schema, not V1-V28 rows.",
        )
    try:
        result = ONLINE.learn(
            tx_dict(req.transaction),
            int(req.label),
            weight=req.weight,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ONLINE_UPDATES.labels("fraud" if req.label else "legit").inc()
    if result["event"].get("drift_detected"):
        ONLINE_DRIFT.inc()
    ONLINE_SAMPLES.set(ONLINE.samples_seen)
    return result


@app.post("/online/bootstrap")
def online_bootstrap(req: OnlineBootstrapRequest):
    """Warm-start the online learner from local history or synthetic data."""
    if not ONLINE.available:
        raise HTTPException(
            status_code=503,
            detail="River is not installed; online learning is unavailable.",
        )
    if req.reset:
        ONLINE.reset()
    path = os.getenv("ONLINE_BOOTSTRAP_PATH", "data/transactions.csv")
    try:
        if os.path.exists(path):
            frame = pd.read_csv(path, nrows=req.rows)
        else:
            from src.make_data import make

            frame = make(req.rows)
        status = ONLINE.bootstrap(frame, max_rows=req.rows)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ONLINE_SAMPLES.set(ONLINE.samples_seen)
    status["source"] = path if os.path.exists(path) else "synthetic"
    return status


@app.get("/federated/status")
def federated_status():
    """Latest federated-training report, if one has been generated."""
    report = load_federated_report(FEDERATED_REPORT_PATH)
    if report is None:
        return {
            "available": False,
            "report_path": FEDERATED_REPORT_PATH,
            "message": (
                "No federated report yet. Run "
                "`python -m src.federated --demo --clients 5 --rounds 5`."
            ),
        }
    report = dict(report)
    report["available"] = True
    report["report_path"] = FEDERATED_REPORT_PATH
    return report


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
