"""Data access and analytics helpers for the SentinelGuard dashboard.

Everything that touches disk, the model bundle, or the scoring API lives here so
the page code in ``streamlit_app.py`` stays focused on presentation.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import COUNTRIES, MCC, TYPES  # noqa: E402

MODELS = ROOT / "models"
DATA = ROOT / "data"
LOG_PATH = MODELS / "predictions.jsonl"
MODEL_SCHEMAS = ("kaggle", "synthetic")


def _artifact_paths(mode: str) -> dict[str, Path]:
    """Artifact filenames for one deployed schema."""
    suffix = "_synthetic" if mode == "synthetic" else ""
    return {
        "bundle": MODELS / f"bundle{suffix}.joblib",
        "metrics": MODELS / f"metrics{suffix}.json",
        "reference": MODELS / f"reference{suffix}.csv",
        "shap": MODELS / f"shap_summary{suffix}.png",
    }


def shap_image_path(mode: str | None = None) -> Path:
    return _artifact_paths(mode or dataset_mode())["shap"]

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# PSI interpretation thresholds (industry-standard rule of thumb).
PSI_WARN = 0.10
PSI_ALERT = 0.25

# Risk bands used across the dashboard.
BAND_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0001]
BAND_LABELS = ["0-2%", "2-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
DECISION_APPROVE = "APPROVE"
DECISION_BLOCK = "BLOCK/REVIEW"

# Largest upload accepted by the batch endpoint.
BATCH_MAX_ROWS = 5000


# ---------------------------------------------------------------------------
# Files and artifacts
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_metrics(mode: str, stamp: float) -> dict[str, Any]:
    try:
        return json.loads(_artifact_paths(mode)["metrics"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_metrics() -> dict[str, Any]:
    """Metrics for the selected model, reloaded when the file changes."""
    mode = dataset_mode()
    try:
        stamp = _artifact_paths(mode)["metrics"].stat().st_mtime
    except OSError:
        return {}
    return _load_metrics(mode, stamp)


@st.cache_data(show_spinner=False)
def _load_reference(mode: str, stamp: float, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(_artifact_paths(mode)["reference"], nrows=nrows)
    except (OSError, ValueError):
        return pd.DataFrame()


def load_reference(
    nrows: int | None = None, mode: str | None = None
) -> pd.DataFrame:
    """Reference feature sample for a schema, defaulting to the active model."""
    mode = mode or dataset_mode()
    try:
        stamp = _artifact_paths(mode)["reference"].stat().st_mtime
    except OSError:
        return pd.DataFrame()
    return _load_reference(mode, stamp, nrows)


@st.cache_data(show_spinner=False)
def _load_policy(mode: str, stamp: float) -> dict[str, Any]:
    suffix = "_synthetic" if mode == "synthetic" else ""
    try:
        return json.loads(
            (MODELS / f"threshold_policy{suffix}.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}


def load_policy(mode: str | None = None) -> dict[str, Any]:
    """Cost-sensitive threshold policies written by the training pipeline."""
    mode = mode or dataset_mode()
    suffix = "_synthetic" if mode == "synthetic" else ""
    path = MODELS / f"threshold_policy{suffix}.json"
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return {}
    return _load_policy(mode, stamp)


def policy_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the policy artifact into display rows, operating cutoff first."""
    rows: list[dict[str, Any]] = []
    for row in [payload.get("operating")] + list(payload.get("policies", [])):
        if not row:
            continue
        rows.append(
            {
                "policy": str(row.get("policy", "")),
                "purpose": str(row.get("purpose", "")),
                "threshold": float(row.get("threshold", 0.0)),
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "fpr": float(row.get("fpr", 0.0)),
                "flagged_rate": float(row.get("flagged_rate", 0.0)),
                "alerts": int(row.get("alerts", 0)),
                "cost": float(row.get("cost", 0.0)),
            }
        )
    return rows


def dataset_mode() -> str:
    """Schema selected in the sidebar, falling back to an available model."""
    try:
        mode = st.session_state.get("model_schema", "kaggle")
    except Exception:  # bare scripts and tests have no Streamlit session
        mode = "kaggle"
    if mode not in MODEL_SCHEMAS:
        mode = "kaggle"
    if not _artifact_paths(mode)["bundle"].exists():
        available = [m for m in MODEL_SCHEMAS if _artifact_paths(m)["bundle"].exists()]
        if available:
            mode = available[0]
    return mode


@st.cache_data(show_spinner=False)
def _artifact_age(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).strftime("%d %b %Y · %H:%M")


def artifact_age() -> str:
    """Human-readable age of the selected model bundle."""
    try:
        stamp = _artifact_paths(dataset_mode())["bundle"].stat().st_mtime
    except OSError:
        return "unknown"
    return _artifact_age(stamp)


@st.cache_data(ttl="5s", show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Prediction log written by the API, newest rows last."""
    if not LOG_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_json(LOG_PATH, lines=True)
    except (OSError, ValueError):
        return pd.DataFrame()
    if df.empty:
        return df
    if "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    if "log_amount" in df.columns:
        df["amount"] = np.expm1(df["log_amount"].astype(float))
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def clear_prediction_cache() -> None:
    load_predictions.clear()


@st.cache_resource(show_spinner=False)
def _load_bundle(mode: str, stamp: float) -> dict[str, Any] | None:
    try:
        import joblib

        return joblib.load(_artifact_paths(mode)["bundle"])
    except Exception:
        return None


def load_bundle() -> dict[str, Any] | None:
    """Load the selected bundle, reloading it after a retrain."""
    mode = dataset_mode()
    try:
        stamp = _artifact_paths(mode)["bundle"].stat().st_mtime
    except OSError:
        return None
    return _load_bundle(mode, stamp)


# ---------------------------------------------------------------------------
# Held-out evaluation analytics
# ---------------------------------------------------------------------------


def _downsample(x: np.ndarray, y: np.ndarray, size: int = 300) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= size:
        return x, y
    idx = np.linspace(0, len(x) - 1, size).astype(int)
    return x[idx], y[idx]


def load_eval_analytics() -> dict[str, Any] | None:
    """Held-out curves and confusion data for the selected model."""
    mode = dataset_mode()
    try:
        stamp = _artifact_paths(mode)["bundle"].stat().st_mtime
    except OSError:
        return None
    return _load_eval_analytics(mode, stamp)


@st.cache_data(show_spinner=False)
def _load_eval_analytics(mode: str, bundle_stamp: float) -> dict[str, Any] | None:
    """Score the held-out test split to build curves and score distributions.

    This mirrors the split in ``src/train.py`` (last 15% of the time-ordered
    dataset). It is cached per bundle version, so the ~1–3 s cost is paid once
    per retrain. Returns ``None`` when the bundle or dataset is unavailable.
    """
    bundle = _load_bundle(mode, bundle_stamp)
    if bundle is None:
        return None

    try:
        from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

        from src.features import build_features

        path = DATA / ("creditcard.csv" if mode == "kaggle" else "transactions.csv")
        if not path.exists():
            return None

        df = pd.read_csv(path)
        time_col = "Time" if "Time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        if time_col is not None:
            df = df.sort_values(time_col).reset_index(drop=True)

        split = int(len(df) * 0.85)
        test = df.iloc[split:].reset_index(drop=True)
        target_col = "Class" if "Class" in test.columns else "is_fraud"
        if target_col not in test.columns:
            return None

        model, iso = bundle["model"], bundle["iso"]
        base_cols, cols = bundle["base_cols"], bundle["cols"]

        X = build_features(test)
        for name in base_cols:
            if name not in X.columns:
                X[name] = 0.0
        X["anomaly"] = -iso.score_samples(X[base_cols])
        X = X[cols]

        y = test[target_col].astype(int).to_numpy()
        scores = model.predict_proba(X)[:, 1]
        threshold = float(bundle["threshold"])
        amount_col = "Amount" if "Amount" in test.columns else "amount"
        amounts = test[amount_col].astype(float).to_numpy()

        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        fpr_ds, tpr_ds = _downsample(fpr, tpr)
        recall_ds, precision_ds = _downsample(recall[::-1], precision[::-1])

        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        prevalence = float((tp + fn) / max(len(y), 1))

        # Risk bands: share of each class that lands in each score band.
        band_index = np.digitize(scores, BAND_EDGES[1:-1], right=False)
        bands = pd.DataFrame({"band": BAND_LABELS, "legitimate": 0.0, "fraud": 0.0})
        for idx, label in enumerate(BAND_LABELS):
            in_band = band_index == idx
            legit = int(np.sum(in_band & (y == 0)))
            fraud = int(np.sum(in_band & (y == 1)))
            total_legit = max(int(np.sum(y == 0)), 1)
            total_fraud = max(int(np.sum(y == 1)), 1)
            bands.loc[idx, "legitimate"] = round(legit / total_legit * 100, 2)
            bands.loc[idx, "fraud"] = round(fraud / total_fraud * 100, 2)

        caught_amount = float(amounts[(y == 1) & (predicted == 1)].sum())
        missed_amount = float(amounts[(y == 1) & (predicted == 0)].sum())

        return {
            "mode": mode,
            "n_test": int(len(y)),
            "n_fraud": int((tp + fn)),
            "threshold": threshold,
            "roc": pd.DataFrame(
                {
                    "fpr": fpr_ds,
                    "tpr": tpr_ds,
                    "baseline": fpr_ds,
                }
            ),
            "pr": pd.DataFrame(
                {
                    "recall": recall_ds,
                    "precision": precision_ds,
                    "prevalence": np.full_like(recall_ds, prevalence),
                }
            ),
            "bands": bands,
            "confusion": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "recall": float(tp / max(tp + fn, 1)),
                "precision": float(tp / max(tp + fp, 1)),
                "fpr": float(fp / max(fp + tn, 1)),
                "specificity": float(tn / max(tn + fp, 1)),
                "accuracy": float((tp + tn) / max(len(y), 1)),
                "prevalence": prevalence,
                "alerts": int(predicted.sum()),
                "alerts_per_10k": float(fp / max(tn + fp, 1) * 10_000),
            },
            "business": {
                "caught_amount": caught_amount,
                "missed_amount": missed_amount,
                "recovery_rate": float(caught_amount / max(caught_amount + missed_amount, 1e-9)),
            },
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring API client
# ---------------------------------------------------------------------------


@st.cache_data(ttl="3s", show_spinner=False)
def api_health() -> tuple[bool, dict[str, Any]]:
    """Cheap liveness probe for the FastAPI scoring service."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.ok:
            return True, response.json()
        return False, {"error": f"HTTP {response.status_code}"}
    except requests.RequestException as exc:
        return False, {"error": type(exc).__name__}


def score_transaction(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """POST one transaction to the scoring API.

    Returns ``(result, error)``. Never raises, so the dashboard can show a
    friendly message when the API is offline.
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    except requests.RequestException as exc:
        return None, (
            f"Could not reach the scoring API at {API_URL} "
            f"({type(exc).__name__}). Start it with "
            "`uvicorn api.main:app --port 8000`."
        )
    if not response.ok:
        detail = response.text[:300].strip() or f"HTTP {response.status_code}"
        return None, f"The scoring API returned an error: {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The scoring API returned a response that was not valid JSON."


def fetch_drift(
    last_n: int = 500, mode: str | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch the PSI drift report for the most recent ``last_n`` predictions."""
    try:
        params: dict[str, Any] = {"last_n": last_n}
        if mode:
            params["mode"] = mode
        response = requests.get(f"{API_URL}/drift", params=params, timeout=20)
    except requests.RequestException as exc:
        return None, (
            f"Could not reach the drift endpoint at {API_URL} "
            f"({type(exc).__name__})."
        )
    if not response.ok:
        return None, f"The drift endpoint returned HTTP {response.status_code}."
    try:
        return response.json(), None
    except ValueError:
        return None, "The drift endpoint returned invalid JSON."


# ---------------------------------------------------------------------------
# Advanced capability clients: agents, cheque VLM, online learning, federation
# ---------------------------------------------------------------------------


def _api_get(path: str, *, timeout: float = 20) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
    except requests.RequestException as exc:
        return None, (
            f"Could not reach {API_URL}{path} ({type(exc).__name__}). "
            "Start the API with `uvicorn api.main:app --port 8000`."
        )
    if not response.ok:
        detail = response.text[:300].strip() or f"HTTP {response.status_code}"
        return None, f"The API returned an error: {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The API returned invalid JSON."


def _api_post(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return None, (
            f"Could not reach {API_URL}{path} ({type(exc).__name__}). "
            "Start the API with `uvicorn api.main:app --port 8000`."
        )
    if not response.ok:
        detail = response.text[:300].strip() or f"HTTP {response.status_code}"
        return None, f"The API returned an error: {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The API returned invalid JSON."


def fetch_agent_catalog() -> tuple[dict[str, Any] | None, str | None]:
    """Catalog of specialist agents and their fusion weights."""
    return _api_get("/agents/catalog")


def analyze_agents(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the transaction, identity, and mule-account agents on one case."""
    return _api_post("/agents/analyze", payload, timeout=30)


def analyze_cheque_image(
    image_bytes: bytes,
    *,
    filename: str = "cheque.png",
    content_type: str = "image/png",
    expected_amount: float | None = None,
    expected_payee: str | None = None,
    account_last4: str | None = None,
    expected_date: str | None = None,
    backend: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Upload a cheque image to the Bedrock VLM endpoint."""
    data: dict[str, Any] = {}
    if expected_amount is not None:
        data["expected_amount"] = str(expected_amount)
    if expected_payee:
        data["expected_payee"] = expected_payee
    if account_last4:
        data["account_last4"] = account_last4
    if expected_date:
        data["expected_date"] = expected_date
    if backend:
        data["backend"] = backend
    try:
        response = requests.post(
            f"{API_URL}/cheque/analyze",
            files={"file": (filename, image_bytes, content_type)},
            data=data,
            timeout=90,
        )
    except requests.RequestException as exc:
        return None, (
            f"Could not reach {API_URL}/cheque/analyze ({type(exc).__name__})."
        )
    if not response.ok:
        detail = response.text[:300].strip() or f"HTTP {response.status_code}"
        return None, f"The cheque endpoint returned an error: {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The cheque endpoint returned invalid JSON."


def fetch_online_status() -> tuple[dict[str, Any] | None, str | None]:
    """River learner state, rolling metrics, adaptive threshold, and drift."""
    return _api_get("/online/status")


def online_predict(
    transaction: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    return _api_post("/online/predict", {"transaction": transaction}, timeout=20)


def online_learn(
    transaction: dict[str, Any],
    label: int,
) -> tuple[dict[str, Any] | None, str | None]:
    return _api_post(
        "/online/learn",
        {"transaction": transaction, "label": int(label)},
        timeout=20,
    )


def bootstrap_online(
    rows: int = 3000,
    *,
    reset: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    return _api_post(
        "/online/bootstrap",
        {"rows": int(rows), "reset": bool(reset)},
        timeout=180,
    )


def fetch_federated_status() -> tuple[dict[str, Any] | None, str | None]:
    """Latest secure-aggregation training report."""
    return _api_get("/federated/status", timeout=10)



# ---------------------------------------------------------------------------
# Demo traffic
# ---------------------------------------------------------------------------


def raw_dataset_path(mode: str | None = None) -> Path:
    mode = mode or dataset_mode()
    return DATA / ("creditcard.csv" if mode == "kaggle" else "transactions.csv")


@st.cache_data(show_spinner=False, ttl="10m")
def _load_traffic_sample(
    mode: str, stamp: float, n: int, seed: int
) -> pd.DataFrame:
    path = raw_dataset_path(mode)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def load_traffic_sample(n: int, seed: int) -> pd.DataFrame:
    """Sample raw transactions for the API traffic simulator."""
    mode = dataset_mode()
    path = raw_dataset_path(mode)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return pd.DataFrame()
    return _load_traffic_sample(mode, stamp, n, seed)


def row_to_payload(row: pd.Series, mode: str) -> dict[str, Any]:
    """Convert a raw dataset row into the API request schema."""
    if mode == "kaggle":
        payload: dict[str, Any] = {"amount": float(row["Amount"])}
        for i in range(1, 29):
            key = f"V{i}"
            if key in row:
                payload[key] = float(row[key])
        return payload
    return {
        "amount": float(row["amount"]),
        "tx_type": str(row["tx_type"]),
        "merchant_category": str(row["merchant_category"]),
        "country": str(row["country"]),
        "device_risk": float(row["device_risk"]),
        "ip_risk": float(row["ip_risk"]),
        "account_age_days": int(row["account_age_days"]),
        "tx_count_24h": int(row["tx_count_24h"]),
    }


def drift_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a drift report into rows, ignoring metadata keys."""
    rows: list[dict[str, Any]] = []
    for feature, value in report.items():
        if feature.startswith("_") or not isinstance(value, dict):
            continue
        if "psi" not in value:
            continue
        rows.append(
            {
                "feature": feature,
                "psi": float(value.get("psi", 0.0)),
                "status": str(value.get("status", "ok")),
            }
        )
    return sorted(rows, key=lambda item: item["psi"], reverse=True)


# ---------------------------------------------------------------------------
# Batch uploads
# ---------------------------------------------------------------------------


def detect_upload_schema(df: pd.DataFrame) -> str | None:
    """Identify an uploaded frame as the Kaggle PCA or synthetic schema."""
    columns = {str(column).strip() for column in df.columns}
    v_columns = [
        column
        for column in columns
        if column.startswith("V") and column[1:].isdigit()
    ]
    if v_columns and ("Amount" in columns or "amount" in columns):
        return "kaggle"
    required = {
        "amount",
        "tx_type",
        "merchant_category",
        "country",
        "device_risk",
        "ip_risk",
        "account_age_days",
        "tx_count_24h",
    }
    if required.issubset(columns):
        return "synthetic"
    return None


def _category_or_none(value: Any, allowed: list[str]) -> str | None:
    """Map a CSV cell to an allowed category, or ``None`` when unrecognised."""
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text in allowed else None


def prepare_upload(
    df: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Clean an uploaded frame and attach ``_amount`` / ``_label`` helpers.

    Returns ``(prepared, issues, errors)``. Issues are recoverable notes (rows
    dropped, columns filled); errors block scoring entirely.
    """
    issues: list[str] = []
    errors: list[str] = []
    data = df.copy()
    data.columns = [str(column).strip() for column in data.columns]

    if mode == "kaggle":
        amount_col = "Amount" if "Amount" in data.columns else "amount"
        v_present = [f"V{i}" for i in range(1, 29) if f"V{i}" in data.columns]
        if amount_col not in data.columns:
            errors.append("Missing the `Amount` column.")
        if not v_present:
            errors.append("No PCA components (`V1`–`V28`) were found.")
        if errors:
            return data, issues, errors

        missing = [f"V{i}" for i in range(1, 29) if f"V{i}" not in data.columns]
        if missing:
            issues.append(
                f"Using {len(v_present)} of 28 PCA components; "
                f"{len(missing)} missing columns were filled with 0."
            )

        numeric = [amount_col] + v_present
        for column in numeric:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid = data[numeric].isna().any(axis=1)
        if invalid.any():
            issues.append(
                f"Skipped {int(invalid.sum()):,} rows with missing or "
                "non-numeric amounts or PCA values."
            )
            data = data[~invalid].copy()
        negative = data[amount_col] < 0
        if negative.any():
            issues.append(
                f"Skipped {int(negative.sum()):,} rows with a negative amount."
            )
            data = data[~negative].copy()

        data["_amount"] = data[amount_col].astype(float)
        if "Class" in data.columns:
            label = pd.to_numeric(data["Class"], errors="coerce")
            data["_label"] = label.where(label.isin([0, 1]))
        return data.reset_index(drop=True), issues, errors

    required = [
        "amount",
        "tx_type",
        "merchant_category",
        "country",
        "device_risk",
        "ip_risk",
        "account_age_days",
        "tx_count_24h",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        errors.append(
            "Missing required columns: "
            + ", ".join(f"`{column}`" for column in missing)
            + "."
        )
        return data, issues, errors

    numeric = ["amount", "device_risk", "ip_risk", "account_age_days", "tx_count_24h"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = data[numeric].notna().all(axis=1)
    valid &= data["amount"] >= 0
    valid &= data["device_risk"].between(0, 1)
    valid &= data["ip_risk"].between(0, 1)
    valid &= data["account_age_days"] >= 0
    valid &= data["tx_count_24h"] >= 0
    dropped = int((~valid).sum())
    if dropped:
        issues.append(
            f"Skipped {dropped:,} rows with invalid amounts, risk scores, "
            "or account fields."
        )
        data = data[valid].copy()

    if "timestamp" in data.columns:
        parsed = pd.to_datetime(data["timestamp"], errors="coerce")
        unparsed = int(parsed.isna().sum())
        if unparsed:
            issues.append(
                f"{unparsed:,} rows had an unreadable timestamp; "
                "the current time was used instead."
            )
        data["timestamp"] = parsed
    else:
        issues.append(
            "No `timestamp` column; the current time was used for "
            "time-of-day features."
        )

    unknown = []
    for column, allowed in (
        ("tx_type", TYPES),
        ("merchant_category", MCC),
        ("country", COUNTRIES),
    ):
        count = int((~data[column].astype(str).isin(allowed)).sum())
        if count:
            unknown.append(f"{count:,} `{column}`")
    if unknown:
        issues.append(
            "Unrecognised categories were treated as unknown: "
            + ", ".join(unknown)
            + "."
        )

    data["_amount"] = data["amount"].astype(float)
    if "is_fraud" in data.columns:
        label = pd.to_numeric(data["is_fraud"], errors="coerce")
        data["_label"] = label.where(label.isin([0, 1]))
    return data.reset_index(drop=True), issues, errors


def upload_row_to_payload(row: pd.Series, mode: str) -> dict[str, Any]:
    """Convert one prepared upload row into the API request schema."""
    if mode == "kaggle":
        payload: dict[str, Any] = {"Amount": float(row["_amount"])}
        for i in range(1, 29):
            key = f"V{i}"
            if key in row.index and pd.notna(row[key]):
                payload[key] = float(row[key])
        return payload

    payload = {
        "amount": float(row["_amount"]),
        "tx_type": _category_or_none(row.get("tx_type"), TYPES),
        "merchant_category": _category_or_none(row.get("merchant_category"), MCC),
        "country": _category_or_none(row.get("country"), COUNTRIES),
        "device_risk": float(row["device_risk"]),
        "ip_risk": float(row["ip_risk"]),
        "account_age_days": int(float(row["account_age_days"])),
        "tx_count_24h": int(float(row["tx_count_24h"])),
    }
    stamp = row.get("timestamp")
    if stamp is not None and pd.notna(stamp):
        payload["timestamp"] = pd.Timestamp(stamp).isoformat()
    return payload


def score_batch(
    transactions: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """POST a batch of transactions and return a score for each row."""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"transactions": transactions},
            timeout=120,
        )
    except requests.RequestException as exc:
        return None, (
            f"Could not reach the batch endpoint at {API_URL} "
            f"({type(exc).__name__}). Start the API with "
            "`uvicorn api.main:app --port 8000`."
        )
    if not response.ok:
        if response.status_code == 404:
            return None, (
                "The running API does not expose `/predict/batch`. Restart it "
                "with `uvicorn api.main:app --port 8000` to load the batch "
                "endpoint."
            )
        detail = response.text[:300].strip() or f"HTTP {response.status_code}"
        return None, f"The scoring API returned an error: {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The scoring API returned a response that was not valid JSON."


@st.cache_data(show_spinner=False)
def batch_template(mode: str) -> pd.DataFrame:
    """Small example CSV for either supported upload schema."""
    if mode == "kaggle":
        reference = load_reference(mode=mode)
        if not reference.empty:
            sample = reference.head(5)
            amount = (
                np.expm1(sample["log_amount"].astype(float))
                if "log_amount" in sample.columns
                else pd.Series([100.0] * len(sample))
            )
            template = pd.DataFrame({"Amount": amount.round(2)})
            for i in range(1, 29):
                column = f"V{i}"
                if column in sample.columns:
                    template[column] = sample[column].astype(float).round(6)
            return template
        return pd.DataFrame(
            {"Amount": [100.0], **{f"V{i}": [0.0] for i in range(1, 29)}}
        )

    path = DATA / "transactions.csv"
    if path.exists():
        try:
            sample = pd.read_csv(path, nrows=5)
            columns = [
                "timestamp",
                "amount",
                "tx_type",
                "merchant_category",
                "country",
                "device_risk",
                "ip_risk",
                "account_age_days",
                "tx_count_24h",
                "is_fraud",
            ]
            return sample[[column for column in columns if column in sample.columns]]
        except (OSError, ValueError):
            pass
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "amount": 85.0,
                "tx_type": "purchase",
                "merchant_category": "grocery",
                "country": "IN",
                "device_risk": 0.10,
                "ip_risk": 0.05,
                "account_age_days": 900,
                "tx_count_24h": 1,
                "is_fraud": 0,
            },
            {
                "timestamp": "2026-01-01T02:15:00",
                "amount": 480.0,
                "tx_type": "online",
                "merchant_category": "electronics",
                "country": "IN",
                "device_risk": 0.55,
                "ip_risk": 0.35,
                "account_age_days": 12,
                "tx_count_24h": 9,
                "is_fraud": 1,
            },
        ]
    )


def score_band_counts(scores: pd.Series, decisions: pd.Series) -> pd.DataFrame:
    """Count approved and blocked rows in each risk band."""
    bands = pd.cut(
        pd.Series(scores).reset_index(drop=True),
        bins=BAND_EDGES,
        labels=BAND_LABELS,
        right=False,
        include_lowest=True,
    )
    frame = pd.DataFrame(
        {
            "band": bands.astype(str),
            "decision": pd.Series(decisions).reset_index(drop=True),
        }
    )
    counts = frame.groupby(["band", "decision"]).size().unstack(fill_value=0)
    counts = counts.reindex(BAND_LABELS, fill_value=0)
    output = pd.DataFrame({"band": BAND_LABELS})
    for decision, column in (
        (DECISION_APPROVE, "approved"),
        (DECISION_BLOCK, "blocked"),
    ):
        output[column] = (
            counts[decision].to_numpy() if decision in counts.columns else 0
        )
    return output
