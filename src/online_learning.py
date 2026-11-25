"""Online fraud learning with River.

The deployed RandomForest is retrained in batches. This module complements it
with a River model that updates after every labelled decision, so new fraud
patterns can be absorbed without waiting for a retrain. It also adapts the
decision threshold from a bounded recent window and watches the error stream
with ADWIN for concept drift.

Usage:
    python -m src.online_learning --data data/transactions.csv \\
        --bootstrap-rows 20000 --stream-rows 5000

The model state is pickled atomically to ``models/online_model.pkl``. The API
loads it on startup and exposes ``/online/predict``, ``/online/learn``,
``/online/status``, and ``/online/bootstrap``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.features import build_features

try:
    import river
    from river import compose as river_compose
    from river import drift as river_drift
    from river import linear_model as river_linear
    from river import optim as river_optim
    from river import preprocessing as river_preprocessing

    RIVER_AVAILABLE = True
    RIVER_VERSION = getattr(river, "__version__", "unknown")
except Exception:  # pragma: no cover - only when the optional dependency is absent
    river = None
    river_compose = river_drift = river_linear = river_optim = river_preprocessing = None
    RIVER_AVAILABLE = False
    RIVER_VERSION = None

DEFAULT_MODEL_PATH = os.getenv("ONLINE_MODEL_PATH", "models/online_model.pkl")
DEFAULT_EVENT_LOG = os.getenv("ONLINE_EVENT_LOG", "models/online_events.jsonl")
WARMUP_SAMPLES = 50
DEFAULT_FRAUD_WEIGHT = 8.0

DEFAULTS = {
    "tx_type": "purchase",
    "merchant_category": "other",
    "country": "IN",
    "device_risk": 0.2,
    "ip_risk": 0.2,
    "account_age_days": 365,
    "tx_count_24h": 1,
}


def featurize_transaction(transaction: dict[str, Any]) -> dict[str, float]:
    """Turn one raw transaction into the shared feature dictionary."""
    row = dict(transaction or {})
    if row.get("amount") is None:
        row["amount"] = row.get("Amount", 0.0)
    for key, value in DEFAULTS.items():
        if row.get(key) is None:
            row[key] = value
    if not row.get("timestamp"):
        row["timestamp"] = datetime.now(timezone.utc)
    frame = pd.DataFrame([row])
    matrix = build_features(frame)
    return {str(key): float(value) for key, value in matrix.iloc[0].items()}


def optimal_threshold(
    y: Iterable[int],
    scores: Iterable[float],
    beta: float = 1.0,
) -> float | None:
    """Best F-beta cutoff on a labelled window, or ``None`` if too little data."""
    y_arr = np.asarray(list(y), dtype=int)
    p_arr = np.asarray(list(scores), dtype=float)
    if len(y_arr) < 20 or len(np.unique(y_arr)) < 2:
        return None
    positives = int(y_arr.sum())
    if positives < 3 or (len(y_arr) - positives) < 10:
        return None
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_arr, p_arr)
    if len(thresholds) == 0:
        return None
    fbeta = ((1 + beta**2) * precision * recall
             / np.clip(beta**2 * precision + recall, 1e-12, None))
    return float(thresholds[int(np.nanargmax(fbeta[:-1]))])


class AdaptiveThreshold:
    """Smooth, bounded threshold updates from a labelled window."""

    def __init__(self, initial: float = 0.5, smoothing: float = 0.15) -> None:
        self.value = float(initial)
        self.smoothing = float(smoothing)
        self.history: list[float] = [self.value]

    def update(self, y: Iterable[int], scores: Iterable[float]) -> float:
        candidate = optimal_threshold(y, scores)
        if candidate is None:
            return self.value
        self.value = float(
            np.clip((1 - self.smoothing) * self.value + self.smoothing * candidate, 0.05, 0.95)
        )
        self.history.append(round(self.value, 6))
        return self.value


def _build_model():
    if not RIVER_AVAILABLE:
        return None
    return river_compose.Pipeline(
        river_preprocessing.StandardScaler(),
        river_linear.LogisticRegression(
            optimizer=river_optim.Adam(0.01),
            l2=1e-5,
            intercept_lr=0.01,
        ),
    )


class OnlineFraudLearner:
    """River-backed online learner with adaptive threshold and ADWIN drift."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        event_log_path: str = DEFAULT_EVENT_LOG,
        *,
        retune_every: int = 50,
        buffer_size: int = 1000,
        save_every: int = 100,
    ) -> None:
        self.model_path = model_path
        self.event_log_path = event_log_path
        self.retune_every = max(int(retune_every), 1)
        self.save_every = max(int(save_every), 0)
        self.buffer: deque[tuple[int, float]] = deque(maxlen=max(int(buffer_size), 20))
        self._lock = threading.RLock()
        self.model = _build_model()
        self.threshold = AdaptiveThreshold(0.5)
        self.samples_seen = 0
        self.positives_seen = 0
        self.adwin = river_drift.ADWIN() if RIVER_AVAILABLE else None
        self.drift_count = 0
        self.last_drift_at: str | None = None
        self.drifted = False
        self.last_event: dict[str, Any] | None = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at: str | None = None
        self.load()

    # -- persistence ------------------------------------------------------
    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        try:
            with open(self.model_path, "rb") as handle:
                state = pickle.load(handle)
            self.model = state.get("model", self.model)
            self.threshold = AdaptiveThreshold(
                float(state.get("threshold", 0.5)),
                float(state.get("smoothing", 0.15)),
            )
            self.threshold.history = list(state.get("threshold_history", [self.threshold.value]))
            self.samples_seen = int(state.get("samples_seen", 0))
            self.positives_seen = int(state.get("positives_seen", 0))
            self.buffer = deque(
                [tuple(item) for item in state.get("buffer", [])], maxlen=self.buffer.maxlen
            )
            self.drift_count = int(state.get("drift_count", 0))
            self.last_drift_at = state.get("last_drift_at")
            self.created_at = state.get("created_at", self.created_at)
            self.updated_at = state.get("updated_at")
            return True
        except Exception:
            return False

    def save(self) -> None:
        if not RIVER_AVAILABLE or self.model is None:
            return
        directory = os.path.dirname(self.model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        state = {
            "version": 1,
            "model": self.model,
            "threshold": self.threshold.value,
            "smoothing": self.threshold.smoothing,
            "threshold_history": self.threshold.history[-200:],
            "samples_seen": self.samples_seen,
            "positives_seen": self.positives_seen,
            "buffer": list(self.buffer),
            "drift_count": self.drift_count,
            "last_drift_at": self.last_drift_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "river_version": RIVER_VERSION,
        }
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=directory or ".", delete=False, suffix=".tmp"
            ) as handle:
                pickle.dump(state, handle)
                temporary = handle.name
            os.replace(temporary, self.model_path)
        except OSError:
            pass

    # -- inference / learning --------------------------------------------
    @property
    def available(self) -> bool:
        return RIVER_AVAILABLE

    @property
    def warm(self) -> bool:
        return self.samples_seen >= WARMUP_SAMPLES

    def _predict_x(self, features: dict[str, float]) -> float:
        if not RIVER_AVAILABLE or self.model is None:
            return 0.5
        try:
            probabilities = self.model.predict_proba_one(features)
        except Exception:
            return 0.5
        if not probabilities:
            return 0.5
        return float(np.clip(probabilities.get(True, probabilities.get(1, 0.5)), 0.0, 1.0))

    def predict(self, transaction: dict[str, Any], *, threshold: float | None = None) -> dict[str, Any]:
        features = featurize_transaction(transaction)
        with self._lock:
            score = self._predict_x(features)
            operating = float(threshold if threshold is not None else self.threshold.value)
        return {
            "available": RIVER_AVAILABLE,
            "model": "River LogisticRegression + StandardScaler" if RIVER_AVAILABLE else None,
            "online_score": round(score, 6),
            "threshold": round(operating, 6),
            "decision": "BLOCK/REVIEW" if score >= operating else "APPROVE",
            "samples_seen": self.samples_seen,
            "warm": self.warm,
            "drift": {
                "detected": self.drifted,
                "events": self.drift_count,
                "last_detected_at": self.last_drift_at,
            },
        }

    def _append_event(self, event: dict[str, Any]) -> None:
        try:
            directory = os.path.dirname(self.event_log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.event_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
        except OSError:
            pass

    def _learn_x(
        self,
        features: dict[str, float],
        label: int,
        *,
        weight: float | None = None,
        record: bool = True,
        persist: bool = True,
    ) -> dict[str, Any]:
        label = int(label)
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if not RIVER_AVAILABLE or self.model is None:
            raise RuntimeError("River is not installed; online learning is unavailable")
        score_before = self._predict_x(features)
        sample_weight = (
            float(weight)
            if weight is not None and weight > 0
            else (DEFAULT_FRAUD_WEIGHT if label == 1 else 1.0)
        )
        self.model.learn_one(features, bool(label), w=sample_weight)
        self.samples_seen += 1
        if label == 1:
            self.positives_seen += 1
        self.buffer.append((label, score_before))
        self.updated_at = datetime.now(timezone.utc).isoformat()

        error = int((score_before >= self.threshold.value) != bool(label))
        if self.adwin is not None:
            self.adwin.update(error)
            if self.adwin.drift_detected:
                self.drift_count += 1
                self.drifted = True
                self.last_drift_at = self.updated_at
        if self.samples_seen % self.retune_every == 0:
            self.threshold.update(
                [item[0] for item in self.buffer],
                [item[1] for item in self.buffer],
            )
        score_after = self._predict_x(features)
        event = {
            "timestamp": self.updated_at,
            "samples_seen": self.samples_seen,
            "label": label,
            "score_before": round(score_before, 6),
            "score_after": round(score_after, 6),
            "threshold": round(self.threshold.value, 6),
            "decision_before": "BLOCK/REVIEW" if score_before >= self.threshold.value else "APPROVE",
            "drift_detected": bool(self.adwin.drift_detected) if self.adwin is not None else False,
        }
        self.last_event = event
        if record:
            self._append_event(event)
        if persist and self.save_every and self.samples_seen % self.save_every == 0:
            self.save()
        return event

    def learn(
        self,
        transaction: dict[str, Any],
        label: int,
        *,
        weight: float | None = None,
    ) -> dict[str, Any]:
        features = featurize_transaction(transaction)
        with self._lock:
            event = self._learn_x(features, label, weight=weight)
            status = self.status()
        return {"event": event, "status": status}

    def bootstrap(
        self,
        frame: pd.DataFrame,
        *,
        max_rows: int = 20_000,
        record: bool = False,
    ) -> dict[str, Any]:
        """Warm-start from historical labelled rows, without event-log noise."""
        if not RIVER_AVAILABLE:
            raise RuntimeError("River is not installed; online learning is unavailable")
        data = frame.head(max_rows).copy()
        if data.empty:
            raise ValueError("empty bootstrap frame")
        if any(
            str(col).startswith("V") and str(col)[1:].isdigit()
            for col in data.columns
        ):
            raise ValueError(
                "online learning supports the transaction schema, not Kaggle V1-V28 rows"
            )
        if "is_fraud" not in data.columns and "Class" not in data.columns:
            raise ValueError("bootstrap frame needs is_fraud or Class")
        if "amount" not in data.columns and "Amount" in data.columns:
            data["amount"] = data["Amount"]
        for key, value in DEFAULTS.items():
            if key not in data.columns:
                data[key] = value
        if "timestamp" not in data.columns:
            data["timestamp"] = datetime.now(timezone.utc)
        features_frame = build_features(data)
        labels = (
            data["is_fraud"].astype(int).to_numpy()
            if "is_fraud" in data.columns
            else data["Class"].astype(int).to_numpy()
        )
        with self._lock:
            for index in range(len(data)):
                features = {
                    str(key): float(value)
                    for key, value in features_frame.iloc[index].items()
                }
                self._learn_x(
                    features,
                    int(labels[index]),
                    record=record,
                    persist=False,
                )
            self.threshold.update(
                [item[0] for item in self.buffer],
                [item[1] for item in self.buffer],
            )
            self.save()
            status = self.status()
        return status

    def reset(self, *, remove_artifacts: bool = True) -> dict[str, Any]:
        with self._lock:
            self.model = _build_model()
            self.threshold = AdaptiveThreshold(0.5)
            self.samples_seen = 0
            self.positives_seen = 0
            self.buffer.clear()
            self.adwin = river_drift.ADWIN() if RIVER_AVAILABLE else None
            self.drift_count = 0
            self.last_drift_at = None
            self.drifted = False
            self.last_event = None
            self.created_at = datetime.now(timezone.utc).isoformat()
            self.updated_at = None
            if remove_artifacts:
                for path in (self.model_path, self.event_log_path):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except OSError:
                        pass
            self.save()
            return self.status()

    # -- reporting --------------------------------------------------------
    def _rolling_metrics(self) -> dict[str, Any]:
        if not self.buffer:
            return {
                "n": 0,
                "positives": 0,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "pr_auc": None,
                "log_loss": None,
                "mean_score": None,
            }
        y = np.asarray([item[0] for item in self.buffer], dtype=int)
        p = np.asarray([item[1] for item in self.buffer], dtype=float)
        predicted = (p >= self.threshold.value).astype(int)
        tp = int(np.sum((predicted == 1) & (y == 1)))
        fp = int(np.sum((predicted == 1) & (y == 0)))
        fn = int(np.sum((predicted == 0) & (y == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        eps = 1e-9
        log_loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        roc_auc = pr_auc = None
        if len(np.unique(y)) > 1:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score

                roc_auc = float(roc_auc_score(y, p))
                pr_auc = float(average_precision_score(y, p))
            except ValueError:
                pass
        return {
            "n": int(len(y)),
            "positives": int(y.sum()),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
            "log_loss": round(log_loss, 6),
            "mean_score": round(float(p.mean()), 6),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            rolling = self._rolling_metrics()
            return {
                "available": RIVER_AVAILABLE,
                "river_version": RIVER_VERSION,
                "model": (
                    "River LogisticRegression + StandardScaler"
                    if RIVER_AVAILABLE
                    else None
                ),
                "samples_seen": self.samples_seen,
                "positives_seen": self.positives_seen,
                "fraud_rate_seen": round(
                    self.positives_seen / max(self.samples_seen, 1), 6
                ),
                "threshold": round(self.threshold.value, 6),
                "threshold_history": self.threshold.history[-50:],
                "warm": self.warm,
                "rolling": rolling,
                "drift": {
                    "detected": self.drifted,
                    "events": self.drift_count,
                    "last_detected_at": self.last_drift_at,
                },
                "last_event": self.last_event,
                "model_path": self.model_path,
                "event_log_path": self.event_log_path,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }


def load_online_history(path: str = DEFAULT_EVENT_LOG, limit: int = 500) -> list[dict[str, Any]]:
    """Read the tail of the online event log for the dashboard."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()[-max(limit, 1):]
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows
    except OSError:
        return []


def main(args: argparse.Namespace) -> None:
    if not RIVER_AVAILABLE:
        raise SystemExit("River is not installed; run pip install -r requirements.txt")
    if args.data and os.path.exists(args.data):
        frame = pd.read_csv(args.data)
        time_col = "Time" if "Time" in frame.columns else (
            "timestamp" if "timestamp" in frame.columns else None
        )
        if time_col:
            frame = frame.sort_values(time_col).reset_index(drop=True)
    else:
        from src.make_data import make

        print("data file not found; generating a synthetic frame", flush=True)
        frame = make(args.fallback_rows)

    learner = OnlineFraudLearner(
        model_path=args.path,
        event_log_path=args.event_log,
        retune_every=args.retune_every,
        save_every=args.save_every,
    )
    if args.reset:
        learner.reset()
    bootstrap_end = min(args.bootstrap_rows, len(frame))
    learner.bootstrap(frame.iloc[:bootstrap_end], max_rows=args.bootstrap_rows)
    print(
        f"bootstrapped {bootstrap_end:,} rows · threshold {learner.threshold.value:.3f}",
        flush=True,
    )

    stream = frame.iloc[bootstrap_end : bootstrap_end + args.stream_rows]
    if stream.empty:
        print(json.dumps(learner.status(), indent=2))
        return
    label_column = "is_fraud" if "is_fraud" in stream.columns else "Class"
    started = time.perf_counter()
    for index, (_, row) in enumerate(stream.iterrows(), start=1):
        learner.learn(row.to_dict(), int(row[label_column]))
        if index % args.report_every == 0 or index == len(stream):
            rolling = learner.status()["rolling"]
            print(
                f"stream {index:,}/{len(stream):,} · threshold {learner.threshold.value:.3f} · "
                f"rolling recall {rolling['recall']} · roll F1 {rolling['f1']}",
                flush=True,
            )
    learner.save()
    print(
        f"processed {len(stream):,} online rows in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(json.dumps(learner.status(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--bootstrap-rows", type=int, default=20_000)
    parser.add_argument("--stream-rows", type=int, default=5_000)
    parser.add_argument("--fallback-rows", type=int, default=50_000)
    parser.add_argument("--retune-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--event-log", default=DEFAULT_EVENT_LOG)
    parser.add_argument("--reset", action="store_true")
    main(parser.parse_args())
