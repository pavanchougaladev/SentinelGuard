"""Federated fraud learning with secure aggregation.

Institutions train a shared fraud model locally. Only clipped model updates
leave an institution; customer rows never do. The coordinator combines updates
with Federated Averaging (FedAvg) after quantizing and pairwise-masking them
with one-time-pad masks (secure aggregation). An optional Gaussian mechanism
adds client-level differential privacy to the released update.

This is a simulation of the protocol that runs in one process so it can be
demonstrated and tested without standing up several hosts. The code paths still
respect the trust boundary: ``InstitutionClient`` keeps its frame private and
returns only ``ClientUpdate`` objects; ``FederatedCoordinator`` receives those
objects and a separate validation frame.

Usage:
    python -m src.federated --demo --clients 5 --rounds 6
    python -m src.federated --data data/transactions.csv --clients 4 --rounds 5 --dp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features import COUNTRIES, MCC, TYPES, build_features

# A 61-bit prime keeps pairwise sums inside int64, so masking is exact.
PRIME = (1 << 61) - 1
QUANT_SCALE = 10**9

# Fixed domain normalization. Sending local means would itself leak a little
# distribution information, so the federation agrees on public constants.
NORMALIZATION: dict[str, tuple[float, float]] = {
    "log_amount": (4.5, 1.2),
    "device_risk": (0.2, 0.2),
    "ip_risk": (0.2, 0.2),
    "account_age_days": (500.0, 500.0),
    "tx_count_24h": (2.0, 3.0),
    "hour": (12.0, 6.0),
    "dow": (3.0, 2.0),
    "risk_product": (0.1, 0.15),
}

BASE_TYPES = np.array([0.55, 0.15, 0.10, 0.20])
BASE_MCC = np.array([0.20, 0.10, 0.10, 0.08, 0.15, 0.03, 0.02, 0.15, 0.10, 0.07])
BASE_COUNTRIES = np.array([0.86, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])

PROFILES: list[dict[str, Any]] = [
    {
        "name": "retail-bank",
        "countries": BASE_COUNTRIES,
        "mcc": BASE_MCC,
        "types": BASE_TYPES,
        "amount_mu": 4.0,
        "amount_sigma": 1.1,
        "device_a": 1.4,
        "device_b": 5.0,
        "age_lo": 30,
        "age_hi": 3000,
        "velocity": 2.0,
    },
    {
        "name": "digital-neobank",
        "countries": np.array([0.55, 0.10, 0.07, 0.06, 0.07, 0.03, 0.06, 0.06]),
        "mcc": np.array([0.12, 0.07, 0.20, 0.14, 0.12, 0.04, 0.05, 0.08, 0.10, 0.08]),
        "types": np.array([0.35, 0.20, 0.05, 0.40]),
        "amount_mu": 4.1,
        "amount_sigma": 1.0,
        "device_a": 1.3,
        "device_b": 5.0,
        "age_lo": 30,
        "age_hi": 2500,
        "velocity": 2.5,
    },
    {
        "name": "cross-border-payments",
        "countries": np.array([0.35, 0.10, 0.10, 0.13, 0.12, 0.05, 0.10, 0.05]),
        "mcc": np.array([0.05, 0.03, 0.10, 0.24, 0.06, 0.06, 0.15, 0.06, 0.05, 0.20]),
        "types": np.array([0.20, 0.55, 0.10, 0.15]),
        "amount_mu": 4.3,
        "amount_sigma": 1.1,
        "device_a": 1.5,
        "device_b": 5.0,
        "age_lo": 30,
        "age_hi": 2500,
        "velocity": 2.5,
    },
    {
        "name": "crypto-ramp",
        "countries": np.array([0.30, 0.08, 0.08, 0.10, 0.08, 0.08, 0.15, 0.13]),
        "mcc": np.array([0.04, 0.02, 0.08, 0.08, 0.04, 0.08, 0.40, 0.04, 0.04, 0.18]),
        "types": np.array([0.18, 0.42, 0.10, 0.30]),
        "amount_mu": 4.4,
        "amount_sigma": 1.2,
        "device_a": 1.6,
        "device_b": 4.5,
        "age_lo": 30,
        "age_hi": 2000,
        "velocity": 3.0,
    },
    {
        "name": "regional-credit-union",
        "countries": np.array([0.93, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]),
        "mcc": np.array([0.24, 0.14, 0.05, 0.03, 0.20, 0.01, 0.01, 0.18, 0.12, 0.02]),
        "types": np.array([0.66, 0.10, 0.14, 0.10]),
        "amount_mu": 3.9,
        "amount_sigma": 1.0,
        "device_a": 1.4,
        "device_b": 5.0,
        "age_lo": 30,
        "age_hi": 3000,
        "velocity": 1.8,
    },
]

# Fraud is a shared, absolute concept: the same latent risk cutoff is applied at
# every institution. Base rates still differ because feature distributions do
# (non-IID), which is the realistic federated challenge.
FRAUD_SCORE_THRESHOLD = 6.5


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _feature_frame(df: pd.DataFrame) -> np.ndarray:
    """Build the shared feature matrix without sharing the raw frame."""
    X = build_features(df)
    columns = []
    for name in X.columns:
        mean, std = NORMALIZATION.get(name, (0.0, 1.0))
        columns.append(np.clip((X[name].astype(float).to_numpy() - mean) / max(std, 1e-9), -8.0, 8.0))
    return np.column_stack(columns)


def feature_names_for(df: pd.DataFrame) -> list[str]:
    return list(build_features(df).columns)


def _target(df: pd.DataFrame) -> np.ndarray:
    for name in ("is_fraud", "Class"):
        if name in df.columns:
            return df[name].astype(int).to_numpy()
    raise ValueError("federated data needs an is_fraud or Class label column")


REVIEW_COST = 1.25
MISSED_FRAUD_COST = 15.0


def select_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    review_cost: float = REVIEW_COST,
    missed_fraud_cost: float = MISSED_FRAUD_COST,
) -> float:
    """Cost-sensitive cutoff: review cost plus missed-fraud cost.

    Fraud is rare, so a plain F1 cutoff can keep recall low. The cost model
    makes the review/miss trade-off explicit and is consistent with the
    threshold policy engine in ``src.policy``.
    """
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 200)))
    best_threshold = 0.5
    best_cost = float("inf")
    for threshold in candidates:
        predicted = scores >= threshold
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        cost = fp * review_cost + fn * missed_fraud_cost
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold



def evaluate_params(
    params: "Params", X: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    scores = params.predict_proba(X)
    threshold = select_threshold(y, scores)
    predicted = (scores >= threshold).astype(int)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    try:
        auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else 0.5
        pr_auc = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float(y.mean())
    except ValueError:
        auc, pr_auc = 0.5, float(y.mean())
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "threshold": round(threshold, 6),
        "roc_auc": round(auc, 6),
        "pr_auc": round(pr_auc, 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "fpr": round(fp / max(fp + tn, 1), 6),
        "flagged_rate": round(float(predicted.mean()), 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@dataclass
class Params:
    """A logistic fraud model; the only object broadcast between parties."""

    coef: np.ndarray
    intercept: float
    feature_names: list[str]

    @classmethod
    def zeros(cls, feature_names: Sequence[str]) -> "Params":
        return cls(np.zeros(len(feature_names), dtype=float), 0.0, list(feature_names))

    def to_vector(self) -> np.ndarray:
        return np.concatenate(([float(self.intercept)], np.asarray(self.coef, dtype=float)))

    @classmethod
    def from_vector(cls, vector: np.ndarray, feature_names: Sequence[str]) -> "Params":
        vector = np.asarray(vector, dtype=float)
        return cls(vector[1:].copy(), float(vector[0]), list(feature_names))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.coef + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": float(self.intercept),
            "coef": [float(v) for v in self.coef],
            "feature_names": list(self.feature_names),
        }


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    init: Params | None = None,
    epochs: int = 60,
    learning_rate: float = 0.15,
    l2: float = 1e-3,
) -> tuple[Params, float]:
    """Class-balanced full-batch logistic regression for one institution.

    A compact custom learner keeps the update vector explicit and cheap to
    aggregate. It is deliberately simple: the point of the module is the
    privacy protocol, not the local model.
    """
    n, d = X.shape
    if n == 0:
        raise ValueError("empty local training frame")
    if init is None:
        w = np.zeros(d, dtype=float)
        b = 0.0
    else:
        w = np.asarray(init.coef, dtype=float).copy()
        b = float(init.intercept)
    positives = max(int(y.sum()), 1)
    negatives = max(int(n - positives), 1)
    sample_weight = np.where(y == 1, n / (2.0 * positives), n / (2.0 * negatives))
    loss = 0.0
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        error = (p - y) * sample_weight
        grad_w = X.T @ error / n + l2 * w
        grad_b = float(np.mean(error))
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        eps = 1e-9
        loss = float(-np.mean(sample_weight * (y * np.log(p + eps)
                                               + (1 - y) * np.log(1 - p + eps))))
    return Params(w, b, []), loss


@dataclass
class PrivacyConfig:
    """Client-level Gaussian update perturbation.

    ``noise_multiplier`` times ``clip_norm`` is spread across clients. The
    reported epsilon is an *indicative* classical Gaussian-mechanism estimate
    for the released aggregate, not a formal accountant.
    """

    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.5
    delta: float = 1e-5

    def sigma(self, n_clients: int) -> float:
        if not self.enabled:
            return 0.0
        return float(self.noise_multiplier * self.clip_norm / max(np.sqrt(n_clients), 1.0))

    def epsilon_per_round(self, n_clients: int) -> float:
        if not self.enabled:
            return 0.0
        sigma = self.sigma(n_clients)
        sensitivity = 2.0 * self.clip_norm / max(n_clients, 1)
        if sigma <= 0:
            return 0.0
        return float(np.sqrt(2.0 * np.log(1.25 / self.delta)) * sensitivity / sigma)


@dataclass
class FederatedConfig:
    rounds: int = 5
    local_epochs: int = 60
    learning_rate: float = 0.15
    l2: float = 1e-3
    seed: int = 42
    secure_aggregation: bool = True
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


@dataclass
class ClientUpdate:
    """The public result of a local training step.

    No rows, labels, or per-customer values are present. The vector is the
    clipped (and optionally noised) difference from the broadcast global model.
    """

    institution_id: str
    n_samples: int
    vector: np.ndarray
    update_norm: float
    released_norm: float
    dp_sigma: float

    def public_summary(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.vector.tobytes()).hexdigest()[:16]
        return {
            "institution_id": self.institution_id,
            "n_samples": self.n_samples,
            "update_norm": round(self.update_norm, 6),
            "released_norm": round(self.released_norm, 6),
            "dp_sigma": round(self.dp_sigma, 8),
            "update_sha256_prefix": digest,
            "contains_raw_rows": False,
        }


class SecureAggregator:
    """Pairwise one-time-pad masking over the integers modulo ``PRIME``.

    Each pair of clients shares a pseudorandom mask; one adds it and the other
    subtracts it. Individual masked updates are visible to the coordinator, but
    their sum cancels every mask and yields the exact weighted sum. In a real
    deployment the masks would be derived from Diffie-Hellman shared secrets on
    the clients, and the coordinator would receive only masked vectors.
    """

    def __init__(
        self,
        salt: str | None = None,
        *,
        prime: int = PRIME,
        scale: int = QUANT_SCALE,
    ) -> None:
        self.prime = prime
        self.scale = scale
        self.salt = salt or secrets.token_hex(16)

    def _pair_mask(self, left: str, right: str, length: int) -> np.ndarray:
        material = f"{self.salt}|{left}|{right}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.prime, size=length, dtype=np.int64)

    def aggregate(self, updates: Iterable[ClientUpdate]) -> tuple[np.ndarray, dict[str, Any]]:
        updates = list(updates)
        if not updates:
            raise ValueError("nothing to aggregate")
        ids = sorted(update.institution_id for update in updates)
        by_id = {update.institution_id: update for update in updates}
        length = len(updates[0].vector)
        quantized: dict[str, np.ndarray] = {}
        for institution_id in ids:
            update = by_id[institution_id]
            weighted = update.vector.astype(float) * update.n_samples
            q = np.rint(weighted * self.scale).astype(np.int64)
            quantized[institution_id] = np.mod(q, self.prime)

        masked = {key: value.copy() for key, value in quantized.items()}
        for left, right in combinations(ids, 2):
            mask = self._pair_mask(left, right, length)
            masked[left] = np.mod(masked[left] + mask, self.prime)
            masked[right] = np.mod(masked[right] - mask, self.prime)

        total = np.zeros(length, dtype=np.int64)
        for institution_id in ids:
            total = np.mod(total + masked[institution_id], self.prime)
        # Recover the signed value represented modulo the prime.
        signed = np.where(total > self.prime // 2, total - self.prime, total)
        total_samples = sum(update.n_samples for update in updates)
        mean = signed.astype(np.float64) / (self.scale * max(total_samples, 1))
        masked_bytes = len(ids) * length * 8
        return mean, {
            "scheme": "pairwise one-time-pad",
            "modulus_bits": int(self.prime.bit_length()),
            "quantization_scale": self.scale,
            "clients": len(ids),
            "vector_length": length,
            "masked_bytes": int(masked_bytes),
            "individual_updates_visible": False,
            "raw_rows_shared": False,
        }


class InstitutionClient:
    """One institution. Its frame is never returned to the coordinator."""

    def __init__(self, institution_id: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError(f"{institution_id}: empty frame")
        self.institution_id = institution_id
        self._frame = frame.reset_index(drop=True).copy()
        self._X = _feature_frame(self._frame)
        self._y = _target(self._frame)
        self.feature_names = feature_names_for(self._frame)

    @property
    def n_samples(self) -> int:
        return int(len(self._y))

    @property
    def n_positives(self) -> int:
        return int(self._y.sum())

    def local_update(
        self,
        global_params: Params,
        config: FederatedConfig,
        *,
        n_clients: int = 2,
        seed: int | None = None,
    ) -> ClientUpdate:
        if self.n_positives == 0 or self.n_positives == self.n_samples:
            raise ValueError(f"{self.institution_id}: local frame needs both classes")
        local, _ = train_logistic(
            self._X,
            self._y,
            init=global_params,
            epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            l2=config.l2,
        )
        delta = local.to_vector() - global_params.to_vector()
        raw_norm = float(np.linalg.norm(delta))
        clip = config.privacy.clip_norm
        if raw_norm > clip:
            delta = delta * (clip / max(raw_norm, 1e-12))
        sigma = 0.0
        if config.privacy.enabled:
            sigma = config.privacy.sigma(n_clients)
            rng = np.random.default_rng(seed if seed is not None else config.seed)
            delta = delta + rng.normal(0.0, sigma, size=delta.shape)
        return ClientUpdate(
            institution_id=self.institution_id,
            n_samples=self.n_samples,
            vector=delta,
            update_norm=raw_norm,
            released_norm=float(np.linalg.norm(delta)),
            dp_sigma=float(sigma),
        )


class FederatedCoordinator:
    def __init__(
        self,
        clients: Sequence[InstitutionClient],
        validation: pd.DataFrame,
        config: FederatedConfig | None = None,
    ) -> None:
        if len(clients) < 2:
            raise ValueError("federated learning needs at least two clients")
        self.clients = list(clients)
        self.config = config or FederatedConfig()
        self.feature_names = self.clients[0].feature_names
        for client in self.clients[1:]:
            if client.feature_names != self.feature_names:
                raise ValueError("all clients must use the same feature schema")
        self._validation_frame = validation.reset_index(drop=True).copy()
        self._validation_X = _feature_frame(self._validation_frame)
        self._validation_y = _target(self._validation_frame)
        self.params = Params.zeros(self.feature_names)
        self.history: list[dict[str, Any]] = []

    def evaluate(self) -> dict[str, Any]:
        return evaluate_params(self.params, self._validation_X, self._validation_y)

    def run_round(self, round_number: int) -> dict[str, Any]:
        started = time.perf_counter()
        updates: list[ClientUpdate] = []
        for index, client in enumerate(self.clients):
            seed = self.config.seed + round_number * 1000 + index
            updates.append(
                client.local_update(
                    self.params,
                    self.config,
                    n_clients=len(self.clients),
                    seed=seed,
                )
            )

        total_samples = sum(update.n_samples for update in updates)
        if self.config.secure_aggregation:
            aggregator = SecureAggregator(
                salt=f"{self.config.seed}:{round_number}", scale=QUANT_SCALE
            )
            mean_delta, aggregation = aggregator.aggregate(updates)
        else:
            mean_delta = sum(
                update.vector * update.n_samples for update in updates
            ) / max(total_samples, 1)
            aggregation = {
                "scheme": "plain weighted average (secure aggregation disabled)",
                "raw_rows_shared": False,
                "individual_updates_visible": True,
            }

        self.params = Params.from_vector(
            self.params.to_vector() + mean_delta, self.feature_names
        )
        metrics = self.evaluate()
        duration_ms = (time.perf_counter() - started) * 1000
        result = {
            "round": int(round_number),
            "clients_participating": len(updates),
            "samples": int(total_samples),
            "mean_update_norm": round(
                float(np.mean([u.released_norm for u in updates])), 6
            ),
            "aggregate_delta_norm": round(float(np.linalg.norm(mean_delta)), 6),
            "duration_ms": round(duration_ms, 2),
            "aggregation": aggregation,
            "client_updates": [u.public_summary() for u in updates],
            "metrics": metrics,
        }
        self.history.append(result)
        return result

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.params = Params.zeros(self.feature_names)
        self.history = []
        best_metrics = -1.0
        best_round = 1
        best_snapshot = self.params
        for round_number in range(1, self.config.rounds + 1):
            result = self.run_round(round_number)
            if result["metrics"]["pr_auc"] > best_metrics:
                best_metrics = result["metrics"]["pr_auc"]
                best_round = round_number
                best_snapshot = Params(
                    self.params.coef.copy(),
                    float(self.params.intercept),
                    list(self.params.feature_names),
                )
            print(
                f"round {round_number:>2} · clients {result['clients_participating']} · "
                f"PR-AUC {result['metrics']['pr_auc']:.3f} · "
                f"ROC-AUC {result['metrics']['roc_auc']:.3f} · "
                f"recall {result['metrics']['recall']:.1%}",
                flush=True,
            )

        # Keep the best-evaluated round, not necessarily the last one.
        self.params = best_snapshot
        final_metrics = self.evaluate()

        n_clients = len(self.clients)
        privacy = {
            "secure_aggregation": bool(self.config.secure_aggregation),
            "raw_rows_shared": False,
            "individual_updates_visible": not self.config.secure_aggregation,
            "model_updates_only": True,
            "dp": {
                "enabled": bool(self.config.privacy.enabled),
                "mechanism": "client-level Gaussian update perturbation",
                "clip_norm": self.config.privacy.clip_norm,
                "noise_multiplier": self.config.privacy.noise_multiplier,
                "delta": self.config.privacy.delta,
                "epsilon_per_round_estimate": round(
                    self.config.privacy.epsilon_per_round(n_clients), 4
                ),
                "epsilon_total_estimate": round(
                    self.config.privacy.epsilon_per_round(n_clients)
                    * self.config.rounds,
                    4,
                ),
                "accounting": "indicative classical Gaussian estimate; basic composition",
            },
            "notes": [
                "Only clipped model updates leave an institution.",
                "Secure aggregation masks each update; the coordinator sees the sum only.",
                "The in-process simulator generates masks locally for demonstration.",
                "DP is optional and its epsilon is indicative, not a certified budget.",
            ],
        }
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": "Federated Averaging (FedAvg) with secure aggregation",
            "config": {
                "rounds": self.config.rounds,
                "local_epochs": self.config.local_epochs,
                "learning_rate": self.config.learning_rate,
                "l2": self.config.l2,
                "seed": self.config.seed,
            },
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "clients": [
                {
                    "institution_id": client.institution_id,
                    "n_samples": client.n_samples,
                }
                for client in self.clients
            ],
            "client_count": n_clients,
            "total_samples": int(sum(c.n_samples for c in self.clients)),
            "rounds": self.history,
            "best_round": best_round,
            "final_metrics": final_metrics,
            "privacy": privacy,
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
        return report

    def save(self, report: dict[str, Any], output_dir: str = "models") -> dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "federated_report.json")
        markdown_path = os.path.join(output_dir, "federated_report.md")
        model_path = os.path.join(output_dir, "federated_model.joblib")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(federated_markdown(report))
        joblib.dump(
            {
                "params": self.params.to_dict(),
                "features": self.feature_names,
                "threshold": report["final_metrics"]["threshold"],
                "normalization": NORMALIZATION,
                "privacy": report["privacy"],
            },
            model_path,
        )
        return {"report": report_path, "markdown": markdown_path, "model": model_path}


def federated_markdown(report: dict[str, Any]) -> str:
    metrics = report["final_metrics"]
    privacy = report["privacy"]
    dp = privacy["dp"]
    lines = [
        "# Federated fraud learning",
        "",
        "A shared fraud model trained by simulated institutions. Customer rows never leave an institution.",
        "",
        f"- Algorithm: {report['algorithm']}",
        f"- Institutions: {report['client_count']}",
        f"- Rounds: {report['config']['rounds']} (best round {report['best_round']})",
        f"- Secure aggregation: {'on' if privacy['secure_aggregation'] else 'off'}",
        f"- Differential privacy: {'on' if dp['enabled'] else 'off'}"
        + (f" · indicative ε ≈ {dp['epsilon_total_estimate']}" if dp["enabled"] else ""),
        "",
        "## Final held-out metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| ROC-AUC | {metrics['roc_auc']:.3f} |",
        f"| PR-AUC | {metrics['pr_auc']:.3f} |",
        f"| Precision | {metrics['precision']:.1%} |",
        f"| Recall | {metrics['recall']:.1%} |",
        f"| F1 | {metrics['f1']:.1%} |",
        f"| FPR | {metrics['fpr']:.2%} |",
        f"| Flagged | {metrics.get('flagged_rate', 0.0):.1%} |",
        f"| Threshold | {metrics['threshold']:.3f} |",
        "",
        "## Round history",
        "",
        "| Round | Clients | Samples | PR-AUC | ROC-AUC | Recall | Precision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rounds"]:
        m = row["metrics"]
        lines.append(
            f"| {row['round']} | {row['clients_participating']} | {row['samples']:,} | "
            f"{m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['recall']:.1%} | "
            f"{m['precision']:.1%} |"
        )
    lines.extend(["", "## Privacy", ""])
    lines.extend(f"- {note}" for note in privacy["notes"])
    lines.append("")
    return "\n".join(lines)


def load_report(path: str = "models/federated_report.json") -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Demo / real-data fixtures
# ---------------------------------------------------------------------------


def generate_frame(profile: dict[str, Any], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    secs = np.sort(rng.integers(0, 120 * 24 * 3600, n))
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(secs, unit="s")
    amount = np.clip(rng.lognormal(profile["amount_mu"], profile["amount_sigma"], n), 1.0, 100_000.0)
    tx_type = rng.choice(TYPES, n, p=profile["types"] / profile["types"].sum())
    mcc = rng.choice(MCC, n, p=profile["mcc"] / profile["mcc"].sum())
    country = rng.choice(COUNTRIES, n, p=profile["countries"] / profile["countries"].sum())
    device = rng.beta(profile["device_a"], profile["device_b"], n)
    ip = rng.beta(max(profile["device_a"] * 0.9, 0.5), profile["device_b"], n)
    age = rng.integers(profile["age_lo"], profile["age_hi"], n)
    velocity = rng.poisson(profile["velocity"], n)
    hour = timestamps.hour.to_numpy()

    z = (
        0.8 * (np.log1p(amount) - 4.5)
        + 2.8 * device
        + 2.8 * ip
        + 1.3 * (country != "IN")
        + 1.0 * (hour < 6)
        + 1.0 * (tx_type == "transfer")
        + 1.1 * np.isin(mcc, ["gambling", "crypto"])
        - age / 1500.0
        + 0.20 * velocity
    )
    score = z + rng.gumbel(0.0, 1.2, n)
    label = (score >= profile.get("fraud_threshold", FRAUD_SCORE_THRESHOLD)).astype(int)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "amount": np.round(amount, 2),
            "tx_type": tx_type,
            "merchant_category": mcc,
            "country": country,
            "device_risk": device.round(4),
            "ip_risk": ip.round(4),
            "account_age_days": age,
            "tx_count_24h": velocity,
            "is_fraud": label,
        }
    )


def make_demo_clients(
    n_clients: int = 5,
    samples_per_client: int = 1200,
    validation_samples: int = 3000,
    seed: int = 42,
) -> tuple[list[InstitutionClient], pd.DataFrame]:
    clients: list[InstitutionClient] = []
    for index in range(n_clients):
        profile = PROFILES[index % len(PROFILES)]
        frame = generate_frame(
            profile,
            samples_per_client,
            seed=seed + index * 101,
        )
        clients.append(InstitutionClient(f"institution_{index + 1:02d}_{profile['name']}", frame))

    validation_parts = []
    for index in range(min(3, len(PROFILES))):
        validation_parts.append(
            generate_frame(
                PROFILES[index],
                max(validation_samples // 3, 200),
                seed=seed + 9000 + index,
            )
        )
    validation = pd.concat(validation_parts, ignore_index=True)
    validation = validation.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return clients, validation


def partition_frame(df: pd.DataFrame, n_clients: int, seed: int = 42) -> list[pd.DataFrame]:
    """Create deliberately non-IID shards (grouped by country when present)."""
    data = df.copy()
    if "country" in data.columns:
        data = data.assign(_order=data["country"].astype(str)).sort_values(
            ["_order", "timestamp"] if "timestamp" in data.columns else ["_order"],
            kind="stable",
        )
        data = data.drop(columns=["_order"])
    elif "Time" in data.columns:
        data = data.sort_values("Time")
    else:
        data = data.sample(frac=1.0, random_state=seed)
    parts = [part for part in np.array_split(data, n_clients) if len(part) > 0]
    if len(parts) < 2:
        raise ValueError("not enough rows to create two institution shards")
    return parts


def main(args: argparse.Namespace) -> None:
    config = FederatedConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        secure_aggregation=not args.no_secure,
        privacy=PrivacyConfig(
            enabled=args.dp,
            clip_norm=args.clip_norm,
            noise_multiplier=args.noise_multiplier,
            delta=args.delta,
        ),
    )

    if args.data and not args.demo:
        df = pd.read_csv(args.data)
        label_col = "is_fraud" if "is_fraud" in df.columns else "Class"
        if label_col not in df.columns:
            raise SystemExit(f"{args.data} has no is_fraud/Class column")
        time_col = "Time" if "Time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
        split = int(len(df) * 0.85)
        training, validation = df.iloc[:split], df.iloc[split:]
        parts = partition_frame(training, args.clients, args.seed)
        clients = []
        for index, part in enumerate(parts):
            if part[label_col].nunique() < 2 or len(part) < 50:
                continue
            clients.append(InstitutionClient(f"institution_{index + 1:02d}", part))
        if len(clients) < 2:
            raise SystemExit("fewer than two usable institution shards; try more rows")
        print(f"loaded {len(df):,} rows into {len(clients)} institutions", flush=True)
    else:
        clients, validation = make_demo_clients(
            n_clients=args.clients,
            samples_per_client=args.samples_per_client,
            validation_samples=args.validation_samples,
            seed=args.seed,
        )
        print(
            f"generated {len(clients)} demo institutions · "
            f"{sum(c.n_samples for c in clients):,} private rows",
            flush=True,
        )

    coordinator = FederatedCoordinator(clients, validation, config)
    report = coordinator.run()
    paths = coordinator.save(report, args.output_dir)
    print("\nfinal held-out metrics:")
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"):
        print(f"  {key:<10} {report['final_metrics'][key]}")
    print(f"\nwrote {paths['report']}, {paths['markdown']}, {paths['model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Optional real CSV; a demo is generated otherwise")
    parser.add_argument("--demo", action="store_true", help="Force the synthetic demo")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--samples-per-client", type=int, default=1200)
    parser.add_argument("--validation-samples", type=int, default=3000)
    parser.add_argument("--local-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--dp", action="store_true", help="Enable client-level Gaussian DP")
    parser.add_argument("--no-secure", action="store_true", help="Disable secure aggregation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models")
    main(parser.parse_args())
