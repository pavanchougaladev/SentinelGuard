"""Analyst-friendly reason codes.

SHAP values explain the model; analysts need a short list of reviewable
reasons. This module maps the most influential SHAP contributions onto stable
code/label pairs so the same transaction always produces the same wording in
the API response, the dashboard and exported reports.

Reason codes are heuristic summaries, not causal explanations.
"""

from __future__ import annotations

from typing import Any, Iterable

# Feature-name prefixes (one-hot columns) mapped to a code and a label.
CATEGORICAL_LABELS: dict[str, tuple[str, str]] = {
    "mcc_gambling": ("HIGH_RISK_MERCHANT", "High-risk merchant category"),
    "mcc_crypto": ("HIGH_RISK_MERCHANT", "High-risk merchant category"),
    "mcc_travel": ("TRAVEL_SPEND", "Travel-related merchant category"),
    "tx_type_transfer": ("RISKY_CHANNEL", "Transfer channel"),
    "tx_type_withdrawal": ("RISKY_CHANNEL", "Cash withdrawal channel"),
    "tx_type_online": ("CARD_NOT_PRESENT", "Card-not-present channel"),
    "is_foreign": ("CROSS_BORDER", "Transaction outside the home market"),
    "is_night": ("UNUSUAL_HOUR", "Transaction outside normal hours"),
    "V1": ("PCA_DEVIATION", "Anomalous PCA component V1"),
}

FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "log_amount": ("HIGH_AMOUNT", "Transaction amount is high for this profile"),
    "amount": ("HIGH_AMOUNT", "Transaction amount is high for this profile"),
    "device_risk": ("HIGH_DEVICE_RISK", "High device risk score"),
    "ip_risk": ("HIGH_IP_RISK", "High IP risk score"),
    "risk_product": ("COMPOUNDED_DEVICE_IP_RISK", "Device and IP risk both elevated"),
    "anomaly": ("ANOMALY_SIGNAL", "Transaction is an outlier against the training profile"),
    "account_age_days": ("NEW_ACCOUNT", "Account is young for this spend"),
    "tx_count_24h": ("VELOCITY_SPIKE", "Unusually high transaction velocity"),
    "hour": ("UNUSUAL_HOUR", "Transaction outside normal hours"),
    "dow": ("UNUSUAL_DAY", "Unusual day of week for this account"),
    "is_night": ("UNUSUAL_HOUR", "Transaction outside normal hours"),
    "is_foreign": ("CROSS_BORDER", "Transaction outside the home market"),
}

COUNTRY_PREFIX = "country_"


def code_for(feature: str) -> tuple[str, str]:
    """Map one model feature name to ``(code, label)``."""
    name = str(feature)
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name in CATEGORICAL_LABELS:
        return CATEGORICAL_LABELS[name]
    if name.startswith(COUNTRY_PREFIX):
        return "HIGH_RISK_GEOGRAPHY", f"Geography {name[len(COUNTRY_PREFIX):]} is elevated for this profile"
    if name.startswith("type_"):
        return "RISKY_CHANNEL", f"{name[len('type_'):].replace('_', ' ').title()} channel"
    if name.startswith("mcc_"):
        return "MERCHANT_CATEGORY", f"{name[len('mcc_'):].replace('_', ' ').title()} merchant category"
    if name.startswith("V") and name[1:].isdigit():
        return "PCA_DEVIATION", f"Anomalous PCA component {name}"
    return name.upper(), name.replace("_", " ").capitalize()


def _severity(rank: int, total: int) -> str:
    if rank == 0:
        return "high"
    if rank == 1:
        return "medium"
    return "low"


def reason_codes(
    reasons: Iterable[dict[str, Any]],
    mode: str = "synthetic",
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Turn SHAP reason dicts into ranked, deduplicated analyst codes.

    ``reasons`` is the list returned by the scoring API: each item has a
    ``feature``, a ``shap`` value and a ``direction``. Only risk-raising
    drivers become reason codes; duplicate codes keep the largest contribution.
    """
    drivers = [
        item
        for item in reasons
        if float(item.get("shap", 0.0)) > 0
        and str(item.get("direction", "raises risk")) == "raises risk"
    ]
    drivers.sort(key=lambda item: abs(float(item.get("shap", 0.0))), reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in drivers:
        code, label = code_for(item.get("feature", ""))
        if code in seen:
            continue
        seen.add(code)
        selected.append(
            {
                "code": code,
                "label": label,
                "feature": item.get("feature"),
                "shap": round(float(item.get("shap", 0.0)), 4),
                "severity": _severity(len(selected), limit),
            }
        )
        if len(selected) >= limit:
            break

    if not selected:
        selected.append(
            {
                "code": "MODEL_RISK_SCORE",
                "label": "Model risk score raised by combined feature effects",
                "feature": None,
                "shap": 0.0,
                "severity": "low",
            }
        )
    return selected


def format_reason_codes(codes: Iterable[dict[str, Any]]) -> str:
    """One-line, semicolon-separated summary for CSV exports."""
    return "; ".join(f"{item['code']}: {item['label']}" for item in codes)
