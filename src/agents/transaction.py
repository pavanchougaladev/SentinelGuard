"""Transaction-fraud agent: model score, SHAP reason codes, and amount/velocity rules."""

from __future__ import annotations

from typing import Any

from src.agents.base import (
    AgentContext,
    AgentFinding,
    BaseAgent,
    Reason,
    clamp,
    noisy_or,
    safe_float,
    safe_int,
)


class TransactionAgent(BaseAgent):
    name = "transaction_agent"
    display_name = "Transaction fraud agent"
    fraud_type = "transaction"
    description = (
        "Scores the payment itself using the supervised model, its SHAP reason "
        "codes, and amount/velocity/geography rules."
    )

    def evaluate(self, context: AgentContext) -> AgentFinding:
        tx = context.transaction or {}
        amount = safe_float(tx.get("amount", tx.get("Amount", 0.0)))
        device = safe_float(tx.get("device_risk"))
        ip = safe_float(tx.get("ip_risk"))
        velocity = safe_int(tx.get("tx_count_24h"))
        account_age = safe_int(tx.get("account_age_days"), 365)
        country = str(tx.get("country", "IN") or "IN").upper()
        tx_type = str(tx.get("tx_type", "purchase") or "purchase").lower()
        merchant = str(tx.get("merchant_category", "other") or "other").lower()

        rules: list[tuple[bool, str, str, float, Any]] = [
            (amount >= 10_000, "HIGH_AMOUNT", "Very high transaction amount", 0.18, amount),
            (5_000 <= amount < 10_000, "ELEVATED_AMOUNT", "High transaction amount", 0.10, amount),
            (device >= 0.8, "HIGH_DEVICE_RISK", "Device risk is very high", 0.16, device),
            (0.6 <= device < 0.8, "ELEVATED_DEVICE_RISK", "Device risk is elevated", 0.08, device),
            (ip >= 0.8, "HIGH_IP_RISK", "IP risk is very high", 0.16, ip),
            (0.6 <= ip < 0.8, "ELEVATED_IP_RISK", "IP risk is elevated", 0.08, ip),
            (velocity >= 20, "VELOCITY_SPIKE", "Unusually high 24-hour velocity", 0.18, velocity),
            (10 <= velocity < 20, "ELEVATED_VELOCITY", "Elevated 24-hour velocity", 0.10, velocity),
            (country != "IN", "CROSS_BORDER", "Transaction is outside the home market", 0.08, country),
            (tx_type in {"transfer", "online"}, "RISKY_CHANNEL", "Transfer or card-not-present channel", 0.05, tx_type),
            (merchant in {"gambling", "crypto"}, "HIGH_RISK_MERCHANT", "High-risk merchant category", 0.10, merchant),
            (account_age <= 7, "NEW_ACCOUNT", "Account is less than a week old", 0.08, account_age),
            (device * ip >= 0.5, "COMPOUNDED_DEVICE_IP_RISK", "Device and IP risk are both elevated", 0.10, round(device * ip, 3)),
        ]
        rule_reasons = [
            Reason(code, label, weight, evidence)
            for triggered, code, label, weight, evidence in rules
            if triggered
        ]
        rule_risk = noisy_or([reason.weight for reason in rule_reasons])

        model_score = None
        if context.model_score is not None:
            model_score = clamp(context.model_score)
        online_score = None
        if context.online_score is not None:
            online_score = clamp(context.online_score)

        if model_score is not None:
            # Rules can corroborate the model but cannot alone push a very low
            # model score to a block decision.
            risk = clamp(1.0 - (1.0 - model_score) * (1.0 - 0.55 * rule_risk))
            confidence = 0.90 if rule_risk > 0 else 0.82
        elif online_score is not None:
            risk = clamp(1.0 - (1.0 - online_score) * (1.0 - 0.55 * rule_risk))
            confidence = 0.72
        else:
            risk = rule_risk
            confidence = 0.62 if rule_risk > 0 else 0.25

        model_reasons = [
            Reason(
                code=str(item.get("code", "MODEL_RISK_SCORE")),
                label=str(item.get("label", "Model risk driver")),
                weight=0.12 if item.get("severity") == "high" else 0.06,
                evidence="SHAP-derived reason code",
            )
            for item in (context.model_reasons or [])[:4]
        ]

        actions = []
        if risk >= 0.75:
            actions.append("Block the payment and contact the customer through an independent channel")
        elif risk >= 0.55:
            actions.append("Send the payment to analyst review before authorisation")
        if velocity >= 20:
            actions.append("Inspect linked transactions in the last 24 hours")
        if country != "IN":
            actions.append("Verify travel or cross-border intent")
        if not actions:
            actions.append("Continue normal monitoring")

        evidence = {
            "model_score": round(model_score, 4) if model_score is not None else None,
            "online_score": round(online_score, 4) if online_score is not None else None,
            "rule_risk": round(rule_risk, 4),
            "amount": amount,
            "device_risk": device,
            "ip_risk": ip,
            "tx_count_24h": velocity,
            "account_age_days": account_age,
            "country": country,
            "tx_type": tx_type,
            "merchant_category": merchant,
            "model_reason_codes": [item.code for item in model_reasons],
        }
        return self.finding(
            risk=risk,
            confidence=confidence,
            reasons=model_reasons + rule_reasons,
            evidence=evidence,
            actions=actions,
        )
