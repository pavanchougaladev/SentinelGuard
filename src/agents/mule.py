"""Mule-account agent: fan-in/fan-out, pass-through, structuring, and dormancy."""

from __future__ import annotations

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


class MuleAgent(BaseAgent):
    name = "mule_agent"
    display_name = "Mule-account agent"
    fraud_type = "mule_account"
    description = (
        "Detects money-mule behaviour: many senders or receivers, fast "
        "pass-through of funds, round amounts, structuring below thresholds, "
        "dormant-then-active accounts, and shared devices."
    )

    def evaluate(self, context: AgentContext) -> AgentFinding:
        signals = context.mule or {}
        inbound = safe_int(signals.get("inbound_count_24h"), 0)
        outbound = safe_int(signals.get("outbound_count_24h"), 0)
        senders = safe_int(signals.get("unique_senders_7d"), 0)
        receivers = safe_int(signals.get("unique_receivers_7d"), 0)
        pass_through = clamp(safe_float(signals.get("pass_through_ratio"), 0.0))
        dormant = safe_int(signals.get("dormant_days_before_activity"), 0)
        round_ratio = clamp(safe_float(signals.get("round_amount_ratio"), 0.0))
        shared_devices = safe_int(signals.get("shared_device_accounts"), 1)
        beneficiary_risk = clamp(safe_float(signals.get("beneficiary_risk"), 0.0))
        avg_outbound = safe_float(signals.get("avg_outbound_amount"), 0.0)
        near_threshold_ratio = clamp(safe_float(signals.get("near_threshold_ratio"), 0.0))
        age_signal = signals.get("account_age_days")
        if age_signal is None:
            age_signal = context.transaction.get("account_age_days")
        account_age = safe_int(age_signal, 365)
        velocity_signal = signals.get("tx_count_24h")
        if velocity_signal is None:
            velocity_signal = context.transaction.get("tx_count_24h")
        velocity = safe_int(velocity_signal, 0)

        contributions: list[tuple[bool, str, str, float, object]] = [
            (pass_through >= 0.85, "FAST_PASS_THROUGH", "Most incoming funds leave the account quickly", 0.30, pass_through),
            (0.65 <= pass_through < 0.85, "PASS_THROUGH", "High share of funds is passed through", 0.16, pass_through),
            (senders >= 20, "FAN_IN", "Funds arrive from many unrelated senders", 0.30, senders),
            (10 <= senders < 20, "FAN_IN", "Funds arrive from several senders", 0.16, senders),
            (receivers >= 15, "FAN_OUT", "Funds are dispersed to many receivers", 0.26, receivers),
            (7 <= receivers < 15, "FAN_OUT", "Funds are dispersed to several receivers", 0.13, receivers),
            (inbound >= 10 and outbound >= 10, "HIGH_TWO_WAY_FLOW", "High inbound and outbound activity in 24 hours", 0.18, {"inbound": inbound, "outbound": outbound}),
            (account_age <= 30 and (inbound + outbound) >= 10, "NEW_ACCOUNT_HIGH_FLOW", "Young account already moving high volume", 0.18, account_age),
            (dormant >= 90 and (inbound + outbound) >= 5, "DORMANT_THEN_ACTIVE", "Dormant account suddenly became active", 0.16, dormant),
            (round_ratio >= 0.60, "ROUND_AMOUNTS", "Most transfers use round amounts", 0.15, round_ratio),
            (near_threshold_ratio >= 0.40 or 8_000 <= avg_outbound < 10_000, "STRUCTURING", "Amounts cluster just below a reporting threshold", 0.22, near_threshold_ratio or avg_outbound),
            (shared_devices >= 5, "SHARED_DEVICE_CLUSTER", "Account shares a device with several accounts", 0.16, shared_devices),
            (beneficiary_risk >= 0.60, "RISKY_BENEFICIARY", "Outbound beneficiary is high risk", 0.25 * beneficiary_risk, beneficiary_risk),
            (velocity >= 30, "EXTREME_VELOCITY", "Extreme transaction velocity", 0.16, velocity),
        ]
        reasons = [
            Reason(code, label, weight, evidence)
            for triggered, code, label, weight, evidence in contributions
            if triggered
        ]
        risk = noisy_or([reason.weight for reason in reasons])

        signal_count = 0
        for key, value in signals.items():
            if key in {"account_age_days", "tx_count_24h"}:
                continue
            if key == "shared_device_accounts":
                if safe_int(value, 1) > 1:
                    signal_count += 1
                continue
            if value not in (None, "", 0, 0.0, False):
                signal_count += 1
        confidence = 0.30 if signal_count == 0 else min(0.95, 0.45 + 0.06 * len(reasons))

        actions = []
        if risk >= 0.70:
            actions.append("Freeze outbound transfers and place the account in mule review")
            actions.append("Trace the beneficiary network and file a suspicious-activity report")
        elif risk >= 0.5:
            actions.append("Hold outbound transfers for analyst review")
        if senders >= 10 or receivers >= 7:
            actions.append("Map the counterparty network and check for shared devices")
        if near_threshold_ratio >= 0.4 or 8_000 <= avg_outbound < 10_000:
            actions.append("Review transaction splitting and threshold-reporting controls")
        if dormant >= 90:
            actions.append("Verify the reactivation with the customer")
        if not actions:
            actions.append("No mule escalation required; continue network monitoring")

        evidence = {
            "inbound_count_24h": inbound,
            "outbound_count_24h": outbound,
            "unique_senders_7d": senders,
            "unique_receivers_7d": receivers,
            "pass_through_ratio": round(pass_through, 4),
            "dormant_days_before_activity": dormant,
            "round_amount_ratio": round(round_ratio, 4),
            "shared_device_accounts": shared_devices,
            "beneficiary_risk": round(beneficiary_risk, 4),
            "avg_outbound_amount": avg_outbound,
            "near_threshold_ratio": round(near_threshold_ratio, 4),
            "account_age_days": account_age,
            "tx_count_24h": velocity,
        }
        return self.finding(
            risk=risk,
            confidence=confidence,
            reasons=reasons,
            evidence=evidence,
            actions=actions,
        )
