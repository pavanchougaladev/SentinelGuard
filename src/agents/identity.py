"""Identity-fraud agent: KYC, synthetic-identity, reuse, and document signals."""

from __future__ import annotations

from src.agents.base import (
    AgentContext,
    AgentFinding,
    BaseAgent,
    Reason,
    clamp,
    noisy_or,
    safe_bool,
    safe_float,
    safe_int,
)


class IdentityAgent(BaseAgent):
    name = "identity_agent"
    display_name = "Identity fraud agent"
    fraud_type = "identity"
    description = (
        "Checks KYC state, account age, identity mismatches, reuse of devices/"
        "emails/phones, sanctions/PEP signals, synthetic-identity scores, and "
        "document or cheque tamper evidence."
    )

    def evaluate(self, context: AgentContext) -> AgentFinding:
        signals = context.identity or {}
        kyc = str(signals.get("kyc_status", "verified") or "verified").lower()
        age_signal = signals.get("account_age_days")
        if age_signal is None:
            age_signal = context.transaction.get("account_age_days")
        account_age = safe_int(age_signal, 365)
        mismatch = safe_bool(signals.get("identity_mismatch"))
        device_accounts = safe_int(signals.get("device_accounts_7d"), 1)
        email_accounts = safe_int(signals.get("email_accounts_7d"), 1)
        phone_accounts = safe_int(signals.get("phone_accounts_7d"), 1)
        geo_mismatch = safe_bool(signals.get("ip_country_mismatch"))
        sanctions = safe_bool(signals.get("sanctions_hit"))
        pep = safe_bool(signals.get("pep_hit"))
        synthetic = clamp(safe_float(signals.get("synthetic_identity_score")))
        document_tamper = clamp(safe_float(signals.get("document_tamper_score")))
        cheque_risk = clamp(safe_float((context.cheque or {}).get("risk_score")))
        cheque_tamper = safe_bool((context.cheque or {}).get("tamper_detected"))

        contributions: list[tuple[bool, str, str, float, object]] = [
            (kyc == "failed", "KYC_FAILED", "KYC verification failed", 0.50, kyc),
            (kyc in {"pending", "review", "unknown"}, "KYC_PENDING", "KYC is not complete", 0.15, kyc),
            (mismatch, "IDENTITY_MISMATCH", "Identity details do not match the account", 0.30, mismatch),
            (account_age <= 7, "NEW_ACCOUNT", "Account is less than a week old", 0.18, account_age),
            (7 < account_age <= 30, "YOUNG_ACCOUNT", "Account is less than a month old", 0.09, account_age),
            (device_accounts >= 10, "DEVICE_REUSE", "Device is linked to many accounts", 0.25, device_accounts),
            (5 <= device_accounts < 10, "DEVICE_REUSE", "Device is linked to several accounts", 0.13, device_accounts),
            (email_accounts >= 4, "EMAIL_REUSE", "Email is linked to several accounts", 0.10, email_accounts),
            (phone_accounts >= 4, "PHONE_REUSE", "Phone is linked to several accounts", 0.12, phone_accounts),
            (geo_mismatch, "GEO_IDENTITY_MISMATCH", "IP geography conflicts with the identity profile", 0.10, geo_mismatch),
            (sanctions, "SANCTIONS_HIT", "Applicant appears on a sanctions list", 0.70, sanctions),
            (pep, "PEP_HIT", "Applicant is a politically exposed person", 0.28, pep),
            (synthetic >= 0.65, "SYNTHETIC_IDENTITY", "Synthetic-identity score is high", 0.38 * synthetic, synthetic),
            (document_tamper >= 0.50, "DOCUMENT_TAMPER", "Identity document tamper score is high", 0.36 * document_tamper, document_tamper),
            (cheque_risk >= 0.55, "CHEQUE_RISK", "Linked cheque analysis is high risk", 0.30 * cheque_risk, cheque_risk),
            (cheque_tamper, "CHEQUE_TAMPER", "Linked cheque has tamper indicators", 0.20, cheque_tamper),
        ]
        reasons = [
            Reason(code, label, weight, evidence)
            for triggered, code, label, weight, evidence in contributions
            if triggered
        ]
        risk = noisy_or([reason.weight for reason in reasons])

        present = sum(
            1
            for key, value in signals.items()
            if not (key == "kyc_status" and str(value).lower() == "verified")
            and value not in (None, "", False, 0, 0.0)
        )
        if present == 0 and not context.cheque:
            confidence = 0.22
        else:
            confidence = min(0.95, 0.45 + 0.07 * len(reasons))

        actions = []
        if sanctions:
            actions.append("Freeze onboarding and escalate to financial-crime compliance immediately")
        if kyc == "failed" or mismatch:
            actions.append("Re-run identity verification and request original documents")
        if synthetic >= 0.65 or document_tamper >= 0.5:
            actions.append("Escalate to the identity-fraud team for document forensics")
        if device_accounts >= 5 or email_accounts >= 4 or phone_accounts >= 4:
            actions.append("Review the linked-account cluster for coordinated onboarding")
        if cheque_risk >= 0.55:
            actions.append("Hold the linked cheque and verify it with the issuing bank")
        if not actions:
            actions.append("No identity escalation required; keep periodic KYC review")

        evidence = {
            "kyc_status": kyc,
            "account_age_days": account_age,
            "identity_mismatch": mismatch,
            "device_accounts_7d": device_accounts,
            "email_accounts_7d": email_accounts,
            "phone_accounts_7d": phone_accounts,
            "ip_country_mismatch": geo_mismatch,
            "sanctions_hit": sanctions,
            "pep_hit": pep,
            "synthetic_identity_score": round(synthetic, 4),
            "document_tamper_score": round(document_tamper, 4),
            "cheque_risk_score": round(cheque_risk, 4),
        }
        return self.finding(
            risk=risk,
            confidence=confidence,
            reasons=reasons,
            evidence=evidence,
            actions=actions,
        )
