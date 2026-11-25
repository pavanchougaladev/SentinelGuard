"""Orchestrator for the specialized fraud agents.

The orchestrator runs the transaction, identity, and mule-account agents,
fuses their findings with explicit weights, adds a corroboration bonus when
independent agents agree, and produces a single decision with the evidence and
recommended actions of every contributing agent.
"""

from __future__ import annotations

from statistics import pstdev
from typing import Any, Sequence

from src.agents.base import AgentContext, AgentFinding, BaseAgent, clamp, risk_status
from src.agents.identity import IdentityAgent
from src.agents.mule import MuleAgent
from src.agents.transaction import TransactionAgent

DEFAULT_WEIGHTS = {
    "transaction_agent": 0.45,
    "identity_agent": 0.30,
    "mule_agent": 0.25,
}


def agent_catalog() -> list[dict[str, Any]]:
    agents = [TransactionAgent(), IdentityAgent(), MuleAgent()]
    return [
        {
            "name": agent.name,
            "display_name": agent.display_name,
            "fraud_type": agent.fraud_type,
            "description": agent.description,
            "default_weight": DEFAULT_WEIGHTS.get(agent.name, 1 / len(agents)),
        }
        for agent in agents
    ]


class FraudAgentOrchestrator:
    def __init__(
        self,
        agents: Sequence[BaseAgent] | None = None,
        weights: dict[str, float] | None = None,
        threshold: float = 0.55,
    ) -> None:
        self.agents = list(agents) if agents is not None else [
            TransactionAgent(),
            IdentityAgent(),
            MuleAgent(),
        ]
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.threshold = float(threshold)

    def _run_agent(self, agent: BaseAgent, context: AgentContext) -> AgentFinding:
        try:
            return agent.evaluate(context)
        except Exception as exc:  # an agent failure must not take down the decision
            return agent.finding(
                risk=0.0,
                confidence=0.0,
                reasons=[],
                evidence={},
                actions=[],
                error=f"{type(exc).__name__}: {exc}",
            )

    def analyze(
        self,
        context: AgentContext,
        *,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        threshold = float(self.threshold if threshold is None else threshold)
        findings = [self._run_agent(agent, context) for agent in self.agents]
        weights = {
            finding.agent: max(float(self.weights.get(finding.agent, 1.0)), 0.0)
            for finding in findings
        }
        weight_total = sum(weights.values()) or 1.0
        weighted_risk = sum(
            finding.risk * weights[finding.agent] for finding in findings
        ) / weight_total
        max_risk = max((finding.risk for finding in findings), default=0.0)

        # A single critical specialist should not be diluted by two quiet
        # agents; keep the fused score close to the strongest finding.
        fused = max(weighted_risk, max_risk * 0.85)
        elevated = [finding for finding in findings if finding.risk >= 0.50]
        if len(elevated) >= 2:
            fused += 0.06
        if len(elevated) >= 3:
            fused += 0.05
        by_type = {finding.fraud_type: finding for finding in findings}
        identity = by_type.get("identity")
        mule = by_type.get("mule_account")
        if identity and mule and identity.risk >= 0.60 and mule.risk >= 0.60:
            fused += 0.05
        risk = clamp(fused)

        review_threshold = max(0.30, threshold * 0.65)
        if risk >= threshold:
            decision = "BLOCK/REVIEW"
        elif risk >= review_threshold:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        if risk >= 0.80:
            band = "CRITICAL"
        elif risk >= 0.65:
            band = "HIGH"
        elif risk >= 0.45:
            band = "MEDIUM"
        else:
            band = "LOW"

        primary = max(findings, key=lambda finding: finding.risk, default=None)
        primary_type = (
            primary.fraud_type if primary is not None and primary.risk >= 0.35 else "none"
        )

        typologies: list[str] = []
        for finding in sorted(findings, key=lambda item: item.risk, reverse=True):
            for reason in finding.reasons:
                if reason.weight > 0 and reason.code not in typologies:
                    typologies.append(reason.code)
        typologies = typologies[:10]

        elevated_agents = [
            finding.agent for finding in findings if finding.risk >= 0.50
        ]
        clear_agents = [
            finding.agent for finding in findings if finding.risk < 0.35
        ]
        disagreements = min(len(elevated_agents), len(clear_agents))
        agreement = 1.0 - min(1.0, pstdev([f.risk for f in findings]) / 0.5)

        top_reasons = []
        for finding in sorted(findings, key=lambda item: item.risk, reverse=True):
            for reason in finding.reasons[:2]:
                if reason.weight > 0:
                    top_reasons.append(f"{reason.label} ({finding.display_name})")
        top_reasons = list(dict.fromkeys(top_reasons))[:4]

        if primary_type == "none" or not top_reasons:
            narrative = (
                "No agent found material fraud evidence. The payment can continue "
                "with normal monitoring."
            )
        else:
            narrative = (
                f"{primary.display_name} is the leading hypothesis. "
                f"The fused risk is {risk:.0%} ({band.lower()}). "
                f"Strongest signals: {', '.join(top_reasons)}."
            )

        actions: list[str] = []
        if risk >= review_threshold:
            for finding in sorted(findings, key=lambda item: item.risk, reverse=True):
                actions.extend(finding.actions)
        if not actions:
            actions = ["Continue normal monitoring; no agent escalation required"]
        actions = list(dict.fromkeys(actions))[:6]

        return {
            "overall_risk": round(risk, 4),
            "decision": decision,
            "band": band,
            "primary_fraud_type": primary_type,
            "threshold": round(threshold, 4),
            "review_threshold": round(review_threshold, 4),
            "agents": [finding.to_dict() for finding in findings],
            "weights": weights,
            "consensus": {
                "elevated_agents": elevated_agents,
                "clear_agents": clear_agents,
                "agreement": round(clamp(agreement), 4),
                "disagreements": disagreements,
            },
            "typologies": typologies,
            "narrative": narrative,
            "recommended_actions": actions,
            "signals": {
                "model_score": (
                    round(float(context.model_score), 4)
                    if context.model_score is not None
                    else None
                ),
                "online_score": (
                    round(float(context.online_score), 4)
                    if context.online_score is not None
                    else None
                ),
                "has_identity_signals": bool(context.identity),
                "has_mule_signals": bool(context.mule),
                "has_cheque_signals": bool(context.cheque),
            },
        }


def analyze_fraud(
    context: AgentContext,
    *,
    threshold: float = 0.55,
) -> dict[str, Any]:
    """Convenience wrapper used by the API and tests."""
    return FraudAgentOrchestrator().analyze(context, threshold=threshold)
