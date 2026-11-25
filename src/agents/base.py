"""Shared primitives for the specialized fraud agents.

The agents are independent, deterministic evaluators. Each one sees only the
signals relevant to its mandate and returns a scored finding with evidence. The
orchestrator fuses those findings; no agent writes to another agent's state.

This is deliberately not an autonomous LLM swarm. Rules remain inspectable and
reproducible, which matters for a regulated fraud decision. A VLM or a human
analyst can supply extra evidence (for example a cheque risk score) through the
context without changing the agents' contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def noisy_or(weights: list[float]) -> float:
    """Combine independent risk contributions in [0, 1] without overflow."""
    remaining = 1.0
    for weight in weights:
        remaining *= 1.0 - clamp(weight, 0.0, 0.999)
    return clamp(1.0 - remaining)


def risk_status(risk: float) -> str:
    if risk >= 0.80:
        return "critical"
    if risk >= 0.60:
        return "elevated"
    if risk >= 0.35:
        return "watch"
    return "clear"


@dataclass
class Reason:
    code: str
    label: str
    weight: float = 0.0
    evidence: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "weight": round(float(self.weight), 4),
            "evidence": self.evidence,
        }


@dataclass
class AgentContext:
    """Normalized evidence handed to every agent."""

    transaction: dict[str, Any] = field(default_factory=dict)
    model_score: float | None = None
    model_reasons: list[dict[str, Any]] = field(default_factory=list)
    identity: dict[str, Any] = field(default_factory=dict)
    mule: dict[str, Any] = field(default_factory=dict)
    cheque: dict[str, Any] = field(default_factory=dict)
    online_score: float | None = None
    threshold: float = 0.55
    history: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentFinding:
    agent: str
    display_name: str
    fraud_type: str
    risk: float
    confidence: float
    status: str
    reasons: list[Reason] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "display_name": self.display_name,
            "fraud_type": self.fraud_type,
            "risk": round(float(self.risk), 4),
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "evidence": self.evidence,
            "actions": self.actions,
            "error": self.error,
        }


class BaseAgent:
    name = "base_agent"
    display_name = "Base agent"
    fraud_type = "unknown"
    description = ""

    def evaluate(self, context: AgentContext) -> AgentFinding:  # pragma: no cover
        raise NotImplementedError

    def finding(
        self,
        *,
        risk: float,
        confidence: float,
        reasons: list[Reason],
        evidence: dict[str, Any],
        actions: list[str],
        error: str | None = None,
    ) -> AgentFinding:
        ordered = sorted(reasons, key=lambda item: item.weight, reverse=True)
        return AgentFinding(
            agent=self.name,
            display_name=self.display_name,
            fraud_type=self.fraud_type,
            risk=clamp(risk),
            confidence=clamp(confidence),
            status=risk_status(risk),
            reasons=ordered,
            evidence=evidence,
            actions=list(dict.fromkeys(actions)),
            error=error,
        )
