"""Unit tests for the specialized fraud agents and orchestrator."""

import unittest

from src.agents import (
    AgentContext,
    BaseAgent,
    FraudAgentOrchestrator,
    IdentityAgent,
    MuleAgent,
    TransactionAgent,
    agent_catalog,
    analyze_fraud,
)


def high_risk_context() -> AgentContext:
    return AgentContext(
        transaction={
            "amount": 9_500,
            "device_risk": 0.92,
            "ip_risk": 0.88,
            "tx_count_24h": 28,
            "account_age_days": 3,
            "country": "RU",
            "tx_type": "transfer",
            "merchant_category": "crypto",
        },
        model_score=0.86,
        model_reasons=[
            {"code": "HIGH_DEVICE_RISK", "label": "High device risk", "severity": "high"}
        ],
        identity={
            "kyc_status": "failed",
            "identity_mismatch": True,
            "device_accounts_7d": 12,
            "sanctions_hit": True,
            "synthetic_identity_score": 0.8,
        },
        mule={
            "inbound_count_24h": 40,
            "outbound_count_24h": 35,
            "unique_senders_7d": 30,
            "unique_receivers_7d": 18,
            "pass_through_ratio": 0.95,
            "dormant_days_before_activity": 120,
            "round_amount_ratio": 0.8,
            "shared_device_accounts": 9,
            "beneficiary_risk": 0.9,
            "near_threshold_ratio": 0.6,
            "account_age_days": 20,
            "tx_count_24h": 40,
        },
    )


class CatalogTests(unittest.TestCase):
    def test_three_specialists_are_exposed(self):
        catalog = agent_catalog()
        names = {item["name"] for item in catalog}
        self.assertEqual(
            names, {"transaction_agent", "identity_agent", "mule_agent"}
        )
        for item in catalog:
            self.assertTrue(item["description"])
            self.assertGreater(item["default_weight"], 0.0)


class AgentTests(unittest.TestCase):
    def test_transaction_agent_combines_model_and_rules(self):
        finding = TransactionAgent().evaluate(high_risk_context())
        self.assertGreater(finding.risk, 0.6)
        codes = {reason.code for reason in finding.reasons}
        self.assertIn("HIGH_DEVICE_RISK", codes)
        self.assertIn("VELOCITY_SPIKE", codes)
        self.assertGreater(finding.confidence, 0.5)

    def test_transaction_agent_handles_missing_model_score(self):
        finding = TransactionAgent().evaluate(
            AgentContext(transaction={"amount": 50.0, "device_risk": 0.1})
        )
        self.assertLess(finding.risk, 0.4)
        self.assertLess(finding.confidence, 0.5)

    def test_identity_agent_escalates_sanctions(self):
        finding = IdentityAgent().evaluate(high_risk_context())
        self.assertGreater(finding.risk, 0.7)
        self.assertIn("SANCTIONS_HIT", {reason.code for reason in finding.reasons})
        self.assertTrue(
            any("compliance" in action.lower() for action in finding.actions)
        )

    def test_identity_agent_is_quiet_without_signals(self):
        finding = IdentityAgent().evaluate(AgentContext())
        self.assertLess(finding.risk, 0.2)
        self.assertLessEqual(finding.confidence, 0.25)

    def test_mule_agent_detects_fan_in_and_pass_through(self):
        finding = MuleAgent().evaluate(high_risk_context())
        codes = {reason.code for reason in finding.reasons}
        self.assertIn("FAST_PASS_THROUGH", codes)
        self.assertIn("FAN_IN", codes)
        self.assertGreater(finding.risk, 0.7)

    def test_mule_agent_flags_structuring_below_threshold(self):
        finding = MuleAgent().evaluate(
            AgentContext(
                mule={
                    "near_threshold_ratio": 0.65,
                    "avg_outbound_amount": 9_200,
                    "unique_receivers_7d": 8,
                }
            )
        )
        self.assertIn("STRUCTURING", {reason.code for reason in finding.reasons})


class OrchestratorTests(unittest.TestCase):
    def test_high_risk_case_blocks(self):
        result = analyze_fraud(high_risk_context())
        self.assertEqual(result["decision"], "BLOCK/REVIEW")
        self.assertGreaterEqual(result["overall_risk"], 0.55)
        self.assertIn(result["primary_fraud_type"], {"transaction", "identity", "mule_account"})
        self.assertGreaterEqual(len(result["agents"]), 3)
        self.assertTrue(result["recommended_actions"])
        self.assertTrue(result["narrative"])

    def test_low_risk_case_approves(self):
        result = analyze_fraud(
            AgentContext(
                transaction={
                    "amount": 42.0,
                    "device_risk": 0.05,
                    "ip_risk": 0.05,
                    "tx_count_24h": 1,
                    "account_age_days": 900,
                    "country": "IN",
                },
                model_score=0.03,
                identity={"kyc_status": "verified", "account_age_days": 900},
            )
        )
        self.assertEqual(result["decision"], "APPROVE")
        self.assertLess(result["overall_risk"], 0.35)
        self.assertEqual(result["primary_fraud_type"], "none")

    def test_corroboration_raises_fused_risk(self):
        context = high_risk_context()
        orchestrator = FraudAgentOrchestrator()
        result = orchestrator.analyze(context)
        individual = [item["risk"] for item in result["agents"]]
        self.assertGreaterEqual(result["overall_risk"], max(individual) * 0.85)
        self.assertGreaterEqual(len(result["consensus"]["elevated_agents"]), 2)

    def test_agent_failure_is_isolated(self):
        class BrokenAgent(BaseAgent):
            name = "broken_agent"
            display_name = "Broken agent"
            fraud_type = "broken"

            def evaluate(self, context):
                raise RuntimeError("boom")

        orchestrator = FraudAgentOrchestrator(
            agents=[TransactionAgent(), BrokenAgent()],
            weights={"transaction_agent": 0.5, "broken_agent": 0.5},
        )
        result = orchestrator.analyze(high_risk_context())
        broken = next(
            item for item in result["agents"] if item["agent"] == "broken_agent"
        )
        self.assertIn("boom", broken["error"])
        self.assertIn("decision", result)

    def test_threshold_override_controls_review_band(self):
        result = analyze_fraud(
            AgentContext(transaction={"amount": 500.0, "device_risk": 0.4}),
            threshold=0.9,
        )
        self.assertEqual(result["threshold"], 0.9)
        self.assertGreaterEqual(result["review_threshold"], 0.3)


if __name__ == "__main__":
    unittest.main()
