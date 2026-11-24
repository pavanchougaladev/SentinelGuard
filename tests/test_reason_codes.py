"""Unit tests for analyst reason codes."""

import unittest

from src.reason_codes import code_for, format_reason_codes, reason_codes


def reason(feature, shap, direction="raises risk"):
    return {"feature": feature, "shap": shap, "direction": direction}


class CodeMappingTests(unittest.TestCase):
    def test_known_features_map_to_stable_codes(self):
        self.assertEqual(code_for("device_risk")[0], "HIGH_DEVICE_RISK")
        self.assertEqual(code_for("anomaly")[0], "ANOMALY_SIGNAL")
        self.assertEqual(code_for("mcc_crypto")[0], "HIGH_RISK_MERCHANT")
        self.assertEqual(code_for("country_RU")[0], "HIGH_RISK_GEOGRAPHY")
        self.assertEqual(code_for("V14")[0], "PCA_DEVIATION")

    def test_unknown_feature_falls_back_to_uppercase_name(self):
        self.assertEqual(code_for("custom_signal")[0], "CUSTOM_SIGNAL")


class ReasonCodeTests(unittest.TestCase):
    def test_codes_are_ranked_by_contribution(self):
        codes = reason_codes(
            [
                reason("log_amount", 0.20),
                reason("device_risk", 0.35),
                reason("ip_risk", 0.10),
                reason("account_age_days", -0.40, "lowers risk"),
            ],
            limit=3,
        )
        self.assertEqual(
            [item["code"] for item in codes],
            ["HIGH_DEVICE_RISK", "HIGH_AMOUNT", "HIGH_IP_RISK"],
        )
        self.assertEqual([item["severity"] for item in codes], ["high", "medium", "low"])

    def test_duplicate_codes_are_collapsed(self):
        codes = reason_codes(
            [reason("mcc_crypto", 0.30), reason("mcc_gambling", 0.20)],
            limit=4,
        )
        self.assertEqual(len(codes), 1)
        self.assertEqual(codes[0]["code"], "HIGH_RISK_MERCHANT")

    def test_mitigating_only_reasons_fall_back_to_model_score(self):
        codes = reason_codes([reason("ip_risk", -0.5, "lowers risk")])
        self.assertEqual([item["code"] for item in codes], ["MODEL_RISK_SCORE"])

    def test_limit_is_respected(self):
        codes = reason_codes(
            [
                reason(feature, 0.1)
                for feature in ("V1", "device_risk", "log_amount", "ip_risk", "anomaly")
            ],
            limit=2,
        )
        self.assertEqual(len(codes), 2)

    def test_summary_is_semicolon_separated(self):
        codes = reason_codes([reason("device_risk", 0.4), reason("log_amount", 0.2)])
        summary = format_reason_codes(codes)
        self.assertIn("HIGH_DEVICE_RISK", summary)
        self.assertIn("; ", summary)


if __name__ == "__main__":
    unittest.main()
