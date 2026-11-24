"""Unit tests for the cost-sensitive threshold policy engine."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.policy import (
    POLICY_ORDER,
    baseline_report,
    build_policy_artifacts,
    calibration_report,
    select_policies,
    threshold_table,
)


class ThresholdTableTests(unittest.TestCase):
    def setUp(self):
        self.y = [0, 0, 1, 1]
        self.scores = [0.10, 0.20, 0.80, 0.90]
        self.amounts = [10.0, 10.0, 100.0, 200.0]
        self.table = threshold_table(self.y, self.scores, self.amounts)

    def test_table_has_one_row_per_candidate(self):
        self.assertFalse(self.table.empty)
        self.assertTrue({"threshold", "precision", "recall", "f1", "cost"} <= set(self.table.columns))

    def test_perfect_cutoff_has_zero_false_positives(self):
        row = self.table.iloc[(self.table["threshold"] - 0.80).abs().argmin()]
        self.assertEqual(int(row["tp"]), 2)
        self.assertEqual(int(row["fp"]), 0)
        self.assertAlmostEqual(float(row["precision"]), 1.0, places=6)
        self.assertAlmostEqual(float(row["recall"]), 1.0, places=6)

    def test_missed_fraud_value_enters_cost(self):
        high = self.table.iloc[(self.table["threshold"] - 0.90).abs().argmin()]
        self.assertGreaterEqual(int(high["fn"]), 1)
        self.assertGreaterEqual(float(high["missed_value"]), 100.0)
        self.assertGreater(float(high["cost"]), 0.0)


class PolicySelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.y = np.r_[np.zeros(180, dtype=int), np.ones(20, dtype=int)]
        self.scores = np.r_[rng.beta(2, 6, 180), rng.beta(6, 2, 20)]
        self.amounts = rng.lognormal(4, 1, 200)
        self.table = threshold_table(self.y, self.scores, self.amounts)
        self.policies = select_policies(self.table)

    def test_all_policies_selected_in_order(self):
        self.assertEqual(
            [row["policy"] for row in self.policies], list(POLICY_ORDER)
        )

    def test_high_precision_respects_target(self):
        row = next(r for r in self.policies if r["policy"] == "high_precision")
        self.assertGreaterEqual(row["precision"], 0.90)

    def test_review_capacity_respects_budget(self):
        row = next(r for r in self.policies if r["policy"] == "review_capacity")
        self.assertLessEqual(row["flagged_rate"], 0.02 + 1e-6)

    def test_cost_optimized_is_cheapest_or_tied(self):
        row = next(r for r in self.policies if r["policy"] == "cost_optimized")
        self.assertAlmostEqual(float(row["cost"]), float(self.table["cost"].min()), places=2)


class DiagnosticTests(unittest.TestCase):
    def test_calibration_of_confident_predictions(self):
        report = calibration_report([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99])
        self.assertLess(report["brier"], 0.01)
        self.assertEqual(len(report["curve"]), 10)
        self.assertEqual(sum(bin_["count"] for bin_ in report["curve"]), 4)

    def test_baselines_cover_trivial_strategies(self):
        report = baseline_report([0] * 90 + [1] * 10, [0.1] * 90 + [0.9] * 10)
        self.assertAlmostEqual(report["prevalence"], 0.10, places=6)
        self.assertEqual(report["flag_all"]["recall"], 1.0)
        self.assertEqual(report["majority"]["flagged_rate"], 0.0)


class ArtifactTests(unittest.TestCase):
    def test_artifacts_are_written(self):
        y = [0] * 50 + [1] * 10
        scores = np.r_[np.linspace(0.01, 0.4, 50), np.linspace(0.6, 0.95, 10)]
        amounts = np.full(60, 50.0)

        with tempfile.TemporaryDirectory() as tmp:
            payload = build_policy_artifacts(
                y, scores, amounts, suffix="_test", output_dir=tmp,
                operating_threshold=0.5,
            )
            for name in ("threshold_policy_test.json", "threshold_policy_test.csv",
                         "threshold_policy_test.md"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)
            with open(os.path.join(tmp, "threshold_policy_test.json")) as fh:
                stored = json.load(fh)
            self.assertEqual(len(stored["policies"]), len(POLICY_ORDER))
            self.assertEqual(stored["operating"]["policy"], "operating")
            self.assertAlmostEqual(payload["operating"]["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
