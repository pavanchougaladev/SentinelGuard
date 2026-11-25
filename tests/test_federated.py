"""Unit tests for secure federated learning."""

import json
import os
import tempfile
import unittest

import numpy as np

from src.federated import (
    ClientUpdate,
    FederatedConfig,
    FederatedCoordinator,
    Params,
    PrivacyConfig,
    SecureAggregator,
    make_demo_clients,
    select_threshold,
)


class ParamsTests(unittest.TestCase):
    def test_vector_roundtrip_preserves_values(self):
        params = Params(np.array([0.5, -1.25, 2.0]), 0.75, ["a", "b", "c"])
        restored = Params.from_vector(params.to_vector(), params.feature_names)
        self.assertAlmostEqual(restored.intercept, 0.75)
        np.testing.assert_allclose(restored.coef, params.coef)

    def test_predict_proba_is_bounded(self):
        params = Params(np.array([50.0]), -50.0, ["x"])
        scores = params.predict_proba(np.array([[-1.0], [0.0], [1.0]]))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))


class SecureAggregationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.updates = [
            ClientUpdate(
                institution_id=f"bank_{index}",
                n_samples=100 + 20 * index,
                vector=rng.normal(0.0, 0.05, size=5),
                update_norm=0.0,
                released_norm=0.0,
                dp_sigma=0.0,
            )
            for index in range(4)
        ]

    def test_masked_sum_recovers_weighted_mean(self):
        aggregator = SecureAggregator(salt="test", scale=10**9)
        mean, metadata = aggregator.aggregate(self.updates)
        expected = sum(
            update.vector * update.n_samples for update in self.updates
        ) / sum(update.n_samples for update in self.updates)
        np.testing.assert_allclose(mean, expected, atol=1e-6)
        self.assertFalse(metadata["individual_updates_visible"])
        self.assertFalse(metadata["raw_rows_shared"])
        self.assertEqual(metadata["clients"], 4)

    def test_public_summary_never_contains_raw_rows(self):
        summary = self.updates[0].public_summary()
        self.assertFalse(summary["contains_raw_rows"])
        self.assertNotIn("vector", summary)
        self.assertEqual(len(summary["update_sha256_prefix"]), 16)


class FederatedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clients, cls.validation = make_demo_clients(
            n_clients=3, samples_per_client=220, validation_samples=400, seed=11
        )
        cls.coordinator = FederatedCoordinator(
            cls.clients,
            cls.validation,
            FederatedConfig(rounds=2, local_epochs=15, seed=11),
        )
        cls.report = cls.coordinator.run()

    def test_each_client_keeps_its_own_frame(self):
        for client in self.clients:
            self.assertGreater(client.n_samples, 0)
            self.assertGreater(client.n_positives, 0)
            self.assertLess(client.n_positives, client.n_samples)

    def test_report_is_json_serializable_and_has_metrics(self):
        text = json.dumps(self.report)
        self.assertGreater(len(text), 100)
        metrics = self.report["final_metrics"]
        for key in ("pr_auc", "roc_auc", "precision", "recall", "threshold"):
            self.assertIn(key, metrics)
        self.assertTrue(0.0 <= metrics["pr_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)

    def test_privacy_contract_is_explicit(self):
        privacy = self.report["privacy"]
        self.assertTrue(privacy["secure_aggregation"])
        self.assertFalse(privacy["raw_rows_shared"])
        self.assertFalse(privacy["individual_updates_visible"])
        self.assertTrue(privacy["model_updates_only"])
        self.assertFalse(privacy["dp"]["enabled"])

    def test_rounds_record_only_update_metadata(self):
        self.assertEqual(len(self.report["rounds"]), 2)
        for round_result in self.report["rounds"]:
            self.assertFalse(round_result["aggregation"]["raw_rows_shared"])
            for update in round_result["client_updates"]:
                self.assertFalse(update["contains_raw_rows"])
                self.assertNotIn("vector", update)

    def test_dp_is_optional_and_reports_epsilon(self):
        clients, validation = make_demo_clients(
            n_clients=2, samples_per_client=180, validation_samples=250, seed=5
        )
        coordinator = FederatedCoordinator(
            clients,
            validation,
            FederatedConfig(
                rounds=1,
                local_epochs=8,
                seed=5,
                privacy=PrivacyConfig(enabled=True, noise_multiplier=0.8),
            ),
        )
        report = coordinator.run()
        self.assertTrue(report["privacy"]["dp"]["enabled"])
        self.assertGreater(report["privacy"]["dp"]["epsilon_total_estimate"], 0.0)
        self.assertGreater(
            report["rounds"][0]["client_updates"][0]["dp_sigma"], 0.0
        )

    def test_save_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.coordinator.save(self.report, output_dir=directory)
            for path in paths.values():
                self.assertTrue(os.path.exists(path))
            with open(paths["report"], encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertEqual(loaded["client_count"], 3)


class ThresholdTests(unittest.TestCase):
    def test_cost_sensitive_threshold_prefers_recall(self):
        y = np.array([0] * 80 + [1] * 20)
        scores = np.r_[
            np.linspace(0.0, 0.4, 80),
            np.linspace(0.6, 1.0, 20),
        ]
        threshold = select_threshold(y, scores)
        predicted = scores >= threshold
        self.assertGreaterEqual(int(np.sum(predicted & (y == 1))), 15)

    def test_single_class_returns_neutral_threshold(self):
        self.assertEqual(select_threshold(np.zeros(10), np.zeros(10)), 0.5)


if __name__ == "__main__":
    unittest.main()
