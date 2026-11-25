"""Unit tests for River online learning, adaptive thresholds, and drift."""

import json
import os
import tempfile
import unittest

import numpy as np

from src.make_data import make
from src.online_learning import (
    RIVER_AVAILABLE,
    AdaptiveThreshold,
    OnlineFraudLearner,
    load_online_history,
    optimal_threshold,
)


class ThresholdTests(unittest.TestCase):
    def test_optimal_threshold_needs_enough_labels(self):
        self.assertIsNone(optimal_threshold([0, 1], [0.1, 0.9]))

    def test_optimal_threshold_separates_classes(self):
        y = np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)]
        scores = np.r_[
            np.linspace(0.01, 0.35, 80),
            np.linspace(0.65, 0.99, 20),
        ]
        threshold = optimal_threshold(y, scores)
        self.assertIsNotNone(threshold)
        self.assertTrue(0.35 < threshold <= 0.65)

    def test_adaptive_threshold_moves_smoothly_and_stays_bounded(self):
        threshold = AdaptiveThreshold(0.5, smoothing=0.2)
        y = np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)]
        scores = np.r_[
            np.linspace(0.01, 0.35, 80),
            np.linspace(0.65, 0.99, 20),
        ]
        updated = threshold.update(y, scores)
        self.assertGreater(updated, 0.5)
        self.assertLessEqual(updated, 0.95)
        self.assertGreaterEqual(updated, 0.05)
        self.assertGreaterEqual(len(threshold.history), 2)


@unittest.skipUnless(RIVER_AVAILABLE, "River is not installed")
class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.directory.name, "online.pkl")
        self.event_path = os.path.join(self.directory.name, "events.jsonl")
        self.frame = make(500, seed=17)

    def tearDown(self):
        self.directory.cleanup()

    def learner(self):
        return OnlineFraudLearner(
            self.model_path,
            self.event_path,
            retune_every=20,
            save_every=50,
        )

    def test_bootstrap_warms_the_model_and_persists(self):
        learner = self.learner()
        status = learner.bootstrap(self.frame.iloc[:300], max_rows=300)
        self.assertEqual(status["samples_seen"], 300)
        self.assertTrue(status["warm"])
        self.assertTrue(os.path.exists(self.model_path))
        self.assertIsNotNone(status["rolling"]["f1"])

        reloaded = self.learner()
        self.assertEqual(reloaded.samples_seen, 300)
        self.assertTrue(reloaded.warm)

    def test_learn_returns_event_and_updates_state(self):
        learner = self.learner()
        learner.bootstrap(self.frame.iloc[:200], max_rows=200)
        row = self.frame.iloc[250].to_dict()
        result = learner.learn(row, int(row["is_fraud"]))
        event = result["event"]
        self.assertEqual(event["samples_seen"], 201)
        self.assertIn(event["label"], (0, 1))
        self.assertGreaterEqual(event["score_before"], 0.0)
        self.assertLessEqual(event["score_after"], 1.0)
        self.assertEqual(result["status"]["samples_seen"], 201)

    def test_rejects_kaggle_schema_rows(self):
        import pandas as pd

        frame = pd.DataFrame({"V1": [0.1], "Amount": [10.0], "Class": [0]})
        with self.assertRaises(ValueError):
            self.learner().bootstrap(frame)

    def test_predict_is_deterministic_and_bounded(self):
        learner = self.learner()
        learner.bootstrap(self.frame.iloc[:250], max_rows=250)
        row = self.frame.iloc[300].to_dict()
        first = learner.predict(row)
        second = learner.predict(row)
        self.assertEqual(first["online_score"], second["online_score"])
        self.assertGreaterEqual(first["online_score"], 0.0)
        self.assertLessEqual(first["online_score"], 1.0)
        self.assertIn(first["decision"], {"APPROVE", "BLOCK/REVIEW"})

    def test_events_are_logged_as_jsonl(self):
        learner = self.learner()
        learner.bootstrap(self.frame.iloc[:200], max_rows=200)
        for index in range(210, 215):
            row = self.frame.iloc[index].to_dict()
            learner.learn(row, int(row["is_fraud"]))
        history = load_online_history(self.event_path)
        self.assertEqual(len(history), 5)
        self.assertTrue(all("score_before" in item for item in history))

    def test_reset_clears_state_and_artifacts(self):
        learner = self.learner()
        learner.bootstrap(self.frame.iloc[:200], max_rows=200)
        status = learner.reset()
        self.assertEqual(status["samples_seen"], 0)
        self.assertFalse(status["warm"])
        self.assertEqual(learner.buffer.maxlen, 1000)
        self.assertFalse(os.path.exists(self.event_path))

    def test_status_is_json_serializable(self):
        learner = self.learner()
        learner.bootstrap(self.frame.iloc[:220], max_rows=220)
        text = json.dumps(learner.status())
        self.assertIn("threshold_history", text)
        self.assertIn("rolling", text)


if __name__ == "__main__":
    unittest.main()
