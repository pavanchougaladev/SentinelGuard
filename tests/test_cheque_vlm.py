"""Unit tests for Bedrock cheque analysis and image forensics."""

import hashlib
import json
import os
import tempfile
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from src.cheque_vlm import (
    ChequeExpected,
    analyze_cheque,
    extract_json,
    inspect_image,
)


def noise_png(width=1200, height=500) -> bytes:
    rng = np.random.default_rng(3)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(array, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def blank_png(width=1200, height=500) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeBedrock:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    def converse(self, **kwargs):
        self.calls += 1
        return {
            "output": {
                "message": {
                    "content": [{"text": self.payload}]
                }
            }
        }


class ExtractJsonTests(unittest.TestCase):
    def test_fenced_json_is_parsed(self):
        text = 'Here is the result:\n```json\n{"is_cheque": true, "risk_score": 0.2}\n```'
        self.assertEqual(extract_json(text)["risk_score"], 0.2)

    def test_missing_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json("no structured output here")

    def test_array_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_json("[1, 2, 3]")


class ForensicsTests(unittest.TestCase):
    def test_blank_image_is_flagged_low_detail(self):
        report = inspect_image(blank_png())
        self.assertEqual(report["dimensions"], {"width": 1200, "height": 500})
        self.assertIn("near_blank_image", report["flags"])
        self.assertIn("low_detail_image", report["flags"])

    def test_noise_image_has_entropy_and_hash(self):
        raw = noise_png()
        report = inspect_image(raw)
        self.assertEqual(report["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertGreater(report["entropy"], 5.0)
        self.assertNotIn("near_blank_image", report["flags"])

    def test_low_resolution_is_flagged(self):
        report = inspect_image(blank_png(400, 200))
        self.assertIn("low_resolution", report["flags"])


class AnalyzeChequeTests(unittest.TestCase):
    def payload(self, **overrides):
        base = {
            "is_cheque": True,
            "cheque_number": "100245",
            "account_number_last4": "1234",
            "payee": "Acme Trading Ltd",
            "amount_numeric": 2400,
            "amount_words": "two thousand four hundred",
            "date": "2026-01-01",
            "bank_name": "Example Bank",
            "signature_present": True,
            "signature_consistent": True,
            "alterations_detected": False,
            "tamper_indicators": [],
            "confidence": 0.86,
            "risk_score": 0.12,
            "reasoning": "The cheque looks consistent.",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_clean_cheque_is_approved(self):
        client = FakeBedrock(self.payload())
        result = analyze_cheque(
            noise_png(),
            expected=ChequeExpected(
                amount=2400, payee="Acme Trading", account_last4="1234"
            ),
            backend="bedrock",
            client=client,
            audit_path=None,
        )
        self.assertEqual(result["backend"], "bedrock")
        self.assertEqual(result["status"], "analyzed")
        self.assertEqual(result["decision"], "APPROVE")
        self.assertLess(result["risk_score"], 0.35)
        self.assertEqual(client.calls, 1)
        self.assertEqual(result["fields"]["account_number_last4"], "1234")

    def test_tampering_and_mismatches_escalate(self):
        client = FakeBedrock(
            self.payload(
                amount_numeric=3200,
                payee="Someone Else",
                account_number_last4="9999",
                alterations_detected=True,
                signature_consistent=False,
                tamper_indicators=["overwritten amount", "cut-and-paste background"],
                risk_score=0.7,
                confidence=0.9,
            )
        )
        result = analyze_cheque(
            noise_png(),
            expected=ChequeExpected(
                amount=2400, payee="Acme Trading", account_last4="1234"
            ),
            backend="bedrock",
            client=client,
            audit_path=None,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertEqual(result["decision"], "BLOCK/REVIEW")
        self.assertIn("CHEQUE_ALTERATION_DETECTED", codes)
        self.assertIn("CHEQUE_AMOUNT_MISMATCH", codes)
        self.assertIn("CHEQUE_PAYEE_MISMATCH", codes)
        self.assertIn("CHEQUE_ACCOUNT_MISMATCH", codes)
        self.assertIn("CHEQUE_SIGNATURE_INCONSISTENT", codes)
        self.assertGreater(result["risk_score"], 0.8)

    def test_invalid_vlm_response_falls_back_to_review(self):
        client = FakeBedrock("I cannot read this image.")
        result = analyze_cheque(
            noise_png(),
            backend="bedrock",
            client=client,
            audit_path=None,
        )
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["decision"], "REVIEW")
        self.assertTrue(result["warnings"])
        self.assertEqual(result["fields"], {})

    def test_heuristic_backend_never_claims_a_clean_cheque(self):
        result = analyze_cheque(
            noise_png(), backend="heuristic", audit_path=None
        )
        self.assertEqual(result["backend"], "heuristic")
        self.assertEqual(result["decision"], "REVIEW")
        self.assertGreaterEqual(result["risk_score"], 0.5)
        self.assertIn(
            "VLM_UNAVAILABLE", {reason["code"] for reason in result["reasons"]}
        )

    def test_unsupported_format_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_cheque(b"not an image", backend="heuristic", audit_path=None)

    def test_audit_log_has_digest_but_no_image_bytes(self):
        raw = noise_png()
        with tempfile.TemporaryDirectory() as directory:
            audit_path = os.path.join(directory, "audit.jsonl")
            result = analyze_cheque(
                raw,
                backend="heuristic",
                audit_path=audit_path,
            )
            with open(audit_path, encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(lines), 1)
        record = lines[0]
        self.assertEqual(record["image_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["decision"], result["decision"])
        self.assertNotIn("image_bytes", record)
        self.assertNotIn("fields", record)
        self.assertGreater(len(raw), 1000)


if __name__ == "__main__":
    unittest.main()
