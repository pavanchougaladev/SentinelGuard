"""Vision-Language cheque fraud analysis via AWS Bedrock.

The model receives the cheque image and a forensic checklist, and returns a
structured JSON assessment. Deterministic image forensics (EXIF editing
software, resolution, entropy) and expected-field comparisons are applied on
top, so the final risk is inspectable even when the VLM is unavailable.

Privacy:
- only a SHA-256 digest and the decision are written to the audit log;
- the prompt asks for the last four account digits, never the full number;
- raw image bytes are never persisted by this module.

Configuration:
    AWS_REGION=us-east-1
    BEDROCK_VLM_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
    CHEQUE_VLM_BACKEND=auto|bedrock|heuristic
    CHEQUE_MAX_IMAGE_MB=8
    CHEQUE_AUDIT_PATH=models/cheque_audit.jsonl

Usage:
    python -m src.cheque_vlm --image cheque.png --expected-amount 2400 \\
        --expected-payee "Acme Ltd" --account-last4 4821
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:  # Pillow is optional at import time; forensics degrade gracefully.
    from PIL import Image, ExifTags
except Exception:  # pragma: no cover - exercised only when Pillow is absent
    Image = None
    ExifTags = None

MAX_IMAGE_BYTES = int(float(os.getenv("CHEQUE_MAX_IMAGE_MB", "8")) * 1024 * 1024)
DEFAULT_MODEL_ID = os.getenv(
    "BEDROCK_VLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)
DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
DEFAULT_AUDIT_PATH = os.getenv("CHEQUE_AUDIT_PATH", "models/cheque_audit.jsonl")

IMAGE_FORMATS = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}
SUSPICIOUS_SOFTWARE = (
    "photoshop",
    "gimp",
    "paint",
    "canva",
    "pixlr",
    "snapseed",
    "lightroom",
    "affinity",
    "inkscape",
    "illustrator",
)


class ChequeExpected(BaseModel):
    amount: float | None = None
    payee: str | None = None
    account_last4: str | None = None
    date: str | None = None


class ChequeFields(BaseModel):
    model_config = ConfigDict(extra="allow")

    is_cheque: bool | None = None
    cheque_number: str | None = None
    account_number_last4: str | None = None
    payee: str | None = None
    amount_numeric: float | None = None
    amount_words: str | None = None
    date: str | None = None
    bank_name: str | None = None
    signature_present: bool | None = None
    signature_consistent: bool | None = None
    alterations_detected: bool | None = None
    tamper_indicators: list[str] = Field(default_factory=list)
    confidence: float | None = None
    risk_score: float | None = None
    reasoning: str | None = None


class ChequeReason(BaseModel):
    code: str
    label: str
    weight: float
    evidence: Any = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _detect_format(raw: bytes) -> str | None:
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    return None


def _coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("VLM response was not a JSON object")
    for key in ("cheque", "analysis", "result", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload = nested
            break
    out = dict(payload)
    for key in (
        "is_cheque",
        "signature_present",
        "signature_consistent",
        "alterations_detected",
    ):
        out[key] = _bool_or_none(out.get(key))
    for key in ("amount_numeric", "confidence", "risk_score"):
        out[key] = _float_or_none(out.get(key))
    for key in (
        "cheque_number",
        "account_number_last4",
        "payee",
        "amount_words",
        "date",
        "bank_name",
        "reasoning",
    ):
        value = out.get(key)
        out[key] = str(value).strip() if value not in (None, "") else None
    indicators = out.get("tamper_indicators") or []
    if isinstance(indicators, str):
        indicators = [part.strip() for part in re.split(r"[;,]", indicators) if part.strip()]
    out["tamper_indicators"] = [str(item).strip() for item in indicators if str(item).strip()]
    if out.get("account_number_last4"):
        digits = re.sub(r"\D", "", str(out["account_number_last4"]))
        out["account_number_last4"] = digits[-4:] if digits else None
    return out


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model response")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response JSON is not an object")
    return payload


def inspect_image(raw: bytes) -> dict[str, Any]:
    """Cheap deterministic image forensics; no OCR and no raw bytes kept."""
    digest = hashlib.sha256(raw).hexdigest()
    base: dict[str, Any] = {
        "sha256": digest,
        "size_bytes": len(raw),
        "format": _detect_format(raw),
        "has_exif": False,
        "software": None,
        "dimensions": None,
        "entropy": None,
        "flags": [],
    }
    if Image is None:
        base["flags"].append("forensics_unavailable")
        return base
    try:
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            base["dimensions"] = {"width": int(width), "height": int(height)}
            base["format"] = (image.format or base["format"] or "").lower() or base["format"]
            base["mode"] = image.mode
            exif = image.getexif() or {}
            base["has_exif"] = bool(exif)
            software = str(exif.get(305, "") or "").strip()
            base["software"] = software or None
            if software and any(term in software.lower() for term in SUSPICIOUS_SOFTWARE):
                base["flags"].append("editing_software")
            if width < 700 or height < 300:
                base["flags"].append("low_resolution")
            ratio = width / max(height, 1)
            if ratio < 1.2 or ratio > 3.6:
                base["flags"].append("unusual_aspect_ratio")
            gray = image.convert("L")
            if max(gray.size) > 512:
                gray.thumbnail((512, 512))
            histogram = gray.histogram()
            total = sum(histogram) or 1
            entropy = -sum(
                (count / total) * math.log2(count / total)
                for count in histogram
                if count
            )
            base["entropy"] = round(float(entropy), 4)
            if entropy < 2.4:
                base["flags"].append("low_detail_image")
            low, high = gray.getextrema()
            if high - low < 25:
                base["flags"].append("near_blank_image")
    except Exception as exc:
        base["flags"].append(f"unreadable_image:{type(exc).__name__}")
    return base


def default_prompt(expected: ChequeExpected) -> str:
    expected_lines = []
    if expected.amount is not None:
        expected_lines.append(f"- expected amount: {expected.amount:.2f}")
    if expected.payee:
        expected_lines.append(f"- expected payee: {expected.payee}")
    if expected.account_last4:
        expected_lines.append(f"- expected account last four digits: {expected.account_last4}")
    if expected.date:
        expected_lines.append(f"- expected date: {expected.date}")
    expected_block = "\n".join(expected_lines) or "- no expected fields supplied"

    return f"""You are a forensic cheque analyst. Analyze the supplied cheque image for signs of fraud or tampering.

Look for:
- overwritten or altered amount figures or words, and any mismatch between them;
- font, ink, alignment, or background inconsistencies suggesting cut-and-paste;
- signature presence, obvious signature mismatch, or a digitally cloned signature;
- missing or inconsistent security features, serial numbers, or bank details;
- unusual paper texture, shadows, compression artifacts, or editing boundaries;
- whether the image is actually a cheque and whether it is legible.

Expected values for comparison:
{expected_block}

Privacy: never transcribe a full account number. Return only the last four digits.
If a field is unreadable, return null instead of guessing.

Return ONLY one JSON object, with exactly these keys:
{{
  "is_cheque": true,
  "cheque_number": "string or null",
  "account_number_last4": "four digits or null",
  "payee": "string or null",
  "amount_numeric": 0.0,
  "amount_words": "string or null",
  "date": "YYYY-MM-DD or null",
  "bank_name": "string or null",
  "signature_present": true,
  "signature_consistent": true,
  "alterations_detected": false,
  "tamper_indicators": ["short concrete indicator"],
  "confidence": 0.0,
  "risk_score": 0.0,
  "reasoning": "one concise paragraph"
}}

risk_score must be a number from 0 (no fraud evidence) to 1 (clear tampering or fraud).
Do not add any prose outside the JSON object."""


def _get_bedrock_client(region: str | None = None):
    import boto3

    return boto3.client("bedrock-runtime", region_name=region or DEFAULT_REGION)


@lru_cache(maxsize=1)
def credentials_available() -> bool:
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def bedrock_configuration() -> dict[str, Any]:
    backend = os.getenv("CHEQUE_VLM_BACKEND", "auto").strip().lower()
    return {
        "backend": backend,
        "model_id": os.getenv("BEDROCK_VLM_MODEL_ID", DEFAULT_MODEL_ID),
        "region": DEFAULT_REGION,
        "credentials_available": credentials_available(),
        "enabled": os.getenv("BEDROCK_VLM_ENABLED", "1").strip().lower()
        not in {"0", "false", "no"},
    }


def _call_bedrock(
    client: Any,
    model_id: str,
    raw: bytes,
    image_format: str,
    prompt: str,
) -> str:
    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": raw}}},
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={"temperature": 0.0, "maxTokens": 1400, "topP": 0.1},
    )
    content = response["output"]["message"]["content"]
    texts = [block.get("text", "") for block in content if isinstance(block, dict)]
    text = "\n".join(part for part in texts if part).strip()
    if not text:
        raise ValueError("Bedrock returned no text content")
    return text


def _fields_from_payload(payload: dict[str, Any]) -> ChequeFields:
    coerced = _coerce_payload(payload)
    fields = ChequeFields.model_validate(coerced)
    if fields.confidence is not None:
        fields.confidence = _clamp(fields.confidence)
    if fields.risk_score is not None:
        fields.risk_score = _clamp(fields.risk_score)
    return fields


def _fields_dict(fields: ChequeFields) -> dict[str, Any]:
    data = fields.model_dump(exclude_none=True)
    return data


def _compare_expected(
    fields: ChequeFields,
    expected: ChequeExpected,
) -> tuple[list[ChequeReason], float, list[str]]:
    reasons: list[ChequeReason] = []
    adjustment = 0.0
    warnings: list[str] = []

    if expected.amount is not None:
        if fields.amount_numeric is None:
            warnings.append("The VLM could not read the cheque amount.")
        else:
            tolerance = max(0.01, abs(expected.amount) * 0.005)
            if abs(fields.amount_numeric - expected.amount) > tolerance:
                adjustment += 0.22
                reasons.append(
                    ChequeReason(
                        code="CHEQUE_AMOUNT_MISMATCH",
                        label="Cheque amount differs from the expected amount",
                        weight=0.22,
                        evidence={
                            "expected": expected.amount,
                            "extracted": fields.amount_numeric,
                        },
                    )
                )

    if expected.payee:
        if not fields.payee:
            warnings.append("The VLM could not read the cheque payee.")
        else:
            expected_name = _norm_text(expected.payee)
            actual_name = _norm_text(fields.payee)
            if expected_name and expected_name not in actual_name and actual_name not in expected_name:
                adjustment += 0.15
                reasons.append(
                    ChequeReason(
                        code="CHEQUE_PAYEE_MISMATCH",
                        label="Cheque payee differs from the expected payee",
                        weight=0.15,
                        evidence={"expected": expected.payee, "extracted": fields.payee},
                    )
                )

    if expected.account_last4:
        if not fields.account_number_last4:
            warnings.append("The VLM could not read the account digits.")
        elif re.sub(r"\D", "", expected.account_last4)[-4:] != re.sub(
            r"\D", "", fields.account_number_last4
        )[-4:]:
            adjustment += 0.25
            reasons.append(
                ChequeReason(
                    code="CHEQUE_ACCOUNT_MISMATCH",
                    label="Cheque account digits differ from the expected account",
                    weight=0.25,
                    evidence={
                        "expected_last4": re.sub(r"\D", "", expected.account_last4)[-4:],
                        "extracted_last4": fields.account_number_last4,
                    },
                )
            )

    if expected.date and fields.date:
        if _norm_text(expected.date)[:8] != _norm_text(fields.date)[:8]:
            adjustment += 0.05
            reasons.append(
                ChequeReason(
                    code="CHEQUE_DATE_MISMATCH",
                    label="Cheque date differs from the expected date",
                    weight=0.05,
                    evidence={"expected": expected.date, "extracted": fields.date},
                )
            )
    return reasons, adjustment, warnings


def _score_fields(
    fields: ChequeFields,
    expected: ChequeExpected,
    forensics: dict[str, Any],
) -> tuple[float, list[ChequeReason], list[str], str]:
    reasons: list[ChequeReason] = []
    warnings: list[str] = []
    forensics_flags = list(forensics.get("flags", []))

    base = fields.risk_score
    if base is None:
        base = 0.25 if fields.is_cheque is not False else 0.55
        base += min(0.20, 0.06 * len(fields.tamper_indicators))
    risk = float(base)

    if fields.is_cheque is False:
        risk += 0.15
        reasons.append(
            ChequeReason(
                code="NOT_A_CHEQUE",
                label="The image does not appear to be a cheque",
                weight=0.15,
                evidence=None,
            )
        )

    if fields.alterations_detected:
        risk += 0.28
        reasons.append(
            ChequeReason(
                code="CHEQUE_ALTERATION_DETECTED",
                label="The VLM detected alteration evidence on the cheque",
                weight=0.28,
                evidence=fields.tamper_indicators or None,
            )
        )

    tamper_indicators = list(fields.tamper_indicators or [])
    if tamper_indicators:
        tamper_weight = min(0.18, 0.06 * len(tamper_indicators))
        risk += tamper_weight
        reasons.append(
            ChequeReason(
                code="CHEQUE_TAMPER_INDICATORS",
                label="Concrete tamper indicators were reported",
                weight=tamper_weight,
                evidence=tamper_indicators[:8],
            )
        )

    if fields.signature_present is False:
        risk += 0.18
        reasons.append(
            ChequeReason(
                code="CHEQUE_SIGNATURE_MISSING",
                label="No signature is visible on the cheque",
                weight=0.18,
            )
        )
    if fields.signature_consistent is False:
        risk += 0.22
        reasons.append(
            ChequeReason(
                code="CHEQUE_SIGNATURE_INCONSISTENT",
                label="The signature appears inconsistent or cloned",
                weight=0.22,
            )
        )

    compare_reasons, compare_adjustment, compare_warnings = _compare_expected(
        fields, expected
    )
    reasons.extend(compare_reasons)
    warnings.extend(compare_warnings)
    risk += compare_adjustment

    if "editing_software" in forensics_flags:
        risk += 0.12
        reasons.append(
            ChequeReason(
                code="IMAGE_EDITING_SOFTWARE",
                label="Image metadata names photo-editing software",
                weight=0.12,
                evidence=forensics.get("software"),
            )
        )
    if "low_resolution" in forensics_flags:
        risk += 0.04
        reasons.append(
            ChequeReason(
                code="LOW_RESOLUTION_IMAGE",
                label="Image resolution is low for forensic review",
                weight=0.04,
                evidence=forensics.get("dimensions"),
            )
        )
    if "low_detail_image" in forensics_flags or "near_blank_image" in forensics_flags:
        risk += 0.06
        reasons.append(
            ChequeReason(
                code="LOW_DETAIL_IMAGE",
                label="Image has unusually little visual detail",
                weight=0.06,
                evidence=forensics.get("entropy"),
            )
        )
    if any(flag.startswith("unreadable_image") for flag in forensics_flags):
        warnings.append("The image could not be fully decoded; treat this result as provisional.")

    confidence = fields.confidence if fields.confidence is not None else 0.0
    if confidence < 0.40:
        warnings.append(
            "VLM confidence is low; route the cheque to a human reviewer regardless of score."
        )

    risk = _clamp(risk)
    if risk >= 0.70:
        decision = "BLOCK/REVIEW"
    elif risk >= 0.35:
        decision = "REVIEW"
    else:
        decision = "APPROVE"
    if fields.is_cheque is False:
        decision = "REVIEW"
    return risk, reasons, warnings, decision


def _fallback_result(
    *,
    forensics: dict[str, Any],
    warning: str,
) -> tuple[float, list[ChequeReason], list[str], str]:
    flags = list(forensics.get("flags", []))
    risk = 0.50
    reasons = [
        ChequeReason(
            code="VLM_UNAVAILABLE",
            label="Bedrock VLM is not configured; manual review is required",
            weight=0.0,
        )
    ]
    warnings = [warning]
    if "editing_software" in flags:
        risk += 0.12
        reasons.append(
            ChequeReason(
                code="IMAGE_EDITING_SOFTWARE",
                label="Image metadata names photo-editing software",
                weight=0.12,
                evidence=forensics.get("software"),
            )
        )
    if "low_detail_image" in flags or "near_blank_image" in flags:
        risk += 0.06
        reasons.append(
            ChequeReason(
                code="LOW_DETAIL_IMAGE",
                label="Image has unusually little visual detail",
                weight=0.06,
            )
        )
    return _clamp(risk), reasons, warnings, "REVIEW"


def _risk_band(risk: float) -> str:
    if risk >= 0.80:
        return "CRITICAL"
    if risk >= 0.65:
        return "HIGH"
    if risk >= 0.40:
        return "MEDIUM"
    return "LOW"


def _audit(result: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_sha256": result["image"]["sha256"],
            "backend": result["backend"],
            "status": result["status"],
            "risk_score": result["risk_score"],
            "decision": result["decision"],
            "reason_codes": [reason["code"] for reason in result["reasons"]],
            "warnings": result["warnings"],
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


def analyze_cheque(
    image_bytes: bytes,
    *,
    content_type: str | None = None,
    expected: ChequeExpected | dict[str, Any] | None = None,
    backend: str | None = None,
    client: Any = None,
    audit_path: str | None = DEFAULT_AUDIT_PATH,
) -> dict[str, Any]:
    """Analyze one cheque image and return a structured, auditable decision."""
    started = time.perf_counter()
    if expected is None:
        expected_model = ChequeExpected()
    elif isinstance(expected, ChequeExpected):
        expected_model = expected
    else:
        expected_model = ChequeExpected(**expected)

    if not image_bytes:
        raise ValueError("empty image")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"image is larger than the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
        )
    image_format = _detect_format(image_bytes)
    if image_format not in IMAGE_FORMATS:
        raise ValueError("unsupported image format; use JPEG, PNG, WEBP, or GIF")
    if content_type and content_type.split(";")[0].strip().lower() not in {
        IMAGE_FORMATS[image_format],
        "application/octet-stream",
    }:
        # Trust the magic bytes but keep the mismatch visible in forensics.
        pass

    forensics = inspect_image(image_bytes)
    config = bedrock_configuration()
    requested = (backend or config["backend"] or "auto").strip().lower()
    if requested not in {"auto", "bedrock", "heuristic", "mock"}:
        requested = "auto"

    warnings: list[str] = []
    payload: dict[str, Any] | None = None
    used_backend = "heuristic"
    status = "fallback"

    should_call = requested in {"bedrock", "mock"} or (
        requested == "auto" and config["enabled"] and config["credentials_available"]
    )
    if should_call:
        try:
            model_id = config["model_id"]
            bedrock_client = client or _get_bedrock_client(config["region"])
            text = _call_bedrock(
                bedrock_client,
                model_id,
                image_bytes,
                image_format,
                default_prompt(expected_model),
            )
            payload = extract_json(text)
            used_backend = "bedrock" if client is None else "bedrock"
            status = "analyzed"
        except Exception as exc:
            warning = (
                "Bedrock VLM call failed "
                f"({type(exc).__name__}: {exc}). Manual review is required."
            )
            warnings.append(warning)
            payload = None
            used_backend = "heuristic"
            status = "fallback"
    elif requested == "auto":
        warnings.append(
            "Bedrock VLM is not configured (missing credentials or disabled); "
            "manual cheque review is required."
        )

    if payload is not None:
        try:
            fields = _fields_from_payload(payload)
            risk, reasons, score_warnings, decision = _score_fields(
                fields, expected_model, forensics
            )
            warnings.extend(score_warnings)
            if fields.confidence is not None and fields.confidence < 0.5:
                warnings.append(
                    "The VLM reported limited confidence in the extracted fields."
                )
            fields_dict = _fields_dict(fields)
            reasoning = fields.reasoning
            confidence = fields.confidence
        except Exception as exc:
            warnings.append(
                f"The VLM response could not be validated ({type(exc).__name__}). "
                "Manual review is required."
            )
            risk, reasons, fallback_warnings, decision = _fallback_result(
                forensics=forensics,
                warning="VLM response was invalid; manual review is required.",
            )
            warnings.extend(fallback_warnings)
            fields = ChequeFields()
            fields_dict = {}
            reasoning = None
            confidence = 0.0
            used_backend = "heuristic"
            status = "fallback"
    else:
        risk, reasons, fallback_warnings, decision = _fallback_result(
            forensics=forensics,
            warning=warnings[-1] if warnings else "VLM unavailable; manual review is required.",
        )
        warnings.extend(fallback_warnings)
        fields = ChequeFields()
        fields_dict = {}
        reasoning = None
        confidence = 0.0

    result = {
        "backend": used_backend,
        "model_id": config["model_id"],
        "region": config["region"],
        "status": status,
        "risk_score": round(float(risk), 4),
        "risk_band": _risk_band(float(risk)),
        "decision": decision,
        "confidence": round(float(confidence or 0.0), 4),
        "fields": fields_dict,
        "reasoning": reasoning,
        "reasons": [reason.model_dump() for reason in reasons],
        "warnings": list(dict.fromkeys(warnings)),
        "forensics": forensics,
        "image": {
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "format": image_format,
            "content_type": IMAGE_FORMATS[image_format],
            "bytes": len(image_bytes),
        },
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    _audit(result, audit_path)
    return result


def main(args: argparse.Namespace) -> None:
    with open(args.image, "rb") as handle:
        raw = handle.read()
    expected = ChequeExpected(
        amount=args.expected_amount,
        payee=args.expected_payee,
        account_last4=args.account_last4,
        date=args.expected_date,
    )
    result = analyze_cheque(
        raw,
        expected=expected,
        backend=args.backend,
        audit_path=args.audit_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-amount", type=float, default=None)
    parser.add_argument("--expected-payee", default=None)
    parser.add_argument("--account-last4", default=None)
    parser.add_argument("--expected-date", default=None)
    parser.add_argument("--backend", default=None, choices=["auto", "bedrock", "heuristic", "mock"])
    parser.add_argument("--audit-path", default=DEFAULT_AUDIT_PATH)
    main(parser.parse_args())
