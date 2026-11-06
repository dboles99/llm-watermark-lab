from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from watermark_lab.statistics import (
    CalibrationManifestSemanticError,
    assert_calibration_manifest_semantics,
    minimum_zero_event_sample_size,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (ROOT / "schemas" / "calibration-manifest.schema.json").read_text(encoding="utf-8")
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
SHA256 = "a" * 64


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "calibration_id": "cal_kgw_en_0001",
        "status": "frozen",
        "method": {
            "method_id": "kgw",
            "implementation_id": "kgw-author-compatible-v1",
            "implementation_sha256": SHA256,
            "author_repository_url": "https://github.com/jwkirchenbauer/lm-watermarking",
            "author_commit": "b" * 40,
            "parity_status": "passed",
            "parity_report_sha256": SHA256,
        },
        "tokenizer": {
            "tokenizer_id": "openai-community/gpt2",
            "revision": "c" * 40,
            "asset_sha256": SHA256,
            "normalization": "tokenizer-native-no-pre-normalization",
            "bos_policy": "exclude",
            "eos_policy": "exclude",
            "special_token_policy": "exclude",
        },
        "locale": "en",
        "locale_fallback_order": ["zh-Hant-TW", "zh-TW", "zh-Hant", "en"],
        "partitions": [
            {"role": role, "partition_id": f"kgw-en-{role}", "manifest_sha256": SHA256}
            for role in ("development", "calibration", "attack-tuning", "final-locked")
        ],
        "sampling": {
            "independent_unit": "author-document-cluster",
            "planned_independent_negatives": 2995,
            "planned_independent_positives": 500,
            "cluster_fields": ["author_id", "document_id"],
            "stopping_rule": "Fixed sample; no optional stopping.",
            "exclusion_policy_sha256": SHA256,
        },
        "scoring": {
            "gamma": 0.25,
            "context_width": 1,
            "seeding_scheme": "simple_1",
            "repeated_ngram_policy": "first-full-ngram-only",
            "minimum_scored_tokens": 100,
            "decision_threshold": 4.0,
        },
        "inference": {
            "target_fpr": 0.001,
            "confidence_level": 0.95,
            "false_positive_interval": "clopper-pearson-exact-one-sided-upper",
            "true_positive_interval": "clopper-pearson-exact-two-sided",
            "claim_basis": "zero-event-upper-bound",
            "multiplicity_family_id": "kgw-en-primary-v1",
            "multiplicity_policy": "single-predeclared-test",
            "analysis_code_sha256": SHA256,
        },
        "created_at": "2026-08-14T12:00:00Z",
        "frozen_at": "2026-08-14T12:05:00Z",
    }


def _messages(manifest: dict[str, object]) -> list[str]:
    return [error.message for error in VALIDATOR.iter_errors(manifest)]


def test_frozen_calibration_manifest_validates() -> None:
    Draft202012Validator.check_schema(SCHEMA)
    assert _messages(_manifest()) == []
    assert_calibration_manifest_semantics(_manifest())


def test_zh_cn_and_noncanonical_fallback_are_rejected() -> None:
    manifest = _manifest()
    manifest["locale"] = "zh-CN"
    assert any("is not one of" in message for message in _messages(manifest))

    manifest = _manifest()
    manifest["locale_fallback_order"] = ["zh-Hant-TW", "zh-CN", "en"]
    assert any("was expected" in message for message in _messages(manifest))


def test_zero_event_point_one_percent_claim_requires_2995_independent_units() -> None:
    manifest = _manifest()
    sampling = manifest["sampling"]
    assert isinstance(sampling, dict)
    sampling["planned_independent_negatives"] = 2994

    assert any("less than the minimum of 2995" in message for message in _messages(manifest))


def test_frozen_manifest_requires_parity_and_analysis_evidence() -> None:
    manifest = _manifest()
    method = manifest["method"]
    inference = manifest["inference"]
    assert isinstance(method, dict)
    assert isinstance(inference, dict)
    method["parity_status"] = "not-run"
    method.pop("parity_report_sha256")
    inference.pop("analysis_code_sha256")

    messages = _messages(manifest)
    assert any("'passed' was expected" in message for message in messages)
    assert any("analysis_code_sha256" in message for message in messages)


def test_all_four_partition_roles_are_required() -> None:
    manifest = _manifest()
    partitions = manifest["partitions"]
    assert isinstance(partitions, list)
    assert isinstance(partitions[-1], dict)
    partitions[-1]["role"] = "development"

    assert any("does not contain items matching" in message for message in _messages(manifest))


def test_semantic_validator_computes_stricter_dynamic_sample_floor() -> None:
    manifest = _manifest()
    inference = manifest["inference"]
    sampling = manifest["sampling"]
    assert isinstance(inference, dict)
    assert isinstance(sampling, dict)
    inference["target_fpr"] = 0.0001
    inference["confidence_level"] = 0.99
    required = minimum_zero_event_sample_size(0.0001, 0.99)
    sampling["planned_independent_negatives"] = required - 1

    with pytest.raises(CalibrationManifestSemanticError, match=f"at least {required}"):
        assert_calibration_manifest_semantics(manifest)


def _ocr_channel(**overrides: object) -> dict[str, object]:
    """A well-formed OCR input-channel block."""

    ocr: dict[str, object] = {
        "engine_id": "rapidocr",
        "engine_version": "1.2.3",
        "profile_fingerprint": SHA256,
        "language": "en",
        "confidence_floor": 0.8,
        "degradation": {"name": "clean", "severity": 0.0, "seed": 0},
    }
    ocr.update(overrides)
    return {"channel": "ocr", "ocr": ocr}


class TestOcrInputChannel:
    """An OCR-derived calibration must be expressible, and tied to its instrument.

    Before this block existed the manifest schema closed additionalProperties at
    the root, so an OCR-specific null could not be written down at all -- which
    meant the OCR detection gate could never be satisfied by any manifest.
    """

    def test_clean_text_channel_is_the_implicit_default(self) -> None:
        manifest = _manifest()
        assert "input_channel" not in manifest
        assert _messages(manifest) == []

    def test_ocr_channel_validates(self) -> None:
        manifest = _manifest()
        manifest["input_channel"] = _ocr_channel()
        manifest["scoring"]["minimum_scored_tokens"] = 50  # type: ignore[index]
        assert _messages(manifest) == []
        assert_calibration_manifest_semantics(manifest)

    def test_ocr_channel_requires_its_profile_block(self) -> None:
        manifest = _manifest()
        manifest["input_channel"] = {"channel": "ocr"}
        assert _messages(manifest) != []

    def test_unknown_channel_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["input_channel"] = {"channel": "telepathy"}
        assert any("is not one of" in message for message in _messages(manifest))

    def test_profile_fingerprint_must_be_a_sha256(self) -> None:
        manifest = _manifest()
        manifest["input_channel"] = _ocr_channel(profile_fingerprint="short")
        assert _messages(manifest) != []

    def test_short_minimum_scored_tokens_is_refused_for_ocr(self) -> None:
        """Recognition error re-segments tokens, so short OCR readings carry less."""

        manifest = _manifest()
        manifest["input_channel"] = _ocr_channel()
        manifest["scoring"]["minimum_scored_tokens"] = 10  # type: ignore[index]
        with pytest.raises(CalibrationManifestSemanticError) as excinfo:
            assert_calibration_manifest_semantics(manifest)
        assert "re-segments tokens" in str(excinfo.value)

    def test_named_degradation_at_severity_zero_is_refused_when_frozen(self) -> None:
        """Severity zero is the identity, so it describes a clean render."""

        manifest = _manifest()
        manifest["input_channel"] = _ocr_channel(
            degradation={"name": "blur", "severity": 0.0, "seed": 0}
        )
        manifest["scoring"]["minimum_scored_tokens"] = 50  # type: ignore[index]
        with pytest.raises(CalibrationManifestSemanticError) as excinfo:
            assert_calibration_manifest_semantics(manifest)
        assert "identity" in str(excinfo.value)

    def test_clean_at_severity_zero_is_accepted(self) -> None:
        manifest = _manifest()
        manifest["input_channel"] = _ocr_channel(
            degradation={"name": "clean", "severity": 0.0}
        )
        manifest["scoring"]["minimum_scored_tokens"] = 50  # type: ignore[index]
        assert_calibration_manifest_semantics(manifest)


class TestOcrChannelMatching:
    def test_matches_the_same_instrument(self) -> None:
        from watermark_lab.statistics.manifest import ocr_channel_matches

        manifest = _manifest()
        manifest["input_channel"] = _ocr_channel()
        matched, reason = ocr_channel_matches(
            manifest, engine_id="rapidocr", profile_fingerprint=SHA256, language="en"
        )
        assert matched is True
        assert "matches" in reason

    def test_rejects_a_clean_text_calibration(self) -> None:
        from watermark_lab.statistics.manifest import ocr_channel_matches

        manifest = _manifest()
        matched, reason = ocr_channel_matches(
            manifest, engine_id="rapidocr", profile_fingerprint=SHA256, language="en"
        )
        assert matched is False
        assert "not an OCR calibration" in reason

    @pytest.mark.parametrize(
        ("engine", "fingerprint", "language"),
        [
            ("tesseract", SHA256, "en"),
            ("rapidocr", "b" * 64, "en"),
            ("rapidocr", SHA256, "zh-Hant-TW"),
        ],
    )
    def test_rejects_a_different_instrument(
        self, engine: str, fingerprint: str, language: str
    ) -> None:
        from watermark_lab.statistics.manifest import ocr_channel_matches

        manifest = _manifest()
        manifest["input_channel"] = _ocr_channel()
        matched, _reason = ocr_channel_matches(
            manifest,
            engine_id=engine,
            profile_fingerprint=fingerprint,
            language=language,
        )
        assert matched is False
