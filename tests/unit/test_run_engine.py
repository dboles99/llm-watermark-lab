"""Unit tests for the run engine: intake, normalization, tokenizer metadata, and detection."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from watermark_lab.core.contracts import DetectionConfig
from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.run_engine.intake import compute_sha256, normalize_text, read_source
from watermark_lab.run_engine.tokenizer_meta import tokenizer_metadata


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world", encoding="utf-8")
    return file_path


def test_read_source_from_file(tmp_text_file: Path) -> None:
    text, media_type, source_path = read_source(str(tmp_text_file))
    assert text == "Hello world"
    assert media_type == "text/plain"
    assert source_path == str(tmp_text_file.resolve())


def test_read_source_from_raw_text() -> None:
    text, media_type, source_path = read_source("Raw pasted text")
    assert text == "Raw pasted text"
    assert media_type == "text/plain"
    assert source_path is None


def test_compute_sha256_is_stable() -> None:
    assert compute_sha256("abc") == compute_sha256("abc")
    assert compute_sha256("abc") != compute_sha256("abcd")


def test_normalize_text_applies_nfc() -> None:
    # LATIN SMALL LETTER E + COMBINING ACUTE ACCENT -> LATIN SMALL LETTER E WITH ACUTE
    composed = "\u00e9"
    decomposed = "e\u0301"
    assert normalize_text(decomposed) == composed
    assert normalize_text(composed) == composed
    assert unicodedata.is_normalized("NFC", normalize_text(decomposed))


def test_tokenizer_metadata_gpt2() -> None:
    meta = tokenizer_metadata("gpt2")
    assert meta["family"] == "gpt2"
    assert meta["vocab_size"] == 50257
    assert meta["revision"] == "local"


def test_tokenizer_metadata_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tokenizer family"):
        tokenizer_metadata("unknown")


def _config(threshold: float = 4.0) -> DetectionConfig:
    return DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=threshold,
        options={"gamma": 0.25, "context_width": 1},
    )


def test_detection_service_short_text_is_inconclusive() -> None:
    service = DetectionService()
    result = service.detect("hello", _config())
    assert result.method_id == "kgw"
    assert result.compatibility == "experimental"
    assert result.decision == "inconclusive"
    assert result.score is None
    assert result.p_value is None
    assert result.token_count == 0
    assert result.estimated_fpr is None
    assert any("inconclusive" in w for w in result.warnings)


def test_detection_service_longer_text_returns_decision() -> None:
    service = DetectionService()
    text = " ".join([f"word{i}" for i in range(80)])
    result = service.detect(text, _config())
    assert result.method_id == "kgw"
    assert result.compatibility == "experimental"
    assert result.decision in {"verified compatible watermark", "not detected", "inconclusive"}
    assert result.token_count == 79
    assert result.estimated_fpr is None
    assert any("blocked-pending-corpus" in w for w in result.warnings)


def test_detection_service_unknown_key_raises() -> None:
    service = DetectionService()
    config = DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="no-such-key",
        threshold=4.0,
        options={"gamma": 0.25, "context_width": 1},
    )
    with pytest.raises(ValueError, match="Unknown key_id"):
        service.detect("some text here", config)


def test_detection_service_unsupported_method_raises() -> None:
    service = DetectionService()
    config = DetectionConfig(
        method_id="mcmark",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=4.0,
        options={"gamma": 0.25, "context_width": 1},
    )
    with pytest.raises(ValueError, match="Unsupported method_id"):
        service.detect("some text here", config)


def test_detection_service_invalid_threshold_raises() -> None:
    service = DetectionService()
    config = DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=-1.0,
        options={"gamma": 0.25, "context_width": 1},
    )
    with pytest.raises(ValueError, match="threshold must be positive"):
        service.detect("some text here", config)


def test_detection_service_applies_repeated_ngram_masking() -> None:
    service = DetectionService()
    config = DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=4.0,
        options={
            "gamma": 0.25,
            "context_width": 1,
            "ignore_repeated_ngrams": True,
        },
    )

    result = service.detect("5 6 5 6 7", config)

    assert result.token_count == 3
    assert result.compatibility == "prototype"
    assert result.decision == "inconclusive"
    assert result.original_scorable_token_count == 4
    assert result.ignored_repeated_ngrams == 1
    assert result.scoring_variant == "unique-ngram-uncalibrated"
    assert result.calibration_id is None
    assert any("no matching frozen calibration" in warning for warning in result.warnings)


def test_detection_service_rejects_non_boolean_repeated_ngram_option() -> None:
    service = DetectionService()
    config = DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=4.0,
        options={"ignore_repeated_ngrams": "true"},
    )

    with pytest.raises(TypeError, match="must be a boolean"):
        service.detect("5 6 5 6 7", config)
