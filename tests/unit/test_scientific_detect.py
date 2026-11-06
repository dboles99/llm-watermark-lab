from __future__ import annotations

import importlib.util

import pytest

from watermark_lab.adapters.kgw_author import AuthorKgwRuntimeError
from watermark_lab.run_engine.scientific_detect import ScientificKgwDetectionService
from watermark_lab.tokenization import (
    NormalizationPolicy,
    SpecialTokenPolicy,
    TokenizerConfigurationError,
    synthetic_numeric_tokenizer,
)


def _tokenizer(*, locale: str = "en", special_tokens: SpecialTokenPolicy | None = None):
    return synthetic_numeric_tokenizer(
        vocab_size=32,
        normalization=NormalizationPolicy(locale=locale),
        special_tokens=special_tokens,
    )


def test_service_requires_a_confirmatory_locale() -> None:
    with pytest.raises(TokenizerConfigurationError, match="exactly"):
        ScientificKgwDetectionService(_tokenizer(locale="und"))


def test_service_requires_author_special_token_boundary() -> None:
    with pytest.raises(TokenizerConfigurationError, match="add_special_tokens=False"):
        ScientificKgwDetectionService(
            _tokenizer(special_tokens=SpecialTokenPolicy(add_bos_token=True))
        )


def test_short_input_is_inconclusive_without_loading_torch() -> None:
    result = ScientificKgwDetectionService(_tokenizer()).detect("7")

    assert result.decision == "inconclusive"
    assert result.score is None
    assert result.candidate_threshold_prediction is None
    assert result.calibration_id is None
    assert result.estimated_fpr is None
    assert result.tokenization.token_ids == (7,)
    assert any("too few" in warning for warning in result.warnings)
    assert any("synthetic numeric" in warning for warning in result.warnings)


def test_missing_or_wrong_torch_runtime_fails_closed() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("default environment unexpectedly provides torch; pinned-runtime test covers it")

    with pytest.raises(AuthorKgwRuntimeError, match="no RNG fallback"):
        ScientificKgwDetectionService(_tokenizer()).detect("7 8")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="optional pinned KGW author PyTorch runtime is not installed",
)
def test_pinned_runtime_wires_tokenizer_provenance_and_author_score() -> None:
    result = ScientificKgwDetectionService(_tokenizer(locale="zh-Hant-TW")).detect(
        "5 6 5 6 7",
        ignore_repeated_ngrams=True,
    )

    assert result.decision == "inconclusive"
    assert result.scientific_status == "mechanics-verified-calibration-required"
    assert result.tokenization.provenance.tokenizer_id == "synthetic-numeric-v1"
    assert len(result.tokenization.provenance.tokenizer_fingerprint_sha256) == 64
    assert result.score is not None
    assert result.score.green_count == 2
    assert result.score.scored_token_count == 3
    assert result.score.scored_positions == (1, 2, 4)
    assert result.score.ignored_repeated_ngram_positions == (3,)
    assert result.score.runtime.torch_version == "2.13.0+cpu"
    assert result.calibration_id is None
    assert result.estimated_fpr is None
