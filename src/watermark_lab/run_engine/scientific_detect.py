"""Fail-closed bridge from pinned tokenization to the KGW author oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from watermark_lab.adapters.kgw_author import (
    AUTHOR_KGW_CPU_PROFILE,
    AuthorKgwProfile,
    AuthorKgwScore,
    kgw_author_detect,
)
from watermark_lab.tokenization import FaithfulTokenizer, TokenizationResult
from watermark_lab.tokenization.contracts import TokenizerConfigurationError


@dataclass(frozen=True, slots=True)
class ScientificKgwDetection:
    """Mechanics result that deliberately abstains without frozen calibration."""

    method_id: Literal["kgw"]
    scientific_status: Literal["mechanics-verified-calibration-required"]
    decision: Literal["inconclusive"]
    author_profile_id: str
    tokenization: TokenizationResult
    score: AuthorKgwScore | None
    candidate_threshold_prediction: bool | None
    calibration_id: None
    estimated_fpr: None
    warnings: tuple[str, ...]


class ScientificKgwDetectionService:
    """Join exact local tokenizer bytes to one exact author RNG profile.

    This service never converts an uncalibrated z threshold into a scientific
    detection decision.  It exposes the threshold comparison only as a
    candidate diagnostic and always returns ``inconclusive`` until a matching
    frozen calibration-manifest implementation is added.
    """

    def __init__(
        self,
        tokenizer: FaithfulTokenizer,
        *,
        profile: AuthorKgwProfile = AUTHOR_KGW_CPU_PROFILE,
    ) -> None:
        locale = tokenizer.spec.normalization.locale
        if locale not in {"en", "zh-Hant-TW"}:
            raise TokenizerConfigurationError(
                "scientific KGW detection locale must be exactly 'en' or 'zh-Hant-TW'"
            )
        specials = tokenizer.spec.special_tokens
        if (
            specials.add_model_special_tokens
            or specials.add_bos_token
            or specials.add_eos_token
        ):
            raise TokenizerConfigurationError(
                "the KGW token-ID author profile requires add_special_tokens=False "
                "with no explicit BOS or EOS insertion"
            )
        self._tokenizer = tokenizer
        self._profile = profile

    def detect(
        self,
        text: str,
        *,
        gamma: float = 0.25,
        threshold: float = 4.0,
        ignore_repeated_ngrams: bool = True,
    ) -> ScientificKgwDetection:
        tokenization = self._tokenizer.encode(text)
        score: AuthorKgwScore | None
        warnings = [
            "no matching frozen calibration manifest; scientific decision is inconclusive",
            (
                "raw-text parity against author normalization remains pending; this run uses the "
                "recorded tokenizer normalization policy and the token-ID author oracle"
            ),
        ]
        if tokenization.provenance.backend == "synthetic-numeric-test-fixture":
            warnings.append(
                "synthetic numeric tokenizer is a mechanics fixture and is not scientific corpus evidence"
            )
        if tokenization.token_count <= self._profile.context_width:
            score = None
            warnings.append("too few token IDs to score after the author context prefix")
        else:
            score = kgw_author_detect(
                tokenization.token_ids,
                tokenization.provenance.vocab_size,
                gamma,
                threshold,
                ignore_repeated_ngrams=ignore_repeated_ngrams,
                profile=self._profile,
            )
        return ScientificKgwDetection(
            method_id="kgw",
            scientific_status="mechanics-verified-calibration-required",
            decision="inconclusive",
            author_profile_id=self._profile.profile_id,
            tokenization=tokenization,
            score=score,
            candidate_threshold_prediction=None if score is None else score.prediction,
            calibration_id=None,
            estimated_fpr=None,
            warnings=tuple(warnings),
        )


__all__ = ["ScientificKgwDetection", "ScientificKgwDetectionService"]
