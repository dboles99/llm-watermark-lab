"""Detection service that wires intake, normalization, tokenization, and KGW."""

from __future__ import annotations

import hashlib
from typing import Any

from watermark_lab.adapters.kgm import kgw_detect
from watermark_lab.core.contracts import DetectionConfig, DetectionResult
from watermark_lab.run_engine.tokenizer_meta import tokenizer_metadata

#: Test-only key vault. Real keys must never be stored in source, prompts, logs,
#: notebooks, or Git. Persisted artifacts must reference keys only by ``key_id``.
TEST_KEY_VAULT: dict[str, bytes] = {
    "test-kgw-v1": b"TEST_KEY_KGW_V1",
}

#: Minimum scored-token count for a conclusive detection decision. Below this the
#: result is reported as ``inconclusive`` rather than forced positive/negative.
MIN_CONCLUSIVE_TOKENS = 30


def _minimal_tokenize(text: str, vocab_size: int) -> list[int]:
    """Placeholder tokenizer: split on whitespace and map words to token ids.

    This is intentionally simple and deterministic so the vertical slice can run
    without downloading real model checkpoints. Numeric words are parsed as
    their integer token id so synthetic provider output can round-trip through
    generation and detection; non-numeric words fall back to a SHA-256 hash.
    It must be replaced with the configured tokenizer family (e.g. gpt2) once
    tokenization assets are available.
    """
    words = text.split()
    tokens: list[int] = []
    for word in words:
        try:
            token_id = int(word) % vocab_size
        except ValueError:
            token_id = hashlib.sha256(word.encode("utf-8")).digest()[0] % vocab_size
        tokens.append(int(token_id))
    return tokens


def _resolve_key(config: DetectionConfig) -> bytes:
    if config.key_id is None:
        raise ValueError("key_id is required for KGW detection")
    if config.key_id not in TEST_KEY_VAULT:
        raise ValueError(f"Unknown key_id: {config.key_id!r}")
    return TEST_KEY_VAULT[config.key_id]


def _run_kgw(text: str, config: DetectionConfig) -> dict[str, Any]:
    key = _resolve_key(config)
    meta = tokenizer_metadata(config.tokenizer_id)
    vocab_size_value = meta["vocab_size"]
    if isinstance(vocab_size_value, bool) or not isinstance(vocab_size_value, int):
        raise TypeError("tokenizer metadata vocab_size must be an integer")
    vocab_size = vocab_size_value
    tokens = _minimal_tokenize(text, vocab_size)
    options = config.options
    ignore_repeated_ngrams = options.get("ignore_repeated_ngrams", False)
    if not isinstance(ignore_repeated_ngrams, bool):
        raise TypeError("ignore_repeated_ngrams must be a boolean")
    return kgw_detect(
        tokens=tokens,
        key=key,
        gamma=float(options.get("gamma", 0.25)),
        context_width=int(options.get("context_width", 1)),
        hash_key=int(options.get("hash_key", 15485863)),
        seeding_scheme=str(options.get("seeding_scheme", "simple_1")),
        ignore_repeated_ngrams=ignore_repeated_ngrams,
        vocab_size=vocab_size,
    )


def _make_decision(raw: dict[str, Any], threshold: float) -> tuple[str, list[str]]:
    warnings: list[str] = []
    warning = raw.get("warning")
    z = raw.get("z")

    if raw.get("ignore_repeated_ngrams") is True:
        warnings.append(
            "repeated-ngram-masked score has no matching frozen calibration; "
            "reported as inconclusive"
        )
        return "inconclusive", warnings

    if warning == "too_few_tokens":
        warnings.append("token_count below scored-token minimum; reported as inconclusive")
        return "inconclusive", warnings
    if z is None:
        warnings.append("could not compute z-score; reported as inconclusive")
        return "inconclusive", warnings
    if raw.get("T", 0) < MIN_CONCLUSIVE_TOKENS:
        warnings.append("token_count below 30; reported as inconclusive")
        return "inconclusive", warnings
    if z >= threshold:
        return "verified compatible watermark", warnings
    return "not detected", warnings


class DetectionService:
    """Run-engine service for detection workflows."""

    def detect(self, text: str, config: DetectionConfig) -> DetectionResult:
        """Run the configured detector and return an explicitly uncalibrated result."""
        if config.threshold <= 0:
            raise ValueError("threshold must be positive")

        if config.method_id != "kgw":
            raise ValueError(f"Unsupported method_id: {config.method_id!r}")

        raw = _run_kgw(text, config)
        decision, warnings = _make_decision(raw, config.threshold)

        # Calibration corpus is not built yet (blocked-pending-corpus).
        warnings.append("estimated_fpr is null; calibration corpus not built (blocked-pending-corpus)")

        return DetectionResult(
            method_id="kgw",
            compatibility=(
                "prototype" if raw.get("ignore_repeated_ngrams") is True else "experimental"
            ),
            decision=decision,
            score=raw.get("z"),
            p_value=raw.get("p_value_one_sided"),
            token_count=raw.get("T", 0),
            estimated_fpr=None,
            warnings=tuple(warnings),
            original_scorable_token_count=raw.get("original_scorable_token_count"),
            ignored_repeated_ngrams=int(raw.get("ignored_repeated_ngrams", 0)),
            scoring_variant=(
                "unique-ngram-uncalibrated"
                if raw.get("ignore_repeated_ngrams") is True
                else "naive-binomial-uncalibrated"
            ),
            calibration_id=None,
        )
