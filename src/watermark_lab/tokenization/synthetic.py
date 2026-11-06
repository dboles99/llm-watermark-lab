"""Explicit synthetic numeric tokenizer for generated mock fixtures only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from watermark_lab.tokenization.boundary import FaithfulTokenizer
from watermark_lab.tokenization.contracts import (
    AssetFileFingerprint,
    NormalizationPolicy,
    SpecialTokenPolicy,
    TokenizerAssetManifest,
    TokenizerEncodingError,
    TokenizerSpec,
)

SYNTHETIC_NUMERIC_TOKENIZER_ID = "synthetic-numeric-v1"
SYNTHETIC_NUMERIC_REVISION = "fixture-v1"


def _manifest(vocab_size: int) -> TokenizerAssetManifest:
    descriptor = json.dumps(
        {
            "algorithm": "ascii-decimal-whitespace-token-ids",
            "strict_range": True,
            "vocab_size": vocab_size,
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(descriptor).hexdigest()
    return TokenizerAssetManifest(
        source="embedded-synthetic-test-fixture",
        files=(
            AssetFileFingerprint(
                name="embedded://synthetic-numeric-v1.json",
                sha256=digest,
                size_bytes=len(descriptor),
            ),
        ),
        aggregate_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class SyntheticNumericBackend:
    """Strict decimal-token parser; never use it as a language-model tokenizer."""

    vocab_size: int = 50257
    backend_name: str = "synthetic-numeric-test-fixture"
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    runtime_versions: tuple[tuple[str, str], ...] = (("synthetic-numeric", "fixture-v1"),)
    asset_manifest: TokenizerAssetManifest = field(init=False)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        object.__setattr__(self, "asset_manifest", _manifest(self.vocab_size))

    def encode(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        if add_special_tokens:
            raise TokenizerEncodingError(
                "The synthetic numeric tokenizer has no model-defined special tokens."
            )
        token_ids: list[int] = []
        for word in text.split():
            if not word.isascii() or not word.isdecimal():
                raise TokenizerEncodingError(
                    "Synthetic numeric fixtures accept ASCII decimal token IDs only."
                )
            token_id = int(word)
            if not 0 <= token_id < self.vocab_size:
                raise TokenizerEncodingError(
                    f"Synthetic token ID {token_id} is outside vocabulary range."
                )
            token_ids.append(token_id)
        return tuple(token_ids)


def synthetic_numeric_tokenizer(
    *,
    vocab_size: int = 50257,
    normalization: NormalizationPolicy | None = None,
    special_tokens: SpecialTokenPolicy | None = None,
) -> FaithfulTokenizer:
    """Return the explicitly labeled tokenizer used by numeric mock output.

    Optional policies let boundary tests exercise the same explicit locale and
    special-token contracts as a real local tokenizer. The backend remains
    labeled synthetic regardless of those policies.
    """

    return FaithfulTokenizer(
        TokenizerSpec(
            tokenizer_id=SYNTHETIC_NUMERIC_TOKENIZER_ID,
            revision=SYNTHETIC_NUMERIC_REVISION,
            normalization=normalization or NormalizationPolicy(),
            special_tokens=special_tokens or SpecialTokenPolicy(),
        ),
        SyntheticNumericBackend(vocab_size=vocab_size),
    )
