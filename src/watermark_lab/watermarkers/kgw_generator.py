"""KGW watermarked text generator (reference implementation).

This module implements the generation side of the Kirchenbauer-Geiping-Wen
(KGW) watermark. For the vertical slice it works with the local mock provider
and the minimal tokenizer, producing synthetic output whose token ids round-trip
through the run-engine tokenizer so self-verification can succeed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from watermark_lab.adapters.kgm import DEFAULT_HASH_KEY
from watermark_lab.run_engine.intake import compute_sha256
from watermark_lab.services.model_gateway import ModelProfile
from watermark_lab.testing.mock_provider import MockProvider

#: Default KGW generation parameters for the vertical slice.
DEFAULT_GAMMA = 0.25
DEFAULT_DELTA = 2.0
DEFAULT_CONTEXT_WIDTH = 1
DEFAULT_SEEDING_SCHEME = "simple_1"

#: Minimum number of new tokens for a conclusive self-verification score.
_MIN_TOKENS_FOR_SELF_VERIFY = 30

#: Test-only key vault. Real keys must never be stored in source, prompts, logs,
#: notebooks, or Git. Persisted artifacts must reference keys only by ``key_id``.
_TEST_KEY_VAULT: dict[str, bytes] = {
    "test-kgw-v1": b"TEST_KEY_KGW_V1",
}


def _resolve_key(key_id: str) -> bytes:
    if key_id not in _TEST_KEY_VAULT:
        raise ValueError(f"Unknown key_id: {key_id!r}")
    return _TEST_KEY_VAULT[key_id]


def _default_vocab_size() -> int:
    return 50257


def _synthetic_decode(tokens: list[int]) -> str:
    """Decode token ids to synthetic text that round-trips through the minimal tokenizer.

    The run-engine tokenizer splits on whitespace and parses numeric words as
    their integer token id. Emitting space-separated ids therefore lets the keyed
    detector recover the original token sequence and detect the watermark.
    """
    return " ".join(str(token_id) for token_id in tokens)


@dataclass(frozen=True)
class KgwGenerationConfig:
    """Validated, effective parameters shared by generation and verification."""

    profile_id: str = "mock"
    key_id: str = "test-kgw-v1"
    tokenizer_id: str = "gpt2"
    max_new_tokens: int = 80
    gamma: float = DEFAULT_GAMMA
    delta: float = DEFAULT_DELTA
    context_width: int = DEFAULT_CONTEXT_WIDTH
    seeding_scheme: str = DEFAULT_SEEDING_SCHEME
    hash_key: int = DEFAULT_HASH_KEY
    vocab_size: int = 50257
    detection_threshold: float = 4.0

    @classmethod
    def from_mapping(
        cls, config: dict[str, Any], *, vocab_size: int | None = None
    ) -> KgwGenerationConfig:
        effective_vocab_size = int(config.get("vocab_size", vocab_size or _default_vocab_size()))
        if vocab_size is not None and effective_vocab_size != vocab_size:
            raise ValueError(
                f"vocab_size {effective_vocab_size} does not match generator vocabulary {vocab_size}"
            )
        effective = cls(
            profile_id=str(config.get("profile_id", "mock")),
            key_id=str(config.get("key_id", "test-kgw-v1")),
            tokenizer_id=str(config.get("tokenizer_id", "gpt2")),
            max_new_tokens=int(config.get("max_new_tokens", 80)),
            gamma=float(config.get("gamma", DEFAULT_GAMMA)),
            delta=float(config.get("delta", DEFAULT_DELTA)),
            context_width=int(config.get("context_width", DEFAULT_CONTEXT_WIDTH)),
            seeding_scheme=str(config.get("seeding_scheme", DEFAULT_SEEDING_SCHEME)),
            hash_key=int(config.get("hash_key", DEFAULT_HASH_KEY)),
            vocab_size=effective_vocab_size,
            detection_threshold=float(config.get("detection_threshold", 4.0)),
        )
        effective.validate()
        return effective

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if not math.isfinite(self.delta):
            raise ValueError("delta must be finite")
        if self.context_width < 1:
            raise ValueError("context_width must be at least 1")
        if self.seeding_scheme != DEFAULT_SEEDING_SCHEME:
            raise ValueError(f"Unsupported seeding_scheme: {self.seeding_scheme!r}")
        if self.seeding_scheme == "simple_1" and self.context_width != 1:
            raise ValueError("simple_1 requires context_width=1")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.detection_threshold <= 0 or not math.isfinite(self.detection_threshold):
            raise ValueError("detection_threshold must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def detection_options(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "context_width": self.context_width,
            "seeding_scheme": self.seeding_scheme,
            "hash_key": self.hash_key,
        }


class KgwGenerator:
    """Generate watermarked text using the KGW green-list logit bias."""

    method_id: str = "kgw"

    def __init__(self, vocab_size: int | None = None) -> None:
        self.vocab_size = vocab_size or _default_vocab_size()
        self._mock = MockProvider(vocab_size=self.vocab_size)

    def _load_profile(self, profile_id: str) -> ModelProfile:
        """Return a simulated model profile for ``profile_id``.

        Only the ``mock`` profile is supported in the vertical slice. Real
        provider endpoints are never called.
        """
        if profile_id != "mock":
            raise ValueError(f"Unsupported model profile: {profile_id!r}")
        return ModelProfile(
            profile_id="mock",
            provider="mock_local",
            model="kgw-test",
            simulated=True,
        )

    def _generate_tokens(
        self,
        key: bytes,
        config: KgwGenerationConfig,
        prompt: str,
    ) -> list[int]:
        """Produce a deterministic KGW-biased token sequence.

        The mock provider seeds its synthetic sampler from the prompt hash,
        so ``prompt`` reaches every generated token while ``gamma``, ``delta``,
        ``context_width``, and ``max_new_tokens`` control the KGW sampling
        process.
        """
        generated = self._mock.generate(
            profile="kgw-test",
            max_new_tokens=config.max_new_tokens,
            key=key,
            gamma=config.gamma,
            delta=config.delta,
            context_width=config.context_width,
            hash_key=config.hash_key,
            seeding_scheme=config.seeding_scheme,
            prompt=prompt,
        )
        return list(generated["tokens"])

    def generate(self, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
        """Generate KGW-watermarked text for ``prompt``.

        Args:
            prompt: Input prompt text.
            config: Generation options. Recognized keys:
                * ``profile_id`` (str): model profile to load; only ``"mock"``.
                * ``key_id`` (str): key identifier; only ``"test-kgw-v1"``.
                * ``max_new_tokens`` (int): length of generated token sequence.
                * ``gamma`` (float): green-list fraction.
                * ``delta`` (float): logit bias added to green-list tokens.
                * ``context_width`` (int): previous tokens used for seeding.

        Returns:
            Generation record with ``text``, ``tokens``, hashes, profile info,
            method/key identifiers, and the generation config.
        """
        effective = KgwGenerationConfig.from_mapping(config, vocab_size=self.vocab_size)

        profile = self._load_profile(effective.profile_id)
        key = _resolve_key(effective.key_id)

        tokens = self._generate_tokens(
            key=key,
            config=effective,
            prompt=prompt,
        )
        text = _synthetic_decode(tokens)

        prompt_sha256 = compute_sha256(prompt)
        response_sha256 = compute_sha256(text)

        return {
            "text": text,
            "tokens": tokens,
            "prompt_sha256": prompt_sha256,
            "response_sha256": response_sha256,
            "model_profile": {
                "profile_id": profile.profile_id,
                "provider": profile.provider,
                "model": profile.model,
                "simulated": profile.simulated,
            },
            "method_id": self.method_id,
            "key_id": effective.key_id,
            "config": effective.to_dict(),
        }


__all__ = ["KgwGenerationConfig", "KgwGenerator"]
