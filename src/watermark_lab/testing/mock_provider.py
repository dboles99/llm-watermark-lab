"""Minimal mock watermark provider for integration tests.

This provider returns deterministic token sequences so the KGW adapter can be
exercised without a real language model. It intentionally covers only the
profiles required by the current integration tests.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from watermark_lab.adapters.kgw import (
    DEFAULT_HASH_KEY,
    kgw_green_list,
)


class WatermarkProviderError(Exception):
    """Base exception for watermark provider failures."""


class ProviderTimeoutError(WatermarkProviderError):
    """Raised when a mock provider request times out."""


class MockProvider:
    """Return deterministic token sequences for watermark regression tests."""

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = vocab_size

    def generate(
        self,
        profile: str = "clean",
        max_new_tokens: int = 40,
        *,
        key: bytes = b"TEST_KEY_KGW_V1",
        gamma: float = 0.25,
        delta: float = 2.0,
        context_width: int = 1,
        hash_key: int = DEFAULT_HASH_KEY,
        seeding_scheme: str = "simple_1",
        prompt: str = "",
    ) -> dict[str, Any]:
        """Generate a deterministic token sequence for ``profile``.

        Args:
            profile: One of ``"clean"``, ``"kgw-test"``, ``"timeout"``,
                ``"malformed"``.
            max_new_tokens: Number of new tokens to produce.
            key: Watermark key used by the synthetic KGW profile.
            gamma: Fraction of the vocabulary assigned to the green list.
            delta: Logit bias represented by the synthetic sampler.
            context_width: Number of preceding tokens supplied to the PRF.
            hash_key: Base PRF salt.
            seeding_scheme: KGW seeding scheme.
            prompt: Optional prompt text. When non-empty, the mock provider
                seeds the synthetic sampler from a stable hash of the prompt
                so that different prompts produce different token sequences.

        Returns:
            Dict with keys ``"tokens"`` and ``"profile"``.

        Raises:
            ProviderTimeoutError: For ``profile="timeout"``.
            WatermarkProviderError: For ``profile="malformed"``.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if not math.isfinite(delta):
            raise ValueError("delta must be finite")
        if context_width < 1:
            raise ValueError("context_width must be at least 1")

        if profile == "timeout":
            raise ProviderTimeoutError("Mock provider timed out")
        if profile == "malformed":
            raise WatermarkProviderError("Mock provider returned malformed output")

        if profile == "clean":
            tokens = self._clean_sequence(max_new_tokens)
        elif profile == "kgw-test":
            tokens = self._kgw_test_sequence(
                max_new_tokens,
                key=key,
                gamma=gamma,
                delta=delta,
                context_width=context_width,
                hash_key=hash_key,
                seeding_scheme=seeding_scheme,
                prompt=prompt,
            )
        else:
            raise WatermarkProviderError(f"Unknown mock provider profile: {profile!r}")

        return {"tokens": tokens, "profile": profile}

    def _clean_sequence(self, length: int) -> list[int]:
        """Unwatermarked deterministic tokens starting at id 100."""
        return [100 + (i % 1000) for i in range(length)]

    def _kgw_test_sequence(
        self,
        length: int,
        *,
        key: bytes,
        gamma: float,
        delta: float,
        context_width: int,
        hash_key: int,
        seeding_scheme: str,
        prompt: str = "",
    ) -> list[int]:
        """Sample deterministic tokens from a KGW-biased synthetic distribution.

        All unmodified mock logits are equal. Applying ``delta`` therefore changes
        the probability mass assigned to the green partition from ``gamma`` to
        ``gamma * exp(delta) / (gamma * exp(delta) + 1 - gamma)``. A fixed RNG
        stream makes the output reproducible while ensuring that non-default
        ``gamma`` and ``delta`` values genuinely affect which partition is sampled.

        When ``prompt`` is non-empty, the RNG is also seeded from a stable hash of
        the prompt so that different prompts reach every generated token.
        """
        # Start from the same token as the golden fixture, then include the
        # largest vocabulary token as a sentinel to guarantee the detector
        # infers the full vocabulary size.
        tokens: list[int] = [15496, self.vocab_size - 1]
        if prompt:
            rng = random.Random(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        else:
            rng = random.Random("watermark-lab-lab:kgw-test:v1")

        # Avoid overflow while retaining the limiting probabilities.
        if delta >= 0:
            exp_neg_delta = math.exp(-min(delta, 745.0))
            green_probability = gamma / (gamma + (1.0 - gamma) * exp_neg_delta)
        else:
            exp_delta = math.exp(max(delta, -745.0))
            green_probability = gamma * exp_delta / (gamma * exp_delta + 1.0 - gamma)

        while len(tokens) < length:
            context = tuple(tokens[-context_width:])
            green_list = kgw_green_list(
                key=key,
                context=context,
                vocab_size=self.vocab_size,
                gamma=gamma,
                hash_key=hash_key,
                seeding_scheme=seeding_scheme,
            )
            choose_green = rng.random() < green_probability
            if choose_green:
                next_token = green_list[rng.randrange(len(green_list))]
            else:
                green_set = set(green_list)
                start = rng.randrange(self.vocab_size)
                next_token = next(
                    candidate
                    for offset in range(self.vocab_size)
                    if (candidate := (start + offset) % self.vocab_size) not in green_set
                )
            tokens.append(next_token)

        return tokens[:length]
