"""Iterative stochastic tinkering workflow for steganographic watermark discovery.

The workflow mutates cover text, applies a chosen steganographic channel,
perturbs the stego text with a set of attacks, and scores each variant by
three metrics:

    - recoverability: can the decoder still extract the payload?
    - integrity: does the HMAC tag still verify?
    - stealth: how close is the perturbed text to the original cover?

The best-scoring variants are kept for comparison.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable
from typing import Any

from watermark_lab.stego.codec import StegoDecoder, StegoEncoder


@dataclasses.dataclass(frozen=True)
class TinkerVariant:
    """One candidate produced by the tinkerer."""

    generation: int
    index: int
    cover: str
    stego: str
    channel: str
    recoverable: bool
    verified: bool
    stealth_score: float
    robustness_score: float
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class StegoTinkerer:
    """Stochastic search over steganographic watermark variants."""

    def __init__(
        self,
        channel_id: str = "whitespace",
        secret: bytes = b"",
        population_size: int = 8,
        generations: int = 3,
        mutation_rate: float = 0.2,
        **kwargs: Any,
    ) -> None:
        self._encoder = StegoEncoder(channel_id=channel_id, secret=secret, **kwargs)
        self._decoder = StegoDecoder(channel_id=channel_id, secret=secret, **kwargs)
        self._channel_id = channel_id
        self._population_size = max(1, population_size)
        self._generations = max(1, generations)
        self._mutation_rate = max(0.0, min(1.0, mutation_rate))
        self._kwargs = kwargs

    @staticmethod
    def _default_perturb(text: str, rng: random.Random, rate: float) -> str:
        """Default perturbation: duplicate, drop, or swap words with probability ``rate``."""
        words = text.split()
        if not words:
            return text
        out: list[str] = []
        for word in words:
            r = rng.random()
            if r < rate * 0.5:
                continue
            if r < rate:
                out.extend([word, word])
            else:
                out.append(word)
        if len(out) > 1 and rng.random() < rate:
            i, j = rng.randrange(len(out)), rng.randrange(len(out))
            out[i], out[j] = out[j], out[i]
        return " ".join(out)

    def run(
        self,
        cover: str,
        payload: str | bytes,
        perturb_fn: Callable[[str, random.Random, float], str] | None = None,
    ) -> dict[str, Any]:
        """Run the iterative tinker workflow and return a summary."""
        rng = random.Random(20260817)
        perturb = perturb_fn or self._default_perturb
        best_variants: list[TinkerVariant] = []
        all_variants: list[TinkerVariant] = []

        for generation in range(self._generations):
            current_population: list[str] = [cover]
            # Seed with previous best stego texts to continue evolution.
            if best_variants:
                current_population.extend(v.stego for v in best_variants[: self._population_size])
            current_population = current_population[: self._population_size]

            generation_variants: list[TinkerVariant] = []
            for index, candidate in enumerate(current_population):
                # Re-encode payload into the candidate cover.
                encoded = self._encoder.encode(candidate, payload)
                if encoded.status != "ok":
                    continue
                stego = encoded.text

                # Perturb the stego text.
                perturbed = perturb(stego, rng, self._mutation_rate)

                # Decode and verify.
                decoded = self._decoder.decode(perturbed)
                recoverable = decoded.status == "ok"
                verified = recoverable and bool(decoded.metadata.get("verified"))

                # Score stealth vs original cover.
                cover_words = cover.split()
                perturbed_words = perturbed.split()
                common = sum(1 for a, b in zip(cover_words, perturbed_words) if a == b)
                stealth = common / max(len(cover_words), len(perturbed_words), 1)

                # Channel-specific robustness.
                try:
                    robustness = self._decoder._channel.robustness_score(cover, perturbed)
                except Exception:
                    robustness = 0.0

                variant = TinkerVariant(
                    generation=generation,
                    index=index,
                    cover=candidate,
                    stego=perturbed,
                    channel=self._channel_id,
                    recoverable=recoverable,
                    verified=verified,
                    stealth_score=stealth,
                    robustness_score=robustness,
                    metadata={
                        "bits_embedded": encoded.bits_embedded,
                        "decoded_warnings": decoded.warnings,
                    },
                )
                generation_variants.append(variant)
                all_variants.append(variant)

            # Keep the top variants by combined score.
            def _score(v: TinkerVariant) -> float:
                return (
                    (1.0 if v.recoverable else 0.0) * 0.45
                    + (1.0 if v.verified else 0.0) * 0.30
                    + v.stealth_score * 0.15
                    + v.robustness_score * 0.10
                )

            generation_variants.sort(key=_score, reverse=True)
            best_variants = generation_variants[: max(1, self._population_size // 2)]

        # Overall best variant across all generations.
        if all_variants:
            overall_best = max(
                all_variants,
                key=lambda v: (
                    (1.0 if v.recoverable else 0.0) * 0.45
                    + (1.0 if v.verified else 0.0) * 0.30
                    + v.stealth_score * 0.15
                    + v.robustness_score * 0.10
                ),
            )
        else:
            overall_best = None

        return {
            "channel": self._channel_id,
            "generations": self._generations,
            "population_size": self._population_size,
            "mutation_rate": self._mutation_rate,
            "total_variants": len(all_variants),
            "overall_best": dataclasses.asdict(overall_best) if overall_best else None,
            "final_best": [dataclasses.asdict(v) for v in best_variants],
            "all_variants": [dataclasses.asdict(v) for v in all_variants],
        }
