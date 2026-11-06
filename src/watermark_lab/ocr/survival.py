"""Measure which provenance channels survive a print-scan-OCR round trip.

The premise this module exists to test:

    OCR annihilates every container-bound provenance channel while leaving the
    statistical channel partly intact.

Zero-width characters, Unicode variation selectors, whitespace encodings, file
metadata and C2PA manifests all live in bytes that a scanner never sees. What
reaches the OCR engine is glyphs on paper. A statistical, logit-biased watermark
lives in *which words were chosen*, and words survive as long as they are read
correctly — so it is the only family with any prospect of crossing the gap.

This also makes OCR an unusually clean attack. The project's evaluation protocol
holds that an attack destroying the text is not a successful attack, which is why
paraphrase must be quality-constrained. OCR is a faithful transcription attempt:
it preserves meaning by construction, so it passes semantic and quality
constraints trivially while still degrading the signal.

The context-corruption prediction
---------------------------------
For a green-list watermark with context width ``h``, the green list for each
token is derived from the preceding ``h`` tokens. One misread token therefore
corrupts its own green-list decision *and* the decisions of the following ``h``
tokens. Detection degradation under OCR should be super-linear in character error
rate and increasing in ``h``. ``context_corruption_span`` computes the affected
fraction so the prediction can be tested rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from watermark_lab.ocr.degrade import Degradation, apply_degradation
from watermark_lab.ocr.render import RenderSettings, render_text_to_image

# Channels whose payload lives in bytes a scanner never captures.
CONTAINER_BOUND_CHANNELS = ("unicode", "whitespace")

# Channels whose payload lives in word choice, which survives correct reading.
CONTENT_BOUND_CHANNELS = ("token_choice",)


def levenshtein(a: str, b: str) -> int:
    """Return the edit distance between two strings.

    Space-optimised dynamic programming: O(len(a) * len(b)) time, O(min) space.
    Adequate for page-sized text; a large corpus should batch this out to a
    compiled implementation.
    """

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return edit distance normalised by reference length.

    Can exceed 1.0 when the hypothesis inserts more than the reference contains,
    which is a real OCR failure mode (noise read as text) and is deliberately not
    clamped.
    """

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return word-level edit distance normalised by reference word count.

    Latin text is split on whitespace. CJK has no inter-word spaces, so for
    scripts without them this collapses toward the character error rate — which
    is itself the point: a CJK character is frequently a whole token, so one
    misread character is one token error, where a Latin misread is often a
    sub-token event the tokenizer may absorb or may cascade through.
    """

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    # Map each distinct word to a private-use codepoint so the character-level
    # DP can be reused directly on word sequences.
    vocabulary: dict[str, str] = {}

    def encode(words: list[str]) -> str:
        out: list[str] = []
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = chr(0xE000 + len(vocabulary))
            out.append(vocabulary[word])
        return "".join(out)

    encoded_reference = encode(reference_words)
    encoded_hypothesis = encode(hypothesis_words)
    return levenshtein(encoded_reference, encoded_hypothesis) / len(reference_words)


def context_corruption_span(
    *,
    token_count: int,
    corrupted_tokens: int,
    context_width: int,
) -> float:
    """Return the fraction of green-list decisions invalidated by misread tokens.

    Each corrupted token invalidates its own decision plus the ``context_width``
    decisions that follow it, giving an upper bound of
    ``corrupted * (1 + context_width) / token_count`` before overlap. The result
    is clamped to 1.0, since overlapping corruption cannot invalidate more than
    every decision.

    This is the quantity behind the prediction that ``h = 1`` (lefthash) is
    materially more OCR-robust than ``h = 4`` (selfhash).
    """

    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if corrupted_tokens < 0:
        raise ValueError("corrupted_tokens must be nonnegative.")
    if context_width < 0:
        raise ValueError("context_width must be nonnegative.")
    return min(1.0, corrupted_tokens * (1 + context_width) / token_count)


@dataclass(frozen=True, slots=True)
class ChannelSurvival:
    """Whether one channel's payload survived a round trip."""

    channel_id: str
    channel_class: str
    embedded: bool
    recovered: bool
    payload_match: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "channel_class": self.channel_class,
            "embedded": self.embedded,
            "recovered": self.recovered,
            "payload_match": self.payload_match,
            "detail": self.detail,
        }


DETECTION_DROPOUT_NOTE = (
    "Recovered text is substantially shorter than the reference, which indicates the "
    "detector failed to find whole regions rather than misreading them. Detection dropout "
    "and recognition error are different failures: counting dropout as signal loss "
    "fabricates watermark fragility that the degradation did not cause."
)

# Below this ratio of recovered to reference length, a cell is treated as a
# detection failure rather than a usable measurement.
DEFAULT_COVERAGE_FLOOR = 0.90


@dataclass(frozen=True, slots=True)
class RoundTripResult:
    """One cell of the OCR survival surface."""

    degradation: dict[str, Any]
    reference_text: str
    recovered_text: str
    character_error_rate: float
    word_error_rate: float
    mean_confidence: float
    min_confidence: float
    engine_id: str
    profile_fingerprint: str
    render_settings: dict[str, Any]
    channels: tuple[ChannelSurvival, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    @property
    def coverage_ratio(self) -> float:
        """Recovered length over reference length.

        Values well below 1.0 indicate the detector missed whole regions. This is
        empirically **not monotone in image quality**: a crisp render can lose
        lines that a mildly blurred one recovers, so coverage must be checked per
        cell rather than assumed to be worst at high severity.
        """

        if not self.reference_text:
            return 1.0
        return len(self.recovered_text) / len(self.reference_text)

    @property
    def detection_dropout(self) -> bool:
        """Whether this cell looks like a detector failure rather than a reading."""

        return self.coverage_ratio < DEFAULT_COVERAGE_FLOOR

    @property
    def usable(self) -> bool:
        """Whether this cell may be used as a survival measurement."""

        return not self.detection_dropout

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "degradation": self.degradation,
            "character_error_rate": self.character_error_rate,
            "word_error_rate": self.word_error_rate,
            "coverage_ratio": self.coverage_ratio,
            "detection_dropout": self.detection_dropout,
            "usable": self.usable,
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "engine_id": self.engine_id,
            "profile_fingerprint": self.profile_fingerprint,
            "render_settings": self.render_settings,
            "channels": [channel.as_dict() for channel in self.channels],
            "recovered_text": self.recovered_text,
            "warnings": list(self.warnings),
        }
        if self.detection_dropout:
            payload["detection_dropout_note"] = DETECTION_DROPOUT_NOTE
        return payload


def round_trip(
    text: str,
    *,
    engine: Any,
    settings: RenderSettings,
    degradation: Degradation,
) -> RoundTripResult:
    """Render, degrade, recognise, and score one text against one degradation."""

    image = render_text_to_image(text, settings)
    degraded = apply_degradation(image, degradation)
    result = engine.recognize(degraded)

    return RoundTripResult(
        degradation=degradation.as_dict(),
        reference_text=text,
        recovered_text=result.text,
        character_error_rate=character_error_rate(text, result.text),
        word_error_rate=word_error_rate(text, result.text),
        mean_confidence=result.mean_confidence,
        min_confidence=result.min_confidence,
        engine_id=result.engine_id,
        profile_fingerprint=result.profile_fingerprint,
        render_settings=settings.as_dict(),
        warnings=result.warnings,
    )


def measure_channel_survival(
    cover_text: str,
    payload: str,
    *,
    engine: Any,
    settings: RenderSettings,
    degradation: Degradation,
    secret: bytes = b"ocr-survival",
    channel_ids: tuple[str, ...] = CONTAINER_BOUND_CHANNELS + CONTENT_BOUND_CHANNELS,
) -> RoundTripResult:
    """Embed a payload in each channel, round-trip through OCR, and test recovery.

    Channels are embedded and measured independently so their fates can be
    compared on identical input. A channel that cannot embed in the given cover
    is recorded as ``embedded=False`` rather than counted as destroyed — failing
    to embed and failing to survive are different findings.
    """

    from watermark_lab.stego import StegoDecoder, StegoEncoder

    base = round_trip(cover_text, engine=engine, settings=settings, degradation=degradation)
    survivals: list[ChannelSurvival] = []

    for channel_id in channel_ids:
        channel_class = (
            "container-bound" if channel_id in CONTAINER_BOUND_CHANNELS else "content-bound"
        )
        try:
            encoder = StegoEncoder(channel_id=channel_id, secret=secret)
            encoded = encoder.encode(cover_text, payload)
        except Exception as exc:
            survivals.append(
                ChannelSurvival(
                    channel_id=channel_id,
                    channel_class=channel_class,
                    embedded=False,
                    recovered=False,
                    payload_match=False,
                    detail=f"encode raised: {exc}",
                )
            )
            continue

        if encoded.status != "ok":
            survivals.append(
                ChannelSurvival(
                    channel_id=channel_id,
                    channel_class=channel_class,
                    embedded=False,
                    recovered=False,
                    payload_match=False,
                    detail=f"encode status {encoded.status}: {list(encoded.warnings)}",
                )
            )
            continue

        trip = round_trip(
            encoded.text, engine=engine, settings=settings, degradation=degradation
        )
        try:
            decoder = StegoDecoder(channel_id=channel_id, secret=secret)
            decoded = decoder.decode(trip.recovered_text)
            recovered = decoded.status == "ok"
            match = recovered and decoded.payload == payload.encode("utf-8")
            detail = f"decode status {decoded.status}"
        except Exception as exc:
            recovered = False
            match = False
            detail = f"decode raised: {exc}"

        survivals.append(
            ChannelSurvival(
                channel_id=channel_id,
                channel_class=channel_class,
                embedded=True,
                recovered=recovered,
                payload_match=match,
                detail=detail,
            )
        )

    return RoundTripResult(
        degradation=base.degradation,
        reference_text=base.reference_text,
        recovered_text=base.recovered_text,
        character_error_rate=base.character_error_rate,
        word_error_rate=base.word_error_rate,
        mean_confidence=base.mean_confidence,
        min_confidence=base.min_confidence,
        engine_id=base.engine_id,
        profile_fingerprint=base.profile_fingerprint,
        render_settings=base.render_settings,
        channels=tuple(survivals),
        warnings=base.warnings,
    )


def summarise_surface(results: list[RoundTripResult]) -> dict[str, Any]:
    """Aggregate round-trip cells into a survival summary.

    Survival is reported per channel *class* because the finding is structural:
    container-bound channels do not survive a scan at any severity, including
    severity zero, because the payload never reaches the page.

    Error-rate statistics are computed over *usable* cells only. Cells where the
    detector dropped whole regions are counted separately, because averaging them
    in would attribute a detector failure to the degradation.
    """

    if not results:
        return {"cells": 0, "channels": {}, "note": "no results"}

    by_channel: dict[str, dict[str, Any]] = {}
    for result in results:
        for channel in result.channels:
            entry = by_channel.setdefault(
                channel.channel_id,
                {
                    "channel_class": channel.channel_class,
                    "embedded_cells": 0,
                    "surviving_cells": 0,
                },
            )
            if channel.embedded:
                entry["embedded_cells"] += 1
                if channel.payload_match:
                    entry["surviving_cells"] += 1

    for entry in by_channel.values():
        embedded = entry["embedded_cells"]
        entry["survival_rate"] = (entry["surviving_cells"] / embedded) if embedded else None

    usable = [result for result in results if result.usable]
    dropped = [result for result in results if not result.usable]
    error_rates = [result.character_error_rate for result in usable]
    if not error_rates:
        return {
            "cells": len(results),
            "usable_cells": 0,
            "detection_dropout_cells": len(dropped),
            "channels": by_channel,
            "note": DETECTION_DROPOUT_NOTE,
        }
    return {
        "cells": len(results),
        "usable_cells": len(usable),
        "detection_dropout_cells": len(dropped),
        "engine_id": results[0].engine_id,
        "profile_fingerprint": results[0].profile_fingerprint,
        "character_error_rate": {
            "min": min(error_rates),
            "max": max(error_rates),
            "mean": sum(error_rates) / len(error_rates),
        },
        "channels": by_channel,
    }
