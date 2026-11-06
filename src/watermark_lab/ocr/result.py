"""OCR result types carrying provenance and recognition confidence.

Text recovered by OCR is not the same evidentiary object as text read from a
file. It is a *reconstruction*, and the reconstruction can be wrong in ways that
look exactly like watermark evidence or exactly like its absence. Every value
produced here therefore travels with the profile that produced it and the
confidence distribution behind it, so a downstream detector can refuse rather
than silently reuse a clean-text threshold.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Any

PROVENANCE_CHANNEL = "ocr"

OCR_RECONSTRUCTION_NOTE = (
    "This text was reconstructed by optical character recognition. It is not a faithful "
    "copy of the source bytes: characters may be substituted, dropped, inserted, or "
    "re-segmented. Detector thresholds calibrated on clean text do not transfer to it."
)


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One recognised region with its own confidence."""

    text: str
    confidence: float
    box: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": list(self.box) if self.box is not None else None,
        }


def sort_reading_order(
    blocks: tuple[TextBlock, ...], *, row_tolerance: float = 0.6
) -> tuple[TextBlock, ...]:
    """Sort detected blocks into top-to-bottom, left-to-right reading order.

    Detection order is not reading order. Engines emit regions in the order their
    detector found them, so joining blocks as returned can scramble a multi-line
    page — which inflates the measured character error rate on a *clean* render
    and makes the degradation ladder non-monotonic. The error then looks like
    fragility in the watermark rather than a defect in the harness.

    Blocks are grouped into rows using a tolerance proportional to median block
    height, then ordered left-to-right within each row. Blocks without geometry
    keep their relative position at the end, since nothing can be inferred about
    where they belong.
    """

    positioned = [block for block in blocks if block.box is not None]
    unpositioned = [block for block in blocks if block.box is None]
    if not positioned:
        return blocks

    heights = sorted(block.box[3] - block.box[1] for block in positioned)  # type: ignore[index]
    median_height = heights[len(heights) // 2] or 1.0
    threshold = median_height * row_tolerance

    rows: list[list[TextBlock]] = []
    for block in sorted(positioned, key=lambda b: b.box[1]):  # type: ignore[index]
        top = block.box[1]  # type: ignore[index]
        for row in rows:
            if abs(row[0].box[1] - top) <= threshold:  # type: ignore[index]
                row.append(block)
                break
        else:
            rows.append([block])

    ordered: list[TextBlock] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.box[0]))  # type: ignore[index]
    ordered.extend(unpositioned)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Text recovered from an image, with the provenance needed to judge it.

    ``mean_confidence`` is length-weighted: a long block recognised poorly should
    not be masked by a short block recognised well. ``min_confidence`` is kept
    separately because a single badly-read region can carry the numeric claim
    that matters most in a document.
    """

    text: str
    blocks: tuple[TextBlock, ...]
    engine_id: str
    profile_fingerprint: str
    language: str
    provenance_channel: str = PROVENANCE_CHANNEL
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def mean_confidence(self) -> float:
        """Length-weighted mean confidence, or 0.0 when nothing was recognised."""

        weighted = 0.0
        total_length = 0
        for block in self.blocks:
            length = len(block.text)
            if length == 0:
                continue
            weighted += block.confidence * length
            total_length += length
        if total_length == 0:
            return 0.0
        return weighted / total_length

    @property
    def min_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return min(block.confidence for block in self.blocks)

    @property
    def median_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return statistics.median(block.confidence for block in self.blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_sha256": self.text_sha256,
            "provenance_channel": self.provenance_channel,
            "engine_id": self.engine_id,
            "profile_fingerprint": self.profile_fingerprint,
            "language": self.language,
            "block_count": self.block_count,
            "mean_confidence": self.mean_confidence,
            "median_confidence": self.median_confidence,
            "min_confidence": self.min_confidence,
            "warnings": list(self.warnings),
            "reconstruction_note": OCR_RECONSTRUCTION_NOTE,
            "metadata": dict(self.metadata),
        }
