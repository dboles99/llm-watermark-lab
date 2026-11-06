"""Does a statistical watermark survive print-scan-OCR, when container marks do not?

This is the measurement the OCR track exists to make, and the one Track H turns
on after the 2026-08-30 sweep found the rest of that track already occupied.

The comparison has two arms over the same page:

- **Container-bound channels** — zero-width characters, whitespace encodings,
  Unicode variation selectors, C2PA manifests. Their payload lives in bytes a
  scanner never captures. Measured elsewhere as 0% recovery at every severity,
  *including a clean render at severity zero*, because the payload never reaches
  the page at all.
- **The content-bound channel** — a statistical, logit-biased watermark, whose
  signal lives in *which words were chosen*. Words survive as long as they are
  read correctly, so this is the only family with any prospect of crossing the
  print-scan gap.

If the content arm survives where the container arm cannot, then a statistical
soft binding is *necessary* rather than merely available — which is precisely
what none of the six TEXT-domain algorithms registered with C2PA provides, and
what none of them publishes a survival matrix for.

The context-corruption prediction
---------------------------------
For context width ``h``, each token's green list derives from the preceding ``h``
tokens, so one misread token corrupts its own decision **and** the following
``h``. Detection loss should therefore be super-linear in character error rate
and increasing in ``h``, predicting that ``h = 1`` (lefthash) is materially more
OCR-robust than ``h = 4`` (selfhash). ``compare_context_widths`` runs that
comparison so the prediction is tested rather than asserted.

Nothing here is promotable. There is no frozen OCR-specific null for any engine
profile or locale, so a surviving z-score is mechanics evidence about signal
degradation, not a calibrated detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from watermark_lab.ocr.degrade import Degradation, apply_degradation
from watermark_lab.ocr.render import RenderSettings, render_text_to_image
from watermark_lab.ocr.survival import character_error_rate, word_error_rate

NOT_CALIBRATED_NOTE = (
    "No OCR-specific calibration null is frozen for this engine profile, locale, or "
    "degradation level. A surviving detector score is evidence about signal degradation, "
    "not a calibrated detection, and no false-positive rate is implied."
)


@dataclass(frozen=True, slots=True)
class StatisticalSurvivalCell:
    """One (degradation, severity) cell of the statistical survival surface."""

    degradation: dict[str, Any]
    character_error_rate: float
    word_error_rate: float
    token_error_rate: float
    reference_tokens: int
    recovered_tokens: int
    reference_score: float | None
    recovered_score: float | None
    reference_detected: bool | None
    recovered_detected: bool | None
    coverage_ratio: float
    detection_dropout: bool
    context_width: int
    predicted_corruption: float
    warnings: tuple[str, ...] = field(default=())

    @property
    def score_retained(self) -> float | None:
        """Fraction of the clean detector score that survived, when both exist."""

        if self.reference_score in (None, 0.0) or self.recovered_score is None:
            return None
        assert self.reference_score is not None
        return self.recovered_score / self.reference_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "degradation": self.degradation,
            "character_error_rate": self.character_error_rate,
            "word_error_rate": self.word_error_rate,
            "token_error_rate": self.token_error_rate,
            "reference_tokens": self.reference_tokens,
            "recovered_tokens": self.recovered_tokens,
            "reference_score": self.reference_score,
            "recovered_score": self.recovered_score,
            "reference_detected": self.reference_detected,
            "recovered_detected": self.recovered_detected,
            "score_retained": self.score_retained,
            "coverage_ratio": self.coverage_ratio,
            "detection_dropout": self.detection_dropout,
            "context_width": self.context_width,
            "predicted_corruption": self.predicted_corruption,
            "warnings": list(self.warnings),
        }


def token_error_rate(reference_ids: list[int], recovered_ids: list[int]) -> float:
    """Edit distance over token IDs, normalised by reference length.

    Token error rate is the quantity the watermark actually cares about.
    Character error rate is only a proxy: in Latin script one misread character
    may leave the token intact or may re-segment a whole word, while in CJK a
    single character is frequently a whole token, so the two rates relate very
    differently by locale.
    """

    if not reference_ids:
        return 0.0 if not recovered_ids else 1.0
    a = "".join(chr(0xE000 + (t % 0x1000)) for t in reference_ids)
    b = "".join(chr(0xE000 + (t % 0x1000)) for t in recovered_ids)
    from watermark_lab.ocr.survival import levenshtein

    return levenshtein(a, b) / len(reference_ids)


def predicted_context_corruption(token_error: float, context_width: int) -> float:
    """Fraction of green-list decisions a given token error rate should invalidate.

    ``min(1, token_error * (1 + h))`` — the upper bound before overlap. This is
    the closed form behind the prediction that wider contexts degrade faster.
    """

    if context_width < 0:
        raise ValueError("context_width must be nonnegative.")
    return min(1.0, max(0.0, token_error) * (1 + context_width))


def measure_statistical_survival(
    watermarked_text: str,
    *,
    engine: Any,
    settings: RenderSettings,
    degradations: list[Degradation],
    scorer: Any,
    tokenizer: Any,
    context_width: int = 1,
    coverage_floor: float = 0.90,
) -> list[StatisticalSurvivalCell]:
    """Render, degrade, OCR, and re-score watermarked text at each degradation.

    ``scorer`` must accept text and return ``(score, detected)`` where ``score``
    is a detector statistic and ``detected`` is its boolean decision, or
    ``(None, None)`` when the text is too short to score. ``tokenizer`` must
    expose ``encode(text) -> list[int]``.

    Cells where the detector dropped whole regions are flagged rather than
    scored, because a detection failure attributed to the degradation would
    fabricate fragility the degradation never caused.
    """

    reference_ids = list(tokenizer.encode(watermarked_text))
    reference_score, reference_detected = scorer(watermarked_text)
    page = render_text_to_image(watermarked_text, settings)

    cells: list[StatisticalSurvivalCell] = []
    for degradation in degradations:
        degraded = apply_degradation(page, degradation)
        result = engine.recognize(degraded)
        recovered = result.text

        coverage = (
            len(recovered) / len(watermarked_text) if watermarked_text else 1.0
        )
        dropout = coverage < coverage_floor

        recovered_ids = list(tokenizer.encode(recovered)) if recovered.strip() else []
        tok_error = token_error_rate(reference_ids, recovered_ids)

        if dropout or not recovered.strip():
            recovered_score, recovered_detected = None, None
        else:
            recovered_score, recovered_detected = scorer(recovered)

        warnings = list(result.warnings)
        if dropout:
            warnings.append(
                "detector dropout: recovered text is far shorter than the reference, so this "
                "cell measures a detection failure rather than watermark degradation"
            )

        cells.append(
            StatisticalSurvivalCell(
                degradation=degradation.as_dict(),
                character_error_rate=character_error_rate(watermarked_text, recovered),
                word_error_rate=word_error_rate(watermarked_text, recovered),
                token_error_rate=tok_error,
                reference_tokens=len(reference_ids),
                recovered_tokens=len(recovered_ids),
                reference_score=reference_score,
                recovered_score=recovered_score,
                reference_detected=reference_detected,
                recovered_detected=recovered_detected,
                coverage_ratio=coverage,
                detection_dropout=dropout,
                context_width=context_width,
                predicted_corruption=predicted_context_corruption(tok_error, context_width),
                warnings=tuple(warnings),
            )
        )

    return cells


def summarise_statistical_survival(cells: list[StatisticalSurvivalCell]) -> dict[str, Any]:
    """Aggregate cells, reporting survival over usable cells only."""

    if not cells:
        return {"cells": 0, "note": "no cells"}

    usable = [cell for cell in cells if not cell.detection_dropout]
    scored = [cell for cell in usable if cell.recovered_detected is not None]
    survived = [cell for cell in scored if cell.recovered_detected]
    retained = [
        cell.score_retained for cell in scored if cell.score_retained is not None
    ]

    return {
        "cells": len(cells),
        "usable_cells": len(usable),
        "detection_dropout_cells": len(cells) - len(usable),
        "scored_cells": len(scored),
        "surviving_cells": len(survived),
        "survival_rate": (len(survived) / len(scored)) if scored else None,
        "reference_detected": cells[0].reference_detected,
        "context_width": cells[0].context_width,
        "mean_score_retained": (sum(retained) / len(retained)) if retained else None,
        "character_error_rate": {
            "min": min(cell.character_error_rate for cell in usable) if usable else None,
            "max": max(cell.character_error_rate for cell in usable) if usable else None,
        },
        "token_error_rate": {
            "min": min(cell.token_error_rate for cell in usable) if usable else None,
            "max": max(cell.token_error_rate for cell in usable) if usable else None,
        },
        "not_calibrated": NOT_CALIBRATED_NOTE,
    }


def compare_context_widths(
    summaries_by_width: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Test the prediction that narrower contexts survive OCR better.

    Returns the survival rate per context width and whether the observed ordering
    is consistent with the prediction. A verdict of ``"not-supported"`` is a real
    result: the prediction is falsifiable and reporting it either way is the point.
    """

    if len(summaries_by_width) < 2:
        return {
            "verdict": "insufficient-data",
            "note": "at least two context widths are required to test the prediction",
        }

    rates = {
        width: summary.get("survival_rate")
        for width, summary in sorted(summaries_by_width.items())
    }
    measured = {width: rate for width, rate in rates.items() if rate is not None}
    if len(measured) < 2:
        return {
            "verdict": "insufficient-data",
            "survival_rate_by_context_width": rates,
            "note": "too few widths produced a scorable survival rate",
        }

    widths = sorted(measured)
    monotone = all(
        measured[widths[i]] >= measured[widths[i + 1]] for i in range(len(widths) - 1)
    )
    strict = measured[widths[0]] > measured[widths[-1]]

    if monotone and strict:
        verdict = "supported"
    elif monotone:
        verdict = "consistent-but-flat"
    else:
        verdict = "not-supported"

    return {
        "verdict": verdict,
        "survival_rate_by_context_width": rates,
        "prediction": (
            "One misread token corrupts its own green-list decision and the following h, so "
            "survival should fall as context width rises."
        ),
        "note": NOT_CALIBRATED_NOTE,
    }


__all__ = [
    "NOT_CALIBRATED_NOTE",
    "StatisticalSurvivalCell",
    "compare_context_widths",
    "measure_statistical_survival",
    "predicted_context_corruption",
    "summarise_statistical_survival",
    "token_error_rate",
]
