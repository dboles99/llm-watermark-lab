"""Is OCR damage to a watermark locale-dependent?

The prediction, stated so it can fail
-------------------------------------
A green-list watermark is damaged in proportion to how many *tokens* change, not
how many characters change. The two rates are not interchangeable, and the
relationship between them is a property of the script-and-tokenizer pair.

**Prediction:** the token-error-to-character-error ratio differs materially
between `en` and `zh-Hant-TW`, so watermark survival under OCR is
locale-dependent. If it holds, a single OCR-robustness figure is not
transferable across locales and per-locale calibration is mandatory rather than
merely prudent.

The verdict may come back `not-supported`. That is a real result and is reported
as one.

Mechanism: corrected after measuring
------------------------------------
An earlier version of this module attributed the effect to CJK errors being
visually-similar character confusions with no whitespace anchor, and to "a
single CJK character being frequently a whole token". **The second half of that
is wrong for the tokenizer actually in use, and the measured data says so.**

Under GPT-2's byte-level BPE the pilot measured **0.169 tokens per character for
`en`** (roughly six characters share a token) and **2.095 tokens per character
for `zh-Hant-TW`** (each character costs about two tokens). The ratio is
therefore driven by *token granularity relative to characters*, not by
whitespace or by one-character-one-token:

- Where many characters share a token, one misread character damages a token
  spanning all of them, and re-segmentation can spread the damage further. The
  pilot measured a token-to-character ratio of **12.0** for `en`.
- Where each character already costs several tokens, one misread character
  damages only its own few tokens out of a proportionally larger total, so the
  two rates move together. The pilot measured **0.92** for `zh-Hant-TW`.

The prediction survives; the explanation for it changed. A CJK-aware tokenizer
would shift these numbers again, which is precisely why the ratio is a property
of the script-and-tokenizer pair rather than of the language alone.

Known limitations of the pilot that produced these numbers
----------------------------------------------------------
- **The character error rates are not matched.** The recogniser read
  Traditional Chinese far worse than English (median CER 0.21 against 0.011).
  Comparing ratios across such different error regimes is weaker evidence than
  comparing them at a matched rate, and the difference in raw legibility is a
  finding in its own right.
- **The tokenizer is English-centric.** GPT-2 byte-level BPE is not a reasonable
  production tokenizer for `zh-Hant-TW`, so the `zh` arm measures this
  tokenizer's behaviour on Chinese rather than a well-matched pairing.
- Synthetic corpora on both arms, one degradation level, one engine.

A second question, specific to this project
-------------------------------------------
The bundled recogniser models are trained principally on Simplified Chinese. If
recognition of Traditional input emits Simplified forms, then **the instrument
itself introduces the `zh-CN` contamination the programme forbids** — and it does
so after corpus screening has already passed. Screening therefore has to run on
recognised output, not only on corpus intake. ``screen_recovered_text`` measures
that directly.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from watermark_lab.corpora.tw_cn_discriminator import screen
from watermark_lab.ocr.degrade import Degradation, apply_degradation
from watermark_lab.ocr.render import build_render_settings, render_text_to_image
from watermark_lab.ocr.statistical_survival import token_error_rate
from watermark_lab.ocr.survival import character_error_rate

#: Ratio difference below which the locales are treated as behaving the same.
DEFAULT_RATIO_MARGIN = 0.25


@dataclass(frozen=True, slots=True)
class LocaleReading:
    """One document round-tripped through render, degrade and recognise."""

    locale: str
    reference_chars: int
    reference_tokens: int
    character_error_rate: float
    token_error_rate: float
    coverage_ratio: float
    dropped: bool
    contamination_verdict: str | None = None
    contamination_introduced: bool = False

    @property
    def token_to_character_ratio(self) -> float | None:
        """How many token errors each character error costs.

        ``None`` when the character error rate is zero: dividing by it would
        report an infinite ratio from a perfect reading, which says nothing.
        """

        if self.character_error_rate <= 0.0:
            return None
        return self.token_error_rate / self.character_error_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "locale": self.locale,
            "reference_chars": self.reference_chars,
            "reference_tokens": self.reference_tokens,
            "character_error_rate": self.character_error_rate,
            "token_error_rate": self.token_error_rate,
            "token_to_character_ratio": self.token_to_character_ratio,
            "coverage_ratio": self.coverage_ratio,
            "dropped": self.dropped,
            "contamination_verdict": self.contamination_verdict,
            "contamination_introduced": self.contamination_introduced,
        }


@dataclass(frozen=True, slots=True)
class LocaleSummary:
    """Aggregated readings for one locale."""

    locale: str
    readings: tuple[LocaleReading, ...]

    @property
    def usable(self) -> tuple[LocaleReading, ...]:
        return tuple(r for r in self.readings if not r.dropped)

    @property
    def median_ratio(self) -> float | None:
        ratios = [
            r.token_to_character_ratio
            for r in self.usable
            if r.token_to_character_ratio is not None
        ]
        return statistics.median(ratios) if ratios else None

    @property
    def median_character_error_rate(self) -> float | None:
        values = [r.character_error_rate for r in self.usable]
        return statistics.median(values) if values else None

    @property
    def median_token_error_rate(self) -> float | None:
        values = [r.token_error_rate for r in self.usable]
        return statistics.median(values) if values else None

    @property
    def tokens_per_character(self) -> float | None:
        """Token granularity relative to characters, the driver of the ratio.

        Many characters per token means one misread character damages a token
        covering all of them. Several tokens per character means the damage
        stays proportional. This is the quantity that actually explains the
        locale difference, so it is reported alongside it.
        """

        chars = sum(r.reference_chars for r in self.usable)
        tokens = sum(r.reference_tokens for r in self.usable)
        return tokens / chars if chars else None

    @property
    def contamination_count(self) -> int:
        return sum(1 for r in self.usable if r.contamination_introduced)

    def as_dict(self) -> dict[str, Any]:
        return {
            "locale": self.locale,
            "readings": len(self.readings),
            "usable_readings": len(self.usable),
            "dropped_readings": len(self.readings) - len(self.usable),
            "median_character_error_rate": self.median_character_error_rate,
            "median_token_error_rate": self.median_token_error_rate,
            "median_token_to_character_ratio": self.median_ratio,
            "tokens_per_character": self.tokens_per_character,
            "contamination_introduced_count": self.contamination_count,
        }


def screen_recovered_text(reference: str, recovered: str) -> tuple[str | None, bool]:
    """Screen recognised output for Simplified forms the source did not contain.

    Returns the verdict for the recovered text and whether contamination is
    *newly introduced* by recognition. A source that already carried mainland
    vocabulary is a corpus problem, not an instrument problem, and the two must
    not be conflated.
    """

    if not recovered.strip():
        return None, False

    before = screen(reference, minimum_cjk=1)
    after = screen(recovered, minimum_cjk=1)

    introduced = bool(
        set(after.simplified_characters) - set(before.simplified_characters)
    ) or (after.mainland_total > before.mainland_total)
    return after.verdict, introduced


def compare_locales(
    documents_by_locale: dict[str, Sequence[str]],
    *,
    engine: Any,
    degradation: Degradation,
    tokenizer: Any,
    font_size: int = 30,
    coverage_floor: float = 0.90,
) -> dict[str, LocaleSummary]:
    """Round-trip documents from each locale under an identical degradation.

    The degradation must be identical across locales, or the comparison measures
    the degradation rather than the script.
    """

    if not documents_by_locale:
        raise ValueError("documents_by_locale must not be empty.")

    summaries: dict[str, LocaleSummary] = {}
    for locale, documents in documents_by_locale.items():
        readings: list[LocaleReading] = []
        for text in documents:
            settings = build_render_settings(text, locale=locale, font_size=font_size)
            page = render_text_to_image(text, settings)
            recovered = engine.recognize(apply_degradation(page, degradation)).text

            coverage = len(recovered) / len(text) if text else 1.0
            dropped = coverage < coverage_floor

            reference_ids = list(tokenizer.encode(text))
            recovered_ids = list(tokenizer.encode(recovered)) if recovered.strip() else []

            verdict, introduced = (None, False)
            if locale.startswith("zh"):
                verdict, introduced = screen_recovered_text(text, recovered)

            readings.append(
                LocaleReading(
                    locale=locale,
                    reference_chars=len(text),
                    reference_tokens=len(reference_ids),
                    character_error_rate=character_error_rate(text, recovered),
                    token_error_rate=token_error_rate(reference_ids, recovered_ids),
                    coverage_ratio=coverage,
                    dropped=dropped,
                    contamination_verdict=verdict,
                    contamination_introduced=introduced,
                )
            )
        summaries[locale] = LocaleSummary(locale=locale, readings=tuple(readings))

    return summaries


@dataclass(frozen=True, slots=True)
class LocaleVerdict:
    """Whether the locale-dependence prediction held."""

    verdict: str
    ratios: dict[str, float | None]
    relative_difference: float | None
    margin: float
    detail: str
    contamination: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "median_token_to_character_ratio": self.ratios,
            "relative_difference": self.relative_difference,
            "margin": self.margin,
            "detail": self.detail,
            "contamination_introduced": self.contamination,
        }


def evaluate_locale_prediction(
    summaries: dict[str, LocaleSummary],
    *,
    margin: float = DEFAULT_RATIO_MARGIN,
) -> LocaleVerdict:
    """Decide whether token-per-character damage differs materially by locale.

    ``margin`` is the relative difference below which the locales are treated as
    behaving the same. It is a preregistered threshold, not a p-value: this is a
    descriptive pilot comparison, and calling it a test would overstate it.
    """

    ratios = {locale: summary.median_ratio for locale, summary in summaries.items()}
    contamination = {
        locale: summary.contamination_count for locale, summary in summaries.items()
    }
    measured = {locale: value for locale, value in ratios.items() if value is not None}

    if len(measured) < 2:
        return LocaleVerdict(
            verdict="insufficient-data",
            ratios=ratios,
            relative_difference=None,
            margin=margin,
            detail=(
                "at least two locales must produce a measurable token-to-character ratio; "
                "a locale whose readings were perfect or entirely dropped cannot contribute"
            ),
            contamination=contamination,
        )

    values = sorted(measured.values())
    low, high = values[0], values[-1]
    relative = (high - low) / low if low > 0 else None

    if relative is None:
        verdict = "insufficient-data"
        detail = "a median ratio of zero makes the relative difference undefined"
    elif relative >= margin:
        verdict = "supported"
        detail = (
            f"token damage per character error differs by {relative:.0%} across locales, "
            f"above the {margin:.0%} margin; a single OCR-robustness figure does not "
            "transfer across locales and per-locale calibration is required"
        )
    else:
        verdict = "not-supported"
        detail = (
            f"token damage per character error differs by only {relative:.0%}, below the "
            f"{margin:.0%} margin; this pilot does not support locale-dependent damage"
        )

    return LocaleVerdict(
        verdict=verdict,
        ratios=ratios,
        relative_difference=relative,
        margin=margin,
        detail=detail,
        contamination=contamination,
    )


__all__ = [
    "DEFAULT_RATIO_MARGIN",
    "LocaleReading",
    "LocaleSummary",
    "LocaleVerdict",
    "compare_locales",
    "evaluate_locale_prediction",
    "screen_recovered_text",
]
