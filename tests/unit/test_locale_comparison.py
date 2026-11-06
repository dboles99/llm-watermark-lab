"""Locale-dependence of OCR damage, and OCR-introduced Simplified contamination.

The prediction under test is falsifiable, so the tests check that it can be
reported either way, and that the ratio is refused rather than fabricated when a
reading was perfect or entirely dropped.
"""

from __future__ import annotations

from typing import Any

import pytest

from watermark_lab.ocr.locale_comparison import (
    DEFAULT_RATIO_MARGIN,
    LocaleReading,
    LocaleSummary,
    compare_locales,
    evaluate_locale_prediction,
    screen_recovered_text,
)
from watermark_lab.ocr.profile import build_profile

TRADITIONAL = "這個軟體的網路功能需要更多的資訊才能運行，使用者可以透過滑鼠點擊螢幕上的選單。"


def _reading(
    locale: str,
    *,
    cer: float,
    ter: float,
    dropped: bool = False,
    contaminated: bool = False,
) -> LocaleReading:
    return LocaleReading(
        locale=locale,
        reference_chars=100,
        reference_tokens=50,
        character_error_rate=cer,
        token_error_rate=ter,
        coverage_ratio=0.5 if dropped else 0.99,
        dropped=dropped,
        contamination_introduced=contaminated,
    )


def _summary(locale: str, readings: list[LocaleReading]) -> LocaleSummary:
    return LocaleSummary(locale=locale, readings=tuple(readings))


class TestRatioArithmetic:
    def test_ratio_is_token_error_over_character_error(self) -> None:
        assert _reading("en", cer=0.02, ter=0.10).token_to_character_ratio == pytest.approx(5.0)

    def test_perfect_reading_has_no_ratio(self) -> None:
        """A zero character error rate would give an infinite ratio from nothing."""

        assert _reading("en", cer=0.0, ter=0.0).token_to_character_ratio is None

    def test_dropped_readings_are_excluded_from_medians(self) -> None:
        summary = _summary(
            "en",
            [
                _reading("en", cer=0.01, ter=0.05),
                _reading("en", cer=0.90, ter=0.99, dropped=True),
            ],
        )
        assert len(summary.usable) == 1
        assert summary.median_character_error_rate == pytest.approx(0.01)

    def test_summary_of_only_dropped_readings_has_no_medians(self) -> None:
        summary = _summary("en", [_reading("en", cer=0.9, ter=0.9, dropped=True)])
        assert summary.median_ratio is None
        assert summary.median_character_error_rate is None


class TestTokenGranularity:
    """Tokens-per-character is what actually drives the locale difference.

    The pilot measured 0.169 for en (about six characters share a token) and
    2.095 for zh-Hant-TW under GPT-2 byte-level BPE. That, not whitespace,
    explains why one misread character costs twelve token errors in one locale
    and about one in the other.
    """

    def test_reports_token_granularity(self) -> None:
        summary = _summary(
            "en",
            [
                LocaleReading(
                    locale="en",
                    reference_chars=480,
                    reference_tokens=80,
                    character_error_rate=0.01,
                    token_error_rate=0.12,
                    coverage_ratio=0.99,
                    dropped=False,
                )
            ],
        )
        assert summary.tokens_per_character == pytest.approx(80 / 480)

    def test_granularity_excludes_dropped_readings(self) -> None:
        summary = _summary("en", [_reading("en", cer=0.9, ter=0.9, dropped=True)])
        assert summary.tokens_per_character is None

    def test_granularity_appears_in_the_summary(self) -> None:
        summary = _summary("en", [_reading("en", cer=0.01, ter=0.05)])
        assert "tokens_per_character" in summary.as_dict()


class TestVerdictIsFalsifiable:
    def test_large_difference_supports_the_prediction(self) -> None:
        summaries = {
            "en": _summary("en", [_reading("en", cer=0.01, ter=0.07)]),        # ratio 7
            "zh-Hant-TW": _summary(
                "zh-Hant-TW", [_reading("zh-Hant-TW", cer=0.02, ter=0.02)]     # ratio 1
            ),
        }
        verdict = evaluate_locale_prediction(summaries)
        assert verdict.verdict == "supported"
        assert verdict.relative_difference is not None
        assert verdict.relative_difference > DEFAULT_RATIO_MARGIN
        assert "per-locale calibration is required" in verdict.detail

    def test_small_difference_does_not_support_it(self) -> None:
        """A null result is a real result and must be reportable."""

        summaries = {
            "en": _summary("en", [_reading("en", cer=0.02, ter=0.040)]),       # ratio 2.0
            "zh-Hant-TW": _summary(
                "zh-Hant-TW", [_reading("zh-Hant-TW", cer=0.02, ter=0.042)]    # ratio 2.1
            ),
        }
        verdict = evaluate_locale_prediction(summaries)
        assert verdict.verdict == "not-supported"
        assert "does not support" in verdict.detail

    def test_margin_is_configurable_and_reported(self) -> None:
        summaries = {
            "en": _summary("en", [_reading("en", cer=0.02, ter=0.04)]),
            "zh-Hant-TW": _summary("zh-Hant-TW", [_reading("zh-Hant-TW", cer=0.02, ter=0.05)]),
        }
        strict = evaluate_locale_prediction(summaries, margin=0.05)
        loose = evaluate_locale_prediction(summaries, margin=0.90)
        assert strict.verdict == "supported"
        assert loose.verdict == "not-supported"
        assert strict.margin == 0.05

    def test_one_measurable_locale_is_insufficient(self) -> None:
        summaries = {
            "en": _summary("en", [_reading("en", cer=0.02, ter=0.04)]),
            "zh-Hant-TW": _summary("zh-Hant-TW", [_reading("zh-Hant-TW", cer=0.0, ter=0.0)]),
        }
        verdict = evaluate_locale_prediction(summaries)
        assert verdict.verdict == "insufficient-data"

    def test_verdict_serialises_with_its_inputs(self) -> None:
        summaries = {
            "en": _summary("en", [_reading("en", cer=0.01, ter=0.07)]),
            "zh-Hant-TW": _summary("zh-Hant-TW", [_reading("zh-Hant-TW", cer=0.02, ter=0.02)]),
        }
        payload = evaluate_locale_prediction(summaries).as_dict()
        assert set(payload["median_token_to_character_ratio"]) == {"en", "zh-Hant-TW"}
        assert payload["margin"] == DEFAULT_RATIO_MARGIN


class TestOcrIntroducedContamination:
    """The recogniser's own language model can inject Simplified forms.

    That matters here more than usual: it happens *after* corpus screening has
    passed, so screening must run on recognised output too.
    """

    def test_clean_recovery_introduces_nothing(self) -> None:
        verdict, introduced = screen_recovered_text(TRADITIONAL, TRADITIONAL)
        assert introduced is False
        assert verdict is not None

    def test_simplified_characters_in_output_are_flagged(self) -> None:
        recovered = TRADITIONAL.replace("這個軟體", "这个软件")
        _verdict, introduced = screen_recovered_text(TRADITIONAL, recovered)
        assert introduced is True

    def test_mainland_vocabulary_in_output_is_flagged(self) -> None:
        """Traditional glyphs, mainland wording: the case a script check misses."""

        recovered = TRADITIONAL.replace("軟體", "軟件").replace("網路", "網絡")
        _verdict, introduced = screen_recovered_text(TRADITIONAL, recovered)
        assert introduced is True

    def test_preexisting_contamination_is_not_blamed_on_the_engine(self) -> None:
        """A corpus problem must not be reported as an instrument problem."""

        source = TRADITIONAL.replace("軟體", "軟件")
        _verdict, introduced = screen_recovered_text(source, source)
        assert introduced is False

    def test_empty_recovery_is_not_contamination(self) -> None:
        verdict, introduced = screen_recovered_text(TRADITIONAL, "")
        assert verdict is None
        assert introduced is False


class _StubEngine:
    """Echoes a fixed recovered text so the comparison logic is under test."""

    engine_id = "stub"

    def __init__(self, recovered_by_length: dict[int, str] | None = None) -> None:
        self._map = recovered_by_length or {}
        self.profile = build_profile(
            engine_id="stub", engine_version="1.0", language="en", assets=()
        )

    def recognize(self, image: Any) -> Any:
        mapping = self._map

        class _Result:
            text = mapping.get("default", "recovered")

        return _Result()


class _Tok:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 1000 for ch in text]


class TestCompareLocales:
    def test_requires_at_least_one_locale(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compare_locales(
                {},
                engine=_StubEngine(),
                degradation=__import__(
                    "watermark_lab.ocr.degrade", fromlist=["Degradation"]
                ).Degradation("clean", 0.0, 0),
                tokenizer=_Tok(),
            )

    def test_only_zh_locales_are_screened_for_contamination(self) -> None:
        from watermark_lab.ocr.degrade import Degradation

        summaries = compare_locales(
            {"en": ["hello world this is a sentence"]},
            engine=_StubEngine({"default": "hello world this is a sentence"}),
            degradation=Degradation("clean", 0.0, 0),
            tokenizer=_Tok(),
            coverage_floor=0.1,
        )
        reading = summaries["en"].readings[0]
        assert reading.contamination_verdict is None
        assert reading.contamination_introduced is False
