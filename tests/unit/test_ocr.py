"""OCR instrument, degradation ladder, and round-trip survival measurement.

Engine-dependent tests are skipped when the engine is unavailable rather than
faked, because a stubbed OCR engine would not exercise the property that matters:
that real recognition loses container-bound payloads entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from watermark_lab.ocr import (
    DEFAULT_COVERAGE_FLOOR,
    Degradation,
    DegradationError,
    OcrProfileError,
    RenderError,
    TextBlock,
    apply_chain,
    apply_degradation,
    available_engine_ids,
    build_ladder,
    build_profile,
    build_render_settings,
    character_error_rate,
    context_corruption_span,
    get_engine,
    levenshtein,
    rapidocr_available,
    render_text_to_image,
    round_trip,
    select_font,
    sort_reading_order,
    summarise_surface,
    tesseract_available,
    word_error_rate,
)
from watermark_lab.ocr.profile import AssetFingerprint

ENGLISH = (
    "Provenance evidence must survive ordinary document handling to be useful. "
    "A watermark that vanishes when a page is printed and scanned again cannot "
    "support any claim about where a document came from."
)

TRADITIONAL_CHINESE = "臺灣繁體中文的浮水印必須在文件掃描之後仍然可以被驗證。"

requires_rapidocr = pytest.mark.skipif(
    not rapidocr_available(), reason="rapidocr_onnxruntime is not installed"
)
requires_tesseract = pytest.mark.skipif(
    not tesseract_available(), reason="tesseract binary is not on PATH"
)


class TestEditDistanceMetrics:
    def test_identical_strings_have_zero_distance(self) -> None:
        assert levenshtein("watermark", "watermark") == 0
        assert character_error_rate("watermark", "watermark") == 0.0
        assert word_error_rate("a b c", "a b c") == 0.0

    def test_distance_is_symmetric(self) -> None:
        assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten") == 3

    def test_empty_reference_handling(self) -> None:
        assert character_error_rate("", "") == 0.0
        assert character_error_rate("", "spurious") == 1.0
        assert character_error_rate("abc", "") == 1.0

    def test_error_rate_can_exceed_one_on_insertion(self) -> None:
        """Noise read as text is a real failure mode and is not clamped away."""

        assert character_error_rate("ab", "ab" + "x" * 20) > 1.0

    def test_word_error_rate_counts_words_not_characters(self) -> None:
        # One word substituted out of four.
        assert word_error_rate("the cat sat down", "the dog sat down") == pytest.approx(0.25)

    def test_word_error_rate_on_unspaced_script(self) -> None:
        """CJK has no inter-word spaces, so the whole line is one unit."""

        assert word_error_rate(TRADITIONAL_CHINESE, TRADITIONAL_CHINESE) == 0.0
        assert word_error_rate(TRADITIONAL_CHINESE, "完全不同的內容") == pytest.approx(1.0)


class TestContextCorruptionSpan:
    def test_wider_context_corrupts_more(self) -> None:
        """The h-dependence prediction: one bad token poisons h+1 decisions."""

        narrow = context_corruption_span(token_count=100, corrupted_tokens=5, context_width=1)
        wide = context_corruption_span(token_count=100, corrupted_tokens=5, context_width=4)
        assert narrow == pytest.approx(0.10)
        assert wide == pytest.approx(0.25)
        assert wide > narrow

    def test_zero_corruption_is_zero_span(self) -> None:
        assert context_corruption_span(
            token_count=100, corrupted_tokens=0, context_width=4
        ) == 0.0

    def test_span_saturates_at_one(self) -> None:
        assert context_corruption_span(
            token_count=10, corrupted_tokens=8, context_width=4
        ) == 1.0

    def test_context_width_zero_is_linear(self) -> None:
        assert context_corruption_span(
            token_count=100, corrupted_tokens=7, context_width=0
        ) == pytest.approx(0.07)

    @pytest.mark.parametrize(
        ("tokens", "corrupted", "width"),
        [(0, 1, 1), (-1, 1, 1), (10, -1, 1), (10, 1, -1)],
    )
    def test_rejects_invalid_inputs(self, tokens: int, corrupted: int, width: int) -> None:
        with pytest.raises(ValueError):
            context_corruption_span(
                token_count=tokens, corrupted_tokens=corrupted, context_width=width
            )


class TestDegradationLadder:
    def _page(self) -> np.ndarray:
        settings = build_render_settings(ENGLISH, locale="en")
        return render_text_to_image(ENGLISH, settings)

    @pytest.mark.parametrize(
        "name", ["clean", "jpeg", "downsample", "rotate", "noise", "blur", "paper", "photo"]
    )
    def test_severity_zero_is_the_identity(self, name: str) -> None:
        """Every rung must have a well-defined null, as the edit-class model requires."""

        page = self._page()
        result = apply_degradation(page, Degradation(name=name, severity=0.0, seed=1))
        assert np.array_equal(result, page)

    @pytest.mark.parametrize("name", ["jpeg", "downsample", "noise", "blur", "paper", "photo"])
    def test_degradation_actually_changes_the_image(self, name: str) -> None:
        page = self._page()
        result = apply_degradation(page, Degradation(name=name, severity=1.0, seed=1))
        assert result.shape == page.shape
        assert not np.array_equal(result, page)

    @pytest.mark.parametrize("name", ["noise", "paper", "photo", "rotate"])
    def test_same_seed_reproduces_the_same_image(self, name: str) -> None:
        page = self._page()
        first = apply_degradation(page, Degradation(name=name, severity=0.7, seed=42))
        second = apply_degradation(page, Degradation(name=name, severity=0.7, seed=42))
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("name", ["noise", "paper"])
    def test_different_seeds_give_different_images(self, name: str) -> None:
        page = self._page()
        first = apply_degradation(page, Degradation(name=name, severity=0.7, seed=1))
        second = apply_degradation(page, Degradation(name=name, severity=0.7, seed=2))
        assert not np.array_equal(first, second)

    def test_rejects_unknown_degradation(self) -> None:
        with pytest.raises(DegradationError, match="Unknown degradation"):
            apply_degradation(self._page(), Degradation(name="teleport", severity=0.5))

    @pytest.mark.parametrize("severity", [-0.1, 1.1])
    def test_rejects_out_of_range_severity(self, severity: float) -> None:
        with pytest.raises(DegradationError):
            apply_degradation(self._page(), Degradation(name="noise", severity=severity))

    def test_chain_applies_in_order(self) -> None:
        page = self._page()
        chained = apply_chain(
            page,
            [Degradation("downsample", 0.5, 3), Degradation("jpeg", 0.5, 3)],
        )
        assert chained.shape == page.shape
        assert not np.array_equal(chained, page)

    def test_ladder_includes_clean_exactly_once(self) -> None:
        ladder = build_ladder()
        clean = [d for d in ladder if d.name == "clean"]
        assert len(clean) == 1
        assert clean[0].severity == 0.0


class TestRendering:
    def test_renders_a_page_with_expected_width(self) -> None:
        settings = build_render_settings(ENGLISH, locale="en", page_width=1200)
        page = render_text_to_image(ENGLISH, settings)
        assert page.shape[1] == 1200
        assert page.shape[2] == 3
        assert page.shape[0] > 0

    def test_render_settings_pin_the_font_by_hash(self) -> None:
        settings = build_render_settings(ENGLISH, locale="en")
        assert len(settings.font_sha256) == 64
        payload = settings.as_dict()
        assert payload["font_sha256"] == settings.font_sha256
        assert payload["font_name"]

    def test_rendering_is_deterministic(self) -> None:
        settings = build_render_settings(ENGLISH, locale="en")
        assert np.array_equal(
            render_text_to_image(ENGLISH, settings),
            render_text_to_image(ENGLISH, settings),
        )

    def test_cjk_text_selects_a_cjk_capable_font(self) -> None:
        """A Latin-only font would render tofu that OCR reads as noise."""

        try:
            font_path, _warnings = select_font(TRADITIONAL_CHINESE, locale="zh-Hant-TW")
        except RenderError:
            pytest.skip("no CJK font available on this machine")
        assert font_path
        latin_path, _ = select_font(ENGLISH, locale="en")
        assert font_path != latin_path

    def test_cjk_renders_without_falling_back_to_latin(self) -> None:
        try:
            settings = build_render_settings(TRADITIONAL_CHINESE, locale="zh-Hant-TW")
        except RenderError:
            pytest.skip("no CJK font available on this machine")
        page = render_text_to_image(TRADITIONAL_CHINESE, settings)
        # Glyphs were drawn: the page is not uniformly white.
        assert page.min() < 128

    def test_rejects_missing_font_path(self) -> None:
        with pytest.raises(RenderError, match="does not exist"):
            build_render_settings(ENGLISH, font_path="/no/such/font.ttf")

    def test_rejects_margin_wider_than_page(self) -> None:
        settings = build_render_settings(ENGLISH, locale="en", page_width=100, margin=400)
        with pytest.raises(RenderError, match="margin"):
            render_text_to_image(ENGLISH, settings)


class TestReadingOrder:
    def test_sorts_top_to_bottom_then_left_to_right(self) -> None:
        blocks = (
            TextBlock("second-right", 0.9, (500.0, 100.0, 700.0, 130.0)),
            TextBlock("first-left", 0.9, (100.0, 20.0, 300.0, 50.0)),
            TextBlock("second-left", 0.9, (100.0, 100.0, 300.0, 130.0)),
            TextBlock("first-right", 0.9, (500.0, 22.0, 700.0, 52.0)),
        )
        ordered = sort_reading_order(blocks)
        assert [b.text for b in ordered] == [
            "first-left",
            "first-right",
            "second-left",
            "second-right",
        ]

    def test_blocks_without_geometry_go_last(self) -> None:
        blocks = (
            TextBlock("no-box", 0.5, None),
            TextBlock("boxed", 0.9, (10.0, 10.0, 20.0, 20.0)),
        )
        ordered = sort_reading_order(blocks)
        assert ordered[-1].text == "no-box"

    def test_empty_input_is_returned_unchanged(self) -> None:
        assert sort_reading_order(()) == ()


class TestProfilePinning:
    def test_fingerprint_changes_when_any_field_changes(self) -> None:
        asset = AssetFingerprint(name="m.onnx", sha256="a" * 64, size_bytes=10)
        base = build_profile(
            engine_id="x", engine_version="1", language="en", assets=(asset,)
        )
        other_version = build_profile(
            engine_id="x", engine_version="2", language="en", assets=(asset,)
        )
        other_language = build_profile(
            engine_id="x", engine_version="1", language="zh-Hant-TW", assets=(asset,)
        )
        other_asset = build_profile(
            engine_id="x",
            engine_version="1",
            language="en",
            assets=(AssetFingerprint(name="m.onnx", sha256="b" * 64, size_bytes=10),),
        )
        fingerprints = {
            base.aggregate_sha256,
            other_version.aggregate_sha256,
            other_language.aggregate_sha256,
            other_asset.aggregate_sha256,
        }
        assert len(fingerprints) == 4

    def test_fingerprint_is_stable_for_identical_input(self) -> None:
        asset = AssetFingerprint(name="m.onnx", sha256="a" * 64, size_bytes=10)
        first = build_profile(engine_id="x", engine_version="1", language="en", assets=(asset,))
        second = build_profile(engine_id="x", engine_version="1", language="en", assets=(asset,))
        assert first.aggregate_sha256 == second.aggregate_sha256

    def test_unknown_engine_is_refused(self) -> None:
        with pytest.raises(OcrProfileError, match="Unknown OCR engine"):
            get_engine("clairvoyance")

    def test_both_engines_are_registered(self) -> None:
        assert set(available_engine_ids()) == {"rapidocr", "tesseract"}


class TestTesseractFailsClosed:
    @pytest.mark.skipif(tesseract_available(), reason="tesseract IS installed here")
    def test_missing_binary_raises_with_actionable_guidance(self) -> None:
        """Absence must not silently fall back to another engine."""

        with pytest.raises(OcrProfileError) as excinfo:
            get_engine("tesseract")
        message = str(excinfo.value)
        assert "not installed" in message
        assert "chi_tra" in message
        assert "rapidocr" in message


@requires_rapidocr
class TestRapidOcrRoundTrip:
    def test_pins_its_onnx_models(self) -> None:
        engine = get_engine("rapidocr")
        assert engine.profile.engine_id == "rapidocr"
        assert engine.profile.assets, "no ONNX model assets were pinned"
        assert all(a.name.endswith(".onnx") for a in engine.profile.assets)
        assert len(engine.profile.aggregate_sha256) == 64

    def test_clean_round_trip_recovers_most_characters(self) -> None:
        settings = build_render_settings(ENGLISH, locale="en", font_size=30)
        engine = get_engine("rapidocr")
        result = round_trip(
            ENGLISH,
            engine=engine,
            settings=settings,
            degradation=Degradation("clean", 0.0, 0),
        )
        assert result.character_error_rate < 0.10
        assert result.mean_confidence > 0.5
        assert result.engine_id == "rapidocr"
        assert result.profile_fingerprint == engine.profile.aggregate_sha256

    def test_result_carries_ocr_provenance(self) -> None:
        settings = build_render_settings(ENGLISH, locale="en", font_size=30)
        engine = get_engine("rapidocr")
        page = render_text_to_image(ENGLISH, settings)
        recognised = engine.recognize(page)
        payload = recognised.as_dict()
        assert payload["provenance_channel"] == "ocr"
        assert "reconstructed" in payload["reconstruction_note"].lower()
        assert "do not transfer" in payload["reconstruction_note"].lower()
        assert payload["text_sha256"]

    def test_severe_degradation_increases_error(self) -> None:
        settings = build_render_settings(ENGLISH, locale="en", font_size=30)
        engine = get_engine("rapidocr")
        clean = round_trip(
            ENGLISH, engine=engine, settings=settings, degradation=Degradation("clean", 0.0, 0)
        )
        wrecked = round_trip(
            ENGLISH, engine=engine, settings=settings, degradation=Degradation("blur", 1.0, 0)
        )
        assert wrecked.character_error_rate > clean.character_error_rate


class TestDetectionDropoutDiagnostic:
    """Detection dropout and recognition error are different failures.

    A detector that misses whole regions produces short output, which would look
    like catastrophic signal loss if scored naively. Cells like that are flagged
    and excluded from error statistics rather than blamed on the degradation.
    """

    def _result(self, recovered: str, reference: str = "x" * 100):
        from watermark_lab.ocr.survival import RoundTripResult

        return RoundTripResult(
            degradation={"name": "clean", "severity": 0.0, "seed": 0},
            reference_text=reference,
            recovered_text=recovered,
            character_error_rate=character_error_rate(reference, recovered),
            word_error_rate=0.0,
            mean_confidence=0.9,
            min_confidence=0.9,
            engine_id="stub",
            profile_fingerprint="f" * 64,
            render_settings={},
        )

    def test_full_coverage_is_usable(self) -> None:
        result = self._result("x" * 100)
        assert result.coverage_ratio == pytest.approx(1.0)
        assert result.detection_dropout is False
        assert result.usable is True

    def test_missing_regions_are_flagged_not_scored(self) -> None:
        result = self._result("x" * 60)
        assert result.coverage_ratio == pytest.approx(0.60)
        assert result.detection_dropout is True
        assert result.usable is False
        assert "detection_dropout_note" in result.as_dict()

    def test_coverage_floor_is_the_boundary(self) -> None:
        assert self._result("x" * 90).usable is True
        assert self._result("x" * 89).usable is False
        assert DEFAULT_COVERAGE_FLOOR == pytest.approx(0.90)

    def test_summary_excludes_dropout_cells_from_error_statistics(self) -> None:
        good = self._result("x" * 100)
        dropped = self._result("x" * 30)
        summary = summarise_surface([good, dropped])
        assert summary["cells"] == 2
        assert summary["usable_cells"] == 1
        assert summary["detection_dropout_cells"] == 1
        assert summary["character_error_rate"]["max"] == pytest.approx(0.0)

    def test_summary_of_all_dropout_reports_no_error_statistics(self) -> None:
        summary = summarise_surface([self._result("x" * 10)])
        assert summary["usable_cells"] == 0
        assert "character_error_rate" not in summary
        assert "note" in summary

    def test_empty_summary(self) -> None:
        assert summarise_surface([])["cells"] == 0
