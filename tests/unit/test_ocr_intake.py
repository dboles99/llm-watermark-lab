"""OCR intake for images and scanned PDFs, and the detection gate.

The gate tests are the important ones: OCR-derived text must not be able to
reach a conclusive decision by accident, and the reasons it cannot must be
legible rather than implicit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from watermark_lab.ocr import build_render_settings, rapidocr_available, render_text_to_image
from watermark_lab.ocr.engine import get_engine
from watermark_lab.run_engine.ocr_intake import (
    DEFAULT_CONFIDENCE_FLOOR,
    IMAGE_MEDIA_TYPES,
    OcrIntakeError,
    OcrIntakeResult,
    PageText,
    ocr_detection_gate,
    read_image,
    read_scanned_source,
)

SAMPLE = (
    "Provenance evidence must survive ordinary document handling to be useful. "
    "A watermark that vanishes when a page is printed and scanned again cannot "
    "support any claim about where a document came from."
)

requires_rapidocr = pytest.mark.skipif(
    not rapidocr_available(), reason="rapidocr_onnxruntime is not installed"
)


def _result(
    *,
    fingerprint: str = "a" * 64,
    confidence: float = 0.95,
    text: str = "recovered text",
) -> OcrIntakeResult:
    page = PageText(
        page_number=1,
        text=text,
        mean_confidence=confidence,
        min_confidence=confidence,
        block_count=1,
        text_sha256="b" * 64,
        image_sha256="c" * 64,
    )
    return OcrIntakeResult(
        text=text,
        pages=(page,),
        media_type="image/png",
        source_path="/tmp/page.png",
        engine_id="stub",
        profile_fingerprint=fingerprint,
        language="en",
    )


def _matching_manifest(
    *,
    status: str = "frozen",
    engine_id: str = "stub",
    fingerprint: str = "a" * 64,
    language: str = "en",
) -> dict[str, object]:
    """A minimal OCR calibration manifest matching the stub reading."""

    return {
        "calibration_id": "cal_ocr_stub_en_v1",
        "status": status,
        "input_channel": {
            "channel": "ocr",
            "ocr": {
                "engine_id": engine_id,
                "engine_version": "1.2.3",
                "profile_fingerprint": fingerprint,
                "language": language,
                "confidence_floor": 0.8,
                "degradation": {"name": "clean", "severity": 0.0, "seed": 0},
            },
        },
    }


class TestDetectionGateFailsClosed:
    def test_missing_calibration_blocks_a_conclusive_decision(self) -> None:
        """The binding precondition today: no OCR null has been frozen."""

        gate = ocr_detection_gate(_result())
        assert gate.permitted_decision == "inconclusive"
        assert gate.calibration_available is False
        assert any("calibration null" in blocker for blocker in gate.blockers)

    def test_all_preconditions_met_permits_a_conclusion(self) -> None:
        gate = ocr_detection_gate(_result(), calibration_manifest=_matching_manifest())
        assert gate.permitted_decision == "conclusive"
        assert gate.blockers == ()

    def test_low_confidence_blocks_even_with_calibration(self) -> None:
        gate = ocr_detection_gate(
            _result(confidence=0.40), calibration_manifest=_matching_manifest()
        )
        assert gate.permitted_decision == "inconclusive"
        assert gate.confidence_ok is False
        assert any("below the" in blocker for blocker in gate.blockers)

    def test_unpinned_profile_blocks_even_with_calibration(self) -> None:
        gate = ocr_detection_gate(
            _result(fingerprint=""), calibration_manifest=_matching_manifest()
        )
        assert gate.permitted_decision == "inconclusive"
        assert gate.profile_pinned is False
        assert any("not pinned" in blocker for blocker in gate.blockers)

    def test_every_blocker_is_reported_not_just_the_first(self) -> None:
        gate = ocr_detection_gate(_result(fingerprint="", confidence=0.1))
        assert len(gate.blockers) == 3

    def test_confidence_floor_is_the_boundary(self) -> None:
        calibration = _matching_manifest()
        at_floor = ocr_detection_gate(
            _result(confidence=DEFAULT_CONFIDENCE_FLOOR), calibration_manifest=calibration
        )
        below = ocr_detection_gate(
            _result(confidence=DEFAULT_CONFIDENCE_FLOOR - 0.01), calibration_manifest=calibration
        )
        assert at_floor.permitted_decision == "conclusive"
        assert below.permitted_decision == "inconclusive"

    def test_gate_serialises_its_reasoning(self) -> None:
        payload = ocr_detection_gate(_result()).as_dict()
        assert payload["permitted_decision"] == "inconclusive"
        assert isinstance(payload["blockers"], list)


class TestResultProvenance:
    def test_result_declares_the_ocr_channel_and_reconstruction(self) -> None:
        payload = _result().as_dict()
        assert payload["provenance_channel"] == "ocr"
        assert "reconstructed" in payload["reconstruction_note"].lower()
        assert payload["text_sha256"]

    def test_pages_are_hashed_individually(self) -> None:
        payload = _result().as_dict()
        assert payload["page_count"] == 1
        assert payload["pages"][0]["image_sha256"]
        assert payload["pages"][0]["text_sha256"]

    def test_confidence_is_length_weighted_across_pages(self) -> None:
        """A long badly-read page must not be masked by a short well-read one."""

        pages = (
            PageText(1, "x" * 900, 0.50, 0.50, 1, "a" * 64, "b" * 64),
            PageText(2, "y" * 100, 1.00, 1.00, 1, "c" * 64, "d" * 64),
        )
        result = OcrIntakeResult(
            text="".join(page.text for page in pages),
            pages=pages,
            media_type="application/pdf",
            source_path="/tmp/x.pdf",
            engine_id="stub",
            profile_fingerprint="e" * 64,
            language="en",
        )
        assert result.mean_confidence == pytest.approx(0.55)
        assert result.min_confidence == pytest.approx(0.50)
        # A naive unweighted mean would be 0.75 and would clear the floor.
        assert result.mean_confidence < DEFAULT_CONFIDENCE_FLOOR

    def test_empty_pages_do_not_divide_by_zero(self) -> None:
        result = OcrIntakeResult(
            text="",
            pages=(),
            media_type="image/png",
            source_path="/tmp/x.png",
            engine_id="stub",
            profile_fingerprint="f" * 64,
            language="en",
        )
        assert result.mean_confidence == 0.0
        assert result.min_confidence == 0.0


class TestDispatch:
    def test_unsupported_extension_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.docx"
        target.write_bytes(b"not really a docx")
        with pytest.raises(OcrIntakeError, match="Unsupported scanned source"):
            read_scanned_source(str(target), engine=object())

    def test_missing_file_is_refused(self) -> None:
        with pytest.raises(OcrIntakeError, match="Not a readable file"):
            read_scanned_source("no/such/file.png", engine=object())

    def test_image_media_types_cover_common_scan_outputs(self) -> None:
        for suffix in (".png", ".jpg", ".tif"):
            assert suffix in IMAGE_MEDIA_TYPES


@requires_rapidocr
class TestRealImageIntake:
    def test_reads_a_rendered_page(self, tmp_path: Path) -> None:
        from PIL import Image

        settings = build_render_settings(SAMPLE, locale="en", font_size=30)
        page = render_text_to_image(SAMPLE, settings)
        target = tmp_path / "page.png"
        Image.fromarray(page).save(target)

        engine = get_engine("rapidocr")
        result = read_image(target, engine=engine, language="en")

        assert result.page_count == 1
        assert result.media_type == "image/png"
        assert result.engine_id == "rapidocr"
        assert result.profile_fingerprint == engine.profile.aggregate_sha256
        assert len(result.text) > 50
        assert result.mean_confidence > 0.5

    def test_even_a_good_reading_is_gated_to_inconclusive(self, tmp_path: Path) -> None:
        """A clean, high-confidence scan still cannot conclude without a null."""

        from PIL import Image

        settings = build_render_settings(SAMPLE, locale="en", font_size=30)
        Image.fromarray(render_text_to_image(SAMPLE, settings)).save(tmp_path / "p.png")
        result = read_image(tmp_path / "p.png", engine=get_engine("rapidocr"), language="en")

        gate = ocr_detection_gate(result)
        assert result.mean_confidence > DEFAULT_CONFIDENCE_FLOOR
        assert gate.confidence_ok is True
        assert gate.profile_pinned is True
        assert gate.permitted_decision == "inconclusive"

    def test_blank_page_yields_no_text_and_fails_the_gate(self, tmp_path: Path) -> None:
        from PIL import Image

        blank = np.full((400, 800, 3), 255, dtype=np.uint8)
        Image.fromarray(blank).save(tmp_path / "blank.png")
        result = read_image(tmp_path / "blank.png", engine=get_engine("rapidocr"), language="en")

        assert result.text.strip() == ""
        assert ocr_detection_gate(result).permitted_decision == "inconclusive"

    def test_corrupt_image_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "broken.png"
        target.write_bytes(b"this is not a png")
        with pytest.raises(OcrIntakeError, match="Could not open image"):
            read_image(target, engine=get_engine("rapidocr"))


class TestCalibrationMustMatchTheInstrument:
    """Presence of *some* OCR calibration is not enough.

    A null measured on one engine, one set of model bytes, or one language does
    not describe a reading produced by another. Accepting a bare identifier would
    let a mismatched null certify a reading, understating the false-positive rate
    in exactly the direction that manufactures a false accusation.
    """

    def test_wrong_engine_is_refused(self) -> None:
        gate = ocr_detection_gate(
            _result(), calibration_manifest=_matching_manifest(engine_id="tesseract")
        )
        assert gate.permitted_decision == "inconclusive"
        assert gate.calibration_available is False
        assert any("does not match reading engine" in b for b in gate.blockers)

    def test_wrong_profile_fingerprint_is_refused(self) -> None:
        """Different model bytes or parameters mean a different instrument."""

        gate = ocr_detection_gate(
            _result(), calibration_manifest=_matching_manifest(fingerprint="9" * 64)
        )
        assert gate.permitted_decision == "inconclusive"
        assert any("different instrument" in b for b in gate.blockers)

    def test_wrong_language_is_refused(self) -> None:
        """OCR error profiles are language-specific."""

        gate = ocr_detection_gate(
            _result(), calibration_manifest=_matching_manifest(language="zh-Hant-TW")
        )
        assert gate.permitted_decision == "inconclusive"
        assert any("language-specific" in b for b in gate.blockers)

    @pytest.mark.parametrize("status", ["draft", "retired"])
    def test_unfrozen_calibration_is_refused(self, status: str) -> None:
        """A draft calibration is still being tuned; a moving threshold is not one."""

        gate = ocr_detection_gate(
            _result(), calibration_manifest=_matching_manifest(status=status)
        )
        assert gate.permitted_decision == "inconclusive"
        assert any("frozen or" in b for b in gate.blockers)

    def test_clean_text_calibration_does_not_certify_an_ocr_reading(self) -> None:
        clean = {"calibration_id": "cal_clean_en_v1", "status": "frozen",
                 "input_channel": {"channel": "clean-text"}}
        gate = ocr_detection_gate(_result(), calibration_manifest=clean)
        assert gate.permitted_decision == "inconclusive"
        assert any("not an OCR calibration" in b for b in gate.blockers)

    def test_calibration_without_a_channel_block_is_refused(self) -> None:
        gate = ocr_detection_gate(
            _result(), calibration_manifest={"calibration_id": "cal_x", "status": "frozen"}
        )
        assert gate.permitted_decision == "inconclusive"
