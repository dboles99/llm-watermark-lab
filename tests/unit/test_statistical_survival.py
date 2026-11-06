"""Statistical watermark survival through a print-scan-OCR round trip.

Most tests here use stub scorers and tokenizers, because the logic under test is
the bookkeeping: dropout exclusion, token-error accounting, and whether the
context-width prediction is reported honestly when it fails. The one integration
test runs the real pipeline and is skipped when the pinned model is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from watermark_lab.ocr import Degradation, build_render_settings, rapidocr_available
from watermark_lab.ocr.engine import get_engine
from watermark_lab.ocr.statistical_survival import (
    StatisticalSurvivalCell,
    compare_context_widths,
    measure_statistical_survival,
    predicted_context_corruption,
    summarise_statistical_survival,
    token_error_rate,
)

PINNED_MODEL = (
    Path(os.environ.get("LOCALAPPDATA", "/nonexistent"))
    / "watermark-lab-lab"
    / "assets"
    / "openai-community-gpt2"
    / "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)


def _cell(
    *,
    name: str = "jpeg",
    dropout: bool = False,
    ref: float | None = 9.0,
    rec: float | None = 8.0,
    detected: bool | None = True,
    width: int = 1,
    cer: float = 0.01,
    ter: float = 0.05,
) -> StatisticalSurvivalCell:
    return StatisticalSurvivalCell(
        degradation={"name": name, "severity": 0.5, "seed": 0},
        character_error_rate=cer,
        word_error_rate=cer,
        token_error_rate=ter,
        reference_tokens=200,
        recovered_tokens=195,
        reference_score=ref,
        recovered_score=rec,
        reference_detected=True,
        recovered_detected=detected,
        coverage_ratio=0.5 if dropout else 0.99,
        detection_dropout=dropout,
        context_width=width,
        predicted_corruption=predicted_context_corruption(ter, width),
    )


class TestTokenErrorRate:
    def test_identical_sequences_score_zero(self) -> None:
        assert token_error_rate([1, 2, 3], [1, 2, 3]) == 0.0

    def test_empty_reference(self) -> None:
        assert token_error_rate([], []) == 0.0
        assert token_error_rate([], [1]) == 1.0

    def test_full_replacement_scores_one(self) -> None:
        assert token_error_rate([1, 2, 3], [4, 5, 6]) == pytest.approx(1.0)

    def test_single_substitution(self) -> None:
        assert token_error_rate([1, 2, 3, 4], [1, 9, 3, 4]) == pytest.approx(0.25)


class TestContextCorruptionPrediction:
    def test_wider_context_predicts_more_corruption(self) -> None:
        narrow = predicted_context_corruption(0.10, 1)
        wide = predicted_context_corruption(0.10, 4)
        assert narrow == pytest.approx(0.20)
        assert wide == pytest.approx(0.50)
        assert wide > narrow

    def test_saturates_at_one(self) -> None:
        assert predicted_context_corruption(0.5, 4) == 1.0

    def test_zero_error_predicts_zero(self) -> None:
        assert predicted_context_corruption(0.0, 4) == 0.0

    def test_rejects_negative_width(self) -> None:
        with pytest.raises(ValueError):
            predicted_context_corruption(0.1, -1)


class TestSummaryExcludesDropout:
    def test_dropout_cells_are_not_scored_as_failures(self) -> None:
        """A detector failure must not be counted as watermark loss."""

        cells = [_cell(), _cell(name="blur", dropout=True, rec=None, detected=None)]
        summary = summarise_statistical_survival(cells)
        assert summary["cells"] == 2
        assert summary["usable_cells"] == 1
        assert summary["detection_dropout_cells"] == 1
        assert summary["scored_cells"] == 1
        assert summary["survival_rate"] == pytest.approx(1.0)

    def test_survival_rate_counts_only_scored_cells(self) -> None:
        cells = [_cell(detected=True), _cell(name="noise", detected=False)]
        summary = summarise_statistical_survival(cells)
        assert summary["survival_rate"] == pytest.approx(0.5)

    def test_score_retained_is_a_ratio_of_reference(self) -> None:
        cell = _cell(ref=10.0, rec=8.0)
        assert cell.score_retained == pytest.approx(0.8)

    def test_score_retained_is_none_without_both_scores(self) -> None:
        assert _cell(rec=None).score_retained is None
        assert _cell(ref=None).score_retained is None
        assert _cell(ref=0.0).score_retained is None

    def test_empty_input(self) -> None:
        assert summarise_statistical_survival([])["cells"] == 0

    def test_summary_declares_it_is_not_calibrated(self) -> None:
        summary = summarise_statistical_survival([_cell()])
        assert "not a calibrated detection" in summary["not_calibrated"]


class TestContextWidthComparison:
    def test_reports_supported_when_narrower_survives_better(self) -> None:
        result = compare_context_widths(
            {1: {"survival_rate": 0.9}, 4: {"survival_rate": 0.4}}
        )
        assert result["verdict"] == "supported"

    def test_reports_not_supported_when_the_prediction_fails(self) -> None:
        """A falsified prediction must be reported, not hidden."""

        result = compare_context_widths(
            {1: {"survival_rate": 0.3}, 4: {"survival_rate": 0.9}}
        )
        assert result["verdict"] == "not-supported"

    def test_flat_result_is_distinguished_from_support(self) -> None:
        result = compare_context_widths(
            {1: {"survival_rate": 0.7}, 4: {"survival_rate": 0.7}}
        )
        assert result["verdict"] == "consistent-but-flat"

    def test_single_width_is_insufficient(self) -> None:
        assert compare_context_widths({1: {"survival_rate": 0.9}})["verdict"] == (
            "insufficient-data"
        )

    def test_unscorable_widths_are_insufficient(self) -> None:
        result = compare_context_widths(
            {1: {"survival_rate": None}, 4: {"survival_rate": 0.5}}
        )
        assert result["verdict"] == "insufficient-data"


@pytest.mark.skipif(
    not rapidocr_available() or not (PINNED_MODEL / "model.safetensors").is_file(),
    reason="requires rapidocr and the pinned GPT-2 weights",
)
@pytest.mark.integration
class TestRealPipeline:
    def test_statistical_watermark_survives_ocr(self) -> None:
        """The measurement the OCR track exists to make.

        Container-bound channels record 0% recovery at every severity. If the
        statistical channel survives the same round trip, a statistical soft
        binding is necessary rather than merely available.
        """

        import transformers

        from watermark_lab.run_engine.real_generate import PinnedWatermarkGenerator

        generator = PinnedWatermarkGenerator(PINNED_MODEL)
        generated = generator.generate(
            "The question of where a document came from",
            max_new_tokens=180,
            seed=11,
            temperature=0.9,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(PINNED_MODEL), local_files_only=True
        )
        detector = transformers.WatermarkDetector(
            model_config=transformers.AutoConfig.from_pretrained(
                str(PINNED_MODEL), local_files_only=True
            ),
            device="cpu",
            watermarking_config=transformers.WatermarkingConfig(
                **generator.watermarking_config
            ),
        )

        def scorer(text: str) -> tuple[float | None, bool | None]:
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            if ids.shape[-1] < 10:
                return None, None
            outcome = detector(ids, return_dict=True)
            return float(outcome.z_score[0]), bool(outcome.prediction[0])

        class _Tok:
            def encode(self, text: str) -> list[int]:
                return list(tokenizer.encode(text))

        settings = build_render_settings(generated.text, locale="en", font_size=30)
        cells = measure_statistical_survival(
            generated.text,
            engine=get_engine("rapidocr"),
            settings=settings,
            degradations=[
                Degradation("jpeg", 0.6, 3),
                Degradation("downsample", 0.5, 3),
            ],
            scorer=scorer,
            tokenizer=_Tok(),
            context_width=1,
        )

        summary = summarise_statistical_survival(cells)
        assert summary["reference_detected"] is True

        if summary["scored_cells"] == 0:
            # Every cell hit detector dropout. That is a property of the OCR
            # engine's line recall on this particular page, not a result about
            # the watermark, so it must not be asserted either way.
            pytest.skip(
                "all cells hit detector dropout; no watermark measurement was possible"
            )

        # The signal degrades but is not annihilated, unlike container-bound
        # channels, which record zero recovery at every severity.
        assert summary["survival_rate"] is not None
        assert summary["mean_score_retained"] is not None
        assert summary["mean_score_retained"] > 0.5

    def test_token_error_exceeds_character_error(self) -> None:
        """One misread character can re-segment a whole word into new tokens.

        This is why token error rate, not character error rate, is the quantity
        the watermark cares about — and why the relationship between them is
        expected to differ by locale.
        """

        import transformers

        from watermark_lab.run_engine.real_generate import PinnedWatermarkGenerator

        generator = PinnedWatermarkGenerator(PINNED_MODEL)
        generated = generator.generate("Provenance evidence", max_new_tokens=120, seed=5)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(PINNED_MODEL), local_files_only=True
        )

        class _Tok:
            def encode(self, text: str) -> list[int]:
                return list(tokenizer.encode(text))

        settings = build_render_settings(generated.text, locale="en", font_size=30)
        cells = measure_statistical_survival(
            generated.text,
            engine=get_engine("rapidocr"),
            settings=settings,
            degradations=[Degradation("jpeg", 0.6, 3)],
            scorer=lambda text: (None, None),
            tokenizer=_Tok(),
        )
        usable = [cell for cell in cells if not cell.detection_dropout]
        if not usable:
            pytest.skip("the degradation produced only detector dropout")
        assert usable[0].token_error_rate >= usable[0].character_error_rate
