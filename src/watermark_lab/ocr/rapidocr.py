"""RapidOCR (ONNX PP-OCR) engine adapter.

RapidOCR is the default profile because it is pure Python, runs fully offline,
and ships its ONNX models as ordinary files inside the installed package — so
those models can be hashed and pinned exactly like a tokenizer's assets. There
is no system binary and no network access at recognition time.

The bundled ``ch_PP-OCRv3`` models are trained principally on Simplified Chinese
and Latin script. Their accuracy on Taiwan Traditional Chinese is an open
empirical question and must be measured per corpus, never assumed: an engine that
silently prefers Simplified forms would introduce exactly the ``zh-CN``
contamination the project forbids.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from watermark_lab.ocr.profile import (
    OcrProfile,
    OcrProfileError,
    build_profile,
    fingerprint_assets,
)
from watermark_lab.ocr.result import OcrResult, TextBlock, sort_reading_order

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

RAPIDOCR_TRADITIONAL_CHINESE_CAVEAT = (
    "The bundled ch_PP-OCRv3 models are trained principally on Simplified Chinese and Latin "
    "script. Traditional Chinese accuracy is unmeasured here and must not be assumed; "
    "recognised zh output requires Simplified-form contamination screening."
)


class RapidOcrEngine:
    """Offline ONNX OCR with pinned, hashed model assets."""

    engine_id: str = "rapidocr"

    def __init__(self, *, language: str = "auto", **engine_kwargs: Any) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - env dependent
            raise OcrProfileError(
                "rapidocr_onnxruntime is not installed. Install it, or select a different "
                "OCR engine profile."
            ) from exc

        self.language = language
        self._engine_kwargs = dict(engine_kwargs)
        try:
            self._reader = RapidOCR(**engine_kwargs)
        except Exception as exc:  # pragma: no cover - env dependent
            raise OcrProfileError(f"Could not initialise RapidOCR: {exc}") from exc
        self._profile = self._build_profile()

    def _build_profile(self) -> OcrProfile:
        import rapidocr_onnxruntime  # type: ignore[import-untyped]

        package_root = Path(rapidocr_onnxruntime.__file__).parent
        model_files = sorted(package_root.rglob("*.onnx"))
        if not model_files:
            raise OcrProfileError(
                f"No .onnx model files found under {package_root}. The RapidOCR install "
                "appears incomplete, so no reproducible profile can be pinned."
            )

        version = getattr(rapidocr_onnxruntime, "__version__", None)
        if not version:
            try:
                from importlib import metadata

                version = metadata.version("rapidocr_onnxruntime")
            except Exception:  # pragma: no cover - env dependent
                version = "unknown"

        return build_profile(
            engine_id=self.engine_id,
            engine_version=str(version),
            language=self.language,
            assets=fingerprint_assets(model_files, root=package_root),
            parameters=dict(self._engine_kwargs),
        )

    @property
    def profile(self) -> OcrProfile:
        return self._profile

    def recognize(self, image: np.ndarray, **kwargs: Any) -> OcrResult:
        """Recognise text in an RGB image array.

        Blocks arrive in detection order, which is not reading order, so they are
        re-sorted top-to-bottom and left-to-right before joining. Skipping that
        step inflates the measured error rate on a clean render and makes the
        degradation ladder non-monotone, which reads as watermark fragility that
        the degradation never caused.
        """

        try:
            detections, _elapsed = self._reader(image, **kwargs)
        except Exception as exc:
            return OcrResult(
                text="",
                blocks=(),
                engine_id=self.engine_id,
                profile_fingerprint=self._profile.aggregate_sha256,
                language=self.language,
                warnings=(f"recognition failed: {exc}",),
            )

        warnings: list[str] = [RAPIDOCR_TRADITIONAL_CHINESE_CAVEAT]
        blocks: list[TextBlock] = []

        for detection in detections or ():
            box, text, confidence = _unpack_detection(detection)
            if text is None:
                continue
            blocks.append(
                TextBlock(text=text, confidence=confidence, box=box)
            )

        ordered = sort_reading_order(tuple(blocks))
        if not ordered:
            warnings.append("no text regions were detected")
        elif any(block.box is None for block in ordered):
            warnings.append(
                "some blocks carried no geometry and could not be placed in reading order"
            )

        return OcrResult(
            text="\n".join(block.text for block in ordered),
            blocks=ordered,
            engine_id=self.engine_id,
            profile_fingerprint=self._profile.aggregate_sha256,
            language=self.language,
            warnings=tuple(warnings),
            metadata={"detection_count": len(ordered)},
        )


def _unpack_detection(
    detection: Any,
) -> tuple[tuple[float, float, float, float] | None, str | None, float]:
    """Normalise one RapidOCR detection into (box, text, confidence).

    RapidOCR returns ``[polygon, text, confidence]`` where the polygon is four
    corner points and the confidence arrives as a string. Both are normalised
    here so downstream code never parses engine-specific shapes.
    """

    try:
        polygon, text, raw_confidence = detection[0], detection[1], detection[2]
    except (TypeError, IndexError, KeyError):
        return None, None, 0.0

    if not isinstance(text, str):
        return None, None, 0.0

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    box: tuple[float, float, float, float] | None = None
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        box = (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError, IndexError):
        box = None

    return box, text, confidence
