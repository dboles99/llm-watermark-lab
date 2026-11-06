"""Intake for images and scanned PDFs, via OCR.

Text recovered from a scan is a *reconstruction*, not a copy of any source bytes,
and the reconstruction can be wrong in ways that look exactly like watermark
evidence or exactly like its absence. Everything here therefore carries the
instrument that produced it — engine, pinned profile fingerprint, per-page
confidence — and refuses to hand back a bare string that downstream code could
mistake for a file read.

Governance rule, enforced by ``ocr_detection_gate``: detection on OCR-derived
text may return at most ``inconclusive`` unless the profile is pinned, mean
character confidence clears a preregistered floor, **and** the calibration
manifest carries an OCR-specific null for that profile, locale, and degradation
level. Reusing a clean-text threshold on reconstructed text is how a false
accusation gets manufactured, so the gate fails closed by default: no OCR null
exists yet for any profile, so today every OCR input is `inconclusive`.

Pages are hashed individually. A multi-page scan is not one artifact — pages get
reordered, dropped, and re-scanned independently, and a document-level hash hides
all of that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watermark_lab.ocr.result import OCR_RECONSTRUCTION_NOTE, PROVENANCE_CHANNEL

IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

#: Preregistered floor. Below this mean character confidence, an OCR reading is
#: not usable as detection input regardless of anything else.
DEFAULT_CONFIDENCE_FLOOR = 0.80

#: Rendering resolution for PDF pages. Recorded because it changes the character
#: error rate as much as any degradation does.
DEFAULT_PDF_DPI = 300


class OcrIntakeError(RuntimeError):
    """Raised when a scanned source cannot be ingested reproducibly."""


@dataclass(frozen=True, slots=True)
class PageText:
    """One page of recovered text with its own provenance."""

    page_number: int
    text: str
    mean_confidence: float
    min_confidence: float
    block_count: int
    text_sha256: str
    image_sha256: str
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text_sha256": self.text_sha256,
            "image_sha256": self.image_sha256,
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "block_count": self.block_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class OcrIntakeResult:
    """Text recovered from a scanned source, with the instrument recorded."""

    text: str
    pages: tuple[PageText, ...]
    media_type: str
    source_path: str
    engine_id: str
    profile_fingerprint: str
    language: str
    provenance_channel: str = PROVENANCE_CHANNEL
    render_dpi: int | None = None
    warnings: tuple[str, ...] = field(default=())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def mean_confidence(self) -> float:
        """Length-weighted mean confidence across pages."""

        weighted = 0.0
        total = 0
        for page in self.pages:
            length = len(page.text)
            if length:
                weighted += page.mean_confidence * length
                total += length
        return weighted / total if total else 0.0

    @property
    def min_confidence(self) -> float:
        return min((page.min_confidence for page in self.pages), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance_channel": self.provenance_channel,
            "media_type": self.media_type,
            "source_path": self.source_path,
            "engine_id": self.engine_id,
            "profile_fingerprint": self.profile_fingerprint,
            "language": self.language,
            "render_dpi": self.render_dpi,
            "page_count": self.page_count,
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "pages": [page.as_dict() for page in self.pages],
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "reconstruction_note": OCR_RECONSTRUCTION_NOTE,
            "warnings": list(self.warnings),
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _recognise_page(engine: Any, image: Any, page_number: int) -> PageText:
    import numpy as np

    result = engine.recognize(image)
    array = np.ascontiguousarray(image)
    return PageText(
        page_number=page_number,
        text=result.text,
        mean_confidence=result.mean_confidence,
        min_confidence=result.min_confidence,
        block_count=result.block_count,
        text_sha256=result.text_sha256,
        image_sha256=_sha256_bytes(array.tobytes()),
        warnings=result.warnings,
    )


def read_image(path: Path, *, engine: Any, language: str = "auto") -> OcrIntakeResult:
    """Recognise a single image file."""

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as handle:
            image = np.array(handle.convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        raise OcrIntakeError(f"Could not open image {path}: {exc}") from exc

    page = _recognise_page(engine, image, 1)
    return OcrIntakeResult(
        text=page.text,
        pages=(page,),
        media_type=IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        source_path=str(path.resolve()),
        engine_id=engine.engine_id,
        profile_fingerprint=engine.profile.aggregate_sha256,
        language=language,
    )


def read_scanned_pdf(
    path: Path,
    *,
    engine: Any,
    language: str = "auto",
    dpi: int = DEFAULT_PDF_DPI,
    max_pages: int | None = None,
) -> OcrIntakeResult:
    """Rasterise a PDF and recognise each page.

    This is the *scanned* path and it deliberately ignores any embedded text
    layer. A PDF can carry both a scanned image and an OCR text layer produced by
    unknown software at unknown quality; silently preferring that layer would
    mean the recorded engine profile describes an instrument that never ran.
    Use the ordinary text-PDF intake when an authored text layer is wanted.
    """

    try:
        import pymupdf  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - env dependent
        try:
            import fitz as pymupdf  # type: ignore[import-untyped,no-redef]
        except ImportError as exc:
            raise OcrIntakeError(
                "PyMuPDF is required to rasterise scanned PDFs. Install the 'ocr' extra."
            ) from exc

    import numpy as np

    warnings: list[str] = []
    pages: list[PageText] = []
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise OcrIntakeError(f"Could not open PDF {path}: {exc}") from exc

    try:
        total = document.page_count
        limit = total if max_pages is None else min(total, max_pages)
        if limit < total:
            warnings.append(f"only the first {limit} of {total} pages were processed")
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for index in range(limit):
            pixmap = document.load_page(index).get_pixmap(matrix=matrix, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            if pixmap.n == 4:  # pragma: no cover - alpha=False makes this unlikely
                image = image[:, :, :3]
            pages.append(_recognise_page(engine, image, index + 1))
    finally:
        document.close()

    if not pages:
        warnings.append("the PDF contained no pages")

    return OcrIntakeResult(
        text="\n\n".join(page.text for page in pages),
        pages=tuple(pages),
        media_type="application/pdf",
        source_path=str(path.resolve()),
        engine_id=engine.engine_id,
        profile_fingerprint=engine.profile.aggregate_sha256,
        language=language,
        render_dpi=dpi,
        warnings=tuple(warnings),
    )


def read_scanned_source(
    path_or_text: str,
    *,
    engine: Any,
    language: str = "auto",
    dpi: int = DEFAULT_PDF_DPI,
    max_pages: int | None = None,
) -> OcrIntakeResult:
    """Dispatch a path to the image or scanned-PDF reader."""

    path = Path(path_or_text)
    if not (path.exists() and path.is_file()):
        raise OcrIntakeError(f"Not a readable file: {path_or_text!r}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_scanned_pdf(
            path, engine=engine, language=language, dpi=dpi, max_pages=max_pages
        )
    if suffix in IMAGE_MEDIA_TYPES:
        return read_image(path, engine=engine, language=language)
    raise OcrIntakeError(
        f"Unsupported scanned source type {suffix!r}. Supported: .pdf and "
        + ", ".join(sorted(IMAGE_MEDIA_TYPES))
    )


@dataclass(frozen=True, slots=True)
class OcrDetectionGate:
    """Whether an OCR reading may be used as detection input, and why not."""

    permitted_decision: str
    profile_pinned: bool
    confidence_ok: bool
    calibration_available: bool
    mean_confidence: float
    confidence_floor: float
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "permitted_decision": self.permitted_decision,
            "profile_pinned": self.profile_pinned,
            "confidence_ok": self.confidence_ok,
            "calibration_available": self.calibration_available,
            "mean_confidence": self.mean_confidence,
            "confidence_floor": self.confidence_floor,
            "blockers": list(self.blockers),
        }


def ocr_detection_gate(
    result: OcrIntakeResult,
    *,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    calibration_manifest: dict[str, Any] | None = None,
) -> OcrDetectionGate:
    """Decide the strongest conclusion an OCR reading may support.

    Returns ``"inconclusive"`` unless every precondition holds.

    The calibration precondition takes a **manifest**, not an identifier, and
    checks that the manifest's OCR profile matches this reading's engine, pinned
    model bytes, and language. Accepting a bare identifier would let a null
    measured on one instrument certify a reading produced by another, which
    understates the false-positive rate in exactly the direction that
    manufactures a false accusation.

    A manifest must also be ``frozen`` or ``validated``. A draft calibration is
    still being tuned, and thresholds selected on data that is still moving are
    not thresholds.
    """

    blockers: list[str] = []

    profile_pinned = bool(result.profile_fingerprint) and len(result.profile_fingerprint) == 64
    if not profile_pinned:
        blockers.append(
            "the OCR engine profile is not pinned, so the reading is not reproducible"
        )

    confidence_ok = result.mean_confidence >= confidence_floor
    if not confidence_ok:
        blockers.append(
            f"mean character confidence {result.mean_confidence:.3f} is below the "
            f"preregistered floor of {confidence_floor:.2f}"
        )

    calibration_available = False
    if calibration_manifest is None:
        blockers.append(
            "no OCR-specific calibration null was supplied for this engine profile, "
            "locale, and degradation level; clean-text thresholds do not transfer to "
            "reconstructed text"
        )
    else:
        status = calibration_manifest.get("status")
        if status not in {"frozen", "validated"}:
            blockers.append(
                f"the supplied calibration has status {status!r}; only a frozen or "
                "validated calibration may support a conclusion"
            )
        else:
            from watermark_lab.statistics.manifest import ocr_channel_matches

            matches, reason = ocr_channel_matches(
                calibration_manifest,
                engine_id=result.engine_id,
                profile_fingerprint=result.profile_fingerprint,
                language=result.language,
            )
            calibration_available = matches
            if not matches:
                blockers.append(reason)

    permitted = "conclusive" if not blockers else "inconclusive"
    return OcrDetectionGate(
        permitted_decision=permitted,
        profile_pinned=profile_pinned,
        confidence_ok=confidence_ok,
        calibration_available=calibration_available,
        mean_confidence=result.mean_confidence,
        confidence_floor=confidence_floor,
        blockers=tuple(blockers),
    )


__all__ = [
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_PDF_DPI",
    "IMAGE_MEDIA_TYPES",
    "OcrDetectionGate",
    "OcrIntakeError",
    "OcrIntakeResult",
    "PageText",
    "ocr_detection_gate",
    "read_image",
    "read_scanned_pdf",
    "read_scanned_source",
]
