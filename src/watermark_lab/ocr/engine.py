"""OCR engine protocol and registry.

Engines are addressed by id so a run manifest can record which instrument
produced a reading. Registration is lazy and failure is explicit: an engine whose
dependencies are missing raises on construction rather than silently degrading to
a different engine, because a silent substitution would change the measurement
without changing the recorded profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from watermark_lab.ocr.profile import OcrProfile, OcrProfileError
from watermark_lab.ocr.result import OcrResult

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@runtime_checkable
class OcrEngine(Protocol):
    """A pinned optical character recognition backend."""

    engine_id: str

    @property
    def profile(self) -> OcrProfile:
        """Return the pinned profile identifying this engine and its assets."""
        ...

    def recognize(self, image: np.ndarray, **kwargs: Any) -> OcrResult:
        """Recognise text in an RGB image array."""
        ...


_ENGINE_BUILDERS: dict[str, Any] = {}


def register_engine(engine_id: str, builder: Any) -> None:
    """Register a zero-argument-friendly builder for ``engine_id``."""

    _ENGINE_BUILDERS[engine_id] = builder


def available_engine_ids() -> tuple[str, ...]:
    """Return every registered engine id, whether or not it can be constructed."""

    _ensure_builtin_engines()
    return tuple(sorted(_ENGINE_BUILDERS))


def get_engine(engine_id: str, **kwargs: Any) -> OcrEngine:
    """Construct a registered engine, or raise ``OcrProfileError``."""

    _ensure_builtin_engines()
    builder = _ENGINE_BUILDERS.get(engine_id)
    if builder is None:
        known = ", ".join(sorted(_ENGINE_BUILDERS)) or "none"
        raise OcrProfileError(f"Unknown OCR engine {engine_id!r}. Registered engines: {known}.")
    return builder(**kwargs)  # type: ignore[no-any-return]


def _ensure_builtin_engines() -> None:
    if _ENGINE_BUILDERS:
        return
    from watermark_lab.ocr.rapidocr import RapidOcrEngine
    from watermark_lab.ocr.tesseract import TesseractEngine

    register_engine("rapidocr", RapidOcrEngine)
    register_engine("tesseract", TesseractEngine)
