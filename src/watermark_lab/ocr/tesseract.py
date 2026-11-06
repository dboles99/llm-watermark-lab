"""Tesseract engine adapter.

Tesseract is the second profile. It exists because engine identity is a
confounder: a survival surface measured with one engine describes that engine's
error profile as much as the watermark's robustness, and a result that holds
across two independently built engines is a far stronger claim than one that
holds for a single ONNX model family.

Tesseract is a system binary, not a Python package. It is deliberately *not*
added to the project's runtime dependencies, and this adapter fails closed with
an actionable message when the binary or its language data is absent — rather
than falling back to another engine, which would change the instrument without
changing the recorded profile.

Traditional Chinese requires the ``chi_tra`` traineddata (and ``chi_tra_vert``
for vertical text). Requesting ``chi_tra`` when only ``chi_sim`` is installed is
treated as an error, because reading Taiwan text with a Simplified model is the
contamination path the project forbids.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from watermark_lab.ocr.profile import (
    AssetFingerprint,
    OcrProfile,
    OcrProfileError,
    build_profile,
    fingerprint_assets,
    tesseract_version,
)
from watermark_lab.ocr.result import OcrResult, TextBlock, sort_reading_order

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

_TSV_COLUMNS = 12
_TESSDATA_ENV = "TESSDATA_PREFIX"


class TesseractEngine:
    """Tesseract OCR with pinned traineddata fingerprints."""

    engine_id: str = "tesseract"

    def __init__(
        self,
        *,
        language: str = "eng",
        psm: int = 3,
        oem: int = 3,
        tessdata_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        binary = shutil.which("tesseract")
        if binary is None:
            raise OcrProfileError(
                "Tesseract is not installed or not on PATH.\n"
                "  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Debian/Ubuntu: apt install tesseract-ocr tesseract-ocr-chi-tra\n"
                "  macOS: brew install tesseract tesseract-lang\n"
                "Then ensure the traineddata for your language is present "
                "('eng', 'chi_tra' for Taiwan Traditional Chinese). "
                "Alternatively select the 'rapidocr' engine profile, which needs no binary."
            )

        self._binary = binary
        self.language = language
        self.psm = psm
        self.oem = oem
        self._tessdata_dir = self._resolve_tessdata_dir(tessdata_dir)
        self._verify_language_available()
        self._profile = self._build_profile()

    def _resolve_tessdata_dir(self, explicit: str | os.PathLike[str] | None) -> Path | None:
        if explicit is not None:
            path = Path(explicit)
            if not path.is_dir():
                raise OcrProfileError(f"tessdata_dir does not exist: {path}")
            return path
        env_value = os.environ.get(_TESSDATA_ENV)
        if env_value:
            candidate = Path(env_value)
            if (candidate / "tessdata").is_dir():
                return candidate / "tessdata"
            if candidate.is_dir():
                return candidate
        binary_parent = Path(self._binary).resolve().parent
        for candidate in (binary_parent / "tessdata", binary_parent.parent / "share" / "tessdata"):
            if candidate.is_dir():
                return candidate
        return None

    def _installed_languages(self) -> tuple[str, ...]:
        try:
            completed = subprocess.run(
                [self._binary, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
            raise OcrProfileError(f"Could not list Tesseract languages: {exc}") from exc
        output = completed.stdout or completed.stderr or ""
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return tuple(line for line in lines if not line.lower().startswith("list of"))

    def _verify_language_available(self) -> None:
        requested = {part for part in self.language.split("+") if part}
        installed = set(self._installed_languages())
        missing = sorted(requested - installed)
        if not missing:
            return
        hint = ""
        if any(code.startswith("chi_tra") for code in missing) and "chi_sim" in installed:
            hint = (
                " Simplified Chinese ('chi_sim') is installed but Traditional is not. "
                "Reading Taiwan text with a Simplified model is not an acceptable "
                "substitution: install 'chi_tra' instead."
            )
        raise OcrProfileError(
            f"Tesseract traineddata missing for: {', '.join(missing)}. "
            f"Installed: {', '.join(sorted(installed)) or 'none'}.{hint}"
        )

    def _build_profile(self) -> OcrProfile:
        assets: tuple[AssetFingerprint, ...] = ()
        if self._tessdata_dir is not None:
            requested = [part for part in self.language.split("+") if part]
            files = [
                path
                for code in requested
                if (path := self._tessdata_dir / f"{code}.traineddata").is_file()
            ]
            assets = fingerprint_assets(files)

        return build_profile(
            engine_id=self.engine_id,
            engine_version=tesseract_version(),
            language=self.language,
            assets=assets,
            parameters={
                "psm": self.psm,
                "oem": self.oem,
                "tessdata_dir": str(self._tessdata_dir) if self._tessdata_dir else None,
            },
        )

    @property
    def profile(self) -> OcrProfile:
        return self._profile

    def recognize(self, image: np.ndarray, **kwargs: Any) -> OcrResult:
        """Recognise text via a TSV run, preserving per-word confidence."""

        from PIL import Image

        warnings: list[str] = []
        if not self._profile.assets:
            warnings.append(
                "traineddata files could not be located, so the profile fingerprint does not "
                "pin the language model; results are not fully reproducible"
            )

        with tempfile.TemporaryDirectory(prefix="lab-ocr-") as tmp:
            image_path = Path(tmp) / "page.png"
            Image.fromarray(image).save(image_path)
            command = [
                self._binary,
                str(image_path),
                "stdout",
                "-l",
                self.language,
                "--psm",
                str(self.psm),
                "--oem",
                str(self.oem),
                "tsv",
            ]
            if self._tessdata_dir is not None:
                command[3:3] = ["--tessdata-dir", str(self._tessdata_dir)]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=float(kwargs.get("timeout", 300.0)),
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return OcrResult(
                    text="",
                    blocks=(),
                    engine_id=self.engine_id,
                    profile_fingerprint=self._profile.aggregate_sha256,
                    language=self.language,
                    warnings=(f"recognition failed: {exc}",),
                )

        if completed.returncode != 0:
            warnings.append(f"tesseract exited {completed.returncode}: {completed.stderr.strip()}")

        blocks = sort_reading_order(_parse_tsv(completed.stdout or ""))
        if not blocks:
            warnings.append("no text regions were detected")

        return OcrResult(
            text=" ".join(block.text for block in blocks),
            blocks=blocks,
            engine_id=self.engine_id,
            profile_fingerprint=self._profile.aggregate_sha256,
            language=self.language,
            warnings=tuple(warnings),
            metadata={"detection_count": len(blocks)},
        )


def _parse_tsv(payload: str) -> tuple[TextBlock, ...]:
    """Parse Tesseract TSV output into confidence-bearing word blocks.

    Tesseract reports ``conf = -1`` for layout rows that carry no text; those are
    structural markers, not low-confidence readings, and are dropped rather than
    counted as zero-confidence words.
    """

    lines = payload.splitlines()
    if len(lines) < 2:
        return ()

    blocks: list[TextBlock] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < _TSV_COLUMNS:
            continue
        text = fields[11].strip()
        if not text:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        if confidence < 0:
            continue
        try:
            left, top, width, height = (float(fields[i]) for i in range(6, 10))
            box: tuple[float, float, float, float] | None = (
                left,
                top,
                left + width,
                top + height,
            )
        except ValueError:
            box = None
        blocks.append(TextBlock(text=text, confidence=confidence / 100.0, box=box))

    return tuple(blocks)
