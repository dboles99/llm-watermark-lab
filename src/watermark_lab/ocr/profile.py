"""Pinned OCR engine profiles.

An OCR engine is an experimental instrument, not a utility. Two engines — or the
same engine with different models — read the same page differently, and the
difference lands directly on the watermark statistic being measured. Engine
identity is therefore a confounder that must be recorded and varied, exactly as
tokenizer identity is.

This module mirrors ``watermark_lab.tokenization.huggingface``: model files
are hashed individually, then combined into an aggregate fingerprint over a
canonical manifest. A profile whose files have changed produces a different
fingerprint, and any result carrying the old fingerprint is no longer reproducible
by the current install.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HASH_CHUNK_BYTES = 1 << 20


class OcrProfileError(RuntimeError):
    """Raised when an OCR profile cannot be pinned or is unavailable."""


@dataclass(frozen=True, slots=True)
class AssetFingerprint:
    """One pinned model or data file."""

    name: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class OcrProfile:
    """A pinned, reproducible OCR configuration.

    ``aggregate_sha256`` covers the engine id, version, language, and every asset
    fingerprint, so it changes if any of them change.
    """

    engine_id: str
    engine_version: str
    language: str
    assets: tuple[AssetFingerprint, ...]
    aggregate_sha256: str
    parameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "language": self.language,
            "assets": [asset.as_dict() for asset in self.assets],
            "aggregate_sha256": self.aggregate_sha256,
            "parameters": dict(self.parameters),
        }


def digest_file(path: Path) -> tuple[str, int]:
    """Return the SHA-256 and byte size of ``path``, streamed."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def fingerprint_assets(paths: list[Path], *, root: Path | None = None) -> tuple[AssetFingerprint, ...]:
    """Hash each file, sorted by name so the result is order-independent."""

    fingerprints: list[AssetFingerprint] = []
    for path in sorted(paths, key=lambda p: p.name):
        if not path.is_file():
            continue
        sha256, size = digest_file(path)
        name = path.name if root is None else str(path.relative_to(root)).replace("\\", "/")
        fingerprints.append(AssetFingerprint(name=name, sha256=sha256, size_bytes=size))
    return tuple(fingerprints)


def build_profile(
    *,
    engine_id: str,
    engine_version: str,
    language: str,
    assets: tuple[AssetFingerprint, ...],
    parameters: dict[str, Any] | None = None,
) -> OcrProfile:
    """Combine engine identity and asset hashes into one aggregate fingerprint."""

    resolved_parameters = dict(parameters or {})
    manifest = json.dumps(
        {
            "engine_id": engine_id,
            "engine_version": engine_version,
            "language": language,
            "assets": [asset.as_dict() for asset in assets],
            "parameters": resolved_parameters,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return OcrProfile(
        engine_id=engine_id,
        engine_version=engine_version,
        language=language,
        assets=assets,
        aggregate_sha256=hashlib.sha256(manifest).hexdigest(),
        parameters=resolved_parameters,
    )


def rapidocr_available() -> bool:
    """Return whether the RapidOCR ONNX runtime is importable."""

    try:
        import rapidocr_onnxruntime  # type: ignore[import-untyped]  # noqa: F401
    except Exception:
        return False
    return True


def tesseract_available() -> bool:
    """Return whether a Tesseract binary is on PATH.

    Tesseract is a system dependency, not a Python package. ``pytesseract``
    importing successfully does not mean the binary exists.
    """

    return shutil.which("tesseract") is not None


def tesseract_version() -> str:
    """Return the reported Tesseract version, or raise if unavailable."""

    binary = shutil.which("tesseract")
    if binary is None:
        raise OcrProfileError(
            "Tesseract is not installed or not on PATH. Install it and the required "
            "traineddata (for example 'eng' and 'chi_tra'), or use the RapidOCR profile."
        )
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        raise OcrProfileError(f"Could not run Tesseract: {exc}") from exc
    first_line = (completed.stdout or completed.stderr or "").strip().splitlines()
    if not first_line:  # pragma: no cover - env dependent
        raise OcrProfileError("Tesseract produced no version output.")
    return first_line[0].strip()


def available_engines() -> dict[str, bool]:
    """Report which engine profiles can be pinned in this environment."""

    return {"rapidocr": rapidocr_available(), "tesseract": tesseract_available()}
