"""Intake helpers: read source text, record media type, and compute hashes."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_source(path_or_text: str) -> tuple[str, str, str | None]:
    """Return ``(text, media_type, source_path)`` for a file path or raw text.

    If ``path_or_text`` points to an existing file it is read and its media type
    is inferred from the extension. Otherwise it is treated as raw pasted text
    with no source path.
    """
    candidate = Path(path_or_text)
    if candidate.exists() and candidate.is_file():
        ext = candidate.suffix.lower()
        text = _read_text_file(candidate)
        return text, MEDIA_TYPES.get(ext, "application/octet-stream"), str(candidate.resolve())
    return path_or_text, "text/plain", None


def compute_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization to ``text``."""
    return unicodedata.normalize("NFC", text)
