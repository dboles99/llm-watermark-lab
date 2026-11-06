"""Run engine for intake, normalization, tokenization, and detection."""

from __future__ import annotations

from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.run_engine.intake import compute_sha256, normalize_text, read_source
from watermark_lab.run_engine.tokenizer_meta import tokenizer_metadata

__all__ = [
    "DetectionService",
    "compute_sha256",
    "normalize_text",
    "read_source",
    "tokenizer_metadata",
]
