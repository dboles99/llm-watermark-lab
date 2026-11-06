"""Shared result type for blind/semi-blind watermark detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class BlindDetectionResult:
    method_id: str
    status: Literal["completed", "inconclusive", "blocked"]
    decision: Literal["watermarked", "not_watermarked", "inconclusive"]
    score: float | None
    p_value: float | None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


__all__ = ["BlindDetectionResult"]
