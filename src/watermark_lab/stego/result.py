"""Shared result types for steganographic operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class StegoResult:
    """Output of a steganographic encode/decode operation."""

    status: Literal["ok", "error"]
    text: str
    payload: bytes = b""
    channel: str = "unknown"
    bits_embedded: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StegoVerifyResult:
    """Output of an anti-injection integrity check."""

    status: Literal["verified", "forged", "inconclusive"]
    reason: str
    payload: bytes = b""
    channel: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
