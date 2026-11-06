"""Steganographic textual watermarks with anti-injection integrity.

This package implements a small, pluggable steganography system for text.
It supports multiple embedding channels (whitespace patterns, Unicode zero-width
characters, and token-choice signals) and binds them together with an HMAC
integrity envelope so that embedded hidden messages can be detected and rejected
when an adversary injects or tampers with the text.

Public API:

    StegoEncoder  - encode a hidden message into cover text.
    StegoDecoder  - decode and verify a hidden message from stego text.
    StegoTinkerer - iterative, stochastic workflow that mutates a watermark
                    and measures channel-robustness metrics.
"""

from __future__ import annotations

from watermark_lab.stego.channels.base import StegoChannel
from watermark_lab.stego.codec import StegoDecoder, StegoEncoder
from watermark_lab.stego.result import StegoResult, StegoVerifyResult
from watermark_lab.stego.tinker import StegoTinkerer

__all__ = [
    "StegoChannel",
    "StegoDecoder",
    "StegoEncoder",
    "StegoResult",
    "StegoTinkerer",
    "StegoVerifyResult",
]
