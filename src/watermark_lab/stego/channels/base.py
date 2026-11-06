"""Base protocol for a steganographic embedding channel."""

from __future__ import annotations

from typing import Any, Protocol


class StegoChannel(Protocol):
    """A pluggable steganographic channel for text."""

    channel_id: str

    def capacity(self, text: str) -> int:
        """Return the maximum number of bits that can be embedded in ``text``."""
        ...

    def encode(self, text: str, payload_bits: str, **kwargs: Any) -> str:
        """Embed ``payload_bits`` into ``text`` and return the stego text."""
        ...

    def decode(self, text: str, **kwargs: Any) -> str:
        """Extract the embedded bit string from ``text``."""
        ...

    def robustness_score(self, cover: str, stego: str, **kwargs: Any) -> float:
        """Return a 0..1 score measuring how much the channel survives perturbation."""
        ...
