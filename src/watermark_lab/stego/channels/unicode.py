"""Unicode zero-width steganography.

Uses U+200B ZERO WIDTH SPACE (0) and U+200C ZERO WIDTH NON-JOINER (1).
High capacity and invisible, but fragile against normalization and many
re-encodings.
"""

from __future__ import annotations

from typing import Any


class UnicodeChannel:
    """Embed bits using zero-width characters."""

    channel_id: str = "unicode"

    BIT_0 = "\u200b"  # ZERO WIDTH SPACE
    BIT_1 = "\u200c"  # ZERO WIDTH NON-JOINER

    def capacity(self, text: str) -> int:
        # We can insert characters between any two visible characters or at the
        # end. Capacity equals the number of insertion positions.
        return max(0, len(text))

    def encode(self, text: str, payload_bits: str, **kwargs: Any) -> str:
        if not payload_bits:
            return text

        # Embed one bit after each visible character. This preserves the
        # original text and adds the hidden signal in the gaps.
        chars = list(text)
        out: list[str] = []
        for i, ch in enumerate(chars):
            out.append(ch)
            if i < len(payload_bits):
                out.append(self.BIT_1 if payload_bits[i] == "1" else self.BIT_0)
        return "".join(out)

    def decode(self, text: str, **kwargs: Any) -> str:
        bits: list[str] = []
        for ch in text:
            if ch == self.BIT_0:
                bits.append("0")
            elif ch == self.BIT_1:
                bits.append("1")
        return "".join(bits)

    def robustness_score(self, cover: str, stego: str, **kwargs: Any) -> float:
        """Unicode channel is fragile; score by surviving zero-width count."""
        expected = len(cover)
        actual = stego.count(self.BIT_0) + stego.count(self.BIT_1)
        if expected == 0:
            return 1.0
        return float(min(actual, expected) / expected)
