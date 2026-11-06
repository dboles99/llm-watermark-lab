"""Whitespace-pattern steganography.

Each word encodes two bits using the first two whitespace characters that
follow it:

    char 1: space = 0, tab = 1
    char 2: space = 0, tab = 1

This is unambiguous, robust, and easy to decode as long as at least two
whitespace characters remain after each word.
"""

from __future__ import annotations

from typing import Any


class WhitespaceChannel:
    """Embed bits using space/tab patterns after words."""

    channel_id: str = "whitespace"

    def capacity(self, text: str) -> int:
        words = text.split()
        # Two bits per word.
        return 2 * len(words)

    def encode(self, text: str, payload_bits: str, **kwargs: Any) -> str:
        if not payload_bits:
            return text

        words = text.split()
        if not words:
            return text

        # Pad to a multiple of 2 bits so they pair into two-char patterns.
        if len(payload_bits) % 2 != 0:
            payload_bits = payload_bits + "0"

        def _char(bit: str) -> str:
            return "\t" if bit == "1" else " "

        out: list[str] = []
        for i, word in enumerate(words):
            out.append(word)
            bit_index = i * 2
            if bit_index < len(payload_bits):
                out.append(_char(payload_bits[bit_index]))
            else:
                out.append(" ")
            if bit_index + 1 < len(payload_bits):
                out.append(_char(payload_bits[bit_index + 1]))
            else:
                out.append(" ")

        return "".join(out)

    def decode(self, text: str, **kwargs: Any) -> str:
        words = text.split()
        if not words:
            return ""

        bits: list[str] = []
        cursor = 0
        for word in words:
            word_start = text.find(word, cursor)
            if word_start == -1:
                break
            cursor = word_start + len(word)

            # Read exactly the next two whitespace characters, defaulting to 0
            # when whitespace has been stripped.
            for _ in range(2):
                if cursor < len(text) and text[cursor] in (" ", "\t"):
                    bits.append("1" if text[cursor] == "\t" else "0")
                    cursor += 1
                else:
                    bits.append("0")

        return "".join(bits)

    def robustness_score(self, cover: str, stego: str, **kwargs: Any) -> float:
        """Whitespace survives copy/paste and minor reformatting."""
        cover_spaces = cover.count(" ") + cover.count("\t")
        stego_spaces = stego.count(" ") + stego.count("\t")
        if cover_spaces == 0:
            return 1.0 if stego_spaces == 0 else 0.5
        ratio = min(stego_spaces, cover_spaces) / max(cover_spaces, 1)
        return float(ratio)
