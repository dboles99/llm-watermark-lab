"""Token-choice steganography that blends into a KGW watermark.

Each generated token can encode a bit by choosing from the green list or the
red list. Because the green list is keyed, the hidden message is only readable
by someone with the watermark key. This channel also functions as a detectable
KGW watermark, giving the dual property of watermarking + steganography.
"""

from __future__ import annotations

from typing import Any

from watermark_lab.adapters.kgw import DEFAULT_HASH_KEY, kgw_count_green, kgw_green_list


class TokenChoiceChannel:
    """Embed bits in KGW green-list / red-list token choices."""

    channel_id: str = "token_choice"

    def __init__(
        self,
        vocab_size: int = 50257,
        gamma: float = 0.25,
        hash_key: int = DEFAULT_HASH_KEY,
        key_id: str = "test-kgw-v1",
        key: bytes | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hash_key = hash_key
        self.key_id = key_id
        self._key = key if key is not None else self._resolve_key(key_id)

    @staticmethod
    def _resolve_key(key_id: str) -> bytes:
        # Mirror the test key vault used by the KGW generator.  Production
        # callers should pass ``key=`` directly so the raw key never lives in
        # this module's memory longer than necessary.
        vault = {
            "test-kgw-v1": b"TEST_KEY_KGW_V1",
        }
        if key_id not in vault:
            raise ValueError(f"Unknown key_id: {key_id!r}")
        return vault[key_id]

    def accepts(self, text: str) -> bool:
        """Return whether ``text`` is in the token-ID form this channel requires."""

        parts = text.split()
        return bool(parts) and all(part.lstrip("-").isdigit() for part in parts)

    def _require_token_ids(self, text: str) -> list[int]:
        """Parse whitespace-separated token IDs, or raise an explanatory error.

        This channel operates on token IDs rather than prose, because a bit is
        encoded by choosing a token from the green or red list and that choice is
        only meaningful in the model's vocabulary. Passing natural language here
        used to fail deep inside the loop with a bare ``int()`` error, which read
        as a bug rather than as the design constraint it is.
        """

        parts = text.split()
        if not parts:
            return []
        bad = next((part for part in parts if not part.lstrip("-").isdigit()), None)
        if bad is not None:
            raise ValueError(
                f"{self.channel_id!r} operates on whitespace-separated token IDs, not prose "
                f"(first offending token: {bad!r}). A bit is encoded by choosing between the "
                "green and red list, which is only defined over the model vocabulary. To use "
                "this channel on real text, tokenize with the pinned tokenizer first and pass "
                "the resulting IDs. Until generation is tied to a real model and tokenizer, "
                "this channel cannot be measured on natural language."
            )
        return [int(part) for part in parts]

    def capacity(self, text: str) -> int:
        # One bit per token after the first context token. Text that is not in
        # token-ID form has zero capacity here, rather than an apparent capacity
        # that would fail later during encoding.
        if not self.accepts(text):
            return 0
        return max(0, len(text.split()) - 1)

    def encode(self, text: str, payload_bits: str, **kwargs: Any) -> str:
        self._require_token_ids(text)
        tokens = text.split()
        if not tokens:
            return text
        if not payload_bits:
            return text

        out: list[str] = [tokens[0]]
        for i, bit in enumerate(payload_bits):
            if i + 1 >= len(tokens):
                break
            prev_token = int(tokens[i])
            green = kgw_green_list(
                key=self._key,
                context=(prev_token,),
                vocab_size=self.vocab_size,
                gamma=self.gamma,
                hash_key=self.hash_key,
            )
            green_set = set(green)
            current = int(tokens[i + 1])
            # If current token already encodes the desired bit, leave it.
            desired_in_green = bit == "1"
            if (current in green_set) == desired_in_green:
                out.append(str(current))
                continue
            # Otherwise flip to a token that encodes the desired bit, preferring
            # small IDs to keep the text looking like synthetic token ids.
            pool = green if desired_in_green else [t for t in range(self.vocab_size) if t not in green_set]
            replacement = str(min(pool) if pool else current)
            tokens[i + 1] = replacement
            out.append(replacement)

        # Append any remaining tokens unchanged.
        out.extend(tokens[len(payload_bits) + 1 :])
        return " ".join(out)

    def decode(self, text: str, **kwargs: Any) -> str:
        tokens = text.split()
        if len(tokens) < 2:
            return ""

        bits: list[str] = []
        for i in range(1, len(tokens)):
            prev_token = int(tokens[i - 1])
            current = int(tokens[i])
            green = kgw_green_list(
                key=self._key,
                context=(prev_token,),
                vocab_size=self.vocab_size,
                gamma=self.gamma,
                hash_key=self.hash_key,
            )
            bits.append("1" if current in green else "0")
        return "".join(bits)

    def robustness_score(self, cover: str, stego: str, **kwargs: Any) -> float:
        """Token-choice robustness is measured by KGW green-fraction survival."""
        try:
            tokens = [int(t) for t in stego.split()]
            green_count, t, _ = kgw_count_green(
                tokens=tokens,
                key=self._key,
                vocab_size=self.vocab_size,
                gamma=self.gamma,
                hash_key=self.hash_key,
            )
            if t == 0:
                return 0.0
            return float(green_count / t)
        except Exception:
            return 0.0
