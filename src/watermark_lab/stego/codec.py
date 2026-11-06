"""Codec that binds payload framing, HMAC integrity, and multiple channels."""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from typing import Any

from watermark_lab.stego.channels.base import StegoChannel
from watermark_lab.stego.channels.token_choice import TokenChoiceChannel
from watermark_lab.stego.channels.unicode import UnicodeChannel
from watermark_lab.stego.channels.whitespace import WhitespaceChannel
from watermark_lab.stego.result import StegoResult, StegoVerifyResult

# Magic + version byte so decoders can recognize our envelope.
# Version 1 = full frame (4-byte length + 4-byte nonce + 4-byte tag).
# Version 2 = short frame (2-byte length + 4-byte tag, no nonce).
_ENVELOPE_MAGIC = b"UEST"
_ENVELOPE_VERSION_FULL = 1
_ENVELOPE_VERSION_SHORT = 2


def _short_frame(payload: bytes, secret: bytes) -> bytes:
    """Minimal integrity frame for short payloads (no nonce, 32-bit HMAC tag).

    Used when the embedding channel has low capacity.  Secrecy still relies on
    the caller-provided secret; this is a tamper-evidence checksum, not
    encryption.
    """
    tag = hmac.new(secret, payload, hashlib.sha256).digest()[:4]
    return (
        _ENVELOPE_MAGIC
        + bytes([_ENVELOPE_VERSION_SHORT])
        + struct.pack("<H", len(payload))
        + payload
        + tag
    )


@dataclass(frozen=True)
class _FramedPayload:
    """Internal framed payload bytes."""

    payload: bytes
    nonce: bytes
    tag: bytes


def _frame_payload(payload: bytes, secret: bytes) -> _FramedPayload:
    """Frame payload with random salt and truncated HMAC-SHA256 tag."""
    try:
        import secrets

        salt = secrets.token_bytes(4)
    except Exception:
        salt = hashlib.sha256(str(id(payload)).encode()).digest()[:4]

    key = _derive_key(secret, salt)
    tag = hmac.new(key, payload, hashlib.sha256).digest()[:4]
    return _FramedPayload(payload=payload, nonce=salt, tag=tag)


def _serialize_frame(frame: _FramedPayload) -> bytes:
    """Serialize framed payload: magic | version | len(payload) | payload | nonce | tag."""
    return (
        _ENVELOPE_MAGIC
        + bytes([_ENVELOPE_VERSION_FULL])
        + struct.pack("<I", len(frame.payload))
        + frame.payload
        + frame.nonce
        + frame.tag
    )


def _parse_frame(data: bytes) -> _FramedPayload:
    """Parse a serialized frame, raising ValueError on corruption.

    Full frames (version 1) and short frames (version 2) are distinguished by
    their version byte, so a full frame can never be mistaken for a short
    frame or vice versa.
    """
    if not data.startswith(_ENVELOPE_MAGIC):
        raise ValueError("Invalid envelope magic")
    if len(data) < 6:
        raise ValueError("Envelope too short")
    version = data[4]

    if version == _ENVELOPE_VERSION_FULL:
        if len(data) < 9:
            raise ValueError("Envelope truncated")
        payload_len = struct.unpack("<I", data[5:9])[0]
        full_total = 9 + payload_len + 4 + 4
        if len(data) < full_total:
            raise ValueError("Envelope truncated")
        payload = data[9 : 9 + payload_len]
        nonce = data[9 + payload_len : 9 + payload_len + 4]
        tag = data[9 + payload_len + 4 : full_total]
        return _FramedPayload(payload=payload, nonce=nonce, tag=tag)

    if version == _ENVELOPE_VERSION_SHORT:
        if len(data) < 7:
            raise ValueError("Envelope truncated")
        short_len = struct.unpack("<H", data[5:7])[0]
        short_total = 7 + short_len + 4
        if len(data) < short_total:
            raise ValueError("Envelope truncated")
        payload = data[7 : 7 + short_len]
        tag = data[7 + short_len : short_total]
        return _FramedPayload(payload=payload, nonce=b"", tag=tag)

    raise ValueError(f"Unsupported envelope version: {version}")


def _derive_key(secret: bytes, salt: bytes) -> bytes:
    """Derive a 256-bit key from a user secret and random salt."""
    return hashlib.pbkdf2_hmac("sha256", secret, salt, iterations=1_000, dklen=32)


def _bytes_to_bits(data: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in data)


def _bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bit string length must be a multiple of 8")
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def _choose_channel(channel_id: str, **kwargs: Any) -> StegoChannel:
    if channel_id == "whitespace":
        return WhitespaceChannel()
    if channel_id == "unicode":
        return UnicodeChannel()
    if channel_id == "token_choice":
        return TokenChoiceChannel(
            vocab_size=kwargs.get("vocab_size", 50257),
            gamma=kwargs.get("gamma", 0.25),
            hash_key=kwargs.get("hash_key", 15485863),
            key_id=kwargs.get("key_id", "test-kgw-v1"),
            key=kwargs.get("key"),
        )
    raise ValueError(f"Unknown channel: {channel_id!r}")


class StegoEncoder:
    """Encode a hidden message into cover text with integrity."""

    def __init__(
        self,
        channel_id: str = "whitespace",
        secret: bytes = b"",
        **kwargs: Any,
    ) -> None:
        self._channel = _choose_channel(channel_id, **kwargs)
        self._channel_id = channel_id
        self._secret = secret or b"LAB_STEGO_DEFAULT_SECRET"

    def encode(self, text: str, payload: str | bytes) -> StegoResult:
        """Embed ``payload`` into ``text`` and return the stego text."""
        try:
            payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
            frame = _frame_payload(payload_bytes, self._secret)
            data = _serialize_frame(frame)
            bits = _bytes_to_bits(data)

            capacity = self._channel.capacity(text)
            if len(bits) > capacity:
                # Fall back to a minimal short frame (no salt, 4-byte HMAC tag).
                data = _short_frame(payload_bytes, self._secret)
                bits = _bytes_to_bits(data)
                if len(bits) > capacity:
                    return StegoResult(
                        status="error",
                        text=text,
                        payload=payload_bytes,
                        channel=self._channel_id,
                        bits_embedded=0,
                        warnings=(
                            f"payload requires {len(bits)} bits but channel capacity is {capacity}",
                        ),
                    )

            # Pad the bit string to a multiple of 8 so it converts cleanly to
            # bytes.  Do not exceed channel capacity.
            while len(bits) % 8 != 0:
                bits = bits + "0"
                if len(bits) > capacity:
                    # Channel cannot even fit a padded byte-multiple frame.
                    return StegoResult(
                        status="error",
                        text=text,
                        payload=payload_bytes,
                        channel=self._channel_id,
                        bits_embedded=0,
                        warnings=(
                            f"payload requires {len(bits)} bits but channel capacity is {capacity}",
                        ),
                    )

            stego = self._channel.encode(text, bits)
            return StegoResult(
                status="ok",
                text=stego,
                payload=payload_bytes,
                channel=self._channel_id,
                bits_embedded=len(bits),
                metadata={
                    "capacity": capacity,
                    "payload_bytes": len(payload_bytes),
                },
            )
        except Exception as exc:
            return StegoResult(
                status="error",
                text=text,
                payload=b"",
                channel=self._channel_id,
                bits_embedded=0,
                warnings=(str(exc),),
            )


class StegoDecoder:
    """Decode and verify a hidden message from stego text."""

    def __init__(
        self,
        channel_id: str = "whitespace",
        secret: bytes = b"",
        **kwargs: Any,
    ) -> None:
        self._channel = _choose_channel(channel_id, **kwargs)
        self._channel_id = channel_id
        self._secret = secret or b"LAB_STEGO_DEFAULT_SECRET"

    def decode(self, text: str) -> StegoResult:
        """Extract payload and verify HMAC tag from ``text``."""
        try:
            bits = self._channel.decode(text)
            # Channels that pad to capacity may return trailing zero bits.
            # Trim to the nearest lower multiple of 8; then scan backward to the
            # first byte-aligned envelope magic to drop padding.
            trimmed_len = (len(bits) // 8) * 8
            if trimmed_len == 0:
                return StegoResult(
                    status="error",
                    text=text,
                    payload=b"",
                    channel=self._channel_id,
                    bits_embedded=len(bits),
                    warnings=("No complete bytes extracted",),
                )
            bits = bits[:trimmed_len]
            magic_bits = _bytes_to_bits(_ENVELOPE_MAGIC)
            magic_pos = bits.rfind(magic_bits)
            if magic_pos != -1:
                bits = bits[magic_pos:]

            data = _bits_to_bytes(bits)
            try:
                frame = _parse_frame(data)
            except ValueError as exc:
                return StegoResult(
                    status="error",
                    text=text,
                    payload=b"",
                    channel=self._channel_id,
                    bits_embedded=len(bits),
                    warnings=(f"Envelope parse failed: {exc}",),
                )

            if frame.nonce:
                key = _derive_key(self._secret, frame.nonce)
            else:
                key = self._secret
            expected_tag = hmac.new(key, frame.payload, hashlib.sha256).digest()[: len(frame.tag)]
            if not hmac.compare_digest(expected_tag, frame.tag):
                return StegoResult(
                    status="error",
                    text=text,
                    payload=b"",
                    channel=self._channel_id,
                    bits_embedded=len(bits),
                    warnings=("Integrity check failed: forged or tampered message",),
                )

            return StegoResult(
                status="ok",
                text=text,
                payload=frame.payload,
                channel=self._channel_id,
                bits_embedded=len(bits),
                metadata={
                    "verified": True,
                    "payload_bytes": len(frame.payload),
                },
            )
        except Exception as exc:
            return StegoResult(
                status="error",
                text=text,
                payload=b"",
                channel=self._channel_id,
                bits_embedded=0,
                warnings=(str(exc),),
            )

    def verify(self, text: str) -> StegoVerifyResult:
        """Only verify integrity without returning the payload."""
        result = self.decode(text)
        if result.status == "ok":
            return StegoVerifyResult(
                status="verified",
                reason="HMAC tag matches",
                payload=result.payload,
                channel=self._channel_id,
                metadata=result.metadata,
            )
        return StegoVerifyResult(
            status="forged" if "Integrity check failed" in (result.warnings or ()) else "inconclusive",
            reason=result.warnings[0] if result.warnings else "unknown",
            payload=b"",
            channel=self._channel_id,
            metadata={"bits_embedded": result.bits_embedded},
        )

    def decode_payload(self, text: str) -> str:
        """Convenience: decode and return the payload as a UTF-8 string."""
        result = self.decode(text)
        if result.status != "ok":
            raise ValueError(result.warnings[0] if result.warnings else "decode failed")
        return result.payload.decode("utf-8")
