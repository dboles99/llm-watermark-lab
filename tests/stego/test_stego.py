"""Tests for steganographic encode/decode/verify and tinker workflow."""

from __future__ import annotations

import pytest

from watermark_lab.stego import StegoDecoder, StegoEncoder, StegoTinkerer

CHANNELS = ["whitespace", "unicode"]


def _long_cover(channel: str) -> str:
    """Return a cover long enough for the short integrity envelope (~104 bits)."""
    if channel == "unicode":
        return (
            "Open sesame for hidden messages inside ordinary looking text "
            "with many more characters so the compact envelope fits cleanly"
        )
    return (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )


def test_whitespace_roundtrip():
    encoder = StegoEncoder(channel_id="whitespace", secret=b"xyzzy")
    decoder = StegoDecoder(channel_id="whitespace", secret=b"xyzzy")
    cover = _long_cover("whitespace")
    payload = "hi"
    result = encoder.encode(cover, payload)
    assert result.status == "ok"
    assert result.bits_embedded > 0

    decoded = decoder.decode(result.text)
    assert decoded.status == "ok"
    assert decoded.payload.decode("utf-8") == payload

    verify = decoder.verify(result.text)
    assert verify.status == "verified"


def test_unicode_roundtrip():
    encoder = StegoEncoder(channel_id="unicode", secret=b"xyzzy")
    decoder = StegoDecoder(channel_id="unicode", secret=b"xyzzy")
    cover = _long_cover("unicode")
    payload = "ok"
    result = encoder.encode(cover, payload)
    assert result.status == "ok"

    decoded = decoder.decode(result.text)
    assert decoded.status == "ok"
    assert decoded.payload.decode("utf-8") == payload


def test_wrong_secret_fails_integrity():
    encoder = StegoEncoder(channel_id="whitespace", secret=b"correct")
    decoder = StegoDecoder(channel_id="whitespace", secret=b"wrong")
    cover = _long_cover("whitespace")
    result = encoder.encode(cover, "x")
    assert result.status == "ok"

    decoded = decoder.decode(result.text)
    assert decoded.status == "error"
    assert "Integrity check failed" in decoded.warnings[0]


@pytest.mark.parametrize("channel", CHANNELS)
def test_capacity_error_for_short_cover(channel: str):
    encoder = StegoEncoder(channel_id=channel, secret=b"s")
    cover = "hi"
    result = encoder.encode(cover, "this payload is far too long to fit")
    assert result.status == "error"


@pytest.mark.parametrize("channel", CHANNELS)
def test_tamper_detection(channel: str):
    """Anti-injection: flipping a carrier bit must break the HMAC tag."""
    encoder = StegoEncoder(channel_id=channel, secret=b"tamper")
    decoder = StegoDecoder(channel_id=channel, secret=b"tamper")
    cover = _long_cover(channel)
    result = encoder.encode(cover, "ok")
    assert result.status == "ok"

    stego = result.text
    if channel == "whitespace":
        # Flip the first tab to a space or vice versa — this corrupts a
        # carrier bit inside the envelope.
        for i, ch in enumerate(stego):
            if ch == "\t":
                tampered = stego[:i] + " " + stego[i + 1 :]
                break
            if ch == " " and i > 0 and stego[i - 1] != " ":
                tampered = stego[:i] + "\t" + stego[i + 1 :]
                break
        else:
            pytest.skip("no carrier whitespace found")
    else:
        # Unicode: flip a zero-width char.
        if "\u200b" in stego:
            tampered = stego.replace("\u200b", "\u200c", 1)
        elif "\u200c" in stego:
            tampered = stego.replace("\u200c", "\u200b", 1)
        else:
            pytest.skip("no zero-width carrier found")

    decoded = decoder.decode(tampered)
    assert decoded.status == "error"
    verify = decoder.verify(tampered)
    assert verify.status in {"forged", "inconclusive"}


@pytest.mark.parametrize("channel", CHANNELS)
def test_injection_attack_breaks_envelope(channel: str):
    """An attacker appending their own stego text must not verify under our secret."""
    encoder = StegoEncoder(channel_id=channel, secret=b"victim-secret")
    decoder = StegoDecoder(channel_id=channel, secret=b"victim-secret")
    cover = _long_cover(channel)
    result = encoder.encode(cover, "ok")
    assert result.status == "ok"

    # Attacker encodes their own message with a different secret and appends it.
    attacker_encoder = StegoEncoder(channel_id=channel, secret=b"attacker-secret")
    # Give the attacker a cover long enough for their own envelope.
    attacker_cover = _long_cover(channel) + " " + _long_cover(channel)
    injected_tail = attacker_encoder.encode(attacker_cover, "evil")
    assert injected_tail.status == "ok"

    injected = result.text + injected_tail.text
    decoded = decoder.decode(injected)
    # The decoder must not return a valid payload under the victim's secret.
    assert decoded.status == "error"


def test_tinker_workflow_runs():
    tinkerer = StegoTinkerer(
        channel_id="whitespace",
        secret=b"tinker",
        population_size=4,
        generations=2,
        mutation_rate=0.1,
    )
    cover = _long_cover("whitespace")
    summary = tinkerer.run(cover, "ok")
    assert summary["total_variants"] > 0
    assert summary["overall_best"] is not None


def test_token_choice_roundtrip():
    encoder = StegoEncoder(channel_id="token_choice", secret=b"k")
    decoder = StegoDecoder(channel_id="token_choice", secret=b"k")
    cover = " ".join(str(i) for i in range(120))
    payload = "ok"
    result = encoder.encode(cover, payload)
    assert result.status == "ok"

    decoded = decoder.decode(result.text)
    assert decoded.status == "ok"
    assert decoded.payload.decode("utf-8") == payload


def test_token_choice_accepts_raw_key():
    """Production callers may pass raw key bytes, bypassing the test vault."""
    encoder = StegoEncoder(
        channel_id="token_choice",
        secret=b"k",
        key=b"CUSTOM_RAW_KEY",
        key_id="custom",
    )
    decoder = StegoDecoder(
        channel_id="token_choice",
        secret=b"k",
        key=b"CUSTOM_RAW_KEY",
        key_id="custom",
    )
    cover = " ".join(str(i) for i in range(120))
    result = encoder.encode(cover, "ok")
    assert result.status == "ok"

    decoded = decoder.decode(result.text)
    assert decoded.status == "ok"
    assert decoded.payload.decode("utf-8") == "ok"


def test_token_choice_tamper_detection():
    """A single token replacement must break the token-choice envelope."""
    encoder = StegoEncoder(channel_id="token_choice", secret=b"k")
    decoder = StegoDecoder(channel_id="token_choice", secret=b"k")
    cover = " ".join(str(i) for i in range(120))
    result = encoder.encode(cover, "ok")
    assert result.status == "ok"

    tokens = result.text.split()
    # Mutate a token near the middle of the embedded region.
    mid = 60
    tokens[mid] = "99999"
    tampered = " ".join(tokens)
    decoded = decoder.decode(tampered)
    assert decoded.status == "error"
