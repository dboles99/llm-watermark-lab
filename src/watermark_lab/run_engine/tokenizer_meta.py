"""Tokenizer metadata lookup for the run engine."""

from __future__ import annotations

TOKENIZER_FAMILIES: dict[str, dict[str, object]] = {
    "gpt2": {
        "family": "gpt2",
        "vocab_size": 50257,
        "revision": "local",
        "implementation": "legacy-minimal-placeholder",
    },
    "synthetic-numeric-v1": {
        "family": "synthetic-numeric",
        "vocab_size": 50257,
        "revision": "fixture-v1",
        "implementation": "explicit-synthetic-test-fixture",
    },
}


def tokenizer_metadata(tokenizer_id: str) -> dict[str, object]:
    """Return metadata for ``tokenizer_id``.

    Unknown families raise ``ValueError`` so the run engine cannot silently
    substitute tokenizers. ``gpt2`` still identifies the legacy placeholder;
    new mock fixtures should use the explicit ``synthetic-numeric-v1`` entry.
    """
    if tokenizer_id not in TOKENIZER_FAMILIES:
        raise ValueError(f"Unknown tokenizer family: {tokenizer_id!r}")
    return dict(TOKENIZER_FAMILIES[tokenizer_id])
