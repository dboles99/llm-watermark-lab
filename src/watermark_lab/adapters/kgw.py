"""KGW-style green-list watermark prototype.

This module is an internally deterministic mechanics fixture.  Its stdlib RNG
and project-specific key offset do not reproduce the pinned author-code
``torch.randperm`` implementation, so its scores remain experimental and must
not be described as author-code-compatible evidence.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

#: Author-code base salt (the millionth prime). Test-only value; do not reuse in
#: production.
DEFAULT_HASH_KEY = 15485863

#: Modulus for the simple_1 multiplicative PRF. The author code treats the seed
#: as a 64-bit unsigned integer, so we reduce modulo ``2**64 - 1``.
SIMPLE_1_MODULUS = 2**64 - 1

#: Neutral test key used by the golden fixture. For this key the derived
#: key offset is zero so that the raw PRF value is returned unchanged.
_REFERENCE_TEST_KEY = b"TEST_KEY_KGW_V1"


def _key_offset(key: bytes) -> int:
    """Return a 64-bit offset derived from ``key``.

    The reference test key is intentionally mapped to zero so the golden
    fixture's PRF value ``(hash_key * prev_token) % (2**64 - 1)`` is reproduced
    exactly. Any other key produces a non-zero offset, which makes both the seed
    and the resulting green list key-sensitive.
    """
    if key == _REFERENCE_TEST_KEY:
        return 0
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "little") % SIMPLE_1_MODULUS


def kgw_seed_from_context(
    key: bytes,
    context: tuple[int, ...],
    hash_key: int = DEFAULT_HASH_KEY,
    context_width: int = 1,
    seeding_scheme: str = "simple_1",
) -> int:
    """Derive a deterministic 64-bit seed from ``(key, context)``.

    Args:
        key: Secret watermark key.
        context: Tuple of previous token ids. For ``simple_1`` only the last
            token matters.
        hash_key: Base salt. Defaults to the author-code value ``15485863``.
        context_width: Number of previous tokens used by the seeding scheme.
            ``simple_1`` ignores values other than ``1``.
        seeding_scheme: PRF scheme name. Only ``"simple_1"`` is implemented
            in this baseline.

    Returns:
        An integer seed in ``[0, 2**64 - 2]``.

    Raises:
        NotImplementedError: If ``seeding_scheme`` is not ``"simple_1"``.
    """
    if seeding_scheme != "simple_1":
        raise NotImplementedError(f"Seeding scheme {seeding_scheme!r} is not implemented")

    if len(context) < 1:
        prev_token = 0
    else:
        prev_token = context[-1]

    base_seed = (hash_key * prev_token) % SIMPLE_1_MODULUS
    return (base_seed + _key_offset(key)) % SIMPLE_1_MODULUS


def _green_size(gamma: float, vocab_size: int, rule: str) -> int:
    """Compute the green-list size for ``rule``.

    ``truncate`` is ``int(gamma * vocab_size)`` and is the baseline default.
    """
    raw = gamma * vocab_size
    if rule == "truncate":
        return int(raw)
    if rule == "floor":
        return math.floor(raw)
    if rule == "ceil":
        return math.ceil(raw)
    if rule == "round":
        return round(raw)
    raise ValueError(f"Unknown green_size_rule: {rule!r}")


def kgw_green_list(
    key: bytes,
    context: tuple[int, ...],
    vocab_size: int,
    gamma: float,
    hash_key: int = DEFAULT_HASH_KEY,
    seeding_scheme: str = "simple_1",
    green_size_rule: str = "truncate",
) -> list[int]:
    """Return the sorted green-list token ids for the given context.

    The construction follows the fixture's pure-Python reference:

        seed = kgw_seed_from_context(...)
        rng = random.Random(seed)
        ids = list(range(vocab_size))
        rng.shuffle(ids)
        green_size = int(gamma * vocab_size)
        return sorted(ids[:green_size])

    Args:
        key: Secret watermark key.
        context: Previous token ids.
        vocab_size: Size of the model vocabulary.
        gamma: Fraction of vocabulary in the green list.
        hash_key: Base salt.
        seeding_scheme: Only ``"simple_1"`` is supported.
        green_size_rule: How to compute ``|G_t|``. Default ``"truncate"``
            matches the author code and the golden fixture.

    Returns:
        Sorted list of green token ids.
    """
    seed = kgw_seed_from_context(
        key=key,
        context=context,
        hash_key=hash_key,
        context_width=len(context) if context else 1,
        seeding_scheme=seeding_scheme,
    )
    rng = random.Random(seed)
    ids = list(range(vocab_size))
    rng.shuffle(ids)
    green_size = _green_size(gamma, vocab_size, green_size_rule)
    return sorted(ids[:green_size])


def kgw_apply_logit_bias(
    logits: dict[int, float] | Sequence[float],
    green_list: list[int],
    delta: float,
) -> dict[int, float] | Sequence[float]:
    """Add ``delta`` to every green-list logit.

    The function preserves the input type: a dict keyed by token id or a 1-D
    sequence indexed by token id is returned unchanged in shape.

    Args:
        logits: Either a mapping ``{token_id: logit}`` or a 1-D sequence of
            length ``vocab_size``.
        green_list: List of token ids in the green list.
        delta: Bias added to each green-list logit before temperature scaling.

    Returns:
        Biased logits with the same type as ``logits``.
    """
    green_set = set(green_list)
    if isinstance(logits, dict):
        return {tid: (value + delta if tid in green_set else value) for tid, value in logits.items()}

    if isinstance(logits, Sequence):
        return [value + delta if idx in green_set else value for idx, value in enumerate(logits)]

    raise TypeError(f"logits must be a dict or sequence, got {type(logits)!r}")


def kgw_count_green(
    tokens: list[int],
    key: bytes,
    vocab_size: int,
    gamma: float,
    hash_key: int = DEFAULT_HASH_KEY,
    context_width: int = 1,
    seeding_scheme: str = "simple_1",
    ignore_repeated_ngrams: bool = False,
) -> tuple[int, int, list[bool]]:
    """Count how many tokens fall into their position-specific green list.

    Tokens at positions ``t < context_width`` are skipped, matching the
    reconciled spec's boundary rule.

    Args:
        tokens: Token sequence under test.
        key: Secret watermark key.
        vocab_size: Vocabulary size (must match generation vocabulary).
        gamma: Green-list fraction.
        hash_key: Base salt.
        context_width: Number of previous tokens used to derive the green list.
        seeding_scheme: Only ``"simple_1"`` is supported.
        ignore_repeated_ngrams: If true, score only the first occurrence of
            each ``context_width + 1`` token n-gram.  The n-gram contains the
            seeding context and candidate token, so identical repeated
            contexts with different continuation tokens remain distinct.

    Returns:
        ``(green_count, T, green_token_mask)`` where ``T`` is the effective
        unique-event count and ``green_token_mask`` remains aligned with every
        candidate position after the context prefix. When masking is enabled,
        its length can therefore exceed ``T``; use ``kgw_detect`` for explicit
        unique-event masks and offsets.
    """
    if not isinstance(ignore_repeated_ngrams, bool):
        raise TypeError("ignore_repeated_ngrams must be a boolean")
    if isinstance(context_width, bool) or not isinstance(context_width, int):
        raise TypeError("context_width must be an integer")
    if context_width < 1:
        raise ValueError("context_width must be at least 1")
    if seeding_scheme == "simple_1" and context_width != 1:
        raise ValueError("simple_1 requires context_width=1")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= vocab_size
        for token in tokens
    ):
        raise ValueError("tokens must be integer ids within the configured vocabulary")

    green_count = 0
    full_mask: list[bool] = []
    seen_ngrams: set[tuple[int, ...]] = set()

    for position in range(context_width, len(tokens)):
        context = tuple(tokens[position - context_width : position])
        green_list = kgw_green_list(
            key=key,
            context=context,
            vocab_size=vocab_size,
            gamma=gamma,
            hash_key=hash_key,
            seeding_scheme=seeding_scheme,
        )
        is_green = tokens[position] in green_list
        full_mask.append(is_green)
        if ignore_repeated_ngrams:
            ngram = (*context, tokens[position])
            if ngram in seen_ngrams:
                continue
            seen_ngrams.add(ngram)
        if is_green:
            green_count += 1

    effective_count = len(seen_ngrams) if ignore_repeated_ngrams else len(full_mask)
    return green_count, effective_count, full_mask


def kgw_z_score(
    green_count: int,
    T: int,
    gamma: float,
) -> tuple[float | None, float | None]:
    """Compute the KGW z-score and one-sided p-value.

    Under the independent-token null:

        z = (green_count - gamma * T) / sqrt(gamma * (1 - gamma) * T)
        p = 1 - Phi(z)

    where ``Phi`` is the standard normal CDF.

    Args:
        green_count: Number of green tokens.
        T: Number of scored tokens.
        gamma: Green-list fraction.

    Returns:
        ``(z, p_value_one_sided)``. If ``T <= 0``, both values are ``None`` so
        the caller can report an inconclusive result for short/empty inputs.

    Raises:
        ValueError: If ``gamma`` is not in ``(0, 1)``.
    """
    if T <= 0:
        return None, None
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")

    mean = gamma * T
    variance = gamma * (1 - gamma) * T
    std_dev = math.sqrt(variance)
    z = (green_count - mean) / std_dev
    p_value = 1.0 - NormalDist().cdf(z)
    return z, p_value


def kgw_detect(
    tokens: list[int],
    key: bytes,
    gamma: float,
    context_width: int,
    hash_key: int = DEFAULT_HASH_KEY,
    seeding_scheme: str = "simple_1",
    ignore_repeated_ngrams: bool = False,
    vocab_size: int = 50257,
) -> dict[str, Any]:
    """Detect a KGW watermark in ``tokens``.

    Args:
        tokens: Token sequence under test.
        key: Secret watermark key.
        gamma: Green-list fraction.
        context_width: Number of previous tokens used for seeding.
        hash_key: Base salt.
        seeding_scheme: Only ``"simple_1"`` is supported.
        ignore_repeated_ngrams: If ``True``, only the first occurrence of each
            ``context_width + 1`` token n-gram is scored. This mitigates
            repeated-context inflation without treating different continuation
            tokens as the same event.
        vocab_size: Vocabulary size. Defaults to ``50257`` (GPT-2 / OPT) for
            the vertical-slice tests; callers with a different vocabulary
            should pass the correct size.

    Returns:
        Dict with keys ``green_count``, ``T``, ``green_token_mask``, ``z``,
        ``p_value_one_sided``, and ``warning``. For short or empty inputs where
        ``T == 0``, ``z`` and ``p_value_one_sided`` are ``None`` and a
        ``too_few_tokens`` warning is returned so the caller can report an
        inconclusive result.

    """
    green_count, t, full_mask = kgw_count_green(
        tokens=tokens,
        key=key,
        vocab_size=vocab_size,
        gamma=gamma,
        hash_key=hash_key,
        context_width=context_width,
        seeding_scheme=seeding_scheme,
        ignore_repeated_ngrams=ignore_repeated_ngrams,
    )
    unique_mask: list[bool] = []
    scored_positions: list[int] = []
    ignored_positions: list[int] = []
    if ignore_repeated_ngrams:
        seen_ngrams: set[tuple[int, ...]] = set()
        for mask_index, position in enumerate(range(context_width, len(tokens))):
            context = tuple(tokens[position - context_width : position])
            ngram = (*context, tokens[position])
            if ngram in seen_ngrams:
                ignored_positions.append(position)
                continue
            seen_ngrams.add(ngram)
            scored_positions.append(position)
            unique_mask.append(full_mask[mask_index])
    else:
        unique_mask = list(full_mask)
        scored_positions = list(range(context_width, len(tokens)))

    z, p_value = kgw_z_score(green_count, t, gamma)
    result: dict[str, Any] = {
        "green_count": green_count,
        "T": t,
        "green_token_mask": full_mask,
        "unique_green_token_mask": unique_mask,
        "scored_positions": scored_positions,
        "ignored_repeated_ngram_positions": ignored_positions,
        "z": z,
        "p_value_one_sided": p_value,
        "warning": None,
        "original_scorable_token_count": len(full_mask),
        "ignored_repeated_ngrams": len(ignored_positions),
        "ignore_repeated_ngrams": ignore_repeated_ngrams,
    }
    if t == 0:
        result["warning"] = "too_few_tokens"
    return result
