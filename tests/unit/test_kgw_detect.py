"""Unit tests for KGW logit bias, green counting, z-score, and detection.

Assumed adapter API:

    def kgw_apply_logit_bias(
        logits: torch.Tensor,
        green_list: list[int],
        delta: float,
    ) -> torch.Tensor:
        ...

    def kgw_count_green(
        tokens: list[int],
        key: bytes,
        vocab_size: int,
        gamma: float,
        hash_key: int = 15485863,
        context_width: int = 1,
        seeding_scheme: str = "simple_1",
    ) -> tuple[int, int, list[bool]]:
        ...

    def kgw_z_score(
        green_count: int,
        T: int,
        gamma: float,
    ) -> tuple[float, float]:
        ...

    def kgw_detect(
        tokens: list[int],
        key: bytes,
        gamma: float,
        context_width: int,
        hash_key: int = 15485863,
        seeding_scheme: str = "simple_1",
        ignore_repeated_ngrams: bool = False,
    ) -> dict:
        ...

Tests will fail until these functions are implemented.
"""

import json
import math
import random
from itertools import pairwise
from pathlib import Path

import pytest

TEST_KEY_KGW_V1 = b"TEST_KEY_KGW_V1"
TEST_KEY_KGW_V2 = b"TEST_KEY_KGW_V2"
DEFAULT_HASH_KEY = 15485863
VOCAB_SIZE = 50257
GAMMA = 0.25


def _import_functions():
    from watermark_lab.adapters.kgm import (
        kgw_apply_logit_bias,
        kgw_count_green,
        kgw_detect,
        kgw_z_score,
    )

    return kgw_apply_logit_bias, kgw_count_green, kgw_z_score, kgw_detect


@pytest.fixture
def kgw_functions():
    return _import_functions()


@pytest.fixture
def golden():
    fixture_path = Path(__file__).parent.parent / "fixtures" / "kgw_golden.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers that reproduce the pure-Python reference from the golden fixture.
# ---------------------------------------------------------------------------


def _ref_green_list(prev_token: int) -> set[int]:
    seed = (DEFAULT_HASH_KEY * prev_token) % (2**64 - 1)
    rng = random.Random(seed)
    ids = list(range(VOCAB_SIZE))
    rng.shuffle(ids)
    green_size = int(GAMMA * VOCAB_SIZE)
    return set(ids[:green_size])


def _all_green_sequence(length: int = 20) -> list[int]:
    """Build a synthetic sequence where every scored token is green."""
    tokens = [15496]
    for _ in range(length):
        prev = tokens[-1]
        green = _ref_green_list(prev)
        # Pick a green token deterministically.  Token 0 is not always green,
        # so choose the smallest green id and continue.
        next_token = min(green)
        tokens.append(next_token)
    return tokens


def _all_red_sequence(length: int = 20) -> list[int]:
    """Build a synthetic sequence where every scored token is red."""
    tokens = [15496]
    for _ in range(length):
        prev = tokens[-1]
        green = _ref_green_list(prev)
        red_iter = (tid for tid in range(VOCAB_SIZE) if tid not in green)
        next_token = next(red_iter)
        tokens.append(next_token)
    return tokens


# ---------------------------------------------------------------------------
# Logit bias
# ---------------------------------------------------------------------------


def test_logit_bias_increases_green_logits(kgw_functions) -> None:
    kgw_apply_logit_bias, *_ = kgw_functions
    green = [3, 9, 12]
    # Use plain Python floats in a dict keyed by token id.  The adapter can
    # accept either a torch.Tensor or a dict; the test is written to avoid a
    # hard torch dependency in the test code.
    logits = {i: float(i) for i in range(64)}
    delta = 2.0
    biased = kgw_apply_logit_bias(logits, green, delta)
    for tid in green:
        assert biased[tid] == logits[tid] + delta


def test_logit_bias_does_not_change_red_logits(kgw_functions) -> None:
    kgw_apply_logit_bias, *_ = kgw_functions
    green = [3, 9, 12]
    logits = {i: float(i) for i in range(64)}
    delta = 2.0
    biased = kgw_apply_logit_bias(logits, green, delta)
    for tid, value in logits.items():
        if tid not in green:
            assert biased[tid] == value


# ---------------------------------------------------------------------------
# Green count
# ---------------------------------------------------------------------------


def test_count_green_all_green(kgw_functions) -> None:
    _, kgw_count_green, *_ = kgw_functions
    tokens = _all_green_sequence(length=20)
    green_count, T, mask = kgw_count_green(
        tokens=tokens,
        key=TEST_KEY_KGW_V1,
        vocab_size=VOCAB_SIZE,
        gamma=GAMMA,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert T == len(tokens) - 1  # first token skipped for context_width=1
    assert green_count == T
    assert all(mask)


def test_count_green_all_red(kgw_functions) -> None:
    _, kgw_count_green, *_ = kgw_functions
    tokens = _all_red_sequence(length=20)
    green_count, T, mask = kgw_count_green(
        tokens=tokens,
        key=TEST_KEY_KGW_V1,
        vocab_size=VOCAB_SIZE,
        gamma=GAMMA,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert T == len(tokens) - 1
    assert green_count == 0
    assert not any(mask)


def test_count_green_mixed_matches_golden_vector(kgw_functions, golden) -> None:
    _, kgw_count_green, *_ = kgw_functions
    green_count, T, mask = kgw_count_green(
        tokens=golden["tokens"],
        key=TEST_KEY_KGW_V1,
        vocab_size=VOCAB_SIZE,
        gamma=GAMMA,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert green_count == golden["green_count"]
    assert T == golden["T"]
    assert mask == golden["green_token_mask"]


def test_count_green_skips_first_token_for_context_width_one(kgw_functions) -> None:
    _, kgw_count_green, *_ = kgw_functions
    # Single-token sequence cannot be scored under context_width=1.
    green_count, T, mask = kgw_count_green(
        tokens=[15496],
        key=TEST_KEY_KGW_V1,
        vocab_size=VOCAB_SIZE,
        gamma=GAMMA,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert T == 0
    assert green_count == 0
    assert mask == []


# ---------------------------------------------------------------------------
# z-score and p-value
# ---------------------------------------------------------------------------


def test_z_score_all_green_is_positive(kgw_functions) -> None:
    _, _, kgw_z_score, _ = kgw_functions
    tokens = _all_green_sequence(length=20)
    T = len(tokens) - 1
    z, _ = kgw_z_score(green_count=T, T=T, gamma=GAMMA)
    assert z > 2.0


def test_z_score_all_red_is_negative(kgw_functions) -> None:
    _, _, kgw_z_score, _ = kgw_functions
    tokens = _all_red_sequence(length=20)
    T = len(tokens) - 1
    z, _ = kgw_z_score(green_count=0, T=T, gamma=GAMMA)
    assert z < -2.0


def test_z_score_expected_count_is_near_zero(kgw_functions) -> None:
    _, _, kgw_z_score, _ = kgw_functions
    T = 40
    expected_green = round(GAMMA * T)
    z, _ = kgw_z_score(green_count=expected_green, T=T, gamma=GAMMA)
    assert abs(z) < 0.5


def test_z_score_p_value_monotonicity(kgw_functions) -> None:
    _, _, kgw_z_score, _ = kgw_functions
    T = 40
    z_low, p_low = kgw_z_score(green_count=10, T=T, gamma=GAMMA)
    z_high, p_high = kgw_z_score(green_count=20, T=T, gamma=GAMMA)
    assert z_high > z_low
    assert p_high < p_low


def test_z_score_matches_golden_vector(kgw_functions, golden) -> None:
    _, _, kgw_z_score, _ = kgw_functions
    z, p = kgw_z_score(
        green_count=golden["green_count"],
        T=golden["T"],
        gamma=golden["gamma"],
    )
    assert abs(z - golden["z_score"]) < 1e-12
    assert abs(p - golden["p_value_one_sided"]) < 1e-12


# ---------------------------------------------------------------------------
# End-to-end detection
# ---------------------------------------------------------------------------


def test_detect_matches_golden_vector(kgw_functions, golden) -> None:
    *_, kgw_detect = kgw_functions
    result = kgw_detect(
        tokens=golden["tokens"],
        key=TEST_KEY_KGW_V1,
        gamma=golden["gamma"],
        context_width=golden["context_width"],
        hash_key=golden["hash_key"],
        seeding_scheme=golden["seeding_scheme"],
    )
    assert result["green_count"] == golden["green_count"]
    assert result["T"] == golden["T"]
    assert result["green_token_mask"] == golden["green_token_mask"]
    assert abs(result["z"] - golden["z_score"]) < 1e-12
    assert abs(result["p_value_one_sided"] - golden["p_value_one_sided"]) < 1e-12


def test_detect_wrong_key_yields_no_signal(kgw_functions, golden) -> None:
    *_, kgw_detect = kgw_functions
    result = kgw_detect(
        tokens=golden["tokens"],
        key=TEST_KEY_KGW_V2,
        gamma=golden["gamma"],
        context_width=golden["context_width"],
        hash_key=golden["hash_key"],
        seeding_scheme=golden["seeding_scheme"],
    )
    # Same T, same gamma, but the green count should be close to expectation.
    assert result["T"] == golden["T"]
    assert abs(result["z"]) < 2.0
    assert result["p_value_one_sided"] > 0.05


# ---------------------------------------------------------------------------
# Repeated n-gram masking
# ---------------------------------------------------------------------------


def test_ignore_repeated_ngrams_scores_only_first_full_ngram(kgw_functions) -> None:
    *_, kgw_detect = kgw_functions
    tokens = [5, 6, 5, 6, 7]

    unmasked = kgw_detect(
        tokens=tokens,
        key=TEST_KEY_KGW_V1,
        gamma=GAMMA,
        context_width=1,
        ignore_repeated_ngrams=False,
        vocab_size=VOCAB_SIZE,
    )
    masked = kgw_detect(
        tokens=tokens,
        key=TEST_KEY_KGW_V1,
        gamma=GAMMA,
        context_width=1,
        ignore_repeated_ngrams=True,
        vocab_size=VOCAB_SIZE,
    )

    assert unmasked["T"] == 4
    assert masked["T"] == 3
    assert masked["ignored_repeated_ngrams"] == 1
    assert masked["green_token_mask"] == unmasked["green_token_mask"]
    assert masked["unique_green_token_mask"] == [
        unmasked["green_token_mask"][0],
        unmasked["green_token_mask"][1],
        unmasked["green_token_mask"][3],
    ]
    assert masked["scored_positions"] == [1, 2, 4]
    assert masked["ignored_repeated_ngram_positions"] == [3]


def test_repeated_context_with_different_continuation_is_not_dropped(kgw_functions) -> None:
    *_, kgw_detect = kgw_functions
    result = kgw_detect(
        tokens=[5, 6, 5, 7],
        key=TEST_KEY_KGW_V1,
        gamma=GAMMA,
        context_width=1,
        ignore_repeated_ngrams=True,
        vocab_size=VOCAB_SIZE,
    )

    assert result["T"] == 3
    assert result["ignored_repeated_ngrams"] == 0


def test_simple_one_rejects_wider_context(kgw_functions) -> None:
    *_, kgw_detect = kgw_functions
    with pytest.raises(ValueError, match="simple_1 requires context_width=1"):
        kgw_detect(
            tokens=[1, 2, 3, 1, 2, 3, 4],
            key=TEST_KEY_KGW_V1,
            gamma=GAMMA,
            context_width=2,
            ignore_repeated_ngrams=True,
            vocab_size=VOCAB_SIZE,
        )


@pytest.mark.parametrize("invalid", ["false", 1, None])
def test_adapter_rejects_non_boolean_repeated_ngram_option(
    kgw_functions, invalid: object
) -> None:
    *_, kgw_detect = kgw_functions
    with pytest.raises(TypeError, match="must be a boolean"):
        kgw_detect(
            tokens=[1, 2, 3],
            key=TEST_KEY_KGW_V1,
            gamma=GAMMA,
            context_width=1,
            ignore_repeated_ngrams=invalid,
            vocab_size=VOCAB_SIZE,
        )


@pytest.mark.parametrize("invalid", [0, -1])
def test_adapter_rejects_invalid_context_width(kgw_functions, invalid: int) -> None:
    *_, kgw_detect = kgw_functions
    with pytest.raises(ValueError, match="at least 1"):
        kgw_detect(
            tokens=[1, 2, 3],
            key=TEST_KEY_KGW_V1,
            gamma=GAMMA,
            context_width=invalid,
            vocab_size=VOCAB_SIZE,
        )


def test_repeated_bigram_counts_match_first_occurrence_property(kgw_functions) -> None:
    *_, kgw_detect = kgw_functions
    rng = random.Random(20260814)

    for _ in range(100):
        tokens = [rng.randrange(16) for _ in range(rng.randrange(1, 80))]
        result = kgw_detect(
            tokens=tokens,
            key=TEST_KEY_KGW_V1,
            gamma=GAMMA,
            context_width=1,
            ignore_repeated_ngrams=True,
            vocab_size=VOCAB_SIZE,
        )
        bigrams = list(pairwise(tokens))
        expected_positions: list[int] = []
        seen: set[tuple[int, int]] = set()
        for position, bigram in enumerate(bigrams, start=1):
            if bigram not in seen:
                seen.add(bigram)
                expected_positions.append(position)

        assert result["T"] == len(seen)
        assert result["ignored_repeated_ngrams"] == len(bigrams) - len(seen)
        assert result["scored_positions"] == expected_positions


@pytest.mark.parametrize(
    ("tokens", "gamma", "message"),
    [
        ([1, -1], GAMMA, "within the configured vocabulary"),
        ([1, VOCAB_SIZE], GAMMA, "within the configured vocabulary"),
        ([1, 2], 0.0, "gamma must be in"),
        ([1, 2], 1.0, "gamma must be in"),
    ],
)
def test_adapter_rejects_invalid_token_and_gamma_inputs(
    kgw_functions,
    tokens: list[int],
    gamma: float,
    message: str,
) -> None:
    *_, kgw_detect = kgw_functions
    with pytest.raises(ValueError, match=message):
        kgw_detect(
            tokens=tokens,
            key=TEST_KEY_KGW_V1,
            gamma=gamma,
            context_width=1,
            vocab_size=VOCAB_SIZE,
        )


# ---------------------------------------------------------------------------
# Sanity check for one-sided p-value formula
# ---------------------------------------------------------------------------


def test_detect_empty_is_inconclusive(kgw_functions) -> None:
    *_, kgw_detect = kgw_functions
    result = kgw_detect(
        tokens=[],
        key=TEST_KEY_KGW_V1,
        gamma=GAMMA,
        context_width=1,
    )
    assert result["T"] == 0
    assert result["green_count"] == 0
    assert result["z"] is None
    assert result["p_value_one_sided"] is None
    assert result["warning"] == "too_few_tokens"


def test_detect_short_text_is_inconclusive(kgw_functions) -> None:
    *_, kgw_detect = kgw_functions
    result = kgw_detect(
        tokens=[15496],
        key=TEST_KEY_KGW_V1,
        gamma=GAMMA,
        context_width=1,
    )
    assert result["T"] == 0
    assert result["z"] is None
    assert result["p_value_one_sided"] is None
    assert result["warning"] == "too_few_tokens"


def test_one_sided_p_value_formula() -> None:
    """Self-contained sanity check: p = 1 - Phi(z) == 0.5 * erfc(z/sqrt(2))."""
    z = 1.1547005383792517
    expected = 0.5 * math.erfc(z / math.sqrt(2))
    assert 0.0 < expected < 1.0
    assert abs(expected - 0.12410653949496178) < 1e-12
