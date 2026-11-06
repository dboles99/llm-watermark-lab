"""Unit tests for KGW green-list construction (kgw_green_list).

Assumed adapter API:

    def kgw_green_list(
        key: bytes,
        context: tuple[int, ...],
        vocab_size: int,
        gamma: float,
        hash_key: int = 15485863,
        seeding_scheme: str = "simple_1",
        green_size_rule: str = "truncate",
    ) -> list[int]:
        ...

Tests will fail until the function is implemented.
"""

import pytest

TEST_KEY_KGW_V1 = b"TEST_KEY_KGW_V1"
TEST_KEY_KGW_V2 = b"TEST_KEY_KGW_V2"
DEFAULT_HASH_KEY = 15485863
VOCAB_SIZE = 50257


def _import_green_list():
    from watermark_lab.adapters.kgm import kgw_green_list

    return kgw_green_list


@pytest.fixture
def kgw_green_list():
    return _import_green_list()


@pytest.mark.parametrize(
    "gamma,expected_size",
    [
        (0.1, int(0.1 * VOCAB_SIZE)),
        (0.25, int(0.25 * VOCAB_SIZE)),
        (0.5, int(0.5 * VOCAB_SIZE)),
    ],
)
def test_green_list_size(kgw_green_list, gamma: float, expected_size: int) -> None:
    """Green-list size equals int(gamma * vocab_size) for the default truncate rule."""
    green = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=gamma,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
        green_size_rule="truncate",
    )
    assert len(green) == expected_size


def test_green_list_determinism(kgw_green_list) -> None:
    """Same inputs produce the same sorted green list."""
    green1 = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    green2 = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    assert green1 == green2
    assert green1 == sorted(green1)


def test_green_list_context_sensitivity(kgw_green_list) -> None:
    """Different context -> different green list."""
    green1 = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    green2 = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(42,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    assert green1 != green2


def test_green_list_key_sensitivity(kgw_green_list) -> None:
    """Different test key -> different green list."""
    green1 = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    green2 = kgw_green_list(
        key=TEST_KEY_KGW_V2,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    assert green1 != green2


def test_green_list_matches_golden_vector(kgw_green_list) -> None:
    """First 20 ids match the deterministic fixture."""
    green = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
        green_size_rule="truncate",
    )
    expected_first_20 = [
        3, 9, 12, 22, 23, 31, 37, 40, 41, 45,
        56, 64, 67, 85, 86, 91, 92, 93, 95, 106,
    ]
    assert green[:20] == expected_first_20


def test_green_list_no_duplicates(kgw_green_list) -> None:
    """A green list must be a set of unique token ids."""
    green = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    assert len(green) == len(set(green))


def test_green_list_all_ids_in_vocab(kgw_green_list) -> None:
    """Every green id must be in [0, vocab_size)."""
    green = kgw_green_list(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        vocab_size=VOCAB_SIZE,
        gamma=0.25,
        hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="simple_1",
    )
    assert all(0 <= token_id < VOCAB_SIZE for token_id in green)
