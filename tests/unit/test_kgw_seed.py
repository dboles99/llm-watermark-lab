"""Unit tests for KGW seed derivation (kgw_seed_from_context).

These tests assume a baseline adapter API roughly matching the reconciled spec:

    def kgw_seed_from_context(
        key: bytes,
        context: tuple[int, ...],
        hash_key: int = 15485863,
        context_width: int = 1,
        seeding_scheme: str = "simple_1",
    ) -> int:
        ...

Tests will fail until the function is implemented, satisfying red/green TDD.
"""

import pytest

TEST_KEY_KGW_V1 = b"TEST_KEY_KGW_V1"
TEST_KEY_KGW_V2 = b"TEST_KEY_KGW_V2"
DEFAULT_HASH_KEY = 15485863


def _import_seed_function():
    # Intentionally defer import so missing code produces an ImportError (red test).
    from watermark_lab.adapters.kgm import kgw_seed_from_context

    return kgw_seed_from_context


@pytest.fixture
def kgw_seed_from_context():
    return _import_seed_function()


def test_seed_determinism(kgw_seed_from_context) -> None:
    """Same inputs must produce the same 64-bit seed."""
    ctx = (15496,)
    s1 = kgw_seed_from_context(
        key=TEST_KEY_KGW_V1,
        context=ctx,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    s2 = kgw_seed_from_context(
        key=TEST_KEY_KGW_V1,
        context=ctx,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert s1 == s2
    assert isinstance(s1, int)


def test_seed_key_sensitivity(kgw_seed_from_context) -> None:
    """Different test key -> different seed for the same context."""
    ctx = (15496,)
    s1 = kgw_seed_from_context(
        key=TEST_KEY_KGW_V1,
        context=ctx,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    s2 = kgw_seed_from_context(
        key=TEST_KEY_KGW_V2,
        context=ctx,
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert s1 != s2


def test_seed_context_sensitivity(kgw_seed_from_context) -> None:
    """Same key, different previous token -> different seed."""
    s1 = kgw_seed_from_context(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    s2 = kgw_seed_from_context(
        key=TEST_KEY_KGW_V1,
        context=(42,),
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert s1 != s2


def test_seed_modulus_is_2_64_minus_1(kgw_seed_from_context) -> None:
    """simple_1 must reduce the product modulo 2**64 - 1."""
    # Pick a context that overflows 64 bits when multiplied by the hash key.
    large_prev = (2**64) // DEFAULT_HASH_KEY + 12345
    s = kgw_seed_from_context(
        key=TEST_KEY_KGW_V1,
        context=(large_prev,),
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert 0 <= s <= 2**64 - 2  # modulus is 2**64 - 1, so max valid seed is mod-1


def test_seed_matches_golden_vector(kgw_seed_from_context) -> None:
    """Reference value from tests/fixtures/kgw_golden.json."""
    s = kgw_seed_from_context(
        key=TEST_KEY_KGW_V1,
        context=(15496,),
        hash_key=DEFAULT_HASH_KEY,
        context_width=1,
        seeding_scheme="simple_1",
    )
    assert s == 239968933048


def test_seed_raises_for_unknown_scheme(kgw_seed_from_context) -> None:
    """Alternative PRF schemes are documented but not baseline; reject unknown names."""
    with pytest.raises(NotImplementedError):
        kgw_seed_from_context(
            key=TEST_KEY_KGW_V1,
            context=(15496,),
            hash_key=DEFAULT_HASH_KEY,
            context_width=1,
            seeding_scheme="selfhash",
        )
