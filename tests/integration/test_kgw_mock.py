"""Integration tests for the KGW adapter against mock provider fixtures.

These tests do not require a real model checkpoint.  They use a small mock
provider that exposes deterministic token sequences:

- profile="clean"       : unwatermarked synthetic tokens.
- profile="kgw-test"    : tokens produced with the reconciled KGW adapter and
                          TEST_KEY_KGW_V1.

Tests will fail until the KGW adapter and the mock provider fixtures exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TEST_KEY_KGW_V1 = b"TEST_KEY_KGW_V1"
TEST_KEY_KGW_V2 = b"TEST_KEY_KGW_V2"
GAMMA = 0.25
DELTA = 2.0
CONTEXT_WIDTH = 1
HASH_KEY = 15485863


def _import_adapter_and_provider():
    from watermark_lab.adapters.kgm import kgw_detect
    from watermark_lab.testing.mock_provider import MockProvider

    return kgw_detect, MockProvider


@pytest.fixture
def adapter_and_provider():
    return _import_adapter_and_provider()


@pytest.fixture
def golden():
    fixture_path = Path(__file__).parent.parent / "fixtures" / "kgw_golden.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def mock_provider(adapter_and_provider):
    _, MockProvider = adapter_and_provider
    return MockProvider()


@pytest.fixture
def kgw_detect(adapter_and_provider):
    kgw_detect_fn, _ = adapter_and_provider
    return kgw_detect_fn


def _detect(provider_result, key: bytes, kgw_detect_fn):
    return kgw_detect_fn(
        tokens=provider_result["tokens"],
        key=key,
        gamma=GAMMA,
        context_width=CONTEXT_WIDTH,
        hash_key=HASH_KEY,
        seeding_scheme="simple_1",
    )


@pytest.mark.integration
def test_clean_profile_returns_no_watermark_signal(
    mock_provider, kgw_detect
) -> None:
    """The clean fixture should not be flagged by the KGW detector."""
    result = mock_provider.generate(profile="clean", max_new_tokens=40)
    detection = _detect(result, TEST_KEY_KGW_V1, kgw_detect)
    assert detection["T"] >= 20
    assert abs(detection["z"]) < 2.0
    assert detection["p_value_one_sided"] > 0.05


@pytest.mark.integration
def test_kgw_test_profile_is_flagged_by_detector(
    mock_provider, kgw_detect
) -> None:
    """The kgw-test fixture, watermarked with TEST_KEY_KGW_V1, must be detected."""
    result = mock_provider.generate(profile="kgw-test", max_new_tokens=40)
    detection = _detect(result, TEST_KEY_KGW_V1, kgw_detect)
    assert detection["T"] >= 20
    assert detection["z"] > 2.0
    assert detection["p_value_one_sided"] < 0.05


@pytest.mark.integration
def test_wrong_key_on_watermarked_text_yields_no_signal(
    mock_provider, kgw_detect
) -> None:
    """Detection with the wrong key should erase the watermark signal."""
    result = mock_provider.generate(profile="kgw-test", max_new_tokens=40)
    detection = _detect(result, TEST_KEY_KGW_V2, kgw_detect)
    assert detection["T"] >= 20
    assert abs(detection["z"]) < 2.0
    assert detection["p_value_one_sided"] > 0.05


@pytest.mark.integration
def test_golden_vector_round_trips_through_detector(kgw_detect, golden) -> None:
    """The stored golden vector is a valid kgw-test fixture."""
    detection = kgw_detect(
        tokens=golden["tokens"],
        key=TEST_KEY_KGW_V1,
        gamma=golden["gamma"],
        context_width=golden["context_width"],
        hash_key=golden["hash_key"],
        seeding_scheme=golden["seeding_scheme"],
    )
    assert detection["green_count"] == golden["green_count"]
    assert detection["z"] > 0.0


@pytest.mark.integration
def test_timeout_profile_raises_documented_exception(mock_provider) -> None:
    """If the mock provider exposes a timeout profile, the adapter raises a documented error."""
    try:
        mock_provider.generate(profile="timeout", max_new_tokens=10)
    except Exception as exc:
        assert "Watermark" in type(exc).__name__ or "Provider" in type(exc).__name__
        return
    pytest.skip("Mock provider has no timeout profile; skipping timeout error-handling test.")


@pytest.mark.integration
def test_malformed_profile_raises_documented_exception(mock_provider) -> None:
    """If the mock provider exposes a malformed profile, the adapter raises a documented error."""
    try:
        mock_provider.generate(profile="malformed", max_new_tokens=10)
    except Exception as exc:
        assert "Watermark" in type(exc).__name__ or "Provider" in type(exc).__name__
        return
    pytest.skip("Mock provider has no malformed profile; skipping malformed error-handling test.")
