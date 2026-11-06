from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from watermark_lab.adapters.kgw import kgw_green_list
from watermark_lab.adapters.kgw_author import (
    AUTHOR_KGW_CPU_PROFILE,
    AuthorKgwRuntimeError,
    kgw_author_detect,
    kgw_author_green_list,
    kgw_author_simple1_seed,
    ordered_token_ids_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "kgw_author_torch_2_13_cpu_win_amd64.json").read_text(
        encoding="utf-8"
    )
)
PINNED_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
requires_pinned_torch = pytest.mark.skipif(
    not PINNED_TORCH_AVAILABLE,
    reason="optional pinned KGW author PyTorch runtime is not installed",
)


def test_author_profile_and_fixture_provenance_are_exact() -> None:
    profile = AUTHOR_KGW_CPU_PROFILE

    assert FIXTURE["fixture_kind"] == "direct-author-code-oracle"
    assert FIXTURE["profile_id"] == profile.profile_id
    assert FIXTURE["author_commit"] == profile.author_commit
    assert FIXTURE["author_source_git_blob"] == profile.author_source_git_blob
    assert FIXTURE["author_source_sha256"] == profile.author_source_sha256
    assert FIXTURE["prf_source_sha256"] == profile.prf_source_sha256
    assert FIXTURE["license_spdx"] == "Apache-2.0"
    assert FIXTURE["oracle"]["torch_version"] == profile.torch_version
    assert FIXTURE["oracle"]["torch_git_version"] == profile.torch_git_version
    assert FIXTURE["oracle"]["rng_device"] == "cpu"
    assert FIXTURE["oracle"]["cuda_build"] is None
    assert FIXTURE["oracle"]["normalizers"] == []


def test_author_seed_and_prefix_boundary_match_pinned_source() -> None:
    assert kgw_author_simple1_seed(15_496) == 239_968_933_048
    with pytest.raises(ValueError, match="prefix context"):
        kgw_author_green_list((), vocab_size=16, gamma=0.25)


def test_optional_torch_dependency_fails_closed_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from watermark_lab.adapters import kgw_author

    def unavailable(_name: str) -> object:
        raise ImportError("test fixture")

    monkeypatch.setattr(kgw_author.importlib, "import_module", unavailable)
    with pytest.raises(AuthorKgwRuntimeError, match="no RNG fallback"):
        kgw_author_green_list((1,), vocab_size=16, gamma=0.25)


@requires_pinned_torch
def test_direct_author_green_list_vectors_match() -> None:
    for vector in FIXTURE["vectors"]:
        green = kgw_author_green_list(
            vector["context"],
            vocab_size=vector["vocab_size"],
            gamma=vector["gamma"],
        )

        assert len(green) == vector["green_count"]
        expected_ids = vector.get("ordered_green_ids")
        if expected_ids is not None:
            assert list(green) == expected_ids
        expected_prefix = vector.get("ordered_first_32")
        if expected_prefix is not None:
            assert list(green[:32]) == expected_prefix
        assert ordered_token_ids_sha256(green) == vector["ordered_sha256"]


@requires_pinned_torch
def test_direct_author_score_vectors_match() -> None:
    for vector in FIXTURE["score_vectors"]:
        if "expected_error" in vector:
            with pytest.raises(ValueError, match="candidate token"):
                kgw_author_detect(
                    vector["tokens"],
                    vector["vocab_size"],
                    vector["gamma"],
                    ignore_repeated_ngrams=vector["ignore_repeated_ngrams"],
                )
            continue

        score = kgw_author_detect(
            vector["tokens"],
            vector["vocab_size"],
            vector["gamma"],
            ignore_repeated_ngrams=vector["ignore_repeated_ngrams"],
        )
        assert score.green_count == vector["green_count"]
        assert score.scored_token_count == vector["scored_token_count"]
        assert list(score.green_token_mask) == vector["green_token_mask"]
        assert score.z_score == pytest.approx(vector["z_score"], rel=1e-15)
        assert score.p_value_one_sided == pytest.approx(
            vector["p_value_one_sided"], rel=1e-15
        )
        if "scored_positions" in vector:
            assert list(score.scored_positions) == vector["scored_positions"]
            assert list(score.ignored_repeated_ngram_positions) == vector["ignored_positions"]


@requires_pinned_torch
def test_profile_rejects_different_torch_build() -> None:
    mismatched = replace(AUTHOR_KGW_CPU_PROFILE, torch_version="different-build")

    with pytest.raises(AuthorKgwRuntimeError, match="runtime mismatch"):
        kgw_author_green_list((1,), vocab_size=16, gamma=0.25, profile=mismatched)


@requires_pinned_torch
def test_project_python_rng_profile_is_an_expected_mismatch() -> None:
    author = kgw_author_green_list((15_496,), vocab_size=16, gamma=0.25)
    project = kgw_green_list(
        key=b"TEST_KEY_KGW_V1",
        context=(15_496,),
        vocab_size=16,
        gamma=0.25,
    )

    assert list(author) == [10, 4, 8, 0]
    assert set(author) != set(project)


@pytest.mark.parametrize(
    ("tokens", "vocab_size", "gamma", "error"),
    [
        ([True, 1], 16, 0.25, TypeError),
        ([0, 16], 16, 0.25, ValueError),
        ([0, 1], True, 0.25, TypeError),
        ([0, 1], 16, 1.0, ValueError),
    ],
)
def test_author_boundary_rejects_invalid_inputs(
    tokens: list[int],
    vocab_size: int,
    gamma: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        kgw_author_detect(tokens, vocab_size, gamma)
