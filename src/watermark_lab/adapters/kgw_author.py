"""Pinned KGW author-code compatibility profile.

This module implements a token-ID oracle profile for the official KGW
``extended_watermark_processor.py`` snapshot.  It is intentionally separate
from :mod:`watermark_lab.adapters.kgw`: the latter uses a project byte-key
extension and Python's RNG and therefore cannot be author-code compatible.

The author source is Apache-2.0 licensed.  This implementation is an
independent, small compatibility boundary derived from the published algorithm
and verified against the pinned source; it does not vendor the author file.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from scipy.stats import norm  # type: ignore[import-untyped]

SIMPLE_1_MODULUS = 2**64 - 1


class AuthorKgwRuntimeError(RuntimeError):
    """Raised when the exact optional author-compatibility runtime is absent."""


@dataclass(frozen=True, slots=True)
class AuthorKgwProfile:
    """Immutable identity of one device/build-specific author-code profile."""

    profile_id: str = "kgw-author-simple1-torch-2.13.0-cpu-win-amd64-v1"
    author_repository_url: str = "https://github.com/jwkirchenbauer/lm-watermarking"
    author_commit: str = "82922516930c02f8aa322765defdb5863d07a00e"
    author_source: str = "extended_watermark_processor.py"
    author_source_git_blob: str = "8c55d58c9cf336ac31d7118c9555ed3c12a9296e"
    author_source_sha256: str = (
        "fc3469e677ba3f2a0e4cdf5a7cae2852d8d423c02cde395cc1e038fca42038a1"
    )
    prf_source_sha256: str = "51c854949c8e9309208dc6e57e2f9edbb8f2d90f0ff61edcd4ed2ef890e19ddc"
    license_spdx: str = "Apache-2.0"
    seeding_scheme: Literal["simple_1"] = "simple_1"
    prf: Literal["additive_prf"] = "additive_prf"
    context_width: Literal[1] = 1
    self_salt: Literal[False] = False
    hash_key: int = 15_485_863
    select_green_tokens: Literal[True] = True
    torch_version: str = "2.13.0+cpu"
    torch_git_version: str = "cf30153c4c131c8164ee7798e5022d810682e2cb"
    rng_device: Literal["cpu"] = "cpu"
    platform_system: Literal["Windows"] = "Windows"
    platform_machine: Literal["AMD64"] = "AMD64"
    raw_text_normalizers: tuple[()] = ()


AUTHOR_KGW_CPU_PROFILE = AuthorKgwProfile()


@dataclass(frozen=True, slots=True)
class TorchRuntimeIdentity:
    """Runtime properties that affect the author RNG oracle."""

    torch_version: str
    torch_git_version: str
    cuda_build: str | None
    rng_device: str
    platform_system: str
    platform_machine: str


@dataclass(frozen=True, slots=True)
class AuthorKgwScore:
    """Token-level score matching the pinned author's ``simple_1`` semantics."""

    profile_id: str
    runtime: TorchRuntimeIdentity
    green_count: int
    scored_token_count: int
    green_fraction: float
    z_score: float
    p_value_one_sided: float
    prediction: bool
    green_token_mask: tuple[bool, ...]
    unique_green_token_mask: tuple[bool, ...]
    scored_positions: tuple[int, ...]
    ignored_repeated_ngram_positions: tuple[int, ...]
    ignore_repeated_ngrams: bool


def _load_pinned_torch(profile: AuthorKgwProfile) -> tuple[Any, TorchRuntimeIdentity]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise AuthorKgwRuntimeError(
            "The optional pinned PyTorch CPU runtime is unavailable; no RNG fallback was attempted."
        ) from exc

    torch_version = str(getattr(torch, "__version__", ""))
    version_namespace = getattr(torch, "version", None)
    torch_git_version = str(getattr(version_namespace, "git_version", ""))
    cuda_build_value = getattr(version_namespace, "cuda", None)
    cuda_build = None if cuda_build_value is None else str(cuda_build_value)
    runtime = TorchRuntimeIdentity(
        torch_version=torch_version,
        torch_git_version=torch_git_version,
        cuda_build=cuda_build,
        rng_device=profile.rng_device,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
    )
    mismatches: list[str] = []
    if runtime.torch_version != profile.torch_version:
        mismatches.append(
            f"torch={runtime.torch_version!r}, expected {profile.torch_version!r}"
        )
    if runtime.torch_git_version != profile.torch_git_version:
        mismatches.append(
            f"torch_git={runtime.torch_git_version!r}, expected {profile.torch_git_version!r}"
        )
    if runtime.cuda_build is not None:
        mismatches.append(f"CUDA build {runtime.cuda_build!r} is not the pinned CPU-only build")
    if runtime.platform_system != profile.platform_system:
        mismatches.append(
            f"system={runtime.platform_system!r}, expected {profile.platform_system!r}"
        )
    if runtime.platform_machine.casefold() != profile.platform_machine.casefold():
        mismatches.append(
            f"machine={runtime.platform_machine!r}, expected {profile.platform_machine!r}"
        )
    if mismatches:
        raise AuthorKgwRuntimeError(
            "Pinned KGW author profile runtime mismatch: " + "; ".join(mismatches)
        )
    return torch, runtime


def _require_plain_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _validate_vocab_gamma(vocab_size: object, gamma: object) -> tuple[int, float]:
    checked_vocab = _require_plain_int(vocab_size, field="vocab_size")
    if checked_vocab < 1:
        raise ValueError("vocab_size must be positive")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be a real number")
    checked_gamma = float(gamma)
    if not math.isfinite(checked_gamma) or not 0.0 < checked_gamma < 1.0:
        raise ValueError("gamma must be finite and in (0, 1)")
    return checked_vocab, checked_gamma


def _validated_tokens(tokens: Sequence[int], vocab_size: int) -> tuple[int, ...]:
    if isinstance(tokens, (str, bytes)):
        raise TypeError("tokens must be a sequence of integer token IDs")
    checked: list[int] = []
    for token in tokens:
        value = _require_plain_int(token, field="token ID")
        if not 0 <= value < vocab_size:
            raise ValueError("token IDs must lie in the configured contiguous vocabulary")
        checked.append(value)
    return tuple(checked)


def _validated_context(tokens: Sequence[int]) -> tuple[int, ...]:
    """Validate an oracle context without imposing the candidate vocabulary.

    The pinned author helper itself accepts any nonnegative integer context.
    Scientific full-sequence detection remains stricter and enforces the
    declared contiguous tokenizer vocabulary through ``_validated_tokens``.
    """

    if isinstance(tokens, (str, bytes)):
        raise TypeError("context must be a sequence of integer token IDs")
    checked: list[int] = []
    for token in tokens:
        value = _require_plain_int(token, field="context token ID")
        if value < 0:
            raise ValueError("context token IDs must be nonnegative")
        checked.append(value)
    return tuple(checked)


def kgw_author_simple1_seed(
    previous_token_id: int,
    *,
    profile: AuthorKgwProfile = AUTHOR_KGW_CPU_PROFILE,
) -> int:
    """Return the pinned author's width-one additive PRF seed."""

    token_id = _require_plain_int(previous_token_id, field="previous_token_id")
    if token_id < 0:
        raise ValueError("previous_token_id must be nonnegative")
    return (profile.hash_key * token_id) % SIMPLE_1_MODULUS


def kgw_author_green_list(
    context: Sequence[int],
    vocab_size: int,
    gamma: float,
    *,
    profile: AuthorKgwProfile = AUTHOR_KGW_CPU_PROFILE,
) -> tuple[int, ...]:
    """Return the ordered author ``torch.randperm`` prefix for one context."""

    checked_vocab, checked_gamma = _validate_vocab_gamma(vocab_size, gamma)
    checked_context = _validated_context(context)
    if len(checked_context) < profile.context_width:
        raise ValueError("simple_1 requires at least one token of prefix context")
    torch, _runtime = _load_pinned_torch(profile)
    generator = torch.Generator(device=profile.rng_device)
    generator.manual_seed(kgw_author_simple1_seed(checked_context[-1], profile=profile))
    permutation = torch.randperm(
        checked_vocab,
        device=profile.rng_device,
        generator=generator,
    )
    green_size = int(checked_vocab * checked_gamma)
    return tuple(int(token_id) for token_id in permutation[:green_size].tolist())


def ordered_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash an ordered token-ID vector using the fixture's canonical JSON form."""

    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def kgw_author_detect(
    tokens: Sequence[int],
    vocab_size: int,
    gamma: float,
    threshold: float = 4.0,
    *,
    ignore_repeated_ngrams: bool = True,
    profile: AuthorKgwProfile = AUTHOR_KGW_CPU_PROFILE,
) -> AuthorKgwScore:
    """Score token IDs using the pinned extended author implementation semantics.

    Raw-text tokenization and the author's optional Unicode sanitizer are
    deliberately outside this token-ID oracle.  A scientific caller must first
    use a separately pinned tokenizer contract and must record that this profile
    uses ``normalizers=[]``.
    """

    checked_vocab, checked_gamma = _validate_vocab_gamma(vocab_size, gamma)
    checked_tokens = _validated_tokens(tokens, checked_vocab)
    if len(checked_tokens) - profile.context_width < 1:
        raise ValueError("at least one candidate token is required after the context prefix")
    if not isinstance(ignore_repeated_ngrams, bool):
        raise TypeError("ignore_repeated_ngrams must be a boolean")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a real number")
    checked_threshold = float(threshold)
    if not math.isfinite(checked_threshold):
        raise ValueError("threshold must be finite")

    _torch, runtime = _load_pinned_torch(profile)
    green_by_context: dict[int, frozenset[int]] = {}
    full_mask: list[bool] = []
    unique_mask: list[bool] = []
    scored_positions: list[int] = []
    ignored_positions: list[int] = []
    seen_pairs: set[tuple[int, int]] = set()

    for position in range(profile.context_width, len(checked_tokens)):
        previous = checked_tokens[position - 1]
        candidate = checked_tokens[position]
        green = green_by_context.get(previous)
        if green is None:
            green = frozenset(
                kgw_author_green_list((previous,), checked_vocab, checked_gamma, profile=profile)
            )
            green_by_context[previous] = green
        is_green = candidate in green
        full_mask.append(is_green)
        pair = (previous, candidate)
        if ignore_repeated_ngrams and pair in seen_pairs:
            ignored_positions.append(position)
            continue
        seen_pairs.add(pair)
        scored_positions.append(position)
        unique_mask.append(is_green)

    effective_mask = unique_mask if ignore_repeated_ngrams else full_mask
    green_count = sum(effective_mask)
    scored_count = len(effective_mask)
    z_score = (green_count - checked_gamma * scored_count) / math.sqrt(
        scored_count * checked_gamma * (1.0 - checked_gamma)
    )
    p_value = float(norm.sf(z_score))
    return AuthorKgwScore(
        profile_id=profile.profile_id,
        runtime=runtime,
        green_count=green_count,
        scored_token_count=scored_count,
        green_fraction=green_count / scored_count,
        z_score=z_score,
        p_value_one_sided=p_value,
        prediction=z_score > checked_threshold,
        green_token_mask=tuple(full_mask),
        unique_green_token_mask=tuple(unique_mask),
        scored_positions=tuple(scored_positions),
        ignored_repeated_ngram_positions=tuple(ignored_positions),
        ignore_repeated_ngrams=ignore_repeated_ngrams,
    )


__all__ = [
    "AUTHOR_KGW_CPU_PROFILE",
    "AuthorKgwProfile",
    "AuthorKgwRuntimeError",
    "AuthorKgwScore",
    "TorchRuntimeIdentity",
    "kgw_author_detect",
    "kgw_author_green_list",
    "kgw_author_simple1_seed",
    "ordered_token_ids_sha256",
]
