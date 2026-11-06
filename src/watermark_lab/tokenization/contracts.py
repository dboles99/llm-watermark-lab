"""Immutable contracts for tokenizer-faithful scientific runs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class TokenizationError(ValueError):
    """Base error for fail-closed tokenization."""


class TokenizerAssetUnavailableError(TokenizationError):
    """Raised when the exact pinned tokenizer assets are unavailable offline."""


class TokenizerConfigurationError(TokenizationError):
    """Raised when tokenizer identity or behavior is incomplete or inconsistent."""


class TokenizerEncodingError(TokenizationError):
    """Raised when a tokenizer backend cannot honor the declared contract."""


class UnicodeNormalization(StrEnum):
    """Supported Unicode normalization choices; implicit normalization is forbidden."""

    NONE = "none"
    NFC = "NFC"


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    """Text normalization applied before encoding.

    No language conversion is performed.  In particular, ``zh-Hant-TW`` text
    remains Traditional Chinese and the contract has no locale-fallback path.
    """

    unicode_form: UnicodeNormalization = UnicodeNormalization.NFC
    preserve_whitespace: bool = True
    locale: str = "und"
    fallback_locale: None = None

    def __post_init__(self) -> None:
        if not self.locale or self.locale.casefold() == "zh-cn":
            raise TokenizerConfigurationError(
                "locale must be explicit and cannot substitute zh-CN for the target locale."
            )
        if self.fallback_locale is not None:
            raise TokenizerConfigurationError("Tokenizer locale fallback is prohibited.")
        if not self.preserve_whitespace:
            raise TokenizerConfigurationError(
                "Whitespace rewriting is not implemented; refusing an ambiguous policy."
            )

    @property
    def policy_id(self) -> str:
        return f"{self.unicode_form.value}-preserve-whitespace-no-locale-fallback-v1"

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be a string.")
        if self.unicode_form is UnicodeNormalization.NONE:
            return text
        return unicodedata.normalize("NFC", text)


@dataclass(frozen=True, slots=True)
class SpecialTokenPolicy:
    """Explicit BOS, EOS, and model-defined special-token behavior."""

    add_model_special_tokens: bool = False
    add_bos_token: bool = False
    add_eos_token: bool = False

    def __post_init__(self) -> None:
        if self.add_model_special_tokens and (self.add_bos_token or self.add_eos_token):
            raise TokenizerConfigurationError(
                "Model-default specials and explicit BOS/EOS cannot be combined."
            )

    @property
    def policy_id(self) -> str:
        if self.add_model_special_tokens:
            return "model-defined-special-tokens-v1"
        names = [
            name
            for enabled, name in (
                (self.add_bos_token, "bos"),
                (self.add_eos_token, "eos"),
            )
            if enabled
        ]
        return f"explicit-{'-'.join(names) if names else 'none'}-v1"


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    """Pinned tokenizer identity and all text-to-token behavior."""

    tokenizer_id: str
    revision: str
    normalization: NormalizationPolicy = NormalizationPolicy()
    special_tokens: SpecialTokenPolicy = SpecialTokenPolicy()
    expected_backend_versions: tuple[tuple[str, str], ...] = ()
    expected_asset_aggregate_sha256: str | None = None
    expected_tokenizer_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.tokenizer_id.strip():
            raise TokenizerConfigurationError("tokenizer_id is required.")
        if not self.revision.strip():
            raise TokenizerConfigurationError("tokenizer revision is required.")
        names: set[str] = set()
        for name, version in self.expected_backend_versions:
            if not name.strip() or not version.strip():
                raise TokenizerConfigurationError(
                    "expected tokenizer backend names and versions must be nonempty."
                )
            if name in names:
                raise TokenizerConfigurationError(
                    f"duplicate expected tokenizer backend version: {name!r}"
                )
            names.add(name)
        expected_fingerprints = (
            self.expected_asset_aggregate_sha256,
            self.expected_tokenizer_fingerprint_sha256,
        )
        if (expected_fingerprints[0] is None) != (expected_fingerprints[1] is None):
            raise TokenizerConfigurationError(
                "expected asset and tokenizer fingerprints must be approved together."
            )
        for value in expected_fingerprints:
            if value is not None and not _SHA256_RE.fullmatch(value):
                raise TokenizerConfigurationError(
                    "Expected tokenizer fingerprints must be lowercase SHA-256 digests."
                )


@dataclass(frozen=True, slots=True)
class AssetFileFingerprint:
    """Digest of one local or virtual tokenizer asset."""

    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise TokenizerConfigurationError("Tokenizer asset name is required.")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise TokenizerConfigurationError("Tokenizer asset SHA-256 is invalid.")
        if self.size_bytes < 0:
            raise TokenizerConfigurationError("Tokenizer asset size cannot be negative.")


@dataclass(frozen=True, slots=True)
class TokenizerAssetManifest:
    """Complete fingerprint metadata supplied by a tokenizer backend."""

    source: str
    files: tuple[AssetFileFingerprint, ...]
    aggregate_sha256: str

    def __post_init__(self) -> None:
        if not self.source:
            raise TokenizerConfigurationError("Tokenizer asset source is required.")
        if not self.files:
            raise TokenizerConfigurationError("At least one tokenizer asset is required.")
        if not _SHA256_RE.fullmatch(self.aggregate_sha256):
            raise TokenizerConfigurationError("Tokenizer aggregate SHA-256 is invalid.")


@dataclass(frozen=True, slots=True)
class TokenizerProvenance:
    """Resolved identity and fingerprints for one tokenizer instance."""

    tokenizer_id: str
    revision: str
    backend: str
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None
    normalization_policy: str
    special_token_policy: str
    backend_versions: tuple[tuple[str, str], ...]
    assets: TokenizerAssetManifest
    tokenizer_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class TokenizationResult:
    """Deterministic, immutable encoding result with full tokenizer provenance."""

    normalized_text: str
    token_ids: tuple[int, ...]
    input_sha256: str
    normalized_text_sha256: str
    encoding_sha256: str
    provenance: TokenizerProvenance

    @property
    def token_count(self) -> int:
        return len(self.token_ids)
