"""Fail-closed text-to-token boundary used by scientific workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from watermark_lab.tokenization.contracts import (
    TokenizationResult,
    TokenizerAssetManifest,
    TokenizerEncodingError,
    TokenizerProvenance,
    TokenizerSpec,
)


class TokenizerBackend(Protocol):
    """Minimal injected backend contract; it must not perform asset discovery."""

    @property
    def backend_name(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def runtime_versions(self) -> tuple[tuple[str, str], ...]: ...

    @property
    def asset_manifest(self) -> TokenizerAssetManifest: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tokenizer_fingerprint_sha256(
    spec: TokenizerSpec,
    *,
    backend_name: str,
    vocab_size: int,
    bos_token_id: int | None,
    eos_token_id: int | None,
    runtime_versions: tuple[tuple[str, str], ...],
    asset_manifest: TokenizerAssetManifest,
) -> str:
    """Derive the fingerprint that an independent asset audit must approve."""

    backend_versions = tuple(sorted(runtime_versions))
    return _canonical_sha256(
        {
            "tokenizer_id": spec.tokenizer_id,
            "revision": spec.revision,
            "backend": backend_name,
            "vocab_size": vocab_size,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "normalization_policy": spec.normalization.policy_id,
            "normalization_locale": spec.normalization.locale,
            "special_token_policy": spec.special_tokens.policy_id,
            "backend_versions": backend_versions,
            "asset_aggregate_sha256": asset_manifest.aggregate_sha256,
            "assets": [
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in asset_manifest.files
            ],
        }
    )


class FaithfulTokenizer:
    """Bind a pinned tokenizer specification to one already-resolved backend."""

    def __init__(self, spec: TokenizerSpec, backend: TokenizerBackend) -> None:
        if not backend.backend_name:
            raise TokenizerEncodingError("Tokenizer backend name is required.")
        if backend.vocab_size <= 0:
            raise TokenizerEncodingError("Tokenizer backend vocab_size must be positive.")
        self.spec = spec
        self._backend = backend
        self.provenance = self._build_provenance()

    def _build_provenance(self) -> TokenizerProvenance:
        manifest = self._backend.asset_manifest
        expected_assets = self.spec.expected_asset_aggregate_sha256
        if expected_assets is not None and manifest.aggregate_sha256 != expected_assets:
            raise TokenizerEncodingError(
                "Tokenizer asset aggregate does not match the approved fingerprint; "
                "no asset substitution was attempted."
            )
        backend_versions = tuple(sorted(self._backend.runtime_versions))
        if self.spec.expected_backend_versions:
            expected = tuple(sorted(self.spec.expected_backend_versions))
            if backend_versions != expected:
                raise TokenizerEncodingError(
                    f"Tokenizer backend versions {backend_versions!r} do not match the pinned "
                    f"versions {expected!r}; no runtime substitution was attempted."
                )
        fingerprint = tokenizer_fingerprint_sha256(
            self.spec,
            backend_name=self._backend.backend_name,
            vocab_size=self._backend.vocab_size,
            bos_token_id=self._backend.bos_token_id,
            eos_token_id=self._backend.eos_token_id,
            runtime_versions=backend_versions,
            asset_manifest=manifest,
        )
        expected_tokenizer = self.spec.expected_tokenizer_fingerprint_sha256
        if expected_tokenizer is not None and fingerprint != expected_tokenizer:
            raise TokenizerEncodingError(
                "Resolved tokenizer does not match the approved tokenizer fingerprint; "
                "no runtime substitution was attempted."
            )
        return TokenizerProvenance(
            tokenizer_id=self.spec.tokenizer_id,
            revision=self.spec.revision,
            backend=self._backend.backend_name,
            vocab_size=self._backend.vocab_size,
            bos_token_id=self._backend.bos_token_id,
            eos_token_id=self._backend.eos_token_id,
            normalization_policy=self.spec.normalization.policy_id,
            special_token_policy=self.spec.special_tokens.policy_id,
            backend_versions=backend_versions,
            assets=manifest,
            tokenizer_fingerprint_sha256=fingerprint,
        )

    def encode(self, text: str) -> TokenizationResult:
        """Normalize and encode without any fallback or tokenizer substitution."""

        normalized = self.spec.normalization.normalize(text)
        try:
            raw_ids = self._backend.encode(
                normalized,
                add_special_tokens=self.spec.special_tokens.add_model_special_tokens,
            )
        except Exception as exc:
            if isinstance(exc, TokenizerEncodingError):
                raise
            raise TokenizerEncodingError(
                f"Tokenizer {self.spec.tokenizer_id!r} at revision "
                f"{self.spec.revision!r} could not encode the input; no fallback was attempted."
            ) from exc

        token_ids = self._validated_ids(raw_ids)
        if (
            not self.spec.special_tokens.add_model_special_tokens
            and self._backend.bos_token_id is not None
            and token_ids
            and token_ids[0] == self._backend.bos_token_id
        ):
            raise TokenizerEncodingError(
                "Tokenizer backend produced a leading BOS token even though "
                "add_special_tokens=False; refusing implicit author-profile input rewriting."
            )
        if self.spec.special_tokens.add_bos_token:
            token_ids = (self._required_special_id("BOS", self._backend.bos_token_id), *token_ids)
        if self.spec.special_tokens.add_eos_token:
            token_ids = (*token_ids, self._required_special_id("EOS", self._backend.eos_token_id))

        input_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        normalized_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        encoding_digest = _canonical_sha256(
            {
                "normalized_text_sha256": normalized_digest,
                "token_ids": token_ids,
                "tokenizer_fingerprint_sha256": self.provenance.tokenizer_fingerprint_sha256,
            }
        )
        return TokenizationResult(
            normalized_text=normalized,
            token_ids=token_ids,
            input_sha256=input_digest,
            normalized_text_sha256=normalized_digest,
            encoding_sha256=encoding_digest,
            provenance=self.provenance,
        )

    def _validated_ids(self, raw_ids: Sequence[int]) -> tuple[int, ...]:
        if isinstance(raw_ids, (str, bytes)):
            raise TokenizerEncodingError("Tokenizer backend returned an invalid token sequence.")
        result: list[int] = []
        for token_id in raw_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TokenizerEncodingError("Tokenizer backend returned a non-integer token ID.")
            if not 0 <= token_id < self._backend.vocab_size:
                raise TokenizerEncodingError(
                    f"Tokenizer backend returned token ID {token_id} outside its vocabulary."
                )
            result.append(token_id)
        return tuple(result)

    def _required_special_id(self, name: str, token_id: int | None) -> int:
        if token_id is None:
            raise TokenizerEncodingError(
                f"Explicit {name} behavior was requested but the tokenizer has no {name} token ID."
            )
        if isinstance(token_id, bool) or not 0 <= token_id < self._backend.vocab_size:
            raise TokenizerEncodingError(f"Tokenizer {name} token ID is invalid.")
        return token_id
