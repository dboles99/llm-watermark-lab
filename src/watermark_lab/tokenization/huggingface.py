"""Offline-only loader for pinned Hugging Face tokenizer assets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from watermark_lab.tokenization.boundary import FaithfulTokenizer
from watermark_lab.tokenization.contracts import (
    AssetFileFingerprint,
    TokenizerAssetManifest,
    TokenizerAssetUnavailableError,
    TokenizerConfigurationError,
    TokenizerEncodingError,
    TokenizerSpec,
)


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def fingerprint_local_assets(asset_directory: Path) -> TokenizerAssetManifest:
    """Hash every regular file below a resolved local tokenizer directory."""

    try:
        root = asset_directory.resolve(strict=True)
    except OSError as exc:
        raise TokenizerAssetUnavailableError(
            f"Pinned tokenizer asset directory is unavailable: {asset_directory}"
        ) from exc
    if not root.is_dir():
        raise TokenizerAssetUnavailableError(
            f"Pinned tokenizer asset path is not a directory: {asset_directory}"
        )

    fingerprints: list[AssetFileFingerprint] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise TokenizerAssetUnavailableError(
                f"Tokenizer asset escapes its pinned directory: {candidate}"
            )
        relative = resolved.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            # Hugging Face materialization metadata contains timestamps and
            # download bookkeeping. It is not a tokenizer input and must not
            # make the tokenizer fingerprint machine- or retrieval-dependent.
            continue
        digest, size = _digest_file(resolved)
        fingerprints.append(
            AssetFileFingerprint(
                name=relative.as_posix(),
                sha256=digest,
                size_bytes=size,
            )
        )
    if not fingerprints:
        raise TokenizerAssetUnavailableError(
            f"Pinned tokenizer asset directory contains no files: {asset_directory}"
        )

    manifest_payload = json.dumps(
        [
            {"name": item.name, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in fingerprints
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return TokenizerAssetManifest(
        source=f"local-directory:{root.name}",
        files=tuple(fingerprints),
        aggregate_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )


class _HuggingFaceBackend:
    def __init__(self, tokenizer: Any, asset_manifest: TokenizerAssetManifest) -> None:
        if getattr(tokenizer, "is_fast", False) is not True:
            raise TokenizerConfigurationError(
                "A Hugging Face fast tokenizer is required for reproducible alignment."
            )
        try:
            vocab_size = len(tokenizer)
        except (TypeError, AttributeError) as exc:
            raise TokenizerConfigurationError("Tokenizer vocabulary size is unavailable.") from exc
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
            raise TokenizerConfigurationError("Tokenizer vocabulary size is invalid.")

        self._tokenizer = tokenizer
        self.backend_name = f"transformers-fast:{type(tokenizer).__name__}"
        self.vocab_size = vocab_size
        self.bos_token_id = self._optional_token_id(tokenizer, "bos_token_id")
        self.eos_token_id = self._optional_token_id(tokenizer, "eos_token_id")
        self.asset_manifest = asset_manifest
        import huggingface_hub
        import tokenizers  # type: ignore[import-untyped]
        import transformers

        self.runtime_versions = (
            ("huggingface_hub", str(huggingface_hub.__version__)),
            ("tokenizers", str(tokenizers.__version__)),
            ("transformers", str(transformers.__version__)),
        )

    @staticmethod
    def _optional_token_id(tokenizer: Any, field: str) -> int | None:
        value = getattr(tokenizer, field, None)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TokenizerConfigurationError(f"Tokenizer {field} is invalid.")
        return value

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]:
        result: object = self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
        if not isinstance(result, Sequence):
            raise TokenizerEncodingError("Tokenizer backend returned no token sequence.")
        return result  # type: ignore[return-value]


def load_local_huggingface_tokenizer(
    spec: TokenizerSpec,
    asset_directory: str | Path,
    *,
    loader: Callable[..., Any] | None = None,
) -> FaithfulTokenizer:
    """Load the exact local assets with network access and remote code disabled.

    ``tokenizer_id`` and ``revision`` remain the scientific identity; the local
    directory supplies the pinned bytes whose fingerprints are recorded.  A
    missing or incompatible directory raises instead of consulting the network
    or substituting another tokenizer.
    """

    asset_path = Path(asset_directory)
    manifest = fingerprint_local_assets(asset_path)
    resolved = asset_path.resolve(strict=True)
    if loader is None:
        from transformers import AutoTokenizer

        loader = AutoTokenizer.from_pretrained
    try:
        tokenizer = loader(
            str(resolved),
            revision=spec.revision,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as exc:
        raise TokenizerAssetUnavailableError(
            f"Pinned tokenizer {spec.tokenizer_id!r} at revision {spec.revision!r} "
            "could not be loaded from local assets; no download or fallback was attempted."
        ) from exc
    return FaithfulTokenizer(spec, _HuggingFaceBackend(tokenizer, manifest))
