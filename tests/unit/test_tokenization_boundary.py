from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from watermark_lab.run_engine.tokenizer_meta import tokenizer_metadata
from watermark_lab.tokenization import (
    AssetFileFingerprint,
    FaithfulTokenizer,
    NormalizationPolicy,
    SpecialTokenPolicy,
    TokenizerAssetManifest,
    TokenizerAssetUnavailableError,
    TokenizerConfigurationError,
    TokenizerEncodingError,
    TokenizerSpec,
    load_local_huggingface_tokenizer,
    synthetic_numeric_tokenizer,
)


def _manifest(content: bytes = b"tiny-tokenizer-fixture") -> TokenizerAssetManifest:
    digest = hashlib.sha256(content).hexdigest()
    return TokenizerAssetManifest(
        source="injected-test-backend",
        files=(
            AssetFileFingerprint(
                name="tiny-tokenizer.json",
                sha256=digest,
                size_bytes=len(content),
            ),
        ),
        aggregate_sha256=digest,
    )


@dataclass(slots=True)
class TinyBackend:
    backend_name: str = "tiny-injected-tokenizer"
    vocab_size: int = 512
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2
    runtime_versions: tuple[tuple[str, str], ...] = (("tiny-backend", "test-v1"),)
    asset_manifest: TokenizerAssetManifest = field(default_factory=_manifest)
    add_special_tokens_calls: list[bool] = field(default_factory=list)

    def encode(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        self.add_special_tokens_calls.append(add_special_tokens)
        ids = tuple(ord(character) % 500 + 3 for character in text)
        if add_special_tokens:
            return (1, *ids, 2)
        return ids


def _spec(
    *,
    locale: str = "zh-Hant-TW",
    special_tokens: SpecialTokenPolicy | None = None,
) -> TokenizerSpec:
    return TokenizerSpec(
        tokenizer_id="org/tiny-traditional-chinese",
        revision="0123456789abcdef",
        normalization=NormalizationPolicy(locale=locale),
        special_tokens=special_tokens or SpecialTokenPolicy(),
    )


def test_encoding_is_deterministic_and_preserves_zh_hant_tw() -> None:
    backend = TinyBackend()
    tokenizer = FaithfulTokenizer(_spec(), backend)
    traditional = "繁體中文與臺灣，後臺資料。"

    first = tokenizer.encode(traditional)
    second = tokenizer.encode(traditional)

    assert first == second
    assert first.normalized_text == traditional
    assert "臺灣" in first.normalized_text
    assert first.token_ids
    assert first.provenance.tokenizer_id == "org/tiny-traditional-chinese"
    assert first.provenance.revision == "0123456789abcdef"
    assert first.provenance.normalization_policy == "NFC-preserve-whitespace-no-locale-fallback-v1"
    assert len(first.encoding_sha256) == 64
    assert backend.add_special_tokens_calls == [False, False]


def test_normalization_is_explicit_nfc_and_does_not_rewrite_whitespace() -> None:
    tokenizer = FaithfulTokenizer(_spec(locale="und"), TinyBackend())

    result = tokenizer.encode("e\u0301  \n臺")

    assert result.normalized_text == "é  \n臺"
    assert result.input_sha256 != result.normalized_text_sha256


def test_zh_cn_cannot_be_configured_as_locale_or_fallback() -> None:
    with pytest.raises(TokenizerConfigurationError, match="cannot substitute zh-CN"):
        NormalizationPolicy(locale="zh-CN")
    with pytest.raises(TokenizerConfigurationError, match="fallback is prohibited"):
        NormalizationPolicy(locale="zh-Hant-TW", fallback_locale="zh-CN")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("policy", "expected", "backend_specials"),
    [
        (SpecialTokenPolicy(), (68,), False),
        (SpecialTokenPolicy(add_bos_token=True), (1, 68), False),
        (SpecialTokenPolicy(add_eos_token=True), (68, 2), False),
        (SpecialTokenPolicy(add_bos_token=True, add_eos_token=True), (1, 68, 2), False),
        (SpecialTokenPolicy(add_model_special_tokens=True), (1, 68, 2), True),
    ],
)
def test_special_token_behavior_is_explicit(
    policy: SpecialTokenPolicy,
    expected: tuple[int, ...],
    backend_specials: bool,
) -> None:
    backend = TinyBackend()
    tokenizer = FaithfulTokenizer(_spec(special_tokens=policy), backend)

    assert tokenizer.encode("A").token_ids == expected
    assert backend.add_special_tokens_calls == [backend_specials]


def test_conflicting_or_unavailable_special_tokens_fail_closed() -> None:
    with pytest.raises(TokenizerConfigurationError, match="cannot be combined"):
        SpecialTokenPolicy(add_model_special_tokens=True, add_bos_token=True)

    backend = TinyBackend(bos_token_id=None)
    tokenizer = FaithfulTokenizer(
        _spec(special_tokens=SpecialTokenPolicy(add_bos_token=True)),
        backend,
    )
    with pytest.raises(TokenizerEncodingError, match="no BOS token ID"):
        tokenizer.encode("A")


def test_identity_revision_assets_and_policy_change_fingerprint() -> None:
    first = FaithfulTokenizer(_spec(), TinyBackend())
    same = FaithfulTokenizer(_spec(), TinyBackend())
    changed_revision = FaithfulTokenizer(
        TokenizerSpec(
            tokenizer_id=first.spec.tokenizer_id,
            revision="different-revision",
            normalization=first.spec.normalization,
            special_tokens=first.spec.special_tokens,
        ),
        TinyBackend(),
    )
    changed_assets = FaithfulTokenizer(_spec(), TinyBackend(asset_manifest=_manifest(b"changed")))
    changed_runtime = FaithfulTokenizer(
        _spec(),
        TinyBackend(runtime_versions=(("tiny-backend", "test-v2"),)),
    )

    assert (
        first.provenance.tokenizer_fingerprint_sha256
        == same.provenance.tokenizer_fingerprint_sha256
    )
    assert (
        first.provenance.tokenizer_fingerprint_sha256
        != changed_revision.provenance.tokenizer_fingerprint_sha256
    )
    assert (
        first.provenance.tokenizer_fingerprint_sha256
        != changed_assets.provenance.tokenizer_fingerprint_sha256
    )
    assert (
        first.provenance.tokenizer_fingerprint_sha256
        != changed_runtime.provenance.tokenizer_fingerprint_sha256
    )


def test_pinned_backend_version_mismatch_fails_closed() -> None:
    spec = TokenizerSpec(
        tokenizer_id="org/tiny-traditional-chinese",
        revision="0123456789abcdef",
        normalization=NormalizationPolicy(locale="zh-Hant-TW"),
        expected_backend_versions=(("tiny-backend", "required-v2"),),
    )

    with pytest.raises(TokenizerEncodingError, match="no runtime substitution"):
        FaithfulTokenizer(spec, TinyBackend())


@pytest.mark.parametrize("revision", ["", "   "])
def test_tokenizer_revision_is_required(revision: str) -> None:
    with pytest.raises(TokenizerConfigurationError, match="revision is required"):
        TokenizerSpec(tokenizer_id="org/model", revision=revision)


@pytest.mark.parametrize("invalid_ids", [(True,), (-1,), (512,), (1.5,)])
def test_invalid_backend_token_ids_fail_closed(invalid_ids: tuple[object, ...]) -> None:
    class InvalidBackend(TinyBackend):
        def encode(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
            del text, add_special_tokens
            return cast(tuple[int, ...], invalid_ids)

    backend = InvalidBackend()
    tokenizer = FaithfulTokenizer(_spec(), backend)

    with pytest.raises(TokenizerEncodingError):
        tokenizer.encode("A")


def test_synthetic_numeric_tokenizer_is_strict_and_explicit() -> None:
    tokenizer = synthetic_numeric_tokenizer(vocab_size=32)

    result = tokenizer.encode("12 7 0")

    assert result.token_ids == (12, 7, 0)
    assert result.provenance.tokenizer_id == "synthetic-numeric-v1"
    assert result.provenance.revision == "fixture-v1"
    assert result.provenance.assets.source == "embedded-synthetic-test-fixture"
    with pytest.raises(TokenizerEncodingError, match="ASCII decimal"):
        tokenizer.encode("not-a-token-id")
    with pytest.raises(TokenizerEncodingError, match="outside vocabulary"):
        tokenizer.encode("32")


def test_run_engine_metadata_labels_legacy_and_synthetic_implementations() -> None:
    assert tokenizer_metadata("gpt2")["implementation"] == "legacy-minimal-placeholder"
    assert (
        tokenizer_metadata("synthetic-numeric-v1")["implementation"]
        == "explicit-synthetic-test-fixture"
    )


class TinyFastTokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 512

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        core = [ord(character) % 500 + 3 for character in text]
        return [1, *core, 2] if add_special_tokens else core


def test_local_loader_is_offline_only_and_records_asset_fingerprints(tmp_path: Path) -> None:
    asset_dir = tmp_path / "tokenizer-assets"
    asset_dir.mkdir()
    (asset_dir / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = []

    def loader(path: str, **kwargs: object) -> TinyFastTokenizer:
        calls.append((path, kwargs))
        return TinyFastTokenizer()

    tokenizer = load_local_huggingface_tokenizer(_spec(), asset_dir, loader=loader)
    result = tokenizer.encode("臺")

    assert calls == [
        (
            str(asset_dir.resolve()),
            {
                "revision": "0123456789abcdef",
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]
    assert result.provenance.assets.files[0].name == "tokenizer.json"
    assert result.provenance.assets.files[0].size_bytes > 0
    assert result.provenance.backend == "transformers-fast:TinyFastTokenizer"


def test_huggingface_download_cache_metadata_is_not_a_tokenizer_asset(tmp_path: Path) -> None:
    asset_dir = tmp_path / "tokenizer-assets"
    asset_dir.mkdir()
    (asset_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    cache = asset_dir / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "download.metadata").write_text("retrieved-at-machine-time", encoding="utf-8")

    tokenizer = load_local_huggingface_tokenizer(
        _spec(),
        asset_dir,
        loader=lambda _path, **_kwargs: TinyFastTokenizer(),
    )

    assert [item.name for item in tokenizer.provenance.assets.files] == ["tokenizer.json"]


def test_unavailable_assets_raise_before_loader_and_never_fallback(tmp_path: Path) -> None:
    calls = 0

    def loader(_path: str, **_kwargs: object) -> TinyFastTokenizer:
        nonlocal calls
        calls += 1
        return TinyFastTokenizer()

    with pytest.raises(TokenizerAssetUnavailableError, match="unavailable"):
        load_local_huggingface_tokenizer(_spec(), tmp_path / "missing", loader=loader)
    assert calls == 0


def test_local_loader_wraps_failure_without_trying_another_tokenizer(tmp_path: Path) -> None:
    asset_dir = tmp_path / "tokenizer-assets"
    asset_dir.mkdir()
    (asset_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    calls = 0

    def loader(_path: str, **_kwargs: object) -> TinyFastTokenizer:
        nonlocal calls
        calls += 1
        raise OSError("fixture load failure")

    with pytest.raises(TokenizerAssetUnavailableError, match="no download or fallback"):
        load_local_huggingface_tokenizer(_spec(), asset_dir, loader=loader)
    assert calls == 1
