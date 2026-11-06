from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (ROOT / "schemas" / "tokenizer-manifest.schema.json").read_text(encoding="utf-8")
)
MANIFEST = json.loads(
    (
        ROOT
        / "config"
        / "tokenizers"
        / "openai-community-gpt2-607a30d7.json"
    ).read_text(encoding="utf-8")
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def _errors(value: object) -> list[str]:
    return [error.message for error in VALIDATOR.iter_errors(value)]


def test_pinned_gpt2_tokenizer_manifest_validates() -> None:
    Draft202012Validator.check_schema(SCHEMA)
    assert _errors(MANIFEST) == []


def test_zh_cn_or_fallback_is_rejected() -> None:
    wrong_locale = dict(MANIFEST)
    wrong_locale["evaluated_locales"] = ["zh-CN"]
    assert any("is not one of" in message for message in _errors(wrong_locale))

    fallback = dict(MANIFEST)
    fallback["fallback_locales"] = ["zh-CN"]
    assert any("was expected" in message for message in _errors(fallback))


def test_mutable_revision_is_rejected() -> None:
    manifest = dict(MANIFEST)
    manifest["revision"] = "main"

    assert any("does not match" in message for message in _errors(manifest))
