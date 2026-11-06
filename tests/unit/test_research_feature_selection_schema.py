from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (ROOT / "schemas" / "research-feature-selection.schema.json").read_text(encoding="utf-8")
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def _selection() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "selection_id": "research-selection-01",
        "selected_features": [
            "edit-receipt",
            "polymark-demixing",
            "robust-fragile",
            "translingua-tw",
        ],
        "execution_target": "api-first-hybrid",
        "execution_profile_id": "workstation-api-first",
        "payload_identity_mode": "person-account-reference",
        "identity_reference": {
            "reference_id": "ref_project_12345",
            "authority_record_id": "auth_record_12345",
        },
        "proof_backend": "tee-simulator",
        "locale": "en",
        "target_locale": "zh-Hant-TW",
        "created_at": "2026-08-14T12:00:00Z",
        "scientific_status": "selection-recorded-not-yet-promoted",
    }


def test_all_four_features_and_person_reference_validate() -> None:
    assert list(VALIDATOR.iter_errors(_selection())) == []


def test_person_reference_is_required_but_raw_identity_is_rejected() -> None:
    missing = _selection()
    missing.pop("identity_reference")
    assert any("identity_reference" in error.message for error in VALIDATOR.iter_errors(missing))

    raw_identity = _selection()
    raw_identity["identity_reference"] = {
        "reference_id": "ref_project_12345",
        "authority_record_id": "auth_record_12345",
        "email": "must-not-be-stored@example.invalid",
    }
    assert any("Additional properties" in error.message for error in VALIDATOR.iter_errors(raw_identity))


def test_translingua_requires_explicit_zh_hant_tw_target() -> None:
    selection = _selection()
    selection.pop("target_locale")

    assert any("target_locale" in error.message for error in VALIDATOR.iter_errors(selection))


def test_duplicate_features_are_rejected() -> None:
    selection = _selection()
    selection["selected_features"] = ["edit-receipt", "edit-receipt"]

    assert any("non-unique" in error.message for error in VALIDATOR.iter_errors(selection))
