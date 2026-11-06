from __future__ import annotations

import json

from typer.testing import CliRunner

from watermark_lab.cli import app


def test_runs_command_reads_canonical_artifact_workspace() -> None:
    result = CliRunner().invoke(app, ["runs", "--limit", "2"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["runs"]) <= 2
    assert payload["locale"] == "en"


def test_artifacts_command_exposes_amendment_mode() -> None:
    result = CliRunner().invoke(
        app, ["artifacts", "--limit", "1", "--history-mode", "amend-in-place"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["artifacts"]) <= 1
    assert payload["history_mode"] == "amend-in-place"


def test_workspace_policy_canonicalizes_zh_tw() -> None:
    result = CliRunner().invoke(app, ["workspace-policy", "--locale", "zh-TW"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["locale"] == "zh-Hant-TW"
    assert payload["locale_fallback_chain"] == ["zh-Hant-TW", "zh-TW", "zh-Hant", "en"]
    assert payload["locale_policy"]["simplified_chinese_allowed"] is False


def test_workspace_policy_rejects_zh_cn() -> None:
    result = CliRunner().invoke(app, ["workspace-policy", "--locale", "zh-CN"])

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert "prohibited" in payload["error"]


def test_research_features_command_exposes_all_choices_and_zh_tw_policy() -> None:
    result = CliRunner().invoke(
        app,
        [
            "research-features",
            "--locale",
            "zh-TW",
            "--select",
            "edit-receipt",
            "--select",
            "translingua-tw",
            "--payload-identity",
            "person-account-reference",
            "--proof-backend",
            "remote-tee",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["feature_id"] for item in payload["features"]] == [
        "edit-receipt",
        "polymark-demixing",
        "robust-fragile",
        "translingua-tw",
    ]
    assert payload["locale"] == "zh-Hant-TW"
    assert payload["research_policy"]["default_execution_target"] == "api-first-hybrid"
    assert payload["research_policy"]["proof_roadmap"][-1]["backend"] == "zero-knowledge"
    assert payload["selection"]["selected_features"] == ["edit-receipt", "translingua-tw"]
    assert payload["selection"]["payload_identity_mode"] == "person-account-reference"
    assert payload["selection"]["proof_status"] == "next-implementation"


def test_tee_simulator_self_test_is_explicitly_non_promotable() -> None:
    result = CliRunner().invoke(app, ["tee-simulator-self-test"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "passed"
    assert payload["backend"] == "tee-simulator"
    assert payload["promotion_eligible"] is False
    assert any("not hardware attestation" in warning for warning in payload["warnings"])
