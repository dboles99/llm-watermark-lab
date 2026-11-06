from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from watermark_lab.i18n import (
    LocalePolicyError,
    canonicalize_locale,
    fallback_chain,
    validate_fallbacks,
)
from watermark_lab.services.product_workspace import ProductWorkspaceService


class FakeRepository:
    def list_runs(self) -> list[dict[str, Any]]:
        return [
            {"run_id": "older", "status": "completed", "updated_at": "2026-08-13T00:00:00Z"},
            {"run_id": "newer", "status": "pending", "updated_at": "2026-08-14T00:00:00Z"},
        ]

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": "report-1",
                "artifact_type": "report",
                "status": "frozen",
                "updated_at": "2026-08-14T00:00:00Z",
                "path": "runs/t03/report.json",
            }
        ]


def test_service_uses_injected_repository_and_sorts_runs() -> None:
    service = ProductWorkspaceService(FakeRepository())

    assert [run.run_id for run in service.list_runs()] == ["newer", "older"]
    assert service.list_artifacts()[0].artifact_id == "report-1"


def test_service_duck_types_typed_storage_records_without_importing_storage() -> None:
    @dataclass
    class RunRecord:
        id: str = "typed-run"
        status: str = "completed"
        method_id: str = "kgw"
        created_at: datetime = datetime(2026, 8, 14, tzinfo=UTC)
        completed_at: datetime | None = None

    @dataclass
    class ArtifactRecord:
        id: str = "typed-artifact"
        artifact_type: str = "report"
        path: str = "project/artifacts/typed-run/report.json"
        frozen: bool = True
        created_at: datetime = datetime(2026, 8, 14, tzinfo=UTC)

    class TypedRepository:
        def list_runs(self) -> list[RunRecord]:
            return [RunRecord()]

        def list_artifacts(self) -> list[ArtifactRecord]:
            return [ArtifactRecord()]

    service = ProductWorkspaceService(TypedRepository())
    assert service.list_runs()[0].run_id == "typed-run"
    assert service.list_runs()[0].updated_at == "2026-08-14T00:00:00+00:00"
    assert service.list_artifacts()[0].status == "frozen"


def test_history_modes_and_policy_are_exposed() -> None:
    service = ProductWorkspaceService(FakeRepository(), history_mode="amend-in-place")

    policy = service.policy()
    assert policy["history_mode"] == "amend-in-place"
    assert policy["history_modes"] == ["supersede-and-preserve", "amend-in-place"]
    assert "superseded" in str(policy["history_rules"]["frozen_artifacts"])


def test_traditional_chinese_alias_is_canonicalized_with_safe_chain() -> None:
    assert canonicalize_locale("zh-TW") == "zh-Hant-TW"
    assert fallback_chain("zh-TW") == ("zh-Hant-TW", "zh-TW", "zh-Hant", "en")


@pytest.mark.parametrize("locale", ["zh-CN", "zh_CN", "zh-Hans", "zh-SG"])
def test_simplified_chinese_is_explicitly_rejected(locale: str) -> None:
    with pytest.raises(LocalePolicyError, match="prohibited"):
        canonicalize_locale(locale)

    with pytest.raises(LocalePolicyError, match="prohibited"):
        validate_fallbacks(["zh-Hant-TW", locale, "en"])


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ProductWorkspaceService(FakeRepository()).list_runs(limit=0)


def test_all_four_frontier_features_and_governance_choices_are_exposed() -> None:
    service = ProductWorkspaceService(FakeRepository())

    assert [item.feature_id.value for item in service.list_research_features()] == [
        "edit-receipt",
        "polymark-demixing",
        "robust-fragile",
        "translingua-tw",
    ]
    policy = service.research_policy()
    assert policy["default_execution_target"] == "api-first-hybrid"
    assert policy["payload_identity_modes"] == [
        "pseudonymous-project-workflow",
        "person-account-reference",
    ]
    assert [item["backend"] for item in policy["proof_roadmap"]] == [
        "tee-simulator",
        "remote-tee",
        "zero-knowledge",
    ]
    assert "Never place" in policy["payload_identity_rules"]["raw_identity"]


def test_research_selection_accepts_multiple_features_and_both_identity_modes() -> None:
    service = ProductWorkspaceService(FakeRepository(), locale="zh-TW")

    selection = service.research_selection(
        ["edit-receipt", "translingua-tw"],
        execution_target="api-only",
        payload_identity_mode="person-account-reference",
        proof_backend="remote-tee",
    )
    assert selection == {
        "selected_features": ["edit-receipt", "translingua-tw"],
        "execution_target": "api-only",
        "payload_identity_mode": "person-account-reference",
        "proof_backend": "remote-tee",
        "proof_status": "next-implementation",
        "scientific_status": "selection-recorded-not-yet-promoted",
        "locale": "zh-Hant-TW",
        "target_locale": "zh-Hant-TW",
    }


@pytest.mark.parametrize(
    "features,match",
    [([], "at least one"), (["edit-receipt", "edit-receipt"], "unique")],
)
def test_research_selection_rejects_empty_or_duplicate_features(
    features: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ProductWorkspaceService(FakeRepository()).research_selection(features)
