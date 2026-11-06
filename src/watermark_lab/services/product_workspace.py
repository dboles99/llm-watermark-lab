"""Shared read boundary for the CLI and desktop product workspace."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from watermark_lab.i18n import LocalePolicy, canonicalize_locale, fallback_chain


class HistoryMode(StrEnum):
    """Supported artifact correction policies."""

    SUPERSEDE_AND_PRESERVE = "supersede-and-preserve"
    AMEND_IN_PLACE = "amend-in-place"


class FrontierFeature(StrEnum):
    """Selectable frontier-research modules exposed by the product shell."""

    EDIT_RECEIPT = "edit-receipt"
    POLYMARK_DEMIXING = "polymark-demixing"
    ROBUST_FRAGILE = "robust-fragile"
    TRANSLINGUA_TW = "translingua-tw"


class ExecutionTarget(StrEnum):
    """Where an experiment is intended to execute."""

    API_FIRST_HYBRID = "api-first-hybrid"
    LOCAL_ONLY = "local-only"
    API_ONLY = "api-only"


class PayloadIdentityMode(StrEnum):
    """Governed payload identity choices; neither mode stores raw identity bits."""

    PSEUDONYMOUS = "pseudonymous-project-workflow"
    PERSON_ACCOUNT = "person-account-reference"


class ProofBackend(StrEnum):
    """Proof backends in their approved implementation sequence."""

    TEE_SIMULATOR = "tee-simulator"
    REMOTE_TEE = "remote-tee"
    ZERO_KNOWLEDGE = "zero-knowledge"


@dataclass(frozen=True, slots=True)
class ResearchFeatureSummary:
    feature_id: FrontierFeature
    label: str
    maturity: str
    description: str
    dependency: str

    def as_dict(self) -> dict[str, str]:
        return {
            "feature_id": self.feature_id.value,
            "label": self.label,
            "maturity": self.maturity,
            "description": self.description,
            "dependency": self.dependency,
        }


FRONTIER_FEATURES = (
    ResearchFeatureSummary(
        FrontierFeature.EDIT_RECEIPT,
        "EditReceipt and counterfactual certificate",
        "skeleton",
        "Signed edit lineage with bounded, uncertainty-aware robustness certificates.",
        "T06 calibrated detectors and transformation registry",
    ),
    ResearchFeatureSummary(
        FrontierFeature.POLYMARK_DEMIXING,
        "PolyMark demixing",
        "skeleton",
        "Localize and demix compatible provider, key, and epoch signals in mixed text.",
        "T06 localization null and collision benchmark",
    ),
    ResearchFeatureSummary(
        FrontierFeature.ROBUST_FRAGILE,
        "Robust-plus-fragile semantic integrity",
        "skeleton-high-risk",
        "Preserve provenance through benign edits while localizing material semantic changes.",
        "T06 quality gates and a preregistered semantic edit ontology",
    ),
    ResearchFeatureSummary(
        FrontierFeature.TRANSLINGUA_TW,
        "Translingua-TW handoff",
        "skeleton",
        "Signed English-to-zh-Hant-TW provenance handoff with Taiwan-language invariants.",
        "T06 Taiwan corpus and native-review panel",
    ),
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    status: str = "unknown"
    method_id: str = ""
    updated_at: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    artifact_id: str
    artifact_type: str = "file"
    status: str = "unknown"
    updated_at: str = ""
    path: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@runtime_checkable
class ProductWorkspaceRepository(Protocol):
    """Minimal repository contract; database adapters can implement this later."""

    def list_runs(self) -> Sequence[Any]: ...

    def list_artifacts(self) -> Sequence[Any]: ...


def _iso_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


class ProjectArtifactRepository:
    """Read-only adapter over the canonical ``project/artifacts`` evidence root."""

    def __init__(self, artifact_root: str | Path = Path("project/artifacts")) -> None:
        self.artifact_root = Path(artifact_root)

    def list_runs(self) -> Sequence[RunSummary]:
        runs_root = self.artifact_root / "runs"
        if not runs_root.is_dir():
            return ()

        summaries: list[RunSummary] = []
        for run_directory in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifests = sorted(run_directory.rglob("run-manifest.json"))
            metadata = sorted(run_directory.rglob("metadata.json"))
            source = manifests[-1] if manifests else (metadata[-1] if metadata else run_directory)
            payload = _load_json(source) if source.is_file() else {}
            summaries.append(
                RunSummary(
                    run_id=str(payload.get("run_id") or payload.get("task_id") or run_directory.name),
                    status=str(payload.get("status") or "unknown"),
                    method_id=str(payload.get("method_id") or payload.get("detector_id") or ""),
                    updated_at=_iso_timestamp(source),
                    source=source.relative_to(self.artifact_root).as_posix(),
                )
            )
        return summaries

    def list_artifacts(self) -> Sequence[ArtifactSummary]:
        if not self.artifact_root.is_dir():
            return ()

        summaries: list[ArtifactSummary] = []
        for path in self.artifact_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.artifact_root).as_posix()
            payload = _load_json(path) if path.suffix.lower() == ".json" else {}
            summaries.append(
                ArtifactSummary(
                    artifact_id=str(payload.get("artifact_id") or relative),
                    artifact_type=str(payload.get("artifact_type") or path.stem),
                    status=str(payload.get("status") or "unknown"),
                    updated_at=_iso_timestamp(path),
                    path=relative,
                )
            )
        return summaries


class ProductWorkspaceService:
    """Application service shared by product front ends."""

    def __init__(
        self,
        repository: ProductWorkspaceRepository | None = None,
        *,
        locale: str = "en",
        history_mode: HistoryMode | str = HistoryMode.SUPERSEDE_AND_PRESERVE,
    ) -> None:
        self.repository = repository or ProjectArtifactRepository()
        self._locale = canonicalize_locale(locale)
        self._history_mode = self._parse_history_mode(history_mode)

    @staticmethod
    def _parse_history_mode(value: HistoryMode | str) -> HistoryMode:
        try:
            return HistoryMode(value)
        except ValueError as exc:
            choices = ", ".join(item.value for item in HistoryMode)
            raise ValueError(f"Unsupported history mode {value!r}; choose {choices}.") from exc

    @property
    def locale(self) -> str:
        return self._locale

    @property
    def history_mode(self) -> HistoryMode:
        return self._history_mode

    def set_locale(self, locale: str) -> str:
        self._locale = canonicalize_locale(locale)
        return self._locale

    def set_history_mode(self, mode: HistoryMode | str) -> HistoryMode:
        self._history_mode = self._parse_history_mode(mode)
        return self._history_mode

    def list_runs(self, *, limit: int | None = None) -> list[RunSummary]:
        runs = [self._coerce_run(item) for item in self.repository.list_runs()]
        runs.sort(key=lambda item: item.updated_at, reverse=True)
        return runs[: self._validate_limit(limit)] if limit is not None else runs

    def list_artifacts(self, *, limit: int | None = None) -> list[ArtifactSummary]:
        artifacts = [self._coerce_artifact(item) for item in self.repository.list_artifacts()]
        artifacts.sort(key=lambda item: item.updated_at, reverse=True)
        return artifacts[: self._validate_limit(limit)] if limit is not None else artifacts

    def policy(self) -> dict[str, object]:
        return {
            "locale": self.locale,
            "locale_fallback_chain": list(fallback_chain(self.locale)),
            "locale_policy": LocalePolicy().as_dict(),
            "history_mode": self.history_mode.value,
            "history_modes": [mode.value for mode in HistoryMode],
            "history_rules": {
                HistoryMode.SUPERSEDE_AND_PRESERVE.value: (
                    "Create a new version and retain the original artifact."
                ),
                HistoryMode.AMEND_IN_PLACE.value: (
                    "Allowed only before an artifact is frozen, published, signed, "
                    "or used for promotion; record an append-only amendment entry."
                ),
                "frozen_artifacts": "Frozen artifacts can only be superseded.",
            },
        }

    def list_research_features(self) -> list[ResearchFeatureSummary]:
        """Return all user-approved frontier features without implying implementation readiness."""

        return list(FRONTIER_FEATURES)

    def research_policy(self) -> dict[str, object]:
        """Expose execution, identity, and proof choices for the research workspace."""

        return {
            "default_execution_target": ExecutionTarget.API_FIRST_HYBRID.value,
            "execution_targets": [target.value for target in ExecutionTarget],
            "local_profile_id": "workstation-api-first",
            "payload_identity_default": PayloadIdentityMode.PSEUDONYMOUS.value,
            "payload_identity_modes": [mode.value for mode in PayloadIdentityMode],
            "payload_identity_rules": {
                PayloadIdentityMode.PSEUDONYMOUS.value: (
                    "Default: embed or bind only an opaque project/workflow reference."
                ),
                PayloadIdentityMode.PERSON_ACCOUNT.value: (
                    "Opt-in: bind an opaque reference to an authorized external identity record; "
                    "require purpose, consent or other authority, retention, access, and revocation."
                ),
                "raw_identity": "Never place a name, email address, or account number in payload bits.",
            },
            "proof_roadmap": [
                {
                    "backend": ProofBackend.TEE_SIMULATOR.value,
                    "status": "implemented-development-only",
                },
                {"backend": ProofBackend.REMOTE_TEE.value, "status": "next-implementation"},
                {
                    "backend": ProofBackend.ZERO_KNOWLEDGE.value,
                    "status": "after-tee-validation",
                },
            ],
        }

    def research_selection(
        self,
        features: Sequence[FrontierFeature | str],
        *,
        execution_target: ExecutionTarget | str = ExecutionTarget.API_FIRST_HYBRID,
        payload_identity_mode: PayloadIdentityMode | str = PayloadIdentityMode.PSEUDONYMOUS,
        proof_backend: ProofBackend | str = ProofBackend.TEE_SIMULATOR,
    ) -> dict[str, object]:
        """Validate and normalize a user feature selection without starting an experiment."""

        if not features:
            raise ValueError("Select at least one frontier feature.")
        try:
            selected = [FrontierFeature(feature) for feature in features]
            target = ExecutionTarget(execution_target)
            identity = PayloadIdentityMode(payload_identity_mode)
            proof = ProofBackend(proof_backend)
        except ValueError as exc:
            raise ValueError(f"Unsupported research selection value: {exc}") from exc
        if len(selected) != len(set(selected)):
            raise ValueError("Frontier feature selections must be unique.")
        proof_status = {
            ProofBackend.TEE_SIMULATOR: "implemented-development-only",
            ProofBackend.REMOTE_TEE: "next-implementation",
            ProofBackend.ZERO_KNOWLEDGE: "after-tee-validation",
        }[proof]
        selection: dict[str, object] = {
            "selected_features": [feature.value for feature in selected],
            "execution_target": target.value,
            "payload_identity_mode": identity.value,
            "proof_backend": proof.value,
            "proof_status": proof_status,
            "scientific_status": "selection-recorded-not-yet-promoted",
            "locale": self.locale,
        }
        if FrontierFeature.TRANSLINGUA_TW in selected:
            selection["target_locale"] = "zh-Hant-TW"
        return selection

    @staticmethod
    def _validate_limit(limit: int | None) -> int:
        if limit is None:
            return 0
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return limit

    @staticmethod
    def _coerce_run(item: Any) -> RunSummary:
        if isinstance(item, RunSummary):
            return item
        if not isinstance(item, Mapping):
            completed_at = getattr(item, "completed_at", None)
            created_at = getattr(item, "created_at", "")
            return RunSummary(
                run_id=str(getattr(item, "run_id", None) or getattr(item, "id", "")),
                status=_display_value(getattr(item, "status", "unknown")),
                method_id=str(getattr(item, "method_id", "")),
                updated_at=_display_value(completed_at or created_at),
                source=str(getattr(item, "source", "")),
            )
        return RunSummary(
            run_id=str(item.get("run_id", "")),
            status=str(item.get("status", "unknown")),
            method_id=str(item.get("method_id", "")),
            updated_at=str(item.get("updated_at", "")),
            source=str(item.get("source", "")),
        )

    @staticmethod
    def _coerce_artifact(item: Any) -> ArtifactSummary:
        if isinstance(item, ArtifactSummary):
            return item
        if not isinstance(item, Mapping):
            status = "frozen" if bool(getattr(item, "frozen", False)) else "active"
            return ArtifactSummary(
                artifact_id=str(getattr(item, "artifact_id", None) or getattr(item, "id", "")),
                artifact_type=str(getattr(item, "artifact_type", "file")),
                status=status,
                updated_at=_display_value(getattr(item, "created_at", "")),
                path=str(getattr(item, "path", "")),
            )
        return ArtifactSummary(
            artifact_id=str(item.get("artifact_id", "")),
            artifact_type=str(item.get("artifact_type", "file")),
            status=str(item.get("status", "unknown")),
            updated_at=str(item.get("updated_at", "")),
            path=str(item.get("path", "")),
        )


def _display_value(value: Any) -> str:
    """Format enums and timestamps from an injected typed repository."""

    if isinstance(value, datetime):
        return value.isoformat()
    enum_value = getattr(value, "value", value)
    return str(enum_value)
