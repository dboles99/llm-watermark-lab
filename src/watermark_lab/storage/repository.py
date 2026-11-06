"""Repository abstractions and a deterministic in-memory implementation."""

from __future__ import annotations

from threading import RLock
from typing import Protocol
from uuid import UUID, uuid4

from watermark_lab.storage.models import (
    AmendmentEntry,
    ArtifactRecord,
    RunRecord,
    RunStatus,
    canonical_artifact_path,
    utc_now,
)


class RecordNotFoundError(LookupError):
    """Raised when a requested persistent record does not exist."""


class RecordConflictError(ValueError):
    """Raised when a write conflicts with persistence or lineage invariants."""


class FrozenArtifactError(RecordConflictError):
    """Raised when attempting to amend a frozen artifact."""


class StorageRepository(Protocol):
    """Persistence contract shared by local and future PostgreSQL adapters."""

    def create_run(self, run: RunRecord) -> RunRecord: ...
    def get_run(self, run_id: UUID) -> RunRecord: ...
    def list_runs(self) -> tuple[RunRecord, ...]: ...
    def update_run_status(self, run_id: UUID, status: RunStatus) -> RunRecord: ...
    def add_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord: ...
    def get_artifact(self, artifact_id: UUID) -> ArtifactRecord: ...
    def list_artifacts(self, run_id: UUID | None = None) -> tuple[ArtifactRecord, ...]: ...
    def freeze_artifact(self, artifact_id: UUID) -> ArtifactRecord: ...
    def supersede_artifact(
        self,
        artifact_id: UUID,
        *,
        relative_name: str,
        sha256: str,
        byte_size: int,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> ArtifactRecord: ...
    def amend_artifact(
        self,
        artifact_id: UUID,
        *,
        sha256: str,
        byte_size: int,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> tuple[ArtifactRecord, AmendmentEntry]: ...
    def list_amendments(self, artifact_id: UUID) -> tuple[AmendmentEntry, ...]: ...


class InMemoryStorageRepository:
    """Thread-safe repository suitable for tests and local ephemeral sessions."""

    def __init__(self) -> None:
        self._runs: dict[UUID, RunRecord] = {}
        self._artifacts: dict[UUID, ArtifactRecord] = {}
        self._amendments: dict[UUID, list[AmendmentEntry]] = {}
        self._lock = RLock()

    def create_run(self, run: RunRecord) -> RunRecord:
        with self._lock:
            if run.id in self._runs:
                raise RecordConflictError(f"run {run.id} already exists")
            if run.parent_run_id is not None and run.parent_run_id not in self._runs:
                raise RecordNotFoundError(f"parent run {run.parent_run_id} does not exist")
            self._runs[run.id] = run.model_copy(deep=True)
            return run.model_copy(deep=True)

    def get_run(self, run_id: UUID) -> RunRecord:
        with self._lock:
            try:
                return self._runs[run_id].model_copy(deep=True)
            except KeyError as error:
                raise RecordNotFoundError(f"run {run_id} does not exist") from error

    def list_runs(self) -> tuple[RunRecord, ...]:
        with self._lock:
            records = sorted(self._runs.values(), key=lambda item: (item.created_at, str(item.id)))
            return tuple(item.model_copy(deep=True) for item in records)

    def update_run_status(self, run_id: UUID, status: RunStatus) -> RunRecord:
        with self._lock:
            current = self.get_run(run_id)
            terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
            if current.status in terminal:
                raise RecordConflictError("terminal run status is immutable")
            updated = current.model_copy(
                update={"status": status, "completed_at": utc_now() if status in terminal else None}
            )
            # Revalidate because model_copy deliberately skips model validation.
            updated = RunRecord.model_validate(updated.model_dump())
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    def add_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self._lock:
            if artifact.id in self._artifacts:
                raise RecordConflictError(f"artifact {artifact.id} already exists")
            if artifact.run_id not in self._runs:
                raise RecordNotFoundError(f"run {artifact.run_id} does not exist")
            if any(item.path == artifact.path for item in self._artifacts.values()):
                raise RecordConflictError(f"artifact path {artifact.path!r} already exists")
            if artifact.version != 1 or artifact.supersedes_artifact_id is not None:
                raise RecordConflictError("new artifacts must start at version 1")
            self._artifacts[artifact.id] = artifact.model_copy(deep=True)
            return artifact.model_copy(deep=True)

    def get_artifact(self, artifact_id: UUID) -> ArtifactRecord:
        with self._lock:
            try:
                return self._artifacts[artifact_id].model_copy(deep=True)
            except KeyError as error:
                raise RecordNotFoundError(f"artifact {artifact_id} does not exist") from error

    def list_artifacts(self, run_id: UUID | None = None) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            records = list(self._artifacts.values())
            if run_id is not None:
                records = [item for item in records if item.run_id == run_id]
            ordered = sorted(records, key=lambda item: (item.created_at, item.version, str(item.id)))
            return tuple(item.model_copy(deep=True) for item in ordered)

    def freeze_artifact(self, artifact_id: UUID) -> ArtifactRecord:
        with self._lock:
            current = self.get_artifact(artifact_id)
            if current.frozen:
                return current
            updated = ArtifactRecord.model_validate(
                current.model_copy(update={"frozen": True}).model_dump()
            )
            self._artifacts[artifact_id] = updated
            return updated.model_copy(deep=True)

    def supersede_artifact(
        self,
        artifact_id: UUID,
        *,
        relative_name: str,
        sha256: str,
        byte_size: int,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> ArtifactRecord:
        """Create a new lineage version and freeze its predecessor (the default history mode)."""
        if not reason.strip():
            raise ValueError("supersession reason must not be empty")
        with self._lock:
            current = self.get_artifact(artifact_id)
            if any(
                item.supersedes_artifact_id == artifact_id for item in self._artifacts.values()
            ):
                raise RecordConflictError("an artifact version may only be superseded once")
            new_path = canonical_artifact_path(current.run_id, relative_name)
            if new_path == current.path:
                raise RecordConflictError("supersession requires a new immutable artifact path")
            if any(item.path == new_path for item in self._artifacts.values()):
                raise RecordConflictError(f"artifact path {new_path!r} already exists")
            replacement = ArtifactRecord(
                id=uuid4(),
                run_id=current.run_id,
                lineage_id=current.lineage_id,
                supersedes_artifact_id=current.id,
                version=current.version + 1,
                artifact_type=current.artifact_type,
                path=new_path,
                sha256=sha256,
                byte_size=byte_size,
                metadata=metadata if metadata is not None else current.metadata,
                change_reason=reason,
            )
            self._artifacts[current.id] = ArtifactRecord.model_validate(
                current.model_copy(update={"frozen": True}).model_dump()
            )
            self._artifacts[replacement.id] = replacement
            return replacement.model_copy(deep=True)

    def amend_artifact(
        self,
        artifact_id: UUID,
        *,
        sha256: str,
        byte_size: int,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> tuple[ArtifactRecord, AmendmentEntry]:
        """Amend an unfrozen artifact and append a before/after hash audit entry."""
        if not reason.strip():
            raise ValueError("amendment reason must not be empty")
        with self._lock:
            current = self.get_artifact(artifact_id)
            if current.frozen:
                raise FrozenArtifactError("frozen artifacts may only be superseded")
            sequence = len(self._amendments.get(artifact_id, [])) + 1
            entry = AmendmentEntry(
                artifact_id=artifact_id,
                sequence=sequence,
                before_sha256=current.sha256,
                after_sha256=sha256,
                before_byte_size=current.byte_size,
                after_byte_size=byte_size,
                reason=reason,
            )
            updated = ArtifactRecord.model_validate(
                current.model_copy(
                    update={
                        "sha256": sha256,
                        "byte_size": byte_size,
                        "metadata": metadata if metadata is not None else current.metadata,
                        "change_reason": reason,
                    }
                ).model_dump()
            )
            self._artifacts[artifact_id] = updated
            self._amendments.setdefault(artifact_id, []).append(entry)
            return updated.model_copy(deep=True), entry.model_copy(deep=True)

    def list_amendments(self, artifact_id: UUID) -> tuple[AmendmentEntry, ...]:
        with self._lock:
            if artifact_id not in self._artifacts:
                raise RecordNotFoundError(f"artifact {artifact_id} does not exist")
            return tuple(item.model_copy(deep=True) for item in self._amendments.get(artifact_id, []))
