"""SQLite-backed local persistence implementing the storage repository contract."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from typing import Self
from uuid import UUID

from watermark_lab.storage.models import AmendmentEntry, ArtifactRecord, RunRecord, RunStatus
from watermark_lab.storage.repository import InMemoryStorageRepository


class SQLiteStorageRepository(InMemoryStorageRepository):
    """Durable local repository with SQLite storage and shared domain invariants.

    Records are serialized as validated JSON. Relational PostgreSQL deployment is described by
    ``database/migrations/0002_artifact_history.sql``; this adapter intentionally stays dependency-free
    for local workflows and tests.
    """

    def __init__(self, database: str | Path = ":memory:") -> None:
        super().__init__()
        self._lock = RLock()
        self._connection = sqlite3.connect(str(database), check_same_thread=False)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()
        self._load_records()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS storage_runs (
                id TEXT PRIMARY KEY,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS storage_artifacts (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                record_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES storage_runs(id)
            );
            CREATE TABLE IF NOT EXISTS storage_amendments (
                id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                UNIQUE (artifact_id, sequence),
                FOREIGN KEY (artifact_id) REFERENCES storage_artifacts(id)
            );
            """
        )
        self._connection.commit()

    def _load_records(self) -> None:
        for (payload,) in self._connection.execute("SELECT record_json FROM storage_runs"):
            run_record = RunRecord.model_validate_json(payload)
            self._runs[run_record.id] = run_record
        for (payload,) in self._connection.execute("SELECT record_json FROM storage_artifacts"):
            artifact_record = ArtifactRecord.model_validate_json(payload)
            self._artifacts[artifact_record.id] = artifact_record
        rows = self._connection.execute(
            "SELECT record_json FROM storage_amendments ORDER BY artifact_id, sequence"
        )
        for (payload,) in rows:
            entry = AmendmentEntry.model_validate_json(payload)
            self._amendments.setdefault(entry.artifact_id, []).append(entry)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _persist_run(self, run: RunRecord) -> None:
        self._connection.execute(
            """INSERT INTO storage_runs (id, record_json) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET record_json = excluded.record_json""",
            (str(run.id), run.model_dump_json()),
        )

    def _persist_artifact(self, artifact: ArtifactRecord) -> None:
        self._connection.execute(
            """INSERT INTO storage_artifacts (id, run_id, path, record_json)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   run_id = excluded.run_id,
                   path = excluded.path,
                   record_json = excluded.record_json""",
            (str(artifact.id), str(artifact.run_id), artifact.path, artifact.model_dump_json()),
        )

    def create_run(self, run: RunRecord) -> RunRecord:
        with self._lock:
            created = super().create_run(run)
            self._persist_run(created)
            self._connection.commit()
            return created

    def update_run_status(self, run_id: UUID, status: RunStatus) -> RunRecord:
        with self._lock:
            updated = super().update_run_status(run_id, status)
            self._persist_run(updated)
            self._connection.commit()
            return updated

    def add_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self._lock:
            created = super().add_artifact(artifact)
            self._persist_artifact(created)
            self._connection.commit()
            return created

    def freeze_artifact(self, artifact_id: UUID) -> ArtifactRecord:
        with self._lock:
            updated = super().freeze_artifact(artifact_id)
            self._persist_artifact(updated)
            self._connection.commit()
            return updated

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
        with self._lock:
            replacement = super().supersede_artifact(
                artifact_id,
                relative_name=relative_name,
                sha256=sha256,
                byte_size=byte_size,
                reason=reason,
                metadata=metadata,
            )
            self._persist_artifact(self._artifacts[artifact_id])
            self._persist_artifact(replacement)
            self._connection.commit()
            return replacement

    def amend_artifact(
        self,
        artifact_id: UUID,
        *,
        sha256: str,
        byte_size: int,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> tuple[ArtifactRecord, AmendmentEntry]:
        with self._lock:
            updated, entry = super().amend_artifact(
                artifact_id,
                sha256=sha256,
                byte_size=byte_size,
                reason=reason,
                metadata=metadata,
            )
            self._persist_artifact(updated)
            self._connection.execute(
                """INSERT INTO storage_amendments (id, artifact_id, sequence, record_json)
                   VALUES (?, ?, ?, ?)""",
                (str(entry.id), str(entry.artifact_id), entry.sequence, entry.model_dump_json()),
            )
            self._connection.commit()
            return updated, entry
