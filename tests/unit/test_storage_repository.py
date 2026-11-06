"""Contract tests for in-memory and SQLite storage repositories."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from watermark_lab.storage import (
    ArtifactRecord,
    FrozenArtifactError,
    InMemoryStorageRepository,
    RecordConflictError,
    RecordNotFoundError,
    RunProvenance,
    RunRecord,
    RunStatus,
    RunType,
    SQLiteStorageRepository,
    StorageRepository,
    canonical_artifact_path,
    sha256_bytes,
)


@pytest.fixture(params=["memory", "sqlite"])
def repository(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StorageRepository]:
    if request.param == "memory":
        yield InMemoryStorageRepository()
        return
    sqlite = SQLiteStorageRepository(tmp_path / "storage.sqlite3")
    try:
        yield sqlite
    finally:
        sqlite.close()


def make_run(parent_run_id: UUID | None = None) -> RunRecord:
    return RunRecord(
        parent_run_id=parent_run_id,
        run_type=RunType.CREATE,
        method_id="kgw",
        provenance=RunProvenance(source_commit="abcdef1"),
    )


def add_artifact(repository: StorageRepository, run: RunRecord, name: str = "v1.json") -> ArtifactRecord:
    content = b'{"version":1}'
    artifact = ArtifactRecord(
        run_id=run.id,
        artifact_type="manifest",
        path=canonical_artifact_path(run.id, name),
        sha256=sha256_bytes(content),
        byte_size=len(content),
    )
    return repository.add_artifact(artifact)


def test_run_parent_must_exist_and_lineage_is_read_only(repository: StorageRepository) -> None:
    missing_parent = make_run(parent_run_id=uuid4())
    with pytest.raises(RecordNotFoundError, match="parent run"):
        repository.create_run(missing_parent)

    parent = repository.create_run(make_run())
    child = repository.create_run(make_run(parent_run_id=parent.id))
    completed = repository.update_run_status(child.id, RunStatus.COMPLETED)
    assert completed.parent_run_id == parent.id
    assert completed.completed_at is not None
    with pytest.raises(RecordConflictError, match="terminal"):
        repository.update_run_status(child.id, RunStatus.FAILED)


def test_repository_returns_detached_record_data(repository: StorageRepository) -> None:
    run = repository.create_run(
        RunRecord(run_type=RunType.DETECT, method_id="kgw", config={"nested": {"value": 1}})
    )
    run.config["nested"]["value"] = 99
    assert repository.get_run(run.id).config == {"nested": {"value": 1}}


def test_supersession_is_default_immutable_history(repository: StorageRepository) -> None:
    run = repository.create_run(make_run())
    original = add_artifact(repository, run)
    replacement_content = b'{"version":2}'
    replacement = repository.supersede_artifact(
        original.id,
        relative_name="v2.json",
        sha256=sha256_bytes(replacement_content),
        byte_size=len(replacement_content),
        reason="correct parameter provenance",
    )

    assert replacement.supersedes_artifact_id == original.id
    assert replacement.lineage_id == original.lineage_id
    assert replacement.version == 2
    assert replacement.change_reason == "correct parameter provenance"
    assert repository.get_artifact(original.id).frozen is True
    with pytest.raises(RecordConflictError, match="only be superseded once"):
        repository.supersede_artifact(
            original.id,
            relative_name="v3.json",
            sha256="b" * 64,
            byte_size=2,
            reason="branching is not allowed",
        )


def test_amendment_is_append_only_and_only_allowed_while_unfrozen(
    repository: StorageRepository,
) -> None:
    run = repository.create_run(make_run())
    artifact = add_artifact(repository, run)
    before = artifact.sha256
    updated, first_entry = repository.amend_artifact(
        artifact.id,
        sha256=sha256_bytes(b'{"version":1,"fixed":true}'),
        byte_size=26,
        reason="repair typo before review",
    )
    updated, second_entry = repository.amend_artifact(
        artifact.id,
        sha256=sha256_bytes(b'{"version":1,"fixed":true,"reviewed":true}'),
        byte_size=42,
        reason="record pre-freeze review",
    )

    entries = repository.list_amendments(artifact.id)
    assert entries == (first_entry, second_entry)
    assert entries[0].before_sha256 == before
    assert entries[0].after_sha256 == entries[1].before_sha256
    assert [entry.sequence for entry in entries] == [1, 2]
    assert updated.path == artifact.path

    repository.freeze_artifact(artifact.id)
    with pytest.raises(FrozenArtifactError, match="only be superseded"):
        repository.amend_artifact(
            artifact.id,
            sha256="c" * 64,
            byte_size=4,
            reason="too late",
        )
    assert repository.list_amendments(artifact.id) == entries


def test_duplicate_paths_and_unknown_runs_are_rejected(repository: StorageRepository) -> None:
    run = repository.create_run(make_run())
    artifact = add_artifact(repository, run)
    duplicate = artifact.model_copy(update={"id": uuid4()})
    with pytest.raises(RecordConflictError, match="path"):
        repository.add_artifact(duplicate)

    unknown_run = make_run()
    orphan = ArtifactRecord(
        run_id=unknown_run.id,
        artifact_type="manifest",
        path=canonical_artifact_path(unknown_run.id, "orphan.json"),
        sha256="a" * 64,
        byte_size=1,
    )
    with pytest.raises(RecordNotFoundError, match="run"):
        repository.add_artifact(orphan)


def test_sqlite_repository_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "durable.sqlite3"
    first = SQLiteStorageRepository(database)
    run = first.create_run(make_run())
    artifact = add_artifact(first, run)
    amended, entry = first.amend_artifact(
        artifact.id,
        sha256="c" * 64,
        byte_size=10,
        reason="pre-freeze correction",
    )
    first.close()

    with SQLiteStorageRepository(database) as reopened:
        assert reopened.get_run(run.id) == run
        assert reopened.get_artifact(artifact.id) == amended
        assert reopened.list_amendments(artifact.id) == (entry,)
