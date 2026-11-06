"""Typed persistence and artifact-history primitives."""

from watermark_lab.storage.hashing import sha256_bytes, sha256_file
from watermark_lab.storage.models import (
    CANONICAL_EVIDENCE_ROOT,
    AmendmentEntry,
    ArtifactRecord,
    ModelProvenance,
    RosettaProvenance,
    RunProvenance,
    RunRecord,
    RunStatus,
    RunType,
    canonical_artifact_path,
    validate_canonical_artifact_path,
)
from watermark_lab.storage.repository import (
    FrozenArtifactError,
    InMemoryStorageRepository,
    RecordConflictError,
    RecordNotFoundError,
    StorageRepository,
)
from watermark_lab.storage.sqlite import SQLiteStorageRepository

__all__ = [
    "CANONICAL_EVIDENCE_ROOT",
    "AmendmentEntry",
    "ArtifactRecord",
    "FrozenArtifactError",
    "InMemoryStorageRepository",
    "ModelProvenance",
    "RecordConflictError",
    "RecordNotFoundError",
    "RosettaProvenance",
    "RunProvenance",
    "RunRecord",
    "RunStatus",
    "RunType",
    "SQLiteStorageRepository",
    "StorageRepository",
    "canonical_artifact_path",
    "sha256_bytes",
    "sha256_file",
    "validate_canonical_artifact_path",
]
