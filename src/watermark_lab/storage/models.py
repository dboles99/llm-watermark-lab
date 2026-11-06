"""Typed persistence records for runs, provenance, and evidence artifacts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
COMMIT_PATTERN = re.compile(r"^[a-fA-F0-9]{7,64}$")
CANONICAL_EVIDENCE_ROOT = PurePosixPath("project/artifacts")

_SENSITIVE_KEY_NAMES = {
    "accesstoken",
    "apikey",
    "bearertoken",
    "clientsecret",
    "password",
    "passwd",
    "privatekey",
    "rawkey",
    "secret",
    "secretkey",
    "token",
}
_SENSITIVE_COMMAND = re.compile(
    r"(--?(api[-_]?key|password|passwd|secret|token|private[-_]?key|raw[-_]?key)(?:=|\s+))"
    r"|((api[-_]?key|password|passwd|secret|token|private[-_]?key|raw[-_]?key)=)",
    re.IGNORECASE,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _reject_sensitive_mapping(value: Any, location: str = "value") -> Any:
    """Reject mappings that could persist credentials or raw key material."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized_key in _SENSITIVE_KEY_NAMES:
                raise ValueError(f"{location} contains forbidden sensitive field {key!r}")
            _reject_sensitive_mapping(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_mapping(item, f"{location}[{index}]")
    return value


def canonical_artifact_path(run_id: UUID, relative_name: str | PurePosixPath) -> str:
    """Build a safe canonical evidence path below ``project/artifacts/<run-id>``."""
    raw_name = str(relative_name)
    if "\\" in raw_name:
        raise ValueError("artifact paths must use forward slashes")
    name = PurePosixPath(raw_name)
    if name.is_absolute() or not name.parts or any(part in {"", ".", ".."} for part in name.parts):
        raise ValueError("artifact path must be a non-empty relative path without traversal")
    return str(CANONICAL_EVIDENCE_ROOT / str(run_id) / name)


def validate_canonical_artifact_path(path: str, run_id: UUID) -> str:
    """Validate that a stored path belongs to the run's canonical evidence directory."""
    if "\\" in path:
        raise ValueError("artifact paths must use forward slashes")
    candidate = PurePosixPath(path)
    required = CANONICAL_EVIDENCE_ROOT / str(run_id)
    if candidate.is_absolute() or candidate == required:
        raise ValueError("artifact path must identify a file below the canonical evidence root")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("artifact path must not contain traversal segments")
    if candidate.parts[: len(required.parts)] != required.parts:
        raise ValueError(f"artifact path must be below {required}")
    return str(candidate)


class RunStatus(StrEnum):
    """Lifecycle states persisted for a run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunType(StrEnum):
    """Supported run categories."""

    DETECT = "detect"
    CREATE = "create"
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    COMPARE = "compare"


class ModelProvenance(BaseModel):
    """Pinned model identity without credentials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class RosettaProvenance(BaseModel):
    """Identity and content hash of the Rosetta used for a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identifier: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class RunProvenance(BaseModel):
    """Reproducibility information attached immutably to a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_commit: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    model: ModelProvenance | None = None
    rosetta: RosettaProvenance | None = None
    commands: tuple[str, ...] = ()

    @field_validator("source_commit")
    @classmethod
    def validate_commit(cls, value: str | None) -> str | None:
        if value is not None and not COMMIT_PATTERN.fullmatch(value):
            raise ValueError("source_commit must be a 7-64 character hexadecimal revision")
        return value.lower() if value is not None else None

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        _reject_sensitive_mapping(value, "environment")
        return value

    @field_validator("commands")
    @classmethod
    def validate_commands(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for command in value:
            if not command.strip():
                raise ValueError("commands must not contain empty entries")
            if _SENSITIVE_COMMAND.search(command):
                raise ValueError("commands must not contain inline credentials or raw keys")
        return value


class RunRecord(BaseModel):
    """Persistent, typed run metadata with immutable lineage and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    parent_run_id: UUID | None = None
    run_type: RunType
    method_id: str = Field(min_length=1)
    status: RunStatus = RunStatus.PENDING
    config: dict[str, Any] = Field(default_factory=dict)
    provenance: RunProvenance = Field(default_factory=RunProvenance)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @field_validator("config")
    @classmethod
    def validate_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_sensitive_mapping(value, "config")
        return value

    @field_validator("created_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> RunRecord:
        terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
        if self.status in terminal and self.completed_at is None:
            raise ValueError("terminal runs require completed_at")
        if self.status not in terminal and self.completed_at is not None:
            raise ValueError("non-terminal runs cannot have completed_at")
        if self.parent_run_id == self.id:
            raise ValueError("a run cannot be its own parent")
        return self


class ArtifactRecord(BaseModel):
    """Current state of an evidence artifact in an immutable lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    lineage_id: UUID = Field(default_factory=uuid4)
    supersedes_artifact_id: UUID | None = None
    version: int = Field(default=1, ge=1)
    artifact_type: str = Field(min_length=1)
    path: str
    sha256: str
    byte_size: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    change_reason: str | None = None
    frozen: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_sensitive_mapping(value, "metadata")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_path_and_lineage(self) -> ArtifactRecord:
        validate_canonical_artifact_path(self.path, self.run_id)
        if self.version == 1 and self.supersedes_artifact_id is not None:
            raise ValueError("version 1 artifacts cannot supersede another artifact")
        if self.version > 1 and self.supersedes_artifact_id is None:
            raise ValueError("superseding artifacts must identify their predecessor")
        if self.supersedes_artifact_id == self.id:
            raise ValueError("an artifact cannot supersede itself")
        return self


class AmendmentEntry(BaseModel):
    """Append-only audit entry for a permitted in-place artifact amendment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    artifact_id: UUID
    sequence: int = Field(ge=1)
    before_sha256: str
    after_sha256: str
    before_byte_size: int = Field(ge=0)
    after_byte_size: int = Field(ge=0)
    reason: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("before_sha256", "after_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def validate_change(self) -> AmendmentEntry:
        if self.before_sha256 == self.after_sha256:
            raise ValueError("an amendment must change the artifact hash")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return self
