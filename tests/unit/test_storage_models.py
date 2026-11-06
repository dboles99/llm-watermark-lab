"""Tests for typed storage records, provenance safety, and canonical paths."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from watermark_lab.storage import (
    ArtifactRecord,
    ModelProvenance,
    RosettaProvenance,
    RunProvenance,
    RunRecord,
    RunStatus,
    RunType,
    canonical_artifact_path,
    sha256_bytes,
)


def test_canonical_artifact_path_is_scoped_to_run() -> None:
    run_id = uuid4()
    path = canonical_artifact_path(run_id, "reports/result.json")
    assert path == f"project/artifacts/{run_id}/reports/result.json"


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", r"bad\windows.json", "."])
def test_canonical_artifact_path_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="artifact path"):
        canonical_artifact_path(uuid4(), name)


def test_run_captures_reproducibility_provenance() -> None:
    run = RunRecord(
        run_type=RunType.DETECT,
        method_id="kgw",
        config={"gamma": 0.25, "key_id": "reference-only"},
        provenance=RunProvenance(
            source_commit="abcdef1234567",
            environment={"python": "3.12.5", "platform": "windows"},
            model=ModelProvenance(provider="mock", profile_id="local", version="1"),
            rosetta=RosettaProvenance(identifier="kgw-v1", sha256="a" * 64),
            commands=("pytest tests/unit",),
        ),
    )
    assert run.provenance.source_commit == "abcdef1234567"
    assert run.provenance.rosetta is not None
    assert run.provenance.rosetta.sha256 == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_key", "raw"),
        ("apiKey", "raw"),
        ("client_secret", "bad"),
        ("password", "bad"),
        ("private-key", "material"),
    ],
)
def test_run_config_rejects_raw_secrets(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match="forbidden sensitive field"):
        RunRecord(run_type=RunType.CREATE, method_id="kgw", config={field: value})


def test_provenance_commands_reject_inline_secrets() -> None:
    with pytest.raises(ValidationError, match="inline credentials"):
        RunProvenance(commands=("provider --api-key very-secret",))


def test_terminal_run_requires_completion_timestamp() -> None:
    with pytest.raises(ValidationError, match="terminal runs require completed_at"):
        RunRecord(run_type=RunType.ANALYZE, method_id="unicode", status=RunStatus.COMPLETED)


def test_artifact_requires_canonical_run_path_and_hash() -> None:
    run_id = uuid4()
    with pytest.raises(ValidationError, match="must be below"):
        ArtifactRecord(
            run_id=run_id,
            artifact_type="report",
            path="artifacts/report.json",
            sha256="a" * 64,
            byte_size=1,
        )
    with pytest.raises(ValidationError, match="64 lowercase"):
        ArtifactRecord(
            run_id=run_id,
            artifact_type="report",
            path=canonical_artifact_path(run_id, "report.json"),
            sha256="not-a-hash",
            byte_size=1,
        )


def test_hash_helper_has_known_digest() -> None:
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_naive_timestamps_are_rejected() -> None:
    naive_timestamp = datetime(2026, 8, 14, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValidationError, match="timezone-aware"):
        RunRecord(
            run_type=RunType.COMPARE,
            method_id="kgw",
            created_at=naive_timestamp,
        )
    valid = RunRecord(
        run_type=RunType.COMPARE,
        method_id="kgw",
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
    )
    assert valid.created_at.tzinfo is UTC
