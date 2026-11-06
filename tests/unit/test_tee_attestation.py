from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from watermark_lab.security import (
    AttestationBundle,
    AttestationVerificationError,
    SimulatedTeeBackend,
    SimulatedTeeVerifier,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (ROOT / "schemas" / "tee-verification-statement.schema.json").read_text(encoding="utf-8")
)
SCHEMA_VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
KEY = b"development-simulator-key-only!!"
MEASUREMENT = "a" * 64


def _components(*, allow_simulated: bool = True) -> tuple[SimulatedTeeBackend, SimulatedTeeVerifier]:
    backend = SimulatedTeeBackend(KEY, enclave_measurement_sha256=MEASUREMENT)
    verifier = SimulatedTeeVerifier(
        KEY,
        expected_measurement_sha256=MEASUREMENT,
        allow_simulated=allow_simulated,
        now=lambda: NOW,
    )
    return backend, verifier


def _bundle(
    backend: SimulatedTeeBackend, verifier: SimulatedTeeVerifier
) -> AttestationBundle:
    return backend.attest(
        nonce=verifier.issue_nonce(),
        normalized_text_sha256="1" * 64,
        detector_id="kgw",
        detector_code_sha256="2" * 64,
        detector_config_sha256="3" * 64,
        calibration_artifact_sha256="4" * 64,
        key_epoch_reference="epoch_test_0001",
        decision="inconclusive",
        calibrated_fpr_upper_bound=None,
        issued_at=NOW,
    )


def test_simulator_statement_validates_and_verifies_but_cannot_promote() -> None:
    backend, verifier = _components()
    bundle = _bundle(backend, verifier)

    assert list(SCHEMA_VALIDATOR.iter_errors(bundle.statement)) == []
    outcome = verifier.verify(bundle)
    assert outcome.valid is True
    assert outcome.backend == "tee-simulator"
    assert outcome.promotion_eligible is False
    assert any("not hardware attestation" in warning for warning in outcome.warnings)
    assert bundle.statement["simulated"] is True
    assert bundle.statement["verification_status"] == "simulated-only"


def test_simulated_evidence_is_rejected_by_default() -> None:
    backend, verifier = _components(allow_simulated=False)

    with pytest.raises(AttestationVerificationError, match="disabled"):
        verifier.verify(_bundle(backend, verifier))


def test_nonce_is_one_time_and_replay_is_rejected() -> None:
    backend, verifier = _components()
    bundle = _bundle(backend, verifier)
    verifier.verify(bundle)

    with pytest.raises(AttestationVerificationError, match="already been consumed"):
        verifier.verify(bundle)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("decision", "not-detected"),
        ("detector_code_sha256", "9" * 64),
        ("calibration_artifact_sha256", "8" * 64),
        ("key_epoch_reference", "epoch_tampered_0001"),
    ],
)
def test_bound_statement_tampering_is_rejected(field: str, replacement: object) -> None:
    backend, verifier = _components()
    bundle = _bundle(backend, verifier)
    statement = deepcopy(dict(bundle.statement))
    statement[field] = replacement

    with pytest.raises(AttestationVerificationError, match="signature is invalid"):
        verifier.verify(AttestationBundle(statement=statement, evidence=bundle.evidence))


def test_evidence_tampering_and_expired_statements_are_rejected() -> None:
    backend, verifier = _components()
    bundle = _bundle(backend, verifier)
    tampered = bundle.evidence[:-1] + bytes([bundle.evidence[-1] ^ 1])
    with pytest.raises(AttestationVerificationError, match="descriptor does not match"):
        verifier.verify(AttestationBundle(statement=bundle.statement, evidence=tampered))

    expired_verifier = SimulatedTeeVerifier(
        KEY,
        expected_measurement_sha256=MEASUREMENT,
        allow_simulated=True,
        now=lambda: NOW + timedelta(minutes=3),
    )
    expired_bundle = _bundle(backend, expired_verifier)
    with pytest.raises(AttestationVerificationError, match="not currently valid"):
        expired_verifier.verify(expired_bundle)
