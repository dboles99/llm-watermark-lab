from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from watermark_lab.security import (
    AttestationBundle,
    AttestationVerificationError,
    PlatformAttestationClaims,
    RemoteAttestationTrustPolicy,
    RemoteTeeVerifier,
    SimulatedTeeBackend,
    statement_binding_sha256,
)

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
FORMAT = "example-provider-signed-report-v1"
PROVIDER = "example-confidential-compute"
PLATFORM = "example-sev-snp-container"
MEASUREMENT = "a" * 64
EXECUTION_POLICY = "b" * 64
CODE = "c" * 64
CONFIG = "d" * 64
CALIBRATION = "e" * 64
EPOCH = "epoch_remote_0001"
EVIDENCE = b"opaque-provider-signed-attestation-report"
ROOT = Path(__file__).resolve().parents[2]
STATEMENT_SCHEMA = json.loads(
    (ROOT / "schemas" / "tee-verification-statement.schema.json").read_text(encoding="utf-8")
)
STATEMENT_VALIDATOR = Draft202012Validator(
    STATEMENT_SCHEMA,
    format_checker=FormatChecker(),
)


class StubPlatformEvidenceVerifier:
    """Test double for an adapter that has already authenticated native evidence."""

    def __init__(self) -> None:
        self.claims: PlatformAttestationClaims | None = None
        self.calls = 0

    def verify_evidence(
        self,
        evidence: bytes,
        *,
        expected_nonce: str,
        verification_time: datetime,
    ) -> PlatformAttestationClaims:
        assert evidence == EVIDENCE
        assert verification_time == NOW
        self.calls += 1
        if self.claims is None:
            raise AssertionError("Test must configure authenticated platform claims.")
        return self.claims


def _policy() -> RemoteAttestationTrustPolicy:
    return RemoteAttestationTrustPolicy(
        allowed_providers=frozenset({PROVIDER}),
        expected_platform=PLATFORM,
        allowed_evidence_formats=frozenset({FORMAT}),
        expected_measurement_sha256=MEASUREMENT,
        expected_execution_policy_sha256=EXECUTION_POLICY,
        expected_detector_id="kgw-v1",
        expected_detector_code_sha256=CODE,
        expected_detector_config_sha256=CONFIG,
        expected_calibration_artifact_sha256=CALIBRATION,
        expected_key_epoch_reference=EPOCH,
    )


def _verifier() -> tuple[RemoteTeeVerifier, StubPlatformEvidenceVerifier]:
    adapter = StubPlatformEvidenceVerifier()
    verifier = RemoteTeeVerifier(_policy(), {FORMAT: adapter}, now=lambda: NOW)
    return verifier, adapter


def _bundle(
    verifier: RemoteTeeVerifier,
    adapter: StubPlatformEvidenceVerifier,
    *,
    issued_at: datetime = NOW,
    expires_at: datetime = NOW + timedelta(minutes=2),
) -> AttestationBundle:
    nonce = verifier.issue_nonce()
    statement = {
        "$schema": "../../../schemas/tee-verification-statement.schema.json",
        "schema_version": "1.0.0",
        "statement_id": "stmt_remote_test_0001",
        "backend": "remote-tee",
        "simulated": False,
        "platform": PLATFORM,
        "enclave_measurement_sha256": MEASUREMENT,
        "nonce": nonce,
        "normalized_text_sha256": "1" * 64,
        "detector_id": "kgw-v1",
        "detector_code_sha256": CODE,
        "detector_config_sha256": CONFIG,
        "calibration_artifact_sha256": CALIBRATION,
        "key_epoch_reference": EPOCH,
        "decision": "inconclusive",
        "calibrated_fpr_upper_bound": None,
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "verification_status": "attested-valid",
    }
    adapter.claims = PlatformAttestationClaims(
        authentic=True,
        provider=PROVIDER,
        platform=PLATFORM,
        evidence_format=FORMAT,
        nonce=nonce,
        enclave_measurement_sha256=MEASUREMENT,
        execution_policy_sha256=EXECUTION_POLICY,
        statement_binding_sha256=statement_binding_sha256(statement),
        issued_at=issued_at,
        expires_at=expires_at,
    )
    statement["attestation_evidence"] = {
        "artifact_id": "attest_remote_test_0001",
        "format": FORMAT,
        "sha256": hashlib.sha256(EVIDENCE).hexdigest(),
        "size_bytes": len(EVIDENCE),
    }
    return AttestationBundle(statement=statement, evidence=EVIDENCE)


def test_remote_verification_authenticates_platform_but_not_science() -> None:
    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)

    assert list(STATEMENT_VALIDATOR.iter_errors(bundle.statement)) == []
    outcome = verifier.verify(bundle)

    assert outcome.valid is True
    assert outcome.backend == "remote-tee"
    assert outcome.platform_attestation_authentic is True
    assert outcome.scientific_validity == "not-assessed"
    assert outcome.promotion_eligible is False
    assert "does not establish scientific validity" in outcome.warnings[0]
    assert adapter.calls == 1


def test_nonce_is_verifier_issued_one_time_and_burned_on_failure() -> None:
    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)
    statement = deepcopy(dict(bundle.statement))
    statement["detector_config_sha256"] = "f" * 64
    invalid = AttestationBundle(statement=statement, evidence=bundle.evidence)

    with pytest.raises(AttestationVerificationError, match="detector_config_sha256"):
        verifier.verify(invalid)
    with pytest.raises(AttestationVerificationError, match="already been consumed"):
        verifier.verify(bundle)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("enclave_measurement_sha256", "f" * 64),
        ("detector_code_sha256", "f" * 64),
        ("detector_config_sha256", "f" * 64),
        ("calibration_artifact_sha256", "f" * 64),
        ("key_epoch_reference", "epoch_wrong_0001"),
    ],
)
def test_statement_commitments_must_match_policy(field: str, replacement: str) -> None:
    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)
    statement = deepcopy(dict(bundle.statement))
    statement[field] = replacement

    with pytest.raises(AttestationVerificationError, match=f"{field} does not match"):
        verifier.verify(AttestationBundle(statement=statement, evidence=bundle.evidence))


def test_execution_policy_and_statement_binding_are_checked_in_platform_claims() -> None:
    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)
    assert adapter.claims is not None
    adapter.claims = replace(adapter.claims, execution_policy_sha256="f" * 64)

    with pytest.raises(AttestationVerificationError, match="Execution policy"):
        verifier.verify(bundle)

    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)
    assert adapter.claims is not None
    adapter.claims = replace(adapter.claims, statement_binding_sha256="f" * 64)
    with pytest.raises(AttestationVerificationError, match="does not bind"):
        verifier.verify(bundle)


def test_unauthenticated_or_untrusted_platform_claims_fail_closed() -> None:
    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)
    assert adapter.claims is not None
    adapter.claims = replace(adapter.claims, authentic=False)
    with pytest.raises(AttestationVerificationError, match="not authentic"):
        verifier.verify(bundle)

    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)
    assert adapter.claims is not None
    adapter.claims = replace(adapter.claims, provider="untrusted-provider")
    with pytest.raises(AttestationVerificationError, match="provider is not trusted"):
        verifier.verify(bundle)


def test_statement_and_platform_evidence_must_be_fresh() -> None:
    verifier, adapter = _verifier()
    stale_bundle = _bundle(
        verifier,
        adapter,
        issued_at=NOW - timedelta(minutes=6),
        expires_at=NOW + timedelta(minutes=1),
    )
    with pytest.raises(AttestationVerificationError, match="stale"):
        verifier.verify(stale_bundle)

    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)
    assert adapter.claims is not None
    adapter.claims = replace(
        adapter.claims,
        issued_at=NOW - timedelta(minutes=2),
        expires_at=NOW - timedelta(minutes=1),
    )
    with pytest.raises(AttestationVerificationError, match="Platform evidence has expired"):
        verifier.verify(bundle)


def test_detached_evidence_tampering_is_rejected_before_adapter() -> None:
    verifier, adapter = _verifier()
    bundle = _bundle(verifier, adapter)

    with pytest.raises(AttestationVerificationError, match="descriptor does not match"):
        verifier.verify(AttestationBundle(bundle.statement, EVIDENCE + b"tampered"))
    assert adapter.calls == 0


def test_simulator_cannot_satisfy_remote_verifier() -> None:
    verifier, adapter = _verifier()
    simulator = SimulatedTeeBackend(
        b"development-simulator-key-only!!",
        enclave_measurement_sha256=MEASUREMENT,
    )
    simulated = simulator.attest(
        nonce=verifier.issue_nonce(),
        normalized_text_sha256="1" * 64,
        detector_id="kgw-v1",
        detector_code_sha256=CODE,
        detector_config_sha256=CONFIG,
        calibration_artifact_sha256=CALIBRATION,
        key_epoch_reference=EPOCH,
        decision="inconclusive",
        calibrated_fpr_upper_bound=None,
        issued_at=NOW,
    )

    with pytest.raises(AttestationVerificationError, match="simulator evidence is rejected"):
        verifier.verify(simulated)
    assert adapter.calls == 0


def test_policy_rejects_simulator_and_adapter_configuration_must_match_exactly() -> None:
    with pytest.raises(ValueError, match="Simulator evidence"):
        replace(
            _policy(),
            allowed_evidence_formats=frozenset({"lab-hmac-tee-simulator-v1"}),
        )

    with pytest.raises(ValueError, match="exactly match"):
        RemoteTeeVerifier(_policy(), {}, now=lambda: NOW)
