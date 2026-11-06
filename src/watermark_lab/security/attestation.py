"""Fail-closed development simulator for the future remote TEE contract.

The HMAC envelope in this module is a local integration fixture, not hardware
attestation.  It exists to stabilize statement semantics, freshness, replay,
and tamper tests before a platform-native remote TEE adapter is selected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
EPOCH_RE = re.compile(r"^epoch_[a-zA-Z0-9_-]{8,}$")
ALLOWED_DECISIONS = {
    "verified-compatible-watermark",
    "not-detected",
    "inconclusive",
}
SIMULATOR_FORMAT = "lab-hmac-tee-simulator-v1"


class AttestationVerificationError(ValueError):
    """Raised when simulated evidence fails closed."""


@dataclass(frozen=True, slots=True)
class AttestationBundle:
    """Statement plus detached evidence bytes.

    Evidence bytes must be stored as a separate protected artifact. They are
    intentionally not included by ``as_dict`` to keep JSON reports bounded.
    """

    statement: Mapping[str, Any]
    evidence: bytes

    def as_dict(self) -> dict[str, Any]:
        return dict(self.statement)


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    valid: bool
    backend: str
    statement_id: str
    evidence_sha256: str
    promotion_eligible: bool
    warnings: tuple[str, ...]


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Attestation timestamps must be timezone-aware.")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise AttestationVerificationError(f"{field} must be an RFC 3339 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AttestationVerificationError(f"{field} is invalid.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AttestationVerificationError(f"{field} must include a timezone.")
    return parsed.astimezone(UTC)


def _require_sha256(value: str, *, field: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest.")
    return value


def _binding_payload(statement: Mapping[str, Any]) -> bytes:
    """Canonical bytes bound by evidence, excluding the evidence descriptor itself."""

    bound = {key: value for key, value in statement.items() if key != "attestation_evidence"}
    return json.dumps(
        bound,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class SimulatedTeeBackend:
    """Create explicitly simulated HMAC evidence for local contract tests."""

    def __init__(self, signing_key: bytes, *, enclave_measurement_sha256: str) -> None:
        if len(signing_key) < 32:
            raise ValueError("The simulator signing key must contain at least 32 bytes.")
        self._signing_key = bytes(signing_key)
        self.enclave_measurement_sha256 = _require_sha256(
            enclave_measurement_sha256, field="enclave_measurement_sha256"
        )

    def attest(
        self,
        *,
        nonce: str,
        normalized_text_sha256: str,
        detector_id: str,
        detector_code_sha256: str,
        detector_config_sha256: str,
        calibration_artifact_sha256: str,
        key_epoch_reference: str,
        decision: str,
        calibrated_fpr_upper_bound: float | None,
        issued_at: datetime | None = None,
        validity: timedelta = timedelta(minutes=2),
    ) -> AttestationBundle:
        if len(nonce) < 16 or not re.fullmatch(r"[a-zA-Z0-9_-]+", nonce):
            raise ValueError("nonce must contain at least 16 URL-safe characters.")
        if not detector_id:
            raise ValueError("detector_id is required.")
        if not EPOCH_RE.fullmatch(key_epoch_reference):
            raise ValueError("key_epoch_reference must be an opaque epoch_* reference.")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Unsupported attested decision: {decision!r}.")
        if calibrated_fpr_upper_bound is not None and not 0 <= calibrated_fpr_upper_bound <= 1:
            raise ValueError("calibrated_fpr_upper_bound must be between 0 and 1.")
        if validity <= timedelta(0) or validity > timedelta(minutes=10):
            raise ValueError("Simulator validity must be greater than zero and at most 10 minutes.")

        issued = (issued_at or _utc_now()).astimezone(UTC)
        statement: dict[str, Any] = {
            "$schema": "../../../schemas/tee-verification-statement.schema.json",
            "schema_version": "1.0.0",
            "statement_id": f"stmt_{uuid.uuid4().hex}",
            "backend": "tee-simulator",
            "simulated": True,
            "platform": "local-development-simulator",
            "enclave_measurement_sha256": self.enclave_measurement_sha256,
            "nonce": nonce,
            "normalized_text_sha256": _require_sha256(
                normalized_text_sha256, field="normalized_text_sha256"
            ),
            "detector_id": detector_id,
            "detector_code_sha256": _require_sha256(
                detector_code_sha256, field="detector_code_sha256"
            ),
            "detector_config_sha256": _require_sha256(
                detector_config_sha256, field="detector_config_sha256"
            ),
            "calibration_artifact_sha256": _require_sha256(
                calibration_artifact_sha256, field="calibration_artifact_sha256"
            ),
            "key_epoch_reference": key_epoch_reference,
            "decision": decision,
            "calibrated_fpr_upper_bound": calibrated_fpr_upper_bound,
            "issued_at": _timestamp(issued),
            "expires_at": _timestamp(issued + validity),
            "verification_status": "simulated-only",
        }
        evidence = hmac.digest(self._signing_key, _binding_payload(statement), "sha256")
        evidence_sha256 = hashlib.sha256(evidence).hexdigest()
        statement["attestation_evidence"] = {
            "artifact_id": f"attest_{evidence_sha256[:24]}",
            "format": SIMULATOR_FORMAT,
            "sha256": evidence_sha256,
            "size_bytes": len(evidence),
        }
        return AttestationBundle(statement=statement, evidence=evidence)


class SimulatedTeeVerifier:
    """Verify freshness, one-time nonce use, measurement policy, and HMAC evidence."""

    def __init__(
        self,
        signing_key: bytes,
        *,
        expected_measurement_sha256: str,
        allow_simulated: bool = False,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        if len(signing_key) < 32:
            raise ValueError("The simulator verification key must contain at least 32 bytes.")
        self._signing_key = bytes(signing_key)
        self.expected_measurement_sha256 = _require_sha256(
            expected_measurement_sha256, field="expected_measurement_sha256"
        )
        self.allow_simulated = allow_simulated
        self._now = now
        self._pending_nonces: set[str] = set()
        self._consumed_nonces: set[str] = set()

    def issue_nonce(self) -> str:
        nonce = secrets.token_urlsafe(24)
        self._pending_nonces.add(nonce)
        return nonce

    def verify(self, bundle: AttestationBundle) -> VerificationOutcome:
        statement = dict(bundle.statement)
        nonce = statement.get("nonce")
        if not isinstance(nonce, str) or nonce not in self._pending_nonces:
            if nonce in self._consumed_nonces:
                raise AttestationVerificationError("Attestation nonce has already been consumed.")
            raise AttestationVerificationError("Attestation nonce was not issued by this verifier.")
        self._pending_nonces.remove(nonce)
        self._consumed_nonces.add(nonce)

        if not self.allow_simulated:
            raise AttestationVerificationError(
                "Simulated TEE evidence is disabled and cannot be used as production evidence."
            )
        if (
            statement.get("backend") != "tee-simulator"
            or statement.get("simulated") is not True
            or statement.get("verification_status") != "simulated-only"
        ):
            raise AttestationVerificationError("Simulator labels are missing or inconsistent.")
        if statement.get("platform") != "local-development-simulator":
            raise AttestationVerificationError("Unexpected simulator platform identifier.")
        if statement.get("enclave_measurement_sha256") != self.expected_measurement_sha256:
            raise AttestationVerificationError("Simulator measurement does not match policy.")

        issued = _parse_timestamp(statement.get("issued_at"), field="issued_at")
        expires = _parse_timestamp(statement.get("expires_at"), field="expires_at")
        now = self._now().astimezone(UTC)
        if expires <= issued or now < issued or now > expires:
            raise AttestationVerificationError("Attestation statement is not currently valid.")

        descriptor = statement.get("attestation_evidence")
        if not isinstance(descriptor, dict):
            raise AttestationVerificationError("Attestation evidence descriptor is missing.")
        evidence_sha256 = hashlib.sha256(bundle.evidence).hexdigest()
        if (
            descriptor.get("format") != SIMULATOR_FORMAT
            or descriptor.get("sha256") != evidence_sha256
            or descriptor.get("size_bytes") != len(bundle.evidence)
            or descriptor.get("artifact_id") != f"attest_{evidence_sha256[:24]}"
        ):
            raise AttestationVerificationError("Attestation evidence descriptor does not match.")

        expected = hmac.digest(self._signing_key, _binding_payload(statement), "sha256")
        if not hmac.compare_digest(expected, bundle.evidence):
            raise AttestationVerificationError("Attestation evidence signature is invalid.")

        statement_id = statement.get("statement_id")
        if not isinstance(statement_id, str):
            raise AttestationVerificationError("statement_id is missing.")
        return VerificationOutcome(
            valid=True,
            backend="tee-simulator",
            statement_id=statement_id,
            evidence_sha256=evidence_sha256,
            promotion_eligible=False,
            warnings=(
                "Development simulator only; this is not hardware attestation.",
                "Simulated evidence cannot support scientific promotion or deployment.",
            ),
        )
