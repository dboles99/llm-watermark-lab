"""Provider-neutral, fail-closed verification of remote TEE evidence.

This module defines the trust boundary between the watermark application and a
platform-specific attestation adapter.  An adapter authenticates native
platform evidence (for example, a signed attestation token) and returns typed
claims.  :class:`RemoteTeeVerifier` then applies application policy, statement
binding, freshness, and replay checks.

Passing this verifier establishes platform-evidence authenticity under the
configured trust policy.  It does *not* establish that a watermark detector,
calibration method, or scientific conclusion is valid.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from watermark_lab.security.attestation import (
    SIMULATOR_FORMAT,
    AttestationBundle,
    AttestationVerificationError,
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_EPOCH_RE = re.compile(r"^epoch_[a-zA-Z0-9_-]{8,}$")
_ARTIFACT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_ALLOWED_DECISIONS = {
    "verified-compatible-watermark",
    "not-detected",
    "inconclusive",
}
_REQUIRED_STATEMENT_FIELDS = {
    "schema_version",
    "statement_id",
    "backend",
    "simulated",
    "platform",
    "enclave_measurement_sha256",
    "nonce",
    "normalized_text_sha256",
    "detector_id",
    "detector_code_sha256",
    "detector_config_sha256",
    "calibration_artifact_sha256",
    "key_epoch_reference",
    "decision",
    "attestation_evidence",
    "issued_at",
    "expires_at",
    "verification_status",
}
_ALLOWED_STATEMENT_FIELDS = _REQUIRED_STATEMENT_FIELDS | {
    "$schema",
    "calibrated_fpr_upper_bound",
}


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AttestationVerificationError(f"{field} must include a timezone.")
    return value.astimezone(UTC)


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise AttestationVerificationError(f"{field} must be an RFC 3339 timestamp.")
    try:
        return _aware_utc(datetime.fromisoformat(value), field=field)
    except ValueError as exc:
        raise AttestationVerificationError(f"{field} is invalid.") from exc


def _policy_sha256(value: str, *, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest.")
    return value


def statement_binding_sha256(statement: Mapping[str, Any]) -> str:
    """Return the canonical digest a platform report must bind.

    The detached evidence descriptor is excluded to avoid a circular digest:
    evidence authenticates the statement binding, while the statement records
    the digest and size of that evidence.
    """

    bound = {key: value for key, value in statement.items() if key != "attestation_evidence"}
    try:
        payload = json.dumps(
            bound,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttestationVerificationError(
            "Attestation statement cannot be canonically encoded."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PlatformAttestationClaims:
    """Authenticated claims returned by a platform-specific evidence adapter."""

    authentic: bool
    provider: str
    platform: str
    evidence_format: str
    nonce: str
    enclave_measurement_sha256: str
    execution_policy_sha256: str
    statement_binding_sha256: str
    issued_at: datetime
    expires_at: datetime


class PlatformEvidenceVerifier(Protocol):
    """Adapter interface for cryptographic verification of native TEE evidence.

    Implementations must validate the provider's signature/certificate chain,
    trust roots, revocation status when available, and report semantics before
    returning ``authentic=True``.  Network access, caching, and provider SDKs
    belong in the adapter, not in the provider-neutral policy verifier.
    """

    def verify_evidence(
        self,
        evidence: bytes,
        *,
        expected_nonce: str,
        verification_time: datetime,
    ) -> PlatformAttestationClaims:
        """Authenticate evidence and return normalized claims or raise."""
        ...


@dataclass(frozen=True, slots=True)
class RemoteAttestationTrustPolicy:
    """Closed allow-list of all commitments required for remote verification."""

    allowed_providers: frozenset[str]
    expected_platform: str
    allowed_evidence_formats: frozenset[str]
    expected_measurement_sha256: str
    expected_execution_policy_sha256: str
    expected_detector_id: str
    expected_detector_code_sha256: str
    expected_detector_config_sha256: str
    expected_calibration_artifact_sha256: str
    expected_key_epoch_reference: str
    maximum_statement_age: timedelta = timedelta(minutes=5)
    maximum_statement_lifetime: timedelta = timedelta(minutes=10)
    maximum_evidence_lifetime: timedelta = timedelta(minutes=10)
    clock_skew: timedelta = timedelta(seconds=30)

    def __post_init__(self) -> None:
        if not self.allowed_providers or any(not item for item in self.allowed_providers):
            raise ValueError("allowed_providers must be a non-empty explicit allow-list.")
        if not self.expected_platform:
            raise ValueError("expected_platform is required.")
        if not self.allowed_evidence_formats or any(
            not item for item in self.allowed_evidence_formats
        ):
            raise ValueError("allowed_evidence_formats must be a non-empty explicit allow-list.")
        if SIMULATOR_FORMAT in self.allowed_evidence_formats:
            raise ValueError("Simulator evidence cannot be trusted by a remote TEE policy.")
        for field in (
            "expected_measurement_sha256",
            "expected_execution_policy_sha256",
            "expected_detector_code_sha256",
            "expected_detector_config_sha256",
            "expected_calibration_artifact_sha256",
        ):
            _policy_sha256(str(getattr(self, field)), field=field)
        if not self.expected_detector_id:
            raise ValueError("expected_detector_id is required.")
        if not _EPOCH_RE.fullmatch(self.expected_key_epoch_reference):
            raise ValueError("expected_key_epoch_reference must be an opaque epoch_* reference.")
        for field in (
            "maximum_statement_age",
            "maximum_statement_lifetime",
            "maximum_evidence_lifetime",
        ):
            if getattr(self, field) <= timedelta(0):
                raise ValueError(f"{field} must be greater than zero.")
        if self.clock_skew < timedelta(0):
            raise ValueError("clock_skew cannot be negative.")


@dataclass(frozen=True, slots=True)
class RemoteVerificationOutcome:
    """Typed result that does not conflate attestation with scientific validity."""

    valid: Literal[True]
    backend: Literal["remote-tee"]
    statement_id: str
    provider: str
    platform: str
    evidence_sha256: str
    platform_attestation_authentic: Literal[True]
    scientific_validity: Literal["not-assessed"]
    promotion_eligible: Literal[False]
    warnings: tuple[str, ...]


class RemoteTeeVerifier:
    """Verify remote TEE statements against an explicit, fail-closed policy."""

    def __init__(
        self,
        policy: RemoteAttestationTrustPolicy,
        evidence_verifiers: Mapping[str, PlatformEvidenceVerifier],
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        configured_formats = frozenset(evidence_verifiers)
        if configured_formats != policy.allowed_evidence_formats:
            raise ValueError(
                "Evidence verifiers must exactly match the policy allow-list; "
                "missing and unapproved adapters fail closed."
            )
        self.policy = policy
        self._evidence_verifiers = dict(evidence_verifiers)
        self._now = now
        self._pending_nonces: set[str] = set()
        self._consumed_nonces: set[str] = set()

    def issue_nonce(self) -> str:
        """Issue and remember a verifier-originated, one-time challenge."""

        nonce = secrets.token_urlsafe(32)
        self._pending_nonces.add(nonce)
        return nonce

    def verify(self, bundle: AttestationBundle) -> RemoteVerificationOutcome:
        """Verify one statement and consume its nonce even when verification fails."""

        statement = dict(bundle.statement)
        nonce = statement.get("nonce")
        if not isinstance(nonce, str) or nonce not in self._pending_nonces:
            if isinstance(nonce, str) and nonce in self._consumed_nonces:
                raise AttestationVerificationError("Attestation nonce has already been consumed.")
            raise AttestationVerificationError("Attestation nonce was not issued by this verifier.")
        self._pending_nonces.remove(nonce)
        self._consumed_nonces.add(nonce)

        self._verify_statement_contract(statement)
        self._verify_statement_labels(statement)
        self._verify_statement_commitments(statement)
        now = _aware_utc(self._now(), field="verification_time")
        self._verify_statement_time(statement, now=now)

        descriptor = self._verify_evidence_descriptor(statement, bundle.evidence)
        evidence_format = str(descriptor["format"])
        adapter = self._evidence_verifiers[evidence_format]
        claims = adapter.verify_evidence(
            bundle.evidence,
            expected_nonce=nonce,
            verification_time=now,
        )
        self._verify_platform_claims(
            claims,
            statement=statement,
            evidence_format=evidence_format,
            nonce=nonce,
            now=now,
        )

        statement_id = statement.get("statement_id")
        if not isinstance(statement_id, str) or not _ARTIFACT_ID_RE.fullmatch(statement_id):
            raise AttestationVerificationError("statement_id is missing or invalid.")
        evidence_sha256 = hashlib.sha256(bundle.evidence).hexdigest()
        return RemoteVerificationOutcome(
            valid=True,
            backend="remote-tee",
            statement_id=statement_id,
            provider=claims.provider,
            platform=claims.platform,
            evidence_sha256=evidence_sha256,
            platform_attestation_authentic=True,
            scientific_validity="not-assessed",
            promotion_eligible=False,
            warnings=(
                "Platform attestation authenticity does not establish scientific validity.",
                "Scientific promotion requires separate calibration, robustness, and review gates.",
            ),
        )

    def _verify_statement_contract(self, statement: Mapping[str, Any]) -> None:
        missing = _REQUIRED_STATEMENT_FIELDS - statement.keys()
        unexpected = statement.keys() - _ALLOWED_STATEMENT_FIELDS
        if missing or unexpected:
            raise AttestationVerificationError(
                "Attestation statement fields do not match schema version 1.0.0."
            )
        statement_id = statement.get("statement_id")
        if not isinstance(statement_id, str) or not _ARTIFACT_ID_RE.fullmatch(statement_id):
            raise AttestationVerificationError("statement_id is missing or invalid.")
        text_digest = statement.get("normalized_text_sha256")
        if not isinstance(text_digest, str) or not _SHA256_RE.fullmatch(text_digest):
            raise AttestationVerificationError("normalized_text_sha256 is invalid.")
        if statement.get("decision") not in _ALLOWED_DECISIONS:
            raise AttestationVerificationError("Attested decision is invalid.")
        fpr = statement.get("calibrated_fpr_upper_bound")
        if fpr is not None and (
            isinstance(fpr, bool) or not isinstance(fpr, (int, float)) or not 0 <= fpr <= 1
        ):
            raise AttestationVerificationError("calibrated_fpr_upper_bound is invalid.")

    def _verify_statement_labels(self, statement: Mapping[str, Any]) -> None:
        if statement.get("schema_version") != "1.0.0":
            raise AttestationVerificationError("Unsupported attestation statement schema version.")
        if (
            statement.get("backend") != "remote-tee"
            or statement.get("simulated") is not False
            or statement.get("verification_status") != "attested-valid"
        ):
            raise AttestationVerificationError(
                "Remote TEE labels are missing or inconsistent; simulator evidence is rejected."
            )
        if statement.get("platform") != self.policy.expected_platform:
            raise AttestationVerificationError("Statement platform does not match trust policy.")

    def _verify_statement_commitments(self, statement: Mapping[str, Any]) -> None:
        expected: tuple[tuple[str, str], ...] = (
            ("enclave_measurement_sha256", self.policy.expected_measurement_sha256),
            ("detector_id", self.policy.expected_detector_id),
            ("detector_code_sha256", self.policy.expected_detector_code_sha256),
            ("detector_config_sha256", self.policy.expected_detector_config_sha256),
            (
                "calibration_artifact_sha256",
                self.policy.expected_calibration_artifact_sha256,
            ),
            ("key_epoch_reference", self.policy.expected_key_epoch_reference),
        )
        for field, required in expected:
            actual = statement.get(field)
            if not isinstance(actual, str) or not hmac.compare_digest(actual, required):
                raise AttestationVerificationError(f"{field} does not match trust policy.")

    def _verify_statement_time(self, statement: Mapping[str, Any], *, now: datetime) -> None:
        issued = _parse_timestamp(statement.get("issued_at"), field="issued_at")
        expires = _parse_timestamp(statement.get("expires_at"), field="expires_at")
        if expires <= issued or expires - issued > self.policy.maximum_statement_lifetime:
            raise AttestationVerificationError("Attestation statement lifetime violates policy.")
        if issued > now + self.policy.clock_skew:
            raise AttestationVerificationError("Attestation statement was issued in the future.")
        if now - issued > self.policy.maximum_statement_age + self.policy.clock_skew:
            raise AttestationVerificationError("Attestation statement is stale.")
        if now > expires + self.policy.clock_skew:
            raise AttestationVerificationError("Attestation statement has expired.")

    def _verify_evidence_descriptor(
        self, statement: Mapping[str, Any], evidence: bytes
    ) -> Mapping[str, Any]:
        descriptor = statement.get("attestation_evidence")
        if not isinstance(descriptor, dict):
            raise AttestationVerificationError("Attestation evidence descriptor is missing.")
        evidence_format = descriptor.get("format")
        if (
            not isinstance(evidence_format, str)
            or evidence_format not in self.policy.allowed_evidence_formats
            or evidence_format == SIMULATOR_FORMAT
        ):
            raise AttestationVerificationError("Attestation evidence format is not trusted.")
        digest = hashlib.sha256(evidence).hexdigest()
        artifact_id = descriptor.get("artifact_id")
        if (
            descriptor.get("sha256") != digest
            or descriptor.get("size_bytes") != len(evidence)
            or not isinstance(artifact_id, str)
            or not _ARTIFACT_ID_RE.fullmatch(artifact_id)
        ):
            raise AttestationVerificationError("Attestation evidence descriptor does not match.")
        return descriptor

    def _verify_platform_claims(
        self,
        claims: PlatformAttestationClaims,
        *,
        statement: Mapping[str, Any],
        evidence_format: str,
        nonce: str,
        now: datetime,
    ) -> None:
        if claims.authentic is not True:
            raise AttestationVerificationError("Platform evidence is not authentic.")
        if not isinstance(claims.provider, str) or claims.provider not in self.policy.allowed_providers:
            raise AttestationVerificationError("Attestation provider is not trusted.")
        if not isinstance(claims.evidence_format, str) or claims.evidence_format != evidence_format:
            raise AttestationVerificationError("Platform evidence format claim does not match.")
        if not isinstance(claims.nonce, str) or claims.nonce != nonce:
            raise AttestationVerificationError("Platform evidence nonce does not match.")
        if not isinstance(claims.platform, str) or claims.platform != self.policy.expected_platform:
            raise AttestationVerificationError("Platform evidence target does not match policy.")
        if not isinstance(claims.enclave_measurement_sha256, str) or not hmac.compare_digest(
            claims.enclave_measurement_sha256,
            self.policy.expected_measurement_sha256,
        ):
            raise AttestationVerificationError("Platform measurement does not match policy.")
        if not isinstance(claims.execution_policy_sha256, str) or not hmac.compare_digest(
            claims.execution_policy_sha256,
            self.policy.expected_execution_policy_sha256,
        ):
            raise AttestationVerificationError("Execution policy does not match trust policy.")
        expected_binding = statement_binding_sha256(statement)
        if not isinstance(claims.statement_binding_sha256, str) or not hmac.compare_digest(
            claims.statement_binding_sha256, expected_binding
        ):
            raise AttestationVerificationError("Platform evidence does not bind the statement.")

        if not isinstance(claims.issued_at, datetime) or not isinstance(
            claims.expires_at, datetime
        ):
            raise AttestationVerificationError("Platform evidence timestamps are invalid.")
        issued = _aware_utc(claims.issued_at, field="platform_evidence.issued_at")
        expires = _aware_utc(claims.expires_at, field="platform_evidence.expires_at")
        if expires <= issued or expires - issued > self.policy.maximum_evidence_lifetime:
            raise AttestationVerificationError("Platform evidence lifetime violates policy.")
        if issued > now + self.policy.clock_skew:
            raise AttestationVerificationError("Platform evidence was issued in the future.")
        if now > expires + self.policy.clock_skew:
            raise AttestationVerificationError("Platform evidence has expired.")
