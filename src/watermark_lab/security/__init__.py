"""Security boundaries for proof-carrying watermark verification."""

from watermark_lab.security.attestation import (
    AttestationBundle,
    AttestationVerificationError,
    SimulatedTeeBackend,
    SimulatedTeeVerifier,
    VerificationOutcome,
)
from watermark_lab.security.remote_attestation import (
    PlatformAttestationClaims,
    PlatformEvidenceVerifier,
    RemoteAttestationTrustPolicy,
    RemoteTeeVerifier,
    RemoteVerificationOutcome,
    statement_binding_sha256,
)

__all__ = [
    "AttestationBundle",
    "AttestationVerificationError",
    "PlatformAttestationClaims",
    "PlatformEvidenceVerifier",
    "RemoteAttestationTrustPolicy",
    "RemoteTeeVerifier",
    "RemoteVerificationOutcome",
    "SimulatedTeeBackend",
    "SimulatedTeeVerifier",
    "VerificationOutcome",
    "statement_binding_sha256",
]
