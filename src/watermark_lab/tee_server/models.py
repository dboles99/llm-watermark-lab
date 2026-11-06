"""Hash-only HTTP contract for the local TEE simulator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SHA256_PATTERN = r"^[a-f0-9]{64}$"
EPOCH_PATTERN = r"^epoch_[a-zA-Z0-9_-]{8,}$"
NONCE_PATTERN = r"^[a-zA-Z0-9_-]{16,}$"
OPAQUE_ID_PATTERN = r"^[a-zA-Z0-9_.:-]+$"
SIMULATION_NOTICE = (
    "LOCAL NON-HARDWARE SIMULATOR ONLY. This receipt is never promotion-eligible and "
    "is not remote or hardware attestation."
)
type DetectionDecision = Literal[
    "verified compatible watermark", "not detected", "inconclusive"
]
type DetectionCompatibility = Literal["experimental", "prototype"]
type ScoringVariant = Literal[
    "naive-binomial-uncalibrated", "unique-ngram-uncalibrated"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SimulatorLabels(StrictModel):
    attestation_class: Literal["simulated-non-hardware"] = "simulated-non-hardware"
    simulated: Literal[True] = True
    hardware_attestation: Literal[False] = False
    promotion_eligible: Literal[False] = False
    notice: str = SIMULATION_NOTICE


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    service: Literal["lab-local-tee-test-server"] = "lab-local-tee-test-server"
    labels: SimulatorLabels = Field(default_factory=SimulatorLabels)
    key_lifecycle: Literal["ephemeral-process", "mounted-secret-file"]


class CapabilitiesResponse(StrictModel):
    service: Literal["lab-local-tee-test-server"] = "lab-local-tee-test-server"
    labels: SimulatorLabels = Field(default_factory=SimulatorLabels)
    hash_only_endpoint_accepts_raw_text: Literal[False] = False
    kgw_test_endpoint_accepts_transient_raw_text: Literal[True] = True
    raw_text_returned_or_persisted: Literal[False] = False
    accepts_key_material: Literal[False] = False
    accepted_input: tuple[str, ...] = (
        "normalized-text-sha256",
        "detector-and-calibration-sha256-commitments",
        "opaque-key-epoch-reference",
        "caller-supplied-simulated-decision",
    )
    freshness: Literal["single-use nonce and expiring simulated statement"] = (
        "single-use nonce and expiring simulated statement"
    )
    persistence: Literal["none; nonce and ephemeral-key state are process-local"] = (
        "none; nonce and ephemeral-key state are process-local"
    )


class NonceResponse(StrictModel):
    nonce: str = Field(pattern=NONCE_PATTERN)
    labels: SimulatorLabels = Field(default_factory=SimulatorLabels)


class SimulatedVerificationRequest(StrictModel):
    """Commitments only; raw text and key material are intentionally absent."""

    nonce: str = Field(pattern=NONCE_PATTERN, max_length=256)
    normalized_text_sha256: str = Field(pattern=SHA256_PATTERN)
    detector_id: str = Field(min_length=1, max_length=128, pattern=OPAQUE_ID_PATTERN)
    detector_code_sha256: str = Field(pattern=SHA256_PATTERN)
    detector_config_sha256: str = Field(pattern=SHA256_PATTERN)
    calibration_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    key_epoch_reference: str = Field(pattern=EPOCH_PATTERN, max_length=256)
    decision: Literal["verified-compatible-watermark", "not-detected", "inconclusive"]
    calibrated_fpr_upper_bound: float | None = Field(default=None, ge=0, le=1)
    validity_seconds: int = Field(default=120, ge=1, le=600)


class VerificationSummary(StrictModel):
    valid: Literal[True] = True
    backend: Literal["tee-simulator"] = "tee-simulator"
    statement_id: str
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    promotion_eligible: Literal[False] = False
    warnings: tuple[str, ...]


class DetachedEvidence(StrictModel):
    artifact_id: str
    format: Literal["lab-hmac-tee-simulator-v1"] = "lab-hmac-tee-simulator-v1"
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=1)
    encoding: Literal["base64"] = "base64"
    value: str


class SimulatedVerificationResponse(StrictModel):
    labels: SimulatorLabels = Field(default_factory=SimulatorLabels)
    statement: dict[str, Any]
    verification: VerificationSummary
    detached_evidence: DetachedEvidence


class KgwTestConfig(StrictModel):
    """Narrow test configuration; no key or key-epoch inputs are exposed."""

    method_id: Literal["kgw"] = "kgw"
    tokenizer_id: Literal["gpt2"] = "gpt2"
    threshold: float = Field(default=4.0, ge=1.0, le=8.0)
    gamma: float = Field(default=0.25, ge=0.25, le=0.25)
    context_width: Literal[1] = 1
    hash_key: Literal[15485863] = 15485863
    seeding_scheme: Literal["simple_1"] = "simple_1"
    ignore_repeated_ngrams: bool = False


class KgwTestDetectionRequest(StrictModel):
    nonce: str = Field(pattern=NONCE_PATTERN, max_length=256)
    text: str = Field(min_length=1, max_length=200_000)
    config: KgwTestConfig = Field(default_factory=KgwTestConfig)


class DetectionResultResponse(StrictModel):
    method_id: Literal["kgw"] = "kgw"
    compatibility: DetectionCompatibility
    decision: DetectionDecision
    score: float | None
    p_value: float | None
    token_count: int = Field(ge=0)
    estimated_fpr: None = None
    warnings: tuple[str, ...]
    original_scorable_token_count: int | None = Field(default=None, ge=0)
    ignored_repeated_ngrams: int = Field(default=0, ge=0)
    scoring_variant: ScoringVariant
    calibration_id: str | None = None


class KgwTestDetectionResponse(StrictModel):
    labels: SimulatorLabels = Field(default_factory=SimulatorLabels)
    calibration_commitment_kind: Literal["configuration-only-uncalibrated"] = (
        "configuration-only-uncalibrated"
    )
    detection: DetectionResultResponse
    statement: dict[str, Any]
    verification: VerificationSummary
    detached_evidence: DetachedEvidence
