"""Stateful process-local service around the existing TEE simulator contract."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import watermark_lab.adapters.kgw as kgw_module
import watermark_lab.run_engine.detect as detect_module
import watermark_lab.run_engine.intake as intake_module
import watermark_lab.run_engine.tokenizer_meta as tokenizer_meta_module
from watermark_lab.core.contracts import DetectionConfig
from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.run_engine.intake import compute_sha256, normalize_text
from watermark_lab.security import (
    AttestationBundle,
    SimulatedTeeBackend,
    SimulatedTeeVerifier,
    VerificationOutcome,
)
from watermark_lab.tee_server.config import TeeServerConfig, load_simulator_signing_key
from watermark_lab.tee_server.models import (
    DetachedEvidence,
    DetectionCompatibility,
    DetectionDecision,
    DetectionResultResponse,
    KgwTestDetectionRequest,
    KgwTestDetectionResponse,
    ScoringVariant,
    SimulatedVerificationRequest,
    SimulatedVerificationResponse,
    VerificationSummary,
)

TEST_KEY_ID = "test-kgw-v1"
TEST_KEY_EPOCH_REFERENCE = "epoch_local_kgw_test_v1"
CALIBRATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "calibration" / "kgw.toml"
)
EXPERIMENTAL_WARNING = (
    "Local simulator used the experimental minimal tokenizer and fixed test key; "
    "this is not scientific, production, or hardware-attested evidence."
)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _detector_code_sha256() -> str:
    source = "\n---component---\n".join(
        inspect.getsource(module)
        for module in (detect_module, kgw_module, intake_module, tokenizer_meta_module)
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration_config_sha256() -> str:
    try:
        contents = CALIBRATION_CONFIG_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "The local KGW calibration configuration is required by the test server."
        ) from exc
    return hashlib.sha256(contents).hexdigest()


def _validated_detection_decision(detection_decision: str) -> DetectionDecision:
    allowed = {"verified compatible watermark", "not detected", "inconclusive"}
    if detection_decision not in allowed:
        raise ValueError("The detector returned an unsupported decision.")
    return cast(DetectionDecision, detection_decision)


def _validated_compatibility(compatibility: str) -> DetectionCompatibility:
    if compatibility not in {"experimental", "prototype"}:
        raise ValueError("The detector returned an unsupported compatibility classification.")
    return cast(DetectionCompatibility, compatibility)


def _validated_scoring_variant(scoring_variant: str) -> ScoringVariant:
    allowed = {"naive-binomial-uncalibrated", "unique-ngram-uncalibrated"}
    if scoring_variant not in allowed:
        raise ValueError("The detector returned an unsupported scoring variant.")
    return cast(ScoringVariant, scoring_variant)


def _attested_decision(detection_decision: DetectionDecision) -> str:
    mapping = {
        "verified compatible watermark": "verified-compatible-watermark",
        "not detected": "not-detected",
        "inconclusive": "inconclusive",
    }
    return mapping[detection_decision]


class TeeSimulatorService:
    """One-worker in-memory nonce authority and simulated receipt issuer."""

    def __init__(
        self,
        config: TeeServerConfig,
        *,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        signing_key, key_lifecycle = load_simulator_signing_key(config.secret_file)
        self.key_lifecycle = key_lifecycle
        self._now = now
        self._detector_code_sha256 = _detector_code_sha256()
        self._calibration_config_sha256 = _calibration_config_sha256()
        self._detection_service = DetectionService()
        self._backend = SimulatedTeeBackend(
            signing_key,
            enclave_measurement_sha256=config.measurement_sha256,
        )
        self._verifier = SimulatedTeeVerifier(
            signing_key,
            expected_measurement_sha256=config.measurement_sha256,
            allow_simulated=True,
            now=now,
        )

    def issue_nonce(self) -> str:
        return self._verifier.issue_nonce()

    def verify(self, request: SimulatedVerificationRequest) -> SimulatedVerificationResponse:
        bundle, outcome = self._attest_and_verify(
            nonce=request.nonce,
            normalized_text_sha256=request.normalized_text_sha256,
            detector_id=request.detector_id,
            detector_code_sha256=request.detector_code_sha256,
            detector_config_sha256=request.detector_config_sha256,
            calibration_artifact_sha256=request.calibration_artifact_sha256,
            key_epoch_reference=request.key_epoch_reference,
            decision=request.decision,
            calibrated_fpr_upper_bound=request.calibrated_fpr_upper_bound,
            validity=timedelta(seconds=request.validity_seconds),
        )
        detached_evidence = self._detached_evidence(bundle.statement, bundle.evidence)

        return SimulatedVerificationResponse(
            statement=dict(bundle.statement),
            verification=VerificationSummary(
                valid=True,
                backend="tee-simulator",
                statement_id=outcome.statement_id,
                evidence_sha256=outcome.evidence_sha256,
                promotion_eligible=False,
                warnings=outcome.warnings,
            ),
            detached_evidence=detached_evidence,
        )

    def detect_kgw(self, request: KgwTestDetectionRequest) -> KgwTestDetectionResponse:
        """Normalize and detect transient text, then attest only commitments/results."""

        normalized_text = normalize_text(request.text)
        options = {
            "gamma": request.config.gamma,
            "context_width": request.config.context_width,
            "hash_key": request.config.hash_key,
            "seeding_scheme": request.config.seeding_scheme,
            "ignore_repeated_ngrams": request.config.ignore_repeated_ngrams,
        }
        detector_config = DetectionConfig(
            method_id="kgw",
            tokenizer_id="gpt2",
            key_id=TEST_KEY_ID,
            threshold=request.config.threshold,
            options=options,
        )
        result = self._detection_service.detect(normalized_text, detector_config)
        detection_decision = _validated_detection_decision(result.decision)
        compatibility = _validated_compatibility(result.compatibility)
        scoring_variant = _validated_scoring_variant(result.scoring_variant)
        detector_config_sha256 = _canonical_sha256(
            {
                "method_id": detector_config.method_id,
                "tokenizer_id": detector_config.tokenizer_id,
                "threshold": detector_config.threshold,
                "options": options,
                "key_epoch_reference": TEST_KEY_EPOCH_REFERENCE,
            }
        )
        bundle, outcome = self._attest_and_verify(
            nonce=request.nonce,
            normalized_text_sha256=compute_sha256(normalized_text),
            detector_id="kgw-local-test-v1",
            detector_code_sha256=self._detector_code_sha256,
            detector_config_sha256=detector_config_sha256,
            calibration_artifact_sha256=self._calibration_config_sha256,
            key_epoch_reference=TEST_KEY_EPOCH_REFERENCE,
            decision=_attested_decision(detection_decision),
            calibrated_fpr_upper_bound=None,
            validity=timedelta(minutes=2),
        )
        detached_evidence = self._detached_evidence(bundle.statement, bundle.evidence)
        warnings = (*result.warnings, EXPERIMENTAL_WARNING)

        return KgwTestDetectionResponse(
            detection=DetectionResultResponse(
                method_id="kgw",
                compatibility=compatibility,
                decision=detection_decision,
                score=result.score,
                p_value=result.p_value,
                token_count=result.token_count,
                estimated_fpr=None,
                warnings=warnings,
                original_scorable_token_count=result.original_scorable_token_count,
                ignored_repeated_ngrams=result.ignored_repeated_ngrams,
                scoring_variant=scoring_variant,
                calibration_id=result.calibration_id,
            ),
            statement=dict(bundle.statement),
            verification=VerificationSummary(
                valid=True,
                backend="tee-simulator",
                statement_id=outcome.statement_id,
                evidence_sha256=outcome.evidence_sha256,
                promotion_eligible=False,
                warnings=outcome.warnings,
            ),
            detached_evidence=detached_evidence,
        )

    def _attest_and_verify(
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
        validity: timedelta,
    ) -> tuple[AttestationBundle, VerificationOutcome]:
        bundle = self._backend.attest(
            nonce=nonce,
            normalized_text_sha256=normalized_text_sha256,
            detector_id=detector_id,
            detector_code_sha256=detector_code_sha256,
            detector_config_sha256=detector_config_sha256,
            calibration_artifact_sha256=calibration_artifact_sha256,
            key_epoch_reference=key_epoch_reference,
            decision=decision,
            calibrated_fpr_upper_bound=calibrated_fpr_upper_bound,
            issued_at=self._now(),
            validity=validity,
        )
        return bundle, self._verifier.verify(bundle)

    @staticmethod
    def _detached_evidence(
        statement: Mapping[str, object], evidence: bytes
    ) -> DetachedEvidence:
        descriptor = statement["attestation_evidence"]
        if not isinstance(descriptor, Mapping):
            raise TypeError("The simulator returned an invalid evidence descriptor.")
        return DetachedEvidence(
            artifact_id=str(descriptor["artifact_id"]),
            format="lab-hmac-tee-simulator-v1",
            sha256=str(descriptor["sha256"]),
            size_bytes=int(descriptor["size_bytes"]),
            encoding="base64",
            value=base64.b64encode(evidence).decode("ascii"),
        )
