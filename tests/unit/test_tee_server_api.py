from __future__ import annotations

import base64
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from watermark_lab.run_engine.intake import compute_sha256, normalize_text
from watermark_lab.tee_server.app import create_app
from watermark_lab.tee_server.config import TeeServerConfig

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
TEST_KEY = b"deterministic-local-simulator-key!!"
ROOT = Path(__file__).resolve().parents[2]


def _client(tmp_path: Path) -> TestClient:
    secret_file = tmp_path / "tee_simulator_key"
    secret_file.write_bytes(TEST_KEY)
    app = create_app(TeeServerConfig(secret_file=secret_file), now=lambda: NOW)
    return TestClient(app)


def _request(nonce: str) -> dict[str, object]:
    return {
        "nonce": nonce,
        "normalized_text_sha256": "1" * 64,
        "detector_id": "kgw",
        "detector_code_sha256": "2" * 64,
        "detector_config_sha256": "3" * 64,
        "calibration_artifact_sha256": "4" * 64,
        "key_epoch_reference": "epoch_local_0001",
        "decision": "inconclusive",
        "calibrated_fpr_upper_bound": None,
    }


def test_tee_server_health_and_capabilities_are_non_hardware_and_non_promotable(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        health = client.get("/health")
        capabilities = client.get("/v1/capabilities")

    assert health.status_code == 200
    assert health.headers["x-lab-attestation-class"] == "simulated-non-hardware"
    assert health.headers["x-lab-hardware-attestation"] == "false"
    assert health.headers["x-lab-promotion-eligible"] == "false"
    assert health.json()["labels"]["hardware_attestation"] is False
    assert health.json()["labels"]["promotion_eligible"] is False
    assert health.json()["key_lifecycle"] == "mounted-secret-file"
    assert capabilities.json()["hash_only_endpoint_accepts_raw_text"] is False
    assert capabilities.json()["kgw_test_endpoint_accepts_transient_raw_text"] is True
    assert capabilities.json()["raw_text_returned_or_persisted"] is False
    assert capabilities.json()["accepts_key_material"] is False


def test_tee_server_issues_nonce_and_returns_verified_simulator_receipt(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        nonce_response = client.post("/v1/nonces")
        nonce = nonce_response.json()["nonce"]
        response = client.post("/v1/simulated-verifications", json=_request(nonce))

    assert nonce_response.status_code == 201
    assert response.status_code == 201
    body = response.json()
    assert body["labels"]["attestation_class"] == "simulated-non-hardware"
    assert body["labels"]["promotion_eligible"] is False
    assert body["statement"]["backend"] == "tee-simulator"
    assert body["statement"]["simulated"] is True
    assert body["statement"]["verification_status"] == "simulated-only"
    assert body["verification"]["valid"] is True
    assert body["verification"]["promotion_eligible"] is False
    evidence = base64.b64decode(body["detached_evidence"]["value"], validate=True)
    assert hashlib.sha256(evidence).hexdigest() == body["detached_evidence"]["sha256"]
    assert len(evidence) == body["detached_evidence"]["size_bytes"]


def test_tee_server_rejects_nonce_replay_with_no_sensitive_echo(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        nonce = client.post("/v1/nonces").json()["nonce"]
        request = _request(nonce)
        assert client.post("/v1/simulated-verifications", json=request).status_code == 201
        replay = client.post("/v1/simulated-verifications", json=request)

    assert replay.status_code == 409
    assert replay.json()["error"] == "simulated_attestation_rejected"
    assert TEST_KEY.decode("ascii") not in replay.text


def test_tee_server_rejects_raw_text_and_key_fields_without_echoing_values(tmp_path: Path) -> None:
    raw_text = "private source text that must never be echoed"
    submitted_key = "private detector key that must never be echoed"
    with _client(tmp_path) as client:
        nonce = client.post("/v1/nonces").json()["nonce"]
        request = _request(nonce)
        request["raw_text"] = raw_text
        request["detector_key"] = submitted_key
        response = client.post("/v1/simulated-verifications", json=request)

    assert response.status_code == 422
    assert response.headers["x-lab-attestation-class"] == "simulated-non-hardware"
    assert response.headers["x-lab-hardware-attestation"] == "false"
    assert response.headers["x-lab-promotion-eligible"] == "false"
    assert raw_text not in response.text
    assert submitted_key not in response.text
    assert response.json()["error"] == "hash_only_contract_validation_failed"


def test_tee_server_openapi_exposes_hash_only_request_contract(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        schema = client.get("/openapi.json").json()

    properties = schema["components"]["schemas"]["SimulatedVerificationRequest"]["properties"]
    assert "normalized_text_sha256" in properties
    assert "raw_text" not in properties
    assert "key" not in properties
    assert "detector_key" not in properties


def test_tee_server_runs_end_to_end_kgw_and_binds_normalized_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_text = "e\u0301 " + " ".join(f"word-{index}" for index in range(80))
    caplog.set_level(logging.DEBUG)
    with _client(tmp_path) as client:
        nonce = client.post("/v1/nonces").json()["nonce"]
        response = client.post(
            "/v1/kgw-test-detections",
            json={"nonce": nonce, "text": raw_text},
        )

    assert response.status_code == 201
    assert raw_text not in response.text
    assert raw_text not in caplog.text
    body = response.json()
    assert body["labels"]["attestation_class"] == "simulated-non-hardware"
    assert body["labels"]["hardware_attestation"] is False
    assert body["labels"]["promotion_eligible"] is False
    assert body["detection"]["method_id"] == "kgw"
    assert body["detection"]["compatibility"] == "experimental"
    assert body["detection"]["estimated_fpr"] is None
    assert body["detection"]["token_count"] == 80
    assert body["detection"]["original_scorable_token_count"] == 80
    assert body["detection"]["ignored_repeated_ngrams"] == 0
    assert body["detection"]["scoring_variant"] == "naive-binomial-uncalibrated"
    assert body["detection"]["calibration_id"] is None
    assert body["calibration_commitment_kind"] == "configuration-only-uncalibrated"
    assert any("blocked-pending-corpus" in item for item in body["detection"]["warnings"])
    assert any("not scientific" in item for item in body["detection"]["warnings"])
    statement = body["statement"]
    assert statement["normalized_text_sha256"] == compute_sha256(normalize_text(raw_text))
    assert statement["detector_id"] == "kgw-local-test-v1"
    assert statement["key_epoch_reference"] == "epoch_local_kgw_test_v1"
    assert statement["calibrated_fpr_upper_bound"] is None
    calibration_bytes = (ROOT / "config" / "calibration" / "kgw.toml").read_bytes()
    assert statement["calibration_artifact_sha256"] == hashlib.sha256(
        calibration_bytes
    ).hexdigest()
    assert statement["simulated"] is True
    assert statement["verification_status"] == "simulated-only"
    assert body["verification"]["promotion_eligible"] is False


def test_tee_server_kgw_code_config_and_calibration_hashes_are_deterministic(
    tmp_path: Path,
) -> None:
    request_text = " ".join(f"stable-{index}" for index in range(40))
    statements: list[dict[str, object]] = []
    with _client(tmp_path) as client:
        for _ in range(2):
            nonce = client.post("/v1/nonces").json()["nonce"]
            response = client.post(
                "/v1/kgw-test-detections",
                json={"nonce": nonce, "text": request_text},
            )
            assert response.status_code == 201
            statements.append(response.json()["statement"])

    committed_fields = (
        "normalized_text_sha256",
        "detector_code_sha256",
        "detector_config_sha256",
        "calibration_artifact_sha256",
    )
    for field in committed_fields:
        assert statements[0][field] == statements[1][field]


def test_tee_server_kgw_short_text_is_inconclusive(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        nonce = client.post("/v1/nonces").json()["nonce"]
        response = client.post(
            "/v1/kgw-test-detections",
            json={"nonce": nonce, "text": "short"},
        )

    assert response.status_code == 201
    assert response.json()["detection"]["decision"] == "inconclusive"
    assert response.json()["detection"]["estimated_fpr"] is None
    assert response.json()["statement"]["decision"] == "inconclusive"


def test_tee_server_kgw_masked_scoring_remains_prototype_and_inconclusive(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        nonce = client.post("/v1/nonces").json()["nonce"]
        response = client.post(
            "/v1/kgw-test-detections",
            json={
                "nonce": nonce,
                "text": "5 6 5 6 5 6 7",
                "config": {"ignore_repeated_ngrams": True},
            },
        )

    assert response.status_code == 201
    detection = response.json()["detection"]
    assert detection["compatibility"] == "prototype"
    assert detection["decision"] == "inconclusive"
    assert detection["scoring_variant"] == "unique-ngram-uncalibrated"
    assert detection["ignored_repeated_ngrams"] > 0
    assert detection["estimated_fpr"] is None


def test_tee_server_kgw_nonce_replay_is_rejected(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        nonce = client.post("/v1/nonces").json()["nonce"]
        request = {"nonce": nonce, "text": "one two three"}
        assert client.post("/v1/kgw-test-detections", json=request).status_code == 201
        replay = client.post("/v1/kgw-test-detections", json=request)

    assert replay.status_code == 409
    assert replay.json()["error"] == "simulated_attestation_rejected"


def test_tee_server_kgw_rejects_unconstrained_config_without_echo(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_text = "sensitive-text-must-not-be-returned-or-logged"
    submitted_key = "sensitive-key-must-not-be-returned-or-logged"
    caplog.set_level(logging.DEBUG)
    with _client(tmp_path) as client:
        nonce = client.post("/v1/nonces").json()["nonce"]
        response = client.post(
            "/v1/kgw-test-detections",
            json={
                "nonce": nonce,
                "text": raw_text,
                "config": {"method_id": "mcmark", "gamma": 0.5},
                "detector_key": submitted_key,
            },
        )

    assert response.status_code == 422
    assert response.json()["error"] == "kgw_test_contract_validation_failed"
    assert raw_text not in response.text
    assert submitted_key not in response.text
    assert raw_text not in caplog.text
    assert submitted_key not in caplog.text
    assert response.headers["x-lab-hardware-attestation"] == "false"
    assert response.headers["x-lab-promotion-eligible"] == "false"
