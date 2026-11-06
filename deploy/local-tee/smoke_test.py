"""Standard-library smoke test for the local non-hardware TEE simulator."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

BASE_URL = "http://127.0.0.1:8787"
SIMULATOR_CLASS = "simulated-non-hardware"


def request_json(
    path: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any], dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return (
                response.status,
                json.loads(response.read().decode("utf-8")),
                {name.lower(): value for name, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            json.loads(exc.read().decode("utf-8")),
            {name.lower(): value for name, value in exc.headers.items()},
        )


def main() -> int:
    status, health, headers = request_json("/health")
    assert status == 200
    assert health["labels"]["attestation_class"] == SIMULATOR_CLASS
    assert health["labels"]["hardware_attestation"] is False
    assert headers["x-lab-promotion-eligible"] == "false"

    _, nonce_body, _ = request_json("/v1/nonces", method="POST")
    raw_text = "local-smoke-sensitive-marker " + " ".join(
        f"token-{index}" for index in range(40)
    )
    request_body = {"nonce": nonce_body["nonce"], "text": raw_text}
    status, receipt, _ = request_json(
        "/v1/kgw-test-detections", method="POST", payload=request_body
    )
    assert status == 201
    assert raw_text not in json.dumps(receipt)
    assert receipt["detection"]["compatibility"] == "experimental"
    assert receipt["detection"]["estimated_fpr"] is None
    assert receipt["statement"]["simulated"] is True
    assert receipt["verification"]["promotion_eligible"] is False

    replay_status, replay, replay_headers = request_json(
        "/v1/kgw-test-detections", method="POST", payload=request_body
    )
    assert replay_status == 409
    assert replay["error"] == "simulated_attestation_rejected"
    assert replay_headers["x-lab-hardware-attestation"] == "false"

    _, short_nonce, _ = request_json("/v1/nonces", method="POST")
    short_status, short_receipt, _ = request_json(
        "/v1/kgw-test-detections",
        method="POST",
        payload={"nonce": short_nonce["nonce"], "text": "short"},
    )
    assert short_status == 201
    assert short_receipt["detection"]["decision"] == "inconclusive"

    _, masked_nonce, _ = request_json("/v1/nonces", method="POST")
    masked_status, masked_receipt, _ = request_json(
        "/v1/kgw-test-detections",
        method="POST",
        payload={
            "nonce": masked_nonce["nonce"],
            "text": "5 6 5 6 5 6 7",
            "config": {"ignore_repeated_ngrams": True},
        },
    )
    assert masked_status == 201
    assert masked_receipt["detection"]["compatibility"] == "prototype"
    assert masked_receipt["detection"]["decision"] == "inconclusive"
    assert masked_receipt["detection"]["ignored_repeated_ngrams"] > 0

    print(
        json.dumps(
            {
                "status": "ok",
                "attestation_class": SIMULATOR_CLASS,
                "hardware_attestation": False,
                "promotion_eligible": False,
                "kgw_detection": receipt["detection"]["decision"],
                "short_text": short_receipt["detection"]["decision"],
                "masked_scoring": masked_receipt["detection"]["compatibility"],
                "replay_status": replay_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
