"""ASGI application for the explicitly non-hardware local TEE test server."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from watermark_lab.security import AttestationVerificationError
from watermark_lab.tee_server.config import TeeServerConfig
from watermark_lab.tee_server.models import (
    CapabilitiesResponse,
    HealthResponse,
    KgwTestDetectionRequest,
    KgwTestDetectionResponse,
    NonceResponse,
    SimulatedVerificationRequest,
    SimulatedVerificationResponse,
)
from watermark_lab.tee_server.service import TeeSimulatorService, utc_now

SIMULATOR_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Lab-Attestation-Class": "simulated-non-hardware",
    "X-Lab-Hardware-Attestation": "false",
    "X-Lab-Promotion-Eligible": "false",
}


def create_app(
    config: TeeServerConfig | None = None,
    *,
    now: Callable[[], datetime] = utc_now,
) -> FastAPI:
    """Create a one-process simulator app; non-local modes fail during construction."""

    resolved_config = config or TeeServerConfig.from_environment()
    service = TeeSimulatorService(resolved_config, now=now)
    application = FastAPI(
        title="Local NON-HARDWARE TEE Simulator",
        description=(
            "Development contract server only. All evidence is simulated, non-hardware, "
            "and permanently ineligible for scientific or deployment promotion."
        ),
        version="1.0.0-simulator",
    )
    application.state.tee_simulator_service = service

    @application.middleware("http")
    async def add_simulator_labels(request: Request, call_next: Callable) -> JSONResponse:
        response = await call_next(request)
        for name, value in SIMULATOR_HEADERS.items():
            response.headers[name] = value
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        is_kgw_test_request = request.url.path == "/v1/kgw-test-detections"
        errors = [
            {
                "location": [str(part) for part in error.get("loc", ())],
                "type": str(error.get("type", "validation_error")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": (
                    "kgw_test_contract_validation_failed"
                    if is_kgw_test_request
                    else "hash_only_contract_validation_failed"
                ),
                "message": (
                    "The request did not match the constrained KGW test contract."
                    if is_kgw_test_request
                    else "The request did not match the hash-only simulator contract."
                ),
                "errors": errors,
            },
        )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(key_lifecycle=service.key_lifecycle)

    @application.get("/v1/capabilities", response_model=CapabilitiesResponse)
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse()

    @application.post("/v1/nonces", response_model=NonceResponse, status_code=201)
    async def issue_nonce() -> NonceResponse:
        return NonceResponse(nonce=service.issue_nonce())

    @application.post(
        "/v1/simulated-verifications",
        response_model=SimulatedVerificationResponse,
        status_code=201,
    )
    async def simulated_verification(
        verification_request: SimulatedVerificationRequest,
    ) -> SimulatedVerificationResponse | JSONResponse:
        try:
            return service.verify(verification_request)
        except AttestationVerificationError:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "simulated_attestation_rejected",
                    "message": "The nonce, freshness, replay, or evidence policy rejected the request.",
                },
            )
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "simulated_attestation_input_rejected",
                    "message": "The commitment-only request was rejected by the simulator contract.",
                },
            )

    @application.post(
        "/v1/kgw-test-detections",
        response_model=KgwTestDetectionResponse,
        status_code=201,
    )
    async def kgw_test_detection(
        detection_request: KgwTestDetectionRequest,
    ) -> KgwTestDetectionResponse | JSONResponse:
        try:
            return service.detect_kgw(detection_request)
        except AttestationVerificationError:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "simulated_attestation_rejected",
                    "message": "The nonce, freshness, replay, or evidence policy rejected the request.",
                },
            )
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=422,
                content={
                    "error": "kgw_test_configuration_rejected",
                    "message": "The constrained KGW test configuration was rejected.",
                },
            )

    return application


app = create_app()
