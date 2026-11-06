# Local TEE Test Server

> **LOCAL NON-HARDWARE SIMULATOR ONLY.** Its receipts are always marked
> `simulated-non-hardware`, always have `promotion_eligible: false`, and cannot
> satisfy a production, scientific-promotion, or remote-attestation gate.

This server exercises the proof-carrying verification HTTP contract around the
existing HMAC simulator. It offers both a hash-only receipt endpoint and a narrow
end-to-end endpoint that normalizes transient text and runs the existing
experimental KGW `DetectionService` with its fixed test key and key epoch. Key
material is never accepted from a request.

## Safety properties

- The application refuses to start unless `WATERMARK_LAB_TEE_DEPLOYMENT_MODE` is exactly
  `local-development`; `production`, `staging`, and unknown values fail closed.
- Every HTTP response carries simulator, non-hardware, and non-promotable headers;
  every successful contract response repeats those labels in its JSON body.
- Access logging is disabled. The application does not log request bodies.
- Requests are strict. The hash-only endpoint never accepts text. The KGW test
  endpoint accepts text only in the request body and never returns or persists it;
  validation errors omit submitted values.
- Nonces are process-local, single-use, and checked by the existing simulator
  verifier. Simulated statements expire after at most ten minutes.
- The default signing key is generated in memory at process start and is lost on
  restart. The only persistent-key option is a file mounted at the fixed path
  `/run/secrets/tee_simulator_key`.
- There is no environment-variable or command-line option for signing-key bytes.

## Run directly

From the repository root:

```powershell
$env:WATERMARK_LAB_TEE_DEPLOYMENT_MODE = "local-development"
python -m watermark_lab.tee_server
```

The direct runner binds only to `127.0.0.1:8787`, uses one worker, and disables
access logs.

## Run with Docker Compose

```powershell
docker compose -f deploy/local-tee/compose.yaml up --build
```

The process listens on the container interface, but Compose publishes it only as
`127.0.0.1:8787` on the host. The container is non-root, read-only, drops all Linux
capabilities, and uses a small temporary filesystem. An `internal: true` Docker
network was tested but removed because Docker Desktop stopped publishing the
loopback port; do not claim container-level egress isolation for this profile.

To use a stable local test key, create a file of at least 32 random bytes without
printing its contents, then apply the example secret override:

```powershell
New-Item -ItemType Directory -Force deploy/local-tee/secrets | Out-Null
$teeKey = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($teeKey)
[IO.File]::WriteAllBytes(
  (Join-Path $PWD "deploy/local-tee/secrets/tee_simulator_key"),
  $teeKey
)
Remove-Variable teeKey
docker compose `
  -f deploy/local-tee/compose.yaml `
  -f deploy/local-tee/compose.secret.override.example.yaml `
  up --build
```

The `deploy/local-tee/secrets/` directory ignores all contents except its
`.gitignore`. Never place the key in Compose environment variables, command-line
arguments, source files, logs, or issue reports.

## HTTP contract

1. `GET /health` — liveness, key lifecycle, and non-hardware labels.
2. `GET /v1/capabilities` — endpoint input and process-local persistence contract.
3. `POST /v1/nonces` — issue a fresh process-local nonce.
4. `POST /v1/simulated-verifications` — create and immediately verify a simulated
   receipt using that nonce; a replay returns HTTP 409.
5. `POST /v1/kgw-test-detections` — transiently normalize raw text, run the fixed-key
   experimental KGW detector, and return `DetectionResult` fields plus the simulated
   statement and detached evidence. Raw text and key bytes are absent from responses.
   The `calibration_artifact_sha256` field commits to `config/calibration/kgw.toml`
   as configuration only; it does not claim that a calibration corpus exists.

Example with Python, which avoids shell interpolation of request data:

```python
import httpx

base_url = "http://127.0.0.1:8787"
with httpx.Client(base_url=base_url, timeout=5.0) as client:
    nonce = client.post("/v1/nonces").raise_for_status().json()["nonce"]
    response = client.post(
        "/v1/simulated-verifications",
        json={
            "nonce": nonce,
            "normalized_text_sha256": "1" * 64,
            "detector_id": "kgw",
            "detector_code_sha256": "2" * 64,
            "detector_config_sha256": "3" * 64,
            "calibration_artifact_sha256": "4" * 64,
            "key_epoch_reference": "epoch_local_0001",
            "decision": "inconclusive",
            "calibrated_fpr_upper_bound": None,
        },
    )
    receipt = response.raise_for_status().json()
    assert receipt["labels"]["promotion_eligible"] is False
    assert receipt["statement"]["simulated"] is True
```

End-to-end KGW simulator request:

```python
with httpx.Client(base_url=base_url, timeout=5.0) as client:
    nonce = client.post("/v1/nonces").raise_for_status().json()["nonce"]
    response = client.post(
        "/v1/kgw-test-detections",
        json={
            "nonce": nonce,
            "text": "Transient input sent only to the local simulator.",
            "config": {
                "method_id": "kgw",
                "tokenizer_id": "gpt2",
                "threshold": 4.0,
                "gamma": 0.25,
                "context_width": 1,
                "hash_key": 15485863,
                "seeding_scheme": "simple_1",
                "ignore_repeated_ngrams": False,
            },
        },
    )
    receipt = response.raise_for_status().json()
    assert receipt["detection"]["estimated_fpr"] is None
    assert receipt["detection"]["compatibility"] == "experimental"
    assert "text" not in receipt
```

With the Compose service running, execute the no-dependency Docker smoke contract:

```powershell
python deploy/local-tee/smoke_test.py
```

The `detached_evidence.value` field is base64-encoded simulator HMAC evidence. It
is not platform attestation and must never be stored or displayed as such.

## Expected failure modes and limitations

- Restarting the process invalidates all outstanding nonces and, with the default
  ephemeral key, all earlier simulator evidence.
- Multiple workers or replicas do not share nonce state. This runbook deliberately
  fixes the service at one worker.
- The service verifies its own newly created simulator envelope. It does not prove
  isolated detector execution, isolate memory, protect against a hostile host, release a
  production key, or supply Azure/AMD/Intel attestation evidence.
- The KGW endpoint uses the repository's minimal deterministic tokenizer, fixed
  test key, and calibration configuration whose corpus status remains
  `blocked-pending-corpus`. `estimated_fpr` is therefore always null, short text is
  inconclusive, and every result includes an experimental warning.
- Repeated-ngram masking is surfaced as `compatibility: prototype`, is forced to
  `inconclusive`, and reports its original scorable count, ignored-ngram count,
  scoring variant, and null calibration identifier.
- Raw text sent to the KGW endpoint exists in ordinary process memory. The local
  container provides no confidentiality from its host and must not receive data
  that requires a real enclave boundary.
- A real TEE adapter must bind actual isolated detector execution and platform
  evidence to the statement and must live outside this simulator package.
- Do not expose port 8787 beyond loopback, add a reverse proxy, or treat TLS as a
  substitute for hardware attestation.

Stop the container with:

```powershell
docker compose -f deploy/local-tee/compose.yaml down
```
