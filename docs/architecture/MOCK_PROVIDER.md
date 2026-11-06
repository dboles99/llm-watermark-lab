# Docker Mock Provider

The mock provider exists to build the application before connecting to Anthropic or another production provider.

## Interfaces

- `POST /v1/messages`: Anthropic-shaped request and response envelope
- `POST /v1/chat/completions`: OpenAI-compatible envelope
- `GET /health`: health check

It deliberately does not reproduce Claude behavior. Every response contains a simulated-provider marker. Watermark profiles are deterministic test-fixture labels, not reverse-engineered provider algorithms.

## Start

```powershell
docker compose up -d --build
```

## Test profiles

Pass `metadata.watermark_profile` as one of:

- `clean`
- `kgw-test`
- `mcmark-inspired-test`
- `mirrormark-inspired-test`
- `malformed-test`
- `timeout-test`

The first scaffold returns deterministic synthetic text. Later implementation tasks will connect these labels to the actual local method plugins and fault-injection middleware.

## Safety boundary

The service name and responses must never use branding that implies Anthropic endorsement or compatibility. “Claude-shaped” refers only to the public API envelope used in local integration tests.

