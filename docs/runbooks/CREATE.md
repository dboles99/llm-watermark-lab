# Create Watermarked Text

## Purpose

Generate text that carries a configured statistical watermark and pass keyed self-verification before the artifact is accepted.

## Prerequisites

- Python 3.12+
- Repository cloned and installed:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run from the CLI

```powershell
.venv\Scripts\Activate.ps1
watermark-lab create --prompt "The quick brown fox jumps over the lazy dog." --disclosure
```

Full example with explicit parameters:

```powershell
watermark-lab create `
  --prompt "Explain the water cycle." `
  --profile mock `
  --method kgw `
  --key-id test-kgw-v1 `
  --disclosure `
  --max-new-tokens 80 `
  --gamma 0.25 `
  --delta 2.0 `
  --context-width 1
```

## Expected output

```json
{
  "run_id": "...",
  "status": "completed",
  "text": "...",
  "tokens": [...],
  "prompt_sha256": "...",
  "response_sha256": "...",
  "model_profile": {
    "profile_id": "mock",
    "provider": "mock_local",
    "model": "kgw-test",
    "simulated": true
  },
  "method_id": "kgw",
  "key_id": "test-kgw-v1",
  "self_verification": {
    "decision": "verified compatible watermark",
    "score": 15.39,
    "p_value": 0.0,
    "token_count": 79,
    "estimated_fpr": null,
    "warnings": [
      "estimated_fpr is null; calibration corpus not built (blocked-pending-corpus)"
    ]
  }
}
```

## Run from the GUI

```powershell
.venv\Scripts\Activate.ps1
watermark-lab-gui
```

Open the **Create** tab, enter a prompt, select the method and key identifier, and click **Generate**. The GUI enforces the same governance checks as the CLI and rejects the output if self-verification fails.

## Governance flags and overrides

The CLI and GUI record three governance fields:

| Field | CLI flag | When to enable |
| --- | --- | --- |
| `disclosure_recorded` | `--disclosure` | Always enable for production use; required by default. |
| `source_is_third_party` | GUI / API option | Only when `governance.allow_third_party_documents` is true in `config/example.toml`. |
| `rewriting_existing_text` | GUI / API option | Only when `governance.allow_existing_text_rewatermarking` is true. |

By default `config/example.toml` blocks third-party documents and rewatermarking. To override an allowed workflow, edit the governance section explicitly and document the decision in the run manifest.

## Self-verification expectation

Every accepted creation must self-verify as `verified compatible watermark` with at least 30 scored tokens. If self-verification returns `not detected` or `inconclusive`, the run status is `failed` and the text is rejected.

## Interpreting results and warnings

- `verified compatible watermark` with a high positive z-score means the keyed detector sees the expected green-list structure.
- `estimated_fpr: null` is expected until a calibration corpus is authorized and built; it does not mean the generation failed.
- The `blocked-pending-corpus` warning remains in results until empirical calibration is completed.

## Safety notes

- Only `key_id` appears in reports and run manifests. Raw key bytes stay in the test-only in-memory vault.
- `test-kgw-v1` is a test-only key. Do not use for production.
- The default mock provider is deterministic only for a fixed seed; regenerated text may vary between runs.
- Do not include third-party text, API keys, or raw watermark keys in prompts or notebooks.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `GovernanceError: Disclosure is required` | Add `--disclosure` to the CLI command. |
| `GovernanceError: Third-party documents are not allowed` | Set `allow_third_party_documents = true` in `config/example.toml` only after authorization. |
| `SelfVerificationError` | The generated text did not verify; try a longer prompt or higher `max-new-tokens`, or check `gamma`/`delta` settings. |
| `Unknown key_id` | Use `--key-id test-kgw-v1` for the test fixture. |
| `Unsupported method_id` | Only `kgw` is implemented. |
| `estimated_fpr is null` | Expected until the calibration corpus is built; see `config/calibration/kgw.toml`. |
