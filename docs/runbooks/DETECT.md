# Detect Watermarked Text

## Purpose

Run a keyed detector on a sample and produce a calibrated `DetectionResult` with input hashing, tokenizer metadata, and a valid run manifest.

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
watermark-lab detect --input "sample.txt" --method kgw --key-id test-kgw-v1 --threshold 4.0
```

Or with pasted text:

```powershell
watermark-lab detect --input "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog again." --method kgw --key-id test-kgw-v1 --threshold 4.0
```

## Expected output

```json
{
  "method_id": "kgw",
  "compatibility": "experimental",
  "decision": "not detected",
  "score": 0.09759000729485333,
  "p_value": 0.46112892982603093,
  "token_count": 35,
  "estimated_fpr": null,
  "threshold": 4.0,
  "warnings": [
    "estimated_fpr is null; calibration corpus not built (blocked-pending-corpus)"
  ],
  "input_media_type": "text/plain",
  "input_source_path": null,
  "input_sha256": "...",
  "normalized_sha256": "..."
}
```

## Decision language

The CLI and GUI use only these decision strings:

- `verified compatible watermark`
- `not detected`
- `inconclusive`

No provider attribution is reported.

## Run from the GUI

```powershell
.venv\Scripts\Activate.ps1
watermark-lab-gui
```

Open the **Detect** tab, paste text, set the key ID and threshold, then click **Run detection**.

## Short or empty input

Inputs that produce fewer than 30 scored tokens (or zero tokens) are reported as `inconclusive` with a `too_few_tokens` warning. The score and p-value are `None`.

## Calibration status

`estimated_fpr` is `None` until the calibration corpus in `config/calibration/kgw.toml` is built or authorized. The warning `blocked-pending-corpus` is always included in results until then.

## Tests

```powershell
.venv\Scripts\Activate.ps1
python -m pytest tests/unit/test_run_engine.py tests/integration/test_cli_detect.py tests/integration/test_gui_detect.py -q
```

## Safety notes

- Only `key_id` appears in reports and run manifests. Raw key bytes stay in the test-only in-memory vault.
- `test-kgw-v1` is a test-only key. Do not use for production.
- The default tokenizer is a deterministic placeholder. Real Hugging Face tokenizer integration is required before `reproduced` status.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Unknown key_id` | Use `--key-id test-kgw-v1` for the test fixture. |
| `Unsupported method_id` | Only `kgw` is implemented. |
| `threshold must be positive` | Use `--threshold` > 0. |
| `Unknown tokenizer family` | Only `gpt2` metadata is configured; use `--tokenizer-id gpt2`. |
