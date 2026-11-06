# War-Dial Pilot Runbook

## Purpose

The war-dial pilot systematically enumerates textual-watermark parameter
combinations — like wardialing phone numbers, but for watermark operating
points.  Each "dial" runs a fast coding test, a KGW creation + stego encoding
roundtrip, a stochastic perturbation, and (optionally) a KGW attack sweep
that feeds an empirical survival surface.

This runbook covers the CLI and the JSONL output.

## CLI usage

Activate the environment and install the package:

```powershell
.venv\Scripts\Activate.ps1
uv pip install -e .
```

The war-dial pilot is available both as a standalone script and as a subcommand
of the main `watermark-lab` CLI.

### Run the default grid

```powershell
uv run python scripts/war_dial.py --output-dir artifacts/war-dial
# or
uv run watermark-lab war-dial --output-dir artifacts/war-dial
```

This writes one JSONL record per dial coordinate to
`artifacts/war_dial_results.jsonl` and a `gate_report.json` with the four-state
gate (artifact_complete, checks_passed, promotion_eligible, deployment_ready).

### Run with KGW attack sweeps and survival surface

```powershell
uv run python scripts/war_dial.py `
  --attack-sweep `
  --survival-surface `
  --output-dir artifacts/war-dial
# or
uv run watermark-lab war-dial --attack-sweep --survival-surface --output-dir artifacts/war-dial
```

With `--attack-sweep`, every KGW coordinate triggers `deletion`, `insertion`,
`substitution`, and `shuffle` attacks at severities `0.1`, `0.3`, `0.5`. The
results are aggregated into a `survival_surface.json` using the EditReceipt
frontier machinery.

### Reconcile existing results and plan the next grid

```powershell
uv run python scripts/war_dial.py --reconcile --output-dir artifacts/war-dial
# or
uv run watermark-lab war-dial --reconcile --output-dir artifacts/war-dial
```

The reconciler loads the JSONL file, reports failures, finds numeric boundaries
where survival flips, and suggests refinements for the next iteration.

### Use a heavier coding test

```powershell
uv run python scripts/war_dial.py `
  --coding-test -m pytest tests/stego/test_stego.py -q `
  --output-dir artifacts/war-dial
```

The default coding test is a fast in-process stego roundtrip. Passing a custom
`--coding-test` runs a subprocess `pytest` subset instead.

## Output files

| File | Purpose |
| --- | --- |
| `war_dial_results.jsonl` | Append-only record of every dial coordinate. |
| `gate_report.json` | Four-state gate report with summary and findings. |
| `survival_surface.json` | Aggregated survival surface (only with `--survival-surface`). |

## JSONL record schema

Each line is a JSON object with:

- `dial_id`: iteration-index coordinate identifier.
- `coordinate`: the KGW, stego, and perturbation settings.
- `coding_passed`, `coding_stdout`, `coding_stderr`: result of the coding test.
- `kgw_created`, `kgw_decision`, `kgw_token_count`: KGW self-verification.
- `kgw_parent_run`: full parent run dict (when KGW succeeded).
- `attack_results`: list of RobustnessService sweep results (when `--attack-sweep`).
- `survival_surface`: per-dial survival surface (when `--attack-sweep`).
- `stego_encoded`, `stego_bits_embedded`, `stego_decode_status`, `stego_verify_status`.
- `perturbation_applied`, `perturbed_decode_status`, `perturbed_text_hash`.


## Governance and limitations

- The demo grid uses the `mock` provider and synthetic token sequences. It is
  explicitly non-scientific and cannot support a frozen calibration claim or
  deployment readiness.
- Do not run attacks on third-party text. Only project-owned or explicitly
  authorized synthetic text may be attacked.
- All keys are test-only placeholders (`test-kgw-v1`, `war-dial-secret`).
- Survival surfaces are empirical fractions, not calibrated probabilities.
  Real claims require T06 calibration and a preregistered transformation
  registry.
- Real-model coordinates require budget/platform approval. The CLI is
  designed so that switching the `profile_id` from `mock` to a real provider
  only requires updating `KgwDial.profile_id` once a provider adapter exists.

## Tests

- `tests/unit/test_war_dial.py` covers the harness, grid resume, reconciler,
  CLI script entry point, attack-sweep survival surfaces, and the four-state
  gate report.
