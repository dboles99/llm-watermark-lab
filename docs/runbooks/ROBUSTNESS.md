# Robustness Measurement Runbook

## Purpose

The transformations below exist to measure detectors on watermarks this lab generated with its own keys. They are not a tool for removing third-party watermarks.

Robustness testing measures how well a generated watermark survives controlled text transformations. It answers the measurement question: "After ordinary edits, paraphrase, or redaction, does the keyed detector still recover the watermark?" This runbook describes the attack catalog, how to run sweeps, how to interpret survival curves, and the governance boundaries that must be observed.

Only synthetic, self-owned, or explicitly authorized text may be attacked in this lab. Attacks must never target third-party content without written authorization.

## Attack catalog

The lab implements five transformation types in `src/watermark_lab/attacks/transformations.py`:

| Attack ID | Description | Severity interpretation |
| --- | --- | --- |
| `deletion` | Randomly remove a fraction of word tokens. | Fraction of words deleted (0.0 to 1.0). |
| `insertion` | Insert random filler words between existing words. | Expected insertion density; number of inserted tokens scales with severity. |
| `substitution` | Replace a fraction of word tokens with words from a small lexicon. | Fraction of words replaced (0.0 to 1.0). |
| `shuffle` | Permute the order of all word tokens. | Always applied at full strength; severity is ignored and reported as 1.0. |
| `paraphrase` | Placeholder attack combining substitution plus deletion. | Simulates a low-fidelity paraphrase; severity controls both sub-attacks. |

These are deterministic given a seed and severity, so sweeps are reproducible. The `paraphrase` attack is a stub; a real paraphrase implementation would call a separate model or service and must be governed under the same authorization rules.

## CLI usage

Activate the environment and install the package:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run a single attack:

```powershell
watermark-lab attack `
  --parent-run-id <uuid> `
  --attack-id deletion `
  --severity 0.2 `
  --seed t04 `
  --method kgw `
  --key-id test-kgw-v1 `
  --threshold 4.0
```

Run a severity sweep:

```powershell
watermark-lab attack-sweep `
  --parent-run-id <uuid> `
  --attack-id deletion `
  --severities 0.0 0.1 0.2 0.3 0.4 0.5 `
  --seed t04 `
  --method kgw `
  --key-id test-kgw-v1
```

Expected flags (not all are wired in the current vertical slice):

| Flag | Meaning |
| --- | --- |
| `--parent-run-id` | UUID of the watermarked generation run to attack. |
| `--attack-id` | One of `deletion`, `insertion`, `substitution`, `shuffle`, `paraphrase`. |
| `--severity` | Per-attack severity; 0.0 is identity, 1.0 is maximum. |
| `--severities` | Space-separated list for a sweep. |
| `--seed` | Reproducibility seed for the random transformation. |
| `--method` | Detector method ID; only `kgw` is supported. |
| `--key-id` | Key identifier for the keyed detector. |
| `--threshold` | Detection threshold (default 4.0). |

Until the CLI commands are fully wired, use the Python API as shown in `notebooks/04_robustness_lab.ipynb`.

## Interpreting survival curves

A survival curve plots severity on the x-axis and a detector outcome on the y-axis. Typical y-axis choices are:

- Post-attack z-score.
- Binary survival flag (1 if `verified compatible watermark`, 0 otherwise).
- Absolute delta score (`post_score - pre_score`).

Use these rules when interpreting results:

- A run "survives" when the post-attack decision string is exactly `verified compatible watermark`.
- A flat curve near the pre-attack score means the watermark is robust to that attack.
- A steep drop or early failure means the attack removes the watermark signal.
- Compare curves across `attack_id` values; deletion and insertion often have different survival profiles.
- Do not overclaim: the mock provider and placeholder tokenizer do not represent real language-model output. Results are labeled `experimental` and `compatibility` must not be promoted to `verified` until real tokenizers, models, and calibration corpora are used.
- `estimated_fpr` is `None` until the calibration corpus is built. Without it, you cannot make empirical false-positive claims or report empirical detection rates.
- The threshold line (default 4.0) is a configuration choice, not a universal guarantee. Sensitivity analysis across thresholds is recommended before any operational decision.

## Governance and safety

- **Authorization**: only attack text that is synthetic, owned by your organization, or covered by a written authorization. User-generated content and third-party documents must not be attacked without explicit approval.
- **Key handling**: only `key_id` references are stored in reports, manifests, notebooks, and runbooks. Raw watermark keys must remain in the test-only in-memory vault or an approved secret store.
- **Outputs**: sweep results may include short excerpts of transformed synthetic text for debugging. Do not paste real third-party text into notebooks or terminal commands.
- **Retention**: keep attack artifacts inside `project/artifacts/`. Do not email or copy them to unauthorized locations.
- **Production readiness**: the current transformations are intentionally simple. Before using robustness results for product decisions, replace the placeholder tokenizer, integrate a real model provider, and build the calibration corpus.

## Tests

Run the attack unit tests and integration tests after any change to the transformation registry:

```powershell
.venv\Scripts\Activate.ps1
python -m pytest tests/unit/test_attack.py tests/unit/test_run_engine.py -q
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Unknown attack_id` | Use one of `deletion`, `insertion`, `substitution`, `shuffle`, `paraphrase`. |
| `Unknown key_id` | Use `--key-id test-kgw-v1` for the test fixture. |
| Pre-attack decision is `not detected` | The parent generation did not self-verify; regenerate with compatible KGW parameters. |
| All severities survive | Expected for synthetic token-id text and weak attacks; real text will usually show faster degradation. |
| No `estimated_fpr` | Calibration corpus is pending; see `config/calibration/kgw.toml`. |
