# LLM Watermark Lab

**[Read the wiki](https://github.com/dboles99/llm-watermark-lab/wiki)** for how it works, the evidence, the design decisions and the limits.

Designed and authored by Daniel Boles; implemented with coding agents.

A research lab for measuring and detecting statistical watermarks in text produced by large
language models. It generates watermarked text with its own keys and matched clean controls, runs
keyed and blind detectors, calibrates them, and records how much of the signal survives ordinary
edits and print-scan-OCR.

A detection identifies a compatible watermark, never an author. A keyed detector can verify only a
compatible watermark. Keyless analysis can flag unusual statistical structure, but it cannot
identify a provider or prove AI authorship. Every result is reported with its method, tokenizer,
threshold and false-positive estimate, and a method-specific result is never collapsed into one "AI
percentage".

## What it measures

- **Keyed detection.** A KGW reference path pinned to the author's profile, plus MCmark-inspired and
  MirrorMark-inspired research adapters. These are independent research implementations, not claims
  of compatibility with any proprietary system.
- **Blind detection.** Seven candidate tests in `src/watermark_lab/blind_detection/`, including
  cross-document variance ratio, perturbation invariance, entropy-conditioned tests and reference
  corpus two-sample tests. All are hypothesis tests, not attribution tools.
- **Calibration.** Empirical nulls, exact false-positive intervals, multiplicity control for
  parameter sweeps, and worst-group rates.
- **Survival.** Controlled transformations applied to watermarks this lab generated, reported as
  survival curves. Provenance channels and OCR round trips are measured the same way.
- **Unknown patterns.** Scans for hidden Unicode, whitespace and document-container anomalies,
  reported without any claim about their origin.

## Measured results

All from `project/artifacts/`, on a pinned GPT-2 model at revision `607a30d7`, fully offline, with
matched unwatermarked controls. None is a calibrated detection claim.

- **A statistical watermark survives print-scan-OCR while container-bound channels do not.** 4 of 4
  scorable cells retained a mean of 88% of the clean z-score. Zero-width and whitespace channels
  recovered 0% at every severity. A fifth cell was not scorable.
  (`ocr-survival/2026-08-30/`)
- **OCR damage depends on locale.** The median token-to-character error ratio is 12.0 for `en` and
  0.92 for `zh-Hant-TW`, so a single OCR-robustness figure does not transfer across locales.
  (`ocr-locale/2026-08-31/`)
- **The default recogniser introduces Simplified Chinese contamination** in 6 of 12 Traditional
  Chinese readings. (`ocr-locale/2026-08-31/`)
- **KGW mechanics match the author's code.** Differential tests found zero mismatches in 10,000
  green-list cases, 1,000 token-score cases and six pinned-tokenizer raw-text cases. This is bounded
  mechanics evidence, not detector promotion.
  (`scientific-validation/kgw-author-parity/`)
- **The pilot OCR calibration cannot support a low false-positive claim.** With 23 clusters the
  tightest attainable one-sided 95% bound is 19%, against a 0.1% target, so every OCR reading returns
  `inconclusive` by design. (`ocr-calibration/2026-08-31/`)

## Layout

```text
src/watermark_lab/   detectors, generators, statistics, OCR, storage, CLI, GUI
crates/              optional Rust core (keyed primitives, blind detection)
tests/               unit, integration and blind-detection tests
schemas/             versioned JSON schemas for manifests and evidence
project/artifacts/   measured evidence that the results above come from
research/            bibliography, evidence ledger, pinned paper manifests (no PDFs)
docs/                architecture, governance, research protocols, runbooks
notebooks/           calibration, benchmark and robustness notebooks
mock-server/         deterministic simulated provider for offline development
deploy/local-tee/    local attestation simulator, always labelled non-hardware
```

## Run it

```powershell
pip install -e ".[dev,ocr]" ./crates/watermark-core
pytest                                   # 679 passed, 8 skipped on 2026-11-06
cargo test --workspace --locked          # 20 passed on 2026-11-06
python scripts/validate_repository.py --skip-notebooks
```

```powershell
watermark-lab runs
watermark-lab ocr-engines
watermark-lab ocr-survival sample.txt
watermark-lab generation-environment
```

Optional dependency groups: `ocr` (offline RapidOCR, no system binary) and `generation` (torch and
transformers). Model weights are never downloaded by the lab. A directory without them is refused,
because a silent fetch would make the recorded model identity describe mutable remote bytes.

See `docs/architecture/SYSTEM_ARCHITECTURE.md` and `docs/research/SCIENTIFIC_ROADMAP.md`. Literature
enters the evidence ledger only through `research/frontier_evidence.json`.

## Scope

This project measures and detects. It does not ship tooling for removing watermarks or evading
detection. The transformations in `src/watermark_lab/attacks/` exist to measure survival of
watermarks generated here with keys held here (`docs/runbooks/ROBUSTNESS.md`).

The two papers from this work are in the companion
[papers repository](https://github.com/dboles99/papers).

## Licence

Apache License 2.0. Copyright 2026 Daniel Boles. See [LICENSE](LICENSE).
