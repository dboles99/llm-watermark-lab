# Runbook: OCR Provenance


OCR is treated here as a scientific instrument and an attack channel, not as an
intake convenience. Text recovered from a scan is a **reconstruction**, never a
copy, and it can be wrong in ways that look exactly like watermark evidence or
exactly like its absence.

## The rule that governs everything below

An OCR reading may support at most **`inconclusive`** unless all of:

1. the engine profile is pinned and its fingerprint recorded;
2. mean character confidence clears the preregistered floor (default 0.80); and
3. a **frozen or validated** calibration manifest exists whose OCR profile
   matches this reading's engine, model bytes, and language.

No frozen OCR null exists for any profile or locale, so **today every scan is
`inconclusive`**. That is the intended behaviour. Clean-text thresholds do not
transfer to reconstructed text, and reusing one is how a false accusation gets
manufactured.

The gate takes a **manifest**, not an identifier. A bare identifier would let a
null measured on one instrument certify a reading produced by another.

## Install

```powershell
python -m pip install -e ".[ocr]"
```

RapidOCR is the default: pure Python, fully offline, and its ONNX models are
ordinary files inside the package, so they can be hashed and pinned exactly like
a tokenizer's assets. No system binary is required.

Tesseract is the second profile and is **not** a Python dependency. Install the
binary plus the language data you need — `chi_tra` for Taiwan Traditional
Chinese. Requesting `chi_tra` when only `chi_sim` is present is an error, not a
fallback: reading Taiwan text with a Simplified model is the contamination path
the programme forbids.

```powershell
watermark-lab ocr-engines
```

## Recover text from a scan

```powershell
watermark-lab ocr-intake scan.pdf --language en --dpi 300
watermark-lab ocr-intake page.png --show-text
```

Pages are hashed individually, because a multi-page scan is not one artifact:
pages get reordered, dropped and re-scanned independently. Confidence is
length-weighted, so a long badly-read page is not masked by a short well-read one.

The scanned-PDF path **ignores any embedded text layer** by design. A PDF can
carry both a scanned image and an OCR layer produced by unknown software at
unknown quality; preferring that layer would mean the recorded engine profile
describes an instrument that never ran. Use the ordinary text intake when an
authored text layer is what you want.

## Measure what survives a print-scan-OCR round trip

```powershell
watermark-lab ocr-survival sample.txt --severities "0.0,0.5,1.0"
```

OCR is the transformation that annihilates every container-bound provenance
channel — zero-width characters, whitespace encodings, Unicode variation
selectors, C2PA manifests, file metadata — while leaving the statistical channel
partly intact. Measured to date: container channels recover **0% at every
severity, including a clean render**; a statistical watermark retained **~88% of
its clean z-score** across scorable cells.

It is also an unusually clean attack. The evaluation protocol holds that an
attack destroying the text is not a successful attack, which is why paraphrase
must be quality-constrained. OCR is a faithful transcription attempt: it
preserves meaning by construction.

### Detection dropout is not signal loss

The recogniser's line recall is **not monotone in image quality** — a crisp
render can lose whole lines that a mildly blurred one recovers. Cells whose
recovered text falls below the coverage floor are flagged `detection_dropout`
and excluded from error statistics. Counting them as signal loss fabricates
fragility the degradation never caused. In the first pilot this excluded 4 of 8
cells, so it is not a corner case.

## Screen recognised Chinese output

```powershell
watermark-lab screen-zh recovered.txt
watermark-lab screen-zh corpus-directory/
```

Character-level Simplified-to-Traditional conversion changes glyphs and not
vocabulary, so converted text passes a script-level Hant check while still
reading 軟件 / 網絡 / 信息 where Taiwan writes 軟體 / 網路 / 資訊.

**Run this on recognised output, not only at corpus intake.** The bundled
`ch_PP-OCRv3` models are Simplified-trained, and in the locale pilot they
introduced Simplified forms into **6 of 12** Traditional documents — after
corpus screening had already passed. The instrument itself is a contamination
source.

A match is evidence of mainland usage, not proof of conversion. Absence of a
match is reported as `insufficient-evidence`, never as a pass.

## Measure a null so the gate can be discharged

```python
from watermark_lab.ocr import measure_ocr_null, build_calibration_manifest
```

Three constraints are enforced rather than assumed:

- **Independent units are clusters, not documents.** Interval arithmetic uses
  cluster counts; a cluster is a false positive if any of its documents exceeds
  the threshold. Adding documents to existing clusters does not help.
- **One stratum per manifest**, keyed on (engine × language × degradation). Each
  combination is a different instrument reading different text.
- **Attainability is reported.** With `n` clusters the tightest one-sided 95%
  bound after zero events is `1 - 0.05**(1/n)`. The first pilot measured 24
  clusters and bounds the FPR at **19%**, not 0.1%, and says so.

`planned_independent_negatives` is the study **design**, not the achieved count;
the schema floors it at 2,995 because a plan that was never large enough cannot
support its target however the run turns out. The achieved count and shortfall
live in the null-measurement artifact alongside.

The builder refuses to mark a calibration `frozen` or `validated` when the
measured clusters cannot evidence its target.

**Budget note.** The first pilot saw **33% document dropout on clean renders**,
so a real null needs roughly 1.5× the documents to reach a given cluster count.

## Locale is a first-class condition, not a parameter

```python
from watermark_lab.ocr import compare_locales, evaluate_locale_prediction
```

Measured: one misread character costs about **12 token errors in `en`** and about
**1 in `zh-Hant-TW`** — a 1206% difference. The driver is token granularity
relative to characters (0.169 tokens/char for `en` against 2.095 for
`zh-Hant-TW` under GPT-2 byte-level BPE), not whitespace.

A single OCR-robustness figure therefore does not transfer across locales.
Per-locale calibration is required, not merely prudent.

Two limitations to carry with that number: the character error rates are **not
matched** (RapidOCR read Traditional Chinese roughly 19× worse), and the GPT-2
tokenizer is not a reasonable production choice for `zh-Hant-TW`.

## Artifacts

| Path | Contents |
| --- | --- |
| `project/artifacts/ocr-survival/2026-08-30/` | container vs content survival comparison |
| `project/artifacts/ocr-locale/2026-08-31/` | locale comparison and contamination counts |
| `project/artifacts/ocr-calibration/2026-08-31/` | pilot null and calibration manifest |

Every one is marked non-promotable and says why in its own `promotion_status`
field.
