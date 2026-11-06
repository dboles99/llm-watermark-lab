# Exact Calibration Statistics

**Status:** Implemented statistical primitive; no calibration corpus or promoted empirical claim  
**Scope:** Independent binary evaluation units only

## What is implemented

`watermark_lab.statistics` provides:

- an exact one-sided Clopper-Pearson upper confidence bound for a binomial event rate;
- a boundary-safe zero-event calculation;
- an integer search for the smallest sample size meeting a requested upper bound while the observed event count is held fixed;
- an equal-tailed, explicitly labeled two-sided Clopper-Pearson interval; and
- a TPR wrapper whose returned estimand is `true-positive rate (TPR)`.

For false-positive calibration, `events` is the number of false positives and
`total` is the number of prespecified independent negative evaluation units.
For TPR, the corresponding counts are true positives and independent positive
evaluation units.

## Exact zero-event target

For zero observed events in `n` independent Bernoulli units, the one-sided
confidence-`C` upper bound is:

```text
upper = 1 - (1 - C) ** (1 / n)
```

The implementation evaluates this as `-expm1(log1p(-C) / n)` to avoid loss of
precision near zero. At one-sided 95% confidence, the smallest sample whose
zero-event upper bound is at most 0.1% is exactly **2,995** independent negative
units:

```python
from watermark_lab.statistics import (
    clopper_pearson_upper_bound,
    minimum_zero_event_sample_size,
)

n = minimum_zero_event_sample_size(0.001, confidence_level=0.95)
assert n == 2_995
assert clopper_pearson_upper_bound(0, n, 0.95) <= 0.001
assert clopper_pearson_upper_bound(0, n - 1, 0.95) > 0.001
```

This is a confidence-bound precision statement after observing zero events. It
is not a prospective power calculation and does not guarantee that zero false
positives will occur.

Frozen manifests first validate against
`schemas/calibration-manifest.schema.json`, then call
`assert_calibration_manifest_semantics`. The second step dynamically computes
the required zero-event sample size from the manifest's own target and
confidence level. This matters because the 2,995 schema floor is sufficient for
95% confidence at 0.1% FPR, but is not sufficient for a stricter confidence or
FPR target.

## Independent-unit assumption

Every interval and sample-size result assumes prespecified, independent
Bernoulli units with one binary result per unit. The code does **not** establish
that assumption and does not correct:

- multiple completions from the same prompt, author, source document, model, or key;
- prompt-, topic-, domain-, model-, language-, or author-level clustering;
- repeated measurements or overlapping text windows;
- adaptive sampling, optional stopping, or threshold selection on evaluation data;
- distribution shift; or
- post-hoc pooling of distinct calibration families.

When records are clustered, the scientific design must define the independent
unit and use a cluster-aware or hierarchical analysis that matches that design.
This module deliberately makes no claim to solve dependence. Sample allocation
must follow the preregistered estimands, error families, and scientifically
meaningful strata; this utility does not recommend mechanically populating a
Cartesian product of every available factor.

## Interpretation rules

- Report an FPR bound as **one-sided** and include the confidence level, event
  count, independent-unit count, calibration family, and corpus version.
- Report TPR with the provided **two-sided equal-tailed** interval, including
  numerator and denominator.
- Do not substitute the two-sided interval's upper endpoint for the one-sided
  FPR upper bound; their tail allocations differ.
- Do not promote a detector because an unattainable or undersized calibration
  family happened to record zero events.
- Keep exploratory threshold selection separate from the frozen evaluation
  sample.

## Evidence provenance

The scientific context for formal watermark tests is the admitted
`STAT_FRAMEWORK_2024` record, *A Statistical Framework of Watermarks for Large
Language Models: Pivot, Detection Efficiency and Optimal Rules*:
[downloadable PDF](https://arxiv.org/pdf/2404.01245) and
[OpenAlex record](https://openalex.org/W4407557405). Its allowed use in
`research/frontier_evidence.json` is `statistical-foundation` and
`optimal-test-design`.

The KGW detector context is the admitted `KGW_2023` record, *A Watermark for
Large Language Models*: [downloadable PMLR PDF](https://proceedings.mlr.press/v202/kirchenbauer23a/kirchenbauer23a.pdf)
and [OpenAlex record](https://openalex.org/W4318149317). Its admitted uses include
`detector-statistic`.

The numerical implementation uses SciPy's beta-distribution percent-point
function: [official SciPy API documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html).
That page is implementation evidence, not scholarly literature, and is not an
entry in the research evidence ledger.

No external empirical result is introduced by this module. A real calibration
claim still requires an authorized corpus, frozen detector/configuration,
prespecified independent units, recorded exclusions, and a review gate.
