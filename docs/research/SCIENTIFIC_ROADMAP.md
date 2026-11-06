# Scientific Roadmap: Evidence-Calibrated Text Provenance Under Real Editing

**Decision version:** 2026-08-14  
**Evidence cutoff:** 2026-08-14  
**Canonical scholarly ledger:** `research/frontier_evidence.json`  
**Admission rule:** `docs/research/EVIDENCE_ADMISSION_POLICY.md`

## Recommended thesis

Build an **evidence-calibrated, adversarially accountable text-provenance system that survives real human editing and treats Taiwan Traditional Chinese (`zh-Hant-TW`) as a first-class scientific condition**.

The product should not answer “was this written by AI?” It should assemble method-specific evidence about compatible watermark signals, payloads, edit history, key epochs, localization, calibration uncertainty, and artifact provenance. Its scientifically defensible output is a lifecycle evidence record with `verified compatible watermark`, `not detected`, or `inconclusive` conclusions.

This thesis combines a necessary standards layer with research-grade “ultra” features. The first publishable result should be either:

1. **EditReceipt + counterfactual robustness certificates** — recommended for the fastest defensible paper and strongest product value; or
2. **PolyMark demixing in `zh-Hant-TW` mixed-source documents** — recommended if method novelty is more important than delivery risk.

The flagship longer-horizon methods paper should study a **robust-plus-fragile dual-channel watermark**: one channel persists through benign editing while a second authenticated channel intentionally breaks at semantically important changes.

These are candidate contributions, not claims of firstness. The bounded novelty statement is: **No directly equivalent end-to-end system was found in the admitted corpus as of 2026-08-14.** A systematic literature and patent search is required before stronger language.

## Recorded product decisions

All four principal tracks are product features, not mutually exclusive thesis alternatives:

1. EditReceipt and counterfactual certificates;
2. PolyMark demixing;
3. robust-plus-fragile semantic integrity; and
4. Translingua-TW handoff.

The Research workspace allows one or more modules to be selected while showing scientific maturity and dependencies. EditReceipt remains the recommended first implementation; the other three remain first-class features rather than being discarded.

Execution defaults to the API-first hybrid profile in `docs/research/EXECUTION_PROFILE.md`: deterministic/statistical work runs locally; pinned large-model and adaptive-attack work uses authorized APIs unless a local preflight proves the exact treatment fits. Human studies with compensated Taiwan-native editors are authorized subject to the study's ethics/consent requirements.

Payload identity is selectable between a pseudonymous project/workflow reference (default) and an authorized person/account reference. Both carry opaque references only; raw names, emails, or account identifiers never appear in watermark bits. Person/account mode requires an explicit purpose, authority/consent, retention, access, selective-disclosure, and revocation policy.

Proof-carrying verification follows `TEE simulator → real remote TEE → TEE validation → zero-knowledge specification → zero-knowledge prototype`. The local workstation is a verifier/development host, not the production enclave.

## What the frontier already covers

The admitted corpus already includes:

- keyed logit-bias, distortion-free, unbiased, and cryptographic watermark constructions [KGW_2023, DISTORTION_FREE_2024, UNBIASED_2024, UNDETECTABLE_2024, SYNTHID_2024];
- semantic/paraphrase-robust, dual, multi-channel, and multi-bit designs [SEMSTAMP_2024, DUWAK_2024, MCMARK_2025, MIRRORMARK_2026];
- translation and cross-lingual evaluation [X_SIR_2024, CROSS_LINGUAL_2025];
- adaptive removal, watermark stealing, spoofing, and impossibility boundaries [ADAPTIVE_ATTACKS_2025, WATERMARK_STEALING_2024, SPOOF_DISCOVERY_2025, WATERMARKS_SAND_2024];
- long-document localization, mixed-source detection, collisions, and broad benchmarks [WATERSEEKER_2025, MIXED_SEGMENTS_2025, COLLISION_2025, WATERBENCH_2024, WATERPARK_2025]; and
- statistical detection, optional stopping, and tokenizer inconsistency [STAT_FRAMEWORK_2024, HUMAN_EDITS_2025, ANYTIME_VALID_2026, TOKENIZATION_2025].

Accordingly, generic “paraphrase-robust watermark,” “multi-bit watermark,” “ensemble detector,” “public verification,” “localization,” or “anytime-valid detection” ideas are too crowded to be the thesis on their own.

The targeted admitted-source search found no paper that explicitly validates a text-watermark system for Taiwan `zh-Hant-TW` as a distinct locale. This is a search-bounded result, not proof that none exists.

## Current scientific blockers in this repository

The existing vertical slices are useful engineering scaffolds, but scientific promotion is blocked by:

1. `detect.py` hashes whitespace-delimited words rather than tokens from the declared model tokenizer.
2. `kgw.py` remains a Python-RNG project mechanics profile. A separate exact
   Windows AMD64 CPU author profile now passes 10,000 direct green-list and
   1,000 direct token-score differentials with zero mismatches. Raw-text and
   generation parity remain open, and other PyTorch/device/platform builds need
   separate profiles.
3. Repeated n-gram masking is now implemented as a prototype full
   `(context, token)` first-occurrence rule. It is restricted to `simple_1` with
   `context_width=1`, preserves full-position and unique-event diagnostics, and
   fails to `inconclusive` because exact author-code parity and a separate
   repeated-context calibration remain open.
4. `kgw_generator.py` emits synthetic token-ID text and is not tied to a pinned real model/tokenizer pair.
5. Attack insertion and paraphrase behavior are English-oriented stubs, not linguistically valid `zh-Hant-TW` transformations.
6. No frozen calibration corpus exists; current outputs do not contain an estimated FPR.
7. The old 200-example-per-cell design has 0.5% empirical resolution and cannot validate a 0.1% FPR claim.
8. The Cartesian calibration grid is simultaneously expensive overall and underpowered within individual cells.

The current T03/T04 evidence therefore remains **vertical slice complete; promotion pending re-review**. New results must supersede rather than rewrite frozen historical artifacts.

## Product feature ladder

### Standard scientific core

These features are mandatory before novel claims:

| Capability | Scientific gate |
| --- | --- |
| Exact method adapters | Author paper and code pinned; equation-to-code map; exact vectors and randomized parity tests |
| Tokenizer-faithful generation/detection | Identical normalization, tokenizer, vocabulary, context, seed/PRF, and repeated-context policy |
| Calibrated decisions | Independent calibration partition, exact intervals, target FPR, family definition, and `inconclusive` policy |
| Real model generation | Pinned open model/tokenizer plus controlled decoding; API adapters clearly separated |
| Multilingual validity | Separate `en` and Taiwan-origin `zh-Hant-TW` corpora, thresholds, quality review, and worst-group reporting |
| Robustness lab | Deterministic fixed attacks plus adaptive attacks with budgets, quality constraints, and semantic/factual checks |
| Mixed-source localization | Span IoU, boundary error, coverage error, scan correction, and document-level uncertainty |
| Artifact provenance | Immutable hashes, source/version lineage, key identifiers rather than keys, supersession, and clean-room reruns |
| Responsible reporting | No provider attribution without an authorized adapter; never report “percent AI-written” |

### Ultra research features

#### A. EditReceipt DAG + counterfactual robustness certificate — recommended first

Represent generation, edits, translation, detection, calibration, and review as a signed content-addressed DAG. For each transformation, compute a counterfactual certificate showing which allowed edit classes and severity region would retain detection at the preregistered threshold, rather than merely saying that one attacked sample passed.

**Novel hypothesis:** lifecycle-aware certificates will better predict post-edit verification and reduce unjustified binary conclusions compared with a terminal detector score.

**Minimum prototype:** local signed receipts, deterministic edit provenance, empirical survival surface, uncertainty band, and explicit out-of-distribution refusal.

**Main risk:** the certificate can only cover a declared transformation family; it must never be presented as universal robustness.

#### B. PolyMark: multi-provider, multi-key, multi-epoch demixing

Infer zero, one, or several compatible watermark processes within a document and assign localized spans to provider adapters, tenant pseudonyms, or key epochs. Model overlap and collision rather than taking the maximum detector score.

**Novel hypothesis:** a structured sparse or Bayesian demixing model can recover ordered lineage and overlapping marks with controlled family-wise error better than independent scanning.

**Key metrics:** provider/key count error, span IoU, ordered-lineage accuracy, payload recovery, collision confusion, family-wise FPR, and abstention quality.

**Main risk:** identifiability may fail when schemes or keys induce near-equivalent statistics. The project must characterize impossibility regions, not hide them.

#### C. Robust-plus-fragile semantic integrity watermark — flagship/high risk

Embed two authenticated channels:

- a robust provenance channel intended to survive benign copyediting, formatting, and approved translation; and
- a fragile semantic channel tied to protected claims, numbers, citations, quotations, or locale-sensitive terminology and intended to fail or localize when those invariants change.

This is not merely two conventional watermarks: the research question is whether transformation-conditioned channel behavior can distinguish benign editorial drift from consequential semantic tampering.

**Novel hypothesis:** joint robust/fragile evidence will detect material tampering at a fixed false-alarm rate while preserving provenance through benign edits better than either channel alone.

**Main risks:** semantic canonicalization is attackable; fragile signals may overreact to acceptable edits; dual-channel capacity may damage quality. Start with constrained factual templates before open-domain prose.

#### D. Translingua-TW: transformation-covariant payloads and signed translation handoff

Bind a payload to semantic units and a declared translation operation. The source verifier produces an authenticated handoff; the target-language verifier tests both provenance continuity and Taiwan terminology/factual constraints.

**Novel hypothesis:** a signed source-to-target handoff plus locale-aware payload mapping will outperform direct token-level survival under English ↔ `zh-Hant-TW` translation.

**Main risk:** this may become a provenance protocol rather than a purely statistical watermark. That is acceptable if evaluated honestly.

#### E. Proof-carrying private verification and EpochSeal

Keep detection keys private while returning a verifiable statement about method, key epoch, detector code hash, calibration version, threshold, result, and revocation state. Implement a trusted-execution proof first; consider zero knowledge only after the statement and leakage model are stable.

**Novel hypothesis:** public accountability can be achieved without publishing a reusable detection oracle or long-lived key.

**Metrics:** proof size/latency, verifier cost, false acceptance, side-channel/oracle leakage, revocation correctness, and compromise blast radius.

**Main risk:** public detection already exists in the literature [PUBLIC_DETECTION_2024]. The contribution must be operational accountability, key lifecycle, calibrated statement semantics, and leakage resistance—not “public verification” alone.

#### F. Federated calibration certificates

Allow independent institutions to contribute privacy-preserving aggregate calibration evidence across languages, domains, and author groups without pooling raw text. Release signed threshold certificates that state the population, time window, uncertainty, and privacy mechanism.

**Novel hypothesis:** hierarchical federated calibration can improve worst-group FPR evidence while preserving source privacy and resisting malicious contributors.

**Main risk:** secure aggregation [SECURE_AGGREGATION_2017] solves only part of the problem; representativeness, poisoning, cluster dependence, and extremely rare-event estimation remain research problems.

#### G. MosaicTrace collusion and mixed-source lineage

Test collusive recombination of outputs carrying different payloads/keys, including sentence mosaics, alternating translations, quotation insertion, and human rewrite. Infer contributor set and localized lineage while controlling false discoveries.

**Novel hypothesis:** code designs coupled to semantic segment boundaries can expose collusive mosaics more reliably than document-level payload decoding.

#### H. TwinAnchor container + text binding

Bind a statistical text watermark to a signed content manifest and record round-trip survival through DOCX, PDF, clipboard, Markdown, normalization, and export. The statistical signal becomes a soft binding when the container metadata disappears.

This uses the [C2PA 2.4 specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html) as a standards reference, not scholarly literature. C2PA already defines soft-binding/invisible-watermark fields and a text-oriented Unicode variation-selector mechanism, so “C2PA for text” is not itself a novelty claim.

> **Correction, 2026-08-30.** This paragraph previously described the text/Unicode
> variation-selector mechanism as “under review”. That is factually wrong and has been
> removed. Text embedding was **normatively standardized in C2PA 2.3 (published 2026-01-08)**
> and is current in 2.4 as Appendix A.7 (HTML), A.8 (unstructured text via Unicode Variation
> Selectors, container `C2PATextManifestWrapper`) and A.9 (structured ASCII armour), with
> §9.2.4 covering hashing of unstructured text assets. A four-language MIT reference
> implementation exists (`encypherai/c2pa-text`). Further, the official
> `c2pa-org/softbinding-algorithm-list` registry already carries **six TEXT-domain
> soft-binding algorithms**, so binding a text watermark to a signed manifest is existing
> registered practice rather than a contribution. The surviving Track H claim is narrowed to
> the **measured round-trip survival surface for a statistical, logit-biased watermark,
> including print→scan→OCR**.

#### I. Chameleon precommitted portfolio

Use a preregistered, rotating portfolio of compatible watermark schedules or detectors to reduce the value of attack adaptation while preserving exact ex-ante error accounting.

**Novel hypothesis:** hidden but precommitted portfolio selection raises attack query cost without invalidating calibration.

**Main risk:** uncontrolled moving targets destroy reproducibility. Every portfolio, selection rule, epoch, and correction must be frozen before evaluation.

## Recommended program and decision gates

### Phase 0 — Evidence and reproducibility foundation (2–3 weeks)

1. Validate every cited record against the evidence schema and perform PDF-header availability checks.
2. Download admitted PDFs into a content-addressed cache; record SHA-256, version, license, and retrieval timestamp.
3. Pin author repositories and commits separately from papers.
4. Repair the bibliography notebook environment with a compatible Polars runtime, then extend it to validate the strict ledger.
5. Preregister the author-code parity and calibration study.

**Exit gate:** no scientific claim depends on a source lacking an admitted index record and downloadable PDF; all frozen inputs are hashed.

### Phase 1 — Exact KGW reproduction (3–5 weeks)

1. Implement tokenizer-faithful detection and generation.
2. Match the author-code PRF/seed, green-list construction, repeated-context handling, normalization, and token indexing.
3. Create fixed golden vectors and at least 10,000 randomized-context parity cases.
4. Run property tests for null behavior, key separation, seed determinism, and generation/detection configuration equality.
5. Generate real text with at least one pinned open model/tokenizer.

**Exit gate:** zero unexplained parity mismatches; all deviations documented; independent blind rerun passes.

### Phase 2 — `zh-Hant-TW` corpus and calibration (5–8 weeks)

1. Build licensed Taiwan-origin corpora spanning news/editorial, government/legal, academic/technical, education, medicine, and conversational domains.
2. Record source region, author/document cluster, license, domain, date, length, native/review status, and contamination checks.
3. Freeze development, calibration, attack-tuning, and final locked partitions by author/document.
4. Use exact binomial intervals and hierarchical/worst-group analysis. Define the family across languages, detectors, keys, lengths, and scans before choosing thresholds.
5. Review a sample with native Taiwan editors for terminology, fluency, meaning, factual/numeric/citation preservation, and Simplified-character contamination.

**Exit gate:** claimed FPR bounds are statistically attainable; `en` and `zh-Hant-TW` are separately calibrated; no hidden `zh-CN` fallback or converted corpus is used.

### Phase 3 — Real editing and adaptive security (5–7 weeks)

1. Collect blinded human copyedits at light, substantive, and translation-review levels.
2. Add fixed attacks, optimization-based adaptive attacks, watermark stealing, spoofing, mixed-source insertion, key collisions, and bounded collusion.
3. Enforce meaning, factuality, numeric/citation integrity, and Taiwan acceptability constraints; an attack is not successful merely because it destroys the signal.
4. Plot detection-quality-query-cost Pareto surfaces and report abstentions.

**Exit gate:** results include calibrated adaptive security, not only canned perturbations; the final set remains locked until policies are frozen.

### Phase 4A — EditReceipt feature and publication track (4–6 weeks)

Build the receipt DAG, empirical counterfactual certificate, and UI evidence view. Compare terminal-score reporting against lifecycle evidence for calibration, interpretability, and post-edit decision quality.

### Phase 4B — PolyMark feature and publication track (6–10 weeks)

Develop a multi-process simulator, demixing model, scan-aware null, collision identifiability study, and mixed-source benchmark in `en` and `zh-Hant-TW`.

### Phase 5 — Robust-plus-fragile and Translingua-TW features (10–16 weeks)

Start the dual-channel method with constrained claims/numbers/citations. Define benign and material edit equivalence classes before algorithm design. Compare robust-only, fragile-only, independent dual, and jointly optimized dual channels. Red-team canonicalization and semantic-oracle attacks. In parallel only after the shared T06 locale gate, build the signed source/translation/`zh-Hant-TW` child handoff and compare it with direct cross-lingual survival.

### Phase 6 — Cryptographic and federated extensions

Add key epochs/revocation and a real remote TEE only after the underlying statistical statement is stable. Freeze the attested statement from the working TEE before designing the zero-knowledge circuit/protocol. Federated calibration follows later. Cryptography cannot repair an invalid estimand or a miscalibrated detector.

## Confirmatory design

### Primary estimands

- TPR at fixed 0.1%, 1%, and 5% FPR with exact intervals.
- Worst-group FPR and TPR by locale, domain, length, author group, tokenizer, and edit condition.
- Post-edit signal survival subject to preregistered semantic, factual, numerical, citation, and locale-quality constraints.
- Span IoU, boundary error, watermarked-coverage error, and process-count error for mixed documents.
- Payload bit error rate and complete-payload recovery.
- Forgery/spoofing advantage at query and compute budgets.

### Sample-size floors

- Zero false positives require at least 2,995 independent negatives for a one-sided 95% upper bound at or below 0.1%.
- Estimating 0.1% FPR to approximately ±20% relative precision requires roughly 95,944 independent negatives per claimed stratum under a simple normal approximation.
- A 90% TPR with approximate ±3 percentage-point precision requires about 385 independent positives; plan for at least 500 per confirmatory cell.
- A balanced preference estimate with approximate ±5 percentage-point precision requires about 385 prompts; plan for 400 prompts, three ratings each, and mixed-effects analysis.

These figures are planning approximations, not permission to ignore dependence. The independent unit is the author/document cluster unless the preregistered model justifies otherwise.

### Mandatory ablations

- correct tokenizer vs whitespace tokenization;
- author-code PRF/permutation vs substitute RNG;
- repeated-context masking vs naive counting;
- theoretical null vs exact/permutation null vs empirical calibration;
- single universal threshold vs language/domain hierarchical thresholds;
- nonadaptive vs adaptive attacks at matched quality;
- robust-only vs fragile-only vs independent dual vs jointly optimized dual;
- document-only vs localized vs demixed multi-process inference;
- no receipt vs unsigned receipt vs signed EditReceipt;
- direct cross-lingual detection vs signed translation handoff;
- static key vs key epochs/revocation;
- clean evaluation vs serving-stack tokenization/normalization round trips.

### Human study

Use blinded randomized method labels. Recruit Taiwan-native editors for the `zh-Hant-TW` arm and domain-qualified reviewers where factual correctness matters. Measure preference, adequacy, fluency, Taiwan terminology, detectable stylistic artifacts, factual/numeric/citation changes, and whether an edit is benign under the preregistered transformation class. Model prompt, document, author/editor, and rater effects.

### Reproducibility package

Freeze and hash:

- admitted PDFs, supplementary materials, and author-code commits;
- model/tokenizer identifiers and hashes or exact provider/API versions;
- generation configurations, prompts, splits, seeds, and PRF specifications;
- corpus manifests, licenses, exclusion logs, and cluster identifiers;
- attack code, budgets, semantic/factual constraints, and stopping rules;
- analysis code, multiplicity families, preregistration, environment lock, and expected figures/tables.

A blind review pass receives the frozen package without producer reasoning. A second independent pass recomputes key vectors and statistical claims. The root orchestrator reconciles evidence; model agreement is never treated as validation.

## Scientific claim language

Allowed:

- “No directly equivalent system was found in the admitted corpus as of 2026-08-14.”
- “The result applies to the declared models, tokenizers, corpora, edit classes, thresholds, and attack budgets.”
- “Compatible watermark evidence was detected at the stated calibrated operating point.”

Disallowed without additional evidence:

- “first,” “unprecedented,” “unbreakable,” or “proves AI authorship”;
- provider attribution from resemblance;
- universal robustness from a finite attack suite;
- a 0.1% FPR claim from a 200-sample cell;
- paragraph-level sample counts presented as independent when passages share authors or documents.

## Literature notation

Bracketed identifiers in this document resolve to exact citations, scholarly index records, and direct PDFs in `research/frontier_evidence.json`. Local Word papers are project hypotheses and design context; they are not counted as independent external validation. C2PA is an official technical standard and is kept outside the scholarly evidence ledger.
