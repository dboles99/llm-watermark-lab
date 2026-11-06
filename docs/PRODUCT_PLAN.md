# Product Plan

## Recommended product position

Build the application and deterministic watermark engine first. Anthropic integration is a later optional provider adapter. During development, a local Docker mock supplies Claude-shaped API responses, known synthetic watermarks, clean controls, and failure cases.

Build a watermark laboratory with five clearly separated operating modes:

1. **Verify:** run a keyed detector for a known compatible method.
2. **Create:** generate model output while applying a project-owned watermark.
3. **Analyze:** inspect token statistics, entropy, repeated contexts, possible periodicity, and passage-level score changes without claiming provider identity.
4. **Stress test:** apply controlled edits, paraphrases, translations, deletions, insertions, and mixed-source attacks, then measure survival.
5. **Provider-neutral forensic ensemble:** ask configured non-Anthropic models to generate hypotheses or classification features, then evaluate those hypotheses against held-out controls. This mode remains keyless and cannot produce an official Claude verification.
6. **Unknown-pattern forensics:** scan text and document containers for unexplained Unicode, whitespace, structural, metadata, periodic, lexical, and token-distribution anomalies without assuming a platform or watermark family.

This separation prevents a keyless forensic score from being mistaken for verified provenance.

## MVP

- Windows desktop GUI using PySide6.
- Intake by pasted text, single file, or folder.
- TXT, Markdown, DOCX, and text-based PDF extraction.
- Run builder that selects method, tokenizer, model endpoint, key identifier, threshold, and transformations.
- KGW creation and detection first.
- MCmark-inspired and MirrorMark-inspired adapters behind experimental flags until reproduced against their papers.
- Local PostgreSQL as the authoritative database, with SQLite allowed only for portable demo mode.
- JSON/CSV/HTML/PDF run reports.
- Jupyter benchmark notebook generated from immutable run records.
- Ollama-compatible endpoint configuration for Kimi 2.7 and GLM 5.2 development and test generation.
- Azure OpenAI and OpenAI-compatible adapters for GPT-5.6 Luna, Kimi 2.7 Code, GLM 5.2, and future non-Anthropic models.
- Docker mock server with both OpenAI-compatible and Anthropic-shaped test endpoints.
- Research workspace with selectable EditReceipt, PolyMark demixing, robust-plus-fragile integrity, and Translingua-TW modules. Each displays its maturity and scientific dependencies.
- API-first hybrid execution choice with local-only and API-only overrides that fail rather than silently changing the pinned treatment.
- Payload identity choice between pseudonymous project/workflow references and governed person/account references; never embed raw identity data.
- Proof roadmap that implements a real remote TEE after a local simulator, then adds zero knowledge after the attested statement is stable.

## Explicit non-goals for the MVP

- Claiming to detect Claude, ChatGPT, Gemini, or another provider without its compatible verifier.
- Covertly watermarking third-party prose without policy authorization and disclosure.
- Automatically accusing a writer of misconduct.
- Attempting to extract or steal proprietary provider keys.
- Treating generic style classification as watermark verification.

## Method options

| Adapter | MVP role | Strength | Main limitation |
| --- | --- | --- | --- |
| KGW-style | Baseline creator and detector | Interpretable z-score and extensive literature | Changes token probabilities and can be attacked |
| MCmark-inspired | Experimental creator and detector | Multi-channel, designed to reduce bias | Exact reproduction and calibration are complex |
| MirrorMark-inspired | Experimental multi-bit creator and detector | Payload and context-anchored scheduling | Current work is preprint-level and payload robustness is limited |

“Inspired” is mandatory in UI and reports unless the implementation passes an independent reproduction protocol and licensing review.

## Phases

### Phase 0: decisions and paper-to-code specifications

- Confirm repository name, license, deployment boundary, database mode, and allowed model endpoints.
- Translate each paper into a testable algorithm specification.
- Create golden vectors for token selection, seeding, encoding, detection, and p-values.

### Phase 0A: mock provider laboratory

- Run a local Docker server that imitates request and response shapes without using Anthropic code, credentials, or services.
- Produce deterministic clean, KGW, MCmark-inspired, and MirrorMark-inspired fixtures.
- Inject timeouts, malformed responses, token-budget errors, and model-version drift for integration testing.
- Keep the simulated-provider label visible in every response and artifact.

### Phase 1: core run engine and KGW baseline

- Immutable run manifests and content hashing.
- Tokenizer abstraction, key vault interface, detector API, watermark API.
- KGW implementation and synthetic benchmark suite.
- CLI and basic GUI intake.

### Phase 2: robustness laboratory

- Controlled edit, deletion, insertion, paraphrase, translation, and mixing attacks.
- Passage localization and score-versus-length plots.
- False-positive calibration on human, edited, multilingual, and domain corpora.

### Phase 3: MCmark-inspired adapter

- Reproduce the paper’s channel construction and unbiasedness tests.
- Compare output quality, detection, and robustness against KGW using matched conditions.

### Phase 4: MirrorMark-inspired adapter

- Implement mirrored pseudorandom mapping, scheduler, multi-bit payload, and decoding.
- Separate watermark presence accuracy from payload recovery accuracy.

### Phase 5: governance and pilot

- Disclosure templates, key rotation, role permissions, audit export, retention, and incident response.
- Pilot only on synthetic or project-owned demonstration content.

## Acceptance gates

- Every score reports method, key compatibility, text length, threshold, estimated false-positive rate, confidence interval, and calibration corpus.
- Cross-method comparisons use the same model, prompts, tokenizer family, length bins, attacks, and false-positive threshold.
- A detector result cannot populate a provider name unless the adapter is provider-authorized.
- Original inputs and generated outputs are content-hashed and never silently overwritten.
- Keys are never stored in notebooks, logs, exported reports, or Git.

## Decisions recorded

- Use `project/artifacts/` as the canonical evidence root.
- Offer supersede-and-preserve and pre-freeze amend-in-place artifact history; frozen evidence is supersede-only.
- Prioritize GUI, storage, and product surfaces before scientific-validity promotion work.
- Authorize public and synthetic English and Taiwan Traditional Chinese calibration data.
- Use `zh-Hant-TW` as the canonical Taiwan locale and accept `zh-TW` as an alias. Never fall back to `zh-CN` or another Simplified Chinese locale.
- Offer all four frontier tracks as selectable features, with EditReceipt implemented first but no feature discarded.
- Use the workstation profile for API-first hybrid execution. Local work owns deterministic/statistical tasks and measured small-model pilots; authorized APIs own primary large-model and confirmatory attack work.
- Authorize compensated Taiwan-native human review, subject to study ethics, consent, and data-governance approval.
- Offer pseudonymous project/workflow and governed person/account payload references. Both use opaque identifiers; raw identity is never embedded.
- Implement proof-carrying verification with a local TEE simulator and a real remote TEE first; design zero knowledge after the TEE statement and leakage model are stable.

Still requiring later product/governance decisions are third-party document admission, exact payload schemas and retention periods, production provider authorization, API spending ceilings, human-study consent/ethics documentation, and the remote confidential-compute platform.
