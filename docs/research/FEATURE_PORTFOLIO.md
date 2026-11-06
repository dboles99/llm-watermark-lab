# Selectable Frontier Feature Portfolio

**Decision date:** 2026-08-14  
**Selection schema:** `schemas/research-feature-selection.schema.json`  
**Shared scientific prerequisite:** T06

## Product contract

The Research workspace offers all four modules. Users may select one or more, an execution target, a payload identity mode, and an available proof backend. A selection records experimental intent; it does not start a run or imply that a feature has passed its scientific, security, or deployment gate.

Every selection and resulting run records:

- selected feature IDs and exact plugin versions;
- locale and transformation profile;
- execution target/profile and actual local/API routing;
- model/tokenizer/API versions and configuration hashes;
- payload identity mode and opaque authority reference where required;
- proof backend and attestation/proof statement version;
- calibration, key epoch, corpus/split, threat model, and preregistration identifiers; and
- separate artifact, check, promotion, scientific, and deployment states.

## Feature plugins

| Feature ID | Role | Required inputs | Primary outputs | First blocking dependency |
| --- | --- | --- | --- | --- |
| `edit-receipt` | Lifecycle evidence wrapper | content/edit DAG, compatible detector, transformation registry | signed receipt, counterfactual survival region, uncertainty, refusal region | calibrated detector and validated transformations |
| `polymark-demixing` | Multi-process analyzer | candidate adapters/keys/epochs, mixed document, scan family | process-count posterior/interval, localized spans, collision and lineage hypotheses | simultaneous scan-aware null and collision corpus |
| `robust-fragile` | Generation and integrity method | protected semantic units, robust/fragile keys, allowed-edit ontology | dual-channel text, provenance verdict, localized integrity breaks | author-code baselines plus human-validated edit ontology |
| `translingua-tw` | Authorized transformation workflow | verified source receipt, translation action, Taiwan terminology/invariants | `zh-Hant-TW` child artifact, signed parent/child handoff, locale-quality evidence | Taiwan corpus, translator protocol, native review |

## Composition

- EditReceipt can wrap every other module and should be the default evidence container.
- PolyMark may analyze documents containing robust-plus-fragile or Translingua outputs, but it may not infer provider identity from an unverified adapter.
- Robust-plus-fragile is a watermarking method, while Translingua-TW is an authorized transformation/handoff workflow. Selecting both means the translation protocol must declare which channel is preserved, reissued, or intentionally invalidated.
- Multi-feature runs retain per-feature gates. One passing module never promotes another.
- Incompatible key, tokenizer, normalization, calibration, or locale states cause `inconclusive` or fail-closed behavior, never silent coercion.

## Execution choices

- `api-first-hybrid` is the default. Deterministic/statistical work is local; primary large-model and confirmatory adaptive work is remote unless exact local feasibility is measured.
- `local-only` is accepted only when every pinned treatment passes memory/model/tokenizer preflight. It must refuse rather than substitute.
- `api-only` is available for reproducible remote jobs, but deterministic scores are recomputed locally where possible as an independent check.

See `docs/research/EXECUTION_PROFILE.md` for the sanitized workstation capacity and routing rules.

## Payload identity choices

### Pseudonymous project/workflow reference — default

Bind only an opaque project/workflow reference such as `ref_*`. Resolution lives in the governed external registry. This minimizes linkability and is appropriate for research, ordinary provenance, and most publication artifacts.

### Authorized person/account reference

Bind an opaque `ref_*` to a separate `auth_*` authority record. Before execution, require a declared purpose, legal/contractual authority or consent, access policy, retention/expiry, selective-disclosure rules, subject-access/correction process, incident response, and revocation behavior.

Neither mode embeds names, emails, employee or account numbers, or raw account identifiers. An identity reference is an auditable pointer, not proof that the referenced person authored every word.

## Proof sequence

1. `tee-simulator`: developer-only statement and verifier integration; always marked simulated.
2. `remote-tee`: next implementation; real platform attestation, pinned detector/config/calibration, nonce/replay protection, key-epoch isolation, rate limiting, and revocation.
3. `zero-knowledge`: available only after the TEE statement and leakage model are validated; it must reproduce the same public semantics or explicitly version the change.

Zero knowledge is not a rewrite of the detector. It is a later proof backend for a stable, scientifically valid statement.

## Product wording

Use `planned`, `development-only`, `experimental`, `reproduced`, or `verified-compatible` exactly as supported by evidence. The selector must always show that these modules are currently planned and T06-dependent.
