# Hybrid Scientific Execution Profile

**Decision date:** 2026-08-14  
**Machine profile:** `config/execution_profiles/workstation-api-first.toml`

## Sanitized workstation capacity

The supplied system report and live capability checks establish:

- two Intel Xeon E5-2696 v2 processors;
- 24 physical cores and 48 logical processors total;
- 64 GiB system memory;
- NVIDIA GeForce RTX 2060 SUPER with 8 GiB VRAM and CUDA capability 7.5; and
- Windows 10 Pro 64-bit.

Serial numbers, network configuration, user/account information, installed-software inventory, and other non-compute identifiers from the report are deliberately excluded.

## Execution policy

The default is **API-first hybrid**, not “run everything remotely” and not an unmeasured local fallback.

Run locally:

- deterministic detectors and exact statistical tests;
- tokenization, normalization, hashing, artifact verification, and author-code parity;
- corpus ingestion, deduplication, partitioning, and most CPU-parallel simulation;
- notebook-small fixtures and reproducibility checks;
- TEE client/verifier development and a clearly labeled attestation simulator; and
- optional quantized small-model pilots after RAM/VRAM preflight.

Run through authorized APIs or remote jobs:

- primary large-model generation;
- confirmatory adaptive-attack sweeps;
- model-diverse quality/robustness evaluation;
- workloads that cannot meet their pinned model, tokenizer, context, or batch contract locally; and
- real confidential-compute/TEE attestation.

The scheduler must fail explicitly when a workload requires an API and no authorized endpoint exists. It must never silently replace a pinned remote model with a smaller local model, because that would change the scientific treatment.

## Practical local limits

The CPUs and memory are well suited to parallel detection and rare-event simulation, although large studies still require checkpointed batches. The 8 GiB GPU can support selected quantized small-model inference with constrained context and batch size. It should not be the default for large-model reproduction, fine-tuning, or the confirmatory adaptive-attack program.

Record actual wall time, CPU worker count, peak RAM, GPU model/driver, peak VRAM, model quantization, context, batch size, and fallback/remote decision in every experiment manifest.

## API and secret boundary

Only provider profiles, environment-variable names, and deployment identifiers belong in configuration. API keys and TEE credentials stay in managed identity, Windows Credential Manager, or an approved secret store. They never enter TOML, notebooks, prompts, artifacts, logs, or chat transcripts.

## TEE-to-zero-knowledge sequence

1. **Local simulator:** stabilize the attested statement, receipt schema, nonce/replay handling, key-epoch reference, calibration commitment, and verifier API. Its outputs say `simulated: true` and are never promotion evidence.
2. **Remote TEE pilot:** run the pinned detector in real confidential compute, verify platform attestation, bind the code/config/calibration/text commitments, rate-limit the oracle, and record revocation behavior.
3. **TEE scientific evaluation:** measure correctness, latency, leakage surface, denial-of-service behavior, replay resistance, proof portability, and historical verification across key/calibration epochs.
4. **Zero-knowledge specification:** freeze the exact statement learned from the TEE pilot, choose proof-friendly detector arithmetic, and define public/private inputs and leakage.
5. **Zero-knowledge prototype and comparison:** compare proof size, prover memory/time, verification time, supported detector complexity, and security assumptions against the TEE implementation.

Cryptographic verification attests execution of a statement; it does not establish that the detector's statistical claim is scientifically valid. T06 calibration remains a prerequisite.

The active implementation is a local Docker simulator. The recommended first future remote backend is Azure Confidential Containers on ACI with Microsoft Azure Attestation and attestation-gated key release, but cloud work is deferred on cost grounds. See `docs/architecture/TEE_DEPLOYMENT.md`. This recommendation uses official provider documentation as implementation evidence; it is not scholarly literature and is not entered in the scientific evidence ledger.
