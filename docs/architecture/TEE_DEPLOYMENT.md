# Remote TEE Deployment Decision

**Decision status:** Future option; local Docker simulator is the active implementation  
**Decision date:** 2026-08-14  
**Evidence class:** Official provider documentation, not scholarly literature

## Recommended first cloud backend, when needed

If the project later approves a real cloud TEE, use **Azure Confidential Containers on Azure Container Instances (ACI)** for the first proof-carrying detector, with **Microsoft Azure Attestation**, an attestation/secure-key-release sidecar, and Azure Key Vault for release of the detector key epoch. Managed HSM is not the pilot default because its continuously allocated pool is disproportionately expensive.

This is the best fit for the first pilot because the detector is a bounded container service and does not yet need Kubernetes. Microsoft's current ACI documentation explicitly describes AMD SEV-SNP confidential container groups, remote guest attestation, verifiable container-enforcement policies, attestation and secure-key-release sidecars, and Key Vault/Managed HSM integration. The Azure confidential-container decision guide recommends ACI for simple or serverless container workloads and AKS when Kubernetes orchestration is actually required.

The local workstation remains the client, verifier, reproducibility runner, and development-simulator host. It is not treated as a production enclave.

## Cost boundary recorded on 2026-08-14

The Microsoft public retail-prices API returned these pay-as-you-go confidential ACI meters for Southeast Asia:

- USD 0.057915 per vCPU-hour;
- USD 0.006364 per GB-hour;
- Basic Azure Container Registry at USD 0.1666 per day; and
- Managed HSM Standard B1 at USD 3.20 per hour.

For a one-vCPU/two-GB pilot, compute is approximately USD 2.83 for 40 hours, USD 7.06 for 100 hours, or USD 51.57 for 730 hours. Adding Basic Container Registry and low-volume Key Vault/logging gives a planning range of roughly USD 8–12, USD 13–18, or USD 58–70 respectively. Microsoft Azure Attestation is currently free.

Managed HSM would add approximately USD 2,336 for a 730-hour month before compute, so it is explicitly excluded from a small pilot unless a later threat model requires the dedicated pool. These are public retail estimates, not a quote; region, agreement, taxes, currency, logging, networking, and data transfer can change the bill.

## Proposed evidence flow

1. The verifier issues a fresh, single-use nonce and the accepted measurement/policy set.
2. The confidential container presents platform evidence through Microsoft Azure Attestation.
3. The relying party validates the attestation token, freshness, nonce binding where supported by the selected integration, platform claims, container policy digest, image/release measurement, and revocation state.
4. Key Vault or Managed HSM releases the detector key epoch only when the attested claims satisfy the pinned release policy.
5. The client submits text over an authenticated encrypted channel only after successful attestation. Raw text remains inside the confidential workload for the detection operation and is not written to the provenance repository.
6. The workload returns the versioned verification statement and detached evidence defined by `schemas/tee-verification-statement.schema.json`.
7. The local verifier checks the platform evidence, statement commitments, nonce, expiry, detector/config/calibration versions, key epoch, and evidence hash before storing a receipt.

The detailed implementation must confirm exactly which application data can be cryptographically bound into the platform report or relying-party token. Unsupported binding must fail closed; it must not be simulated with an unsigned metadata field.

## Security boundary

TEE attestation can support a claim that an approved workload ran in an approved confidential-compute environment. It does not by itself:

- validate the detector's false-positive or power claims;
- prove that an identity reference denotes the human author of every word;
- prevent input-length, timing, traffic-analysis, availability, or all microarchitectural side channels;
- make the cloud control plane or application dependencies irrelevant;
- justify retaining raw text; or
- make simulator evidence promotable.

The remote gate therefore includes adversarial tests for replay, rollback, stale images and policies, key release, rate limiting, revocation, malformed evidence, unavailable attestation services, logging leakage, crash dumps, outbound network policy, and denial of service. T06 calibration remains a scientific prerequisite.

## Alternatives and promotion triggers

| Backend | Use when | Current disposition |
| --- | --- | --- |
| ACI confidential containers | One bounded service, minimal orchestration, rapid attested-container pilot | **Recommended first** |
| Azure confidential VM | The detector needs full guest-OS control or direct guest-attestation integration | Secondary option |
| AKS confidential containers | Multi-service scale, Kubernetes policy, or autoscaling becomes a demonstrated requirement | Defer; the current Azure comparison page labels the confidential-container offering preview |
| Azure Container Apps confidential compute | Its current attestation, evidence-export, and key-release contract is verified to satisfy the statement schema | Evaluate, not the first commitment |

Moving between backends is a versioned deployment change. It requires a new measurement policy and a regression of attestation, key isolation, scientific parity, and receipt verification.

## Human approvals required before provisioning

- Azure subscription and resource-group boundary;
- allowed region and text/data-residency policy;
- monthly pilot budget and shutdown policy;
- tenant/managed-identity and Key Vault or Managed HSM boundary;
- inbound and outbound network policy;
- whether raw input may transit the selected Azure region, even transiently; and
- retention, audit access, incident response, and key-revocation owners.

Credentials, subscription secrets, and key material must not be placed in repository files, notebooks, prompts, artifacts, logs, or chat transcripts.

Until these approvals are given, Azure remains documentation only. The active test path is `deploy/local-tee/`, which provides protocol and failure-mode testing but no hardware isolation or platform attestation.

## Official platform sources

- [Microsoft Azure Attestation overview](https://learn.microsoft.com/en-us/azure/attestation/overview)
- [Confidential containers on Azure Container Instances](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-confidential-overview)
- [Guest attestation for confidential VMs](https://learn.microsoft.com/en-us/azure/confidential-computing/guest-attestation-confidential-vms)
- [Choose a confidential-container offering](https://learn.microsoft.com/en-us/azure/confidential-computing/choose-confidential-containers-offerings)
- [Azure confidential containers overview](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-containers)
- [Azure Container Instances pricing](https://azure.microsoft.com/en-us/pricing/details/container-instances/)
- [Azure retail-prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices)
- [Azure Attestation pricing](https://azure.microsoft.com/en-us/pricing/details/azure-attestation/)
- [Azure Key Vault pricing](https://azure.microsoft.com/en-us/pricing/details/key-vault/)

These are implementation/vendor sources. They are intentionally excluded from `research/frontier_evidence.json`, whose admission contract is limited to scholarly literature with a verified index record and downloadable PDF.
