# Annotated Bibliography: LLM Text Watermarking, Detection, Attacks, and Governance

**Scope:** Generation-time text watermarking and directly relevant evidence on detection, attacks, fairness, benchmarking, and Anthropic connections.  
**Canonical admitted evidence:** `research/frontier_evidence.json`
**Broader discovery data:** `research/annotated_bibliography.csv`
**Last evidence review:** August 14, 2026

## How to maintain this bibliography

The CSV is a discovery and annotation table, not an admission decision. A paper may support a scientific claim only after it appears in `frontier_evidence.json` with a verified scholarly index record and directly downloadable PDF. Preserve the exact paper version, DOI, arXiv identifier, status, and check date. Never replace a preprint silently with a conference version.

The research agent should search Google Scholar, Semantic Scholar, Crossref, OpenAlex, DBLP, and venue repositories for discovery and metadata reconciliation. Claims must be verified in the primary paper or official provider documentation.

## Foundational methods

### Kirchenbauer et al. (2023), A Watermark for Large Language Models — PRIMARY CLAUDE HYPOTHESIS

Introduces keyed green-list logit bias and an interpretable statistical detector. This is the recommended first implementation because the main detector statistic and generation intervention are relatively transparent. Use it for the Python reference, Rust parity tests, length calibration, and initial GUI report.

**Evidence role (promoted 2026-08-15):** `PRIMARY_CLAUDE_WATERMARK_HYPOTHESIS` — the KGW family is treated as the primary candidate mechanism for hypothesis testing against Claude-generated text (see `docs/research/CLAUDE_WATERMARK_HYPOTHESIS.md`). Implementation roles: canonical generator, canonical detector, positive control, Claude hypothesis model, perturbation baseline, hybridization anchor, parameter-search target, robustness benchmark. **This is a hypothesis, not a finding: no public evidence establishes that Anthropic deploys this or any text watermark.** Section 7 of the paper attacks its own watermark with black-box span replacement under a rising budget, providing the methodological precedent for constrained perturbation analysis (signal suppression subject to semantic and quality floors).

Persistent records: PMLR 202:17061–17084; arXiv:2301.10226; arXiv/DataCite DOI 10.48550/arXiv.2301.10226 (no separate PMLR/Crossref DOI for the proceedings paper).  
Primary source: https://proceedings.mlr.press/v202/kirchenbauer23a.html  
Local pinned PDF: `research/papers/a-watermark-for-large-language-models/paper.pdf` (SHA-256 `192bbc0d3929b72b…`)

### Kuditipudi et al. (2024), Robust Distortion-Free Watermarks for Language Models

Develops a distribution-preserving construction based on alignment with a secret random sequence. The paper is important because its experiments demonstrate how sharply detection can depend on task entropy and output length. A reproduction should use it to test claims of distortion-free behavior and to develop alignment-based attack analysis.

Persistent record: arXiv:2307.15593.  
Primary source: https://openreview.net/forum?id=FpaCL1MO2C

### Hu et al. (2024), Unbiased Watermark for Large Language Models

Proposes an unbiased construction that preserves the marginal output distribution under specified assumptions. It is the conceptual precursor to MCmark. Any implementation using the word “unbiased” must reproduce the relevant marginal-distribution property tests.

Persistent record: arXiv:2310.10669.  
Primary source: https://openreview.net/forum?id=uWVC5FVidc

### Christ, Gunn, and Zamir (2024), Undetectable Watermarks for Language Models

Develops cryptographic watermark constructions intended to be computationally indistinguishable without a secret key. The paper clarifies why a secure watermark can be invisible to an ordinary observer while remaining detectable to a compatible verifier.

Persistent records: PMLR 247:1125–1139; arXiv:2306.09194.  
Primary source: https://proceedings.mlr.press/v247/christ24a.html

### Zhao et al. (2024), Provable Robust Watermarking for AI-Generated Text

Provides formal robustness guarantees under a defined editing model. The value here is not simply the guarantee, but the opportunity to compare the formal attack assumptions with real proofreading, academic editing, and translation operations.

Persistent record: arXiv:2306.17439.  
Primary source: https://arxiv.org/abs/2306.17439

### Dathathri et al. (2024), Scalable Watermarking for Identifying Large Language Model Outputs

Describes SynthID-Text and deployment-oriented evaluation. It is relevant to production architecture and large-scale verification, although an outside researcher cannot assume access to proprietary model internals or keys.

DOI: 10.1038/s41586-024-08025-4.  
Primary source: https://www.nature.com/articles/s41586-024-08025-4

## Robustness and attacks

### Kirchenbauer et al. (2024), On the Reliability of Watermarks for Large Language Models

Extends analysis of the KGW family across attacks and reliability conditions. It is essential for repeated-context handling, attack design, and calibration choices.

Persistent record: arXiv:2306.04634.  
Primary source: https://openreview.net/forum?id=aX8ig9X2a7

### He et al. (2024), Can Watermarks Survive Translation?

Shows that pivot-language translation can reduce tested watermark discrimination toward chance and proposes X-SIR. This paper is central here because translation and bilingual editing are ordinary workflows, not exotic attacks.

DOI: 10.18653/v1/2024.acl-long.226.  
Primary source: https://aclanthology.org/2024.acl-long.226/

### Wu and Chandrasekaran (2024), Bypassing LLM Watermarks with Color-Aware Substitutions

Demonstrates black-box inference of token-class information and targeted substitution attacks. It motivates rate controls, key rotation, secret partition policies, and adversarial query tests.

DOI: 10.18653/v1/2024.acl-long.464.  
Primary source: https://aclanthology.org/2024.acl-long.464/

### Krishna et al. (2023), Paraphrasing Evades Detectors of AI-Generated Text

Shows that paraphrasing can evade several AI-text detectors, while retrieval can help under some provenance conditions. It is not limited to generation-time watermarks, but it is a necessary adversarial baseline.

Persistent record: arXiv:2303.13408.  
Primary source: https://proceedings.neurips.cc/paper_files/paper/2023/hash/575c450013d0e99e4b0ecf82bd1afaa4-Abstract-Conference.html

### Pang et al. (2024), No Free Lunch in LLM Watermarking

Formalizes design trade-offs among robustness, utility, and usability. It supports reporting multiple operational metrics rather than selecting one preferred detection result.

Primary source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa86a9c7b9f341716ccb679d1aeb9afa-Abstract-Conference.html

### Zhang et al. (2024), Watermarks in the Sand

Provides impossibility results for sufficiently strong watermarking guarantees in broad settings. This is a strong argument for layered provenance evidence rather than watermark-only governance.

Persistent record: arXiv:2311.04378.  
Primary source: https://openreview.net/forum?id=bM2s12t4hR

## Benchmarks, reviews, and fairness

### Dugan et al. (2024), RAID

Benchmarks AI-text detectors across models, domains, decoding strategies, and attacks. Its main contribution to this project is evaluation design: detector performance should be stratified and stress-tested rather than summarized by one accuracy number.

DOI: 10.18653/v1/2024.acl-long.674.  
Primary source: https://aclanthology.org/2024.acl-long.674/

### Liu et al. (2024), A Survey of Text Watermarking in the Era of Large Language Models

Synthesizes method taxonomies, requirements, attacks, and evaluation practices. It is useful for research routing, while implementation claims should still be checked in original method papers.

DOI: 10.1145/3691626.  
Primary source: https://dl.acm.org/doi/10.1145/3691626

### Piet et al. (2025), Mark My Words

Benchmarks quality, token requirements, and tamper resistance. It emphasizes task dependence and reports that code is a particularly difficult domain for some watermarks.

Persistent record: arXiv:2312.00273.  
Primary source: https://arxiv.org/abs/2312.00273

### Zhao et al. (2025), SoK: Watermarking for AI-Generated Content

Systematizes threat models, properties, evaluation criteria, and open problems. Nicholas Carlini is listed with an Anthropic affiliation. This is the clearest verified Anthropic research connection in the core corpus, but it does not reveal Claude’s implementation.

DOI: 10.1109/SP61157.2025.00178; arXiv:2411.18479.  
Primary source: https://arxiv.org/abs/2411.18479

### Liang et al. (2023), GPT Detectors Are Biased Against Non-Native English Writers

Finds substantial false-positive risk for non-native English writing in generic detectors. It directly supports multilingual calibration, fairness reporting, and a ban on automatic accusations.

DOI: 10.1016/j.patter.2023.100779.  
Primary source: https://www.cell.com/patterns/fulltext/S2666-3899(23)00130-7

### Sadasivan et al. (2023), Can AI-Generated Text Be Reliably Detected?

Argues that generic detection degrades under distribution shift and paraphrasing. The paper helps define the boundary between keyed watermark verification and weaker keyless classification.

Persistent record: arXiv:2303.11156.  
Primary source: https://arxiv.org/abs/2303.11156

## Recent candidate methods

### Chen et al. (2025), Improved Unbiased Watermark for Large Language Models

Introduces MCmark, a multi-channel method that reports stronger detection than previous unbiased approaches. Robustness varies substantially across rephrasing, back translation, and stronger paraphrase attacks. Any reproduction should therefore reproduce attack-specific results rather than adopting one headline value.

DOI: 10.18653/v1/2025.acl-long.1005; arXiv:2502.11268.  
Primary source: https://aclanthology.org/2025.acl-long.1005/

### Jiang et al. (2026), MirrorMark: Generalizable Mirrored Sampling for Multi-bit LLM Watermarking

Uses mirrored pseudorandom mappings and context-anchored scheduling for multi-bit watermarking. Detection can survive some rewriting conditions better than payload recovery. The current version is a preprint and has no verified Anthropic connection.

Persistent record: arXiv:2601.22246v3.  
Primary source: https://arxiv.org/abs/2601.22246

## Research-agent update protocol

1. Search for new citing and cited papers, current versions, corrections, retractions, and author code.
2. Reconcile DOI, arXiv, venue, and title metadata across indexes.
3. Read the primary paper before changing an annotation.
4. Record version and check date.
5. Add new controlled vocabulary values only through a documented schema change.
6. Run the bibliography notebook and investigate duplicate identifiers, missing fields, stale preprints, and uncovered method or attack categories.
7. Open a paper-ingestion task before any new method enters the codebase.
