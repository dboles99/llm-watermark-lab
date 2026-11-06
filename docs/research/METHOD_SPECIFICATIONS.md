# Method Specifications and Reproduction Gates

## KGW-style baseline

- Seed a pseudorandom function from a secret key and token context.
- Partition the vocabulary into green and red subsets.
- Add a configurable logit bias to green tokens during generation.
- Detect by comparing observed green-token count with its null expectation using a z-score or exact test.
- Calibrate by language, domain, tokenizer, length, entropy, and repeated-context policy.

The exact bounded author profile, runtime invariants, direct-oracle evidence,
and remaining claim limits are specified in
`docs/research/KGW_AUTHOR_COMPATIBILITY.md`. The project Python-RNG adapter and
the author-compatible Torch adapter are distinct profiles and must never share
a compatibility label or calibration.

## MCmark-inspired adapter

- Implement the paper’s multi-channel construction and unbiased sampling assumptions.
- Keep channel count, mapping, key derivation, and aggregation explicit.
- Test marginal distribution preservation before calling the implementation “unbiased.”
- Reproduce matched no-attack, rephrase, back-translation, and strong paraphrase conditions.
- Label as inspired until the reproduction report passes.

## MirrorMark-inspired adapter

- Implement mod-1 mirrored pseudorandom mapping.
- Implement context-anchored scheduling and configurable payload positions.
- Report detection and payload recovery as different metrics.
- Test desynchronization, paraphrase, deletion, insertion, and translation.
- Label as inspired until the current paper version is reproduced.

## Claude and other proprietary systems

Create a provider adapter only when an official detector/API, normalization specification, authorization, and calibration documentation are available. Until then, Claude is represented only as an unsupported external provider in the capability matrix.

## Reproduction gate

Each method must have:

1. Equation-to-code mapping
2. Paper version and persistent identifier
3. Golden deterministic vectors
4. Unit and property tests
5. Matched baseline and attack corpus
6. False-positive calibration
7. Independent review by the research agent
8. Known-deviation report
