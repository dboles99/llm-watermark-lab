# KGW Author Compatibility Contract

**Status:** bounded token-ID mechanics passed; detector promotion pending  
**Profile:** `kgw-author-simple1-torch-2.13.0-cpu-win-amd64-v1`  
**Evidence date:** 2026-08-14

## Claim boundary

The repository now reproduces the ordered `simple_1` green list and complete
token-ID score semantics of one exact official KGW source/runtime profile. The
result is not a claim that every PyTorch, CPU, CUDA, operating-system, tokenizer,
or normalization configuration is equivalent.

The bounded profile fixes:

- official repository commit
  `82922516930c02f8aa322765defdb5863d07a00e`;
- `extended_watermark_processor.py` SHA-256
  `fc3469e677ba3f2a0e4cdf5a7cae2852d8d423c02cde395cc1e038fca42038a1`;
- `alternative_prf_schemes.py` SHA-256
  `51c854949c8e9309208dc6e57e2f9edbb8f2d90f0ff61edcd4ed2ef890e19ddc`;
- `simple_1`, additive PRF, context width 1, no self salt, integer hash key
  15,485,863, direct green partition, and `int(gamma * vocab_size)`;
- PyTorch `2.13.0+cpu`, upstream Git version
  `cf30153c4c131c8164ee7798e5022d810682e2cb`, Windows AMD64, CPU RNG; and
- token-ID mode with author normalizers disabled.

Any material change creates a different profile. The compatibility backend
refuses mismatches and never substitutes Python RNG or another device.

## Evidence

The immutable direct-author fixture covers fixed green lists, context boundary,
repeated full-bigram behavior, z scores, and survival-function p-values. The
differential script verifies source digests before import and passed:

- 10,000 randomized ordered green-list comparisons, zero mismatches;
- 1,000 randomized complete token-score comparisons, zero mismatches;
- six fixed raw-text comparisons through pinned GPT-2 tokenizer bytes (three
  `en`, three `zh-Hant-TW`), zero mismatches;
- sequence lengths 2–80, repeat masking enabled and disabled;
- vocabulary sizes 16, 17, 31, 64, 127, 257, and 1,024; and
- gamma values 0.10, 0.25, 0.50, and 0.75.

The evidence record is
`project/artifacts/scientific-validation/kgw-author-parity/step-01-greenlist/gate-report.json`.

## Tokenizer bridge

`ScientificKgwDetectionService` joins this author token-ID backend to a
`FaithfulTokenizer`. It requires exactly `en` or `zh-Hant-TW`, forbids special
token insertion for this profile, records exact tokenizer asset fingerprints,
and returns `inconclusive` because no matching frozen calibration exists. The
synthetic numeric tokenizer is accepted only as a labeled mechanics fixture.

The pinned GPT-2 profile is recorded in
`config/tokenizers/openai-community-gpt2-607a30d7.json`. Its five input files,
aggregate asset digest, tokenizer fingerprint, revision, and Transformers /
Tokenizers versions are fixed. The local loader excludes Hugging Face `.cache`
download bookkeeping because it is not tokenizer input and is retrieval-time
dependent.

The next raw-text gate must expand beyond six fixed strings and explicitly
compare randomized Unicode plus the author's optional Unicode sanitizer with
the repository normalization policy. GPT-2 is an English baseline; its ability
to encode `zh-Hant-TW` bytes is not evidence of Taiwan-language generation
quality.

## Remaining scientific gates

1. Randomized raw-text and optional author-normalizer parity beyond the fixed
   pinned-tokenizer cases.
2. Generation/logits-processor parity on a pinned model/tokenizer/runtime.
3. Permitted local PDF-byte hashes for the primary papers.
4. Separate frozen `en` and `zh-Hant-TW` calibration manifests and corpora.
5. Cluster-aware inference, adaptive attacks, and a blind clean-room rerun.

Until all applicable gates pass, the product detector remains prototype or
experimental and no estimated FPR is emitted.

## Evidence provenance

The scientific method is supported by Kirchenbauer et al., *A Watermark for
Large Language Models*, ICML 2023: [downloadable PMLR PDF](https://proceedings.mlr.press/v202/kirchenbauer23a/kirchenbauer23a.pdf)
and [DBLP index](https://dblp.org/rec/conf/icml/KirchenbauerGWK23).

Repeated full-n-gram scoring and robustness context are supported by
Kirchenbauer et al., *On the Reliability of Watermarks for Large Language
Models*, ICLR 2024: [downloadable ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/d78e9e4316e1714fbb0f20be66f8044c-Paper-Conference.pdf)
and [DBLP index](https://dblp.org/rec/conf/iclr/KirchenbauerGWS24).

The [official author repository](https://github.com/jwkirchenbauer/lm-watermarking/tree/82922516930c02f8aa322765defdb5863d07a00e)
is Apache-2.0 implementation evidence and is not treated as scholarly
literature.
