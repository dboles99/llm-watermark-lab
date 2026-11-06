# Evaluation Protocol

## Evidence partitions and independence

Freeze four non-overlapping partitions before confirmatory work: development, calibration, attack-tuning, and final locked evaluation. Split by source document and author before extracting passages so repeated passages are not counted as independent evidence. Thresholds, attack policies, and model choices are locked before the final partition is opened.

## Minimum datasets

- Human academic prose by native and non-native English writers
- English, Taiwan Traditional Chinese (`zh-Hant-TW`), Japanese, and bilingual samples
- Academic abstracts, full paragraphs, references, code, equations, and tables
- Outputs from each configured generation model and decoding strategy
- Human-edited, lightly paraphrased, strongly paraphrased, translated, and mixed-source variants
- Licensed Taiwan-origin `zh-Hant-TW` material reviewed by native Taiwan editors; never mechanically convert `zh-CN` into the evaluation corpus

## Metrics

- ROC AUC, TPR at fixed 0.1%, 1%, and 5% FPR
- False positives by language, domain, author group, and length bin
- Detection power versus tokens and entropy
- Payload bit accuracy and complete payload recovery
- Quality change, semantic similarity, perplexity, and human review
- Robustness curves by attack type and severity
- Calibration error and confidence intervals
- Exact one-sided false-positive bounds, effective cluster sample size, and worst-group performance
- Localization span intersection-over-union, boundary error, and watermarked-coverage error
- Payload bit error rate, complete payload recovery, and colliding-key/provider demixing accuracy
- Taiwan acceptability, terminology correctness, factual/numeric/citation preservation, and Simplified-character contamination
- Adversarial query cost, forgery advantage, proof-verification cost, and key-compromise blast radius where applicable

## Statistical claim floor

For a one-sided 95% upper confidence bound below 0.1% after observing zero false positives, use at least 2,995 independent negative units in the applicable claim family. Two hundred examples have a 0.5% empirical resolution and cannot substantiate a 0.1% FPR claim. For about ±20% relative precision around a true 0.1% FPR under a simple normal approximation, plan for roughly 95,944 independent negatives per claimed stratum or explicitly weaken the claim. Model author/document clustering and report simultaneous or hierarchical uncertainty when scanning passages, detectors, keys, or thresholds.

For a 90% TPR estimate with an approximate ±3 percentage-point 95% margin, use at least 385 independent positives; the default planning floor is 500 per confirmatory cell. For a balanced human preference near 50% with an approximate ±5 percentage-point margin, use about 385 independent prompts; the default is 400 prompts with three ratings each and a mixed-effects analysis.

## Confirmatory hypotheses

- Exact author-code parity has zero unexplained mismatches on fixed vectors and at least 10,000 randomized contexts.
- False-positive error is controlled separately for `en` and `zh-Hant-TW`, with no universal threshold assumed.
- Repeated-context masking or a justified exact/permutation null improves calibration over a naive independent-token z-test.
- Worst-group FPR, TPR, and quality gaps remain within preregistered equivalence or non-inferiority margins.
- Watermarking does not increase Simplified-character or non-Taiwan lexical contamination in `zh-Hant-TW`.
- Adaptive attacks are evaluated separately from fixed attacks, and spoofing power is calibrated rather than reported anecdotally.

## Comparison rule

Never compare headline values from different papers as a ranking unless model, tokenizer, prompts, length, threshold, attack, and key policy are matched.

Every confirmatory study preregisters hypotheses, estimands, margins, exclusions, stopping rules, analysis code, multiple-testing families, and permissible exploratory analyses. Hash papers, code, author-code commits, tokenizers, model weights or API versions, corpora, prompts, splits, seeds, and environment manifests. A clean-room rerun by a blind reviewer is required before scientific promotion.

Frozen detector calibrations must validate against
`schemas/calibration-manifest.schema.json`. Schema validity is necessary but not
sufficient: analysis code must also call the semantic validator, recompute
interval values, and confirm that
the declared author/document clustering justifies the independent-unit count.

## Report language

Use `verified compatible watermark`, `not detected`, or `inconclusive`. Reserve provider attribution for authorized provider adapters. Do not output a generic “percent AI-written.”
