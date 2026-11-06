"""Blind and semi-blind watermark detection orchestration.

This package implements seven candidate blind detection methods:

1. Parameter-space sweep detector
2. Cross-document variance-ratio under random keys
3. Perturbation-invariance detector
4. Entropy-conditioned blind test
5. Reference-corpus two-sample test
6. Embedding-space cluster anomaly detector
7. Length-stratified meta-analysis

All methods are explicitly hypothesis-testing instruments, not attribution tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from watermark_lab.blind_detection.embedding import EmbeddingClusterAnomalyDetector
from watermark_lab.blind_detection.entropy import EntropyConditionedDetector
from watermark_lab.blind_detection.length_meta import LengthStratifiedMetaAnalysis
from watermark_lab.blind_detection.perturbation import PerturbationInvarianceDetector
from watermark_lab.blind_detection.result import BlindDetectionResult
from watermark_lab.blind_detection.sweep import ParameterSweepDetector
from watermark_lab.blind_detection.two_sample import ReferenceCorpusTwoSampleDetector
from watermark_lab.blind_detection.variance_ratio import VarianceRatioDetector


@dataclass(frozen=True)
class BlindDetectionSuiteResult:
    method_id: str
    status: Literal["completed", "inconclusive", "blocked"]
    decision: Literal["watermarked", "not_watermarked", "inconclusive"]
    score: float | None
    p_value: float | None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class BlindDetectionOrchestrator:
    """Run the full suite of blind/semi-blind watermark tests on tokenized text."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._sweep = ParameterSweepDetector(tokenizer)
        self._variance = VarianceRatioDetector(tokenizer)
        self._perturbation = PerturbationInvarianceDetector(tokenizer)
        self._length_meta = LengthStratifiedMetaAnalysis(tokenizer)
        self._two_sample = ReferenceCorpusTwoSampleDetector(tokenizer)
        self._entropy = EntropyConditionedDetector(tokenizer)
        self._embedding = EmbeddingClusterAnomalyDetector()

    def run_all(
        self,
        texts: list[str],
        control_texts: list[str] | None = None,
    ) -> dict[str, BlindDetectionResult | BlindDetectionSuiteResult]:
        """Run all blind detection methods and return a map of method_id to result.

        Args:
            texts: target texts (Claude/AI-generated candidates).
            control_texts: optional human/unwatermarked control texts.
        """
        results: dict[str, BlindDetectionResult | BlindDetectionSuiteResult] = {}
        # 1. Parameter sweep on the first/concatenated target text
        if texts:
            results["parameter_sweep"] = self._sweep.detect(
                texts[0],
                gammas=[0.1, 0.25, 0.5],
                context_widths=[1],
                hash_keys=[15485863, 999999, 123456789, 3141592653],
            )
        else:
            results["parameter_sweep"] = _blocked("parameter_sweep", "no target texts provided")

        # 2. Variance-ratio across all target docs
        results["variance_ratio"] = self._variance.detect(
            texts,
            gamma=0.25,
            candidate_keys=[15485863, 999999, 123456789, 3141592653],
            random_key_trials=20,
        )

        # 3. Perturbation invariance
        if control_texts and texts:
            results["perturbation_invariance"] = self._perturbation.compare(
                texts[0], control_texts[0]
            )
        else:
            results["perturbation_invariance"] = _blocked(
                "perturbation_invariance", "target and control texts required"
            )

        # 4. Length-stratified meta-analysis
        results["length_stratified"] = self._length_meta.analyze(texts)

        # 5. Reference-corpus two-sample test
        if control_texts:
            results["reference_corpus_two_sample"] = self._two_sample.compare(
                texts[:50], control_texts[:50]
            )
        else:
            results["reference_corpus_two_sample"] = _blocked(
                "reference_corpus_two_sample", "control corpus required"
            )

        # 6. Entropy-conditioned (blocked until reference model wired)
        results["entropy_conditioned"] = self._entropy.detect(texts[0] if texts else "")

        # 7. Embedding cluster anomaly (blocked until embedding model wired)
        results["embedding_cluster_anomaly"] = self._embedding.compare(
            texts[:20], (control_texts or [])[:20]
        )

        return results


def _blocked(method_id: str, reason: str) -> BlindDetectionSuiteResult:
    return BlindDetectionSuiteResult(
        method_id=method_id,
        status="blocked",
        decision="inconclusive",
        score=None,
        p_value=None,
        warnings=(f"Blocked: {reason}",),
    )


__all__ = ["BlindDetectionOrchestrator", "BlindDetectionSuiteResult"]
