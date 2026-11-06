"""Reference-corpus two-sample watermark detection.

Compares token-frequency distributions of target texts against a control corpus
using per-token two-sample tests. Detects provider-independent distributional
anomalies, not only KGW-family signals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from watermark_lab.blind_detection.result import BlindDetectionResult

LOGGER = logging.getLogger(__name__)

try:
    import watermark_core as _core  # type: ignore[import-untyped,import-not-found]

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass(frozen=True)
class TwoSampleResult:
    n_target: int
    n_control: int
    significant_tokens: list[dict[str, Any]] = field(default_factory=list)
    n_significant: int = 0


class ReferenceCorpusTwoSampleDetector:
    """Provider-independent two-sample test on token frequency vectors."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        if not _HAS_RUST:
            raise RuntimeError(
                "ReferenceCorpusTwoSampleDetector requires the watermark_core Rust extension"
            )

    def compare(
        self,
        target_texts: list[str],
        control_texts: list[str],
        alpha: float = 0.001,
    ) -> BlindDetectionResult:
        if not target_texts or not control_texts:
            return BlindDetectionResult(
                method_id="reference_corpus_two_sample",
                status="blocked",
                decision="inconclusive",
                score=None,
                p_value=None,
                warnings=("target and control corpora required",),
            )
        target_docs = [self._tokenizer.encode(t).token_ids for t in target_texts]
        control_docs = [self._tokenizer.encode(t).token_ids for t in control_texts]
        vocab_size = max(
            max(max(d) for d in target_docs) + 1 if target_docs else 1,
            max(max(d) for d in control_docs) + 1 if control_docs else 1,
        )
        raw = _core.py_token_frequency_vectors(target_docs + control_docs, vocab_size)
        vectors = json.loads(raw)
        t_vecs = np.array(vectors[: len(target_docs)], dtype=np.float64)
        c_vecs = np.array(vectors[len(target_docs) :], dtype=np.float64)
        # Per-token two-sample t-tests with Bonferroni correction
        significant: list[dict[str, Any]] = []
        n_tests = vocab_size
        for token_id in range(vocab_size):
            t_counts = t_vecs[:, token_id]
            c_counts = c_vecs[:, token_id]
            if t_counts.sum() == 0 and c_counts.sum() == 0:
                continue
            try:
                t_stat, p = stats.ttest_ind(t_counts, c_counts, equal_var=False)
            except Exception:
                LOGGER.debug("t-test failed for token %d", token_id, exc_info=True)
                continue
            if not np.isfinite(t_stat):
                continue
            if p * n_tests < alpha:
                significant.append(
                    {
                        "token_id": token_id,
                        "t_statistic": float(t_stat),
                        "p_value": float(p),
                        "target_mean": float(t_counts.mean()),
                        "control_mean": float(c_counts.mean()),
                    }
                )
        n_sig = len(significant)
        # Fisher's combined probability test across all tokens
        valid_ps = []
        for token_id in range(vocab_size):
            t_counts = t_vecs[:, token_id]
            c_counts = c_vecs[:, token_id]
            if t_counts.sum() == 0 and c_counts.sum() == 0:
                continue
            try:
                _, p = stats.ttest_ind(t_counts, c_counts, equal_var=False)
            except Exception:
                LOGGER.debug("t-test failed for token %d", token_id, exc_info=True)
                continue
            if p > 0 and p < 1:
                valid_ps.append(p)
        fisher_stat = -2.0 * sum(np.log(p) for p in valid_ps if p > 0)
        fisher_df = 2 * len(valid_ps)
        fisher_p = float(stats.chi2.sf(fisher_stat, fisher_df)) if fisher_df > 0 else 1.0
        result = TwoSampleResult(
            n_target=len(target_texts),
            n_control=len(control_texts),
            significant_tokens=significant,
            n_significant=n_sig,
        )
        # Distributional anomaly if many tokens differ or Fisher's test is significant
        if n_sig > 5 and fisher_p < 0.001:
            decision: Literal["watermarked", "not_watermarked", "inconclusive"] = "watermarked"
        else:
            decision = "inconclusive"
        return BlindDetectionResult(
            method_id="reference_corpus_two_sample",
            status="completed",
            decision=decision,
            score=float(n_sig),
            p_value=fisher_p,
            details=result.__dict__,
            warnings=(
                f"{n_sig} tokens differ significantly after Bonferroni correction",
                "distributional anomaly, not a specific watermark mechanism",
            ),
        )


__all__ = ["ReferenceCorpusTwoSampleDetector"]
