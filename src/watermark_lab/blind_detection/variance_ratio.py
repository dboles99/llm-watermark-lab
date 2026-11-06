"""Cross-document variance-ratio detector for fixed-key watermarks.

The test statistic exploits the fact that a fixed-key watermark creates
cross-document correlation: under the correct key, z-scores across documents
are systematically shifted and their cross-document variance shrinks.
Under H0 (no watermark), z-scores are iid N(0,1) and cross-document
variance is ~1.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from watermark_lab.blind_detection.result import BlindDetectionResult

try:
    import watermark_core as _core  # type: ignore[import-untyped,import-not-found]

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass(frozen=True)
class VarianceRatioResult:
    best_key: int
    best_mean_z: float
    best_variance_ratio: float
    key_results: list[dict[str, Any]] = field(default_factory=list)
    permutation_p_value: float | None = None


class VarianceRatioDetector:
    """Detect fixed-key KGW-family watermarks via cross-document variance ratio."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        if not _HAS_RUST:
            raise RuntimeError(
                "VarianceRatioDetector requires the watermark_core Rust extension"
            )

    def detect(
        self,
        texts: list[str],
        gamma: float = 0.25,
        candidate_keys: list[int] | None = None,
        random_key_trials: int = 50,
    ) -> BlindDetectionResult:
        if len(texts) < 3:
            return BlindDetectionResult(
                method_id="variance_ratio",
                status="inconclusive",
                decision="inconclusive",
                score=None,
                p_value=None,
                warnings=("need at least 3 documents for variance-ratio test",),
            )
        token_docs = []
        for text in texts:
            tok = self._tokenizer.encode(text)
            token_docs.append(tok.token_ids)
        vocab_size = 0
        if token_docs:
            vocab_size = max(max(d) for d in token_docs) + 1
        try:
            vocab_size = max(vocab_size, self._tokenizer.spec.vocab_size)
        except AttributeError:
            pass
        keys = [int(k) for k in (candidate_keys or [])]
        # Add random keys for the H0 reference distribution
        rng = random.Random(20260815)
        keys.extend(rng.randint(1, 2**32 - 1) for _ in range(random_key_trials))
        raw_json = _core.py_variance_ratio_matrix(token_docs, vocab_size, gamma, keys)
        matrix = json.loads(raw_json)
        # Find key with smallest variance ratio (most correlated z-scores)
        best = min(matrix, key=lambda r: r["variance_ratio"])
        # Permutation p-value: compare best variance ratio to random-key ratios
        random_ratios = [r["variance_ratio"] for r in matrix[1:]]  # first is a candidate
        if random_ratios:
            rank = sum(1 for v in random_ratios if v <= best["variance_ratio"]) + 1
            p_value = rank / (len(random_ratios) + 1)
        else:
            p_value = None
        result = VarianceRatioResult(
            best_key=best["hash_key"],
            best_mean_z=best["mean_z"],
            best_variance_ratio=best["variance_ratio"],
            key_results=matrix,
            permutation_p_value=p_value,
        )
        # Decision: small variance ratio + high mean z is suspicious
        if best["mean_z"] > 2.0 and best["variance_ratio"] < 0.5:
            decision: Literal["watermarked", "not_watermarked", "inconclusive"] = "watermarked"
        else:
            decision = "inconclusive"
        return BlindDetectionResult(
            method_id="variance_ratio",
            status="completed",
            decision=decision,
            score=best["mean_z"],
            p_value=p_value,
            details=result.__dict__,
            warnings=(
                (
                    f"best key {best['hash_key']}: mean_z={best['mean_z']:.3f}, "
                    f"variance_ratio={best['variance_ratio']:.3f}"
                ),
                "small variance ratio with high mean z suggests a shared secret key",
            ),
        )
