"""Length-stratified meta-analysis for watermark detection power.

Short documents have noisy z-scores; long documents have precise ones.
This module pools z-scores by length bins using inverse-variance weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from watermark_lab.blind_detection.result import BlindDetectionResult

try:
    import watermark_core as _core  # type: ignore[import-untyped,import-not-found]

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass(frozen=True)
class LengthStratum:
    min_len: int
    max_len: int
    n_docs: int
    pooled_z: float
    pooled_var: float


@dataclass(frozen=True)
class LengthMetaResult:
    strata: list[LengthStratum] = field(default_factory=list)
    pooled_z: float | None = None
    pooled_p: float | None = None


class LengthStratifiedMetaAnalysis:
    """Recover power from variable-length documents via inverse-variance pooling."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        if not _HAS_RUST:
            raise RuntimeError(
                "LengthStratifiedMetaAnalysis requires the watermark_core Rust extension"
            )

    def analyze(
        self,
        texts: list[str],
        gamma: float = 0.25,
        hash_key: int = 15485863,
    ) -> BlindDetectionResult:
        import json

        z_scores: list[tuple[int, float]] = []
        for text in texts:
            tok = self._tokenizer.encode(text)
            if tok.token_count < 50:
                continue
            raw = _core.py_parameter_sweep(
                tok.token_ids,
                tok.provenance.vocab_size,
                [gamma],
                [1],
                [hash_key],
            )
            results = json.loads(raw)
            z_scores.append((tok.token_count, results[0]["z_score"]))
        if not z_scores:
            return BlindDetectionResult(
                method_id="length_stratified",
                status="inconclusive",
                decision="inconclusive",
                score=None,
                p_value=None,
                warnings=("no documents with >=50 tokens available",),
            )
        # Strata: short (<100), medium (100-300), long (>300)
        bins = [(0, 100), (100, 300), (300, 10_000)]
        strata: list[LengthStratum] = []
        all_z: list[float] = []
        all_w: list[float] = []
        for lo, hi in bins:
            subset = [z for n, z in z_scores if lo <= n < hi]
            if not subset:
                continue
            # Within stratum variance is approx 1 / effective count, but we have z-scores
            # with unit variance. Use the empirical variance as estimate.
            var = float(np.var(subset, ddof=1)) if len(subset) > 1 else 1.0
            if var <= 0:
                var = 1.0
            pooled_z = float(np.mean(subset))
            strata.append(
                LengthStratum(
                    min_len=lo,
                    max_len=hi,
                    n_docs=len(subset),
                    pooled_z=pooled_z,
                    pooled_var=var,
                )
            )
            all_z.append(pooled_z)
            all_w.append(1.0 / var)
        if not all_z:
            return BlindDetectionResult(
                method_id="length_stratified",
                status="inconclusive",
                decision="inconclusive",
                score=None,
                p_value=None,
                warnings=("no length strata populated",),
            )
        weights = np.array(all_w)
        zs = np.array(all_z)
        pooled_z = float(np.sum(weights * zs) / np.sum(weights))
        pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
        pooled_p = float(stats.norm.sf(pooled_z))
        meta_result = LengthMetaResult(
            strata=strata, pooled_z=pooled_z, pooled_p=pooled_p
        )
        decision: Literal["watermarked", "not_watermarked", "inconclusive"] = (
            "watermarked" if pooled_z > 4.0 else "inconclusive"
        )
        return BlindDetectionResult(
            method_id="length_stratified",
            status="completed",
            decision=decision,
            score=pooled_z,
            p_value=pooled_p,
            details={"strata": [s.__dict__ for s in strata], "pooled_z": pooled_z, "meta": meta_result.__dict__},
            warnings=(
                f"length-stratified pooled z={pooled_z:.3f} (se={pooled_se:.3f})",
            ),
        )
