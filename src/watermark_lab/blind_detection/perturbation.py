"""Perturbation-invariance detector.

For unwatermarked text, a paraphrase changes the KGW z-score by a mean-zero
amount. For watermarked text, paraphrase tends to *reduce* the z-score because
the green-list token choices are disrupted. This detector compares the ΔW
distribution between a target text and a control text.
"""

from __future__ import annotations

import random
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
class PerturbationInvarianceResult:
    target_mean_delta: float
    control_mean_delta: float
    difference: float
    t_statistic: float
    p_value: float
    target_deltas: list[float] = field(default_factory=list)
    control_deltas: list[float] = field(default_factory=list)


class PerturbationInvarianceDetector:
    """Detect watermark presence via degradation direction under perturbation."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        if not _HAS_RUST:
            raise RuntimeError(
                "PerturbationInvarianceDetector requires the watermark_core Rust extension"
            )

    def compare(
        self,
        target_text: str,
        control_text: str,
        gamma: float = 0.25,
        hash_key: int = 15485863,
        n_variants: int = 30,
    ) -> BlindDetectionResult:
        target_deltas = self._delta_distribution(
            target_text, gamma, hash_key, n_variants, seed=20260815
        )
        control_deltas = self._delta_distribution(
            control_text, gamma, hash_key, n_variants, seed=20260816
        )
        if not target_deltas or not control_deltas:
            return BlindDetectionResult(
                method_id="perturbation_invariance",
                status="inconclusive",
                decision="inconclusive",
                score=None,
                p_value=None,
                warnings=("insufficient tokens to compute perturbation deltas",),
            )
        t, p_value = stats.ttest_ind(target_deltas, control_deltas, equal_var=False)
        result = PerturbationInvarianceResult(
            target_mean_delta=float(np.mean(target_deltas)),
            control_mean_delta=float(np.mean(control_deltas)),
            difference=float(np.mean(target_deltas) - np.mean(control_deltas)),
            t_statistic=float(t),
            p_value=float(p_value),
            target_deltas=target_deltas,
            control_deltas=control_deltas,
        )
        # Watermarked text should have more negative ΔW than control
        if result.difference < -0.3 and p_value < 0.05:
            decision: Literal["watermarked", "not_watermarked", "inconclusive"] = "watermarked"
        else:
            decision = "inconclusive"
        return BlindDetectionResult(
            method_id="perturbation_invariance",
            status="completed",
            decision=decision,
            score=result.difference,
            p_value=p_value,
            details=result.__dict__,
            warnings=(
                f"target ΔW mean={result.target_mean_delta:.3f}, control={result.control_mean_delta:.3f}",
                "negative difference suggests watermark degradation under perturbation",
            ),
        )

    def _delta_distribution(
        self,
        text: str,
        gamma: float,
        hash_key: int,
        n: int,
        seed: int,
    ) -> list[float]:
        base_z = self._z_score(text, gamma, hash_key)
        if base_z is None:
            return []
        deltas: list[float] = []
        rng = random.Random(seed)
        for i in range(n):
            variant = _core.py_word_window_shuffle(text, 4, rng.randint(0, 2**32 - 1))
            z = self._z_score(variant, gamma, hash_key)
            if z is not None:
                deltas.append(z - base_z)
        return deltas

    def _z_score(self, text: str, gamma: float, hash_key: int) -> float | None:
        tok = self._tokenizer.encode(text)
        if tok.token_count < 50:
            return None
        # Use parameter sweep with a single setting for speed
        import json

        raw = _core.py_parameter_sweep(
            tok.token_ids,
            tok.provenance.vocab_size,
            [gamma],
            [1],
            [hash_key],
        )
        results = json.loads(raw)
        return results[0]["z_score"] if results else None
