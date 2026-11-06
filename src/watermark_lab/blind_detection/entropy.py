"""Entropy-conditioned blind KGW detection.

KGW applies stronger bias when the model has high entropy (more equiprobable
next tokens). Without access to the model's logits, we can estimate entropy
from a reference model. High-entropy positions should show stronger green-list
bias if a watermark is present.

This module is a stub because it requires a local reference model (e.g., GPT-2)
to estimate per-position entropy. It documents the interface and returns
inconclusive until such a model is wired.
"""

from __future__ import annotations

from typing import Any

from watermark_lab.blind_detection.result import BlindDetectionResult


class EntropyConditionedDetector:
    """Detect KGW-family watermark via entropy-conditioned green-list bias.

    Requires a reference model to estimate next-token entropy for the target text.
    """

    def __init__(self, tokenizer: Any, reference_model: Any | None = None) -> None:
        self._tokenizer = tokenizer
        self._reference_model = reference_model

    def detect(
        self,
        text: str,
        gamma: float = 0.25,
        hash_key: int = 15485863,
    ) -> BlindDetectionResult:
        if self._reference_model is None:
            return BlindDetectionResult(
                method_id="entropy_conditioned",
                status="blocked",
                decision="inconclusive",
                score=None,
                p_value=None,
                warnings=("requires a local reference model for entropy estimation",),
            )
        # Future: compute logits from reference model, partition positions by
        # estimated entropy (high/medium/low), run green-list test per partition.
        return BlindDetectionResult(
            method_id="entropy_conditioned",
            status="blocked",
            decision="inconclusive",
            score=None,
            p_value=None,
            warnings=("entropy-conditioned test not yet implemented",),
        )


__all__ = ["EntropyConditionedDetector"]
