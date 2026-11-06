"""Parameter-space sweep detector for KGW-family watermarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from watermark_lab.blind_detection.result import BlindDetectionResult

try:
    import watermark_core as _core  # type: ignore[import-untyped,import-not-found]

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass(frozen=True)
class ParameterSweepResult:
    best_z: float
    best_p: float
    best_setting: dict[str, Any]
    top_results: list[dict[str, Any]] = field(default_factory=list)
    total_tests: int = 0


class ParameterSweepDetector:
    """Scan gamma/hash/context-width settings and report the strongest KGW signal."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        if not _HAS_RUST:
            raise RuntimeError(
                "ParameterSweepDetector requires the watermark_core Rust extension"
            )

    def detect(
        self,
        text: str,
        gammas: list[float] | None = None,
        context_widths: list[int] | None = None,
        hash_keys: list[int] | None = None,
    ) -> BlindDetectionResult:
        tok = self._tokenizer.encode(text)
        if tok.token_count < 50:
            return BlindDetectionResult(
                method_id="parameter_sweep",
                status="inconclusive",
                decision="inconclusive",
                score=None,
                p_value=None,
                warnings=("too few tokens for parameter sweep",),
            )
        gammas = gammas or [0.1, 0.25, 0.5, 0.75]
        context_widths = context_widths or [1]
        hash_keys = hash_keys or [15485863, 999999, 123456789, 3141592653]
        raw_json = _core.py_parameter_sweep(
            tok.token_ids,
            tok.provenance.vocab_size,
            gammas,
            context_widths,
            [int(k) for k in hash_keys],
        )
        results = json.loads(raw_json)
        best = results[0]
        total = len(results)
        # Bonferroni correction over tested settings
        min_p = min(r["p_value"] for r in results)
        adjusted_p = min(min_p * total, 1.0)
        sweep = ParameterSweepResult(
            best_z=best["z_score"],
            best_p=best["p_value"],
            best_setting=best["setting"],
            top_results=results[:5],
            total_tests=total,
        )
        decision: Literal["watermarked", "not_watermarked", "inconclusive"] = (
            "watermarked" if best["z_score"] > 4.0 else "inconclusive"
        )
        return BlindDetectionResult(
            method_id="parameter_sweep",
            status="completed",
            decision=decision,
            score=best["z_score"],
            p_value=adjusted_p,
            details=sweep.__dict__,
            warnings=(
                f"parameter sweep over {total} settings; best setting {best['setting']}",
                "this is a candidate signal, not confirmed attribution",
            ),
        )
