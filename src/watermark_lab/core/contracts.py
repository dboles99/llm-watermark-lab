from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class DetectionConfig:
    method_id: str = "kgw"
    tokenizer_id: str = "gpt2"
    key_id: str | None = None
    threshold: float = 4.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    method_id: str
    compatibility: str
    decision: str
    score: float | None
    p_value: float | None
    token_count: int
    estimated_fpr: float | None
    warnings: tuple[str, ...] = ()
    original_scorable_token_count: int | None = None
    ignored_repeated_ngrams: int = 0
    scoring_variant: str = "naive-binomial"
    calibration_id: str | None = None


class DetectorPlugin(Protocol):
    method_id: str

    def detect(self, text: str, config: DetectionConfig) -> DetectionResult: ...


class WatermarkerPlugin(Protocol):
    method_id: str

    def generate(self, prompt: str, config: dict[str, Any]) -> dict[str, Any]: ...
