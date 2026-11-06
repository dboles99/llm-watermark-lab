"""War-dial parameter spaces for textual watermark experiments.

A "dial" is one coordinate in a watermark parameter grid.  The war-dial runner
enumerates the grid, runs a deterministic coding test and an encoding
roundtrip for each coordinate, applies a stochastic perturbation, and saves
the result so the grid can be refined in the next iteration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KgwDial:
    """One KGW generation/detection coordinate."""

    prompt: str
    gamma: float
    delta: float
    context_width: int
    max_new_tokens: int
    key_id: str = "test-kgw-v1"
    tokenizer_id: str = "gpt2"
    hash_key: int = 15485863
    seeding_scheme: str = "simple_1"
    vocab_size: int = 50257
    detection_threshold: float = 4.0

    def to_config(self) -> dict[str, Any]:
        return {
            "profile_id": "mock",
            "key_id": self.key_id,
            "tokenizer_id": self.tokenizer_id,
            "max_new_tokens": self.max_new_tokens,
            "gamma": self.gamma,
            "delta": self.delta,
            "context_width": self.context_width,
            "seeding_scheme": self.seeding_scheme,
            "hash_key": self.hash_key,
            "vocab_size": self.vocab_size,
            "detection_threshold": self.detection_threshold,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StegoDial:
    """One steganographic encoding coordinate."""

    channel: str
    cover: str
    payload: str
    secret: str
    key_id: str = "test-kgw-v1"
    vocab_size: int = 50257
    gamma: float = 0.25
    hash_key: int = 15485863

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationDial:
    """One stochastic perturbation coordinate."""

    attack_id: str
    severity: float
    seed: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WarDialCoordinate:
    """A full coordinate combining generation, stego, and perturbation settings."""

    kgw: KgwDial | None = None
    stego: StegoDial | None = None
    perturbation: PerturbationDial | None = None
    iteration: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "kgw": self.kgw.as_dict() if self.kgw else None,
            "stego": self.stego.as_dict() if self.stego else None,
            "perturbation": self.perturbation.as_dict() if self.perturbation else None,
        }
