"""Transformation plugins for robustness/attack experiments."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Any, ClassVar, Protocol


def _word_tokenize(text: str) -> list[str]:
    """Split ``text`` into a sequence of word and non-word tokens."""
    return re.findall(r"\w+|\W+", text, flags=re.UNICODE)


def _word_count(text: str) -> int:
    return len([t for t in _word_tokenize(text) if re.search(r"\w", t, flags=re.UNICODE)])


@dataclass(frozen=True)
class AttackConfig:
    attack_id: str
    severity: float = 0.0
    seed: str = "default"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.severity, bool) or not isinstance(self.severity, (int, float)):
            raise TypeError("severity must be a finite number in [0, 1]")
        if not isfinite(self.severity) or not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")


class TransformationPlugin(Protocol):
    attack_id: str

    def transform(self, text: str, config: AttackConfig) -> dict[str, Any]: ...


class DeletionPlugin:
    """Delete a fraction of word tokens."""

    attack_id = "deletion"

    def transform(self, text: str, config: AttackConfig) -> dict[str, Any]:
        rng = random.Random(config.seed)
        tokens = _word_tokenize(text)
        severity = float(config.severity)
        kept: list[str] = []
        for token in tokens:
            if not re.search(r"\w", token, flags=re.UNICODE) or rng.random() > severity:
                kept.append(token)
        transformed = "".join(kept)
        return {
            "attack_id": self.attack_id,
            "severity": severity,
            "seed": config.seed,
            "transformed_text": transformed,
            "tokens_deleted": _word_count(text) - _word_count(transformed),
        }


class InsertionPlugin:
    """Insert random filler words between existing word tokens."""

    attack_id = "insertion"

    _FILLER: ClassVar[list[str]] = ["the", "a", "very", "quite", "really", "then", "thus", "so"]

    def transform(self, text: str, config: AttackConfig) -> dict[str, Any]:
        rng = random.Random(config.seed)
        severity = float(config.severity)
        tokens = _word_tokenize(text)
        word_positions = [
            index for index, token in enumerate(tokens) if re.search(r"\w", token, flags=re.UNICODE)
        ]
        insertion_candidates = word_positions[:-1]
        if not insertion_candidates or severity == 0.0:
            return {
                "attack_id": self.attack_id,
                "severity": severity,
                "seed": config.seed,
                "transformed_text": text,
                "tokens_inserted": 0,
            }

        insertion_count = min(
            len(insertion_candidates), ceil(severity * len(insertion_candidates))
        )
        insertion_positions = set(rng.sample(insertion_candidates, insertion_count))
        out: list[str] = []
        for index, token in enumerate(tokens):
            out.append(token)
            if index in insertion_positions:
                out.extend((" ", rng.choice(self._FILLER)))
        transformed = "".join(out)
        return {
            "attack_id": self.attack_id,
            "severity": severity,
            "seed": config.seed,
            "transformed_text": transformed,
            "tokens_inserted": insertion_count,
        }


class SubstitutionPlugin:
    """Replace a fraction of word tokens with random words from a small lexicon."""

    attack_id = "substitution"

    _LEXICON: ClassVar[list[str]] = ["apple", "river", "blue", "swift", "quiet", "stone", "lunar", "open", "bright", "matter"]

    def transform(self, text: str, config: AttackConfig) -> dict[str, Any]:
        rng = random.Random(config.seed)
        severity = float(config.severity)
        tokens = _word_tokenize(text)
        replaced = 0
        out: list[str] = []
        for token in tokens:
            if re.search(r"\w", token, flags=re.UNICODE) and rng.random() < severity:
                out.append(rng.choice(self._LEXICON))
                replaced += 1
            else:
                out.append(token)
        return {
            "attack_id": self.attack_id,
            "severity": severity,
            "seed": config.seed,
            "transformed_text": "".join(out),
            "tokens_replaced": replaced,
        }


class ShufflePlugin:
    """Shuffle a severity-controlled subset of words in their original slots."""

    attack_id = "shuffle"

    def transform(self, text: str, config: AttackConfig) -> dict[str, Any]:
        severity = float(config.severity)
        if severity == 0.0:
            return {
                "attack_id": self.attack_id,
                "severity": severity,
                "seed": config.seed,
                "transformed_text": text,
                "tokens_shuffled": 0,
            }

        rng = random.Random(config.seed)
        tokens = _word_tokenize(text)
        word_positions = [
            index for index, token in enumerate(tokens) if re.search(r"\w", token, flags=re.UNICODE)
        ]
        selected_count = min(len(word_positions), ceil(severity * len(word_positions)))
        if selected_count < 2:
            return {
                "attack_id": self.attack_id,
                "severity": severity,
                "seed": config.seed,
                "transformed_text": text,
                "tokens_shuffled": 0,
            }
        selected_positions = rng.sample(word_positions, selected_count)
        original_words = [tokens[index] for index in selected_positions]
        shuffled_words = list(original_words)
        rng.shuffle(shuffled_words)
        if shuffled_words == original_words:
            shuffled_words = original_words[1:] + original_words[:1]
        for index, word in zip(selected_positions, shuffled_words, strict=True):
            tokens[index] = word
        tokens_shuffled = sum(
            before != after for before, after in zip(original_words, shuffled_words, strict=True)
        )
        return {
            "attack_id": self.attack_id,
            "severity": severity,
            "seed": config.seed,
            "transformed_text": "".join(tokens),
            "tokens_shuffled": tokens_shuffled,
        }


class ParaphraseStubPlugin:
    """Placeholder paraphrase attack that applies substitution + deletion."""

    attack_id = "paraphrase"

    def transform(self, text: str, config: AttackConfig) -> dict[str, Any]:
        if config.severity == 0.0:
            return {
                "attack_id": self.attack_id,
                "severity": 0.0,
                "seed": config.seed,
                "transformed_text": text,
                "substitutions": 0,
                "deletions": 0,
            }
        sub_cfg = AttackConfig(
            attack_id="substitution", severity=config.severity, seed=config.seed
        )
        sub = SubstitutionPlugin().transform(text, sub_cfg)
        del_cfg = AttackConfig(attack_id="deletion", severity=config.severity / 2, seed=config.seed)
        deleted = DeletionPlugin().transform(sub["transformed_text"], del_cfg)
        return {
            "attack_id": self.attack_id,
            "severity": config.severity,
            "seed": config.seed,
            "transformed_text": deleted["transformed_text"],
            "substitutions": sub.get("tokens_replaced", 0),
            "deletions": deleted.get("tokens_deleted", 0),
        }


class TransformationRegistry:
    """Registry of built-in transformation plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, TransformationPlugin] = {}
        for plugin in (
            DeletionPlugin(),
            InsertionPlugin(),
            SubstitutionPlugin(),
            ShufflePlugin(),
            ParaphraseStubPlugin(),
        ):
            self._plugins[plugin.attack_id] = plugin

    def transform(self, text: str, config: AttackConfig) -> dict[str, Any]:
        if config.attack_id not in self._plugins:
            raise ValueError(f"Unknown attack_id: {config.attack_id!r}")
        return self._plugins[config.attack_id].transform(text, config)

    def list_attacks(self) -> list[str]:
        return sorted(self._plugins)


__all__ = [
    "AttackConfig",
    "DeletionPlugin",
    "InsertionPlugin",
    "ParaphraseStubPlugin",
    "ShufflePlugin",
    "SubstitutionPlugin",
    "TransformationPlugin",
    "TransformationRegistry",
]
