"""Tests for transformation plugins and robustness service."""

from __future__ import annotations

import re
from copy import deepcopy

import pytest

from watermark_lab.attacks.transformations import (
    AttackConfig,
    DeletionPlugin,
    InsertionPlugin,
    ParaphraseStubPlugin,
    ShufflePlugin,
    SubstitutionPlugin,
    TransformationRegistry,
)
from watermark_lab.core.contracts import DetectionConfig, DetectionResult
from watermark_lab.run_engine.attack import (
    AttackAuthorizationError,
    AttackConfigurationError,
    RobustnessService,
)
from watermark_lab.run_engine.create import CreationService


def test_deletion_is_deterministic() -> None:
    plugin = DeletionPlugin()
    cfg = AttackConfig("deletion", severity=0.3, seed="abc")
    a = plugin.transform("one two three four five", cfg)
    b = plugin.transform("one two three four five", cfg)
    assert a["transformed_text"] == b["transformed_text"]
    assert a["tokens_deleted"] == b["tokens_deleted"]


def test_registry_unknown_attack_raises() -> None:
    registry = TransformationRegistry()
    with pytest.raises(ValueError, match="Unknown attack_id"):
        registry.transform("hello", AttackConfig("bad_attack"))


def test_substitution_changes_text() -> None:
    plugin = SubstitutionPlugin()
    cfg = AttackConfig("substitution", severity=1.0, seed="x")
    result = plugin.transform("alpha beta gamma", cfg)
    assert result["transformed_text"] != "alpha beta gamma"
    assert result["tokens_replaced"] == 3


def test_insertion_increases_length() -> None:
    plugin = InsertionPlugin()
    cfg = AttackConfig("insertion", severity=1.0, seed="y")
    result = plugin.transform("alpha beta gamma", cfg)
    assert len(result["transformed_text"]) > len("alpha beta gamma")
    assert result["tokens_inserted"] >= 1


@pytest.mark.parametrize(
    "plugin",
    [
        DeletionPlugin(),
        InsertionPlugin(),
        SubstitutionPlugin(),
        ShufflePlugin(),
        ParaphraseStubPlugin(),
    ],
    ids=["deletion", "insertion", "substitution", "shuffle", "paraphrase"],
)
def test_all_transformations_are_deterministic_and_zero_is_exact_identity(plugin) -> None:
    text = "Hello, world! Keep this suffix: (intact)."
    zero = AttackConfig(plugin.attack_id, severity=0.0, seed="fixed")
    assert plugin.transform(text, zero)["transformed_text"] == text

    full = AttackConfig(plugin.attack_id, severity=1.0, seed="fixed")
    first = plugin.transform(text, full)
    second = plugin.transform(text, full)
    assert first == second


@pytest.mark.parametrize(
    ("plugin", "expected_key", "expected_at_full_strength"),
    [
        (DeletionPlugin(), "tokens_deleted", 4),
        (SubstitutionPlugin(), "tokens_replaced", 4),
        (InsertionPlugin(), "tokens_inserted", 3),
        (ShufflePlugin(), "tokens_shuffled", 2),
        (ParaphraseStubPlugin(), "substitutions", 4),
    ],
    ids=["deletion", "substitution", "insertion", "shuffle", "paraphrase"],
)
def test_full_strength_boundary_report_is_consistent(
    plugin, expected_key: str, expected_at_full_strength: int
) -> None:
    """At severity=1.0 each plugin reports the maximum change it can apply."""
    text = "one, two; three! four?"
    result = plugin.transform(text, AttackConfig(plugin.attack_id, severity=1.0, seed="boundary"))
    assert result[expected_key] >= expected_at_full_strength


def test_deletion_only_removes_words_preserves_order_and_punctuation() -> None:
    """Deletion removes a subset of words but keeps the remaining ones in order."""
    text = "Alpha, beta; gamma... FINAL!"
    result = DeletionPlugin().transform(text, AttackConfig("deletion", severity=0.5, seed="order"))
    transformed = result["transformed_text"]
    remaining_words = [w for w in ["Alpha", "beta", "gamma", "FINAL"] if w in transformed]
    # Remaining words must appear in the same relative order as in the source.
    positions = [transformed.find(word) for word in remaining_words]
    assert sorted(positions) == positions
    assert all(char in transformed for char in ",;...!")


def _word_count(text: str) -> int:
    return len([t for t in re.findall(r"\w+|\W+", text, flags=re.UNICODE) if re.search(r"\w", t, flags=re.UNICODE)])


def test_substitution_only_replaces_words_preserves_count_and_structure() -> None:
    """Substitution replaces words without changing word count or punctuation."""
    text = "Alpha, beta; gamma... FINAL!"
    result = SubstitutionPlugin().transform(
        text, AttackConfig("substitution", severity=1.0, seed="replace")
    )
    transformed = result["transformed_text"]
    assert _word_count(transformed) == _word_count(text)
    assert transformed.endswith("!")
    assert "," in transformed and ";" in transformed and "..." in transformed


def test_shuffle_only_permutes_words_preserves_multiset() -> None:
    """Shuffle rearranges words but never changes or removes them."""
    text = "Alpha beta gamma delta"
    result = ShufflePlugin().transform(text, AttackConfig("shuffle", severity=1.0, seed="permute"))
    transformed = result["transformed_text"]
    assert sorted(transformed.split()) == sorted(text.split())
    assert result["tokens_shuffled"] > 0


def test_insertion_preserves_punctuation_and_complete_suffix() -> None:
    text = "Alpha, beta; gamma... FINAL!"
    result = InsertionPlugin().transform(
        text, AttackConfig("insertion", severity=0.5, seed="suffix")
    )
    transformed = result["transformed_text"]
    source_cursor = iter(transformed)
    assert all(character in source_cursor for character in text)
    assert transformed.endswith("FINAL!")
    assert result["tokens_inserted"] == 2


@pytest.mark.parametrize("severity", [-0.01, 1.01, float("inf"), float("nan")])
def test_attack_config_rejects_out_of_range_or_non_finite_severity(severity: float) -> None:
    with pytest.raises(ValueError, match="severity"):
        AttackConfig("deletion", severity=severity)


def test_transform_boundaries_report_full_strength_changes() -> None:
    text = "one, two; three! four?"
    assert DeletionPlugin().transform(
        text, AttackConfig("deletion", severity=1.0, seed="boundary")
    )["tokens_deleted"] == 4
    assert SubstitutionPlugin().transform(
        text, AttackConfig("substitution", severity=1.0, seed="boundary")
    )["tokens_replaced"] == 4
    assert InsertionPlugin().transform(
        text, AttackConfig("insertion", severity=1.0, seed="boundary")
    )["tokens_inserted"] == 3
    assert ShufflePlugin().transform(
        text, AttackConfig("shuffle", severity=1.0, seed="boundary")
    )["tokens_shuffled"] >= 2


def test_robustness_service_attack_survival() -> None:
    creation = CreationService()
    parent = creation.create(
        prompt="the quick brown fox jumps over the lazy dog",
        profile_id="mock",
        method_id="kgw",
        key_id="test-kgw-v1",
        config={"disclosure_recorded": True, "max_new_tokens": 80},
    )
    detector_config = DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=4.0,
        options={"gamma": 0.25, "context_width": 1},
    )
    service = RobustnessService()
    result = service.attack(
        parent_run=parent,
        attack_config=AttackConfig("shuffle", severity=1.0, seed="shuffle-1"),
        detector_config=detector_config,
    )
    assert result["parent_run_id"] == parent["run_id"]
    assert result["pre_attack"]["decision"] == "verified compatible watermark"
    assert "post_attack" in result
    assert isinstance(result["survived"], bool)


def test_robustness_service_sweep() -> None:
    creation = CreationService()
    parent = creation.create(
        prompt="a b c d e f g h i j",
        profile_id="mock",
        method_id="kgw",
        key_id="test-kgw-v1",
        config={"disclosure_recorded": True, "max_new_tokens": 80},
    )
    detector_config = DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=4.0,
        options={"gamma": 0.25, "context_width": 1},
    )
    service = RobustnessService()
    results = service.sweep(parent, "deletion", [0.1, 0.5, 0.9], detector_config)
    assert len(results) == 3
    assert all(r["attack_id"] == "deletion" for r in results)
    assert [r["severity"] for r in results] == [0.1, 0.5, 0.9]
    assert results[0]["pre_attack"]["decision"] == "verified compatible watermark"


def _detector_config(**changes) -> DetectionConfig:
    values = {
        "method_id": "kgw",
        "tokenizer_id": "gpt2",
        "key_id": "test-kgw-v1",
        "threshold": 4.0,
        "options": {
            "gamma": 0.25,
            "context_width": 1,
            "seeding_scheme": "simple_1",
            "hash_key": 15485863,
        },
    }
    values.update(changes)
    return DetectionConfig(**values)


def _synthetic_parent() -> dict:
    effective_config = {
        "profile_id": "mock",
        "key_id": "test-kgw-v1",
        "tokenizer_id": "gpt2",
        "max_new_tokens": 3,
        "gamma": 0.25,
        "delta": 2.0,
        "context_width": 1,
        "seeding_scheme": "simple_1",
        "hash_key": 15485863,
        "vocab_size": 50257,
        "detection_threshold": 4.0,
    }
    return {
        "run_id": "parent-1",
        "run_type": "create",
        "status": "completed",
        "method_id": "kgw",
        "key_id": "test-kgw-v1",
        "config": dict(effective_config),
        "governance": {
            "source_is_third_party": False,
            "project_owned": True,
            "attack_authorized": False,
        },
        "generation": {
            "text": "15496 50256 123",
            "method_id": "kgw",
            "key_id": "test-kgw-v1",
            "config": dict(effective_config),
            "model_profile": {"simulated": True},
        },
    }


@pytest.mark.parametrize(
    "detector_config",
    [
        _detector_config(method_id="other"),
        _detector_config(key_id="other-key"),
        _detector_config(tokenizer_id="other-tokenizer"),
        _detector_config(options={"gamma": 0.4, "context_width": 1}),
    ],
    ids=["method", "key", "tokenizer", "parameters"],
)
def test_attack_rejects_detector_mismatch(detector_config: DetectionConfig) -> None:
    with pytest.raises(AttackConfigurationError, match="does not match"):
        RobustnessService().attack(
            _synthetic_parent(), AttackConfig("deletion", 0.0), detector_config
        )


def test_attack_rejects_unauthorized_non_owned_parent() -> None:
    parent = _synthetic_parent()
    parent["generation"]["model_profile"]["simulated"] = False
    parent["governance"]["project_owned"] = False
    with pytest.raises(AttackAuthorizationError, match="synthetic, project-owned"):
        RobustnessService().attack(
            parent, AttackConfig("deletion", 0.0), _detector_config()
        )


def test_third_party_parent_requires_explicit_attack_authorization() -> None:
    parent = _synthetic_parent()
    parent["governance"]["source_is_third_party"] = True
    with pytest.raises(AttackAuthorizationError, match="explicit"):
        RobustnessService().attack(
            parent, AttackConfig("deletion", 0.0), _detector_config()
        )


def test_attack_preserves_missing_score_and_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubDetectionService:
        def detect(self, text: str, config: DetectionConfig) -> DetectionResult:
            return DetectionResult(
                method_id="kgw",
                compatibility="experimental",
                decision="inconclusive",
                score=None,
                p_value=None,
                token_count=0,
                estimated_fpr=None,
            )

    monkeypatch.setattr(
        "watermark_lab.run_engine.attack.DetectionService", StubDetectionService
    )
    result = RobustnessService().attack(
        deepcopy(_synthetic_parent()),
        AttackConfig("deletion", 0.0),
        _detector_config(),
    )
    assert result["pre_attack"]["score"] is None
    assert result["post_attack"]["score"] is None
    assert result["delta_score"] is None
