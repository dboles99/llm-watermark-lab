"""Unit tests for the watermark creation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from watermark_lab.core.contracts import DetectionConfig
from watermark_lab.run_engine.create import (
    CreationService,
    GovernanceError,
    SelfVerificationError,
    _load_governance,
)
from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.watermarkers.kgw_generator import KgwGenerator


def test_kgw_generator_is_deterministic() -> None:
    """The same prompt and config must produce identical output."""
    generator = KgwGenerator()
    config = {"profile_id": "mock", "key_id": "test-kgw-v1", "max_new_tokens": 50}
    result_a = generator.generate("the quick brown fox", config)
    result_b = generator.generate("the quick brown fox", config)
    assert result_a["text"] == result_b["text"]
    assert result_a["tokens"] == result_b["tokens"]
    assert result_a["response_sha256"] == result_b["response_sha256"]


@pytest.mark.parametrize(
    ("parameter", "non_default"),
    [("gamma", 0.4), ("delta", 1.0)],
)
def test_kgw_generator_output_is_parameter_sensitive(
    parameter: str, non_default: float
) -> None:
    """Non-default generation parameters must affect the sampled token sequence."""
    generator = KgwGenerator()
    config = {"profile_id": "mock", "key_id": "test-kgw-v1", "max_new_tokens": 50}
    baseline = generator.generate("parameter sensitivity", config)
    changed = generator.generate(
        "parameter sensitivity", {**config, parameter: non_default}
    )
    assert changed["tokens"] != baseline["tokens"]
    assert changed["config"][parameter] == non_default


def test_kgw_generator_records_hashes_and_profile() -> None:
    """A generation record carries prompt/response hashes and a simulated profile."""
    generator = KgwGenerator()
    config = {"profile_id": "mock", "key_id": "test-kgw-v1", "max_new_tokens": 50}
    result = generator.generate("hello world", config)
    assert result["prompt_sha256"] is not None and len(result["prompt_sha256"]) == 64
    assert result["response_sha256"] is not None and len(result["response_sha256"]) == 64
    assert result["model_profile"]["profile_id"] == "mock"
    assert result["model_profile"]["provider"] == "mock_local"
    assert result["model_profile"]["simulated"] is True
    assert result["method_id"] == "kgw"
    assert result["key_id"] == "test-kgw-v1"


def test_kgw_generator_self_verifies() -> None:
    """Generated text is detectable with the keyed detector."""
    generator = KgwGenerator()
    config = {"profile_id": "mock", "key_id": "test-kgw-v1", "max_new_tokens": 80}
    generation = generator.generate("self verify test", config)
    detection_config = DetectionConfig(
        method_id="kgw",
        tokenizer_id="gpt2",
        key_id="test-kgw-v1",
        threshold=4.0,
        options={"gamma": 0.25, "context_width": 1},
    )
    result = DetectionService().detect(generation["text"], detection_config)
    assert result.decision == "verified compatible watermark"
    assert result.token_count >= 30


@pytest.mark.parametrize(
    "parameter",
    ["gamma", "delta", "context_width", "max_new_tokens", "prompt"],
)
def test_kgw_generator_parameter_reaches_every_token(parameter: str) -> None:
    """Each listed parameter must propagate to the generated token sequence."""
    generator = KgwGenerator()
    base_config: dict[str, Any] = {
        "profile_id": "mock",
        "key_id": "test-kgw-v1",
        "max_new_tokens": 80,
    }
    base_prompt = "parameter sensitivity"

    if parameter == "prompt":
        base = generator.generate(base_prompt, base_config)
        changed = generator.generate("different prompt reaches tokens", base_config)
        assert base["prompt_sha256"] != changed["prompt_sha256"]
    elif parameter == "context_width":
        # simple_1 only supports context_width=1; verify the parameter is
        # validated and therefore reaches the generation path.
        with pytest.raises(ValueError, match="simple_1 requires context_width=1"):
            generator.generate(base_prompt, {**base_config, "context_width": 2})
        return
    elif parameter == "max_new_tokens":
        base = generator.generate(base_prompt, base_config)
        changed = generator.generate(
            base_prompt, {**base_config, "max_new_tokens": 40}
        )
    else:
        base = generator.generate(base_prompt, base_config)
        non_default: Any = {"gamma": 0.4, "delta": 1.0}[parameter]
        changed = generator.generate(base_prompt, {**base_config, parameter: non_default})

    assert changed["tokens"] != base["tokens"]
    if parameter in base_config:
        assert changed["config"][parameter] != base["config"][parameter]


def test_kgw_generator_prompt_is_propagated_to_mock_provider() -> None:
    """The same prompt produces identical output; a different prompt differs."""
    generator = KgwGenerator()
    config = {"profile_id": "mock", "key_id": "test-kgw-v1", "max_new_tokens": 80}
    a = generator.generate("prompt seeding one", config)
    b = generator.generate("prompt seeding one", config)
    c = generator.generate("prompt seeding two", config)
    assert a["tokens"] == b["tokens"]
    assert a["tokens"] != c["tokens"]
    assert a["prompt_sha256"] != c["prompt_sha256"]


def test_creation_service_rejects_without_disclosure(tmp_path: None) -> None:
    """Governance default requires disclosure."""
    service = CreationService()
    with pytest.raises(GovernanceError, match="Disclosure is required"):
        service.create("hello", "mock", "kgw", "test-kgw-v1", {})


def test_creation_service_rejects_third_party_documents() -> None:
    """Third-party documents are disallowed by default governance."""
    service = CreationService()
    config = {"disclosure_recorded": True, "source_is_third_party": True}
    with pytest.raises(GovernanceError, match="Third-party documents are not allowed"):
        service.create("hello", "mock", "kgw", "test-kgw-v1", config)


def test_creation_service_rejects_rewatermarking() -> None:
    """Existing-text rewatermarking is disallowed by default governance."""
    service = CreationService()
    config = {"disclosure_recorded": True, "rewriting_existing_text": True}
    with pytest.raises(GovernanceError, match="Rewatermarking existing text is not allowed"):
        service.create("hello", "mock", "kgw", "test-kgw-v1", config)


def test_creation_service_rejects_short_unverifiable_text() -> None:
    """A generation with too few tokens must fail self-verification as inconclusive."""
    service = CreationService()
    config = {"disclosure_recorded": True, "max_new_tokens": 5}
    with pytest.raises(SelfVerificationError, match="too short"):
        service.create("hello world", "mock", "kgw", "test-kgw-v1", config)


def test_creation_service_returns_generation_result() -> None:
    """A valid request produces a completed result with required fields."""
    service = CreationService()
    config = {"disclosure_recorded": True}
    result = service.create("hello world", "mock", "kgw", "test-kgw-v1", config)
    assert result["status"] == "completed"
    assert result["input_sha256"] is not None and len(result["input_sha256"]) == 64
    assert result["generation"]["prompt_sha256"] == result["input_sha256"]
    assert result["generation"]["response_sha256"] is not None
    assert result["self_verification"]["decision"] == "verified compatible watermark"


def test_creation_service_self_verifies_with_non_default_effective_config() -> None:
    """Self-verification consumes the exact effective KGW detector parameters."""
    service = CreationService()
    config = {
        "disclosure_recorded": True,
        "gamma": 0.4,
        "delta": 3.0,
        "context_width": 1,
        "max_new_tokens": 80,
        "detection_threshold": 5.0,
    }
    result = service.create("non-default config", "mock", "kgw", "test-kgw-v1", config)
    assert result["status"] == "completed"
    assert result["config"] == result["generation"]["config"]
    assert result["config"]["gamma"] == 0.4
    assert result["config"]["context_width"] == 1
    assert result["config"]["detection_threshold"] == 5.0
    assert result["self_verification"]["decision"] == "verified compatible watermark"


def test_creation_service_rejects_simple_one_with_wider_context() -> None:
    service = CreationService()
    config = {
        "disclosure_recorded": True,
        "context_width": 2,
    }

    with pytest.raises(ValueError, match="simple_1 requires context_width=1"):
        service.create("invalid context", "mock", "kgw", "test-kgw-v1", config)


def test_creation_service_manifest_is_valid() -> None:
    """The run manifest contains the required schema fields."""
    service = CreationService()
    config = {"disclosure_recorded": True}
    result = service.create("hello world", "mock", "kgw", "test-kgw-v1", config)
    manifest = service.create_manifest(result)
    assert manifest["run_id"] == result["run_id"]
    assert manifest["run_type"] == "create"
    assert manifest["method_id"] == "kgw"
    assert manifest["input_sha256"] == result["input_sha256"]
    assert manifest["created_at"] is not None
    assert manifest["software_version"] == "0.1.0"
    assert "config" in manifest


def test_load_governance_defaults() -> None:
    """Missing config file falls back to default governance."""
    governance = _load_governance(Path("/nonexistent/config.toml"))
    assert governance["require_disclosure"] is True
    assert governance["allow_third_party_documents"] is False
    assert governance["allow_existing_text_rewatermarking"] is False
