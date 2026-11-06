"""Robustness/attack run engine."""

from __future__ import annotations

from datetime import UTC
from typing import Any

from watermark_lab.attacks.transformations import AttackConfig, TransformationRegistry
from watermark_lab.core.contracts import DetectionConfig
from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.run_engine.intake import compute_sha256


class AttackAuthorizationError(ValueError):
    """Raised when the parent text is not eligible for controlled attacks."""


class AttackConfigurationError(ValueError):
    """Raised when parent, generator, and detector configurations do not match."""


_MATCHED_DETECTION_OPTIONS: dict[str, Any] = {
    "gamma": 0.25,
    "context_width": 1,
    "seeding_scheme": "simple_1",
    "hash_key": 15485863,
}


def _parent_authorization(parent_run: dict[str, Any]) -> dict[str, bool]:
    generation = parent_run.get("generation")
    if not isinstance(generation, dict):
        raise AttackAuthorizationError("Parent run must contain a generation record")
    model_profile = generation.get("model_profile", {})
    governance = parent_run.get("governance", {})
    if not isinstance(model_profile, dict) or not isinstance(governance, dict):
        raise AttackAuthorizationError("Parent ownership and authorization metadata are invalid")

    synthetic = model_profile.get("simulated") is True
    project_owned = governance.get("project_owned") is True
    explicitly_authorized = governance.get("attack_authorized") is True
    third_party_source = governance.get("source_is_third_party") is True

    if third_party_source and not explicitly_authorized:
        raise AttackAuthorizationError(
            "Third-party parent text requires explicit attack authorization"
        )
    if not (synthetic or project_owned or explicitly_authorized):
        raise AttackAuthorizationError(
            "Parent text must be synthetic, project-owned, or explicitly authorized"
        )
    return {
        "synthetic": synthetic,
        "project_owned": project_owned,
        "explicitly_authorized": explicitly_authorized,
    }


def _validate_parent_and_detector(
    parent_run: dict[str, Any], detector_config: DetectionConfig
) -> dict[str, bool]:
    if parent_run.get("run_type") != "create" or parent_run.get("status") != "completed":
        raise AttackAuthorizationError("Parent must be a completed create run")
    authorization = _parent_authorization(parent_run)

    generation = parent_run["generation"]
    parent_method = parent_run.get("method_id")
    generation_method = generation.get("method_id")
    if not parent_method or parent_method != generation_method:
        raise AttackConfigurationError("Parent and generation method_id values do not match")
    if detector_config.method_id != parent_method:
        raise AttackConfigurationError("Detector method_id does not match the parent generation")

    parent_key = parent_run.get("key_id")
    generation_key = generation.get("key_id")
    if not parent_key or parent_key != generation_key:
        raise AttackConfigurationError("Parent and generation key_id values do not match")
    if detector_config.key_id != parent_key:
        raise AttackConfigurationError("Detector key_id does not match the parent generation")

    parent_config = parent_run.get("config")
    generation_config = generation.get("config")
    if not isinstance(parent_config, dict) or not isinstance(generation_config, dict):
        raise AttackConfigurationError("Parent and generation configs must be recorded")
    for name, generation_value in generation_config.items():
        if parent_config.get(name) != generation_value:
            raise AttackConfigurationError(
                f"Parent config {name!r} does not match the effective generation config"
            )

    tokenizer_id = str(generation_config.get("tokenizer_id", "gpt2"))
    if detector_config.tokenizer_id != tokenizer_id:
        raise AttackConfigurationError("Detector tokenizer_id does not match the parent generation")
    detection_threshold = generation_config.get("detection_threshold", 4.0)
    if detector_config.threshold != detection_threshold:
        raise AttackConfigurationError(
            "Detector threshold does not match the parent generation verification threshold"
        )

    for name, default in _MATCHED_DETECTION_OPTIONS.items():
        generation_value = generation_config.get(name, default)
        detector_value = detector_config.options.get(name, default)
        if detector_value != generation_value:
            raise AttackConfigurationError(
                f"Detector option {name!r} does not match the parent generation"
            )
    return authorization


class RobustnessService:
    """Run controlled attacks against a parent generation and measure survival."""

    def __init__(self, registry: TransformationRegistry | None = None) -> None:
        self.registry = registry or TransformationRegistry()

    def attack(
        self,
        parent_run: dict[str, Any],
        attack_config: AttackConfig,
        detector_config: DetectionConfig,
    ) -> dict[str, Any]:
        authorization = _validate_parent_and_detector(parent_run, detector_config)
        original_text = parent_run["generation"]["text"]
        pre = DetectionService().detect(original_text, detector_config)
        transformed = self.registry.transform(original_text, attack_config)
        transformed_text = transformed["transformed_text"]
        post = DetectionService().detect(transformed_text, detector_config)
        survived = post.decision == "verified compatible watermark"
        run_id = _new_run_id()
        delta_score = (
            None if pre.score is None or post.score is None else post.score - pre.score
        )
        return {
            "run_id": run_id,
            "run_type": "transform",
            "parent_run_id": parent_run["run_id"],
            "method_id": parent_run["method_id"],
            "compatibility": "experimental",
            "attack_id": attack_config.attack_id,
            "severity": attack_config.severity,
            "seed": attack_config.seed,
            "authorization": authorization,
            "input_sha256": compute_sha256(original_text),
            "transformed_sha256": compute_sha256(transformed_text),
            "pre_attack": {
                "decision": pre.decision,
                "score": pre.score,
                "token_count": pre.token_count,
                "estimated_fpr": pre.estimated_fpr,
                "original_scorable_token_count": pre.original_scorable_token_count,
                "ignored_repeated_ngrams": pre.ignored_repeated_ngrams,
                "scoring_variant": pre.scoring_variant,
                "calibration_id": pre.calibration_id,
                "warnings": list(pre.warnings),
            },
            "post_attack": {
                "decision": post.decision,
                "score": post.score,
                "token_count": post.token_count,
                "estimated_fpr": post.estimated_fpr,
                "original_scorable_token_count": post.original_scorable_token_count,
                "ignored_repeated_ngrams": post.ignored_repeated_ngrams,
                "scoring_variant": post.scoring_variant,
                "calibration_id": post.calibration_id,
                "warnings": list(post.warnings),
            },
            "survived": survived,
            "delta_score": delta_score,
            "transformed_text": transformed_text,
            "software_version": "0.1.0",
            "git_commit": _git_commit(),
            "created_at": _now_iso(),
        }

    def sweep(
        self,
        parent_run: dict[str, Any],
        attack_id: str,
        severities: list[float],
        detector_config: DetectionConfig,
        seed: str = "sweep",
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for severity in severities:
            attack_config = AttackConfig(attack_id=attack_id, severity=severity, seed=seed)
            results.append(self.attack(parent_run, attack_config, detector_config))
        return results


def _new_run_id() -> str:
    import uuid

    return str(uuid.uuid4())


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    return None


__all__ = ["AttackAuthorizationError", "AttackConfigurationError", "RobustnessService"]
