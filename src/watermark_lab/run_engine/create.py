"""Creation service for watermarked text generation workflows."""

from __future__ import annotations

import json
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from watermark_lab.core.contracts import DetectionConfig, DetectionResult
from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.run_engine.intake import compute_sha256, normalize_text
from watermark_lab.watermarkers.kgw_generator import KgwGenerationConfig, KgwGenerator

#: Default governance values used when no config file is loaded.
GOVERNANCE_DEFAULTS: dict[str, Any] = {
    "require_disclosure": True,
    "allow_third_party_documents": False,
    "allow_existing_text_rewatermarking": False,
}

#: Minimum scored-token count for a conclusive self-verification result.
_MIN_CONCLUSIVE_TOKENS = 30


class GovernanceError(Exception):
    """Raised when a governance check blocks generation."""


class SelfVerificationError(Exception):
    """Raised when generated text fails keyed self-verification."""


def _load_governance(config_path: Path | None = None) -> dict[str, Any]:
    """Load governance flags from ``config/example.toml`` or return defaults."""
    if config_path is None:
        config_path = Path("config/example.toml")
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[3] / config_path

    if not config_path.exists():
        return dict(GOVERNANCE_DEFAULTS)

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return dict(GOVERNANCE_DEFAULTS)

    governance = data.get("governance", {})
    return {
        "require_disclosure": governance.get(
            "require_disclosure", GOVERNANCE_DEFAULTS["require_disclosure"]
        ),
        "allow_third_party_documents": governance.get(
            "allow_third_party_documents", GOVERNANCE_DEFAULTS["allow_third_party_documents"]
        ),
        "allow_existing_text_rewatermarking": governance.get(
            "allow_existing_text_rewatermarking",
            GOVERNANCE_DEFAULTS["allow_existing_text_rewatermarking"],
        ),
    }


def _check_governance(config: dict[str, Any], governance: dict[str, Any]) -> dict[str, Any]:
    """Run governance checks and return a recorded decision set."""
    checks: dict[str, Any] = {
        "disclosure_recorded": bool(config.get("disclosure_recorded", False)),
        "source_is_third_party": bool(config.get("source_is_third_party", False)),
        "rewriting_existing_text": bool(config.get("rewriting_existing_text", False)),
    }

    if governance["require_disclosure"] and not checks["disclosure_recorded"]:
        raise GovernanceError("Disclosure is required but was not recorded")

    if checks["source_is_third_party"] and not governance["allow_third_party_documents"]:
        raise GovernanceError("Third-party documents are not allowed by governance")

    if checks["rewriting_existing_text"] and not governance["allow_existing_text_rewatermarking"]:
        raise GovernanceError("Rewatermarking existing text is not allowed by governance")

    return checks


def _default_run_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` if git is unavailable."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _self_verify(
    text: str,
    method_id: str,
    config: KgwGenerationConfig,
) -> DetectionResult:
    """Run the keyed detector using the validated effective generation config."""
    detection_config = DetectionConfig(
        method_id=method_id,
        tokenizer_id=config.tokenizer_id,
        key_id=config.key_id,
        threshold=config.detection_threshold,
        options=config.detection_options(),
    )
    return DetectionService().detect(normalize_text(text), detection_config)


class CreationService:
    """Run-engine service for watermark generation workflows."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path
        self.governance = _load_governance(config_path)

    def create(
        self,
        prompt: str,
        profile_id: str,
        method_id: str,
        key_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate and self-verify watermarked text.

        Args:
            prompt: Input prompt text.
            profile_id: Model profile identifier.
            method_id: Watermarker method identifier.
            key_id: Key identifier (references ``TEST_KEY_VAULT`` only).
            config: Generation options and governance flags.

        Returns:
            Run result containing the generation record, self-verification result,
            governance checks, and run metadata. If self-verification fails, the
            ``status`` field is ``failed`` and the text is not accepted.
        """
        if method_id != "kgw":
            raise ValueError(f"Unsupported method_id: {method_id!r}")

        governance_checks = _check_governance(config, self.governance)

        requested_generator_config = {
            "profile_id": profile_id,
            "key_id": key_id,
            "tokenizer_id": str(config.get("tokenizer_id", "gpt2")),
            "max_new_tokens": int(config.get("max_new_tokens", 80)),
            "gamma": float(config.get("gamma", 0.25)),
            "delta": float(config.get("delta", 2.0)),
            "context_width": int(config.get("context_width", 1)),
            "seeding_scheme": str(config.get("seeding_scheme", "simple_1")),
            "hash_key": int(config.get("hash_key", 15485863)),
            "detection_threshold": float(config.get("detection_threshold", 4.0)),
        }
        generator = KgwGenerator()
        effective_config = KgwGenerationConfig.from_mapping(
            requested_generator_config, vocab_size=generator.vocab_size
        )
        generation = generator.generate(prompt, effective_config.to_dict())

        verification = _self_verify(generation["text"], method_id, effective_config)
        accepted = (
            verification.decision == "verified compatible watermark"
            and verification.token_count >= _MIN_CONCLUSIVE_TOKENS
        )

        if not accepted:
            raise SelfVerificationError(
                "Self-verification did not detect the watermark or the text was too short"
            )

        run_id = _default_run_id()
        governance_checks.update(
            {
                "synthetic_output": bool(generation["model_profile"]["simulated"]),
                "project_owned": bool(
                    config.get("project_owned", generation["model_profile"]["simulated"])
                ),
                "attack_authorized": bool(config.get("attack_authorized", False)),
            }
        )
        generator_config = effective_config.to_dict()
        result: dict[str, Any] = {
            "run_id": run_id,
            "run_type": "create",
            "method_id": method_id,
            "compatibility": "experimental",
            "input_sha256": compute_sha256(prompt),
            "normalized_sha256": compute_sha256(normalize_text(prompt)),
            "model_profile": generation["model_profile"],
            "key_id": key_id,
            "config": generator_config,
            "generation": generation,
            "self_verification": {
                "decision": verification.decision,
                "score": verification.score,
                "p_value": verification.p_value,
                "token_count": verification.token_count,
                "estimated_fpr": verification.estimated_fpr,
                "original_scorable_token_count": verification.original_scorable_token_count,
                "ignored_repeated_ngrams": verification.ignored_repeated_ngrams,
                "scoring_variant": verification.scoring_variant,
                "calibration_id": verification.calibration_id,
                "warnings": list(verification.warnings),
            },
            "governance": governance_checks,
            "status": "completed",
            "created_at": _now_iso(),
            "software_version": "0.1.0",
            "git_commit": _git_commit(),
            "artifacts": [],
        }

        return result

    def create_manifest(self, create_result: dict[str, Any]) -> dict[str, Any]:
        """Return a run manifest validated against ``schemas/run-manifest.schema.json``."""
        return {
            "run_id": create_result["run_id"],
            "run_type": create_result["run_type"],
            "method_id": create_result["method_id"],
            "compatibility": create_result["compatibility"],
            "input_sha256": create_result["input_sha256"],
            "normalized_sha256": create_result["normalized_sha256"],
            "created_at": create_result["created_at"],
            "software_version": create_result["software_version"],
            "git_commit": create_result.get("git_commit"),
            "key_id": create_result.get("key_id"),
            "config": create_result["config"],
            "artifacts": create_result.get("artifacts", []),
        }


def create_and_manifest(
    prompt: str,
    profile_id: str,
    method_id: str,
    key_id: str,
    config: dict[str, Any],
    config_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convenience wrapper that runs ``CreationService.create`` and emits a manifest."""
    service = CreationService(config_path=config_path)
    result = service.create(prompt, profile_id, method_id, key_id, config)
    return result, service.create_manifest(result)


def write_artifacts(result: dict[str, Any], manifest: dict[str, Any], base_dir: Path) -> None:
    """Write generation record, verification, and manifest JSON files to ``base_dir``."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "generation.json").write_text(
        json.dumps(result["generation"], indent=2), encoding="utf-8"
    )
    (base_dir / "verification.json").write_text(
        json.dumps(result["self_verification"], indent=2), encoding="utf-8"
    )
    (base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
