"""Semantic checks that JSON Schema cannot express for calibration manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from watermark_lab.statistics.calibration import minimum_zero_event_sample_size


@dataclass(frozen=True, slots=True)
class CalibrationManifestIssue:
    """One fail-closed semantic defect in an otherwise structured manifest."""

    path: str
    message: str


class CalibrationManifestSemanticError(ValueError):
    """Raised when a calibration manifest makes a mathematically invalid claim."""

    def __init__(self, issues: tuple[CalibrationManifestIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(f"calibration manifest semantic validation failed: {detail}")


def _mapping(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def validate_calibration_manifest_semantics(
    manifest: dict[str, Any],
) -> tuple[CalibrationManifestIssue, ...]:
    """Return dynamic calibration errors after Draft 2020-12 validation.

    Structural validation remains the schema's job. This layer computes the
    exact minimum independent sample size for any zero-event confidence/FPR
    combination and checks cross-field scientific invariants.
    """

    issues: list[CalibrationManifestIssue] = []
    inference = _mapping(manifest.get("inference"))
    sampling = _mapping(manifest.get("sampling"))
    method = _mapping(manifest.get("method"))
    if inference is None or sampling is None or method is None:
        return (
            CalibrationManifestIssue(
                path="<root>",
                message="schema validation must supply method, sampling, and inference objects",
            ),
        )

    if inference.get("claim_basis") == "zero-event-upper-bound":
        target = inference.get("target_fpr")
        confidence = inference.get("confidence_level")
        planned = sampling.get("planned_independent_negatives")
        if (
            isinstance(target, (int, float))
            and not isinstance(target, bool)
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and isinstance(planned, int)
            and not isinstance(planned, bool)
        ):
            required = minimum_zero_event_sample_size(
                float(target),
                float(confidence),
            )
            if planned < required:
                issues.append(
                    CalibrationManifestIssue(
                        path="sampling/planned_independent_negatives",
                        message=(
                            f"{planned} is insufficient; exact zero-event design requires "
                            f"at least {required} independent units for upper bound "
                            f"{float(target):g} at confidence {float(confidence):g}"
                        ),
                    )
                )

    status = manifest.get("status")
    if status in {"frozen", "validated"} and method.get("parity_status") != "passed":
        issues.append(
            CalibrationManifestIssue(
                path="method/parity_status",
                message="a frozen or validated calibration requires passed author-code parity",
            )
        )
    if manifest.get("locale") == "zh-Hant-TW":
        fallback = manifest.get("locale_fallback_order")
        expected = ["zh-Hant-TW", "zh-TW", "zh-Hant", "en"]
        if fallback != expected:
            issues.append(
                CalibrationManifestIssue(
                    path="locale_fallback_order",
                    message="zh-Hant-TW calibration must use the canonical non-zh-CN fallback order",
                )
            )

    issues.extend(_validate_input_channel(manifest))
    return tuple(issues)


def _validate_input_channel(manifest: dict[str, Any]) -> list[CalibrationManifestIssue]:
    """Check invariants for a calibration built on OCR-derived text.

    An OCR reading is a reconstruction, so its null must be tied to the exact
    instrument that produced it. A null recorded against one engine profile,
    language, or degradation level says nothing about a reading produced by
    another, and silently reusing it would understate the false-positive rate in
    precisely the direction that manufactures false accusations.
    """

    issues: list[CalibrationManifestIssue] = []
    channel_block = _mapping(manifest.get("input_channel"))
    if channel_block is None:
        return issues
    if channel_block.get("channel") != "ocr":
        return issues

    ocr = _mapping(channel_block.get("ocr"))
    if ocr is None:
        issues.append(
            CalibrationManifestIssue(
                path="input_channel/ocr",
                message="an OCR channel requires the ocr profile block",
            )
        )
        return issues

    scoring = _mapping(manifest.get("scoring")) or {}
    minimum_tokens = scoring.get("minimum_scored_tokens")
    if (
        isinstance(minimum_tokens, int)
        and not isinstance(minimum_tokens, bool)
        and minimum_tokens < 30
    ):
        issues.append(
            CalibrationManifestIssue(
                path="scoring/minimum_scored_tokens",
                message=(
                    f"{minimum_tokens} is too few for an OCR calibration: recognition "
                    "error re-segments tokens, so short OCR readings carry far less "
                    "usable signal than short clean readings of the same length"
                ),
            )
        )

    degradation = _mapping(ocr.get("degradation")) or {}
    severity = degradation.get("severity")
    status = manifest.get("status")
    if (
        status in {"frozen", "validated"}
        and isinstance(severity, (int, float))
        and not isinstance(severity, bool)
        and float(severity) == 0.0
        and degradation.get("name") not in {"clean", None}
    ):
        issues.append(
            CalibrationManifestIssue(
                path="input_channel/ocr/degradation",
                message=(
                    f"severity 0 for degradation {degradation.get('name')!r} is the identity, "
                    "so this manifest describes a clean render rather than the named "
                    "degradation; record it as 'clean' or raise the severity"
                ),
            )
        )

    return issues


def ocr_channel_matches(
    manifest: dict[str, Any],
    *,
    engine_id: str,
    profile_fingerprint: str,
    language: str,
) -> tuple[bool, str]:
    """Return whether an OCR calibration applies to a specific reading.

    Presence of *some* OCR calibration is not enough. The null must have been
    measured on the same engine, the same pinned model bytes, and the same
    language, or it describes a different instrument.
    """

    channel_block = _mapping(manifest.get("input_channel"))
    if channel_block is None or channel_block.get("channel") != "ocr":
        return False, "the calibration is not an OCR calibration"

    ocr = _mapping(channel_block.get("ocr"))
    if ocr is None:
        return False, "the calibration declares an OCR channel but carries no OCR profile"

    if ocr.get("engine_id") != engine_id:
        return False, (
            f"calibration engine {ocr.get('engine_id')!r} does not match reading engine "
            f"{engine_id!r}"
        )
    if ocr.get("profile_fingerprint") != profile_fingerprint:
        return False, (
            "calibration profile fingerprint does not match the reading's; the engine "
            "model bytes or parameters differ, so this is a different instrument"
        )
    if ocr.get("language") != language:
        return False, (
            f"calibration language {ocr.get('language')!r} does not match reading language "
            f"{language!r}; OCR error profiles are language-specific"
        )
    return True, "the OCR calibration matches this reading's instrument"


def assert_calibration_manifest_semantics(manifest: dict[str, Any]) -> None:
    """Raise with all semantic issues when a manifest cannot support its claim."""

    issues = validate_calibration_manifest_semantics(manifest)
    if issues:
        raise CalibrationManifestSemanticError(issues)


__all__ = [
    "CalibrationManifestIssue",
    "CalibrationManifestSemanticError",
    "assert_calibration_manifest_semantics",
    "validate_calibration_manifest_semantics",
]
