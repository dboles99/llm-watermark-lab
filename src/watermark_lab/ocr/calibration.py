"""Measure an OCR-specific null distribution and emit a calibration manifest.

A clean-text null does not transfer to OCR-derived text. Recognition changes
which tokens are present and how words are segmented, so the detector's null
distribution on reconstructed text is a different distribution — and using a
clean-text threshold on a scan would understate the false-positive rate in the
one direction that matters, the direction that produces false accusations.

This module builds the null the OCR detection gate requires: score unwatermarked
text that has been through the *same* render-degrade-recognise pipeline as the
readings the threshold will be applied to, then take the empirical quantile.

Three constraints from `docs/research/CALIBRATION_STATISTICS.md` are enforced
rather than assumed:

**Clusters, not documents.** Documents from one cluster are dependent
replicates. Interval arithmetic uses cluster counts. A cluster counts as a false
positive if *any* of its documents exceeds the threshold, which is the
conservative reading and the one that matches how a scan would actually be
adjudicated.

**Strata, not pools.** The null is measured per (engine, language, degradation),
because each combination is a different instrument reading different text. A
manifest records exactly one stratum; pooling them would hide the variation the
whole exercise exists to expose.

**Attainability, stated.** With ``n`` clusters the tightest one-sided 95% bound
after zero events is ``1 - 0.05**(1/n)``. A pilot of 40 clusters bounds the FPR
near 7.2%, not 0.1%, and the manifest says so in its own fields rather than
leaving a reader to infer it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from watermark_lab.ocr.degrade import Degradation, apply_degradation
from watermark_lab.ocr.render import RenderSettings, build_render_settings, render_text_to_image
from watermark_lab.statistics.calibration import (
    clopper_pearson_upper_bound,
    minimum_zero_event_sample_size,
)
from watermark_lab.statistics.empirical_null import empirical_threshold

MINIMUM_PILOT_CLUSTERS = 20

PILOT_STATUS_NOTE = (
    "Pilot calibration on synthetic negatives. Synthetic text is explicitly "
    "non-scientific: it validates the pipeline and the threshold arithmetic, but it "
    "cannot support a claim about real human text. Status stays 'draft' until measured "
    "on a licensed corpus."
)


class OcrCalibrationError(ValueError):
    """Raised when an OCR calibration cannot be computed or is not attainable."""


@dataclass(frozen=True, slots=True)
class ClusterScore:
    """Scores for every document in one independent cluster."""

    cluster_id: str
    scores: tuple[float, ...]
    dropped: int = 0

    @property
    def max_score(self) -> float:
        return max(self.scores) if self.scores else float("-inf")

    @property
    def scored(self) -> int:
        return len(self.scores)


@dataclass(frozen=True, slots=True)
class OcrNullResult:
    """An OCR null distribution measured over independent clusters."""

    stratum: dict[str, Any]
    clusters: tuple[ClusterScore, ...]
    threshold: float
    target_fpr: float
    confidence_level: float
    false_positive_clusters: int
    attainable_upper_bound: float
    dropped_documents: int
    scored_documents: int
    warnings: tuple[str, ...] = field(default=())

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    @property
    def observed_fpr(self) -> float:
        return (
            self.false_positive_clusters / self.cluster_count if self.cluster_count else 0.0
        )

    @property
    def target_attainable(self) -> bool:
        """Whether the cluster count can evidence the requested target at all."""

        return self.attainable_upper_bound <= self.target_fpr

    def as_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "cluster_count": self.cluster_count,
            "scored_documents": self.scored_documents,
            "dropped_documents": self.dropped_documents,
            "threshold": self.threshold,
            "target_fpr": self.target_fpr,
            "confidence_level": self.confidence_level,
            "false_positive_clusters": self.false_positive_clusters,
            "observed_fpr": self.observed_fpr,
            "attainable_upper_bound": self.attainable_upper_bound,
            "target_attainable": self.target_attainable,
            "cluster_scores": [
                {
                    "cluster_id": cluster.cluster_id,
                    "max_score": cluster.max_score if cluster.scores else None,
                    "scored": cluster.scored,
                    "dropped": cluster.dropped,
                }
                for cluster in self.clusters
            ],
            "warnings": list(self.warnings),
        }


def measure_ocr_null(
    documents: Sequence[Any],
    *,
    engine: Any,
    degradation: Degradation,
    scorer: Callable[[str], float | None],
    language: str,
    target_fpr: float = 0.001,
    confidence_level: float = 0.95,
    coverage_floor: float = 0.90,
    render_settings: RenderSettings | None = None,
    font_size: int = 30,
) -> OcrNullResult:
    """Score unwatermarked documents through the OCR pipeline and derive a threshold.

    ``documents`` must expose ``text`` and ``cluster_id``. ``scorer`` returns the
    detector statistic for recovered text, or ``None`` when the text is too short
    to score.

    Cells where the recogniser dropped whole regions are excluded rather than
    scored: a detection failure is not a null observation, and counting it as one
    would bias the threshold downward and the apparent false-positive rate with it.
    """

    if not documents:
        raise OcrCalibrationError("at least one document is required.")

    grouped: dict[str, list[str]] = {}
    for document in documents:
        grouped.setdefault(document.cluster_id, []).append(document.text)

    if len(grouped) < MINIMUM_PILOT_CLUSTERS:
        raise OcrCalibrationError(
            f"{len(grouped)} clusters is below the pilot floor of {MINIMUM_PILOT_CLUSTERS}. "
            "Independent units are clusters, not documents, so adding documents to the same "
            "clusters does not help."
        )

    warnings: list[str] = []
    clusters: list[ClusterScore] = []
    total_dropped = 0
    total_scored = 0

    for cluster_id in sorted(grouped):
        scores: list[float] = []
        dropped = 0
        for text in grouped[cluster_id]:
            settings = render_settings or build_render_settings(
                text, locale=language, font_size=font_size
            )
            page = render_text_to_image(text, settings)
            recovered = engine.recognize(apply_degradation(page, degradation)).text

            coverage = len(recovered) / len(text) if text else 1.0
            if coverage < coverage_floor:
                dropped += 1
                continue
            score = scorer(recovered)
            if score is None:
                dropped += 1
                continue
            scores.append(float(score))

        total_dropped += dropped
        total_scored += len(scores)
        clusters.append(ClusterScore(cluster_id=cluster_id, scores=tuple(scores), dropped=dropped))

    usable = [cluster for cluster in clusters if cluster.scores]
    if not usable:
        raise OcrCalibrationError(
            "no cluster produced a scorable reading; the degradation may be too severe or "
            "the recogniser may be dropping every page"
        )
    if len(usable) < len(clusters):
        warnings.append(
            f"{len(clusters) - len(usable)} of {len(clusters)} clusters produced no scorable "
            "reading and are excluded from the null"
        )

    # One value per independent unit: the cluster's worst-case (highest) score.
    cluster_maxima = [cluster.max_score for cluster in usable]
    threshold_result = empirical_threshold(
        cluster_maxima,
        alpha=target_fpr if target_fpr * len(usable) >= 1 else 1.0 / len(usable),
        corpus_id=f"ocr-null-{language}-{degradation.name}-{degradation.severity}",
        confidence_level=confidence_level,
    )
    threshold = threshold_result.threshold
    false_positive_clusters = sum(1 for value in cluster_maxima if value > threshold)

    attainable = clopper_pearson_upper_bound(
        false_positive_clusters, len(usable), confidence_level
    )
    if attainable > target_fpr:
        warnings.append(
            f"{len(usable)} clusters cannot evidence a {target_fpr:g} FPR: the tightest "
            f"attainable one-sided {confidence_level:g} bound here is {attainable:.4g}"
        )

    return OcrNullResult(
        stratum={
            "engine_id": engine.engine_id,
            "engine_version": engine.profile.engine_version,
            "profile_fingerprint": engine.profile.aggregate_sha256,
            "language": language,
            "degradation": degradation.as_dict(),
        },
        clusters=tuple(clusters),
        threshold=threshold,
        target_fpr=target_fpr,
        confidence_level=confidence_level,
        false_positive_clusters=false_positive_clusters,
        attainable_upper_bound=attainable,
        dropped_documents=total_dropped,
        scored_documents=total_scored,
        warnings=tuple(warnings),
    )


def build_calibration_manifest(
    null: OcrNullResult,
    *,
    calibration_id: str,
    locale: str,
    method: dict[str, Any],
    tokenizer: dict[str, Any],
    partitions: list[dict[str, Any]],
    confidence_floor: float = 0.80,
    coverage_floor: float = 0.90,
    minimum_scored_tokens: int = 50,
    gamma: float = 0.25,
    context_width: int = 1,
    seeding_scheme: str = "lefthash",
    status: str = "draft",
) -> dict[str, Any]:
    """Assemble a manifest matching `schemas/calibration-manifest.schema.json`.

    Status defaults to ``draft``. A pilot on synthetic negatives must not be
    frozen: freezing asserts the threshold is usable for decisions, and a
    threshold derived from text nobody wrote is not.
    """

    if not calibration_id.startswith("cal_"):
        raise OcrCalibrationError("calibration_id must start with 'cal_'.")
    if locale not in ("en", "zh-Hant-TW"):
        raise OcrCalibrationError("locale must be 'en' or 'zh-Hant-TW'.")

    # `planned_independent_negatives` is the study *design*, not what this run
    # achieved. The schema enforces a floor on it because a calibration whose
    # plan was never large enough cannot support its target however it turns
    # out. The achieved cluster count and the shortfall live in the null
    # measurement artifact written alongside.
    planned_negatives = minimum_zero_event_sample_size(
        null.target_fpr, null.confidence_level
    )
    if status in {"frozen", "validated"} and not null.target_attainable:
        raise OcrCalibrationError(
            f"cannot mark this calibration {status!r}: it measured {null.cluster_count} "
            f"independent clusters, which bounds the FPR at "
            f"{null.attainable_upper_bound:.4g}, short of the {null.target_fpr:g} target "
            f"that needs {planned_negatives} clusters. Keep it 'draft' or measure more."
        )

    stratum = null.stratum
    return {
        "$schema": "../schemas/calibration-manifest.schema.json",
        "schema_version": "1.0.0",
        "calibration_id": calibration_id,
        "status": status,
        "method": method,
        "tokenizer": tokenizer,
        "locale": locale,
        "locale_fallback_order": ["zh-Hant-TW", "zh-TW", "zh-Hant", "en"],
        "input_channel": {
            "channel": "ocr",
            "ocr": {
                "engine_id": stratum["engine_id"],
                "engine_version": stratum["engine_version"],
                "profile_fingerprint": stratum["profile_fingerprint"],
                "language": stratum["language"],
                "confidence_floor": confidence_floor,
                "coverage_floor": coverage_floor,
                "degradation": {
                    "name": stratum["degradation"]["name"],
                    "severity": stratum["degradation"]["severity"],
                    "seed": stratum["degradation"]["seed"],
                },
            },
        },
        "partitions": partitions,
        "sampling": {
            "independent_unit": "author-document-cluster",
            "planned_independent_negatives": planned_negatives,
            "planned_independent_positives": planned_negatives,
            "cluster_fields": ["document_id"],
            "stopping_rule": (
                "fixed sample: every cluster in the calibration partition is scored once; "
                "no interim looks and no threshold selection on this partition"
            ),
        },
        "scoring": {
            "gamma": gamma,
            "context_width": context_width,
            "seeding_scheme": seeding_scheme,
            "repeated_ngram_policy": "first-full-ngram-only",
            "minimum_scored_tokens": minimum_scored_tokens,
            "decision_threshold": null.threshold,
        },
        "inference": {
            "target_fpr": null.target_fpr,
            "confidence_level": null.confidence_level,
            "false_positive_interval": "clopper-pearson-exact-one-sided-upper",
            "true_positive_interval": "clopper-pearson-exact-two-sided",
            "claim_basis": (
                "zero-event-upper-bound"
                if null.false_positive_clusters == 0
                else "estimated-rate"
            ),
            "multiplicity_family_id": (
                f"ocr-null-{stratum['engine_id']}-{stratum['language']}-"
                f"{stratum['degradation']['name']}"
            ),
            "multiplicity_policy": "single-predeclared-test",
        },
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def write_calibration_artifact(
    null: OcrNullResult,
    manifest: dict[str, Any],
    directory: Path,
    *,
    note: str = PILOT_STATUS_NOTE,
) -> dict[str, Path]:
    """Write the manifest and its supporting null measurement side by side."""

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "calibration-manifest.json"
    null_path = directory / "null-measurement.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload = null.as_dict()
    payload["note"] = note
    payload["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    null_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest_path, "null": null_path}


__all__ = [
    "MINIMUM_PILOT_CLUSTERS",
    "PILOT_STATUS_NOTE",
    "ClusterScore",
    "OcrCalibrationError",
    "OcrNullResult",
    "build_calibration_manifest",
    "measure_ocr_null",
    "write_calibration_artifact",
]
