"""PolyMark: multi-provider, multi-key, multi-epoch demixing skeleton.

Infers zero, one, or several compatible watermark processes within a document
and assigns localized spans to provider adapters, tenant pseudonyms, or key
epochs.  Models overlap and collision rather than taking a single maximum
detector score.

Scientific status: skeleton.  A real demixing model requires a structured
sparse or Bayesian formulation, a scan-aware null, collision identifiability
characterization, and a mixed-source benchmark in en and zh-Hant-TW.  This
module provides the data contracts and a placeholder scan-based estimator
so the pipeline can run end-to-end without claiming scientific validity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal


class PolymarkDemixingError(ValueError):
    """Raised when demixing parameters or inputs fail closed."""


@dataclass(frozen=True, slots=True)
class PolymarkProcess:
    """One hypothesized watermark process in a mixed document."""

    process_id: str
    provider_label: str
    key_epoch: str
    gamma: float
    threshold: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarkSpan:
    """A localized span assigned to one process."""

    process_id: str
    start_token: int
    end_token: int
    z_score: float
    p_value: float
    overlap: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarkDemixingResult:
    """Outcome of a demixing pass over one document."""

    scientific_status: Literal["skeleton-uncalibrated"]
    process_count: int
    detected_processes: tuple[PolymarkProcess, ...]
    spans: tuple[PolymarkSpan, ...]
    overlap_count: int
    collision_count: int
    family_wise_fpr_estimate: float | None
    abstained: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "scientific_status": self.scientific_status,
            "process_count": self.process_count,
            "detected_processes": [p.as_dict() for p in self.detected_processes],
            "spans": [s.as_dict() for s in self.spans],
            "overlap_count": self.overlap_count,
            "collision_count": self.collision_count,
            "family_wise_fpr_estimate": self.family_wise_fpr_estimate,
            "abstained": self.abstained,
            "warnings": list(self.warnings),
        }


class PolymarkDemixer:
    """Skeleton scan-based multi-process demixer.

    The prototype scans token spans with each registered process detector and
    reports significant spans.  It does not implement a structured sparse or
    Bayesian model yet; it abstains when no process reaches threshold and
    reports overlaps honestly.

    A scientific demixer requires:
    - a scan-aware null distribution (not a single-doc null);
    - family-wise error control across processes, keys, and scans;
    - collision identifiability characterization;
    - a mixed-source benchmark in en and zh-Hant-TW; and
    - T06 calibrated per-process detectors.
    """

    def __init__(self, processes: Sequence[PolymarkProcess]) -> None:
        if not processes:
            raise PolymarkDemixingError("At least one process must be registered.")
        ids = [p.process_id for p in processes]
        if len(ids) != len(set(ids)):
            raise PolymarkDemixingError("Process IDs must be unique.")
        for process in processes:
            if not 0.0 < process.gamma < 1.0:
                raise PolymarkDemixingError("gamma must be in (0, 1) for every process")
            if not 0.0 < process.threshold:
                raise PolymarkDemixingError("threshold must be positive for every process")
        self._processes = tuple(processes)

    @property
    def processes(self) -> tuple[PolymarkProcess, ...]:
        return self._processes

    def demix(
        self,
        span_scores: Sequence[tuple[str, int, int, float, float]],
    ) -> PolymarkDemixingResult:
        """Demix a document from pre-computed per-process span scores.

        ``span_scores`` is a sequence of (process_id, start, end, z_score,
        p_value) tuples produced by per-process detectors.  This skeleton
        performs threshold filtering and overlap counting; it does not fit
        a statistical model.
        """

        process_map = {p.process_id: p for p in self._processes}
        spans: list[PolymarkSpan] = []
        for process_id, start, end, z, p_val in span_scores:
            if process_id not in process_map:
                raise PolymarkDemixingError(f"Unknown process_id: {process_id}")
            if start < 0 or end <= start:
                raise PolymarkDemixingError("span must have start >= 0 and end > start")
            if not math.isfinite(z) or not math.isfinite(p_val):
                raise PolymarkDemixingError("z_score and p_value must be finite")
            spans.append(
                PolymarkSpan(
                    process_id=process_id,
                    start_token=start,
                    end_token=end,
                    z_score=z,
                    p_value=p_val,
                    overlap=False,
                )
            )

        significant = [s for s in spans if s.z_score > process_map[s.process_id].threshold]
        significant.sort(key=lambda s: (s.start_token, s.process_id))

        overlap_count = self._count_overlaps(significant)
        detected_ids = {s.process_id for s in significant}
        detected_processes = tuple(
            process_map[pid] for pid in sorted(detected_ids)
        )
        collision_count = self._count_collisions(significant)
        abstained = len(detected_processes) == 0

        warnings: list[str] = [
            "Skeleton scan-based demixer: no structured sparse or Bayesian model.",
            "No scan-aware null or family-wise error control implemented.",
            "Scientific demixing requires T06 calibrated per-process detectors.",
        ]
        if abstained:
            warnings.append("No process reached threshold; demixer abstained.")
        if overlap_count > 0:
            warnings.append(
                f"{overlap_count} overlapping significant spans detected; "
                "identifiability under overlap is not characterized."
            )

        return PolymarkDemixingResult(
            scientific_status="skeleton-uncalibrated",
            process_count=len(detected_processes),
            detected_processes=detected_processes,
            spans=tuple(significant),
            overlap_count=overlap_count,
            collision_count=collision_count,
            family_wise_fpr_estimate=None,
            abstained=abstained,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _count_overlaps(spans: list[PolymarkSpan]) -> int:
        overlaps = 0
        for i, a in enumerate(spans):
            for b in spans[i + 1 :]:
                if a.start_token < b.end_token and b.start_token < a.end_token:
                    overlaps += 1
        return overlaps

    @staticmethod
    def _count_collisions(spans: list[PolymarkSpan]) -> int:
        """Count spans from different processes that share token positions."""

        collisions = 0
        for i, a in enumerate(spans):
            for b in spans[i + 1 :]:
                if (
                    a.process_id != b.process_id
                    and a.start_token < b.end_token
                    and b.start_token < a.end_token
                ):
                    collisions += 1
        return collisions


def bonferroni_threshold(
    nominal_alpha: float,
    num_tests: int,
) -> float:
    """Return the Bonferroni-adjusted significance threshold.

    The family-wise error rate is controlled at ``nominal_alpha`` across
    ``num_tests`` independent or positively-dependent tests.  This is a
    conservative correction; it does not require independence but may
    over-correct under strong positive dependence.
    """

    if not 0.0 < nominal_alpha < 1.0:
        raise PolymarkDemixingError("nominal_alpha must be in (0, 1)")
    if num_tests < 1:
        raise PolymarkDemixingError("num_tests must be at least 1")
    return nominal_alpha / num_tests


def scan_aware_p_value(
    observed_z: float,
    num_scans: int,
    *,
    null_mean: float = 0.0,
    null_sd: float = 1.0,
) -> float:
    """Approximate scan-aware p-value using a Bonferroni-style supremum correction.

    For a scan of ``num_scans`` correlated z-score tests, the family-wise
    upper bound on the p-value is ``num_scans * P(Z >= observed_z)`` under
    the null.  This is conservative; a proper scan-aware null would model
    the correlation structure of overlapping windows.

    The result is clamped to [0, 1].
    """

    if num_scans < 1:
        raise PolymarkDemixingError("num_scans must be at least 1")
    if not math.isfinite(observed_z):
        raise PolymarkDemixingError("observed_z must be finite")
    standardized = (observed_z - null_mean) / null_sd if null_sd > 0 else observed_z
    single_p = float(math.erfc(standardized / math.sqrt(2.0)) / 2.0)
    return min(1.0, single_p * num_scans)


__all__ = [
    "PolymarkDemixer",
    "PolymarkDemixingError",
    "PolymarkDemixingResult",
    "PolymarkProcess",
    "PolymarkSpan",
    "bonferroni_threshold",
    "scan_aware_p_value",
]