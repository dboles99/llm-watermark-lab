"""Family-wise error control for sweeps over keys, configurations, and windows.

A war-dial sweep tests one text against many watermark configurations. Each test
carries its own Type-I error probability, so the probability that *at least one*
configuration fires on unwatermarked text grows with the size of the sweep. A
per-test threshold chosen for a single comparison therefore does not control the
error rate of the sweep that reports it.

The scale of the problem is not subtle. At a per-test level of 0.05, a sweep of
100 independent configurations produces at least one false detection with
probability ``1 - 0.95 ** 100``, which is about 0.994. A "hit" from an
uncorrected sweep of that size is close to uninformative.

This module supplies the corrections and the family bookkeeping that a run
manifest must record. It deliberately does not decide what the family *is*:
that is a preregistration question, and declaring it after seeing results
reintroduces exactly the error the correction removes.

Two dependence regimes matter here and are handled separately:

``sidak_threshold``
    Assumes the tests are independent. Exact under that assumption and slightly
    less conservative than Bonferroni.

``bonferroni_threshold``
    Makes no independence assumption. Valid under arbitrary dependence, so it is
    the correct default when tests share a corpus, a key schedule, or overlapping
    windows.

Overlapping sliding windows are dependent by construction, and no analytic
document-level bound follows from a per-window target. ``per_window_target_level``
implements the alternative used in the segment-detection literature: pick a very
small per-window level and certify the document-level rate by simulation. The
returned record marks the document-level rate as *requiring* simulation rather
than implying one has been done.

Both the Šidák and Bonferroni corrections are standard multiple-comparison
procedures, not methods introduced by any watermarking paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

CorrectionMethod = Literal["sidak", "bonferroni", "none"]

ARBITRARY_DEPENDENCE_NOTE = (
    "Bonferroni controls the family-wise error rate under arbitrary dependence between "
    "tests. It is the correct choice when swept configurations share a corpus, key "
    "schedule, tokenizer, or overlapping windows."
)

INDEPENDENCE_ASSUMPTION_NOTE = (
    "Sidak is exact only when the tests in the family are mutually independent. Swept "
    "configurations that share a corpus or overlapping windows are not independent; "
    "prefer Bonferroni unless independence is argued explicitly."
)

WINDOW_DEPENDENCE_NOTE = (
    "Overlapping windows are dependent by construction, so no analytic document-level "
    "bound follows from a per-window level. The document-level rate must be certified "
    "by simulation over a frozen null corpus with a recorded seed."
)

UNCORRECTED_NOTE = (
    "No multiplicity correction was applied. The per-test level is NOT the family-wise "
    "level. Any 'zero false positives' claim from a multi-configuration sweep reported "
    "at this level is a multiplicity artifact."
)


class MultiplicityError(ValueError):
    """Raised when a family definition or correction request is invalid."""


def _validate_level(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiplicityError(f"{field_name} must be a real number.")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 < checked < 1.0:
        raise MultiplicityError(f"{field_name} must be finite and strictly between 0 and 1.")
    return checked


def _validate_family_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiplicityError("family_size must be an integer.")
    if value < 1:
        raise MultiplicityError("family_size must be at least 1.")
    return value


def sidak_threshold(family_level: float, family_size: int) -> float:
    """Return the per-test level giving ``family_level`` under independence.

    ``alpha_test = 1 - (1 - alpha_family) ** (1 / K)``. Exact when the ``K`` tests
    are mutually independent; anti-conservative when they are positively
    dependent, which swept configurations sharing a corpus generally are.
    """

    level = _validate_level(family_level, field_name="family_level")
    size = _validate_family_size(family_size)
    return -math.expm1(math.log1p(-level) / size)


def bonferroni_threshold(family_level: float, family_size: int) -> float:
    """Return the per-test level giving ``family_level`` under any dependence."""

    level = _validate_level(family_level, field_name="family_level")
    size = _validate_family_size(family_size)
    return level / size


def family_wise_error_rate(per_test_level: float, family_size: int) -> float:
    """Return P(at least one rejection) for independent null tests.

    This is the quantity an uncorrected sweep actually incurs. It is exact only
    under independence; under positive dependence the true rate is lower, and
    under adversarial dependence it can approach ``min(1, K * per_test_level)``.
    """

    level = _validate_level(per_test_level, field_name="per_test_level")
    size = _validate_family_size(family_size)
    return -math.expm1(size * math.log1p(-level))


@dataclass(frozen=True, slots=True)
class SweepFamily:
    """A declared multiple-comparison family and its corrected per-test level.

    The family must be declared before the sweep runs. ``dimensions`` records
    what was varied so a reviewer can check that ``family_size`` matches the grid
    that was actually executed.
    """

    family_level: float
    family_size: int
    per_test_level: float
    method: CorrectionMethod
    uncorrected_family_wise_rate: float
    dimensions: tuple[str, ...] = ()
    note: str = ""
    requires_simulation: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "family_level": self.family_level,
            "family_size": self.family_size,
            "per_test_level": self.per_test_level,
            "method": self.method,
            "uncorrected_family_wise_rate": self.uncorrected_family_wise_rate,
            "dimensions": list(self.dimensions),
            "note": self.note,
            "requires_simulation": self.requires_simulation,
        }


def declare_sweep_family(
    *,
    family_level: float,
    family_size: int,
    method: CorrectionMethod = "bonferroni",
    dimensions: tuple[str, ...] = (),
) -> SweepFamily:
    """Declare a comparison family and compute its corrected per-test level.

    ``method`` defaults to ``bonferroni`` because swept configurations sharing a
    corpus are not independent. ``method="none"`` is permitted so that an
    uncorrected baseline can be recorded explicitly for an ablation, but the
    resulting record carries a note saying the per-test level is not the
    family-wise level.
    """

    level = _validate_level(family_level, field_name="family_level")
    size = _validate_family_size(family_size)
    if method not in ("sidak", "bonferroni", "none"):
        raise MultiplicityError("method must be 'sidak', 'bonferroni', or 'none'.")
    if not isinstance(dimensions, tuple) or not all(isinstance(d, str) for d in dimensions):
        raise MultiplicityError("dimensions must be a tuple of strings.")

    if method == "sidak":
        per_test = sidak_threshold(level, size)
        note = INDEPENDENCE_ASSUMPTION_NOTE
    elif method == "bonferroni":
        per_test = bonferroni_threshold(level, size)
        note = ARBITRARY_DEPENDENCE_NOTE
    else:
        per_test = level
        note = UNCORRECTED_NOTE

    return SweepFamily(
        family_level=level,
        family_size=size,
        per_test_level=per_test,
        method=method,
        uncorrected_family_wise_rate=family_wise_error_rate(level, size),
        dimensions=dimensions,
        note=note,
    )


def per_window_target_level(
    *,
    family_level: float,
    window_count: int,
    per_window_level: float = 1e-6,
) -> SweepFamily:
    """Declare a per-window level for a scan whose windows overlap.

    Overlapping windows are dependent, so the document-level rate cannot be
    derived analytically from ``per_window_level``. This returns a record with
    ``requires_simulation=True``; the document-level rate must be certified by
    simulation over a frozen null corpus before any FPR claim is made.

    ``family_level`` is recorded as the document-level rate being *targeted*, not
    one that has been demonstrated.
    """

    level = _validate_level(family_level, field_name="family_level")
    windows = _validate_family_size(window_count)
    window_level = _validate_level(per_window_level, field_name="per_window_level")

    return SweepFamily(
        family_level=level,
        family_size=windows,
        per_test_level=window_level,
        method="none",
        uncorrected_family_wise_rate=family_wise_error_rate(window_level, windows),
        dimensions=("window",),
        note=WINDOW_DEPENDENCE_NOTE,
        requires_simulation=True,
    )


@dataclass(frozen=True, slots=True)
class EffectiveSampleSize:
    """Cluster-adjusted sample size for dependent observations.

    Passages drawn from the same document or author are not independent units.
    Reporting ``n_passages`` where the analysis assumes independence overstates
    precision. The design effect for equal-sized clusters is
    ``1 + (m - 1) * rho`` for cluster size ``m`` and intracluster correlation
    ``rho``; the effective size is ``n / design_effect``.

    No LLM-watermarking paper found in the 2026-08-30 sweep addresses clustering,
    so ``rho`` must be estimated from the corpus or bounded conservatively, and
    the choice recorded. This is an open methodological gap, not a solved one.
    """

    n_observations: int
    n_clusters: int
    mean_cluster_size: float
    intracluster_correlation: float
    design_effect: float
    n_effective: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_observations": self.n_observations,
            "n_clusters": self.n_clusters,
            "mean_cluster_size": self.mean_cluster_size,
            "intracluster_correlation": self.intracluster_correlation,
            "design_effect": self.design_effect,
            "n_effective": self.n_effective,
        }


def effective_sample_size(
    *,
    n_observations: int,
    n_clusters: int,
    intracluster_correlation: float,
) -> EffectiveSampleSize:
    """Return the cluster-adjusted effective sample size.

    With ``rho = 0`` this returns ``n_observations`` unchanged. With one
    observation per cluster the design effect is 1 regardless of ``rho``.
    """

    if isinstance(n_observations, bool) or not isinstance(n_observations, int):
        raise MultiplicityError("n_observations must be an integer.")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise MultiplicityError("n_clusters must be an integer.")
    if n_observations < 1:
        raise MultiplicityError("n_observations must be at least 1.")
    if not 1 <= n_clusters <= n_observations:
        raise MultiplicityError("n_clusters must be between 1 and n_observations.")
    if isinstance(intracluster_correlation, bool) or not isinstance(
        intracluster_correlation, (int, float)
    ):
        raise MultiplicityError("intracluster_correlation must be a real number.")
    rho = float(intracluster_correlation)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise MultiplicityError("intracluster_correlation must be between 0 and 1.")

    mean_cluster_size = n_observations / n_clusters
    design_effect = 1.0 + (mean_cluster_size - 1.0) * rho
    return EffectiveSampleSize(
        n_observations=n_observations,
        n_clusters=n_clusters,
        mean_cluster_size=mean_cluster_size,
        intracluster_correlation=rho,
        design_effect=design_effect,
        n_effective=n_observations / design_effect,
    )


@dataclass(frozen=True, slots=True)
class WorstGroupRate:
    """A per-stratum rate summary reporting the worst group, not the pool.

    Pooled false-positive rates hide subgroup disparity, and that disparity is
    structural for text-only detectors rather than a fixable defect. Sample-size
    floors apply *per stratum*: a 2,995-negative floor for a 0.1% zero-event
    bound must be met within each stratum, not across the pool.
    """

    strata: tuple[str, ...]
    rates: tuple[float, ...]
    counts: tuple[int, ...]
    pooled_rate: float
    worst_group: str
    worst_group_rate: float
    underpowered_strata: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, object]:
        return {
            "strata": list(self.strata),
            "rates": list(self.rates),
            "counts": list(self.counts),
            "pooled_rate": self.pooled_rate,
            "worst_group": self.worst_group,
            "worst_group_rate": self.worst_group_rate,
            "underpowered_strata": list(self.underpowered_strata),
        }


def worst_group_rate(
    events_by_stratum: dict[str, int],
    totals_by_stratum: dict[str, int],
    *,
    minimum_per_stratum: int = 0,
) -> WorstGroupRate:
    """Summarise a rate across strata, surfacing the worst group.

    ``minimum_per_stratum`` flags strata too small to support the intended claim;
    they are reported rather than silently pooled away.
    """

    if not events_by_stratum:
        raise MultiplicityError("events_by_stratum must not be empty.")
    if set(events_by_stratum) != set(totals_by_stratum):
        raise MultiplicityError("events_by_stratum and totals_by_stratum must share keys.")
    if isinstance(minimum_per_stratum, bool) or not isinstance(minimum_per_stratum, int):
        raise MultiplicityError("minimum_per_stratum must be an integer.")
    if minimum_per_stratum < 0:
        raise MultiplicityError("minimum_per_stratum must be nonnegative.")

    strata = tuple(sorted(events_by_stratum))
    rates: list[float] = []
    counts: list[int] = []
    underpowered: list[str] = []
    total_events = 0
    total_units = 0

    for name in strata:
        events = events_by_stratum[name]
        total = totals_by_stratum[name]
        if isinstance(events, bool) or not isinstance(events, int):
            raise MultiplicityError(f"events for stratum {name!r} must be an integer.")
        if isinstance(total, bool) or not isinstance(total, int):
            raise MultiplicityError(f"total for stratum {name!r} must be an integer.")
        if total < 1:
            raise MultiplicityError(f"total for stratum {name!r} must be at least 1.")
        if not 0 <= events <= total:
            raise MultiplicityError(f"events for stratum {name!r} must be within [0, total].")
        rates.append(events / total)
        counts.append(total)
        total_events += events
        total_units += total
        if total < minimum_per_stratum:
            underpowered.append(name)

    worst_index = max(range(len(strata)), key=lambda i: rates[i])
    return WorstGroupRate(
        strata=strata,
        rates=tuple(rates),
        counts=tuple(counts),
        pooled_rate=total_events / total_units,
        worst_group=strata[worst_index],
        worst_group_rate=rates[worst_index],
        underpowered_strata=tuple(underpowered),
    )
