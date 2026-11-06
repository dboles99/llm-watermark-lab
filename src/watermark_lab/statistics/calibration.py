"""Exact finite-binomial intervals used by calibration studies.

These functions model each supplied observation as an independent Bernoulli
unit. They do not diagnose or correct dependence, prompt/model clustering,
repeated measurements, adaptive sampling, or distribution shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import beta  # type: ignore[import-untyped]

INDEPENDENT_UNIT_ASSUMPTION = (
    "Observations are prespecified independent Bernoulli units with one binary outcome per "
    "unit. This calculation does not establish independence or correct clustering, repeated "
    "measurements, adaptive sampling, or distribution shift."
)


@dataclass(frozen=True, slots=True)
class BinomialInterval:
    """An explicitly labeled exact interval for a binomial proportion."""

    lower: float
    upper: float
    confidence_level: float
    sides: Literal["two-sided"]
    method: Literal["clopper-pearson-exact"]
    estimand: str
    assumption: str = INDEPENDENT_UNIT_ASSUMPTION


def _require_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer.")
    return value


def _validate_counts(events: object, total: object) -> tuple[int, int]:
    checked_events = _require_integer(events, field="events")
    checked_total = _require_integer(total, field="total")
    if checked_total < 1:
        raise ValueError("total must be at least 1.")
    if not 0 <= checked_events <= checked_total:
        raise ValueError("events must be between 0 and total, inclusive.")
    return checked_events, checked_total


def _validate_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number.")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 < checked < 1.0:
        raise ValueError(f"{field} must be finite and strictly between 0 and 1.")
    return checked


def _beta_quantile(probability: float, shape_a: int, shape_b: int) -> float:
    value = float(beta.ppf(probability, shape_a, shape_b))
    if not math.isfinite(value):
        raise ArithmeticError("SciPy returned a non-finite beta quantile.")
    return value


def clopper_pearson_upper_bound(
    events: int,
    total: int,
    confidence_level: float = 0.95,
) -> float:
    """Return the exact one-sided upper confidence bound.

    ``events`` is the count of false positives for an FPR bound, or any other
    prespecified binary event. The result assumes independent Bernoulli units.
    For zero events, the boundary-safe expression is
    ``1 - (1 - confidence_level) ** (1 / total)``.
    """

    checked_events, checked_total = _validate_counts(events, total)
    confidence = _validate_probability(confidence_level, field="confidence_level")

    if checked_events == checked_total:
        return 1.0
    if checked_events == 0:
        return -math.expm1(math.log1p(-confidence) / checked_total)
    return _beta_quantile(confidence, checked_events + 1, checked_total - checked_events)


def clopper_pearson_two_sided_interval(
    events: int,
    total: int,
    confidence_level: float = 0.95,
    *,
    estimand: str = "binomial proportion",
) -> BinomialInterval:
    """Return an equal-tailed, two-sided exact Clopper-Pearson interval."""

    checked_events, checked_total = _validate_counts(events, total)
    confidence = _validate_probability(confidence_level, field="confidence_level")
    if not isinstance(estimand, str):
        raise TypeError("estimand must be a string.")
    checked_estimand = estimand.strip()
    if not checked_estimand:
        raise ValueError("estimand must not be empty.")

    tail_probability = (1.0 - confidence) / 2.0
    if checked_events == 0:
        lower = 0.0
        upper = -math.expm1(math.log(tail_probability) / checked_total)
    elif checked_events == checked_total:
        lower = math.exp(math.log(tail_probability) / checked_total)
        upper = 1.0
    else:
        lower = _beta_quantile(
            tail_probability,
            checked_events,
            checked_total - checked_events + 1,
        )
        upper = _beta_quantile(
            1.0 - tail_probability,
            checked_events + 1,
            checked_total - checked_events,
        )

    return BinomialInterval(
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        sides="two-sided",
        method="clopper-pearson-exact",
        estimand=checked_estimand,
    )


def tpr_clopper_pearson_interval(
    true_positives: int,
    positive_units: int,
    confidence_level: float = 0.95,
) -> BinomialInterval:
    """Return an explicitly labeled two-sided interval for true-positive rate."""

    return clopper_pearson_two_sided_interval(
        true_positives,
        positive_units,
        confidence_level,
        estimand="true-positive rate (TPR)",
    )


def minimum_sample_size_for_upper_bound(
    events: int,
    target_upper_bound: float,
    confidence_level: float = 0.95,
    *,
    max_total: int = 100_000_000,
) -> int:
    """Find the smallest total giving the requested one-sided upper bound.

    The observed event count is held fixed while ``total`` increases. This is
    a precision calculation under the independent-unit model, not a power
    analysis or a license to treat dependent records as independent.
    """

    checked_events = _require_integer(events, field="events")
    if checked_events < 0:
        raise ValueError("events must be nonnegative.")
    target = _validate_probability(target_upper_bound, field="target_upper_bound")
    confidence = _validate_probability(confidence_level, field="confidence_level")
    checked_max = _require_integer(max_total, field="max_total")
    minimum_total = max(1, checked_events)
    if checked_max < minimum_total:
        raise ValueError("max_total is smaller than the minimum valid total.")

    if clopper_pearson_upper_bound(checked_events, minimum_total, confidence) <= target:
        return minimum_total
    if clopper_pearson_upper_bound(checked_events, checked_max, confidence) > target:
        raise ValueError("target_upper_bound is not attainable within max_total.")

    low = minimum_total
    high = checked_max
    while low + 1 < high:
        midpoint = low + (high - low) // 2
        if clopper_pearson_upper_bound(checked_events, midpoint, confidence) <= target:
            high = midpoint
        else:
            low = midpoint
    return high


def minimum_zero_event_sample_size(
    target_upper_bound: float,
    confidence_level: float = 0.95,
    *,
    max_total: int = 100_000_000,
) -> int:
    """Find the minimum independent sample size after observing zero events."""

    return minimum_sample_size_for_upper_bound(
        0,
        target_upper_bound,
        confidence_level,
        max_total=max_total,
    )
