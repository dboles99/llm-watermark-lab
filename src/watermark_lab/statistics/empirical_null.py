"""Empirical null distributions and calibrated thresholds.

The analytic null for a green-list detector assumes independent
``Bernoulli(gamma)`` draws. Real text does not satisfy that: tokens repeat,
contexts recur, style is autocorrelated, and the tokenizer imposes its own
structure. The published consequence is that a closed-form threshold **misses its
nominal false-positive rate on natural language**, which is why serious work
recalibrates against a held-out human corpus before shipping a threshold.

This module implements the two-stage discipline the plan requires:

1. compute the analytic threshold;
2. set the *shipped* threshold to the empirical ``1 - alpha`` quantile of the
   detector statistic on a frozen, versioned human-text corpus;
3. report ``estimated_fpr`` from that corpus, never from the analytic tail.

It also provides the **wrong-key permutation null**, which is the right tool when
no human corpus is available for the exact configuration. Rescoring the *same*
text under many independent keys preserves every within-text dependence — token
repetition, context recurrence, autocorrelation — and erases only the keyed
alignment. What remains is a null that already contains the structure an analytic
null wrongly assumes away.

The ``+1`` in the permutation p-value numerator and denominator is not a fudge
factor: it is required for exact validity of a Monte-Carlo permutation p-value,
because the observed statistic is itself one draw from the null under the
hypothesis being tested. Omitting it produces anti-conservative p-values and can
report zero.
"""

from __future__ import annotations

import math
import statistics as _statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from watermark_lab.statistics.calibration import clopper_pearson_upper_bound

ANALYTIC_NULL_MISMATCH_NOTE = (
    "The analytic null assumes independent Bernoulli draws. Natural language violates "
    "that through token repetition, recurring contexts, and style autocorrelation, so an "
    "analytic threshold does not generally attain its nominal false-positive rate."
)

EMPIRICAL_THRESHOLD_NOTE = (
    "This threshold was read off a finite null corpus, so it inherits that corpus's "
    "population, domain, locale, and length distribution. It does not transfer to text "
    "drawn from a different population."
)


class EmpiricalNullError(ValueError):
    """Raised when a null distribution or threshold cannot be computed."""


@dataclass(frozen=True, slots=True)
class EmpiricalThreshold:
    """A threshold read from an empirical null distribution."""

    alpha: float
    threshold: float
    analytic_threshold: float | None
    null_size: int
    exceedances: int
    estimated_fpr: float
    estimated_fpr_upper_bound: float
    corpus_id: str
    note: str = EMPIRICAL_THRESHOLD_NOTE

    @property
    def analytic_gap(self) -> float | None:
        """Signed difference between empirical and analytic thresholds.

        A positive gap means the analytic threshold was **too permissive**: using
        it would have produced more false positives than nominal.
        """

        if self.analytic_threshold is None:
            return None
        return self.threshold - self.analytic_threshold

    @property
    def attainable(self) -> bool:
        """Whether the corpus is large enough to evidence the requested alpha.

        With ``n`` null samples the smallest resolvable rate is ``1 / n``. A
        corpus smaller than ``1 / alpha`` cannot demonstrate that alpha at all,
        however few exceedances are observed.
        """

        return self.null_size >= math.ceil(1.0 / self.alpha)

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "analytic_threshold": self.analytic_threshold,
            "analytic_gap": self.analytic_gap,
            "null_size": self.null_size,
            "exceedances": self.exceedances,
            "estimated_fpr": self.estimated_fpr,
            "estimated_fpr_upper_bound": self.estimated_fpr_upper_bound,
            "attainable": self.attainable,
            "corpus_id": self.corpus_id,
            "note": self.note,
        }


def _validate_alpha(alpha: object) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise EmpiricalNullError("alpha must be a real number.")
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise EmpiricalNullError("alpha must be strictly between 0 and 1.")
    return value


def _validate_scores(scores: Sequence[float]) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        raise EmpiricalNullError("the null distribution must contain at least one score.")
    if any(not math.isfinite(value) for value in values):
        raise EmpiricalNullError("null scores must all be finite.")
    return values


def empirical_threshold(
    null_scores: Sequence[float],
    *,
    alpha: float,
    corpus_id: str,
    analytic_threshold: float | None = None,
    confidence_level: float = 0.95,
) -> EmpiricalThreshold:
    """Return the ``1 - alpha`` quantile of an empirical null distribution.

    The quantile is taken conservatively: the threshold is the smallest observed
    score such that **at most** ``floor(alpha * n)`` null samples exceed it. This
    errs toward a higher threshold and therefore a lower realised false-positive
    rate, which is the correct direction when the consequence of a false positive
    is an accusation.

    ``estimated_fpr`` is the realised exceedance rate on this corpus.
    ``estimated_fpr_upper_bound`` is the exact Clopper-Pearson one-sided bound,
    which is the number that should be reported, because the point estimate can
    be zero while the true rate is not.
    """

    values = sorted(_validate_scores(null_scores), reverse=True)
    checked_alpha = _validate_alpha(alpha)
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise EmpiricalNullError("corpus_id must be a non-empty string identifying the corpus.")

    total = len(values)
    allowed = math.floor(checked_alpha * total)
    # values is descending, so values[allowed] is the largest score that leaves at
    # most `allowed` samples strictly above it.
    threshold = values[allowed] if allowed < total else values[-1]
    exceedances = sum(1 for value in values if value > threshold)

    return EmpiricalThreshold(
        alpha=checked_alpha,
        threshold=threshold,
        analytic_threshold=analytic_threshold,
        null_size=total,
        exceedances=exceedances,
        estimated_fpr=exceedances / total,
        estimated_fpr_upper_bound=clopper_pearson_upper_bound(
            exceedances, total, confidence_level
        ),
        corpus_id=corpus_id.strip(),
    )


@dataclass(frozen=True, slots=True)
class PermutationResult:
    """A p-value from rescoring one text under many wrong keys."""

    observed: float
    p_value: float
    draws: int
    null_mean: float
    null_stdev: float
    resolution: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "p_value": self.p_value,
            "draws": self.draws,
            "null_mean": self.null_mean,
            "null_stdev": self.null_stdev,
            "resolution": self.resolution,
            "note": (
                "Wrong-key permutation null. Preserves within-text dependence and erases "
                "only the keyed alignment. The smallest attainable p-value is 1/(draws+1)."
            ),
        }


def wrong_key_p_value(observed: float, null_scores: Sequence[float]) -> PermutationResult:
    """Return the permutation p-value of ``observed`` against wrong-key scores.

    ``p = (1 + #{r : S_r >= S_observed}) / (M + 1)``.

    The smallest attainable p-value is ``1 / (M + 1)``, so a 0.1% claim needs at
    least 999 draws before it is even representable. ``resolution`` reports that
    floor so an unattainably small claim cannot be made by accident.
    """

    values = _validate_scores(null_scores)
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise EmpiricalNullError("observed must be a real number.")
    checked = float(observed)
    if not math.isfinite(checked):
        raise EmpiricalNullError("observed must be finite.")

    draws = len(values)
    at_least_as_extreme = sum(1 for value in values if value >= checked)
    return PermutationResult(
        observed=checked,
        p_value=(1 + at_least_as_extreme) / (draws + 1),
        draws=draws,
        null_mean=_statistics.fmean(values),
        null_stdev=_statistics.pstdev(values) if draws > 1 else 0.0,
        resolution=1.0 / (draws + 1),
    )


def build_wrong_key_null(
    score_fn: Callable[[bytes], float],
    keys: Sequence[bytes],
) -> list[float]:
    """Score one text under each wrong key, producing an empirical null.

    ``score_fn`` must recompute the detector statistic for the *same* text under
    the supplied key. Keys must be drawn independently from the same distribution
    as the real key, and must exclude the real key: including it would place the
    alternative inside the null.
    """

    if not keys:
        raise EmpiricalNullError("at least one wrong key is required.")
    return [float(score_fn(key)) for key in keys]


def uniformity_check(p_values: Sequence[float], *, bins: int = 10) -> dict[str, Any]:
    """Test whether null p-values are uniform, via a chi-square bin count.

    If the null is correctly specified, p-values computed on null data are
    Uniform(0, 1). A significant departure means the null is wrong and every
    threshold derived from it is wrong too. This is the check that catches a
    miscalibrated detector before it reaches a decision.
    """

    values = _validate_scores(p_values)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise EmpiricalNullError("p-values must lie in [0, 1].")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise EmpiricalNullError("bins must be an integer of at least 2.")

    total = len(values)
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int(value * bins))
        counts[index] += 1

    expected = total / bins
    chi_square = sum((count - expected) ** 2 / expected for count in counts)

    from scipy.stats import chi2  # type: ignore[import-untyped]

    p_value = float(chi2.sf(chi_square, bins - 1))
    return {
        "bins": bins,
        "counts": counts,
        "expected_per_bin": expected,
        "chi_square": chi_square,
        "p_value": p_value,
        "uniform": p_value >= 0.05,
        "sample_size": total,
        "note": (
            "Null p-values must be Uniform(0, 1). A significant departure means the null "
            "is misspecified and every threshold derived from it is invalid."
        ),
    }


def calibration_report(
    null_scores: Sequence[float],
    *,
    alphas: Sequence[float] = (0.05, 0.01, 0.001),
    corpus_id: str,
    analytic_thresholds: dict[float, float] | None = None,
) -> dict[str, Any]:
    """Produce thresholds at several operating points from one null corpus.

    Operating points the corpus cannot evidence are still reported, but marked
    ``attainable: false`` rather than omitted — silently dropping an unattainable
    target is how an unsupportable claim survives review.
    """

    values = _validate_scores(null_scores)
    analytic = analytic_thresholds or {}
    thresholds = {}
    for alpha in alphas:
        result = empirical_threshold(
            values,
            alpha=alpha,
            corpus_id=corpus_id,
            analytic_threshold=analytic.get(alpha),
        )
        thresholds[str(alpha)] = result.as_dict()

    unattainable = [
        alpha for alpha in alphas if len(values) < math.ceil(1.0 / _validate_alpha(alpha))
    ]
    return {
        "corpus_id": corpus_id,
        "null_size": len(values),
        "thresholds": thresholds,
        "unattainable_alphas": unattainable,
        "analytic_null_note": ANALYTIC_NULL_MISMATCH_NOTE,
    }


__all__ = [
    "ANALYTIC_NULL_MISMATCH_NOTE",
    "EMPIRICAL_THRESHOLD_NOTE",
    "EmpiricalNullError",
    "EmpiricalThreshold",
    "PermutationResult",
    "build_wrong_key_null",
    "calibration_report",
    "empirical_threshold",
    "uniformity_check",
    "wrong_key_p_value",
]
