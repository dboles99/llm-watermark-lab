"""Goodness-of-fit detection statistics for green-list watermarks.

The classical KGW detector counts green tokens and forms a normal-approximation
z-statistic against ``Binomial(n, gamma)``. That throws away structure. It uses
only the *total* count, so it cannot see where the green tokens fall, and it
compares against an asymptotic null that is inaccurate in the short-text regime
where operational detection actually happens.

Goodness-of-fit tests use the same per-token green indicators but ask a stronger
question: is the *sequence* of indicators consistent with independent
``Bernoulli(gamma)`` draws? Under a watermark it is not, and several classical
statistics detect that departure with more power than the count alone —
particularly at low decoding temperature, where repetition makes the indicator
sequence structured in ways a count is blind to.

Every statistic here consumes the same input the z-score does, so this is a
drop-in change of test statistic rather than a change of detector inputs.

Two cautions that the reporting code must preserve:

- **These p-values are asymptotic or approximate.** For a calibrated operating
  point, use them as the statistic and obtain the threshold empirically from a
  frozen null corpus (see ``watermark_lab.statistics.empirical_null``), rather
  than trusting the analytic tail.
- **Repeated contexts break independence.** A token whose context was already
  scored is not an independent draw, so the repeated-context policy must be
  applied *before* these tests, not after.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from scipy.stats import binomtest, kstwobign, norm  # type: ignore[import-untyped]

StatisticName = Literal[
    "z_score",
    "exact_binomial",
    "kolmogorov_smirnov",
    "berk_jones",
    "higher_criticism",
    "runs",
]

ASYMPTOTIC_NULL_NOTE = (
    "This p-value comes from an asymptotic or approximate null distribution. A calibrated "
    "operating point requires an empirical threshold from a frozen null corpus; do not "
    "report this tail probability as a false-positive rate."
)

INDEPENDENCE_NOTE = (
    "These tests assume the green indicators are independent Bernoulli(gamma) draws under "
    "the null. Repeated (context, token) pairs violate that assumption and must be masked "
    "before scoring, not after."
)


class GoodnessOfFitError(ValueError):
    """Raised when a detection statistic cannot be computed as specified."""


@dataclass(frozen=True, slots=True)
class DetectionStatistic:
    """One test statistic with its approximate p-value."""

    name: str
    statistic: float
    p_value: float
    scored_tokens: int
    green_tokens: int
    gamma: float
    note: str = ASYMPTOTIC_NULL_NOTE

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "scored_tokens": self.scored_tokens,
            "green_tokens": self.green_tokens,
            "gamma": self.gamma,
            "note": self.note,
        }


def _validate(indicators: Sequence[bool | int], gamma: float) -> tuple[list[int], float]:
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise GoodnessOfFitError("gamma must be a real number.")
    checked_gamma = float(gamma)
    if not math.isfinite(checked_gamma) or not 0.0 < checked_gamma < 1.0:
        raise GoodnessOfFitError("gamma must be strictly between 0 and 1.")
    values = [1 if bool(value) else 0 for value in indicators]
    if not values:
        raise GoodnessOfFitError("at least one green indicator is required.")
    return values, checked_gamma


def z_score_statistic(indicators: Sequence[bool | int], gamma: float) -> DetectionStatistic:
    """The classical KGW normal-approximation z-statistic.

    Retained as the labelled baseline for the ablation the roadmap requires, not
    as the recommended default.
    """

    values, checked_gamma = _validate(indicators, gamma)
    total = len(values)
    green = sum(values)
    denominator = math.sqrt(total * checked_gamma * (1.0 - checked_gamma))
    statistic = (green - checked_gamma * total) / denominator
    return DetectionStatistic(
        name="z_score",
        statistic=statistic,
        p_value=float(norm.sf(statistic)),
        scored_tokens=total,
        green_tokens=green,
        gamma=checked_gamma,
    )


def exact_binomial_statistic(
    indicators: Sequence[bool | int], gamma: float
) -> DetectionStatistic:
    """Exact one-sided binomial test on the green count.

    Uses the same information as the z-score but without the normal
    approximation, so it is correct in the short-text regime where the
    approximation is worst. Its p-value is exact under the independence
    assumption, which the note still qualifies.
    """

    values, checked_gamma = _validate(indicators, gamma)
    total = len(values)
    green = sum(values)
    result = binomtest(green, total, checked_gamma, alternative="greater")
    return DetectionStatistic(
        name="exact_binomial",
        statistic=float(green),
        p_value=float(result.pvalue),
        scored_tokens=total,
        green_tokens=green,
        gamma=checked_gamma,
        note=(
            "Exact under the independence assumption; no normal approximation. "
            + INDEPENDENCE_NOTE
        ),
    )


def _uniform_scores(values: list[int], gamma: float) -> list[float]:
    """Map green indicators to Uniform(0, 1) scores under the null.

    A green token is assigned a draw from ``[0, gamma)`` and a red token from
    ``[gamma, 1)``, placed deterministically at the midpoint of its interval so
    the transform is reproducible. Under the null the resulting scores are
    stochastically uniform; under a watermark they concentrate below ``gamma``.
    This lets distribution-shape tests be applied to binary indicators.
    """

    return [gamma * 0.5 if value else gamma + (1.0 - gamma) * 0.5 for value in values]


def kolmogorov_smirnov_statistic(
    indicators: Sequence[bool | int], gamma: float
) -> DetectionStatistic:
    """One-sided KS departure of the empirical green rate from ``gamma``.

    For binary data the KS statistic against the null CDF reduces to the absolute
    deviation of the empirical green proportion from ``gamma``, scaled by
    ``sqrt(n)``. It is reported separately because its null distribution differs
    from the normal tail and is less sensitive to the variance approximation.
    """

    values, checked_gamma = _validate(indicators, gamma)
    total = len(values)
    green = sum(values)
    deviation = green / total - checked_gamma
    statistic = math.sqrt(total) * abs(deviation)
    p_value = float(kstwobign.sf(statistic)) if deviation > 0 else 1.0
    return DetectionStatistic(
        name="kolmogorov_smirnov",
        statistic=statistic,
        p_value=min(1.0, max(0.0, p_value)),
        scored_tokens=total,
        green_tokens=green,
        gamma=checked_gamma,
    )


def berk_jones_statistic(indicators: Sequence[bool | int], gamma: float) -> DetectionStatistic:
    """Berk-Jones style maximum-likelihood-ratio departure statistic.

    Computes the binomial log-likelihood ratio of the observed green rate against
    ``gamma``, one-sided. It is more sensitive than a count-based test when the
    departure is moderate but sustained, which is the regime a diluted or
    partially-edited watermark produces.
    """

    values, checked_gamma = _validate(indicators, gamma)
    total = len(values)
    green = sum(values)
    rate = green / total
    if rate <= checked_gamma:
        return DetectionStatistic(
            name="berk_jones",
            statistic=0.0,
            p_value=1.0,
            scored_tokens=total,
            green_tokens=green,
            gamma=checked_gamma,
        )

    def _kl(p: float, q: float) -> float:
        terms = 0.0
        if p > 0.0:
            terms += p * math.log(p / q)
        if p < 1.0:
            terms += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
        return terms

    statistic = total * _kl(rate, checked_gamma)
    # Wilks: 2 * log-likelihood-ratio is asymptotically chi-square with 1 df, so
    # the one-sided tail is the normal survival of sqrt(2 * statistic).
    p_value = float(norm.sf(math.sqrt(2.0 * statistic)))
    return DetectionStatistic(
        name="berk_jones",
        statistic=statistic,
        p_value=p_value,
        scored_tokens=total,
        green_tokens=green,
        gamma=checked_gamma,
    )


def higher_criticism_statistic(
    indicators: Sequence[bool | int], gamma: float, *, window: int = 25
) -> DetectionStatistic:
    """Higher-criticism style scan over fixed windows.

    Splits the indicator sequence into windows and takes the maximum standardised
    departure across them. This detects a *localised* watermark that a
    document-level count would dilute below significance — the mixed-source case
    where only part of a document is machine-generated.

    Because it maximises over windows, its null is not the normal tail: the
    reported p-value applies a Bonferroni correction across windows, which is
    conservative under the positive dependence that overlapping content induces.
    """

    values, checked_gamma = _validate(indicators, gamma)
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise GoodnessOfFitError("window must be an integer of at least 2.")

    total = len(values)
    green = sum(values)
    windows = [values[i : i + window] for i in range(0, total, window)]
    windows = [chunk for chunk in windows if len(chunk) >= 2]
    if not windows:
        return z_score_statistic(values, checked_gamma)

    best = -math.inf
    for chunk in windows:
        size = len(chunk)
        count = sum(chunk)
        denominator = math.sqrt(size * checked_gamma * (1.0 - checked_gamma))
        best = max(best, (count - checked_gamma * size) / denominator)

    p_value = min(1.0, float(norm.sf(best)) * len(windows))
    return DetectionStatistic(
        name="higher_criticism",
        statistic=best,
        p_value=p_value,
        scored_tokens=total,
        green_tokens=green,
        gamma=checked_gamma,
        note=(
            "Maximum standardised departure over non-overlapping windows, Bonferroni-"
            "corrected across windows. Detects localised watermarks a document-level "
            "count would dilute. " + ASYMPTOTIC_NULL_NOTE
        ),
    )


def runs_statistic(indicators: Sequence[bool | int], gamma: float) -> DetectionStatistic:
    """Wald-Wolfowitz runs test on the green/red sequence.

    This is the one statistic here that is genuinely blind to the green *count*
    and sensitive only to *ordering*. It catches clustering that a count cannot
    see — for example a watermarked passage spliced into human text, or the
    repetition-induced structure that appears at low decoding temperature.

    Reported two-sided, because both excessive clustering and excessive
    alternation are departures from independence.
    """

    values, checked_gamma = _validate(indicators, gamma)
    total = len(values)
    green = sum(values)
    red = total - green
    if green == 0 or red == 0:
        return DetectionStatistic(
            name="runs",
            statistic=0.0,
            p_value=1.0,
            scored_tokens=total,
            green_tokens=green,
            gamma=checked_gamma,
            note="Degenerate sequence (all green or all red); the runs test is undefined.",
        )

    runs = 1 + sum(1 for i in range(1, total) if values[i] != values[i - 1])
    expected = 1.0 + 2.0 * green * red / total
    variance = (2.0 * green * red * (2.0 * green * red - total)) / (total**2 * (total - 1))
    if variance <= 0.0:
        return DetectionStatistic(
            name="runs",
            statistic=0.0,
            p_value=1.0,
            scored_tokens=total,
            green_tokens=green,
            gamma=checked_gamma,
            note="Runs variance is degenerate for this sequence.",
        )

    statistic = (runs - expected) / math.sqrt(variance)
    return DetectionStatistic(
        name="runs",
        statistic=statistic,
        p_value=float(2.0 * norm.sf(abs(statistic))),
        scored_tokens=total,
        green_tokens=green,
        gamma=checked_gamma,
        note=(
            "Two-sided test of ordering only; blind to the green count. Detects clustering "
            "a count-based statistic cannot see. " + ASYMPTOTIC_NULL_NOTE
        ),
    )


_BATTERY = {
    "z_score": z_score_statistic,
    "exact_binomial": exact_binomial_statistic,
    "kolmogorov_smirnov": kolmogorov_smirnov_statistic,
    "berk_jones": berk_jones_statistic,
    "higher_criticism": higher_criticism_statistic,
    "runs": runs_statistic,
}

DEFAULT_BATTERY: tuple[StatisticName, ...] = (
    "z_score",
    "exact_binomial",
    "berk_jones",
    "higher_criticism",
    "runs",
)


def compute_statistic(
    name: StatisticName, indicators: Sequence[bool | int], gamma: float, **kwargs: Any
) -> DetectionStatistic:
    """Compute one named statistic."""

    if name not in _BATTERY:
        known = ", ".join(sorted(_BATTERY))
        raise GoodnessOfFitError(f"Unknown statistic {name!r}. Known: {known}.")
    return _BATTERY[name](indicators, gamma, **kwargs)


def compute_battery(
    indicators: Sequence[bool | int],
    gamma: float,
    *,
    names: tuple[StatisticName, ...] = DEFAULT_BATTERY,
) -> dict[str, Any]:
    """Compute several statistics on the same indicators.

    **The battery is a diagnostic, not a decision rule.** Reporting the smallest
    p-value across several tests is itself a multiple-comparison problem: the
    test must be preregistered, or the family declared and corrected via
    ``watermark_lab.statistics.multiplicity``. The returned payload records the
    number of tests so that correction cannot be forgotten.
    """

    results = {name: compute_statistic(name, indicators, gamma) for name in names}
    return {
        "statistics": {name: result.as_dict() for name, result in results.items()},
        "test_count": len(results),
        "selection_warning": (
            "Selecting the most significant of several statistics after seeing the data "
            "inflates the false-positive rate. Preregister the statistic, or declare a "
            "family of size test_count and correct for it."
        ),
        "independence_note": INDEPENDENCE_NOTE,
    }


__all__ = [
    "ASYMPTOTIC_NULL_NOTE",
    "DEFAULT_BATTERY",
    "INDEPENDENCE_NOTE",
    "DetectionStatistic",
    "GoodnessOfFitError",
    "berk_jones_statistic",
    "compute_battery",
    "compute_statistic",
    "exact_binomial_statistic",
    "higher_criticism_statistic",
    "kolmogorov_smirnov_statistic",
    "runs_statistic",
    "z_score_statistic",
]
