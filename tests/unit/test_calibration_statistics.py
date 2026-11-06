from __future__ import annotations

import math

import pytest

from watermark_lab.statistics import (
    INDEPENDENT_UNIT_ASSUMPTION,
    clopper_pearson_two_sided_interval,
    clopper_pearson_upper_bound,
    minimum_sample_size_for_upper_bound,
    minimum_zero_event_sample_size,
    tpr_clopper_pearson_interval,
)


def test_zero_false_positive_bound_uses_exact_boundary_formula() -> None:
    bound = clopper_pearson_upper_bound(events=0, total=2_995, confidence_level=0.95)
    expected = 1.0 - 0.05 ** (1.0 / 2_995)

    assert bound == pytest.approx(expected, rel=1e-14)
    assert bound <= 0.001


def test_zero_event_sample_size_for_one_tenth_percent_is_exactly_2995() -> None:
    required = minimum_zero_event_sample_size(0.001, confidence_level=0.95)

    assert required == 2_995
    assert clopper_pearson_upper_bound(0, required, 0.95) <= 0.001
    assert clopper_pearson_upper_bound(0, required - 1, 0.95) > 0.001


def test_one_sided_bound_handles_interior_and_all_event_cases() -> None:
    assert clopper_pearson_upper_bound(5, 100, 0.95) == pytest.approx(
        0.1022533832947319
    )
    assert clopper_pearson_upper_bound(10, 10, 0.95) == 1.0


def test_two_sided_interval_is_equal_tailed_and_explicitly_labeled() -> None:
    interval = clopper_pearson_two_sided_interval(
        events=5,
        total=100,
        confidence_level=0.95,
        estimand="false-positive rate (FPR)",
    )

    assert interval.lower == pytest.approx(0.016431879182052158)
    assert interval.upper == pytest.approx(0.11283491110546276)
    assert interval.sides == "two-sided"
    assert interval.method == "clopper-pearson-exact"
    assert interval.estimand == "false-positive rate (FPR)"
    assert interval.assumption == INDEPENDENT_UNIT_ASSUMPTION


def test_tpr_interval_is_two_sided_and_names_the_estimand() -> None:
    interval = tpr_clopper_pearson_interval(90, 100, 0.95)

    assert interval.lower == pytest.approx(0.8237774022599773)
    assert interval.upper == pytest.approx(0.9509953107785141)
    assert interval.sides == "two-sided"
    assert interval.estimand == "true-positive rate (TPR)"
    assert "independent Bernoulli units" in interval.assumption
    assert "does not establish independence" in interval.assumption


def test_two_sided_interval_handles_boundary_counts() -> None:
    no_events = clopper_pearson_two_sided_interval(0, 10)
    all_events = clopper_pearson_two_sided_interval(10, 10)

    assert no_events.lower == 0.0
    assert no_events.upper == pytest.approx(0.3084971078187608)
    assert all_events.lower == pytest.approx(1.0 - no_events.upper)
    assert all_events.upper == 1.0


def test_sample_size_search_supports_a_fixed_nonzero_event_count() -> None:
    required = minimum_sample_size_for_upper_bound(
        events=1,
        target_upper_bound=0.01,
        confidence_level=0.95,
    )

    assert clopper_pearson_upper_bound(1, required, 0.95) <= 0.01
    assert clopper_pearson_upper_bound(1, required - 1, 0.95) > 0.01


@pytest.mark.parametrize(
    ("events", "total", "error"),
    [
        (True, 10, TypeError),
        (0, True, TypeError),
        (-1, 10, ValueError),
        (11, 10, ValueError),
        (0, 0, ValueError),
        (1.5, 10, TypeError),
    ],
)
def test_count_validation(events: object, total: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        clopper_pearson_upper_bound(events, total)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, math.inf, math.nan, True, "0.95"])
def test_confidence_validation(confidence: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        clopper_pearson_upper_bound(0, 10, confidence)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, math.inf, math.nan, True, "0.001"])
def test_target_validation(target: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        minimum_zero_event_sample_size(target)  # type: ignore[arg-type]


def test_sample_size_search_fails_when_target_is_outside_limit() -> None:
    with pytest.raises(ValueError, match="not attainable"):
        minimum_zero_event_sample_size(0.001, max_total=2_994)


@pytest.mark.parametrize("estimand", ["", "   ", 123])
def test_estimand_validation(estimand: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        clopper_pearson_two_sided_interval(1, 2, estimand=estimand)  # type: ignore[arg-type]
