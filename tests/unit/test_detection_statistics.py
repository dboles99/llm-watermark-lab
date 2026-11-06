"""Detection statistics and empirical calibration.

These tests check the properties that matter for a detector that will be used to
make accusations: that the null behaves like a null, that thresholds are
conservative rather than optimistic, and that unattainable claims are refused
rather than silently reported.
"""

from __future__ import annotations

import random

import pytest

from watermark_lab.statistics.empirical_null import (
    EmpiricalNullError,
    build_wrong_key_null,
    calibration_report,
    empirical_threshold,
    uniformity_check,
    wrong_key_p_value,
)
from watermark_lab.statistics.goodness_of_fit import (
    DEFAULT_BATTERY,
    GoodnessOfFitError,
    berk_jones_statistic,
    compute_battery,
    compute_statistic,
    exact_binomial_statistic,
    higher_criticism_statistic,
    runs_statistic,
    z_score_statistic,
)

GAMMA = 0.25


def _null_indicators(n: int, rng: random.Random, gamma: float = GAMMA) -> list[int]:
    return [1 if rng.random() < gamma else 0 for _ in range(n)]


def _watermarked_indicators(
    n: int, rng: random.Random, rate: float = 0.55
) -> list[int]:
    return [1 if rng.random() < rate else 0 for _ in range(n)]


class TestNullBehaviour:
    @pytest.mark.parametrize("name", DEFAULT_BATTERY)
    def test_null_data_is_not_flagged(self, name: str) -> None:
        """Each statistic must behave on data generated from its own null."""

        rng = random.Random(11)
        flagged = 0
        trials = 200
        for _ in range(trials):
            indicators = _null_indicators(300, rng)
            result = compute_statistic(name, indicators, GAMMA)
            if result.p_value < 0.05:
                flagged += 1
        # Allow generous Monte-Carlo slack; the point is that it is near nominal,
        # not that it is exact.
        assert flagged / trials < 0.15, f"{name} flagged {flagged}/{trials} null samples"

    @pytest.mark.parametrize("name", DEFAULT_BATTERY)
    def test_p_values_are_probabilities(self, name: str) -> None:
        rng = random.Random(5)
        result = compute_statistic(name, _null_indicators(200, rng), GAMMA)
        assert 0.0 <= result.p_value <= 1.0


class TestPower:
    @pytest.mark.parametrize(
        "name", ["z_score", "exact_binomial", "berk_jones", "higher_criticism"]
    )
    def test_watermarked_text_is_detected(self, name: str) -> None:
        rng = random.Random(7)
        result = compute_statistic(name, _watermarked_indicators(300, rng), GAMMA)
        assert result.p_value < 0.001

    def test_runs_test_sees_ordering_a_count_cannot(self) -> None:
        """A perfectly clustered sequence has null count but non-null ordering."""

        indicators = [1] * 75 + [0] * 225  # exactly gamma = 0.25 overall
        count_based = z_score_statistic(indicators, GAMMA)
        ordering_based = runs_statistic(indicators, GAMMA)

        assert count_based.p_value > 0.4, "count-based test should see nothing here"
        assert ordering_based.p_value < 0.001, "runs test should see the clustering"

    def test_higher_criticism_finds_a_localised_watermark(self) -> None:
        """A short watermarked span diluted in human text evades a global count."""

        rng = random.Random(3)
        human = _null_indicators(600, rng)
        span = _watermarked_indicators(60, rng, rate=0.95)
        mixed = human[:300] + span + human[300:]

        global_test = z_score_statistic(mixed, GAMMA)
        scan_test = higher_criticism_statistic(mixed, GAMMA, window=30)
        assert scan_test.p_value < global_test.p_value


class TestStatisticContracts:
    def test_exact_binomial_needs_no_normal_approximation(self) -> None:
        """On short text the exact test is the defensible one."""

        indicators = [1] * 6 + [0] * 4
        exact = exact_binomial_statistic(indicators, GAMMA)
        approximate = z_score_statistic(indicators, GAMMA)
        assert 0.0 <= exact.p_value <= 1.0
        assert exact.p_value != approximate.p_value

    def test_berk_jones_is_zero_below_gamma(self) -> None:
        result = berk_jones_statistic([0] * 100, GAMMA)
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_runs_test_refuses_degenerate_sequences(self) -> None:
        result = runs_statistic([1] * 50, GAMMA)
        assert result.p_value == 1.0
        assert "undefined" in result.note.lower()

    def test_statistics_record_their_inputs(self) -> None:
        result = z_score_statistic([1, 0, 1, 0, 0, 0, 0, 0], GAMMA)
        assert result.scored_tokens == 8
        assert result.green_tokens == 2
        assert result.gamma == GAMMA

    @pytest.mark.parametrize("gamma", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_invalid_gamma(self, gamma: float) -> None:
        with pytest.raises(GoodnessOfFitError):
            z_score_statistic([1, 0, 1], gamma)

    def test_rejects_empty_indicators(self) -> None:
        with pytest.raises(GoodnessOfFitError):
            z_score_statistic([], GAMMA)

    def test_rejects_unknown_statistic(self) -> None:
        with pytest.raises(GoodnessOfFitError, match="Unknown statistic"):
            compute_statistic("telepathy", [1, 0], GAMMA)  # type: ignore[arg-type]


class TestBatteryIsADiagnosticNotADecision:
    def test_battery_warns_about_selecting_the_best_p_value(self) -> None:
        rng = random.Random(2)
        payload = compute_battery(_null_indicators(200, rng), GAMMA)
        assert payload["test_count"] == len(DEFAULT_BATTERY)
        assert "inflates the false-positive rate" in payload["selection_warning"]
        assert "repeated" in payload["independence_note"].lower()

    def test_battery_reports_every_requested_statistic(self) -> None:
        rng = random.Random(4)
        payload = compute_battery(_null_indicators(100, rng), GAMMA)
        assert set(payload["statistics"]) == set(DEFAULT_BATTERY)


class TestEmpiricalThreshold:
    def test_threshold_is_conservative(self) -> None:
        """At most floor(alpha * n) null samples may exceed the threshold."""

        rng = random.Random(9)
        scores = [rng.gauss(0.0, 1.0) for _ in range(1000)]
        result = empirical_threshold(scores, alpha=0.05, corpus_id="synthetic-normal")
        assert result.exceedances <= 50
        assert result.estimated_fpr <= 0.05

    def test_estimated_fpr_is_populated_not_none(self) -> None:
        """The whole point: estimated_fpr comes from a corpus, not a Gaussian tail."""

        rng = random.Random(1)
        scores = [rng.gauss(0.0, 1.0) for _ in range(500)]
        result = empirical_threshold(scores, alpha=0.01, corpus_id="c")
        payload = result.as_dict()
        assert payload["estimated_fpr"] is not None
        assert payload["estimated_fpr_upper_bound"] >= payload["estimated_fpr"]
        assert payload["corpus_id"] == "c"

    def test_upper_bound_exceeds_point_estimate_when_zero_events(self) -> None:
        scores = [0.0] * 100
        result = empirical_threshold(scores, alpha=0.01, corpus_id="degenerate")
        assert result.estimated_fpr == 0.0
        assert result.estimated_fpr_upper_bound > 0.0

    def test_analytic_gap_flags_a_permissive_analytic_threshold(self) -> None:
        """A positive gap means the analytic threshold would over-fire."""

        rng = random.Random(6)
        # Heavy-tailed null: the normal threshold is far too permissive.
        scores = [rng.gauss(0.0, 1.0) * (3.0 if rng.random() < 0.1 else 1.0) for _ in range(2000)]
        result = empirical_threshold(
            scores, alpha=0.01, corpus_id="heavy-tailed", analytic_threshold=2.326
        )
        assert result.analytic_gap is not None
        assert result.analytic_gap > 0.0

    def test_attainability_tracks_corpus_size(self) -> None:
        small = empirical_threshold([0.0] * 100, alpha=0.001, corpus_id="small")
        large = empirical_threshold([0.0] * 1000, alpha=0.001, corpus_id="large")
        assert small.attainable is False
        assert large.attainable is True

    def test_rejects_empty_null(self) -> None:
        with pytest.raises(EmpiricalNullError):
            empirical_threshold([], alpha=0.05, corpus_id="x")

    def test_requires_a_corpus_identifier(self) -> None:
        with pytest.raises(EmpiricalNullError, match="corpus_id"):
            empirical_threshold([1.0, 2.0], alpha=0.05, corpus_id="   ")

    def test_report_lists_unattainable_operating_points(self) -> None:
        report = calibration_report(
            [float(i) for i in range(200)], corpus_id="c", alphas=(0.05, 0.001)
        )
        assert 0.001 in report["unattainable_alphas"]
        assert 0.05 not in report["unattainable_alphas"]
        assert report["thresholds"]["0.001"]["attainable"] is False


class TestWrongKeyPermutationNull:
    def test_p_value_uses_the_plus_one_correction(self) -> None:
        """Without +1 the p-value can be zero, which is never valid."""

        result = wrong_key_p_value(100.0, [1.0] * 99)
        assert result.p_value == pytest.approx(1 / 100)
        assert result.p_value > 0.0

    def test_resolution_floor_is_reported(self) -> None:
        result = wrong_key_p_value(5.0, [0.0] * 99)
        assert result.resolution == pytest.approx(1 / 100)
        # A 0.1% claim is not representable with 99 draws.
        assert result.resolution > 0.001

    def test_observed_inside_the_null_is_not_significant(self) -> None:
        rng = random.Random(8)
        null = [rng.gauss(0.0, 1.0) for _ in range(499)]
        result = wrong_key_p_value(0.0, null)
        assert result.p_value > 0.2

    def test_observed_far_outside_the_null_is_significant(self) -> None:
        rng = random.Random(8)
        null = [rng.gauss(0.0, 1.0) for _ in range(499)]
        result = wrong_key_p_value(9.0, null)
        assert result.p_value == pytest.approx(1 / 500)

    def test_build_null_scores_every_key(self) -> None:
        keys = [b"k1", b"k2", b"k3"]
        null = build_wrong_key_null(lambda key: float(len(key)), keys)
        assert null == [2.0, 2.0, 2.0]

    def test_build_null_requires_keys(self) -> None:
        with pytest.raises(EmpiricalNullError):
            build_wrong_key_null(lambda key: 1.0, [])


class TestUniformityCheck:
    def test_uniform_p_values_pass(self) -> None:
        rng = random.Random(12)
        result = uniformity_check([rng.random() for _ in range(2000)])
        assert result["uniform"] is True

    def test_skewed_p_values_fail(self) -> None:
        """A miscalibrated null produces non-uniform p-values, and must be caught."""

        rng = random.Random(13)
        skewed = [rng.random() ** 3 for _ in range(2000)]
        result = uniformity_check(skewed)
        assert result["uniform"] is False
        assert result["p_value"] < 0.05

    def test_rejects_out_of_range_values(self) -> None:
        with pytest.raises(EmpiricalNullError):
            uniformity_check([0.5, 1.5])

    def test_rejects_too_few_bins(self) -> None:
        with pytest.raises(EmpiricalNullError):
            uniformity_check([0.1, 0.2], bins=1)
