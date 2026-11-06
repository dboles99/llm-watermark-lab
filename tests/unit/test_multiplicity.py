"""Multiplicity control for sweeps over keys, configurations, and windows.

The central test here is ``test_uncorrected_sweep_inflates_family_wise_error``,
which demonstrates the defect the war-dial had before correction was available:
sweeping many configurations against one text at an uncorrected per-test level
produces a false detection almost every time on pure-noise input.
"""

from __future__ import annotations

import random

import pytest

from watermark_lab.statistics.multiplicity import (
    MultiplicityError,
    SweepFamily,
    bonferroni_threshold,
    declare_sweep_family,
    effective_sample_size,
    family_wise_error_rate,
    per_window_target_level,
    sidak_threshold,
    worst_group_rate,
)


def _sweep_fires(per_test_level: float, family_size: int, rng: random.Random) -> bool:
    """Simulate one sweep over independent null tests.

    Under the null each test's p-value is Uniform(0, 1), so a test rejects with
    probability equal to its level. The sweep "fires" if any test rejects.
    """

    return any(rng.random() < per_test_level for _ in range(family_size))


class TestUncorrectedSweepIsBroken:
    """The defect: an uncorrected multi-key sweep does not control error."""

    @pytest.mark.parametrize(
        ("family_size", "minimum_observed_rate"),
        [(10, 0.30), (100, 0.90), (1000, 0.99)],
    )
    def test_uncorrected_sweep_inflates_family_wise_error(
        self, family_size: int, minimum_observed_rate: float
    ) -> None:
        """Sweeping K configs at alpha=0.05 fires on noise far more than 5%.

        This is the regression guard for the war-dial defect. A sweep that
        reports per-configuration significance without declaring a family is
        not reporting a 5% false-positive rate, and at K=1000 it is wrong on
        essentially every null document.
        """

        rng = random.Random(20260830 + family_size)
        trials = 2000
        fired = sum(_sweep_fires(0.05, family_size, rng) for _ in range(trials))
        observed = fired / trials

        assert observed > minimum_observed_rate, (
            f"K={family_size} uncorrected sweep fired on {observed:.1%} of null documents; "
            "the per-test level is not the family-wise level."
        )

        analytic = family_wise_error_rate(0.05, family_size)
        assert abs(observed - analytic) < 0.05

    def test_family_wise_rate_matches_closed_form(self) -> None:
        # 1 - 0.95 ** 100 ~= 0.9941
        assert family_wise_error_rate(0.05, 100) == pytest.approx(0.99408, abs=1e-4)
        assert family_wise_error_rate(0.05, 1) == pytest.approx(0.05)


class TestCorrectionRestoresControl:
    """With a declared family, the sweep holds its nominal rate."""

    @pytest.mark.parametrize("family_size", [1, 10, 100, 1000])
    def test_corrected_sweep_holds_family_wise_rate(self, family_size: int) -> None:
        """A pure-noise corpus swept at K configurations must hold alpha_family."""

        family = declare_sweep_family(
            family_level=0.05,
            family_size=family_size,
            method="bonferroni",
            dimensions=("key_id",),
        )
        rng = random.Random(31337 + family_size)
        trials = 4000
        fired = sum(_sweep_fires(family.per_test_level, family_size, rng) for _ in range(trials))
        observed = fired / trials

        # Bonferroni is conservative, so the observed rate must not exceed the
        # nominal family level beyond Monte-Carlo noise.
        assert observed <= 0.05 + 0.015, (
            f"K={family_size} corrected sweep fired on {observed:.1%}, above the 5% family level."
        )

    @pytest.mark.parametrize("family_size", [1, 2, 10, 997, 100_000])
    def test_sidak_is_exact_under_independence(self, family_size: int) -> None:
        per_test = sidak_threshold(0.05, family_size)
        assert family_wise_error_rate(per_test, family_size) == pytest.approx(0.05, abs=1e-12)

    @pytest.mark.parametrize("family_size", [1, 2, 10, 997, 100_000])
    def test_bonferroni_is_more_conservative_than_sidak(self, family_size: int) -> None:
        bonferroni = bonferroni_threshold(0.05, family_size)
        sidak = sidak_threshold(0.05, family_size)
        assert bonferroni <= sidak
        if family_size > 1:
            assert bonferroni < sidak

    def test_family_size_one_is_a_no_op(self) -> None:
        assert sidak_threshold(0.05, 1) == pytest.approx(0.05)
        assert bonferroni_threshold(0.05, 1) == pytest.approx(0.05)

    def test_small_levels_do_not_underflow(self) -> None:
        """log1p/expm1 formulation must stay accurate at a 0.1% family level."""

        per_test = sidak_threshold(0.001, 100_000)
        assert per_test > 0.0
        assert family_wise_error_rate(per_test, 100_000) == pytest.approx(0.001, abs=1e-12)


class TestSweepFamilyRecord:
    def test_declares_dimensions_and_carries_the_uncorrected_rate(self) -> None:
        family = declare_sweep_family(
            family_level=0.05,
            family_size=240,
            method="bonferroni",
            dimensions=("key_id", "gamma", "delta", "context_width"),
        )
        assert isinstance(family, SweepFamily)
        assert family.per_test_level == pytest.approx(0.05 / 240)
        assert family.uncorrected_family_wise_rate > 0.99
        assert family.dimensions == ("key_id", "gamma", "delta", "context_width")
        assert "arbitrary dependence" in family.note

        payload = family.as_dict()
        assert payload["family_size"] == 240
        assert payload["dimensions"] == ["key_id", "gamma", "delta", "context_width"]

    def test_uncorrected_is_permitted_but_labelled(self) -> None:
        """An uncorrected baseline is allowed for ablation, never silently."""

        family = declare_sweep_family(family_level=0.05, family_size=100, method="none")
        assert family.per_test_level == pytest.approx(0.05)
        assert family.method == "none"
        assert "multiplicity artifact" in family.note

    def test_bonferroni_is_the_default(self) -> None:
        family = declare_sweep_family(family_level=0.05, family_size=50)
        assert family.method == "bonferroni"

    @pytest.mark.parametrize(
        ("level", "size"),
        [(0.0, 10), (1.0, 10), (-0.1, 10), (0.05, 0), (0.05, -1)],
    )
    def test_rejects_invalid_families(self, level: float, size: int) -> None:
        with pytest.raises(MultiplicityError):
            declare_sweep_family(family_level=level, family_size=size)

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(MultiplicityError):
            declare_sweep_family(family_level=0.05, family_size=10, method="holm")  # type: ignore[arg-type]

    def test_rejects_boolean_family_size(self) -> None:
        with pytest.raises(MultiplicityError):
            declare_sweep_family(family_level=0.05, family_size=True)  # type: ignore[arg-type]


class TestWindowScanning:
    def test_window_scan_requires_simulation(self) -> None:
        """Overlapping windows admit no analytic document-level bound."""

        family = per_window_target_level(family_level=0.01, window_count=5000)
        assert family.requires_simulation is True
        assert family.per_test_level == pytest.approx(1e-6)
        assert "simulation" in family.note

    def test_window_family_level_is_a_target_not_a_result(self) -> None:
        family = per_window_target_level(family_level=0.01, window_count=100)
        # The recorded family level is what is targeted; the analytic rate under
        # a (false) independence assumption is reported separately so the two
        # are never conflated.
        assert family.family_level == pytest.approx(0.01)
        assert family.uncorrected_family_wise_rate == pytest.approx(1e-4, rel=1e-3)


class TestEffectiveSampleSize:
    def test_independent_units_are_unchanged(self) -> None:
        result = effective_sample_size(
            n_observations=3000, n_clusters=3000, intracluster_correlation=0.4
        )
        assert result.design_effect == pytest.approx(1.0)
        assert result.n_effective == pytest.approx(3000)

    def test_clustering_reduces_effective_size(self) -> None:
        """40 passages from each of 100 documents is not 4000 independent units."""

        result = effective_sample_size(
            n_observations=4000, n_clusters=100, intracluster_correlation=0.5
        )
        assert result.mean_cluster_size == pytest.approx(40.0)
        assert result.design_effect == pytest.approx(1.0 + 39.0 * 0.5)
        assert result.n_effective < 200
        assert result.n_effective == pytest.approx(4000 / 20.5)

    def test_zero_correlation_is_a_no_op(self) -> None:
        result = effective_sample_size(
            n_observations=1000, n_clusters=10, intracluster_correlation=0.0
        )
        assert result.n_effective == pytest.approx(1000)

    @pytest.mark.parametrize(
        ("n_obs", "n_clusters", "rho"),
        [(0, 1, 0.1), (10, 0, 0.1), (10, 11, 0.1), (10, 5, -0.1), (10, 5, 1.1)],
    )
    def test_rejects_invalid_inputs(self, n_obs: int, n_clusters: int, rho: float) -> None:
        with pytest.raises(MultiplicityError):
            effective_sample_size(
                n_observations=n_obs, n_clusters=n_clusters, intracluster_correlation=rho
            )


class TestWorstGroupRate:
    def test_surfaces_the_worst_stratum_not_the_pool(self) -> None:
        """A pooled rate can look fine while one group is badly served."""

        result = worst_group_rate(
            events_by_stratum={"en": 1, "zh-Hant-TW": 40},
            totals_by_stratum={"en": 1000, "zh-Hant-TW": 1000},
        )
        assert result.pooled_rate == pytest.approx(41 / 2000)
        assert result.worst_group == "zh-Hant-TW"
        assert result.worst_group_rate == pytest.approx(0.04)
        assert result.worst_group_rate > result.pooled_rate

    def test_flags_underpowered_strata(self) -> None:
        """The 2,995 floor applies per stratum, not to the pool."""

        result = worst_group_rate(
            events_by_stratum={"en": 0, "zh-Hant-TW": 0},
            totals_by_stratum={"en": 3000, "zh-Hant-TW": 40},
            minimum_per_stratum=2995,
        )
        assert result.underpowered_strata == ("zh-Hant-TW",)

    def test_no_underpowered_strata_when_all_meet_floor(self) -> None:
        result = worst_group_rate(
            events_by_stratum={"en": 0, "zh-Hant-TW": 0},
            totals_by_stratum={"en": 3000, "zh-Hant-TW": 3000},
            minimum_per_stratum=2995,
        )
        assert result.underpowered_strata == ()

    def test_rejects_mismatched_strata(self) -> None:
        with pytest.raises(MultiplicityError):
            worst_group_rate(
                events_by_stratum={"en": 1},
                totals_by_stratum={"zh-Hant-TW": 10},
            )

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(MultiplicityError):
            worst_group_rate(events_by_stratum={}, totals_by_stratum={})

    def test_rejects_events_exceeding_total(self) -> None:
        with pytest.raises(MultiplicityError):
            worst_group_rate(
                events_by_stratum={"en": 11},
                totals_by_stratum={"en": 10},
            )
