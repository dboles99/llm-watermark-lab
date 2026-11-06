"""Tests for the robust-plus-fragile dual-channel frontier feature."""

from __future__ import annotations

import pytest

from watermark_lab.frontier.robust_fragile import (
    InvariantKind,
    RobustFragileError,
    RobustFragileWatermarker,
    SemanticInvariant,
)


def _make_invariants() -> tuple[SemanticInvariant, ...]:
    return (
        SemanticInvariant(
            invariant_id="number-42",
            kind=InvariantKind.NUMBER,
            pattern=r"\b42\b",
            description="The number 42 must not change.",
        ),
        SemanticInvariant(
            invariant_id="citation-smith",
            kind=InvariantKind.CITATION,
            pattern=r"Smith et al\.",
            description="Citation to Smith et al. must be preserved.",
        ),
    )


class TestRobustFragileWatermarker:
    def test_both_channels_intact(self) -> None:
        wm = RobustFragileWatermarker(_make_invariants())
        result = wm.verify(
            text="The answer is 42 as shown by Smith et al. in their study.",
            payload="project-abc",
        )
        assert result.classification == "provenance-intact-semantics-intact"
        assert result.fragile_channel.intact is True
        assert result.robust_channel.survives is True

    def test_fragile_channel_violated_by_missing_number(self) -> None:
        wm = RobustFragileWatermarker(_make_invariants())
        result = wm.verify(
            text="The answer is 43 as shown by Smith et al. in their study.",
            payload="project-abc",
        )
        assert result.classification == "provenance-intact-semantics-violated"
        assert result.fragile_channel.intact is False
        assert "number-42" in result.fragile_channel.violated_invariants

    def test_fragile_channel_violated_by_missing_citation(self) -> None:
        wm = RobustFragileWatermarker(_make_invariants())
        result = wm.verify(
            text="The answer is 42 as shown by Jones et al. in their study.",
            payload="project-abc",
        )
        assert result.fragile_channel.intact is False
        assert "citation-smith" in result.fragile_channel.violated_invariants

    def test_robust_channel_lost(self) -> None:
        wm = RobustFragileWatermarker(_make_invariants())
        result = wm.verify(
            text="The answer is 42 as shown by Smith et al. in their study.",
            payload="project-abc",
            robust_survives=False,
        )
        assert result.classification == "provenance-lost-semantics-intact"
        assert result.robust_channel.survives is False

    def test_both_channels_lost(self) -> None:
        wm = RobustFragileWatermarker(_make_invariants())
        result = wm.verify(
            text="The answer is 99 as shown by Nobody.",
            payload="project-abc",
            robust_survives=False,
        )
        assert result.classification == "provenance-lost-semantics-violated"

    def test_empty_invariants_rejected(self) -> None:
        with pytest.raises(RobustFragileError, match="At least one"):
            RobustFragileWatermarker([])

    def test_duplicate_invariant_ids_rejected(self) -> None:
        inv = SemanticInvariant("id1", InvariantKind.NUMBER, r"\b1\b", "test")
        with pytest.raises(RobustFragileError, match="unique"):
            RobustFragileWatermarker([inv, inv])

    def test_result_serializable(self) -> None:
        wm = RobustFragileWatermarker(_make_invariants())
        result = wm.verify(text="42 Smith et al.", payload="p")
        data = result.as_dict()
        assert data["scientific_status"] == "skeleton-uncalibrated"
        assert isinstance(data["warnings"], list)

    def test_localization_spans_present_on_violation(self) -> None:
        wm = RobustFragileWatermarker(_make_invariants())
        result = wm.verify(
            text="The answer is 43 and the citation is Jones et al.",
            payload="p",
        )
        assert len(result.fragile_channel.violated_invariants) == 2