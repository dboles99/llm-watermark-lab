"""Tests for the PolyMark demixing frontier feature."""

from __future__ import annotations

import pytest

from watermark_lab.frontier.polymark import (
    PolymarkDemixer,
    PolymarkDemixingError,
    PolymarkProcess,
)


def _make_processes() -> tuple[PolymarkProcess, ...]:
    return (
        PolymarkProcess("p-a", "provider-a", "epoch-001", 0.25, 4.0),
        PolymarkProcess("p-b", "provider-b", "epoch-002", 0.25, 4.0),
    )


class TestPolymarkDemixer:
    def test_empty_processes_rejected(self) -> None:
        with pytest.raises(PolymarkDemixingError, match="At least one"):
            PolymarkDemixer([])

    def test_duplicate_process_ids_rejected(self) -> None:
        with pytest.raises(PolymarkDemixingError, match="unique"):
            PolymarkDemixer(
                [PolymarkProcess("p", "a", "e", 0.25, 4.0),
                 PolymarkProcess("p", "b", "e", 0.25, 4.0)]
            )

    def test_invalid_gamma_rejected(self) -> None:
        with pytest.raises(PolymarkDemixingError, match="gamma"):
            PolymarkDemixer([PolymarkProcess("p", "a", "e", 1.5, 4.0)])

    def test_abstention_when_no_process_reaches_threshold(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        result = demixer.demix([
            ("p-a", 0, 50, 1.0, 0.15),
            ("p-b", 50, 100, 2.0, 0.02),
        ])
        assert result.abstained is True
        assert result.process_count == 0
        assert result.scientific_status == "skeleton-uncalibrated"

    def test_significant_spans_detected(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        result = demixer.demix([
            ("p-a", 0, 50, 5.0, 0.0001),
            ("p-b", 50, 100, 1.0, 0.15),
        ])
        assert result.process_count == 1
        assert result.detected_processes[0].process_id == "p-a"
        assert len(result.spans) == 1
        assert result.spans[0].process_id == "p-a"

    def test_overlap_counted(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        result = demixer.demix([
            ("p-a", 0, 60, 5.0, 0.0001),
            ("p-b", 40, 100, 5.0, 0.0001),
        ])
        assert result.overlap_count >= 1
        assert "overlapping" in result.warnings[3] or any("overlapping" in w for w in result.warnings)

    def test_collision_counted_for_different_processes(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        result = demixer.demix([
            ("p-a", 0, 50, 5.0, 0.0001),
            ("p-b", 30, 80, 5.0, 0.0001),
        ])
        assert result.collision_count >= 1

    def test_unknown_process_rejected(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        with pytest.raises(PolymarkDemixingError, match="Unknown process"):
            demixer.demix([("p-c", 0, 50, 5.0, 0.0001)])

    def test_invalid_span_rejected(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        with pytest.raises(PolymarkDemixingError, match="span"):
            demixer.demix([("p-a", 50, 50, 5.0, 0.0001)])

    def test_result_serializable(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        result = demixer.demix([("p-a", 0, 50, 5.0, 0.0001)])
        data = result.as_dict()
        assert isinstance(data["spans"], list)
        assert data["scientific_status"] == "skeleton-uncalibrated"

    def test_warnings_present(self) -> None:
        demixer = PolymarkDemixer(_make_processes())
        result = demixer.demix([("p-a", 0, 50, 5.0, 0.0001)])
        assert len(result.warnings) >= 3
        assert any("Skeleton" in w for w in result.warnings)