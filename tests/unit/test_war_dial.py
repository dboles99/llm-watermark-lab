"""Tests for the watermark war-dial harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from watermark_lab.war_dial.harness import WarDialHarness
from watermark_lab.war_dial.reconcile import WarDialReconciler
from watermark_lab.war_dial.space import (
    KgwDial,
    PerturbationDial,
    StegoDial,
    WarDialCoordinate,
)


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "war-dial"


def test_fast_coding_test_passes(tmp_output: Path) -> None:
    harness = WarDialHarness(output_dir=tmp_output, coding_test_args=None)
    passed, stdout, stderr = harness.run_coding_test()
    assert passed
    assert "roundtrip ok" in stdout
    assert stderr == ""


def test_war_dial_runs_single_coordinate(tmp_output: Path) -> None:
    harness = WarDialHarness(output_dir=tmp_output, coding_test_args=None)
    cover = (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )
    coordinate = WarDialCoordinate(
        kgw=KgwDial(
            prompt="the quick brown fox",
            gamma=0.25,
            delta=2.0,
            context_width=1,
            max_new_tokens=40,
        ),
        stego=StegoDial(
            channel="unicode",
            cover=cover,
            payload="ok",
            secret="test-secret",
        ),
        perturbation=PerturbationDial("builtin-word-jitter", 0.05, "seed-1"),
    )
    result = harness.dial(coordinate, dial_id="test-0000")
    assert result.coding_passed
    assert result.kgw_created
    assert result.stego_encoded
    assert result.stego_decode_status == "ok"
    assert result.stego_verify_status == "verified"


def test_run_grid_appends_jsonl_and_respects_resume(tmp_output: Path) -> None:
    harness = WarDialHarness(output_dir=tmp_output, coding_test_args=None)
    cover = (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )
    coords = [
        WarDialCoordinate(
            stego=StegoDial(channel="unicode", cover=cover, payload="ok", secret="s"),
        )
    ]
    harness.run_grid(coords, resume=True)
    result_file = tmp_output / "war_dial_results.jsonl"
    assert result_file.exists()
    lines_before = len(result_file.read_text().splitlines())

    # Resume should skip already-completed dial IDs.
    harness.run_grid(coords, resume=True)
    lines_after = len(result_file.read_text().splitlines())
    assert lines_after == lines_before


def test_reconciler_produces_summary_and_next_grid(tmp_output: Path) -> None:
    harness = WarDialHarness(output_dir=tmp_output, coding_test_args=None)
    cover = (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )
    harness.run_grid(
        [
            WarDialCoordinate(
                stego=StegoDial(channel="unicode", cover=cover, payload="ok", secret="s"),
            ),
        ],
        resume=False,
    )
    reconciler = WarDialReconciler(tmp_output)
    report = reconciler.reconcile()
    assert report["summary"]["total_dials"] == 1
    assert "next_grid" in report
    assert "recommendation" in report["next_grid"]


def test_war_dial_runs_single_coordinate_with_attack_sweep(tmp_output: Path) -> None:
    """Attack sweep produces attack_results and a survival surface."""
    harness = WarDialHarness(output_dir=tmp_output, coding_test_args=None)
    cover = (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )
    coordinate = WarDialCoordinate(
        kgw=KgwDial(
            prompt="the quick brown fox",
            gamma=0.25,
            delta=2.0,
            context_width=1,
            max_new_tokens=40,
        ),
        stego=StegoDial(
            channel="unicode",
            cover=cover,
            payload="ok",
            secret="test-secret",
        ),
        perturbation=PerturbationDial("builtin-word-jitter", 0.05, "seed-1"),
    )
    result = harness.dial(coordinate, dial_id="test-0000", run_attack_sweep=True)
    assert result.coding_passed
    assert result.kgw_created
    assert result.attack_results
    assert result.survival_surface is not None
    assert result.survival_surface["sample_size"] > 0


def test_gate_report_is_four_state(tmp_output: Path) -> None:
    harness = WarDialHarness(output_dir=tmp_output, coding_test_args=None)
    cover = (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )
    harness.run_grid(
        [
            WarDialCoordinate(
                stego=StegoDial(channel="unicode", cover=cover, payload="ok", secret="s"),
            ),
        ],
        resume=False,
    )
    report = harness.gate_report()
    assert report["report_type"] == "four-state-gate"
    assert "artifact_complete" in report
    assert "checks_passed" in report
    assert "promotion_eligible" in report
    assert "deployment_ready" in report
    assert report["artifact_complete"] is True

    path = harness.write_gate_report()
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["report_type"] == "four-state-gate"
