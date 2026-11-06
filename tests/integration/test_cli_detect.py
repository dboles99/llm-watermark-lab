"""Integration tests for the ``watermark-lab detect`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from watermark_lab.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_detect_command_short_text(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "detect",
            "--input",
            "hello world",
            "--method",
            "kgw",
            "--key-id",
            "test-kgw-v1",
            "--threshold",
            "4.0",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["method_id"] == "kgw"
    assert payload["compatibility"] == "experimental"
    assert payload["decision"] == "inconclusive"
    assert payload["estimated_fpr"] is None
    assert any("inconclusive" in w for w in payload["warnings"])


def test_detect_command_longer_text(runner: CliRunner) -> None:
    text = " ".join([f"token{i}" for i in range(80)])
    result = runner.invoke(
        app,
        [
            "detect",
            "--input",
            text,
            "--method",
            "kgw",
            "--key-id",
            "test-kgw-v1",
            "--threshold",
            "4.0",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["method_id"] == "kgw"
    assert payload["decision"] in {"verified compatible watermark", "not detected", "inconclusive"}
    assert payload["token_count"] == 79


def test_detect_command_via_console_script(tmp_path: Path) -> None:
    """Smoke test that the installed console script runs."""
    proc = subprocess.run(
        [sys.executable, "-m", "watermark_lab.cli", "detect", "--input", "hello world"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["method_id"] == "kgw"


def test_detect_command_surfaces_uncalibrated_repeated_ngram_diagnostics(
    runner: CliRunner,
) -> None:
    result = runner.invoke(
        app,
        [
            "detect",
            "--input",
            "5 6 5 6 7",
            "--ignore-repeated-ngrams",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["compatibility"] == "prototype"
    assert payload["decision"] == "inconclusive"
    assert payload["token_count"] == 3
    assert payload["original_scorable_token_count"] == 4
    assert payload["ignored_repeated_ngrams"] == 1
    assert payload["scoring_variant"] == "unique-ngram-uncalibrated"
    assert payload["calibration_id"] is None


def test_detect_command_scientific_flag_without_assets(runner: CliRunner) -> None:
    """Scientific mode should fail gracefully if tokenizer assets are missing."""

    result = runner.invoke(
        app,
        [
            "detect",
            "--input",
            "The quick brown fox jumps over the lazy dog.",
            "--scientific",
            "--locale",
            "en",
            "--tokenizer-assets",
            "/nonexistent/path",
        ],
    )
    assert result.exit_code == 1, result.output
    # Transformers may print warnings to stdout; extract the JSON payload
    json_start = result.output.find("{")
    assert json_start >= 0, result.output
    payload = json.loads(result.output[json_start:])
    assert payload.get("mode") == "scientific"
    assert payload.get("status") == "failed"
    assert "error" in payload


def test_detect_command_attack_command(runner: CliRunner) -> None:
    """The attack CLI command should produce a pre/post comparison."""

    text = " ".join([f"word{i}" for i in range(60)])
    result = runner.invoke(
        app,
        [
            "attack",
            "--text",
            text,
            "--attack-id",
            "deletion",
            "--severity",
            "0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_type"] == "transform"
    assert payload["attack_id"] == "deletion"
    assert "pre_attack" in payload
    assert "post_attack" in payload
    assert "survived" in payload


def test_detect_command_attack_sweep_command(runner: CliRunner) -> None:
    """The attack-sweep CLI command should produce a survival curve."""

    text = " ".join([f"word{i}" for i in range(60)])
    result = runner.invoke(
        app,
        [
            "attack-sweep",
            "--text",
            text,
            "--attack-id",
            "deletion",
            "--severities",
            "0.0,0.2,0.4",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["attack_id"] == "deletion"
    assert len(payload["survival_curve"]) == 3
    assert all("severity" in item for item in payload["survival_curve"])
    assert all("survived" in item for item in payload["survival_curve"])
