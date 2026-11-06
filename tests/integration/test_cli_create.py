"""Integration tests for the ``watermark-lab create`` CLI."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from watermark_lab.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_create_command_succeeds_with_disclosure(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "create",
            "--prompt",
            "hello world",
            "--profile",
            "mock",
            "--method",
            "kgw",
            "--key-id",
            "test-kgw-v1",
            "--disclosure",
            "--max-new-tokens",
            "80",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "completed"
    assert payload["method_id"] == "kgw"
    assert payload["key_id"] == "test-kgw-v1"
    assert payload["self_verification"]["decision"] == "verified compatible watermark"
    assert len(payload["prompt_sha256"]) == 64
    assert len(payload["response_sha256"]) == 64


def test_create_command_fails_without_disclosure(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "create",
            "--prompt",
            "hello world",
            "--profile",
            "mock",
            "--method",
            "kgw",
            "--key-id",
            "test-kgw-v1",
        ],
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert "Disclosure" in payload["error"]
