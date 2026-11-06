from __future__ import annotations

from pathlib import Path

import pytest

from watermark_lab.tee_server.config import (
    DEFAULT_SECRET_FILE,
    TeeServerConfig,
    load_simulator_signing_key,
)


@pytest.mark.parametrize("deployment_mode", ["production", "staging", "prod", "unknown"])
def test_tee_server_refuses_every_non_local_deployment_mode(deployment_mode: str) -> None:
    with pytest.raises(RuntimeError, match="refuses to start"):
        TeeServerConfig(deployment_mode=deployment_mode)


def test_tee_server_environment_cannot_select_key_or_secret_path() -> None:
    config = TeeServerConfig.from_environment(
        {
            "WATERMARK_LAB_TEE_DEPLOYMENT_MODE": "local-development",
            "WATERMARK_LAB_TEE_SIMULATOR_KEY": "must-be-ignored",
            "WATERMARK_LAB_TEE_SIMULATOR_KEY_FILE": "must-also-be-ignored",
        }
    )
    assert config.secret_file == DEFAULT_SECRET_FILE


def test_tee_server_production_environment_fails_during_configuration() -> None:
    with pytest.raises(RuntimeError, match="refuses to start"):
        TeeServerConfig.from_environment({"WATERMARK_LAB_TEE_DEPLOYMENT_MODE": "production"})


def test_tee_server_uses_ephemeral_key_when_secret_file_is_absent(tmp_path: Path) -> None:
    first, first_lifecycle = load_simulator_signing_key(tmp_path / "missing")
    second, second_lifecycle = load_simulator_signing_key(tmp_path / "missing")

    assert len(first) == 32
    assert len(second) == 32
    assert first != second
    assert first_lifecycle == second_lifecycle == "ephemeral-process"


def test_tee_server_rejects_invalid_mounted_secret_without_fallback(tmp_path: Path) -> None:
    secret_file = tmp_path / "tee_simulator_key"
    secret_file.write_bytes(b"too-short")

    with pytest.raises(RuntimeError, match="between 32 and 4096 bytes"):
        load_simulator_signing_key(secret_file)
