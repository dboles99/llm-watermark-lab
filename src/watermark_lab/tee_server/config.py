"""Fail-closed configuration for the local TEE simulator server."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LOCAL_DEVELOPMENT_MODE = "local-development"
DEFAULT_SECRET_FILE = Path("/run/secrets/tee_simulator_key")
DEFAULT_MEASUREMENT_SHA256 = hashlib.sha256(
    b"watermark-lab-lab/local-non-hardware-tee-simulator/v1"
).hexdigest()
KeyLifecycle = Literal["ephemeral-process", "mounted-secret-file"]


@dataclass(frozen=True, slots=True)
class TeeServerConfig:
    """Configuration with no switch that can promote simulator evidence."""

    deployment_mode: str = LOCAL_DEVELOPMENT_MODE
    secret_file: Path = DEFAULT_SECRET_FILE
    measurement_sha256: str = DEFAULT_MEASUREMENT_SHA256

    def __post_init__(self) -> None:
        if self.deployment_mode != LOCAL_DEVELOPMENT_MODE:
            raise RuntimeError(
                "The local TEE test server contains only a simulator and refuses to start "
                "outside local-development mode. It cannot run in production or staging."
            )

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> TeeServerConfig:
        """Read only the non-secret deployment classification from the environment.

        Signing-key bytes and their path are deliberately not configurable through
        environment variables or command-line arguments. The optional Docker secret
        is read only from the fixed ``/run/secrets/tee_simulator_key`` path.
        """

        source = os.environ if environ is None else environ
        return cls(deployment_mode=source.get("WATERMARK_LAB_TEE_DEPLOYMENT_MODE", LOCAL_DEVELOPMENT_MODE))


def load_simulator_signing_key(secret_file: Path) -> tuple[bytes, KeyLifecycle]:
    """Load a fixed-path secret file, or create a process-ephemeral simulator key."""

    try:
        file_stat = secret_file.stat()
    except FileNotFoundError:
        return secrets.token_bytes(32), "ephemeral-process"
    except OSError as exc:
        raise RuntimeError("The simulator secret file could not be inspected.") from exc

    if secret_file.is_symlink():
        raise RuntimeError("The simulator secret path must not be a symbolic link.")
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError("The simulator secret path must be a regular mounted file.")
    if file_stat.st_size < 32 or file_stat.st_size > 4096:
        raise RuntimeError("The simulator secret file must contain between 32 and 4096 bytes.")

    try:
        signing_key = secret_file.read_bytes()
    except OSError as exc:
        raise RuntimeError("The simulator secret file could not be read.") from exc
    if len(signing_key) < 32:
        raise RuntimeError("The simulator secret file must contain at least 32 bytes.")
    return signing_key, "mounted-secret-file"
