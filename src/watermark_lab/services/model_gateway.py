from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    provider: str
    model: str
    base_url: str | None = None
    deployment: str | None = None
    simulated: bool = False


@dataclass(frozen=True)
class ModelEvidence:
    profile_id: str
    prompt_sha256: str
    response_sha256: str
    hypothesis: str
    proposed_feature: str
    attribution_allowed: bool = False


class ModelClient(Protocol):
    async def generate(self, messages: list[dict[str, Any]], profile: ModelProfile) -> str: ...

