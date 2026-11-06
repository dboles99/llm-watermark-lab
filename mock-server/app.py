from __future__ import annotations

import hashlib
import os
import random
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Simulated Provider", version="0.1.0")
API_KEY = os.getenv("MOCK_PROVIDER_API_KEY", "local-development-only")
BASE_SEED = int(os.getenv("MOCK_SEED", "20260814"))


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class MessageRequest(BaseModel):
    model: str = "mock-claude-watermark-lab"
    max_tokens: int = Field(default=512, ge=1, le=8192)
    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)


def authorize(x_api_key: str | None) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid mock API key")


def render_text(request: MessageRequest) -> str:
    joined = "\n".join(str(message.content) for message in request.messages)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    rng = random.Random(BASE_SEED ^ int(digest[:12], 16))
    profile = request.metadata.get("watermark_profile", "clean")
    vocabulary = [
        "analysis", "evidence", "method", "context", "result", "calibration",
        "language", "sample", "threshold", "uncertainty", "review", "pattern",
    ]
    tokens = [rng.choice(vocabulary) for _ in range(min(request.max_tokens, 96))]
    return (
        f"[SIMULATED PROVIDER: {profile}] "
        + " ".join(tokens)
        + ". This output is synthetic and must not be attributed to Anthropic."
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "simulated": True}


@app.post("/v1/messages")
def anthropic_shaped_messages(
    request: MessageRequest,
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    authorize(x_api_key)
    text = render_text(request)
    return {
        "id": f"msg_mock_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": len(text.split())},
        "simulated_provider": True,
        "created": int(time.time()),
    }


@app.post("/v1/chat/completions")
def openai_shaped_messages(
    request: MessageRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid mock bearer token")
    text = render_text(request)
    return {
        "id": f"chatcmpl_mock_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": len(text.split())},
        "simulated_provider": True,
    }

