"""Real-model watermarked generation against a pinned local model and tokenizer.

This replaces the synthetic token-ID text that `watermarkers/kgw_generator.py`
emits. That generator could never support a scientific claim: text made of
whitespace-separated integers has none of the entropy structure, repetition, or
tokenizer behaviour that determines whether a watermark is detectable.

Two properties are load-bearing here.

**Offline by construction.** Every load passes ``local_files_only=True``. A
silent download would mean the recorded model identity describes weights that
were fetched at run time from a mutable remote, which is exactly the provenance
failure the project exists to prevent. Absent weights raise instead.

**Author-matching defaults.** Hugging Face's ``WatermarkingConfig`` defaults are
the KGW author's: ``greenlist_ratio=0.25``, ``bias=2.0``,
``hashing_key=15485863``, ``seeding_scheme="lefthash"``, ``context_width=1``.
That makes this path a closer parity target than third-party reimplementations,
some of which reduce the PRF seed modulo the vocabulary size and interpose a
permutation table.

Generation still is not a promotion. Detectability depends on the sampling
temperature, prompt entropy, and repeated-context policy, and none of those has
a frozen calibration yet — so results carry ``scientific_status`` and the
detector cross-check is reported as mechanics evidence, not as an operating
point.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Author-matching KGW defaults, restated so a caller can see them without
#: reading Hugging Face internals, and so a drift in their defaults is visible.
AUTHOR_KGW_DEFAULTS: dict[str, Any] = {
    "greenlist_ratio": 0.25,
    "bias": 2.0,
    "hashing_key": 15485863,
    "seeding_scheme": "lefthash",
    "context_width": 1,
}

MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

NOT_PROMOTABLE_NOTE = (
    "Real-model generation is mechanics evidence, not a calibrated result. Detectability "
    "depends on decoding temperature, prompt entropy, and the repeated-context policy, none "
    "of which has a frozen calibration. No estimated FPR is implied."
)


class RealGenerationError(RuntimeError):
    """Raised when pinned generation cannot proceed reproducibly."""


def _digest_directory(directory: Path, filenames: tuple[str, ...]) -> str:
    """Hash the named files in a directory, in sorted order, as one fingerprint."""

    digest = hashlib.sha256()
    found = False
    for name in sorted(filenames):
        path = directory / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest() if found else ""


def find_model_weights(model_dir: Path) -> tuple[str, ...]:
    """Return which recognised weight files exist in ``model_dir``."""

    return tuple(name for name in MODEL_WEIGHT_FILENAMES if (model_dir / name).is_file())


@dataclass(frozen=True, slots=True)
class GenerationProvenance:
    """Everything needed to reproduce or refuse to trust a generation."""

    model_dir: str
    model_id: str
    weight_files: tuple[str, ...]
    weights_sha256: str
    tokenizer_dir: str
    tokenizer_files_sha256: str
    watermarking_config: dict[str, Any]
    generation_config: dict[str, Any]
    seed: int
    transformers_version: str
    torch_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "model_id": self.model_id,
            "weight_files": list(self.weight_files),
            "weights_sha256": self.weights_sha256,
            "tokenizer_dir": self.tokenizer_dir,
            "tokenizer_files_sha256": self.tokenizer_files_sha256,
            "watermarking_config": self.watermarking_config,
            "generation_config": self.generation_config,
            "seed": self.seed,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
        }


@dataclass(frozen=True, slots=True)
class RealGenerationResult:
    """Watermarked text produced by a pinned local model."""

    text: str
    prompt: str
    token_ids: tuple[int, ...]
    new_token_count: int
    provenance: GenerationProvenance
    self_check: dict[str, Any] | None = None
    scientific_status: str = "mechanics-only"
    warnings: tuple[str, ...] = field(default=())

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_sha256": self.text_sha256,
            "prompt": self.prompt,
            "new_token_count": self.new_token_count,
            "token_ids": list(self.token_ids),
            "provenance": self.provenance.as_dict(),
            "self_check": self.self_check,
            "scientific_status": self.scientific_status,
            "not_promotable": NOT_PROMOTABLE_NOTE,
            "warnings": list(self.warnings),
        }


class PinnedWatermarkGenerator:
    """Generate watermarked text with a pinned, local, offline model."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        tokenizer_dir: str | Path | None = None,
        watermarking_config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RealGenerationError(
                "Real-model generation requires torch and transformers. Install the "
                "'generation' extra, or use the mock provider."
            ) from exc

        self._torch = torch
        self._transformers = transformers

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise RealGenerationError(f"Model directory does not exist: {self.model_dir}")

        weights = find_model_weights(self.model_dir)
        if not weights:
            raise RealGenerationError(
                f"No model weights found in {self.model_dir}. Expected one of: "
                f"{', '.join(MODEL_WEIGHT_FILENAMES)}. The pinned tokenizer assets are "
                "separate from the weights; a tokenizer-only directory cannot generate. "
                "Weights are never downloaded automatically, because a silent fetch would "
                "make the recorded model identity describe mutable remote bytes."
            )
        self.weight_files = weights

        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else self.model_dir
        if not self.tokenizer_dir.is_dir():
            raise RealGenerationError(f"Tokenizer directory does not exist: {self.tokenizer_dir}")

        self.watermarking_config = {**AUTHOR_KGW_DEFAULTS, **(watermarking_config or {})}

        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self.tokenizer_dir), local_files_only=True
            )
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
        except Exception as exc:
            raise RealGenerationError(
                f"Could not load the pinned model or tokenizer offline: {exc}"
            ) from exc

        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def model_id(self) -> str:
        return str(getattr(self._model.config, "name_or_path", "") or self.model_dir.name)

    def _provenance(self, generation_config: dict[str, Any], seed: int) -> GenerationProvenance:
        tokenizer_files = (
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        return GenerationProvenance(
            model_dir=str(self.model_dir.resolve()),
            model_id=self.model_id,
            weight_files=self.weight_files,
            weights_sha256=_digest_directory(self.model_dir, self.weight_files),
            tokenizer_dir=str(self.tokenizer_dir.resolve()),
            tokenizer_files_sha256=_digest_directory(self.tokenizer_dir, tokenizer_files),
            watermarking_config=dict(self.watermarking_config),
            generation_config=dict(generation_config),
            seed=seed,
            transformers_version=self._transformers.__version__,
            torch_version=self._torch.__version__,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        seed: int = 0,
        watermarked: bool = True,
        self_check: bool = True,
    ) -> RealGenerationResult:
        """Generate text, optionally watermarked, from a pinned local model.

        ``watermarked=False`` produces the matched negative control from the same
        model, prompt and seed. Without that control a positive detection cannot
        be distinguished from a property of the model or the prompt.
        """

        torch = self._torch
        transformers = self._transformers

        if not isinstance(prompt, str) or not prompt.strip():
            raise RealGenerationError("prompt must be a non-empty string.")
        if max_new_tokens < 1:
            raise RealGenerationError("max_new_tokens must be at least 1.")

        generation_config: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "watermarked": watermarked,
        }

        inputs = self._tokenizer(prompt, return_tensors="pt")
        prompt_length = int(inputs["input_ids"].shape[-1])

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            if top_k:
                kwargs["top_k"] = top_k
        if watermarked:
            kwargs["watermarking_config"] = transformers.WatermarkingConfig(
                **self.watermarking_config
            )

        torch.manual_seed(seed)
        with torch.no_grad():
            output = self._model.generate(**inputs, **kwargs)  # type: ignore[misc]

        sequence = output[0]
        new_tokens = sequence[prompt_length:]
        decoded = self._tokenizer.decode(sequence, skip_special_tokens=True)
        # decode() is annotated str | list[str] upstream; a single sequence
        # always yields a str, but narrow explicitly rather than assume.
        text = decoded if isinstance(decoded, str) else "".join(decoded)

        warnings: list[str] = []
        if int(new_tokens.shape[-1]) < max_new_tokens:
            warnings.append(
                "generation stopped early, so the sample is shorter than requested; short "
                "samples are the regime where detection is weakest"
            )

        check: dict[str, Any] | None = None
        if self_check and watermarked:
            check = self._self_check(new_tokens.unsqueeze(0))

        return RealGenerationResult(
            text=text,
            prompt=prompt,
            token_ids=tuple(int(value) for value in sequence.tolist()),
            new_token_count=int(new_tokens.shape[-1]),
            provenance=self._provenance(generation_config, seed),
            self_check=check,
            warnings=tuple(warnings),
        )

    def _self_check(self, token_ids: Any) -> dict[str, Any]:
        """Score generated tokens with Hugging Face's own detector.

        This is a mechanics cross-check: it verifies the processor actually
        biased token selection. It is *not* a calibrated decision, and its
        prediction must never be reported as an operating point.
        """

        transformers = self._transformers
        try:
            detector = transformers.WatermarkDetector(
                model_config=self._model.config,
                device="cpu",
                watermarking_config=transformers.WatermarkingConfig(**self.watermarking_config),
            )
            outcome = detector(token_ids, return_dict=True)
            prediction = getattr(outcome, "prediction", None)
            z_score = getattr(outcome, "num_tokens_scored", None)
            return {
                "detector": "transformers.WatermarkDetector",
                "prediction": bool(prediction[0]) if prediction is not None else None,
                "num_tokens_scored": (
                    int(z_score[0]) if z_score is not None else None
                ),
                "note": (
                    "Mechanics cross-check only: confirms the logits processor biased token "
                    "selection. Not a calibrated decision and not an operating point."
                ),
            }
        except Exception as exc:
            return {
                "detector": "transformers.WatermarkDetector",
                "error": str(exc),
                "note": "Self-check failed; generation output is still returned unchanged.",
            }


def describe_environment() -> dict[str, Any]:
    """Report whether real-model generation is possible here, and what is missing."""

    payload: dict[str, Any] = {"torch": None, "transformers": None, "ready": False}
    try:
        import torch

        payload["torch"] = torch.__version__
    except ImportError:
        payload["missing"] = "torch"
        return payload
    try:
        import transformers

        payload["transformers"] = transformers.__version__
    except ImportError:
        payload["missing"] = "transformers"
        return payload

    payload["author_kgw_defaults"] = dict(AUTHOR_KGW_DEFAULTS)
    payload["ready"] = True
    payload["note"] = (
        "torch and transformers are available. A pinned local model directory containing "
        "weights is still required; weights are never downloaded automatically."
    )
    return payload


def write_generation_record(result: RealGenerationResult, path: Path) -> Path:
    """Persist a generation result as a JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


__all__ = [
    "AUTHOR_KGW_DEFAULTS",
    "MODEL_WEIGHT_FILENAMES",
    "NOT_PROMOTABLE_NOTE",
    "GenerationProvenance",
    "PinnedWatermarkGenerator",
    "RealGenerationError",
    "RealGenerationResult",
    "describe_environment",
    "find_model_weights",
    "write_generation_record",
]
