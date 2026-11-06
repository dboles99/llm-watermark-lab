"""Differentially verify the local KGW profile against a pinned author checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

from watermark_lab.adapters.kgw_author import (
    AUTHOR_KGW_CPU_PROFILE,
    kgw_author_detect,
    kgw_author_green_list,
    ordered_token_ids_sha256,
)
from watermark_lab.run_engine.scientific_detect import ScientificKgwDetectionService
from watermark_lab.tokenization import (
    NormalizationPolicy,
    SpecialTokenPolicy,
    TokenizerSpec,
    UnicodeNormalization,
    load_local_huggingface_tokenizer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_author_checkout(path: Path) -> Path:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("author repository path must be a directory")
    profile = AUTHOR_KGW_CPU_PROFILE
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != profile.author_commit:
        raise ValueError(f"author checkout is {commit!r}, expected {profile.author_commit!r}")
    expected = {
        profile.author_source: profile.author_source_sha256,
        "alternative_prf_schemes.py": profile.prf_source_sha256,
    }
    for relative, wanted in expected.items():
        actual = _sha256(root / relative)
        if actual != wanted:
            raise ValueError(f"author source digest mismatch for {relative}: {actual}")
    return root


def _load_author_module(root: Path) -> Any:
    # Token-ID parity uses normalizers=[]; a tiny import stub avoids adding the
    # unrelated homoglyph corpus dependency to this bounded oracle runtime.
    normalizer_stub = types.ModuleType("normalizers")
    normalizer_stub.__dict__["normalization_strategy_lookup"] = lambda _name: lambda text: text
    sys.modules["normalizers"] = normalizer_stub
    sys.path.insert(0, str(root))
    return importlib.import_module("extended_watermark_processor")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--score-cases", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_814)
    parser.add_argument("--tokenizer-assets", type=Path)
    return parser


def _raw_text_parity(author: Any, torch: Any, asset_path: Path) -> tuple[int, str]:
    from transformers import AutoTokenizer

    revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    specs = {
        locale: TokenizerSpec(
            tokenizer_id="openai-community/gpt2",
            revision=revision,
            normalization=NormalizationPolicy(
                unicode_form=UnicodeNormalization.NONE,
                locale=locale,
            ),
            special_tokens=SpecialTokenPolicy(),
            expected_backend_versions=(
                ("huggingface_hub", "1.27.0"),
                ("tokenizers", "0.22.2"),
                ("transformers", "5.15.0"),
            ),
        )
        for locale in ("en", "zh-Hant-TW")
    }
    services = {
        locale: ScientificKgwDetectionService(
            load_local_huggingface_tokenizer(spec, asset_path)
        )
        for locale, spec in specs.items()
    }
    direct_tokenizer = AutoTokenizer.from_pretrained(
        str(asset_path.resolve(strict=True)),
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    detector = author.WatermarkDetector(
        vocab=list(direct_tokenizer.get_vocab().values()),
        gamma=0.25,
        seeding_scheme="simple_1",
        device=torch.device("cpu"),
        tokenizer=direct_tokenizer,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )
    texts = (
        ("en", "The quick brown fox jumps over the lazy dog, then returns home."),
        ("en", "Cafe\u0301 and café remain distinct when normalization is disabled."),
        ("en", "Line one.\nLine two contains repeated repeated words and punctuation!"),
        ("zh-Hant-TW", "繁體中文與臺灣用語必須保留，不能轉成簡體中文。"),
        ("zh-Hant-TW", "這份研究記錄模型、分詞器、金鑰時期與校準版本。"),
        ("zh-Hant-TW", "第一行。\n第二行包含數字 2026、標點，以及英文 API。"),
    )
    aggregate = hashlib.sha256()
    for case_index, (locale, text) in enumerate(texts):
        raw_tokens = tuple(
            int(value)
            for value in direct_tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].tolist()
        )
        local = services[locale].detect(text, ignore_repeated_ngrams=True)
        expected = detector.detect(
            text=text,
            return_green_token_mask=True,
            return_z_at_T=False,
        )
        if local.tokenization.token_ids != raw_tokens or local.score is None:
            raise RuntimeError(f"raw-text tokenizer mismatch at case {case_index}")
        expected_mask_value = expected["green_token_mask"]
        if hasattr(expected_mask_value, "tolist"):
            expected_mask_value = expected_mask_value.tolist()
        expected_mask = tuple(bool(value) for value in expected_mask_value)
        score = local.score
        mismatch = (
            score.scored_token_count != int(expected["num_tokens_scored"])
            or score.green_count != int(expected["num_green_tokens"])
            or score.green_token_mask != expected_mask
            or abs(score.z_score - float(expected["z_score"])) > 1e-15
            or abs(score.p_value_one_sided - float(expected["p_value"])) > 1e-15
        )
        if mismatch:
            raise RuntimeError(f"raw-text score mismatch at case {case_index}")
        aggregate.update(
            json.dumps(
                {
                    "case_index": case_index,
                    "locale": locale,
                    "input_sha256": local.tokenization.input_sha256,
                    "encoding_sha256": local.tokenization.encoding_sha256,
                    "token_ids_sha256": ordered_token_ids_sha256(raw_tokens),
                    "green_count": score.green_count,
                    "scored_token_count": score.scored_token_count,
                    "z_score": score.z_score,
                    "p_value": score.p_value_one_sided,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return len(texts), aggregate.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    if args.cases < 1 or args.score_cases < 1:
        raise ValueError("cases and score-cases must be positive")
    root = _verified_author_checkout(args.author_repo)
    author = _load_author_module(root)
    torch = importlib.import_module("torch")
    rng = random.Random(args.seed)
    vocab_sizes = (16, 17, 31, 64, 127, 257, 1024)
    gammas = (0.10, 0.25, 0.50, 0.75)
    dummy_tokenizer = object()
    detectors: dict[tuple[int, float], Any] = {}
    greenlist_aggregate = hashlib.sha256()

    for case_index in range(args.cases):
        vocab_size = vocab_sizes[rng.randrange(len(vocab_sizes))]
        gamma = gammas[rng.randrange(len(gammas))]
        previous_token = rng.randrange(vocab_size)
        detector_key = (vocab_size, gamma)
        detector = detectors.get(detector_key)
        if detector is None:
            detector = author.WatermarkDetector(
                vocab=list(range(vocab_size)),
                gamma=gamma,
                seeding_scheme="simple_1",
                device=torch.device("cpu"),
                tokenizer=dummy_tokenizer,
                normalizers=[],
                ignore_repeated_ngrams=True,
            )
            detectors[detector_key] = detector
        expected = tuple(
            int(value)
            for value in detector._get_greenlist_ids(
                torch.tensor([previous_token], device="cpu")
            ).tolist()
        )
        greenlist_actual = kgw_author_green_list((previous_token,), vocab_size, gamma)
        if greenlist_actual != expected:
            print(
                json.dumps(
                    {
                        "status": "mismatch",
                        "case_index": case_index,
                        "previous_token": previous_token,
                        "vocab_size": vocab_size,
                        "gamma": gamma,
                        "expected_sha256": ordered_token_ids_sha256(expected),
                        "actual_sha256": ordered_token_ids_sha256(greenlist_actual),
                    },
                    sort_keys=True,
                )
            )
            return 1
        greenlist_aggregate.update(
            json.dumps(
                {
                    "case_index": case_index,
                    "previous_token": previous_token,
                    "vocab_size": vocab_size,
                    "gamma": gamma,
                    "ordered_sha256": ordered_token_ids_sha256(greenlist_actual),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    score_aggregate = hashlib.sha256()
    score_detectors: dict[tuple[int, float, bool], Any] = {}
    for case_index in range(args.score_cases):
        vocab_size = vocab_sizes[rng.randrange(len(vocab_sizes))]
        gamma = gammas[rng.randrange(len(gammas))]
        ignore_repeats = bool(rng.randrange(2))
        length = rng.randrange(2, 81)
        if case_index % 4 == 0:
            first = rng.randrange(vocab_size)
            second = rng.randrange(vocab_size)
            tokens = ([first, second] * ((length + 1) // 2))[:length]
        else:
            tokens = [rng.randrange(vocab_size) for _ in range(length)]
        score_detector_key = (vocab_size, gamma, ignore_repeats)
        detector = score_detectors.get(score_detector_key)
        if detector is None:
            detector = author.WatermarkDetector(
                vocab=list(range(vocab_size)),
                gamma=gamma,
                seeding_scheme="simple_1",
                device=torch.device("cpu"),
                tokenizer=dummy_tokenizer,
                normalizers=[],
                ignore_repeated_ngrams=ignore_repeats,
            )
            score_detectors[score_detector_key] = detector
        expected = detector._score_sequence(
            torch.tensor(tokens, device="cpu"),
            return_green_token_mask=True,
            return_z_at_T=False,
        )
        score_actual = kgw_author_detect(
            tokens,
            vocab_size,
            gamma,
            ignore_repeated_ngrams=ignore_repeats,
        )
        expected_mask_value = expected["green_token_mask"]
        if hasattr(expected_mask_value, "tolist"):
            expected_mask_value = expected_mask_value.tolist()
        expected_mask = tuple(bool(value) for value in expected_mask_value)
        mismatch = (
            score_actual.scored_token_count != int(expected["num_tokens_scored"])
            or score_actual.green_count != int(expected["num_green_tokens"])
            or score_actual.green_token_mask != expected_mask
            or abs(score_actual.z_score - float(expected["z_score"])) > 1e-15
            or abs(score_actual.p_value_one_sided - float(expected["p_value"])) > 1e-15
        )
        if mismatch:
            print(
                json.dumps(
                    {
                        "status": "score-mismatch",
                        "case_index": case_index,
                        "vocab_size": vocab_size,
                        "gamma": gamma,
                        "ignore_repeated_ngrams": ignore_repeats,
                        "tokens_sha256": ordered_token_ids_sha256(tokens),
                    },
                    sort_keys=True,
                )
            )
            return 1
        score_aggregate.update(
            json.dumps(
                {
                    "case_index": case_index,
                    "tokens_sha256": ordered_token_ids_sha256(tokens),
                    "vocab_size": vocab_size,
                    "gamma": gamma,
                    "ignore_repeated_ngrams": ignore_repeats,
                    "green_count": score_actual.green_count,
                    "scored_token_count": score_actual.scored_token_count,
                    "z_score": score_actual.z_score,
                    "p_value": score_actual.p_value_one_sided,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    profile = AUTHOR_KGW_CPU_PROFILE
    raw_text_cases = 0
    raw_text_stream_sha256 = None
    if args.tokenizer_assets is not None:
        raw_text_cases, raw_text_stream_sha256 = _raw_text_parity(
            author,
            torch,
            args.tokenizer_assets,
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "greenlist_cases": args.cases,
                "score_cases": args.score_cases,
                "mismatches": 0,
                "case_seed": args.seed,
                "greenlist_case_stream_sha256": greenlist_aggregate.hexdigest(),
                "score_case_stream_sha256": score_aggregate.hexdigest(),
                "raw_text_cases": raw_text_cases,
                "raw_text_case_stream_sha256": raw_text_stream_sha256,
                "profile_id": profile.profile_id,
                "author_commit": profile.author_commit,
                "author_source_sha256": profile.author_source_sha256,
                "torch_version": str(torch.__version__),
                "torch_git_version": str(torch.version.git_version),
                "rng_device": "cpu",
                "normalizers": [],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
