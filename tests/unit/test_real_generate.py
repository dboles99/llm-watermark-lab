"""Real-model watermarked generation against a pinned local model.

The model used here is a tiny, randomly-initialised GPT-2 built from config in a
temporary directory. It downloads nothing and produces meaningless prose, which
is exactly right for the question these tests ask: does the generation path load
pinned weights offline, apply the watermark processor, record reproducible
provenance, and refuse when it cannot? Text *quality* is not under test and no
result here is scientific evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from watermark_lab.run_engine.real_generate import (
    AUTHOR_KGW_DEFAULTS,
    MODEL_WEIGHT_FILENAMES,
    PinnedWatermarkGenerator,
    RealGenerationError,
    describe_environment,
    find_model_weights,
    write_generation_record,
)

torch = pytest.importorskip("torch", reason="real generation requires torch")
transformers = pytest.importorskip("transformers", reason="real generation requires transformers")

PROMPT = "The provenance of a document"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny randomly-initialised GPT-2 saved locally. Nothing is downloaded."""

    target = tmp_path_factory.mktemp("tiny-gpt2")
    tokenizer = transformers.GPT2TokenizerFast(
        vocab_file=None,
        merges_file=None,
        tokenizer_object=_byte_level_tokenizer(),
    )
    config = transformers.GPT2Config(
        vocab_size=tokenizer.vocab_size, n_positions=128, n_embd=64, n_layer=2, n_head=2
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    return target


def _byte_level_tokenizer():
    """Build a minimal byte-level BPE tokenizer with no vocabulary files."""

    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {chr(i): i for i in range(256)}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return tokenizer


class TestEnvironmentReporting:
    def test_reports_readiness_and_author_defaults(self) -> None:
        payload = describe_environment()
        assert payload["ready"] is True
        assert payload["torch"]
        assert payload["transformers"]
        assert payload["author_kgw_defaults"] == AUTHOR_KGW_DEFAULTS

    def test_author_defaults_match_upstream(self) -> None:
        """If Hugging Face changes its defaults, parity silently breaks."""

        import inspect

        signature = inspect.signature(transformers.WatermarkingConfig.__init__)
        for key, expected in AUTHOR_KGW_DEFAULTS.items():
            assert signature.parameters[key].default == expected, (
                f"transformers default for {key} drifted from the KGW author value"
            )


class TestFailsClosed:
    def test_missing_directory_is_refused(self) -> None:
        with pytest.raises(RealGenerationError, match="does not exist"):
            PinnedWatermarkGenerator("/no/such/model/dir")

    def test_tokenizer_only_directory_is_refused(self, tmp_path: Path) -> None:
        """The pinned tokenizer assets alone cannot generate, and must say so."""

        (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
        with pytest.raises(RealGenerationError) as excinfo:
            PinnedWatermarkGenerator(tmp_path)
        message = str(excinfo.value)
        assert "No model weights found" in message
        assert "never downloaded automatically" in message

    def test_find_model_weights_reports_what_is_present(self, tmp_path: Path) -> None:
        assert find_model_weights(tmp_path) == ()
        (tmp_path / "model.safetensors").write_bytes(b"x")
        assert find_model_weights(tmp_path) == ("model.safetensors",)

    def test_weight_filenames_cover_both_formats(self) -> None:
        assert "model.safetensors" in MODEL_WEIGHT_FILENAMES
        assert "pytorch_model.bin" in MODEL_WEIGHT_FILENAMES


class TestGeneration:
    def test_generates_the_requested_number_of_tokens(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        result = generator.generate(PROMPT, max_new_tokens=40, seed=1)
        assert result.new_token_count == 40
        assert result.prompt == PROMPT
        assert result.text_sha256

    def test_is_deterministic_for_a_fixed_seed(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        first = generator.generate(PROMPT, max_new_tokens=30, seed=5)
        second = generator.generate(PROMPT, max_new_tokens=30, seed=5)
        assert first.token_ids == second.token_ids

    def test_different_seeds_diverge(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        first = generator.generate(PROMPT, max_new_tokens=30, seed=1)
        second = generator.generate(PROMPT, max_new_tokens=30, seed=2)
        assert first.token_ids != second.token_ids

    def test_watermark_changes_token_selection(self, tiny_model_dir: Path) -> None:
        """The matched negative control: same model, prompt and seed, no watermark.

        Without this control a positive detection cannot be distinguished from a
        property of the model or the prompt.
        """

        generator = PinnedWatermarkGenerator(tiny_model_dir)
        marked = generator.generate(PROMPT, max_new_tokens=50, seed=7, watermarked=True)
        control = generator.generate(PROMPT, max_new_tokens=50, seed=7, watermarked=False)
        assert marked.token_ids != control.token_ids

    def test_greedy_decoding_is_supported(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        result = generator.generate(PROMPT, max_new_tokens=20, do_sample=False, seed=0)
        assert result.new_token_count == 20

    @pytest.mark.parametrize("prompt", ["", "   "])
    def test_rejects_empty_prompt(self, tiny_model_dir: Path, prompt: str) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        with pytest.raises(RealGenerationError, match="non-empty"):
            generator.generate(prompt)

    def test_rejects_zero_tokens(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        with pytest.raises(RealGenerationError, match="at least 1"):
            generator.generate(PROMPT, max_new_tokens=0)


class TestProvenance:
    def test_records_hashes_and_versions(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        provenance = generator.generate(PROMPT, max_new_tokens=20, seed=3).provenance

        assert len(provenance.weights_sha256) == 64
        assert len(provenance.tokenizer_files_sha256) == 64
        assert provenance.torch_version
        assert provenance.transformers_version
        assert provenance.seed == 3
        assert provenance.watermarking_config == AUTHOR_KGW_DEFAULTS

    def test_uses_author_defaults_unless_overridden(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(
            tiny_model_dir, watermarking_config={"bias": 5.0}
        )
        config = generator.generate(PROMPT, max_new_tokens=20).provenance.watermarking_config
        assert config["bias"] == 5.0
        assert config["greenlist_ratio"] == AUTHOR_KGW_DEFAULTS["greenlist_ratio"]
        assert config["hashing_key"] == AUTHOR_KGW_DEFAULTS["hashing_key"]

    def test_weights_hash_is_stable(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        first = generator.generate(PROMPT, max_new_tokens=10, seed=1).provenance
        second = generator.generate(PROMPT, max_new_tokens=10, seed=2).provenance
        assert first.weights_sha256 == second.weights_sha256


class TestNotPromotable:
    def test_result_declares_mechanics_only_status(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        payload = generator.generate(PROMPT, max_new_tokens=20).as_dict()
        assert payload["scientific_status"] == "mechanics-only"
        assert "not a calibrated result" in payload["not_promotable"]

    def test_self_check_is_labelled_as_mechanics_not_an_operating_point(
        self, tiny_model_dir: Path
    ) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        result = generator.generate(PROMPT, max_new_tokens=60, seed=7, self_check=True)
        assert result.self_check is not None
        note = result.self_check.get("note", "")
        assert "not an operating point" in note.lower() or "error" in result.self_check

    def test_control_generation_skips_the_self_check(self, tiny_model_dir: Path) -> None:
        generator = PinnedWatermarkGenerator(tiny_model_dir)
        control = generator.generate(PROMPT, max_new_tokens=20, watermarked=False)
        assert control.self_check is None


class TestArtifactWriting:
    def test_writes_a_json_record(self, tiny_model_dir: Path, tmp_path: Path) -> None:
        import json

        generator = PinnedWatermarkGenerator(tiny_model_dir)
        result = generator.generate(PROMPT, max_new_tokens=20, seed=11)
        target = write_generation_record(result, tmp_path / "runs" / "gen.json")

        assert target.is_file()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["provenance"]["seed"] == 11
        assert payload["scientific_status"] == "mechanics-only"
