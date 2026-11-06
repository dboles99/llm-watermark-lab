"""Tests for blind/semi-blind watermark detection methods."""

from __future__ import annotations

import random

import pytest

from watermark_lab.blind_detection import BlindDetectionOrchestrator
from watermark_lab.tokenization.synthetic import synthetic_numeric_tokenizer


@pytest.fixture
def synth_tokenizer():
    return synthetic_numeric_tokenizer(vocab_size=200)


def _watermarked_seq(
    length: int, vocab_size: int, gamma: float, hash_key: int, *, token_choice_seed: int = 20260817
) -> list[int]:
    """Build a deterministic KGW-like sequence using the Rust portable RNG.

    The watermark sampling is done with the Rust splitmix64 green list; a
    separate Python RNG only chooses which green token to emit at each step so
    the sequence has realistic token diversity while remaining watermarked.
    """
    import watermark_core as _core

    seq = [0]
    rng = random.Random(token_choice_seed)
    for _ in range(1, length):
        prev = seq[-1]
        seed = _core.simple_1_seed_py(prev, hash_key)
        green = _core.py_green_list_splitmix(vocab_size, gamma, seed)
        # Choose a deterministic green token to avoid pathological repetition
        seq.append(rng.choice(green))
    return seq


def test_parameter_sweep_detects_synthetic_watermark(synth_tokenizer):
    vocab_size = 200
    gamma = 0.25
    hash_key = 15485863
    token_ids = _watermarked_seq(300, vocab_size, gamma, hash_key)
    text = " ".join(str(t) for t in token_ids)
    det = BlindDetectionOrchestrator(synth_tokenizer)
    result = det._sweep.detect(
        text,
        gammas=[0.1, 0.25, 0.5],
        context_widths=[1],
        hash_keys=[15485863, 999999],
    )
    assert result.status == "completed"
    assert result.score is not None
    assert result.score > 10.0
    assert result.details["best_setting"]["hash_key"] == hash_key
    assert abs(result.details["best_setting"]["gamma"] - gamma) < 0.01


def test_variance_ratio_detects_shared_key(synth_tokenizer):
    vocab_size = 200
    gamma = 0.25
    hash_key = 15485863
    docs = []
    for _ in range(5):
        token_ids = _watermarked_seq(250, vocab_size, gamma, hash_key)
        docs.append(" ".join(str(t) for t in token_ids))
    det = BlindDetectionOrchestrator(synth_tokenizer)
    result = det._variance.detect(
        docs,
        gamma=gamma,
        candidate_keys=[hash_key, 111111],
        random_key_trials=10,
    )
    assert result.status == "completed"
    assert result.details["best_key"] == hash_key
    assert result.details["best_mean_z"] > 3.0


def test_length_stratified_pools_z_scores(synth_tokenizer):
    vocab_size = 200
    gamma = 0.25
    hash_key = 15485863
    docs = []
    for length in [80, 150, 400]:
        token_ids = _watermarked_seq(length, vocab_size, gamma, hash_key)
        docs.append(" ".join(str(t) for t in token_ids))
    det = BlindDetectionOrchestrator(synth_tokenizer)
    result = det._length_meta.analyze(docs, gamma=gamma, hash_key=hash_key)
    assert result.status == "completed"
    assert result.score is not None
    assert result.score > 3.0


def test_orchestrator_blocks_without_control(synth_tokenizer):
    text = " ".join(str(t) for t in _watermarked_seq(100, 200, 0.25, 15485863))
    det = BlindDetectionOrchestrator(synth_tokenizer)
    results = det.run_all([text])
    assert results["parameter_sweep"].status == "completed"
    assert results["perturbation_invariance"].status == "blocked"
    assert results["reference_corpus_two_sample"].status == "blocked"
    assert results["entropy_conditioned"].status == "blocked"
    assert results["embedding_cluster_anomaly"].status == "blocked"


def test_two_sample_detects_distributional_difference(synth_tokenizer):
    # Target: multiple watermarked docs; Control: random uniform tokens.
    # A single target doc has no per-token variance, so we need several
    # watermarked samples to expose token-frequency anomalies. A smaller
    # vocabulary concentrates token mass and improves per-token test power.
    vocab_size = 50
    target = [
        " ".join(str(t) for t in _watermarked_seq(200, vocab_size, 0.25, 15485863, token_choice_seed=seed))
        for seed in range(30)
    ]
    control = [
        " ".join(str(random.randint(0, vocab_size - 1)) for _ in range(200))
        for _ in range(30)
    ]
    det = BlindDetectionOrchestrator(synth_tokenizer)
    result = det._two_sample.compare(target, control)
    assert result.status == "completed"
    # Should detect at least some tokens differ strongly
    assert result.details["n_significant"] > 0
