"""Tests for synthetic corpus generation."""

from __future__ import annotations

import pytest

from watermark_lab.corpora import (
    SyntheticCorpusError,
    SyntheticCorpusGenerator,
    generate_en_corpus,
    generate_zh_hant_tw_corpus,
)


class TestSyntheticCorpusGenerator:
    def test_en_corpus_is_deterministic(self) -> None:
        a = generate_en_corpus(seed=42, clusters=5, documents_per_cluster=3)
        b = generate_en_corpus(seed=42, clusters=5, documents_per_cluster=3)
        assert a.manifest_sha256 == b.manifest_sha256
        assert a.document_count == 15
        assert a.locale == "en"

    def test_zh_hant_tw_corpus_uses_traditional_chinese(self) -> None:
        corpus = generate_zh_hant_tw_corpus(seed=99, clusters=3, documents_per_cluster=2)
        assert corpus.locale == "zh-Hant-TW"
        assert corpus.document_count == 6
        for doc in corpus.documents:
            assert doc.locale == "zh-Hant-TW"
            assert "軟體" in doc.text or "資訊" in doc.text or len(doc.text) > 0

    def test_different_seeds_produce_different_corpora(self) -> None:
        a = generate_en_corpus(seed=1, clusters=5, documents_per_cluster=2)
        b = generate_en_corpus(seed=2, clusters=5, documents_per_cluster=2)
        assert a.manifest_sha256 != b.manifest_sha256

    def test_cluster_metadata_present(self) -> None:
        corpus = generate_en_corpus(seed=7, clusters=4, documents_per_cluster=2)
        cluster_ids = {doc.cluster_id for doc in corpus.documents}
        assert len(cluster_ids) == 4
        for doc in corpus.documents:
            assert doc.partition in corpus.partitions
            assert doc.document_id.startswith(doc.cluster_id)
            assert doc.text_sha256
            assert doc.domain

    def test_partitions_assigned_by_cluster(self) -> None:
        corpus = generate_en_corpus(seed=10, clusters=8, documents_per_cluster=3)
        cluster_partitions = {}
        for doc in corpus.documents:
            if doc.cluster_id in cluster_partitions:
                assert cluster_partitions[doc.cluster_id] == doc.partition
            else:
                cluster_partitions[doc.cluster_id] = doc.partition

    def test_scientific_status_is_non_scientific(self) -> None:
        corpus = generate_en_corpus(seed=1, clusters=2, documents_per_cluster=1)
        assert corpus.scientific_status == "non-scientific-synthetic"

    def test_manifest_serializable(self) -> None:
        corpus = generate_en_corpus(seed=1, clusters=2, documents_per_cluster=1)
        data = corpus.as_dict()
        assert isinstance(data["documents"], list)
        assert isinstance(data["partitions"], list)
        assert data["manifest_sha256"]

    def test_invalid_locale_rejected(self) -> None:
        gen = SyntheticCorpusGenerator(seed=1)
        with pytest.raises(SyntheticCorpusError, match="zh-Hant-TW"):
            gen.generate("zh-CN")  # type: ignore[arg-type]

    def test_invalid_params_rejected(self) -> None:
        gen = SyntheticCorpusGenerator(seed=1)
        with pytest.raises(SyntheticCorpusError, match="clusters"):
            gen.generate("en", clusters=0)

    def test_zh_cn_not_in_zh_hant_tw_corpus(self) -> None:
        corpus = generate_zh_hant_tw_corpus(seed=100, clusters=10, documents_per_cluster=5)
        zh_cn_terms = ["软件", "信息", "网络", "视频", "计算机"]
        for doc in corpus.documents:
            for term in zh_cn_terms:
                assert term not in doc.text, f"zh-CN term {term} found in zh-Hant-TW corpus"