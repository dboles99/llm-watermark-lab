"""Tests for the Taiwan public/government corpus adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from watermark_lab.corpora.taiwan_public import (
    TaiwanCorpusError,
    ingest_taiwan_public_corpus,
)

_SOURCE_MANIFEST = {
    "sources": [
        {
            "source_id": "gov-gazette-2025",
            "title": "Executive Yuan Gazette excerpts",
            "url": "https://data.gov.tw/dataset/example-gazette",
            "license_kind": "government-open-data",
            "license_reference": "https://data.gov.tw/license",
            "retrieved_at": "2026-08-15T00:00:00Z",
            "domain": "government-legal",
            "author_group": "executive-yuan",
        },
        {
            "source_id": "open-news-2025",
            "title": "Open-licensed news archive",
            "url": "https://example.org/news",
            "license_kind": "spdx-open",
            "license_reference": "CC-BY-4.0",
            "retrieved_at": "2026-08-15T00:00:00Z",
            "domain": "news",
        },
    ]
}


def _write_corpus(root: Path) -> None:
    (root / "sources" / "gov-gazette-2025").mkdir(parents=True)
    (root / "sources" / "open-news-2025").mkdir(parents=True)
    (root / "source-manifest.json").write_text(
        json.dumps(_SOURCE_MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "sources" / "gov-gazette-2025" / "doc-001.txt").write_text(
        "行政院今日公布新的資訊安全政策，網路防護列為優先項目。", encoding="utf-8"
    )
    (root / "sources" / "gov-gazette-2025" / "doc-002.txt").write_text(
        "該政策將由行政院資通安全處統籌，各部會須配合辦理。", encoding="utf-8"
    )
    (root / "sources" / "open-news-2025" / "doc-003.txt").write_text(
        "立法院昨日通過軟體採購條例修正案，明年一月施行。", encoding="utf-8"
    )


class TestTaiwanPublicCorpus:
    def test_successful_ingest(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        report = ingest_taiwan_public_corpus(tmp_path)
        assert report.locale == "zh-Hant-TW"
        assert report.scientific_status == "real-public-domain"
        assert report.native_review_status == "pending"
        assert len(report.documents) == 3
        assert len(report.rejected) == 0
        assert report.cluster_count == 2
        assert len(report.sources) == 2

    def test_cluster_metadata_preserved(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        report = ingest_taiwan_public_corpus(tmp_path)
        cluster_ids = {doc.cluster_id for doc in report.documents}
        assert len(cluster_ids) == 2
        for doc in report.documents:
            assert doc.document_id.startswith(doc.cluster_id)
            assert doc.text_sha256

    def test_partitions_never_split_cluster(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        report = ingest_taiwan_public_corpus(tmp_path)
        cluster_partitions: dict[str, str] = {}
        for doc in report.documents:
            if doc.cluster_id in cluster_partitions:
                assert cluster_partitions[doc.cluster_id] == doc.partition
            else:
                cluster_partitions[doc.cluster_id] = doc.partition
        assert set(report.partition_counts.values()) <= {0, 1, 2}

    def test_deterministic_partitions(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        first = ingest_taiwan_public_corpus(tmp_path)
        second = ingest_taiwan_public_corpus(tmp_path)
        assert first.manifest_sha256 == second.manifest_sha256
        assert {d.document_id: d.partition for d in first.documents} == {
            d.document_id: d.partition for d in second.documents
        }

    def test_zh_cn_contamination_rejected_and_recorded(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        (tmp_path / "sources" / "open-news-2025" / "bad.txt").write_text(
            "这份软件使用网络传输视频数据。", encoding="utf-8"
        )
        report = ingest_taiwan_public_corpus(tmp_path)
        assert len(report.documents) == 3
        assert len(report.rejected) == 1
        assert "zh-CN" in report.rejected[0].reason
        assert any("rejected" in c for c in report.caveats)

    def test_additional_markers_respected(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        manifest = dict(_SOURCE_MANIFEST)
        manifest["additional_zh_cn_markers"] = ["国务院"]
        (tmp_path / "source-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        (tmp_path / "sources" / "open-news-2025" / "marker.txt").write_text(
            "文件提到国务院的相關報導。", encoding="utf-8"
        )
        report = ingest_taiwan_public_corpus(tmp_path)
        assert len(report.rejected) == 1
        assert "国务院" in report.rejected[0].reason

    def test_missing_source_manifest_fails_closed(self, tmp_path: Path) -> None:
        (tmp_path / "sources").mkdir(parents=True)
        with pytest.raises(TaiwanCorpusError, match="source manifest not found"):
            ingest_taiwan_public_corpus(tmp_path)

    def test_undeclared_source_directory_rejected(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        undeclared = tmp_path / "sources" / "undeclared-source"
        undeclared.mkdir()
        (undeclared / "doc.txt").write_text("未宣告來源的文件。", encoding="utf-8")
        with pytest.raises(TaiwanCorpusError, match="no declaration"):
            ingest_taiwan_public_corpus(tmp_path)

    def test_closed_license_rejected(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        manifest = json.loads((tmp_path / "source-manifest.json").read_text(encoding="utf-8"))
        manifest["sources"][0]["license_kind"] = "all-rights-reserved"
        (tmp_path / "source-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(TaiwanCorpusError, match="accepted open license"):
            ingest_taiwan_public_corpus(tmp_path)

    def test_empty_corpus_rejected(self, tmp_path: Path) -> None:
        (root := tmp_path).mkdir(exist_ok=True)
        (root / "sources" / "gov-gazette-2025").mkdir(parents=True)
        (root / "source-manifest.json").write_text(
            json.dumps(_SOURCE_MANIFEST, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(TaiwanCorpusError, match="zero documents"):
            ingest_taiwan_public_corpus(root)

    def test_zh_cn_locale_rejected(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        with pytest.raises(TaiwanCorpusError, match="zh-Hant-TW"):
            ingest_taiwan_public_corpus(tmp_path, locale="zh-CN")

    def test_zh_tw_alias_accepted(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        report = ingest_taiwan_public_corpus(tmp_path, locale="zh-TW")
        assert report.locale == "zh-Hant-TW"

    def test_report_serializable(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        data = ingest_taiwan_public_corpus(tmp_path).as_dict()
        assert data["scientific_status"] == "real-public-domain"
        assert isinstance(data["documents"], list)
        assert isinstance(data["rejected"], list)
        json.dumps(data, ensure_ascii=False)