"""Integration tests for the Taiwan public corpus CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from watermark_lab.cli import app


def _write_corpus(root: Path) -> None:
    (root / "sources" / "gov-test").mkdir(parents=True)
    (root / "source-manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "gov-test",
                        "title": "Test gazette",
                        "url": "https://data.gov.tw/dataset/test",
                        "license_kind": "government-open-data",
                        "license_reference": "https://data.gov.tw/license",
                        "retrieved_at": "2026-08-15T00:00:00Z",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "sources" / "gov-test" / "a.txt").write_text(
        "行政院公布資訊安全政策。", encoding="utf-8"
    )


def test_ingest_taiwan_public_corpus_command(tmp_path: Path) -> None:
    runner = CliRunner()
    _write_corpus(tmp_path)
    result = runner.invoke(app, ["ingest-taiwan-public-corpus", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    json_start = result.output.find("{")
    payload = json.loads(result.output[json_start:])
    assert payload["scientific_status"] == "real-public-domain"
    assert payload["native_review_status"] == "pending"
    assert payload["document_count"] == 1
    assert payload["cluster_count"] == 1
    assert "documents" not in payload


def test_ingest_taiwan_public_corpus_missing_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    (tmp_path / "sources").mkdir()
    result = runner.invoke(app, ["ingest-taiwan-public-corpus", "--root", str(tmp_path)])
    assert result.exit_code == 2
    json_start = result.output.find("{")
    payload = json.loads(result.output[json_start:])
    assert payload["status"] == "failed"
    assert "source manifest" in payload["error"]