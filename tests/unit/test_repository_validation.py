from __future__ import annotations

import json
import runpy
from pathlib import Path

_VALIDATOR = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "validate_repository.py")
)
DRAFT_2020_12 = _VALIDATOR["DRAFT_2020_12"]
validate_repository = _VALIDATOR["validate_repository"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _repository(root: Path) -> None:
    workflow_schema = {
        "$schema": DRAFT_2020_12,
        "type": "object",
        "required": ["tasks"],
        "properties": {"tasks": {"type": "array"}},
    }
    run_schema = {
        "$schema": DRAFT_2020_12,
        "type": "object",
        "required": ["run_id"],
        "properties": {"run_id": {"type": "string"}},
    }
    _write_json(root / "schemas/workflow-manifest.schema.json", workflow_schema)
    _write_json(root / "schemas/run-manifest.schema.json", run_schema)
    _write_json(
        root / "project/workflow-manifest.json",
        {
            "tasks": [
                {
                    "task_document": "project/tasks/T01.md",
                    "steps": [{"rosettas": ["prompts/rosettas/worker.md"]}],
                }
            ]
        },
    )
    (root / "project/tasks").mkdir(parents=True)
    (root / "project/tasks/T01.md").write_text("task", encoding="utf-8")
    (root / "prompts/rosettas").mkdir(parents=True)
    (root / "prompts/rosettas/worker.md").write_text("prompt", encoding="utf-8")
    _write_json(
        root / "project/artifacts/run/outputs/run-manifest.json",
        {"run_id": "run-1"},
    )
    _write_json(
        root / "notebooks/example.ipynb",
        {"cells": [{"cell_type": "code", "source": ["value = 1\n"]}]},
    )


def test_valid_repository_reports_legacy_provenance(tmp_path: Path) -> None:
    _repository(tmp_path)

    result = validate_repository(tmp_path)

    assert result.ok
    assert len(result.legacy_warnings) == 1
    assert "run-manifest.json" in result.legacy_warnings[0]


def test_missing_workflow_references_are_errors(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "prompts/rosettas/worker.md").unlink()

    result = validate_repository(tmp_path, skip_notebooks=True)

    assert not result.ok
    assert any("missing Rosetta" in error for error in result.errors)


def test_notebook_syntax_errors_are_reported(tmp_path: Path) -> None:
    _repository(tmp_path)
    _write_json(
        tmp_path / "notebooks/example.ipynb",
        {"cells": [{"cell_type": "code", "source": ["if True print('bad')\n"]}]},
    )

    result = validate_repository(tmp_path, notebooks_only=True)

    assert not result.ok
    assert any("does not compile" in error for error in result.errors)


def test_schema_must_declare_draft_2020_12(tmp_path: Path) -> None:
    _repository(tmp_path)
    _write_json(
        tmp_path / "schemas/old.schema.json",
        {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"},
    )

    result = validate_repository(tmp_path, skip_notebooks=True)

    assert not result.ok
    assert any("old.schema.json" in error and "Draft 2020-12" in error for error in result.errors)


def test_invalid_literature_ledger_is_reported(tmp_path: Path) -> None:
    _repository(tmp_path)
    _write_json(
        tmp_path / "schemas/literature-evidence.schema.json",
        {
            "$schema": DRAFT_2020_12,
            "type": "object",
            "required": ["records"],
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["record_id", "pdf_url", "index_records"],
                    },
                }
            },
        },
    )
    _write_json(
        tmp_path / "research/frontier_evidence.json",
        {"records": [{"record_id": "PAPER_WITHOUT_EVIDENCE"}]},
    )

    result = validate_repository(tmp_path, skip_notebooks=True)

    assert not result.ok
    assert any("frontier_evidence.json" in error and "pdf_url" in error for error in result.errors)


def test_duplicate_literature_record_ids_are_reported(tmp_path: Path) -> None:
    _repository(tmp_path)
    _write_json(
        tmp_path / "schemas/literature-evidence.schema.json",
        {
            "$schema": DRAFT_2020_12,
            "type": "object",
            "required": ["records"],
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["record_id"],
                        "properties": {"record_id": {"type": "string"}},
                    },
                }
            },
        },
    )
    _write_json(
        tmp_path / "research/frontier_evidence.json",
        {"records": [{"record_id": "DUP"}, {"record_id": "DUP"}]},
    )

    result = validate_repository(tmp_path, skip_notebooks=True)

    assert not result.ok
    assert any("duplicate literature record IDs: DUP" in error for error in result.errors)
