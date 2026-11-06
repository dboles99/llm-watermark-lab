"""Offline, cross-platform validation for repository evidence and notebooks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


@dataclass
class ValidationResult:
    """Accumulate validation findings without stopping at the first defect."""

    checked: int = 0
    errors: list[str] = field(default_factory=list)
    legacy_warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path, root: Path, result: ValidationResult) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result.errors.append(f"{_display(path, root)}: invalid JSON: {exc}")
        return None


def _schema_errors(instance: Any, schema: Any) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.path)):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{location}: {error.message}")
    return errors


def validate_schemas(root: Path, result: ValidationResult) -> dict[str, Any]:
    """Validate every repository schema as Draft 2020-12."""

    schemas: dict[str, Any] = {}
    schema_dir = root / "schemas"
    for path in sorted(schema_dir.rglob("*.json")):
        schema = _load_json(path, root, result)
        result.checked += 1
        if schema is None:
            continue
        if not isinstance(schema, dict) or schema.get("$schema") != DRAFT_2020_12:
            result.errors.append(
                f"{_display(path, root)}: $schema must declare Draft 2020-12"
            )
            continue
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            result.errors.append(f"{_display(path, root)}: invalid Draft 2020-12 schema: {exc.message}")
            continue
        schemas[path.name] = schema
    if not schemas:
        result.errors.append("schemas: no valid JSON schemas found")
    return schemas


def _repository_reference(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts:
        return None
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def validate_workflow(root: Path, schemas: dict[str, Any], result: ValidationResult) -> None:
    """Validate the workflow manifest and every referenced task/Rosetta file."""

    path = root / "project" / "workflow-manifest.json"
    if not path.exists():
        return
    manifest = _load_json(path, root, result)
    result.checked += 1
    schema = schemas.get("workflow-manifest.schema.json")
    if manifest is None or schema is None:
        if schema is None:
            result.errors.append("schemas/workflow-manifest.schema.json: valid schema required")
        return
    for detail in _schema_errors(manifest, schema):
        result.errors.append(f"{_display(path, root)}: {detail}")
    if not isinstance(manifest, dict):
        return
    for task in manifest.get("tasks", []):
        if not isinstance(task, dict):
            continue
        references = [("task document", task.get("task_document"))]
        for step in task.get("steps", []):
            if isinstance(step, dict):
                references.extend(("Rosetta", value) for value in step.get("rosettas", []))
        for kind, value in references:
            target = _repository_reference(root, value)
            if target is None:
                result.errors.append(f"{_display(path, root)}: unsafe or invalid {kind} path: {value!r}")
            elif not target.is_file():
                result.errors.append(f"{_display(path, root)}: missing {kind}: {value}")


def validate_literature(root: Path, schemas: dict[str, Any], result: ValidationResult) -> None:
    """Validate the strict scholarly evidence ledger and unique record IDs."""

    path = root / "research" / "frontier_evidence.json"
    schema = schemas.get("literature-evidence.schema.json")
    if not path.exists() and schema is None:
        return
    if schema is None:
        result.errors.append("schemas/literature-evidence.schema.json: valid schema required")
        return
    if not path.is_file():
        result.errors.append("research/frontier_evidence.json: strict evidence ledger required")
        return
    ledger = _load_json(path, root, result)
    result.checked += 1
    if ledger is None:
        return
    for detail in _schema_errors(ledger, schema):
        result.errors.append(f"{_display(path, root)}: {detail}")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("records"), list):
        return
    record_ids: list[str] = []
    for record in ledger["records"]:
        if not isinstance(record, dict):
            continue
        record_id = record.get("record_id")
        if isinstance(record_id, str):
            record_ids.append(record_id)
    duplicates = sorted({record_id for record_id in record_ids if record_ids.count(record_id) > 1})
    if duplicates:
        result.errors.append(
            f"{_display(path, root)}: duplicate literature record IDs: {', '.join(duplicates)}"
        )
    for identifier in ("doi", "arxiv"):
        values: list[str] = []
        for record in ledger["records"]:
            if not isinstance(record, dict):
                continue
            stable = record.get("stable_identifier")
            value = stable.get(identifier) if isinstance(stable, dict) else None
            if isinstance(value, str):
                values.append(value)
        duplicate_values = sorted({value for value in values if values.count(value) > 1})
        if duplicate_values:
            result.errors.append(
                f"{_display(path, root)}: duplicate literature {identifier} values: "
                f"{', '.join(duplicate_values)}"
            )


def _has_source_provenance(manifest: dict[str, Any]) -> bool:
    source = manifest.get("source")
    return (
        isinstance(source, dict)
        and bool(source.get("git_commit"))
        and isinstance(source.get("working_tree_dirty"), bool)
        and isinstance(source.get("environment"), dict)
    )


def validate_evidence(root: Path, schemas: dict[str, Any], result: ValidationResult) -> None:
    """Validate run manifests and report incomplete historical provenance."""

    evidence_root = root / "project" / "artifacts"
    run_schema = schemas.get("run-manifest.schema.json")
    run_paths = sorted(evidence_root.rglob("run-manifest.json")) if evidence_root.is_dir() else []
    if run_schema is None:
        result.errors.append("schemas/run-manifest.schema.json: valid schema required")
    else:
        for path in run_paths:
            manifest = _load_json(path, root, result)
            result.checked += 1
            if manifest is None:
                continue
            for detail in _schema_errors(manifest, run_schema):
                result.errors.append(f"{_display(path, root)}: {detail}")

    candidates = sorted(
        path
        for path in evidence_root.rglob("*.json")
        if path.name == "manifest.json" or path.name.endswith("-manifest.json")
    ) if evidence_root.is_dir() else []
    for path in candidates:
        manifest = _load_json(path, root, result)
        if not isinstance(manifest, dict) or _has_source_provenance(manifest):
            continue
        result.legacy_warnings.append(
            f"{_display(path, root)}: canonical evidence manifest lacks complete source "
            "provenance (git commit, dirty-tree state, and environment)"
        )


def notebook_paths(root: Path) -> list[Path]:
    paths = list(root.glob("*.ipynb"))
    notebook_dir = root / "notebooks"
    if notebook_dir.is_dir():
        paths.extend(notebook_dir.rglob("*.ipynb"))
    return sorted(set(paths))


def validate_notebooks(root: Path, result: ValidationResult) -> None:
    """Compile every code cell without executing it or importing notebook dependencies."""

    for path in notebook_paths(root):
        notebook = _load_json(path, root, result)
        result.checked += 1
        if not isinstance(notebook, dict):
            continue
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            result.errors.append(f"{_display(path, root)}: notebook cells must be an array")
            continue
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list) and all(isinstance(line, str) for line in source):
                code = "".join(source)
            elif isinstance(source, str):
                code = source
            else:
                result.errors.append(
                    f"{_display(path, root)}: code cell {index + 1} has invalid source"
                )
                continue
            try:
                compile(code, f"{_display(path, root)}:cell-{index + 1}", "exec")
            except (SyntaxError, ValueError, OverflowError) as exc:
                result.errors.append(
                    f"{_display(path, root)}: code cell {index + 1} does not compile: {exc}"
                )


def validate_repository(
    root: Path, *, skip_notebooks: bool = False, notebooks_only: bool = False
) -> ValidationResult:
    root = root.resolve()
    result = ValidationResult()
    if notebooks_only:
        validate_notebooks(root, result)
        return result
    schemas = validate_schemas(root, result)
    validate_workflow(root, schemas, result)
    validate_literature(root, schemas, result)
    validate_evidence(root, schemas, result)
    if not skip_notebooks:
        validate_notebooks(root, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-notebooks", action="store_true")
    mode.add_argument("--notebooks-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_repository(
        args.root, skip_notebooks=args.skip_notebooks, notebooks_only=args.notebooks_only
    )
    for warning in result.legacy_warnings:
        print(f"LEGACY WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(
        f"Repository validation: {result.checked} files checked, "
        f"{len(result.errors)} errors, {len(result.legacy_warnings)} legacy warnings"
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
