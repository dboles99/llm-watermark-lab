from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from watermark_lab.core.contracts import DetectionConfig
from watermark_lab.run_engine.create import CreationService
from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.run_engine.intake import compute_sha256, normalize_text, read_source
from watermark_lab.security import SimulatedTeeBackend, SimulatedTeeVerifier
from watermark_lab.services.product_workspace import ProductWorkspaceService


def _configure_utf8_output() -> None:
    """Force UTF-8 on stdout and stderr.

    The default Windows console encoding is cp1252, which cannot represent CJK.
    For a project that treats Taiwan Traditional Chinese as a first-class
    condition that is not a cosmetic problem: any command emitting zh-Hant-TW
    text either crashes with a UnicodeEncodeError or silently escapes every
    character to ``\\uXXXX``, which makes locale output unreadable and makes
    Simplified-versus-Traditional contamination impossible to eyeball.

    ``errors="replace"`` keeps a terminal that genuinely cannot render a glyph
    from aborting the command.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - stream-dependent
            pass


_configure_utf8_output()

app = typer.Typer(help="LLM Watermark Lab research CLI")


def _run_scientific_detect(
    *,
    text: str,
    locale: str,
    tokenizer_id: str,
    tokenizer_assets: str,
    gamma: float,
    threshold: float,
    ignore_repeated_ngrams: bool,
    input_media_type: str,
    input_source_path: str | None,
) -> dict[str, Any]:
    """Run the fail-closed scientific detector with pinned tokenizer and author RNG."""

    from watermark_lab.i18n import canonicalize_locale
    from watermark_lab.run_engine.scientific_detect import (
        ScientificKgwDetectionService,
    )
    from watermark_lab.tokenization import load_local_huggingface_tokenizer
    from watermark_lab.tokenization.contracts import (
        NormalizationPolicy,
        SpecialTokenPolicy,
        TokenizerSpec,
        UnicodeNormalization,
    )

    canonical = canonicalize_locale(locale)
    if canonical not in ("en", "zh-Hant-TW"):
        raise ValueError(f"Scientific mode requires locale 'en' or 'zh-Hant-TW'; got {canonical!r}")

    tokenizer_config_path = Path("config/tokenizers") / "openai-community-gpt2-607a30d7.json"
    if tokenizer_config_path.is_file():
        tok_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"Tokenizer configuration not found: {tokenizer_config_path}. "
            "Scientific mode requires a pinned tokenizer profile."
        )

    revision = tok_config.get("revision", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
    asset_dir = tokenizer_assets or tok_config.get("local_asset_directory", "")
    if not asset_dir:
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            asset_dir = os.path.join(
                local_app_data,
                "watermark-lab-lab",
                "assets",
                "openai-community-gpt2",
                revision,
            )
    if not asset_dir or not Path(asset_dir).is_dir():
        raise ValueError(
            f"Pinned tokenizer assets not found at {asset_dir!r}. "
            "Use --tokenizer-assets to specify the local directory."
        )

    norm_form = UnicodeNormalization.NONE if tok_config.get("normalization") == "none" else UnicodeNormalization.NFC
    spec = TokenizerSpec(
        tokenizer_id=tokenizer_id,
        revision=revision,
        normalization=NormalizationPolicy(unicode_form=norm_form, locale=canonical),
        special_tokens=SpecialTokenPolicy(),
        expected_backend_versions=tuple(
            (item["name"], item["version"])
            for item in tok_config.get("backend_versions", [])
        ),
        expected_asset_aggregate_sha256=tok_config.get("asset_aggregate_sha256"),
        expected_tokenizer_fingerprint_sha256=tok_config.get("tokenizer_fingerprint_sha256"),
    )
    tokenizer = load_local_huggingface_tokenizer(spec, asset_dir)
    service = ScientificKgwDetectionService(tokenizer)
    result = service.detect(
        text,
        gamma=gamma,
        threshold=threshold,
        ignore_repeated_ngrams=ignore_repeated_ngrams,
    )
    return {
        "mode": "scientific",
        "method_id": result.method_id,
        "scientific_status": result.scientific_status,
        "decision": result.decision,
        "author_profile_id": result.author_profile_id,
        "calibration_id": result.calibration_id,
        "estimated_fpr": result.estimated_fpr,
        "candidate_threshold_prediction": result.candidate_threshold_prediction,
        "token_count": result.tokenization.token_count,
        "tokenizer_fingerprint": result.tokenization.provenance.tokenizer_fingerprint_sha256,
        "input_sha256": result.tokenization.input_sha256,
        "normalized_sha256": result.tokenization.normalized_text_sha256,
        "encoding_sha256": result.tokenization.encoding_sha256,
        "warnings": list(result.warnings),
        "input_media_type": input_media_type,
        "input_source_path": input_source_path,
        "score": result.score.z_score if result.score is not None else None,
        "p_value": result.score.p_value_one_sided if result.score is not None else None,
        "green_count": result.score.green_count if result.score is not None else None,
        "scored_token_count": result.score.scored_token_count if result.score is not None else None,
    }


def get_product_workspace_service(
    *, locale: str = "en", history_mode: str = "supersede-and-preserve"
) -> ProductWorkspaceService:
    """Build the product service; tests and applications may replace this seam."""

    return ProductWorkspaceService(locale=locale, history_mode=history_mode)


@app.command()
def capabilities() -> None:
    """Show planned method adapters and their current status."""
    typer.echo("kgw: experimental")
    typer.echo("mcmark-inspired: experimental specification")
    typer.echo("mirrormark-inspired: experimental specification")
    typer.echo("anthropic/claude-style: unsupported without an official compatible detector")


@app.command(name="runs")
def list_runs(
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum runs to return"),
    locale: str = typer.Option("en", "--locale", help="en, zh-Hant-TW, or zh-TW"),
) -> None:
    """List persisted research runs through the shared product service."""

    try:
        service = get_product_workspace_service(locale=locale)
        payload = {
            "runs": [item.as_dict() for item in service.list_runs(limit=limit)],
            "locale": service.locale,
        }
    except ValueError as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command(name="artifacts")
def list_artifacts(
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum artifacts to return"),
    locale: str = typer.Option("en", "--locale", help="en, zh-Hant-TW, or zh-TW"),
    history_mode: str = typer.Option(
        "supersede-and-preserve",
        "--history-mode",
        help="supersede-and-preserve or amend-in-place",
    ),
) -> None:
    """List evidence artifacts and the selected history policy."""

    try:
        service = get_product_workspace_service(locale=locale, history_mode=history_mode)
        payload = {
            "artifacts": [item.as_dict() for item in service.list_artifacts(limit=limit)],
            "locale": service.locale,
            "history_mode": service.history_mode.value,
        }
    except ValueError as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command(name="workspace-policy")
def workspace_policy(
    locale: str = typer.Option("en", "--locale", help="en, zh-Hant-TW, or zh-TW"),
    history_mode: str = typer.Option(
        "supersede-and-preserve",
        "--history-mode",
        help="supersede-and-preserve or amend-in-place",
    ),
) -> None:
    """Show artifact-history choices and the strict locale policy."""

    try:
        service = get_product_workspace_service(locale=locale, history_mode=history_mode)
        payload = service.policy()
    except ValueError as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command(name="research-features")
def research_features(
    locale: str = typer.Option("en", "--locale", help="en, zh-Hant-TW, or zh-TW"),
    select: Annotated[
        list[str] | None,
        typer.Option("--select", help="Repeat to record one or more feature selections."),
    ] = None,
    execution_target: str = typer.Option(
        "api-first-hybrid",
        "--execution-target",
        help="api-first-hybrid, local-only, or api-only",
    ),
    payload_identity: str = typer.Option(
        "pseudonymous-project-workflow",
        "--payload-identity",
        help="pseudonymous-project-workflow or person-account-reference",
    ),
    proof_backend: str = typer.Option(
        "tee-simulator",
        "--proof-backend",
        help="tee-simulator, remote-tee, or zero-knowledge",
    ),
) -> None:
    """List the selectable frontier features and their governed execution choices."""

    try:
        service = get_product_workspace_service(locale=locale)
        payload = {
            "features": [item.as_dict() for item in service.list_research_features()],
            "research_policy": service.research_policy(),
            "locale": service.locale,
        }
        if select:
            payload["selection"] = service.research_selection(
                select,
                execution_target=execution_target,
                payload_identity_mode=payload_identity,
                proof_backend=proof_backend,
            )
    except ValueError as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command(name="tee-simulator-self-test")
def tee_simulator_self_test() -> None:
    """Exercise the local statement/freshness/tamper boundary with ephemeral test material."""

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    ephemeral_key = secrets.token_bytes(32)
    measurement = digest("watermark-lab-lab-tee-simulator-v1")
    backend = SimulatedTeeBackend(
        ephemeral_key,
        enclave_measurement_sha256=measurement,
    )
    verifier = SimulatedTeeVerifier(
        ephemeral_key,
        expected_measurement_sha256=measurement,
        allow_simulated=True,
    )
    bundle = backend.attest(
        nonce=verifier.issue_nonce(),
        normalized_text_sha256=digest("development-self-test-text-commitment"),
        detector_id="self-test-detector",
        detector_code_sha256=digest("development-self-test-code"),
        detector_config_sha256=digest("development-self-test-config"),
        calibration_artifact_sha256=digest("development-self-test-calibration"),
        key_epoch_reference="epoch_self_test_0001",
        decision="inconclusive",
        calibrated_fpr_upper_bound=None,
    )
    outcome = verifier.verify(bundle)
    typer.echo(
        json.dumps(
            {
                "status": "passed",
                "backend": outcome.backend,
                "statement_id": outcome.statement_id,
                "evidence_sha256": outcome.evidence_sha256,
                "promotion_eligible": outcome.promotion_eligible,
                "warnings": list(outcome.warnings),
            },
            indent=2,
        )
    )


@app.command(name="detect")
def detect(
    input: str = typer.Option(..., "--input", help="File path or raw text to analyze"),
    method: str = typer.Option("kgw", "--method", help="Detector method ID"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="Key identifier"),
    threshold: float = typer.Option(4.0, "--threshold", help="Detection threshold"),
    tokenizer_id: str = typer.Option("gpt2", "--tokenizer-id", help="Tokenizer family"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction"),
    context_width: int = typer.Option(1, "--context-width", help="KGW context width"),
    ignore_repeated_ngrams: bool = typer.Option(
        False,
        "--ignore-repeated-ngrams",
        help="Score each unique context/token n-gram once; remains inconclusive until calibrated",
    ),
    scientific: bool = typer.Option(
        False,
        "--scientific",
        help="Use the pinned tokenizer + author-code RNG scientific detector (returns inconclusive without frozen calibration)",
    ),
    tokenizer_assets: str = typer.Option(
        "",
        "--tokenizer-assets",
        help="Local directory of pinned tokenizer assets (required for --scientific with a real tokenizer)",
    ),
    locale: str = typer.Option("en", "--locale", help="en or zh-Hant-TW (scientific mode)"),
) -> None:
    """Run a detector on a sample and print a JSON DetectionResult."""
    text, media_type, source_path = read_source(input)
    normalized = normalize_text(text)

    if scientific:
        try:
            sci_payload = _run_scientific_detect(
                text=normalized,
                locale=locale,
                tokenizer_id=tokenizer_id,
                tokenizer_assets=tokenizer_assets,
                gamma=gamma,
                threshold=threshold,
                ignore_repeated_ngrams=ignore_repeated_ngrams,
                input_media_type=media_type,
                input_source_path=source_path,
            )
        except Exception as exc:
            typer.echo(json.dumps({"error": str(exc), "mode": "scientific", "status": "failed"}, indent=2))
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(sci_payload, indent=2))
        return
    else:
        config = DetectionConfig(
            method_id=method,
            tokenizer_id=tokenizer_id,
            key_id=key_id,
            threshold=threshold,
            options={
                "gamma": gamma,
                "context_width": context_width,
                "ignore_repeated_ngrams": ignore_repeated_ngrams,
                "input_media_type": media_type,
                "input_source_path": source_path,
                "input_sha256": compute_sha256(text),
                "normalized_sha256": compute_sha256(normalized),
            },
        )

        result = DetectionService().detect(normalized, config)
        payload: dict[str, Any] = {
            "method_id": result.method_id,
            "compatibility": result.compatibility,
            "decision": result.decision,
            "score": result.score,
            "p_value": result.p_value,
            "token_count": result.token_count,
            "estimated_fpr": result.estimated_fpr,
            "original_scorable_token_count": result.original_scorable_token_count,
            "ignored_repeated_ngrams": result.ignored_repeated_ngrams,
            "scoring_variant": result.scoring_variant,
            "calibration_id": result.calibration_id,
            "threshold": config.threshold,
            "warnings": list(result.warnings),
            "input_media_type": media_type,
            "input_source_path": source_path,
            "input_sha256": config.options["input_sha256"],
            "normalized_sha256": config.options["normalized_sha256"],
        }
    typer.echo(json.dumps(payload, indent=2))


@app.command(name="create")
def create(
    prompt: str = typer.Option(..., "--prompt", help="Prompt text for generation"),
    profile: str = typer.Option("mock", "--profile", help="Model profile ID"),
    method: str = typer.Option("kgw", "--method", help="Watermark method ID"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="Key identifier"),
    disclosure: bool = typer.Option(False, "--disclosure", help="Record disclosure flag"),
    max_new_tokens: int = typer.Option(80, "--max-new-tokens", help="Number of tokens to generate"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction"),
    delta: float = typer.Option(2.0, "--delta", help="KGW logit bias"),
    context_width: int = typer.Option(1, "--context-width", help="KGW context width"),
) -> None:
    """Generate watermarked text and print a JSON creation result."""
    config = {
        "disclosure_recorded": disclosure,
        "max_new_tokens": max_new_tokens,
        "gamma": gamma,
        "delta": delta,
        "context_width": context_width,
    }

    service = CreationService()
    try:
        result = service.create(prompt, profile, method, key_id, config)
    except Exception as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=1)

    payload: dict[str, Any] = {
        "run_id": result["run_id"],
        "status": result["status"],
        "text": result["generation"]["text"],
        "tokens": result["generation"]["tokens"],
        "prompt_sha256": result["input_sha256"],
        "response_sha256": result["generation"]["response_sha256"],
        "model_profile": result["generation"]["model_profile"],
        "method_id": result["method_id"],
        "key_id": result["key_id"],
        "self_verification": result["self_verification"],
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command(name="attack")
def attack(
    text: str = typer.Option(..., "--text", help="Parent watermarked text to attack"),
    attack_id: str = typer.Option("deletion", "--attack-id", help="Attack type: deletion, insertion, substitution, shuffle, paraphrase"),
    severity: float = typer.Option(0.3, "--severity", min=0.0, max=1.0, help="Attack severity"),
    seed: str = typer.Option("attack-cli", "--seed", help="Random seed for the attack"),
    method: str = typer.Option("kgw", "--method", help="Detector method ID"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="Key identifier"),
    threshold: float = typer.Option(4.0, "--threshold", help="Detection threshold"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction"),
    context_width: int = typer.Option(1, "--context-width", help="KGW context width"),
) -> None:
    """Run a single attack on watermarked text and compare pre/post detection."""

    from watermark_lab.attacks.transformations import AttackConfig
    from watermark_lab.run_engine.attack import RobustnessService

    parent_run: dict[str, Any] = {
        "run_id": "cli-attack-parent",
        "run_type": "create",
        "status": "completed",
        "method_id": method,
        "key_id": key_id,
        "generation": {
            "text": text,
            "method_id": method,
            "key_id": key_id,
            "model_profile": {"simulated": True},
            "config": {
                "gamma": gamma,
                "context_width": context_width,
                "detection_threshold": threshold,
                "seeding_scheme": "simple_1",
                "hash_key": 15485863,
            },
        },
        "governance": {"project_owned": True, "source_is_third_party": False},
        "config": {
            "gamma": gamma,
            "context_width": context_width,
            "detection_threshold": threshold,
            "seeding_scheme": "simple_1",
            "hash_key": 15485863,
        },
    }
    detector_config = DetectionConfig(
        method_id=method,
        tokenizer_id="gpt2",
        key_id=key_id,
        threshold=threshold,
        options={
            "gamma": gamma,
            "context_width": context_width,
            "seeding_scheme": "simple_1",
            "hash_key": 15485863,
        },
    )
    attack_config = AttackConfig(attack_id=attack_id, severity=severity, seed=seed)
    try:
        service = RobustnessService()
        result = service.attack(parent_run, attack_config, detector_config)
    except Exception as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(result, indent=2))


@app.command(name="attack-sweep")
def attack_sweep(
    text: str = typer.Option(..., "--text", help="Parent watermarked text to attack"),
    attack_id: str = typer.Option("deletion", "--attack-id", help="Attack type"),
    severities: str = typer.Option(
        "0.0,0.1,0.2,0.3,0.4,0.5",
        "--severities",
        help="Comma-separated severity values",
    ),
    seed: str = typer.Option("sweep-cli", "--seed", help="Random seed"),
    method: str = typer.Option("kgw", "--method", help="Detector method ID"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="Key identifier"),
    threshold: float = typer.Option(4.0, "--threshold", help="Detection threshold"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction"),
    context_width: int = typer.Option(1, "--context-width", help="KGW context width"),
) -> None:
    """Run a severity sweep and report survival across multiple attack strengths."""

    from watermark_lab.run_engine.attack import RobustnessService

    try:
        severity_values = [float(s.strip()) for s in severities.split(",")]
    except ValueError as exc:
        typer.echo(json.dumps({"error": f"Invalid severities: {exc}", "status": "failed"}, indent=2))
        raise typer.Exit(code=2) from exc

    parent_run: dict[str, Any] = {
        "run_id": "cli-sweep-parent",
        "run_type": "create",
        "status": "completed",
        "method_id": method,
        "key_id": key_id,
        "generation": {
            "text": text,
            "method_id": method,
            "key_id": key_id,
            "model_profile": {"simulated": True},
            "config": {
                "gamma": gamma,
                "context_width": context_width,
                "detection_threshold": threshold,
                "seeding_scheme": "simple_1",
                "hash_key": 15485863,
            },
        },
        "governance": {"project_owned": True, "source_is_third_party": False},
        "config": {
            "gamma": gamma,
            "context_width": context_width,
            "detection_threshold": threshold,
            "seeding_scheme": "simple_1",
            "hash_key": 15485863,
        },
    }
    detector_config = DetectionConfig(
        method_id=method,
        tokenizer_id="gpt2",
        key_id=key_id,
        threshold=threshold,
        options={
            "gamma": gamma,
            "context_width": context_width,
            "seeding_scheme": "simple_1",
            "hash_key": 15485863,
        },
    )
    try:
        service = RobustnessService()
        results = service.sweep(parent_run, attack_id, severity_values, detector_config, seed=seed)
    except Exception as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=1) from exc

    summary = {
        "attack_id": attack_id,
        "severities": severity_values,
        "survival_curve": [
            {"severity": r["severity"], "survived": r["survived"], "pre_score": r["pre_attack"]["score"], "post_score": r["post_attack"]["score"]}
            for r in results
        ],
        "results": results,
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command(name="generate-synthetic-corpus")
def generate_synthetic_corpus(
    locale: str = typer.Option(..., "--locale", help="en or zh-Hant-TW"),
    seed: int = typer.Option(20260815, "--seed", help="Random seed"),
    clusters: int = typer.Option(20, "--clusters", min=1, help="Number of clusters"),
    docs_per_cluster: int = typer.Option(5, "--docs-per-cluster", min=1, help="Documents per cluster"),
) -> None:
    """Generate a synthetic corpus for pipeline validation (non-scientific)."""

    from watermark_lab.corpora import SyntheticCorpusError, SyntheticCorpusGenerator

    try:
        gen = SyntheticCorpusGenerator(seed=seed)
        target: Literal["en", "zh-Hant-TW"] = (
            "zh-Hant-TW" if locale in ("zh-Hant-TW", "zh-TW") else "en"
        )
        corpus = gen.generate(
            target, clusters=clusters, documents_per_cluster=docs_per_cluster
        )
    except (ValueError, SyntheticCorpusError) as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(corpus.as_dict(), indent=2))


@app.command(name="ingest-taiwan-public-corpus")
def ingest_taiwan_public_corpus_cli(
    root: str = typer.Option(
        ...,
        "--root",
        help="Local corpus directory containing source-manifest.json and sources/<source_id>/*.txt",
    ),
    locale: str = typer.Option("zh-Hant-TW", "--locale", help="zh-Hant-TW or zh-TW"),
) -> None:
    """Ingest a locally prepared Taiwan public/government corpus with license screening."""

    from watermark_lab.corpora import TaiwanCorpusError, ingest_taiwan_public_corpus

    try:
        report = ingest_taiwan_public_corpus(root, locale=locale)
    except TaiwanCorpusError as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=2) from exc
    payload = report.as_dict()
    payload.pop("documents", None)
    payload["document_count"] = len(report.documents)
    typer.echo(json.dumps(payload, indent=2))


@app.command(name="edit-receipt-demo")
def edit_receipt_demo(
    content: str = typer.Option(..., "--content", help="Original content"),
    edited: str = typer.Option(..., "--edited", help="Edited content"),
    edit_class: str = typer.Option("word", "--edit-class", help="Edit classification"),
    severity: float = typer.Option(0.3, "--severity", min=0.0, max=1.0, help="Edit severity"),
) -> None:
    """Demonstrate EditReceipt DAG with a generation and edit event."""

    from watermark_lab.frontier.edit_receipt import EditReceiptStore

    store = EditReceiptStore()
    store.record_generation(
        event_id="gen-1", content=content, actor="generator",
        timestamp="2026-08-15T10:00:00Z",
    )
    store.record_edit(
        event_id="edit-1", parent_event_id="gen-1",
        original_content=content, edited_content=edited,
        actor="editor", timestamp="2026-08-15T11:00:00Z",
        edit_class=edit_class, severity=severity,
    )
    chain = store.dag.chain("edit-1")
    typer.echo(json.dumps({
        "receipts": [r.as_dict() for r in chain],
        "scientific_status": "skeleton-non-promotable",
    }, indent=2))


@app.command(name="polymark-demo")
def polymark_demo(
    scores_json: str = typer.Option(..., "--scores", help="JSON list of [process_id, start, end, z_score, p_value]"),
) -> None:
    """Demonstrate PolyMark demixing from pre-computed span scores."""

    from watermark_lab.frontier.polymark import PolymarkDemixer, PolymarkProcess

    try:
        raw = json.loads(scores_json)
        processes = (
            PolymarkProcess("p-a", "provider-a", "epoch-001", 0.25, 4.0),
            PolymarkProcess("p-b", "provider-b", "epoch-002", 0.25, 4.0),
        )
        demixer = PolymarkDemixer(processes)
        span_scores = [
            (str(item[0]), int(item[1]), int(item[2]), float(item[3]), float(item[4]))
            for item in raw
        ]
        result = demixer.demix(span_scores)
    except Exception as exc:
        typer.echo(json.dumps({"error": str(exc), "status": "failed"}, indent=2))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.as_dict(), indent=2))


@app.command(name="robust-fragile-demo")
def robust_fragile_demo(
    text: str = typer.Option(..., "--text", help="Text to verify"),
    number: str = typer.Option("42", "--number", help="Number that must be preserved"),
) -> None:
    """Demonstrate robust-plus-fragile dual-channel verification."""

    from watermark_lab.frontier.robust_fragile import (
        InvariantKind,
        RobustFragileWatermarker,
        SemanticInvariant,
    )

    invariants = (
        SemanticInvariant(
            invariant_id="protected-number",
            kind=InvariantKind.NUMBER,
            pattern=rf"\b{number}\b",
            description=f"The number {number} must not change.",
        ),
    )
    wm = RobustFragileWatermarker(invariants)
    result = wm.verify(text=text, payload="demo-payload")
    typer.echo(json.dumps(result.as_dict(), indent=2))


def _stego_kwargs(
    channel: str,
    key_id: str,
    vocab_size: int,
    gamma: float,
    hash_key: int,
) -> dict[str, object]:
    """Build channel-specific kwargs for token_choice; empty for the others."""
    if channel != "token_choice":
        return {}
    return {
        "key_id": key_id,
        "vocab_size": vocab_size,
        "gamma": gamma,
        "hash_key": hash_key,
    }


@app.command(name="stego-encode")
def stego_encode(
    text: str = typer.Option(..., "--text", help="Cover text to hide the payload in"),
    payload: str = typer.Option(..., "--payload", help="Hidden message to embed"),
    channel: str = typer.Option("whitespace", "--channel", help="Steganographic channel: whitespace, unicode, token_choice"),
    secret: str = typer.Option("", "--secret", help="Shared secret for HMAC integrity"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="KGW key identifier (token_choice only)"),
    vocab_size: int = typer.Option(50257, "--vocab-size", help="KGW vocab size (token_choice only)"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction (token_choice only)"),
    hash_key: int = typer.Option(15485863, "--hash-key", help="KGW hash seed (token_choice only)"),
) -> None:
    """Embed a hidden message into cover text with anti-injection HMAC."""
    from watermark_lab.stego import StegoEncoder

    encoder = StegoEncoder(
        channel_id=channel,
        secret=secret.encode("utf-8") if secret else b"",
        **_stego_kwargs(channel, key_id, vocab_size, gamma, hash_key),
    )
    result = encoder.encode(text, payload)
    response = {
        "status": result.status,
        "channel": result.channel,
        "bits_embedded": result.bits_embedded,
        "text": result.text,
        "warnings": list(result.warnings),
        "metadata": result.metadata,
    }
    typer.echo(json.dumps(response, indent=2))


@app.command(name="stego-decode")
def stego_decode(
    text: str = typer.Option(..., "--text", help="Stego text to extract the payload from"),
    channel: str = typer.Option("whitespace", "--channel", help="Steganographic channel: whitespace, unicode, token_choice"),
    secret: str = typer.Option("", "--secret", help="Shared secret for HMAC integrity"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="KGW key identifier (token_choice only)"),
    vocab_size: int = typer.Option(50257, "--vocab-size", help="KGW vocab size (token_choice only)"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction (token_choice only)"),
    hash_key: int = typer.Option(15485863, "--hash-key", help="KGW hash seed (token_choice only)"),
) -> None:
    """Extract and verify a hidden message from stego text."""
    from watermark_lab.stego import StegoDecoder

    decoder = StegoDecoder(
        channel_id=channel,
        secret=secret.encode("utf-8") if secret else b"",
        **_stego_kwargs(channel, key_id, vocab_size, gamma, hash_key),
    )
    result = decoder.decode(text)
    response = {
        "status": result.status,
        "channel": result.channel,
        "payload": result.payload.decode("utf-8", errors="replace") if result.payload else None,
        "bits_embedded": result.bits_embedded,
        "warnings": list(result.warnings),
        "metadata": result.metadata,
    }
    typer.echo(json.dumps(response, indent=2))


@app.command(name="stego-verify")
def stego_verify(
    text: str = typer.Option(..., "--text", help="Stego text to verify"),
    channel: str = typer.Option("whitespace", "--channel", help="Steganographic channel"),
    secret: str = typer.Option("", "--secret", help="Shared secret for HMAC integrity"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="KGW key identifier (token_choice only)"),
    vocab_size: int = typer.Option(50257, "--vocab-size", help="KGW vocab size (token_choice only)"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction (token_choice only)"),
    hash_key: int = typer.Option(15485863, "--hash-key", help="KGW hash seed (token_choice only)"),
) -> None:
    """Verify the integrity of a stego message without revealing the payload."""
    from watermark_lab.stego import StegoDecoder

    decoder = StegoDecoder(
        channel_id=channel,
        secret=secret.encode("utf-8") if secret else b"",
        **_stego_kwargs(channel, key_id, vocab_size, gamma, hash_key),
    )
    result = decoder.verify(text)
    response = {
        "status": result.status,
        "channel": result.channel,
        "reason": result.reason,
        "metadata": result.metadata,
    }
    typer.echo(json.dumps(response, indent=2))


@app.command(name="stego-tinker")
def stego_tinker(
    text: str = typer.Option(..., "--text", help="Cover text to experiment with"),
    payload: str = typer.Option(..., "--payload", help="Hidden message to embed"),
    channel: str = typer.Option("whitespace", "--channel", help="Steganographic channel"),
    secret: str = typer.Option("", "--secret", help="Shared secret for HMAC integrity"),
    generations: int = typer.Option(3, "--generations", help="Number of evolutionary generations"),
    population_size: int = typer.Option(6, "--population-size", help="Variants per generation"),
    mutation_rate: float = typer.Option(0.15, "--mutation-rate", help="Per-token perturbation probability"),
    key_id: str = typer.Option("test-kgw-v1", "--key-id", help="KGW key identifier (token_choice only)"),
    vocab_size: int = typer.Option(50257, "--vocab-size", help="KGW vocab size (token_choice only)"),
    gamma: float = typer.Option(0.25, "--gamma", help="KGW green-list fraction (token_choice only)"),
    hash_key: int = typer.Option(15485863, "--hash-key", help="KGW hash seed (token_choice only)"),
) -> None:
    """Run an iterative stochastic tinker workflow on a steganographic watermark."""
    from watermark_lab.stego import StegoTinkerer

    tinkerer = StegoTinkerer(
        channel_id=channel,
        secret=secret.encode("utf-8") if secret else b"",
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        **_stego_kwargs(channel, key_id, vocab_size, gamma, hash_key),
    )
    summary = tinkerer.run(text, payload)
    typer.echo(json.dumps(summary, indent=2))


@app.command(name="translingua-tw-demo")
def translingua_tw_demo(
    source: str = typer.Option(..., "--source", help="English source text"),
    target: str = typer.Option(..., "--target", help="zh-Hant-TW target text"),
) -> None:
    """Demonstrate Translingua-TW translation handoff verification."""

    from watermark_lab.frontier.translingua_tw import TranslinguaTWHandoffService

    svc = TranslinguaTWHandoffService()
    handoff = svc.create_handoff(
        handoff_id="demo-ho-1",
        source_locale="en",
        target_locale="zh-Hant-TW",
        source_content=source,
        target_content=target,
        payload_ref="demo-workflow-001",
        translation_operation="en-to-zh-Hant-TW-demo",
        invariants=["terminology"],
        signer="demo",
        timestamp="2026-08-15T10:00:00Z",
    )
    result = svc.verify_handoff(handoff, target_content=target)
    typer.echo(json.dumps({
        "handoff": handoff.as_dict(),
        "verification": result.as_dict(),
    }, indent=2))


@app.command(name="war-dial")
def war_dial(
    output_dir: str = typer.Option("artifacts/war-dial", "--output-dir", help="Directory for war-dial artifacts"),
    iterations: int = typer.Option(1, "--iterations", help="Number of grid iterations"),
    attack_sweep: bool = typer.Option(False, "--attack-sweep", help="Run KGW attack sweeps and survival surface"),
    survival_surface: bool = typer.Option(False, "--survival-surface", help="Write survival_surface.json"),
    reconcile: bool = typer.Option(False, "--reconcile", help="Reconcile existing results and plan next grid"),
) -> None:
    """Run the watermark parameter-space war-dial pilot."""
    from pathlib import Path

    from watermark_lab.war_dial.harness import WarDialHarness
    from watermark_lab.war_dial.reconcile import WarDialReconciler

    output_path = Path(output_dir)
    if reconcile:
        reconciler = WarDialReconciler(output_dir=output_path)
        report = reconciler.reconcile()
        typer.echo(json.dumps(report, indent=2))
        return

    harness = WarDialHarness(output_dir=output_path)
    from watermark_lab.war_dial.space import (
        KgwDial,
        PerturbationDial,
        StegoDial,
        WarDialCoordinate,
    )

    prompts = ["the quick brown fox"]
    if not attack_sweep:
        prompts.append("a different prompt text")
    gammas = [0.25, 0.4]
    deltas = [2.0]
    max_new_tokens_list = [80]
    stego_channels = ["whitespace", "unicode"]
    perturbations = [PerturbationDial("builtin-word-jitter", 0.1, "seed-a")]
    if not attack_sweep:
        perturbations.append(PerturbationDial("deletion", 0.1, "seed-b"))

    cover = (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )

    coordinates: list[WarDialCoordinate] = []
    iteration = 0
    for prompt in prompts:
        for gamma in gammas:
            for delta in deltas:
                for max_new_tokens in max_new_tokens_list:
                    for channel in stego_channels:
                        for perturbation in perturbations:
                            coordinates.append(
                                WarDialCoordinate(
                                    kgw=KgwDial(
                                        prompt=prompt,
                                        gamma=gamma,
                                        delta=delta,
                                        context_width=1,
                                        max_new_tokens=max_new_tokens,
                                    ),
                                    stego=StegoDial(
                                        channel=channel,
                                        cover=cover,
                                        payload=f"demo-{iteration:04d}",
                                        secret="war-dial-secret",
                                    ),
                                    perturbation=perturbation,
                                )
                            )
                            iteration += 1

    harness.run_grid(coordinates, resume=True, run_attack_sweep=attack_sweep)
    if survival_surface and attack_sweep:
        from watermark_lab.frontier.edit_receipt import compute_survival_surface

        results_file = output_path / "war_dial_results.jsonl"
        attack_results: list[dict[str, Any]] = []
        if results_file.exists():
            for line in results_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                attack_results.extend(record.get("attack_results", []))
        if attack_results:
            surface = compute_survival_surface(
                attack_results,
                edit_family=["deletion", "insertion", "substitution", "shuffle"],
            )
            surface_path = output_path / "survival_surface.json"
            surface_path.write_text(
                json.dumps(surface.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            typer.echo(f"Wrote {surface_path}")
    summary = harness.summary()
    summary["attack_sweep_enabled"] = attack_sweep
    gate_path = harness.write_gate_report()
    summary["gate_report_path"] = str(gate_path)
    typer.echo(json.dumps(summary, indent=2))


@app.command(name="ocr-engines")
def ocr_engines() -> None:
    """Report which OCR engine profiles can be pinned in this environment."""

    from watermark_lab.ocr import available_engines, get_engine
    from watermark_lab.ocr.profile import OcrProfileError

    payload: dict[str, Any] = {"engines": {}}
    for engine_id, present in available_engines().items():
        entry: dict[str, Any] = {"available": present}
        if present:
            try:
                engine = get_engine(engine_id)
                entry["profile"] = engine.profile.as_dict()
            except OcrProfileError as exc:
                entry["available"] = False
                entry["error"] = str(exc)
        payload["engines"][engine_id] = entry
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command(name="ocr-survival")
def ocr_survival(
    source: Annotated[Path, typer.Argument(help="Text file to round-trip through OCR")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("ocr_survival"),
    engine_id: str = typer.Option("rapidocr", "--engine", help="rapidocr or tesseract"),
    locale: str = typer.Option("en", "--locale", help="en or zh-Hant-TW"),
    payload: str = typer.Option("A", "--payload", help="Payload to embed in each channel"),
    font_size: int = typer.Option(30, "--font-size", min=8, max=200),
    severities: str = typer.Option("0.0,0.25,0.5,0.75,1.0", "--severities"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Measure which provenance channels survive a print-scan-OCR round trip.

    Renders the source text to a page, walks the degradation ladder, recognises
    each degraded page, and reports per-channel payload recovery. Cells where the
    detector dropped whole regions are flagged and excluded from the error
    statistics, because a detection failure is not signal loss.
    """

    from watermark_lab.ocr import (
        build_ladder,
        build_render_settings,
        get_engine,
        measure_channel_survival,
        summarise_surface,
    )
    from watermark_lab.ocr.profile import OcrProfileError
    from watermark_lab.ocr.render import RenderError

    if not source.is_file():
        typer.echo(f"Source file not found: {source}", err=True)
        raise typer.Exit(code=2)

    text = source.read_text(encoding="utf-8").strip()
    if not text:
        typer.echo("Source file is empty.", err=True)
        raise typer.Exit(code=2)

    try:
        parsed = tuple(float(value) for value in severities.split(",") if value.strip())
    except ValueError:
        typer.echo("--severities must be a comma-separated list of numbers.", err=True)
        raise typer.Exit(code=2) from None

    try:
        engine = get_engine(engine_id)
        settings = build_render_settings(text, locale=locale, font_size=font_size)
    except (OcrProfileError, RenderError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for degradation in build_ladder(severities=parsed, seed=seed):
        results.append(
            measure_channel_survival(
                text,
                payload,
                engine=engine,
                settings=settings,
                degradation=degradation,
            )
        )

    records_path = output_dir / "ocr_survival.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")

    summary = summarise_surface(results)
    summary["source_sha256"] = compute_sha256(text)
    summary["locale"] = locale
    summary["records_path"] = str(records_path)
    summary["engine_profile"] = engine.profile.as_dict()
    summary["render_settings"] = settings.as_dict()
    summary["promotion_status"] = (
        "non-promotable: OCR-derived text is a reconstruction, and no OCR-specific "
        "calibration null is frozen for this profile and locale"
    )

    summary_path = output_dir / "ocr_survival_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command(name="generate-real")
def generate_real(
    prompt: Annotated[str, typer.Argument(help="Prompt to generate from")],
    model_dir: Annotated[Path, typer.Option("--model-dir", help="Pinned local model directory")],
    tokenizer_dir: Annotated[Path | None, typer.Option("--tokenizer-dir")] = None,
    max_new_tokens: int = typer.Option(128, "--max-new-tokens", min=1),
    seed: int = typer.Option(0, "--seed"),
    temperature: float = typer.Option(1.0, "--temperature"),
    greedy: bool = typer.Option(False, "--greedy", help="Disable sampling"),
    control: bool = typer.Option(False, "--control", help="Generate the unwatermarked control"),
    output: Annotated[Path | None, typer.Option("--output", help="Write a JSON record here")] = None,
) -> None:
    """Generate watermarked text with a pinned local model, fully offline.

    Weights are never downloaded: a directory without them is refused, because a
    silent fetch would make the recorded model identity describe mutable remote
    bytes. Use --control to produce the matched negative control from the same
    model, prompt and seed.
    """

    from watermark_lab.run_engine.real_generate import (
        PinnedWatermarkGenerator,
        RealGenerationError,
        write_generation_record,
    )

    try:
        generator = PinnedWatermarkGenerator(model_dir, tokenizer_dir=tokenizer_dir)
        result = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=temperature,
            seed=seed,
            watermarked=not control,
        )
    except RealGenerationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    if output is not None:
        written = write_generation_record(result, output)
        typer.echo(f"Wrote {written}", err=True)
    typer.echo(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


@app.command(name="generation-environment")
def generation_environment() -> None:
    """Report whether real-model generation can run in this environment."""

    from watermark_lab.run_engine.real_generate import describe_environment

    typer.echo(json.dumps(describe_environment(), indent=2, ensure_ascii=False))


@app.command(name="ocr-intake")
def ocr_intake(
    source: Annotated[Path, typer.Argument(help="Image or scanned PDF")],
    engine_id: str = typer.Option("rapidocr", "--engine"),
    language: str = typer.Option("auto", "--language"),
    dpi: int = typer.Option(300, "--dpi", min=72, max=1200),
    max_pages: Annotated[int | None, typer.Option("--max-pages", min=1)] = None,
    show_text: bool = typer.Option(False, "--show-text", help="Include recovered text"),
) -> None:
    """Recover text from a scan and report whether it may support a conclusion.

    Recovered text is a reconstruction, not a copy. The detection gate returns at
    most `inconclusive` unless the engine profile is pinned, confidence clears the
    preregistered floor, and an OCR-specific calibration null exists for that
    profile and locale. No such null is frozen yet, so today every scan is
    `inconclusive` by design.
    """

    from watermark_lab.ocr.engine import get_engine
    from watermark_lab.ocr.profile import OcrProfileError
    from watermark_lab.run_engine.ocr_intake import (
        OcrIntakeError,
        ocr_detection_gate,
        read_scanned_source,
    )

    try:
        engine = get_engine(engine_id)
        result = read_scanned_source(
            str(source), engine=engine, language=language, dpi=dpi, max_pages=max_pages
        )
    except (OcrIntakeError, OcrProfileError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    payload = result.as_dict()
    payload["detection_gate"] = ocr_detection_gate(result).as_dict()
    if show_text:
        payload["text"] = result.text
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command(name="screen-zh")
def screen_zh(
    source: Annotated[Path, typer.Argument(help="Text file or directory of .txt files")],
    minimum_cjk: int = typer.Option(50, "--minimum-cjk", min=1),
) -> None:
    """Screen Chinese text for mainland vocabulary carried in Traditional glyphs.

    Character-level Simplified-to-Traditional conversion changes glyphs and not
    vocabulary, so converted text passes a script-level Hant check while still
    reading 軟件 / 網絡 / 信息 where Taiwan writes 軟體 / 網路 / 資訊. This screens for
    that directly.

    A match is evidence of mainland usage, not proof of conversion; absence of a
    match is much weaker evidence than its presence, and is reported as
    `insufficient-evidence` rather than as a pass.
    """

    from watermark_lab.corpora.tw_cn_discriminator import screen, screen_corpus

    if source.is_dir():
        documents = {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(source.glob("*.txt"))
        }
        if not documents:
            typer.echo(f"No .txt files found in {source}", err=True)
            raise typer.Exit(code=2)
        payload: dict[str, Any] = screen_corpus(documents, minimum_cjk=minimum_cjk)
    elif source.is_file():
        payload = screen(source.read_text(encoding="utf-8"), minimum_cjk=minimum_cjk).as_dict()
        payload["source"] = str(source)
    else:
        typer.echo(f"Source not found: {source}", err=True)
        raise typer.Exit(code=2)

    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    app()
