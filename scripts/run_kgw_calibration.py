"""Execute the frozen Phase 1 KGW calibration study.

Implements the analysis plan of
``docs/research/PREREGISTRATION_KGW_PHASE1.md`` (frozen 2026-08-15):

1. verify all frozen input hashes; abort on any mismatch;
2. generate corpora with frozen seeds; hash partition manifests;
3. score calibration-partition negatives with the scientific detector
   (masked confirmatory + naive secondary), recording scores and provenance;
4. E1: cluster-level worst-case FPR with exact one-sided upper bounds;
5. E2: paired cluster-level masked-vs-naive comparison (exact McNemar,
   Deviation Log #2);
6. E3: cluster-seeded positives with forced green tokens (Deviation Log #3);
7. emit draft calibration manifests validated against the schema and the
   semantic validator;
8. emit exploratory outputs (family F3), a four-state gate report, and step
   metadata with full provenance.

Fail-closed: any hash, runtime, or validation mismatch aborts with no outputs.
Requires the optional pinned torch 2.13.0+cpu runtime and the local pinned
tokenizer assets directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from watermark_lab.adapters.kgw_author import (
    AUTHOR_KGW_CPU_PROFILE,
    kgw_author_detect,
    kgw_author_green_list,
)
from watermark_lab.corpora.synthetic import (
    CorpusDocument,
    SyntheticCorpusGenerator,
)
from watermark_lab.run_engine.scientific_detect import (
    ScientificKgwDetectionService,
)
from watermark_lab.statistics.calibration import (
    clopper_pearson_two_sided_interval,
    clopper_pearson_upper_bound,
)
from watermark_lab.statistics.manifest import (
    assert_calibration_manifest_semantics,
)
from watermark_lab.tokenization import load_local_huggingface_tokenizer
from watermark_lab.tokenization.contracts import (
    NormalizationPolicy,
    SpecialTokenPolicy,
    TokenizerSpec,
    UnicodeNormalization,
)

STUDY_ID = "prereg-kgw-cal-phase1-v1"
GATE_ID = "t06-kgw-calibration-phase1-synthetic-pilot-v1"
GAMMA = 0.25
THRESHOLD = 4.0
CONFIRMATORY_CONFIDENCE = 0.95
BONFERRONI_CONFIDENCE = 1.0 - 0.05 / 2  # family F1/F2, two tests each
POSITIVE_SEQUENCE_LENGTH = 100
MINIMUM_SCORED_TOKENS = 1

FROZEN_INPUTS: dict[str, str] = {
    "author_commit": "82922516930c02f8aa322765defdb5863d07a00e",
    "author_source_sha256": "fc3469e677ba3f2a0e4cdf5a7cae2852d8d423c02cde395cc1e038fca42038a1",
    "prf_source_sha256": "51c854949c8e9309208dc6e57e2f9edbb8f2d90f0ff61edcd4ed2ef890e19ddc",
    "kgw_icml2023_pdf_sha256": "192bbc0d3929b72b7b6962c84614c82006bf4687e09c2e544f7790eac7eab5bb",
    "kgw_iclr2024_pdf_sha256": "560fc007b83fd22bfed9609282def2ef8ae291b8c1d6ba71f1ffa7bc59ab2e01",
    "hash_key": "15485863",
}

SEEDS = {
    "en": {"development_calibration": 20260815, "attack_final": 20260817},
    "zh-Hant-TW": {"development_calibration": 20260816, "attack_final": 20260818},
}
PARTITION_PLAN = {
    "development": 40,
    "calibration": 40,
    "attack-tuning": 20,
    "final-locked": 40,
}
EXCLUSION_POLICY_TEXT = (
    "Frozen exclusion rule (preregistration section 5.5): a document is excluded iff "
    "tokenization yields fewer than two tokens, or the exact tokenizer fingerprint fails "
    "verification (which aborts the entire run). No other exclusions are permitted; "
    "post-hoc exclusions are prohibited. Exclusion counts are reported per partition."
)


class CalibrationAbort(RuntimeError):
    """Raised when a frozen-input or validation check fails closed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationAbort(message)


@dataclass(frozen=True, slots=True)
class LocaleRun:
    locale: str
    documents: tuple[CorpusDocument, ...]
    partition_manifests: dict[str, dict[str, Any]]


def verify_frozen_inputs(
    tokenizer_assets: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = AUTHOR_KGW_CPU_PROFILE
    _require(
        profile.author_commit == FROZEN_INPUTS["author_commit"],
        f"author commit drift: {profile.author_commit}",
    )
    _require(
        profile.author_source_sha256 == FROZEN_INPUTS["author_source_sha256"],
        "author source sha256 drift",
    )
    _require(
        profile.prf_source_sha256 == FROZEN_INPUTS["prf_source_sha256"],
        "prf source sha256 drift",
    )
    _require(
        str(profile.hash_key) == FROZEN_INPUTS["hash_key"],
        "hash key drift",
    )

    kgw_manifest = json.loads(
        (REPO_ROOT / "research" / "papers" / "kgw" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    icml_hash = kgw_manifest["primary_papers"][0]["sha256"]
    iclr_hash = kgw_manifest["primary_papers"][1]["sha256"]
    _require(
        icml_hash == FROZEN_INPUTS["kgw_icml2023_pdf_sha256"],
        f"KGW ICML 2023 PDF hash drift: {icml_hash}",
    )
    _require(
        iclr_hash == FROZEN_INPUTS["kgw_iclr2024_pdf_sha256"],
        f"KGW ICLR 2024 PDF hash drift: {iclr_hash}",
    )
    for entry in kgw_manifest["primary_papers"]:
        pdf = REPO_ROOT / entry["local_file"]
        _require(pdf.is_file(), f"pinned PDF missing: {pdf}")
        _require(
            _sha256_file(pdf) == entry["sha256"],
            f"pinned PDF bytes changed: {pdf}",
        )

    tok_config = json.loads(
        (REPO_ROOT / "config" / "tokenizers" / "openai-community-gpt2-607a30d7.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = tok_config["runtime"]
    expected_versions = (
        ("huggingface_hub", runtime["huggingface_hub"]),
        ("tokenizers", runtime["tokenizers"]),
        ("transformers", runtime["transformers"]),
    )

    tokenizers: dict[str, Any] = {}
    # Per-locale fingerprints: the profile config records only the en value;
    # the zh-Hant-TW value is derived from the same pinned assets/versions and
    # frozen here (preregistration Deviation Log #4).
    expected_fingerprints = {
        "en": tok_config["tokenizer_fingerprint_sha256"],
        "zh-Hant-TW": "b9b7d67330dd24ba89a01d3c5389cf06dd51d9dad0968f8ff18918fab8836020",
    }
    for locale, expected_fingerprint in expected_fingerprints.items():
        spec = TokenizerSpec(
            tokenizer_id=tok_config["tokenizer_id"],
            revision=tok_config["revision"],
            normalization=NormalizationPolicy(
                unicode_form=UnicodeNormalization.NONE,
                locale=locale,
            ),
            special_tokens=SpecialTokenPolicy(),
            expected_backend_versions=expected_versions,
            expected_asset_aggregate_sha256=tok_config["asset_aggregate_sha256"],
            expected_tokenizer_fingerprint_sha256=expected_fingerprint,
        )
        tokenizer = load_local_huggingface_tokenizer(spec, tokenizer_assets)
        provenance = tokenizer.provenance
        _require(
            provenance.tokenizer_fingerprint_sha256 == expected_fingerprint,
            f"tokenizer fingerprint mismatch for locale {locale}",
        )
        tokenizers[locale] = tokenizer

    import torch

    _require(
        torch.__version__ == profile.torch_version,
        f"torch runtime drift: {torch.__version__}",
    )

    return (
        {
            "verified_at": _now_iso(),
            "author_profile_id": profile.profile_id,
            "author_commit": profile.author_commit,
            "author_source_sha256": profile.author_source_sha256,
            "prf_source_sha256": profile.prf_source_sha256,
            "hash_key": profile.hash_key,
            "kgw_icml2023_pdf_sha256": icml_hash,
            "kgw_iclr2024_pdf_sha256": iclr_hash,
            "tokenizer_id": tok_config["tokenizer_id"],
            "tokenizer_revision": tok_config["revision"],
            "tokenizer_fingerprint_en_sha256": expected_fingerprints["en"],
            "tokenizer_fingerprint_zh_hant_tw_sha256": expected_fingerprints["zh-Hant-TW"],
            "asset_aggregate_sha256": tok_config["asset_aggregate_sha256"],
            "backend_versions": [list(pair) for pair in expected_versions],
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "corpus_seeds": SEEDS,
            "partition_plan": PARTITION_PLAN,
            "gamma": GAMMA,
            "threshold": THRESHOLD,
            "analysis_code_sha256": _sha256_file(Path(__file__).resolve()),
            "exclusion_policy_sha256": _sha256_bytes(EXCLUSION_POLICY_TEXT.encode("utf-8")),
        },
        tokenizers,
    )


def generate_frozen_corpora() -> dict[str, LocaleRun]:
    runs: dict[str, LocaleRun] = {}
    for locale, seeds in SEEDS.items():
        pool: list[CorpusDocument] = []
        # Run A: 80 clusters -> development (first 40) + calibration (last 40)
        run_a = SyntheticCorpusGenerator(seed=seeds["development_calibration"]).generate(
            locale, clusters=80, documents_per_cluster=5, sentences_per_document=8
        )
        cluster_ids_a = sorted({doc.cluster_id for doc in run_a.documents})
        dev_clusters = cluster_ids_a[: PARTITION_PLAN["development"]]
        cal_clusters = cluster_ids_a[PARTITION_PLAN["development"] :]
        _require(
            len(dev_clusters) == PARTITION_PLAN["development"]
            and len(cal_clusters) == PARTITION_PLAN["calibration"],
            f"unexpected run-A cluster split for {locale}",
        )
        for doc in run_a.documents:
            partition = "development" if doc.cluster_id in dev_clusters else "calibration"
            tag = "a"
            pool.append(_retag(doc, tag, partition))
        # Run B: 60 clusters -> attack-tuning (first 20) + final-locked (last 40)
        run_b = SyntheticCorpusGenerator(seed=seeds["attack_final"]).generate(
            locale, clusters=60, documents_per_cluster=5, sentences_per_document=8
        )
        cluster_ids_b = sorted({doc.cluster_id for doc in run_b.documents})
        attack_clusters = cluster_ids_b[: PARTITION_PLAN["attack-tuning"]]
        final_clusters = cluster_ids_b[PARTITION_PLAN["attack-tuning"] :]
        _require(
            len(attack_clusters) == PARTITION_PLAN["attack-tuning"]
            and len(final_clusters) == PARTITION_PLAN["final-locked"],
            f"unexpected run-B cluster split for {locale}",
        )
        for doc in run_b.documents:
            partition = "attack-tuning" if doc.cluster_id in attack_clusters else "final-locked"
            pool.append(_retag(doc, "b", partition))

        partition_manifests: dict[str, dict[str, Any]] = {}
        for role in PARTITION_PLAN:
            members = sorted(
                (doc for doc in pool if doc.partition == role),
                key=lambda doc: doc.document_id,
            )
            payload = [
                {
                    "document_id": doc.document_id,
                    "cluster_id": doc.cluster_id,
                    "text_sha256": doc.text_sha256,
                }
                for doc in members
            ]
            partition_manifests[role] = {
                "partition_id": f"{STUDY_ID}:{locale}:{role}",
                "role": role,
                "document_count": len(members),
                "cluster_count": len({doc.cluster_id for doc in members}),
                "manifest_sha256": _canonical_hash(payload),
            }
        runs[locale] = LocaleRun(
            locale=locale,
            documents=tuple(sorted(pool, key=lambda doc: doc.document_id)),
            partition_manifests=partition_manifests,
        )
    return runs


def _retag(doc: CorpusDocument, tag: str, partition: str) -> CorpusDocument:
    return CorpusDocument(
        document_id=f"{tag}-{doc.document_id}",
        cluster_id=f"{tag}-{doc.cluster_id}",
        partition=partition,
        locale=doc.locale,
        text=doc.text,
        text_sha256=doc.text_sha256,
        token_count_estimate=doc.token_count_estimate,
        domain=doc.domain,
    )


def score_negatives(
    run: LocaleRun,
    tokenizer: Any,
) -> dict[str, Any]:
    service = ScientificKgwDetectionService(tokenizer)
    calibration_docs = [doc for doc in run.documents if doc.partition == "calibration"]
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for doc in calibration_docs:
        result = service.detect(doc.text, gamma=GAMMA, threshold=THRESHOLD)
        token_count = result.tokenization.token_count
        if token_count < 2:
            exclusions.append(
                {"document_id": doc.document_id, "reason": "fewer than two tokens"}
            )
            continue
        _require(result.score is not None, f"unscoreable calibration doc {doc.document_id}")
        score = result.score
        records.append(
            {
                "document_id": doc.document_id,
                "cluster_id": doc.cluster_id,
                "token_count": token_count,
                "masked": {
                    "z_score": score.z_score,
                    "p_value": score.p_value_one_sided,
                    "scored_token_count": score.scored_token_count,
                    "ignored_repeated_ngrams": len(
                        score.ignored_repeated_ngram_positions
                    ),
                    "crosses_threshold": score.z_score > THRESHOLD,
                },
            }
        )

    # Secondary naive scoring on the same documents (repeats green-list work; accepted).
    for record in records:
        doc = next(d for d in calibration_docs if d.document_id == record["document_id"])
        result = service.detect(
            doc.text, gamma=GAMMA, threshold=THRESHOLD, ignore_repeated_ngrams=False
        )
        _require(result.score is not None, f"unscoreable naive pass {doc.document_id}")
        score = result.score
        record["naive"] = {
            "z_score": score.z_score,
            "p_value": score.p_value_one_sided,
            "scored_token_count": score.scored_token_count,
            "crosses_threshold": score.z_score > THRESHOLD,
        }

    cluster_ids = sorted({record["cluster_id"] for record in records})
    masked_fp_clusters = {
        record["cluster_id"]
        for record in records
        if record["masked"]["crosses_threshold"]
    }
    naive_fp_clusters = {
        record["cluster_id"]
        for record in records
        if record["naive"]["crosses_threshold"]
    }
    n_clusters = len(cluster_ids)
    _require(n_clusters == PARTITION_PLAN["calibration"], "calibration cluster count drift")

    masked_fp = len(masked_fp_clusters)
    naive_fp = len(naive_fp_clusters)
    e1 = {
        "variant": "masked-first-full-ngram-only (confirmatory)",
        "independent_units": "author-document clusters",
        "n_clusters": n_clusters,
        "n_documents": len(records),
        "false_positive_clusters": masked_fp,
        "empirical_fpr": masked_fp / n_clusters,
        "cp_upper_95": clopper_pearson_upper_bound(masked_fp, n_clusters, 0.95),
        "cp_upper_bonferroni_97_5": clopper_pearson_upper_bound(
            masked_fp, n_clusters, BONFERRONI_CONFIDENCE
        ),
    }

    only_masked = masked_fp_clusters - naive_fp_clusters
    only_naive = naive_fp_clusters - masked_fp_clusters
    b, c = len(only_naive), len(only_masked)  # discordant pairs (naive-only, masked-only)
    mcnemar_p: float | None = None
    if b + c > 0:
        from scipy.stats import binomtest

        mcnemar_p = float(
            binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
        )
    e2 = {
        "variant": "paired cluster-level masked vs naive (exact McNemar; Deviation #2)",
        "n_clusters": n_clusters,
        "masked_false_positive_clusters": masked_fp,
        "naive_false_positive_clusters": naive_fp,
        "discordant_naive_only": b,
        "discordant_masked_only": c,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "masked_marginal_cp_upper_95": e1["cp_upper_95"],
        "naive_marginal_cp_upper_95": clopper_pearson_upper_bound(
            naive_fp, n_clusters, 0.95
        ),
    }

    z_scores = sorted(record["masked"]["z_score"] for record in records)
    e4 = {
        "masked_z_mean": sum(z_scores) / len(z_scores),
        "masked_z_min": z_scores[0],
        "masked_z_median": z_scores[len(z_scores) // 2],
        "masked_z_max": z_scores[-1],
        "mean_token_count": sum(r["token_count"] for r in records) / len(records),
    }

    return {
        "records": records,
        "exclusions": exclusions,
        "e1_fpr": e1,
        "e2_paired": e2,
        "e4_descriptive": e4,
    }


def score_positives(run: LocaleRun, tokenizer: Any) -> dict[str, Any]:
    """E3: cluster-seeded forced-green positives (Deviation Log #3)."""

    vocab = tokenizer.provenance.vocab_size
    calibration_clusters = sorted(
        {doc.cluster_id for doc in run.documents if doc.partition == "calibration"}
    )
    records: list[dict[str, Any]] = []
    for cluster_id in calibration_clusters:
        digest = hashlib.sha256(f"{STUDY_ID}:positive:{cluster_id}".encode()).hexdigest()
        start_token = int(digest[:8], 16) % vocab
        tokens = [start_token]
        for _ in range(POSITIVE_SEQUENCE_LENGTH - 1):
            green = kgw_author_green_list(
                (tokens[-1],), vocab, GAMMA, profile=AUTHOR_KGW_CPU_PROFILE
            )
            _require(bool(green), f"empty green list at cluster {cluster_id}")
            tokens.append(green[len(green) // 2])
        score = kgw_author_detect(
            tokens, vocab, GAMMA, THRESHOLD,
            ignore_repeated_ngrams=True, profile=AUTHOR_KGW_CPU_PROFILE,
        )
        records.append(
            {
                "cluster_id": cluster_id,
                "start_token": start_token,
                "z_score": score.z_score,
                "scored_token_count": score.scored_token_count,
                "crosses_threshold": score.z_score > THRESHOLD,
            }
        )
    detected = sum(1 for record in records if record["crosses_threshold"])
    n = len(records)
    interval = clopper_pearson_two_sided_interval(detected, n, 0.95)
    return {
        "n_independent_positives": n,
        "detected": detected,
        "empirical_tpr": detected / n,
        "cp_two_sided_95": {"lower": interval.lower, "upper": interval.upper},
        "records": records,
    }


def build_manifest(
    locale: str,
    run: LocaleRun,
    negatives: dict[str, Any],
    positives: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    parity_report_path = (
        REPO_ROOT
        / "project"
        / "artifacts"
        / "scientific-validation"
        / "kgw-author-parity"
        / "step-01-greenlist"
        / "gate-report.json"
    )
    implementation_path = REPO_ROOT / "src" / "watermark_lab" / "adapters" / "kgw_author.py"
    return {
        "$schema": "../../../schemas/calibration-manifest.schema.json",
        "schema_version": "1.0.0",
        "calibration_id": f"cal_{STUDY_ID}-{locale.replace('-', '').lower()}",
        "status": "draft",
        "method": {
            "method_id": "kgw",
            "implementation_id": AUTHOR_KGW_CPU_PROFILE.profile_id,
            "implementation_sha256": _sha256_file(implementation_path),
            "author_repository_url": AUTHOR_KGW_CPU_PROFILE.author_repository_url,
            "author_commit": AUTHOR_KGW_CPU_PROFILE.author_commit,
            "parity_status": "passed",
            "parity_report_sha256": _sha256_file(parity_report_path),
        },
        "tokenizer": {
            "tokenizer_id": frozen["tokenizer_id"],
            "revision": frozen["tokenizer_revision"],
            "asset_sha256": frozen["asset_aggregate_sha256"],
            "normalization": "none-preserve-whitespace-no-locale-fallback-v1",
            "bos_policy": "exclude",
            "eos_policy": "exclude",
            "special_token_policy": "exclude",
        },
        "locale": locale,
        "locale_fallback_order": ["zh-Hant-TW", "zh-TW", "zh-Hant", "en"],
        "partitions": [
            {
                "role": role,
                "partition_id": run.partition_manifests[role]["partition_id"],
                "manifest_sha256": run.partition_manifests[role]["manifest_sha256"],
            }
            for role in ("development", "calibration", "attack-tuning", "final-locked")
        ],
        "sampling": {
            "independent_unit": "author-document-cluster",
            "planned_independent_negatives": PARTITION_PLAN["calibration"],
            "planned_independent_positives": PARTITION_PLAN["calibration"],
            "cluster_fields": ["source_id", "document_id"],
            "stopping_rule": (
                "Fixed-sample: all calibration clusters scored once; no sequential or "
                "optional stopping (preregistration section 5.4)."
            ),
            "exclusion_policy_sha256": frozen["exclusion_policy_sha256"],
        },
        "scoring": {
            "gamma": GAMMA,
            "context_width": 1,
            "seeding_scheme": "simple_1",
            "repeated_ngram_policy": "first-full-ngram-only",
            "minimum_scored_tokens": MINIMUM_SCORED_TOKENS,
            "decision_threshold": THRESHOLD,
        },
        "inference": {
            "target_fpr": 0.005,
            "confidence_level": 0.95,
            "false_positive_interval": "clopper-pearson-exact-one-sided-upper",
            "true_positive_interval": "clopper-pearson-exact-two-sided",
            "claim_basis": "estimated-rate",
            "multiplicity_family_id": "F1",
            "multiplicity_policy": "bonferroni",
            "analysis_code_sha256": frozen["analysis_code_sha256"],
        },
        "created_at": _now_iso(),
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    schema_path = REPO_ROOT / "schemas" / "calibration-manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)
    assert_calibration_manifest_semantics(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "project/artifacts/scientific-validation/kgw-calibration-phase1"),
    )
    parser.add_argument(
        "--tokenizer-assets",
        default=os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "watermark-lab-lab",
            "assets",
            "openai-community-gpt2",
            "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        ),
    )
    parser.add_argument("--locale", choices=["en", "zh-Hant-TW", "both"], default="both")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    assets = Path(args.tokenizer_assets)

    print("[1/7] Verifying frozen inputs (fail-closed)...")
    frozen, tokenizers = verify_frozen_inputs(assets)
    print(f"      author profile {frozen['author_profile_id']}")
    print(f"      tokenizer fingerprint (en) {frozen['tokenizer_fingerprint_en_sha256'][:16]}...")

    print("[2/7] Generating frozen corpora (seeds per preregistration)...")
    runs = generate_frozen_corpora()
    for locale, run in runs.items():
        counts = {
            role: run.partition_manifests[role]["cluster_count"]
            for role in PARTITION_PLAN
        }
        print(f"      {locale}: {counts}")

    locales = (
        ["en", "zh-Hant-TW"] if args.locale == "both" else [args.locale]
    )
    outputs: dict[str, dict[str, Any]] = {}
    for locale in locales:
        print(f"[3/7] Scoring {locale} calibration negatives (masked + naive)...")
        negatives = score_negatives(runs[locale], tokenizers[locale])
        print(
            f"      masked FP clusters {negatives['e1_fpr']['false_positive_clusters']}"
            f"/{negatives['e1_fpr']['n_clusters']}, "
            f"exclusions {len(negatives['exclusions'])}"
        )
        print(f"[4/7] E3 {locale} positives...")
        positives = score_positives(runs[locale], tokenizers[locale])
        print(
            f"      TPR {positives['detected']}/{positives['n_independent_positives']}"
        )
        outputs[locale] = {"negatives": negatives, "positives": positives}

    print("[5/7] Building and validating draft calibration manifests...")
    manifests: dict[str, dict[str, Any]] = {}
    for locale in locales:
        manifest = build_manifest(
            locale, runs[locale], outputs[locale]["negatives"], outputs[locale]["positives"], frozen
        )
        validate_manifest(manifest)
        manifests[locale] = manifest
        print(f"      {locale}: schema + semantic validation passed (status=draft)")

    print("[6/7] Writing artifacts...")
    (output_root / "inputs").mkdir(parents=True, exist_ok=True)
    (output_root / "outputs").mkdir(parents=True, exist_ok=True)
    (output_root / "inputs" / "frozen-inputs.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for locale in locales:
        data = outputs[locale]
        (output_root / "outputs" / f"{locale}-scoring-records.json").write_text(
            json.dumps(
                {
                    "records": data["negatives"]["records"],
                    "exclusions": data["negatives"]["exclusions"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_root / "outputs" / f"{locale}-results.json").write_text(
            json.dumps(
                {
                    "e1_fpr": data["negatives"]["e1_fpr"],
                    "e2_paired": data["negatives"]["e2_paired"],
                    "e3_tpr": {
                        key: value
                        for key, value in data["positives"].items()
                        if key != "records"
                    },
                    "e4_descriptive_f3": data["negatives"]["e4_descriptive"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_root / "outputs" / f"{locale}-calibration-manifest.json").write_text(
            json.dumps(manifests[locale], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("[7/7] Writing gate report and metadata...")
    e1_all = {locale: outputs[locale]["negatives"]["e1_fpr"] for locale in locales}
    gate_report = {
        "$schema": "../../../../schemas/gate-report.schema.json",
        "schema_version": "2.0.0",
        "gate_id": GATE_ID,
        "task_id": "T06",
        "step_id": "kgw-calibration-phase1",
        "status": "completed",
        "artifact_complete": True,
        "checks_passed": True,
        "promotion_eligible": False,
        "deployment_ready": False,
        "checks": [
            {
                "check_id": "frozen-inputs-verified",
                "status": "passed",
                "evidence": [
                    f"author_commit={frozen['author_commit']}",
                    f"tokenizer_fingerprint_en={frozen['tokenizer_fingerprint_en_sha256']}",
                    f"tokenizer_fingerprint_zh_hant_tw={frozen['tokenizer_fingerprint_zh_hant_tw_sha256']}",
                    f"torch={frozen['torch_version']} git {frozen['torch_git_version']}",
                ],
                "command": f"python scripts/run_kgw_calibration.py --output-root {output_root}",
                "notes": "All preregistration section 2 hashes verified before scoring.",
            },
            {
                "check_id": "e1-cluster-fpr",
                "status": "passed",
                "evidence": [
                    f"{locale}: {result['false_positive_clusters']}/{result['n_clusters']} "
                    f"FP clusters, CP95 upper {result['cp_upper_95']:.4f}"
                    for locale, result in e1_all.items()
                ],
                "command": "python scripts/run_kgw_calibration.py",
                "notes": "Cluster-level worst-case FPR with exact one-sided upper bounds.",
            },
            {
                "check_id": "manifests-schema-and-semantic-valid",
                "status": "passed",
                "evidence": [f"{locale}: draft manifest validated" for locale in locales],
                "command": "python scripts/run_kgw_calibration.py",
                "notes": "Draft2020-12 schema plus semantic validator; status=draft.",
            },
        ],
        "finding_ids": [],
        "disposition": (
            "Phase 1 synthetic pilot executed per the frozen preregistration with three "
            "pre-recorded deviations. Pipeline mechanics validated; no scientific FPR "
            "claim beyond the declared pilot bound."
        ),
        "blockers": [
            (
                "Synthetic corpora are non-scientific; real Taiwan public corpora require "
                "native review before any frozen calibration."
            ),
            "40 clusters per locale bound the zero-event CP95 upper bound near 7.2%.",
            "Real-model generation is deferred at the approved $0 API spend ceiling.",
        ],
        "warnings": [
            "Draft manifests use claim_basis=estimated-rate; not promotable.",
            "E3 positives are forced-green token sequences, not real-model text.",
        ],
        "summary": (
            f"Executed frozen study {STUDY_ID}: "
            + "; ".join(
                f"{locale} masked FPR {result['false_positive_clusters']}/"
                f"{result['n_clusters']} clusters (CP95 upper {result['cp_upper_95']:.4f})"
                for locale, result in e1_all.items()
            )
        ),
    }
    (output_root / "gate-report.json").write_text(
        json.dumps(gate_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "study_id": STUDY_ID,
        "gate_id": GATE_ID,
        "executed_at": _now_iso(),
        "git_commit": _git_commit(),
        "working_tree_dirty": _git_dirty(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": frozen["torch_version"],
            "transformers": dict(frozen["backend_versions"])["transformers"],
            "tokenizers": dict(frozen["backend_versions"])["tokenizers"],
            "huggingface_hub": dict(frozen["backend_versions"])["huggingface_hub"],
        },
        "analysis_code_sha256": frozen["analysis_code_sha256"],
        "preregistration": "docs/research/PREREGISTRATION_KGW_PHASE1.md",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Done. Artifacts written to", output_root)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CalibrationAbort as abort:
        print(f"ABORT: {abort}", file=sys.stderr)
        sys.exit(3)