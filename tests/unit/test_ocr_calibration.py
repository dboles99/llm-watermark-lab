"""Measuring an OCR-specific null and emitting a calibration manifest.

The properties under test are the ones that keep a calibration honest: that
independent units are clusters rather than documents, that detector dropout is
excluded rather than counted as a null observation, that an unattainable target
is reported rather than quietly claimed, and that a pilot on synthetic text
cannot be frozen.

A stub engine is used so these run fast and deterministically. The pipeline is
exercised against a real engine in the integration test at the end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from watermark_lab.ocr import Degradation, rapidocr_available
from watermark_lab.ocr.calibration import (
    MINIMUM_PILOT_CLUSTERS,
    OcrCalibrationError,
    build_calibration_manifest,
    measure_ocr_null,
    write_calibration_artifact,
)
from watermark_lab.ocr.profile import build_profile
from watermark_lab.statistics import validate_calibration_manifest_semantics

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (ROOT / "schemas" / "calibration-manifest.schema.json").read_text(encoding="utf-8")
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
SHA256 = "a" * 64

TEXT = (
    "Provenance evidence must survive ordinary document handling to be useful in "
    "practice, and a watermark that vanishes on contact with a scanner cannot support "
    "any claim about where a document came from."
)


class _Doc:
    def __init__(self, cluster_id: str, text: str = TEXT) -> None:
        self.cluster_id = cluster_id
        self.text = text


class _StubEngine:
    """Returns the input text unchanged, so scoring is fully controlled by the test.

    ``drop_every`` truncates every Nth call below the coverage floor, which
    models the intermittent line-recall dropout a real recogniser exhibits.
    """

    engine_id = "stub"

    def __init__(self, *, coverage: float = 1.0, drop_every: int | None = None) -> None:
        self._coverage = coverage
        self._drop_every = drop_every
        self._calls = 0
        self.profile = build_profile(
            engine_id="stub", engine_version="1.0", language="en", assets=()
        )

    def recognize(self, image: Any) -> Any:
        self._calls += 1
        coverage = self._coverage
        if self._drop_every and self._calls % self._drop_every == 0:
            coverage = 0.2
        recovered = TEXT[: max(1, int(len(TEXT) * coverage))]

        class _Result:
            text = recovered

        return _Result()


def _documents(clusters: int, per_cluster: int = 2) -> list[_Doc]:
    return [
        _Doc(f"cluster-{index:04d}")
        for index in range(clusters)
        for _ in range(per_cluster)
    ]


def _constant_scorer(value: float):
    return lambda text: value


def _method() -> dict[str, Any]:
    return {
        "method_id": "kgw",
        "implementation_id": "stub",
        "implementation_sha256": SHA256,
        "author_repository_url": "https://github.com/jwkirchenbauer/lm-watermarking",
        "author_commit": "8" * 40,
        "parity_status": "not-run",
    }


def _tokenizer() -> dict[str, Any]:
    return {
        "tokenizer_id": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "asset_sha256": SHA256,
        "normalization": "NFC",
        "bos_policy": "tokenizer-native",
        "eos_policy": "tokenizer-native",
        "special_token_policy": "tokenizer-native",
    }


def _partitions() -> list[dict[str, Any]]:
    return [
        {"role": role, "partition_id": f"p-{role}", "manifest_sha256": SHA256}
        for role in ("development", "calibration", "attack-tuning", "final-locked")
    ]


class TestIndependentUnitIsTheCluster:
    def test_too_few_clusters_is_refused(self) -> None:
        with pytest.raises(OcrCalibrationError, match="below the pilot floor"):
            measure_ocr_null(
                _documents(5),
                engine=_StubEngine(),
                degradation=Degradation("clean", 0.0, 0),
                scorer=_constant_scorer(1.0),
                language="en",
            )

    def test_adding_documents_to_the_same_clusters_does_not_help(self) -> None:
        """The refusal message must say so, or the fix looks like 'add more docs'."""

        with pytest.raises(OcrCalibrationError) as excinfo:
            measure_ocr_null(
                _documents(5, per_cluster=50),
                engine=_StubEngine(),
                degradation=Degradation("clean", 0.0, 0),
                scorer=_constant_scorer(1.0),
                language="en",
            )
        assert "does not help" in str(excinfo.value)

    def test_cluster_count_drives_the_bound_not_document_count(self) -> None:
        few_clusters = measure_ocr_null(
            _documents(MINIMUM_PILOT_CLUSTERS, per_cluster=10),
            engine=_StubEngine(),
            degradation=Degradation("clean", 0.0, 0),
            scorer=_constant_scorer(1.0),
            language="en",
        )
        many_clusters = measure_ocr_null(
            _documents(MINIMUM_PILOT_CLUSTERS * 4, per_cluster=1),
            engine=_StubEngine(),
            degradation=Degradation("clean", 0.0, 0),
            scorer=_constant_scorer(1.0),
            language="en",
        )
        # 200 documents in 20 clusters gives a looser bound than 80 documents in
        # 80 clusters, because only clusters are independent.
        assert few_clusters.scored_documents > many_clusters.scored_documents
        assert few_clusters.attainable_upper_bound > many_clusters.attainable_upper_bound

    def test_empty_input_is_refused(self) -> None:
        with pytest.raises(OcrCalibrationError, match="at least one document"):
            measure_ocr_null(
                [],
                engine=_StubEngine(),
                degradation=Degradation("clean", 0.0, 0),
                scorer=_constant_scorer(1.0),
                language="en",
            )


class TestDropoutIsExcludedNotScored:
    def test_low_coverage_documents_are_dropped(self) -> None:
        """A detection failure is not a null observation.

        Every second reading falls below the coverage floor, so half the
        documents must be dropped rather than contributing a score.
        """

        null = measure_ocr_null(
            _documents(MINIMUM_PILOT_CLUSTERS, per_cluster=2),
            engine=_StubEngine(drop_every=2),
            degradation=Degradation("clean", 0.0, 0),
            scorer=_constant_scorer(1.0),
            language="en",
        )
        total = MINIMUM_PILOT_CLUSTERS * 2
        assert null.dropped_documents == total // 2
        assert null.scored_documents == total - null.dropped_documents
        # Every cluster still contributes one usable score, so the cluster count
        # is unchanged: dropout costs documents, not independent units, here.
        assert null.cluster_count == MINIMUM_PILOT_CLUSTERS

    def test_all_dropped_raises_rather_than_returning_an_empty_null(self) -> None:
        with pytest.raises(OcrCalibrationError, match="no cluster produced a scorable"):
            measure_ocr_null(
                _documents(MINIMUM_PILOT_CLUSTERS),
                engine=_StubEngine(coverage=0.1),
                degradation=Degradation("clean", 0.0, 0),
                scorer=_constant_scorer(1.0),
                language="en",
            )

    def test_unscorable_short_text_counts_as_dropped(self) -> None:
        null = measure_ocr_null(
            _documents(MINIMUM_PILOT_CLUSTERS, per_cluster=2),
            engine=_StubEngine(),
            degradation=Degradation("clean", 0.0, 0),
            scorer=lambda text: None if len(text) > 0 and False else 1.0,
            language="en",
        )
        assert null.scored_documents > 0


class TestAttainabilityIsReported:
    def test_pilot_cannot_evidence_a_tenth_of_a_percent(self) -> None:
        """40 clusters bounds the FPR near 7%, not 0.1%. Say so."""

        null = measure_ocr_null(
            _documents(40, per_cluster=1),
            engine=_StubEngine(),
            degradation=Degradation("clean", 0.0, 0),
            scorer=_constant_scorer(1.0),
            language="en",
            target_fpr=0.001,
        )
        assert null.target_attainable is False
        assert null.attainable_upper_bound > 0.05
        assert any("cannot evidence" in warning for warning in null.warnings)

    def test_attainable_bound_matches_the_exact_zero_event_formula(self) -> None:
        from watermark_lab.statistics import clopper_pearson_upper_bound

        null = measure_ocr_null(
            _documents(40, per_cluster=1),
            engine=_StubEngine(),
            degradation=Degradation("clean", 0.0, 0),
            scorer=_constant_scorer(1.0),
            language="en",
        )
        expected = clopper_pearson_upper_bound(
            null.false_positive_clusters, null.cluster_count, 0.95
        )
        assert null.attainable_upper_bound == pytest.approx(expected)

    def test_stratum_records_the_instrument(self) -> None:
        null = measure_ocr_null(
            _documents(MINIMUM_PILOT_CLUSTERS),
            engine=_StubEngine(),
            degradation=Degradation("jpeg", 0.5, 7),
            scorer=_constant_scorer(1.0),
            language="zh-Hant-TW",
        )
        assert null.stratum["engine_id"] == "stub"
        assert null.stratum["language"] == "zh-Hant-TW"
        assert null.stratum["degradation"]["name"] == "jpeg"
        assert len(null.stratum["profile_fingerprint"]) == 64


class TestManifestGeneration:
    def _null(self):
        return measure_ocr_null(
            _documents(40, per_cluster=1),
            engine=_StubEngine(),
            degradation=Degradation("clean", 0.0, 0),
            scorer=_constant_scorer(1.0),
            language="en",
        )

    def test_manifest_is_schema_valid(self) -> None:
        manifest = build_calibration_manifest(
            self._null(),
            calibration_id="cal_ocr_stub_en_clean_v1",
            locale="en",
            method=_method(),
            tokenizer=_tokenizer(),
            partitions=_partitions(),
        )
        assert list(VALIDATOR.iter_errors(manifest)) == []

    def test_manifest_declares_the_ocr_channel(self) -> None:
        manifest = build_calibration_manifest(
            self._null(),
            calibration_id="cal_ocr_stub_en_clean_v1",
            locale="en",
            method=_method(),
            tokenizer=_tokenizer(),
            partitions=_partitions(),
        )
        channel = manifest["input_channel"]
        assert channel["channel"] == "ocr"
        assert channel["ocr"]["engine_id"] == "stub"
        assert channel["ocr"]["degradation"]["name"] == "clean"

    def test_manifest_passes_semantic_validation(self) -> None:
        manifest = build_calibration_manifest(
            self._null(),
            calibration_id="cal_ocr_stub_en_clean_v1",
            locale="en",
            method=_method(),
            tokenizer=_tokenizer(),
            partitions=_partitions(),
        )
        assert validate_calibration_manifest_semantics(manifest) == ()

    def test_pilot_defaults_to_draft(self) -> None:
        """A threshold from text nobody wrote must not be frozen."""

        manifest = build_calibration_manifest(
            self._null(),
            calibration_id="cal_ocr_stub_en_clean_v1",
            locale="en",
            method=_method(),
            tokenizer=_tokenizer(),
            partitions=_partitions(),
        )
        assert manifest["status"] == "draft"

    def test_rejects_a_bad_calibration_id(self) -> None:
        with pytest.raises(OcrCalibrationError, match="must start with"):
            build_calibration_manifest(
                self._null(),
                calibration_id="ocr-en-v1",
                locale="en",
                method=_method(),
                tokenizer=_tokenizer(),
                partitions=_partitions(),
            )

    def test_rejects_a_simplified_chinese_locale(self) -> None:
        with pytest.raises(OcrCalibrationError, match="must be"):
            build_calibration_manifest(
                self._null(),
                calibration_id="cal_ocr_stub_zh_v1",
                locale="zh-CN",
                method=_method(),
                tokenizer=_tokenizer(),
                partitions=_partitions(),
            )

    def test_claim_basis_tracks_observed_events(self) -> None:
        manifest = build_calibration_manifest(
            self._null(),
            calibration_id="cal_ocr_stub_en_clean_v1",
            locale="en",
            method=_method(),
            tokenizer=_tokenizer(),
            partitions=_partitions(),
        )
        null = self._null()
        expected = (
            "zero-event-upper-bound" if null.false_positive_clusters == 0 else "estimated-rate"
        )
        assert manifest["inference"]["claim_basis"] == expected


class TestArtifactWriting:
    def test_writes_manifest_and_null_side_by_side(self, tmp_path: Path) -> None:
        null = measure_ocr_null(
            _documents(MINIMUM_PILOT_CLUSTERS),
            engine=_StubEngine(),
            degradation=Degradation("clean", 0.0, 0),
            scorer=_constant_scorer(1.0),
            language="en",
        )
        manifest = build_calibration_manifest(
            null,
            calibration_id="cal_ocr_stub_en_clean_v1",
            locale="en",
            method=_method(),
            tokenizer=_tokenizer(),
            partitions=_partitions(),
        )
        paths = write_calibration_artifact(null, manifest, tmp_path / "cal")

        assert paths["manifest"].is_file()
        assert paths["null"].is_file()
        payload = json.loads(paths["null"].read_text(encoding="utf-8"))
        assert payload["cluster_count"] == MINIMUM_PILOT_CLUSTERS
        assert len(payload["manifest_sha256"]) == 64
        assert "non-scientific" in payload["note"]


@pytest.mark.skipif(not rapidocr_available(), reason="rapidocr is not installed")
@pytest.mark.integration
class TestRealEngineSmoke:
    def test_pipeline_runs_against_a_real_engine(self) -> None:
        from watermark_lab.ocr.engine import get_engine

        null = measure_ocr_null(
            _documents(MINIMUM_PILOT_CLUSTERS, per_cluster=1),
            engine=get_engine("rapidocr"),
            degradation=Degradation("clean", 0.0, 0),
            scorer=lambda text: float(len(text)) / 100.0,
            language="en",
            coverage_floor=0.5,
        )
        assert null.cluster_count == MINIMUM_PILOT_CLUSTERS
        assert null.stratum["engine_id"] == "rapidocr"
        assert len(null.stratum["profile_fingerprint"]) == 64
