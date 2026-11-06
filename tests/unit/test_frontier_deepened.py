"""Tests for deepened frontier features: survival surface, scan-aware null, edit ontology, factual invariants."""

from __future__ import annotations

import pytest

from watermark_lab.frontier.edit_receipt import (
    EditReceiptError,
    EditReceiptStore,
    compute_survival_surface,
)
from watermark_lab.frontier.polymark import (
    bonferroni_threshold,
    scan_aware_p_value,
)
from watermark_lab.frontier.robust_fragile import (
    EditClass,
    EditOntology,
    is_benign_edit,
    is_material_edit,
)
from watermark_lab.frontier.translingua_tw import (
    FactualInvariant,
    TranslinguaTWHandoffService,
    check_factual_invariants,
    create_factual_handoff,
)


class TestSurvivalSurface:
    def _make_attack_results(self) -> list[dict[str, object]]:
        return [
            {"attack_id": "deletion", "severity": 0.0, "survived": True,
             "pre_attack": {"score": 5.0}, "post_attack": {"score": 5.0}},
            {"attack_id": "deletion", "severity": 0.1, "survived": True,
             "pre_attack": {"score": 5.0}, "post_attack": {"score": 4.5}},
            {"attack_id": "deletion", "severity": 0.3, "survived": False,
             "pre_attack": {"score": 5.0}, "post_attack": {"score": 2.0}},
            {"attack_id": "insertion", "severity": 0.1, "survived": True,
             "pre_attack": {"score": 5.0}, "post_attack": {"score": 4.8}},
        ]

    def test_survival_surface_computed(self) -> None:
        surface = compute_survival_surface(
            self._make_attack_results(),
            edit_family=["deletion", "insertion"],
        )
        assert surface.sample_size == 4
        assert "deletion" in surface.survival_rate_by_class
        assert surface.survival_rate_by_class["deletion"] == pytest.approx(2 / 3)
        assert surface.survival_rate_by_class["insertion"] == pytest.approx(1.0)

    def test_survival_surface_serializable(self) -> None:
        surface = compute_survival_surface(
            self._make_attack_results(),
            edit_family=["deletion", "insertion"],
        )
        data = surface.as_dict()
        assert isinstance(data["points"], list)
        assert len(data["points"]) == 4

    def test_empty_results_rejected(self) -> None:
        with pytest.raises(EditReceiptError, match="empty"):
            compute_survival_surface([], edit_family=["deletion"])

    def test_no_matching_family_rejected(self) -> None:
        with pytest.raises(EditReceiptError, match="matched"):
            compute_survival_surface(
                self._make_attack_results(),
                edit_family=["nonexistent"],
            )

    def test_certificate_with_survival_surface(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1", content="test", actor="g", timestamp="t1"
        )
        surface = compute_survival_surface(
            self._make_attack_results(),
            edit_family=["deletion", "insertion"],
        )
        cert = store.certificate(
            source_receipt_id="gen-1",
            detector_config_hash="abc",
            threshold=4.0,
            edit_family=surface.edit_family,
            survival_estimate=surface.survival_rate_by_class["deletion"],
            survival_lower=0.5,
            survival_upper=0.9,
            sample_size=surface.sample_size,
        )
        assert cert.sample_size == 4


class TestScanAwareNull:
    def test_bonferroni_threshold(self) -> None:
        assert bonferroni_threshold(0.05, 1) == pytest.approx(0.05)
        assert bonferroni_threshold(0.05, 10) == pytest.approx(0.005)
        assert bonferroni_threshold(0.01, 100) == pytest.approx(0.0001)

    def test_bonferroni_invalid_alpha(self) -> None:
        with pytest.raises(Exception, match="alpha"):
            bonferroni_threshold(1.5, 10)

    def test_scan_aware_p_value_increases_with_scans(self) -> None:
        p1 = scan_aware_p_value(3.0, 1)
        p10 = scan_aware_p_value(3.0, 10)
        assert p10 > p1
        assert p1 < 0.01
        assert p10 <= 1.0

    def test_scan_aware_p_value_clamped(self) -> None:
        p = scan_aware_p_value(0.0, 1000)
        assert p == 1.0

    def test_scan_aware_p_value_high_z(self) -> None:
        p = scan_aware_p_value(6.0, 100)
        assert p < 0.01


class TestEditOntology:
    def test_benign_classification(self) -> None:
        ontology = EditOntology()
        assert ontology.classify(EditClass.BENIGN_PUNCTUATION) == "benign"
        assert ontology.classify("benign-whitespace") == "benign"

    def test_material_classification(self) -> None:
        ontology = EditOntology()
        assert ontology.classify(EditClass.MATERIAL_NUMBER_CHANGE) == "material"
        assert ontology.classify("material-citation-change") == "material"

    def test_unknown_classification(self) -> None:
        ontology = EditOntology()
        assert ontology.classify("nonexistent-edit") == "unknown"

    def test_expected_behavior_benign(self) -> None:
        ontology = EditOntology()
        behavior = ontology.expected_behavior(EditClass.BENIGN_TYPO_FIX)
        assert behavior["robust_channel"] == "survives"
        assert behavior["fragile_channel"] == "intact"

    def test_expected_behavior_material(self) -> None:
        ontology = EditOntology()
        behavior = ontology.expected_behavior(EditClass.MATERIAL_NUMBER_CHANGE)
        assert behavior["robust_channel"] == "survives"
        assert behavior["fragile_channel"] == "violated"

    def test_is_benign_edit(self) -> None:
        assert is_benign_edit("benign-punctuation") is True
        assert is_benign_edit("material-number-change") is False

    def test_is_material_edit(self) -> None:
        assert is_material_edit("material-translation") is True
        assert is_material_edit("benign-formatting") is False

    def test_ontology_serializable(self) -> None:
        ontology = EditOntology()
        data = ontology.as_dict()
        assert "benign_classes" in data
        assert "material_classes" in data
        assert len(data["benign_classes"]) == 4
        assert len(data["material_classes"]) == 7


class TestFactualInvariants:
    def test_factual_invariants_preserved(self) -> None:
        invariants = (
            FactualInvariant("num-42", "number", "42", "42", "The answer"),
            FactualInvariant("cite-smith", "citation", "Smith et al.", "Smith 等人", "Citation"),
        )
        violated, intact = check_factual_invariants(
            "答案是 42，根據 Smith 等人的研究。",
            invariants,
        )
        assert len(violated) == 0
        assert len(intact) == 2

    def test_factual_invariants_violated(self) -> None:
        invariants = (
            FactualInvariant("num-42", "number", "42", "42", "The answer"),
            FactualInvariant("cite-smith", "citation", "Smith et al.", "Smith 等人", "Citation"),
        )
        violated, _intact = check_factual_invariants(
            "答案是 43，根據 Jones 等人的研究。",
            invariants,
        )
        assert "num-42" in violated
        assert "cite-smith" in violated

    def test_create_factual_handoff(self) -> None:
        svc = TranslinguaTWHandoffService()
        invariants = (
            FactualInvariant("num-100", "number", "100", "100", "A percentage"),
        )
        _handoff, verification = create_factual_handoff(
            svc,
            handoff_id="factual-1",
            source_content="The accuracy was 100%.",
            target_content="準確率為 100%。",
            payload_ref="project-001",
            translation_operation="en-to-zh-Hant-TW",
            factual_invariants=invariants,
            signer="translator",
            timestamp="2026-08-15T10:00:00Z",
        )
        assert verification.handoff_valid is True
        assert "num-100" not in verification.violated_invariants

    def test_create_factual_handoff_with_violation(self) -> None:
        svc = TranslinguaTWHandoffService()
        invariants = (
            FactualInvariant("num-100", "number", "100", "100", "A percentage"),
        )
        _handoff, verification = create_factual_handoff(
            svc,
            handoff_id="factual-2",
            source_content="The accuracy was 100%.",
            target_content="準確率為 95%。",
            payload_ref="project-001",
            translation_operation="en-to-zh-Hant-TW",
            factual_invariants=invariants,
            signer="translator",
            timestamp="2026-08-15T10:00:00Z",
        )
        assert "num-100" in verification.violated_invariants