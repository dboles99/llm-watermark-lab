"""Tests for the EditReceipt frontier feature."""

from __future__ import annotations

import pytest

from watermark_lab.frontier.edit_receipt import (
    CounterfactualCertificate,
    EditReceiptDAG,
    EditReceiptError,
    EditReceiptStore,
    ReceiptEventKind,
)


class TestEditReceiptDAG:
    def test_generation_receipt_has_no_parents(self) -> None:
        dag = EditReceiptDAG()
        store = EditReceiptStore(dag)
        receipt = store.record_generation(
            event_id="gen-1",
            content="Hello world",
            actor="generator",
            timestamp="2026-08-15T10:00:00Z",
        )
        assert receipt.event.kind == ReceiptEventKind.GENERATION
        assert receipt.event.parent_event_ids == ()
        assert receipt.receipt_sha256
        assert receipt.signature
        assert dag.verify(receipt)

    def test_edit_receipt_links_to_parent(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1",
            content="Hello world",
            actor="gen",
            timestamp="2026-08-15T10:00:00Z",
        )
        edit = store.record_edit(
            event_id="edit-1",
            parent_event_id="gen-1",
            original_content="Hello world",
            edited_content="Hello, world!",
            actor="editor",
            timestamp="2026-08-15T11:00:00Z",
            edit_class="punctuation",
            severity=0.1,
        )
        assert edit.event.parent_event_ids == ("gen-1",)
        assert edit.event.metadata["edit_class"] == "punctuation"
        assert edit.event.metadata["severity"] == 0.1

    def test_unknown_parent_rejected(self) -> None:
        store = EditReceiptStore()
        with pytest.raises(EditReceiptError, match="Unknown parent"):
            store.record_edit(
                event_id="edit-1",
                parent_event_id="nonexistent",
                original_content="a",
                edited_content="b",
                actor="editor",
                timestamp="2026-08-15T11:00:00Z",
                edit_class="punctuation",
                severity=0.1,
            )

    def test_duplicate_event_rejected(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1",
            content="Hello",
            actor="gen",
            timestamp="2026-08-15T10:00:00Z",
        )
        with pytest.raises(EditReceiptError, match="Duplicate"):
            store.record_generation(
                event_id="gen-1",
                content="World",
                actor="gen",
                timestamp="2026-08-15T10:00:00Z",
            )

    def test_chain_returns_ordered_ancestry(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1", content="A", actor="g", timestamp="t1"
        )
        store.record_edit(
            event_id="edit-1", parent_event_id="gen-1",
            original_content="A", edited_content="B", actor="e",
            timestamp="t2", edit_class="word", severity=0.3,
        )
        store.record_edit(
            event_id="edit-2", parent_event_id="edit-1",
            original_content="B", edited_content="C", actor="e",
            timestamp="t3", edit_class="word", severity=0.3,
        )
        chain = store.dag.chain("edit-2")
        assert [r.event.event_id for r in chain] == ["gen-1", "edit-1", "edit-2"]

    def test_leaf_and_root_receipts(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1", content="A", actor="g", timestamp="t1"
        )
        store.record_edit(
            event_id="edit-1", parent_event_id="gen-1",
            original_content="A", edited_content="B", actor="e",
            timestamp="t2", edit_class="word", severity=0.3,
        )
        roots = store.dag.root_receipts()
        leaves = store.dag.leaf_receipts()
        assert len(roots) == 1 and roots[0].event.event_id == "gen-1"
        assert len(leaves) == 1 and leaves[0].event.event_id == "edit-1"

    def test_invalid_severity_rejected(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1", content="A", actor="g", timestamp="t1"
        )
        with pytest.raises(EditReceiptError, match="severity"):
            store.record_edit(
                event_id="edit-1", parent_event_id="gen-1",
                original_content="A", edited_content="B", actor="e",
                timestamp="t2", edit_class="word", severity=1.5,
            )

    def test_certificate_valid_bounds(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1", content="A", actor="g", timestamp="t1"
        )
        cert = store.certificate(
            source_receipt_id="gen-1",
            detector_config_hash="abc123",
            threshold=4.0,
            edit_family=["punctuation", "word-swap"],
            survival_estimate=0.85,
            survival_lower=0.75,
            survival_upper=0.92,
            sample_size=500,
        )
        assert isinstance(cert, CounterfactualCertificate)
        assert cert.source_receipt_id == "gen-1"
        assert cert.survival_lower_bound <= cert.survival_estimate <= cert.survival_upper_bound
        assert "not universal robustness" in cert.caveats[0]

    def test_certificate_invalid_bounds_rejected(self) -> None:
        store = EditReceiptStore()
        store.record_generation(
            event_id="gen-1", content="A", actor="g", timestamp="t1"
        )
        with pytest.raises(EditReceiptError, match="survival bounds"):
            store.certificate(
                source_receipt_id="gen-1",
                detector_config_hash="abc",
                threshold=4.0,
                edit_family=["x"],
                survival_estimate=0.5,
                survival_lower=0.6,
                survival_upper=0.4,
            )

    def test_tampered_receipt_fails_verification(self) -> None:
        dag = EditReceiptDAG()
        store = EditReceiptStore(dag)
        receipt = store.record_generation(
            event_id="gen-1", content="A", actor="g", timestamp="t1"
        )
        from watermark_lab.frontier.edit_receipt import EditReceipt

        tampered = EditReceipt(
            event=receipt.event,
            receipt_sha256="0" * 64,
            signature=receipt.signature,
        )
        assert not dag.verify(tampered)