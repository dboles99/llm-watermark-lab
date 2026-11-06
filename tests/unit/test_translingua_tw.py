"""Tests for the Translingua-TW translation handoff frontier feature."""

from __future__ import annotations

import pytest

from watermark_lab.frontier.translingua_tw import (
    TranslinguaTWError,
    TranslinguaTWHandoffService,
)


class TestTranslinguaTWHandoff:
    def test_valid_handoff_verification(self) -> None:
        svc = TranslinguaTWHandoffService()
        handoff = svc.create_handoff(
            handoff_id="ho-1",
            source_locale="en",
            target_locale="zh-Hant-TW",
            source_content="The software runs on the network.",
            target_content="這個軟體在網路上運行。",
            payload_ref="project-workflow-001",
            translation_operation="en-to-zh-Hant-TW-human",
            invariants=["terminology"],
            signer="translator",
            timestamp="2026-08-15T10:00:00Z",
        )
        result = svc.verify_handoff(handoff, target_content="這個軟體在網路上運行。")
        assert result.handoff_valid is True
        assert result.provenance_continuous is True
        assert result.taiwan_terminology_ok is True

    def test_zh_cn_terminology_detected(self) -> None:
        svc = TranslinguaTWHandoffService()
        handoff = svc.create_handoff(
            handoff_id="ho-2",
            source_locale="en",
            target_locale="zh-Hant-TW",
            source_content="The software runs on the network.",
            target_content="这个软件在网络上运行。",
            payload_ref="project-001",
            translation_operation="en-to-zh-Hant-TW-machine",
            invariants=["terminology"],
            signer="mt",
            timestamp="2026-08-15T10:00:00Z",
        )
        result = svc.verify_handoff(handoff, target_content="这个软件在网络上运行。")
        assert result.taiwan_terminology_ok is False
        assert "software" in result.violated_invariants
        assert "network" in result.violated_invariants

    def test_content_mismatch_breaks_provenance(self) -> None:
        svc = TranslinguaTWHandoffService()
        handoff = svc.create_handoff(
            handoff_id="ho-3",
            source_locale="en",
            target_locale="zh-Hant-TW",
            source_content="Hello",
            target_content="你好",
            payload_ref="p-1",
            translation_operation="en-to-zh-Hant-TW",
            invariants=[],
            signer="t",
            timestamp="2026-08-15T10:00:00Z",
        )
        result = svc.verify_handoff(handoff, target_content="再見")
        assert result.provenance_continuous is False
        assert "content hash does not match" in " ".join(result.warnings).lower()

    def test_same_locale_rejected(self) -> None:
        svc = TranslinguaTWHandoffService()
        with pytest.raises(TranslinguaTWError, match="must differ"):
            svc.create_handoff(
                handoff_id="ho-x",
                source_locale="en",
                target_locale="en",
                source_content="A",
                target_content="A",
                payload_ref="p",
                translation_operation="none",
                invariants=[],
                signer="t",
                timestamp="2026-08-15T10:00:00Z",
            )

    def test_zh_cn_locale_rejected(self) -> None:
        svc = TranslinguaTWHandoffService()
        with pytest.raises(TranslinguaTWError, match="prohibited"):
            svc.create_handoff(
                handoff_id="ho-zhcn",
                source_locale="en",
                target_locale="zh-CN",
                source_content="A",
                target_content="A",
                payload_ref="p",
                translation_operation="en-to-zh-CN",
                invariants=[],
                signer="t",
                timestamp="2026-08-15T10:00:00Z",
            )

    def test_tampered_signature_detected(self) -> None:
        svc = TranslinguaTWHandoffService()
        handoff = svc.create_handoff(
            handoff_id="ho-tamper",
            source_locale="en",
            target_locale="zh-Hant-TW",
            source_content="Hello",
            target_content="你好",
            payload_ref="p",
            translation_operation="en-to-zh-Hant-TW",
            invariants=[],
            signer="t",
            timestamp="2026-08-15T10:00:00Z",
        )
        from watermark_lab.frontier.translingua_tw import TranslationHandoff

        tampered = TranslationHandoff(
            **{**handoff.as_dict(), "signature": "0" * 64}
        )
        result = svc.verify_handoff(tampered, target_content="你好")
        assert result.handoff_valid is False

    def test_result_serializable(self) -> None:
        svc = TranslinguaTWHandoffService()
        handoff = svc.create_handoff(
            handoff_id="ho-ser",
            source_locale="en",
            target_locale="zh-Hant-TW",
            source_content="Hello",
            target_content="你好",
            payload_ref="p",
            translation_operation="en-to-zh-Hant-TW",
            invariants=["term"],
            signer="t",
            timestamp="2026-08-15T10:00:00Z",
        )
        result = svc.verify_handoff(handoff, target_content="你好")
        data = result.as_dict()
        assert data["scientific_status"] == "skeleton-uncalibrated"
        assert isinstance(data["warnings"], list)

    def test_taiwan_invariants_default_set_present(self) -> None:
        svc = TranslinguaTWHandoffService()
        assert len(svc.taiwan_invariants) >= 5
        ids = {inv.term_id for inv in svc.taiwan_invariants}
        assert "software" in ids
        assert "information" in ids