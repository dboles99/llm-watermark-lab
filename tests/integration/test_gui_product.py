"""Headless-friendly smoke tests for the initial product views."""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from watermark_lab.gui.main import MainWindow
from watermark_lab.services.product_workspace import ProductWorkspaceService


class FakeRepository:
    def list_runs(self) -> list[dict[str, Any]]:
        return [{"run_id": "run-1", "status": "completed", "method_id": "kgw"}]

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": "artifact-1",
                "artifact_type": "report",
                "status": "frozen",
                "path": "runs/run-1/report.json",
            }
        ]


def test_product_views_use_shared_service_without_showing_window() -> None:
    _ = QApplication.instance() or QApplication([])
    service = ProductWorkspaceService(FakeRepository())
    window = MainWindow(service)

    assert window.runs_tab.table.rowCount() == 1
    assert window.runs_tab.table.item(0, 0).text() == "run-1"
    assert window.artifact_history_tab.table.rowCount() == 1
    assert window.artifact_history_tab.table.item(0, 0).text() == "artifact-1"
    labels = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert "Intake" in labels
    assert "Detect" in labels
    assert "Create" in labels
    assert "Runs" in labels
    assert "Artifact History" in labels
    assert "Research" in labels


def test_settings_offer_only_safe_locale_and_both_history_modes() -> None:
    _ = QApplication.instance() or QApplication([])
    service = ProductWorkspaceService(FakeRepository())
    window = MainWindow(service)

    locale_values = [
        window.settings_tab.locale_combo.itemData(index)
        for index in range(window.settings_tab.locale_combo.count())
    ]
    mode_values = [
        window.settings_tab.history_mode_combo.itemData(index)
        for index in range(window.settings_tab.history_mode_combo.count())
    ]
    assert locale_values == ["en", "zh-Hant-TW"]
    assert "zh-CN" not in locale_values
    assert mode_values == ["supersede-and-preserve", "amend-in-place"]

    window.settings_tab.locale_combo.setCurrentIndex(1)
    window.settings_tab.history_mode_combo.setCurrentIndex(1)
    assert service.locale == "zh-Hant-TW"
    assert service.history_mode.value == "amend-in-place"


def test_research_tab_offers_all_features_and_governed_choices() -> None:
    _ = QApplication.instance() or QApplication([])
    service = ProductWorkspaceService(FakeRepository())
    window = MainWindow(service)

    assert list(window.research_tab.feature_checkboxes) == [
        "edit-receipt",
        "polymark-demixing",
        "robust-fragile",
        "translingua-tw",
    ]
    execution_values = [
        str(window.research_tab.execution_combo.itemData(index))
        for index in range(window.research_tab.execution_combo.count())
    ]
    payload_values = [
        str(window.research_tab.payload_combo.itemData(index))
        for index in range(window.research_tab.payload_combo.count())
    ]
    assert execution_values == ["api-first-hybrid", "local-only", "api-only"]
    assert payload_values == ["pseudonymous-project-workflow", "person-account-reference"]

    window.research_tab.feature_checkboxes["edit-receipt"].setChecked(True)
    rendered = window.research_tab.selection_display.toPlainText()
    assert '"edit-receipt"' in rendered
    assert '"tee-simulator"' in rendered
    assert '"remote-tee"' in rendered
    assert '"zero-knowledge"' in rendered
