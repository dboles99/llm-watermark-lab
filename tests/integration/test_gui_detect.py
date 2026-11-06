"""Optional smoke tests for the GUI Detect tab."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_detect_tab_can_be_instantiated() -> None:
    from PySide6.QtWidgets import QApplication

    from watermark_lab.gui.main import DetectTab

    _ = QApplication.instance() or QApplication([])
    tab = DetectTab()
    assert tab.method_label.text() == "Method: kgw"
    assert tab.key_id_input.text() == "test-kgw-v1"
    assert tab.threshold_input.text() == "4.0"
