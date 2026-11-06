"""Optional smoke tests for the GUI Create tab."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_create_tab_can_be_instantiated() -> None:
    from PySide6.QtWidgets import QApplication

    from watermark_lab.gui.main import CreateTab

    _ = QApplication.instance() or QApplication([])
    tab = CreateTab()
    assert tab.profile_input.text() == "mock"
    assert tab.method_input.text() == "kgw"
    assert tab.key_id_input.text() == "test-kgw-v1"
    assert tab.disclosure_checkbox.isChecked() is False
