from __future__ import annotations

import json
import sys

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from watermark_lab.core.contracts import DetectionConfig
from watermark_lab.run_engine.create import (
    CreationService,
    GovernanceError,
    SelfVerificationError,
)
from watermark_lab.run_engine.detect import DetectionService
from watermark_lab.run_engine.intake import compute_sha256, normalize_text, read_source
from watermark_lab.services.product_workspace import (
    ExecutionTarget,
    HistoryMode,
    PayloadIdentityMode,
    ProductWorkspaceService,
)


class IntakeTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.path_label = QLabel("No file or folder selected")
        self.text = QTextEdit()
        self.text.setPlaceholderText("Paste text here, or browse for a file or folder...")
        file_button = QPushButton("Browse file")
        folder_button = QPushButton("Browse folder")
        file_button.clicked.connect(self._browse_file)
        folder_button.clicked.connect(self._browse_folder)
        buttons = QHBoxLayout()
        buttons.addWidget(file_button)
        buttons.addWidget(folder_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addLayout(buttons)
        layout.addWidget(self.text)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select text or document",
            "",
            "Supported (*.txt *.md *.docx *.pdf);;All files (*.*)",
        )
        if path:
            self.path_label.setText(path)

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select intake folder")
        if path:
            self.path_label.setText(path)


class DetectTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Paste text here or load it from the Intake tab...")

        self.method_label = QLabel("Method: kgw")
        self.key_id_input = QLineEdit("test-kgw-v1")
        self.threshold_input = QLineEdit("4.0")
        self.scientific_checkbox = QCheckBox(
            "Scientific mode (pinned tokenizer + author-code RNG; returns inconclusive without frozen calibration)"
        )
        self.run_button = QPushButton("Run detection")
        self.run_button.clicked.connect(self._run_detection)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        form = QVBoxLayout()
        form.addWidget(self.method_label)
        form.addWidget(QLabel("Key ID:"))
        form.addWidget(self.key_id_input)
        form.addWidget(QLabel("Threshold:"))
        form.addWidget(self.threshold_input)
        form.addWidget(self.scientific_checkbox)
        form.addWidget(self.run_button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Input text:"))
        layout.addWidget(self.input_text)
        layout.addLayout(form)
        layout.addWidget(QLabel("Result:"))
        layout.addWidget(self.result_display)

    def _run_detection(self) -> None:
        raw_input = self.input_text.toPlainText()
        if not raw_input.strip():
            QMessageBox.warning(self, "Missing input", "Please paste or load text before running detection.")
            return

        try:
            threshold = float(self.threshold_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid threshold", "Threshold must be a number.")
            return

        text, media_type, source_path = read_source(raw_input)
        normalized = normalize_text(text)

        if self.scientific_checkbox.isChecked():
            try:
                from watermark_lab.cli import _run_scientific_detect

                payload = _run_scientific_detect(
                    text=normalized,
                    locale="en",
                    tokenizer_id="gpt2",
                    tokenizer_assets="",
                    gamma=0.25,
                    threshold=threshold,
                    ignore_repeated_ngrams=False,
                    input_media_type=media_type,
                    input_source_path=source_path,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Scientific detection failed", str(exc))
                return
            self.result_display.setPlainText(json.dumps(payload, indent=2))
            return

        config = DetectionConfig(
            tokenizer_id="gpt2",
            key_id=self.key_id_input.text(),
            threshold=threshold,
            options={
                "gamma": 0.25,
                "context_width": 1,
                "input_media_type": media_type,
                "input_source_path": source_path,
                "input_sha256": compute_sha256(text),
                "normalized_sha256": compute_sha256(normalized),
            },
        )

        try:
            result = DetectionService().detect(normalized, config)
        except Exception as exc:
            QMessageBox.critical(self, "Detection failed", str(exc))
            return

        payload = {
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
        self.result_display.setPlainText(json.dumps(payload, indent=2))


class CreateTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter a prompt for generation...")

        self.profile_input = QLineEdit("mock")
        self.method_input = QLineEdit("kgw")
        self.key_id_input = QLineEdit("test-kgw-v1")
        self.disclosure_checkbox = QCheckBox("Disclosure recorded")

        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self._run_generation)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        form = QVBoxLayout()
        form.addWidget(QLabel("Profile:"))
        form.addWidget(self.profile_input)
        form.addWidget(QLabel("Method:"))
        form.addWidget(self.method_input)
        form.addWidget(QLabel("Key ID:"))
        form.addWidget(self.key_id_input)
        form.addWidget(self.disclosure_checkbox)
        form.addWidget(self.generate_button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Prompt:"))
        layout.addWidget(self.prompt_input)
        layout.addLayout(form)
        layout.addWidget(QLabel("Result:"))
        layout.addWidget(self.result_display)

    def _run_generation(self) -> None:
        prompt = self.prompt_input.toPlainText()
        if not prompt.strip():
            QMessageBox.warning(self, "Missing prompt", "Please enter a prompt before generating.")
            return

        profile_id = self.profile_input.text() or "mock"
        method_id = self.method_input.text() or "kgw"
        key_id = self.key_id_input.text() or "test-kgw-v1"
        disclosure = self.disclosure_checkbox.isChecked()

        config = {
            "disclosure_recorded": disclosure,
            "max_new_tokens": 80,
            "gamma": 0.25,
            "delta": 2.0,
            "context_width": 1,
        }

        try:
            result = CreationService().create(prompt, profile_id, method_id, key_id, config)
        except GovernanceError as exc:
            QMessageBox.warning(self, "Governance check failed", str(exc))
            return
        except SelfVerificationError as exc:
            QMessageBox.critical(self, "Self-verification failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Generation failed", str(exc))
            return

        payload = {
            "run_id": result["run_id"],
            "status": result["status"],
            "text": result["generation"]["text"],
            "prompt_sha256": result["input_sha256"],
            "response_sha256": result["generation"]["response_sha256"],
            "model_profile": result["generation"]["model_profile"],
            "method_id": result["method_id"],
            "key_id": result["key_id"],
            "self_verification": result["self_verification"],
        }
        self.result_display.setPlainText(json.dumps(payload, indent=2))


class RunsTab(QWidget):
    """Read-only run browser backed by the shared product service."""

    def __init__(self, service: ProductWorkspaceService) -> None:
        super().__init__()
        self.service = service
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Run ID", "Status", "Method", "Updated", "Evidence source"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.empty_label = QLabel()

        layout = QVBoxLayout(self)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self) -> None:
        runs = self.service.list_runs(limit=200)
        self.table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            values = (run.run_id, run.status, run.method_id, run.updated_at, run.source)
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.empty_label.setText("No runs found." if not runs else f"{len(runs)} run(s)")


class ArtifactHistoryTab(QWidget):
    """Read-only artifact browser with an explicit correction policy."""

    def __init__(self, service: ProductWorkspaceService) -> None:
        super().__init__()
        self.service = service
        self.mode_label = QLabel()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Artifact", "Type", "Status", "Updated", "Path"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.empty_label = QLabel()

        layout = QVBoxLayout(self)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self) -> None:
        artifacts = self.service.list_artifacts(limit=500)
        self.mode_label.setText(f"History mode: {self.service.history_mode.value}")
        self.table.setRowCount(len(artifacts))
        for row, artifact in enumerate(artifacts):
            values = (
                artifact.artifact_id,
                artifact.artifact_type,
                artifact.status,
                artifact.updated_at,
                artifact.path,
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.empty_label.setText(
            "No artifacts found." if not artifacts else f"{len(artifacts)} artifact(s)"
        )


class ResearchTab(QWidget):
    """Selectable frontier modules with explicit maturity and governance choices."""

    def __init__(self, service: ProductWorkspaceService) -> None:
        super().__init__()
        self.service = service
        self.feature_checkboxes: dict[str, QCheckBox] = {}
        self.execution_combo = QComboBox()
        self.execution_combo.addItem("API-first hybrid (recommended)", ExecutionTarget.API_FIRST_HYBRID)
        self.execution_combo.addItem("Local only", ExecutionTarget.LOCAL_ONLY)
        self.execution_combo.addItem("API only", ExecutionTarget.API_ONLY)
        self.payload_combo = QComboBox()
        self.payload_combo.addItem(
            "Pseudonymous project/workflow reference (default)",
            PayloadIdentityMode.PSEUDONYMOUS,
        )
        self.payload_combo.addItem(
            "Authorized person/account reference",
            PayloadIdentityMode.PERSON_ACCOUNT,
        )
        self.selection_display = QTextEdit()
        self.selection_display.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Select one or more frontier modules. These are planned research features; "
                "selection does not imply that their scientific gates have passed."
            )
        )
        for feature in self.service.list_research_features():
            checkbox = QCheckBox(f"{feature.label} [{feature.maturity}]")
            checkbox.setToolTip(f"{feature.description}\nDependency: {feature.dependency}")
            checkbox.stateChanged.connect(self._render_selection)
            self.feature_checkboxes[feature.feature_id.value] = checkbox
            layout.addWidget(checkbox)

        layout.addWidget(QLabel("Execution target:"))
        layout.addWidget(self.execution_combo)
        layout.addWidget(QLabel("Payload identity mode:"))
        layout.addWidget(self.payload_combo)
        layout.addWidget(
            QLabel(
                "Proof sequence: local TEE simulator → remote attested TEE → zero knowledge. "
                "Raw personal identifiers are never embedded in payload bits."
            )
        )
        layout.addWidget(QLabel("Selection and policy:"))
        layout.addWidget(self.selection_display)
        self.execution_combo.currentIndexChanged.connect(self._render_selection)
        self.payload_combo.currentIndexChanged.connect(self._render_selection)
        self._render_selection()

    def _render_selection(self) -> None:
        selected = [
            feature_id
            for feature_id, checkbox in self.feature_checkboxes.items()
            if checkbox.isChecked()
        ]
        execution = self.execution_combo.currentData()
        payload_identity = self.payload_combo.currentData()
        payload: dict[str, object] = {"policy": self.service.research_policy()}
        if selected:
            payload["selection"] = self.service.research_selection(
                selected,
                execution_target=execution,
                payload_identity_mode=payload_identity,
            )
        else:
            payload["selection"] = {
                "selected_features": [],
                "status": "select-at-least-one-feature",
            }
        self.selection_display.setPlainText(json.dumps(payload, indent=2))


class SettingsTab(QWidget):
    """Locale and artifact-history controls for the initial product shell."""

    def __init__(self, service: ProductWorkspaceService, artifact_tab: ArtifactHistoryTab) -> None:
        super().__init__()
        self.service = service
        self.artifact_tab = artifact_tab
        self.locale_combo = QComboBox()
        self.locale_combo.addItem("English", "en")
        self.locale_combo.addItem("繁體中文（台灣）", "zh-Hant-TW")
        self.history_mode_combo = QComboBox()
        self.history_mode_combo.addItem(
            "Supersede and preserve", HistoryMode.SUPERSEDE_AND_PRESERVE.value
        )
        self.history_mode_combo.addItem("Amend in place", HistoryMode.AMEND_IN_PLACE.value)
        self.policy_display = QTextEdit()
        self.policy_display.setReadOnly(True)

        self.locale_combo.currentIndexChanged.connect(self._apply_settings)
        self.history_mode_combo.currentIndexChanged.connect(self._apply_settings)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Locale:"))
        layout.addWidget(self.locale_combo)
        layout.addWidget(QLabel("Artifact history mode:"))
        layout.addWidget(self.history_mode_combo)
        layout.addWidget(QLabel("Policy:"))
        layout.addWidget(self.policy_display)
        self._select_current_values()
        self._render_policy()

    def _select_current_values(self) -> None:
        locale_index = self.locale_combo.findData(self.service.locale)
        if locale_index >= 0:
            self.locale_combo.setCurrentIndex(locale_index)
        mode_index = self.history_mode_combo.findData(self.service.history_mode.value)
        if mode_index >= 0:
            self.history_mode_combo.setCurrentIndex(mode_index)

    def _apply_settings(self) -> None:
        locale = self.locale_combo.currentData()
        mode = self.history_mode_combo.currentData()
        if locale:
            self.service.set_locale(str(locale))
        if mode:
            self.service.set_history_mode(str(mode))
        self._render_policy()
        self.artifact_tab.refresh()

    def _render_policy(self) -> None:
        self.policy_display.setPlainText(json.dumps(self.service.policy(), indent=2))


class MainWindow(QMainWindow):
    def __init__(self, workspace_service: ProductWorkspaceService | None = None) -> None:
        super().__init__()
        self.workspace_service = workspace_service or ProductWorkspaceService()
        self.setWindowTitle("LLM Watermark Lab")
        self.resize(1180, 760)
        self.tabs = QTabWidget()
        self.tabs.addTab(IntakeTab(), "Intake")
        self.tabs.addTab(DetectTab(), "Detect")
        self.tabs.addTab(CreateTab(), "Create")
        for label in ("Analyze", "Stress Test", "Compare"):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.addWidget(QLabel(f"{label} workspace will be implemented in its project phase."))
            self.tabs.addTab(panel, label)
        self.research_tab = ResearchTab(self.workspace_service)
        self.tabs.addTab(self.research_tab, "Research")
        self.runs_tab = RunsTab(self.workspace_service)
        self.artifact_history_tab = ArtifactHistoryTab(self.workspace_service)
        self.settings_tab = SettingsTab(self.workspace_service, self.artifact_history_tab)
        self.tabs.addTab(self.runs_tab, "Runs")
        self.tabs.addTab(self.artifact_history_tab, "Artifact History")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(self.tabs)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
