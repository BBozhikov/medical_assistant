"""Session-only settings; no secrets or conversation data written to disk."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
)

from src.rag.rag_pipeline import GenerationSettings


class SettingsDialog(QDialog):
    def __init__(self, settings: GenerationSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Local model settings")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.model_path = QLineEdit(settings.model_path)
        self.model_path.setAccessibleName("Local GGUF model path")
        browse = QToolButton()
        browse.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse.setToolTip("Choose a local GGUF model")
        browse.setAccessibleName("Choose model file")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.model_path, 1)
        path_row.addWidget(browse)
        form.addRow("GGUF model", path_row)

        self.temperature = QSlider(Qt.Horizontal)
        self.temperature.setRange(0, 100)
        self.temperature.setValue(round(settings.temperature * 100))
        self.temperature.setAccessibleName("Temperature")
        self.temperature_value = QLabel()
        temperature_row = QHBoxLayout()
        temperature_row.addWidget(self.temperature, 1)
        temperature_row.addWidget(self.temperature_value)
        form.addRow("Temperature", temperature_row)

        self.max_tokens = QSlider(Qt.Horizontal)
        self.max_tokens.setRange(64, 2048)
        self.max_tokens.setSingleStep(64)
        self.max_tokens.setPageStep(128)
        self.max_tokens.setValue(settings.max_tokens)
        self.max_tokens.setAccessibleName("Maximum output tokens")
        self.tokens_value = QLabel()
        tokens_row = QHBoxLayout()
        tokens_row.addWidget(self.max_tokens, 1)
        tokens_row.addWidget(self.tokens_value)
        form.addRow("Max tokens", tokens_row)

        self.show_context = QCheckBox("Show retrieved context")
        self.show_context.setChecked(settings.show_context)
        form.addRow(self.show_context)
        self.temperature.valueChanged.connect(self._update_values)
        self.max_tokens.valueChanged.connect(self._update_values)
        self._update_values()
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Choose local model", self.model_path.text(), "GGUF models (*.gguf)"
        )
        if path:
            self.model_path.setText(path)

    def _update_values(self, *_args) -> None:
        self.temperature_value.setText(f"{self.temperature.value() / 100:.2f}")
        self.tokens_value.setText(str(self.max_tokens.value()))

    def generation_settings(self) -> GenerationSettings:
        return GenerationSettings(
            model_path=self.model_path.text().strip(),
            temperature=self.temperature.value() / 100,
            max_tokens=self.max_tokens.value(),
            show_context=self.show_context.isChecked(),
        )