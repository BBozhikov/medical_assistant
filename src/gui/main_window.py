"""Main window; all widgets are accessed exclusively on the GUI thread."""

import json
from dataclasses import asdict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.chat_widget import ChatWidget, MessageInput
from src.gui.settings_dialog import SettingsDialog
from src.gui.sidebar_widget import SidebarWidget
from src.gui.worker_thread import GenerationWorker
from src.rag.rag_pipeline import (
    MEDICAL_DISCLAIMER,
    GenerationSettings,
    Pipeline,
    RAGPipeline,
    RAGResponse,
)


STYLESHEET = """
QMainWindow, QDialog { background: #ffffff; color: #273440; }
QWidget { font-size: 13px; color: #273440; }
QLabel { background: transparent; }
QWidget#appHeader { background: #ffffff; border-bottom: 1px solid #e6eaef; }
QLabel#title { font-size: 20px; font-weight: 600; color: #233b47; }
QLabel#sectionHeading { font-weight: 600; color: #6c7787; margin-top: 8px; }
QLabel#messageHeading { font-size: 12px; font-weight: 600; color: #647684; }
QLabel#messageBody { font-size: 14px; }
QLabel#emptyChat { font-size: 24px; font-weight: 500; color: #8c97a4; }
QLabel#disclaimer { background: #fffdf7; color: #786a45; padding: 12px 24px;
                    border-top: 1px solid #eee9da; font-size: 12px; }
QWidget#sidebar, QScrollArea#sidebarScroll > QWidget { background: #f1f3f6; }
QScrollArea { border: none; background: #ffffff; }
QScrollArea#topicsPanel, QScrollArea#topicsPanel > QWidget, QWidget#topicsContent { background: #f1f3f6; }
QWidget#topicsContent QCheckBox { font-size: 12px; spacing: 8px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #9aa8b3;
                       border-radius: 3px; background: #ffffff; }
QCheckBox::indicator:checked { background: #246b60; border-color: #246b60; }
QCheckBox::indicator:disabled { background: #e5ebf0; border-color: #cfd8de; }
QCheckBox::indicator:checked:disabled { background: #a5b9b4; border-color: #a5b9b4; }
QWidget#chatContent { background: #ffffff; }
QFrame#userMessage { background: #edf3f7; border: none; border-radius: 8px; }
QFrame#assistantMessage { background: #ffffff; border: none; }
QPlainTextEdit, QLineEdit { background: #ffffff; color: #273440;
                         border: 1px solid #dce3ea; border-radius: 6px;
                         padding: 10px; selection-background-color: #c7e3df; }
QPlainTextEdit:focus, QLineEdit:focus { border: 1px solid #48877e; }
QWidget#composer { background: #f7f9fb; border: 1px solid #dce3ea; border-radius: 8px; }
QPlainTextEdit#questionInput { border: none; background: transparent; padding: 4px; }
QPushButton, QToolButton { background: transparent; border: none;
                         border-radius: 6px; padding: 9px 12px; }
QPushButton:hover, QToolButton:hover { background: #e5ebf0; }
QToolButton#topicsToggle { text-align: left; font-weight: 600; }
QPushButton#clearChat { color: #64747d; text-align: left; }
QPushButton#sendButton { background: #246b60; color: white; border-radius: 6px; padding: 0; }
QPushButton#sendButton:hover { background: #1b574e; }
QPushButton#sendButton:disabled { background: #a5b9b4; }
QProgressBar { max-height: 3px; border: none; background: #edf1f4; }
QProgressBar::chunk { background: #398979; }
QSplitter::handle { background: #e6eaef; width: 1px; }
QStatusBar { color: #7a8794; background: #ffffff; font-size: 12px; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #b9c8cc; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0; }
QScrollBar::handle:horizontal { background: #b9c8cc; border-radius: 4px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
"""


class MainWindow(QMainWindow):
    def __init__(
        self,
        pipeline: Pipeline | None = None,
        settings: GenerationSettings | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline if pipeline is not None else RAGPipeline()
        self.settings = settings or GenerationSettings()
        self.history: list[dict] = []
        self._worker: GenerationWorker | None = None
        self._error_box: QMessageBox | None = None
        self.setWindowTitle("Local Medical RAG Assistant")
        self.resize(1100, 780)
        self.setMinimumSize(760, 620)
        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header_panel = QWidget()
        header_panel.setObjectName("appHeader")
        header = QHBoxLayout(header_panel)
        header.setContentsMargins(24, 12, 24, 12)
        title = QLabel("Medical RAG Assistant")
        title.setObjectName("title")
        header.addWidget(title, 1)
        self.export_button = QToolButton()
        self.export_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.export_button.setToolTip("Export conversation to a local JSON file")
        self.export_button.setAccessibleName("Export conversation")
        self.export_button.setFixedSize(40, 40)
        self.export_button.clicked.connect(self._export_history)
        self.settings_button = QToolButton()
        self.settings_button.setIcon(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        )
        self.settings_button.setToolTip("Local model settings")
        self.settings_button.setAccessibleName("Settings")
        self.settings_button.setFixedSize(40, 40)
        self.settings_button.clicked.connect(self._open_settings)
        header.addWidget(self.export_button)
        header.addWidget(self.settings_button)
        layout.addWidget(header_panel)

        splitter = QSplitter(Qt.Horizontal)
        self.sidebar = SidebarWidget()
        self.sidebar.setAttribute(Qt.WA_StyledBackground, True)
        self.sidebar.clear_requested.connect(self.clear_conversation)
        self.sidebar.set_context_visible(self.settings.show_context)
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QScrollArea.NoFrame)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(262)
        self.sidebar_scroll.setWidget(self.sidebar)
        splitter.addWidget(self.sidebar_scroll)
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(24, 16, 24, 20)
        chat_layout.setSpacing(12)
        self.chat = ChatWidget()
        chat_layout.addWidget(self.chat, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setAccessibleName("Generating answer")
        self.progress.hide()
        chat_layout.addWidget(self.progress)
        composer_panel = QWidget()
        composer_panel.setObjectName("composer")
        composer = QHBoxLayout(composer_panel)
        composer.setContentsMargins(12, 10, 10, 10)
        composer.setSpacing(10)
        self.message_input = MessageInput()
        self.message_input.setObjectName("questionInput")
        self.message_input.submitted.connect(self.send_message)
        self.send_button = QPushButton()
        self.send_button.setObjectName("sendButton")
        self.send_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.send_button.setAccessibleName("Send message")
        self.send_button.setToolTip("Send message")
        self.send_button.setFixedSize(40, 40)
        self.send_button.clicked.connect(self.send_message)
        composer.addWidget(self.message_input, 1)
        composer.addWidget(self.send_button, 0, Qt.AlignBottom)
        chat_layout.addWidget(composer_panel)
        splitter.addWidget(chat_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 740])  # wide enough for the longest topic name
        layout.addWidget(splitter, 1)
        disclaimer = QLabel(MEDICAL_DISCLAIMER)
        disclaimer.setTextFormat(Qt.PlainText)
        disclaimer.setObjectName("disclaimer")
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)
        self.setCentralWidget(root)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("Ready | Local processing")

    def send_message(self) -> None:
        question = self.message_input.toPlainText().strip()
        if not question or self._worker is not None:
            return
        topics = self.sidebar.selected_topics()
        if not topics:
            self._show_error("Select a topic", "Choose at least one medical topic.")
            return
        self.chat.add_message("user", question)
        self.history.append({"role": "user", "content": question})
        self.message_input.clear()
        self.sidebar.set_sources((), "")
        self._set_busy(True)
        self.statusBar().showMessage("Retrieving sources and generating answer...")
        self._worker = GenerationWorker(self.pipeline, question, topics, self.settings)
        self._worker.response_ready.connect(self._on_response)
        self._worker.failed.connect(self._on_failure)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_response(self, response: RAGResponse) -> None:
        answer = response.answer
        if MEDICAL_DISCLAIMER not in answer:
            answer += "\n\n" + MEDICAL_DISCLAIMER
        self.chat.add_message("assistant", answer)
        self.sidebar.set_sources(response.sources, response.context)
        self.history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [asdict(source) for source in response.sources],
            }
        )
        count = len(response.sources)
        self.statusBar().showMessage(
            f"Answer complete | {count} source(s)"
            if count
            else "Answer complete | No sources returned; answer is not verified"
        )

    def _on_failure(self, detail: str) -> None:
        answer = "Unable to generate an answer. " + MEDICAL_DISCLAIMER
        self.chat.add_message("assistant", answer)
        self.history.append({"role": "assistant", "content": answer, "sources": []})
        self.statusBar().showMessage("Generation failed | Check the local backend and model")
        self._show_error(
            "Local generation failed",
            "The local backend could not answer. Check the GGUF model path, "
            "local embedding files, and document index, then try again.",
            detail,
        )

    def _on_finished(self) -> None:
        self._worker = None
        self._set_busy(False)
        self.message_input.setFocus()

    def _set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.message_input.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.sidebar.set_busy(busy)

    def clear_conversation(self) -> None:
        if self._worker is not None:
            return
        self.history.clear()
        self.chat.clear()
        self.sidebar.set_sources((), "")
        self.statusBar().showMessage("Conversation cleared")

    def _open_settings(self) -> None:
        if self._worker is not None:
            return
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.generation_settings()
            self.sidebar.set_context_visible(self.settings.show_context)

    def _export_history(self) -> None:
        if self._worker is not None or not self.history:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export conversation (may contain private health information)",
            "conversation.json", "JSON files (*.json)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(self.history, handle, ensure_ascii=False, indent=2)
            except OSError as error:
                self._show_error("Export failed", "Could not write the local file.", str(error))
            else:
                self.statusBar().showMessage("Conversation exported to a local file")

    def _show_error(self, title: str, message: str, detail: str = "") -> None:
        self._error_box = QMessageBox(self)
        self._error_box.setIcon(QMessageBox.Warning)
        self._error_box.setWindowTitle(title)
        self._error_box.setTextFormat(Qt.PlainText)
        self._error_box.setText(message)
        if detail:
            self._error_box.setDetailedText(detail)
        self._error_box.setAttribute(Qt.WA_DeleteOnClose)
        self._error_box.open()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None:
            self.statusBar().showMessage(
                "Generation is still running. Close the window after it finishes."
            )
            event.ignore()
        else:
            event.accept()