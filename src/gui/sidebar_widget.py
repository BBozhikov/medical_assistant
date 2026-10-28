"""Topic filters and the last answer's local source evidence."""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.rag.rag_pipeline import MEDICAL_TOPICS, SourceCitation


class SidebarWidget(QWidget):
    clear_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setMinimumWidth(230)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.topics_toggle = QToolButton()
        self.topics_toggle.setObjectName("topicsToggle")
        self.topics_toggle.setCheckable(True)
        self.topics_toggle.setArrowType(Qt.RightArrow)
        self.topics_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.topics_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.topics_toggle.setMinimumHeight(38)
        self.topics_toggle.setToolTip("Choose medical topics for retrieval")
        self.topics_toggle.setAccessibleName("Expand medical topic filters")
        layout.addWidget(self.topics_toggle)
        # The long topic list scrolls inside a fixed-height panel
        self.topics_panel = QScrollArea()
        self.topics_panel.setObjectName("topicsPanel")  # styled in main_window.STYLESHEET
        self.topics_panel.setWidgetResizable(True)
        self.topics_panel.setFrameShape(QFrame.NoFrame)
        self.topics_panel.setMaximumHeight(240)
        topics_content = QWidget()
        topics_content.setObjectName("topicsContent")
        topics_content.setAttribute(Qt.WA_StyledBackground, True)
        self.topics_panel.setWidget(topics_content)
        topics_layout = QVBoxLayout(topics_content)
        topics_layout.setContentsMargins(4, 0, 0, 4)
        topics_layout.setSpacing(6)
        bulk_row = QHBoxLayout()
        self.select_all_button = QPushButton("Select all")
        self.select_none_button = QPushButton("Clear")
        self.select_all_button.clicked.connect(lambda: self._set_all_topics(True))
        self.select_none_button.clicked.connect(lambda: self._set_all_topics(False))
        bulk_row.addWidget(self.select_all_button)
        bulk_row.addWidget(self.select_none_button)
        topics_layout.addLayout(bulk_row)
        self.topic_checks = []
        for topic in MEDICAL_TOPICS:
            checkbox = QCheckBox(topic)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._update_topic_summary)
            self.topic_checks.append(checkbox)
            topics_layout.addWidget(checkbox)
        self.topics_panel.hide()
        self.topics_toggle.toggled.connect(self._toggle_topics)
        self._update_topic_summary()
        layout.addWidget(self.topics_panel)

        self.sources_heading = QLabel("Sources")
        self.sources_heading.setObjectName("sectionHeading")
        layout.addWidget(self.sources_heading)
        self.sources_view = QPlainTextEdit()
        self.sources_view.setReadOnly(True)
        self.sources_view.setAccessibleName("Retrieved source citations")
        self.sources_view.setMinimumHeight(100)
        self.sources_view.setMaximumHeight(220)
        layout.addWidget(self.sources_view)
        self.sources_heading.hide()
        self.sources_view.hide()

        self.context_heading = QLabel("Retrieved context")
        self.context_heading.setObjectName("sectionHeading")
        self.context_view = QPlainTextEdit()
        self.context_view.setReadOnly(True)
        self.context_view.setAccessibleName("Retrieved context")
        self.context_view.setMinimumHeight(60)
        self.context_view.setMaximumHeight(180)
        layout.addWidget(self.context_heading)
        layout.addWidget(self.context_view)
        self.set_context_visible(False)

        layout.addStretch(1)
        self.clear_button = QPushButton("Clear chat")
        self.clear_button.setObjectName("clearChat")
        self.clear_button.setToolTip("Clear conversation and source citations")
        self.clear_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.clear_button.clicked.connect(self.clear_requested.emit)
        layout.addWidget(self.clear_button)

    def _toggle_topics(self, expanded: bool) -> None:
        self.topics_panel.setVisible(expanded)
        self.topics_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.topics_toggle.setAccessibleName(
            "Collapse medical topic filters" if expanded else "Expand medical topic filters"
        )

    def _set_all_topics(self, checked: bool) -> None:
        for checkbox in self.topic_checks:
            checkbox.setChecked(checked)

    def _update_topic_summary(self, *_args) -> None:
        count = len(self.selected_topics())
        summary = "all" if count == len(MEDICAL_TOPICS) else str(count)
        self.topics_toggle.setText(f"Medical topics ({summary})")

    def selected_topics(self) -> tuple[str, ...]:
        return tuple(check.text() for check in self.topic_checks if check.isChecked())

    def set_sources(self, sources: tuple[SourceCitation, ...], context: str) -> None:
        self.sources_heading.setText(f"Sources ({len(sources)})")
        self.sources_heading.setVisible(bool(sources))
        self.sources_view.setVisible(bool(sources))
        self.sources_view.setPlainText(
            "\n\n".join(
                f"[{index}] {source.title}\n{source.location}"
                + (f"\n{source.excerpt}" if source.excerpt else "")
                for index, source in enumerate(sources, 1)
            )
        )
        self.context_view.setPlainText(context)

    def set_context_visible(self, visible: bool) -> None:
        self.context_heading.setVisible(visible)
        self.context_view.setVisible(visible)

    def set_busy(self, busy: bool) -> None:
        self.topics_toggle.setEnabled(not busy)
        for checkbox in self.topic_checks:
            checkbox.setEnabled(not busy)
        self.select_all_button.setEnabled(not busy)
        self.select_none_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)