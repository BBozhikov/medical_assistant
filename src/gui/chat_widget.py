"""Reusable, plain-text chat messages and keyboard-aware composer."""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("userMessage" if role == "user" else "assistantMessage")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        heading = QLabel("You" if role == "user" else "Medical assistant")
        heading.setObjectName("messageHeading")
        self.body = QLabel()
        self.body.setObjectName("messageBody")
        self.body.setTextFormat(Qt.PlainText)
        self.body.setText(text)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setMinimumWidth(0)
        layout.addWidget(heading)
        layout.addWidget(self.body)


class ChatWidget(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content.setObjectName("chatContent")
        self.messages_layout = QVBoxLayout(self.content)
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(20)
        self.empty_state = QLabel("New conversation")
        self.empty_state.setObjectName("emptyChat")
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.messages_layout.addWidget(self.empty_state, 1)
        self.messages_layout.addStretch()
        self.setWidget(self.content)
        self.verticalScrollBar().rangeChanged.connect(self._scroll_to_bottom)

    def add_message(self, role: str, text: str) -> None:
        self.empty_state.hide()
        self.messages_layout.insertWidget(
            self.messages_layout.count() - 1, MessageBubble(role, text)
        )
        QTimer.singleShot(0, self._scroll_to_bottom)

    def clear(self) -> None:
        while self.messages_layout.count() > 2:
            item = self.messages_layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
            self.empty_state.show()

    def _scroll_to_bottom(self, *_args) -> None:
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class MessageInput(QPlainTextEdit):
    submitted = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setPlaceholderText("Ask a medical information question...")
        self.setAccessibleName("Medical question")
        self.setToolTip("Enter to send; Shift+Enter for a new line")
        self.setFixedHeight(72)
        self.setTabChangesFocus(True)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.submitted.emit()
            event.accept()
        else:
            super().keyPressEvent(event)