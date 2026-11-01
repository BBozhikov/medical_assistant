"""Launch the local PyQt5 desktop interface."""

import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Medical RAG Assistant")
    font = QFont()
    font.setFamilies(["Avenir Next", "Segoe UI", "sans-serif"])
    font.setPointSize(10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())