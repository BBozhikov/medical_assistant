"""Blocking backend calls isolated from the Qt GUI thread."""

from PyQt5.QtCore import QThread, pyqtSignal

from src.rag.rag_pipeline import GenerationSettings, Pipeline, RAGResponse


class GenerationWorker(QThread):
    response_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        pipeline: Pipeline,
        question: str,
        topics: tuple[str, ...],
        settings: GenerationSettings,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.question = question
        self.topics = topics
        self.settings = settings

    def run(self) -> None:
        try:
            response = self.pipeline.answer(
                self.question, topics=self.topics, settings=self.settings
            )
            if not isinstance(response, RAGResponse):
                raise TypeError("The backend must return a RAGResponse.")
            if not response.answer.strip():
                raise ValueError("The backend returned an empty answer.")
            self.response_ready.emit(response)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")