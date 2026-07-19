from abc import ABC, abstractmethod
from typing import Optional


class ExperimentTracker(ABC):

    @abstractmethod
    def start(self, project: str, name: str, config: dict):
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict,
        step: Optional[int] = None
    ):
        pass

    @abstractmethod
    def finish(self):
        pass