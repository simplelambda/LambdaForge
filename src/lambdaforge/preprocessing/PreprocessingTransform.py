"""Base transformation contract for one preprocessing record."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.tasks.TaskContext import TaskContext


class PreprocessingTransform(ABC):
    """Transform one record while preserving its stable key."""

    @abstractmethod
    def transform(
        self,
        record: PreprocessingRecord,
        context: TaskContext,
    ) -> PreprocessingRecord:
        """Return the transformed record."""
