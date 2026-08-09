"""Base record-source contract for preprocessing pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.tasks.TaskContext import TaskContext


class PreprocessingSource(ABC):
    """Produce deterministic-key records without owning pipeline execution."""

    @abstractmethod
    def records(self, context: TaskContext) -> Iterable[PreprocessingRecord]:
        """Yield source records in a deterministic order when possible."""
