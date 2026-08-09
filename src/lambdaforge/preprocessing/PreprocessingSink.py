"""Base output-sink contract for preprocessing pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.tasks.ArtifactDeclaration import ArtifactDeclaration
from lambdaforge.tasks.TaskContext import TaskContext


class PreprocessingSink(ABC):
    """Persist processed records and declare final run-relative artifacts."""

    @abstractmethod
    def write(self, record: PreprocessingRecord, context: TaskContext) -> None:
        """Persist one complete record or raise without marking it complete."""

    def is_complete(self, key: str, context: TaskContext) -> bool:
        """Return whether a manifest-complete key still has a valid sink output."""
        del key, context
        return True

    def finalize(self, context: TaskContext) -> Sequence[ArtifactDeclaration]:
        """Flush the sink and declare its aggregate artifacts."""
        del context
        return ()
