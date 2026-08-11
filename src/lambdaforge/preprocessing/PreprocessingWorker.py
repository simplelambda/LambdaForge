"""Spawn-safe record transformation worker."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.tasks.TaskContext import TaskContext


class PreprocessingWorker:
    """Apply CPU transforms in a child while leaving sink ownership in the parent."""

    @staticmethod
    def transform(
        transforms: Sequence[Any],
        record: PreprocessingRecord,
        context: TaskContext,
    ) -> PreprocessingRecord:
        """Return one transformed record through a pickle-safe process boundary."""
        transformed = record
        for transform in transforms:
            transformed = transform.transform(transformed, context)
            if not isinstance(transformed, PreprocessingRecord):
                raise TypeError("Preprocessing transforms must return PreprocessingRecord objects.")
            if transformed.key != record.key:
                raise ValueError("Preprocessing transforms cannot change stable record keys.")
        return transformed
