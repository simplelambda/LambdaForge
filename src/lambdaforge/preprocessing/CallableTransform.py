"""Explicit adapter from a Python callable to a preprocessing transform."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.preprocessing.PreprocessingTransform import PreprocessingTransform
from lambdaforge.tasks.TaskContext import TaskContext


class CallableTransform(PreprocessingTransform):
    """Apply a YAML ``ref`` callable to each record value."""

    def __init__(self, function: Callable[[Any], Any]) -> None:
        if not callable(function):
            raise TypeError("CallableTransform.function must be callable.")
        self.function = function

    def transform(
        self,
        record: PreprocessingRecord,
        context: TaskContext,
    ) -> PreprocessingRecord:
        """Replace the value with the callable result and retain record identity."""
        del context
        return record.with_value(self.function(record.value))
