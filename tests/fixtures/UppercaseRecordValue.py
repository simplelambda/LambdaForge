"""Test-only preprocessing transform with explicit record/context contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lambdaforge.preprocessing import PreprocessingRecord, PreprocessingTransform
from lambdaforge.tasks import TaskContext


class UppercaseRecordValue(PreprocessingTransform):
    """Uppercase the configured field of a mapping-shaped record value."""

    def __init__(self, field: str = "text") -> None:
        self.field = field

    def transform(
        self,
        record: PreprocessingRecord,
        context: TaskContext,
    ) -> PreprocessingRecord:
        """Return a copied mapping with one uppercased field."""
        del context
        if not isinstance(record.value, Mapping):
            raise TypeError("UppercaseRecordValue expects a mapping value.")
        value: dict[str, Any] = dict(record.value)
        value[self.field] = str(value[self.field]).upper()
        return record.with_value(value)
