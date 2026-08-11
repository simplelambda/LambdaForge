"""JSON Lines record source for structured preprocessing inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.preprocessing.PreprocessingSource import PreprocessingSource
from lambdaforge.tasks.TaskContext import TaskContext


class JsonLinesSource(PreprocessingSource):
    """Read one JSON value per non-empty line with optional mapping key selection."""

    def __init__(
        self,
        path: str | Path | None = None,
        key_field: str | None = None,
        input_name: str | None = None,
    ) -> None:
        if path is None and input_name is None:
            raise ValueError("JsonLinesSource requires path or input_name.")
        if path is not None and input_name is not None:
            raise ValueError("JsonLinesSource path and input_name are mutually exclusive.")
        if key_field is not None and not key_field.strip():
            raise ValueError("JsonLinesSource.key_field cannot be empty.")
        self.path = Path(path) if path is not None else None
        self.input_name = input_name
        self.key_field = key_field

    def records(self, context: TaskContext) -> Iterable[PreprocessingRecord]:
        """Yield parsed JSON values in source order with stable keys."""
        path = (
            context.input(self.input_name)
            if self.input_name is not None
            else context.declared_input_path(self.path or "")
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value: Any = json.loads(line)
                if self.key_field is None:
                    key = f"line-{line_number:012d}"
                elif isinstance(value, Mapping) and self.key_field in value:
                    key = str(value[self.key_field])
                else:
                    raise KeyError(f"JSONL line {line_number} has no key field {self.key_field!r}.")
                yield PreprocessingRecord(
                    key=key,
                    value=value,
                    metadata={"source_line": line_number},
                )
