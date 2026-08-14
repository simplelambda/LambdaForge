"""Crash-safe one-JSON-file-per-record preprocessing sink."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.preprocessing.PreprocessingSink import PreprocessingSink
from lambdaforge.tasks.artifacts import ArtifactDeclaration, ArtifactType
from lambdaforge.tasks.TaskContext import TaskContext


class JsonDirectorySink(PreprocessingSink):
    """Atomically store each record under a filename derived from its stable key."""

    def __init__(
        self,
        output_dir: str | Path = "processed",
        output_name: str | None = None,
    ) -> None:
        output = Path(output_dir)
        if output.is_absolute() or not output.parts or ".." in output.parts:
            raise ValueError("JsonDirectorySink.output_dir must be run-relative.")
        self.output_dir = output
        self.output_name = output_name

    def write(self, record: PreprocessingRecord, context: TaskContext) -> None:
        """Atomically publish one JSON-compatible record envelope."""
        path = self._record_path(record.key, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"key": record.key, "value": record.value, "metadata": record.metadata},
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def is_complete(self, key: str, context: TaskContext) -> bool:
        """Verify that the deterministic output remains readable and owns the expected key."""
        path = self._record_path(key, context)
        if not path.is_file() or path.is_symlink():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return (
            isinstance(value, Mapping)
            and value.get("key") == key
            and "value" in value
            and isinstance(value.get("metadata"), Mapping)
        )

    def finalize(self, context: TaskContext) -> tuple[ArtifactDeclaration, ...]:
        """Declare the complete JSON directory as one dataset-like artifact."""
        self._output_root(context).mkdir(parents=True, exist_ok=True)
        return (
            ArtifactDeclaration(
                path=self._output_root(context).relative_to(context.run_dir),
                kind=ArtifactType.DIRECTORY,
                media_type="application/json",
            ),
        )

    def _record_path(self, key: str, context: TaskContext) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._output_root(context) / f"{digest}.json"

    def _output_root(self, context: TaskContext) -> Path:
        return (
            context.output(self.output_name)
            if self.output_name
            else context.output_path(self.output_dir)
        )
