"""Immutable preview and persistence result for one migration chain."""

from __future__ import annotations

import copy
import difflib
import json
import math
import os
from collections.abc import Mapping, Sequence, Set
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.migrations.ExperimentConfigMigrationStep import (
    ExperimentConfigMigrationStep,
)
from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)
from lambdaforge.experiments.migrations.MigrationPreviewFormat import (
    MigrationPreviewFormat,
)


class ExperimentConfigMigrationResult(JsonResult):
    """Expose a deterministic preview while keeping source writes explicit."""

    RESULT_VERSION = 1

    def __init__(
        self,
        *,
        source: str | None,
        source_version: ExperimentSchemaVersion,
        target_version: ExperimentSchemaVersion,
        steps: Sequence[ExperimentConfigMigrationStep],
        config: Mapping[str, Any],
        original_yaml: str,
        migrated_yaml: str,
        warnings: Sequence[str] = (),
    ) -> None:
        self.source = source
        self.source_version = source_version
        self.target_version = target_version
        self.steps = tuple(steps)
        self._config = copy.deepcopy(dict(config))
        self.original_yaml = original_yaml
        self.migrated_yaml = migrated_yaml
        self.warnings = tuple(str(warning) for warning in warnings)
        payload = FrozenJsonMapping(
            {
                "migration_result_version": self.RESULT_VERSION,
                "source": self.source,
                "source_version": self.source_version.to_json_value(),
                "target_version": self.target_version.to_json_value(),
                "changed": self.changed,
                "steps": [step.to_dict() for step in self.steps],
                "warnings": list(self.warnings),
                "diff": self.diff(),
                "config": self._json_projection(self._config),
            }
        )
        self._freeze_mapping(dict(payload))

    @property
    def config(self) -> dict[str, Any]:
        """Return an independent semantic configuration for framework use."""
        return copy.deepcopy(self._config)

    @property
    def changed(self) -> bool:
        """Return whether at least one semantic migration step was applied."""
        return bool(self.steps)

    def diff(self) -> str:
        """Render a deterministic unified diff without touching the source."""
        source_label = self.source or "configuration.yaml"
        target_label = f"{source_label} [migrated to {self.target_version}]"
        lines = difflib.unified_diff(
            self.original_yaml.splitlines(),
            self.migrated_yaml.splitlines(),
            fromfile=source_label,
            tofile=target_label,
            lineterm="",
        )
        value = "\n".join(lines)
        return f"{value}\n" if value else ""

    def render(self, output_format: MigrationPreviewFormat | str) -> str:
        """Render diff, complete YAML or the stable JSON envelope."""
        resolved = MigrationPreviewFormat(output_format)
        if resolved is MigrationPreviewFormat.DIFF:
            return self.diff()
        if resolved is MigrationPreviewFormat.YAML:
            return self.migrated_yaml
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def write_yaml(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically write migrated YAML to a distinct explicit destination."""
        path = Path(path)
        if self.source is not None:
            source_path = Path(self.source)
            if os.path.normcase(str(source_path.resolve())) == os.path.normcase(
                str(path.resolve())
            ):
                raise ValueError("Migration output must not overwrite the source file.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Migration output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                handle.write(self.migrated_yaml)
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                try:
                    os.link(temporary, path)
                except FileExistsError as error:
                    raise FileExistsError(f"Migration output already exists: {path}") from error
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible migration envelope."""
        return copy.deepcopy(dict(self))

    @classmethod
    def _json_projection(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return str(value)
        if isinstance(value, Enum):
            return cls._json_projection(value.value)
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, Mapping):
            return {str(key): cls._json_projection(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return [cls._json_projection(item) for item in value]
        if isinstance(value, Set):
            projected = [cls._json_projection(item) for item in value]
            return sorted(projected, key=lambda item: json.dumps(item, sort_keys=True))
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            return isoformat()
        value_type = type(value)
        return f"<{value_type.__module__}.{value_type.__qualname__}>"
