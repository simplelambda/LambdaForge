"""Content-addressed local input declared by a task configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.tasks.TaskArtifact import TaskArtifact


class TaskInput(JsonResult):
    """Resolve and fingerprint a YAML-relative file or directory input."""

    def __init__(
        self,
        *,
        name: str,
        path: str,
        resolved_path: str | Path,
        sha256: str,
        size_bytes: int,
    ) -> None:
        if not name.strip():
            raise ValueError("Task input names must be non-empty.")
        self.name = str(name)
        self.path = str(path)
        self.resolved_path = str(resolved_path)
        self.sha256 = str(sha256)
        self.size_bytes = int(size_bytes)
        self._freeze_mapping(self.to_dict())

    @classmethod
    def materialize(
        cls,
        value: Mapping[str, Any],
        source_dir: str | Path,
        index: int,
    ) -> TaskInput:
        """Resolve, validate and hash one configured local input."""
        configured = Path(str(value["path"]))
        unresolved = configured if configured.is_absolute() else Path(source_dir) / configured
        if unresolved.is_symlink():
            raise ValueError(f"Task inputs cannot be symbolic links: {unresolved}")
        resolved = unresolved.resolve(strict=False)
        if not resolved.exists():
            raise FileNotFoundError(f"Task input does not exist: {resolved}")
        digest, size = TaskArtifact.fingerprint_path(resolved)
        return cls(
            name=str(value.get("name", f"input_{index}")),
            path=configured.as_posix(),
            resolved_path=resolved,
            sha256=digest,
            size_bytes=size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable input provenance mapping."""
        return {
            "name": self.name,
            "path": self.path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
