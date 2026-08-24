"""Run-owned state used to resume compatible Work attempts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lambdaforge.work.managed import ManagedFile, ManagedFileStore, owned_path
from lambdaforge.work.models import atomic_json


class CheckpointCollection:
    """Own safe resumable state for a Run across Attempts."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._store = ManagedFileStore(self._root, scope="checkpoint")

    def path(self, name: str, *, create_parent: bool = True) -> Path:
        """Return a contained checkpoint path for advanced state formats."""
        selected = owned_path(self._root, name)
        if create_parent:
            selected.parent.mkdir(parents=True, exist_ok=True)
        return selected

    def exists(self, name: str) -> bool:
        """Return whether a safe checkpoint path exists."""
        return self.path(name, create_parent=False).exists()

    def save_json(self, name: str, value: Any) -> Path:
        """Atomically store trusted JSON-shaped checkpoint state."""
        return atomic_json(self.path(name), value)

    def load_json(self, name: str) -> Any:
        """Load existing JSON state; corrupt state raises rather than becoming empty."""
        path = self.path(name, create_parent=False)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def file(
        self,
        name: str,
        *,
        build: Callable[[Path], Any] | None = None,
        validate: Callable[[ManagedFile], bool] | None = None,
    ) -> ManagedFile:
        """Return a valid checkpoint file or atomically rebuild it."""
        return self._store.file(name, build=build, validate=validate)

    def restore_reference(
        self,
        name: str,
        *,
        sha256: str,
        size_bytes: int,
    ) -> ManagedFile | None:
        """Restore a logical map dependency only when its checkpoint bytes still match."""
        return self._store.restore(name, sha256=sha256, size_bytes=size_bytes)


__all__ = ["CheckpointCollection"]
