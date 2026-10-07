"""Run-owned state used to resume compatible Work attempts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
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

    def pin(self, name: str, *, reason: str = "researcher-retained") -> Path:
        """Keep one owned checkpoint beyond terminal retention; outputs remain preferable."""
        selected = owned_path(self._root, name, must_exist=True)
        if selected == self._root:
            raise ValueError("Pin a named checkpoint, not the collection root.")
        self._update_policy(
            lambda policy: policy.setdefault("pins", {}).__setitem__(
                selected.relative_to(self._root).as_posix(), str(reason)
            )
        )
        return selected

    def unpin(self, name: str) -> None:
        """Remove only an explicit retention pin; this never deletes bytes immediately."""
        selected = owned_path(self._root, name)
        self._update_policy(
            lambda policy: policy.setdefault("pins", {}).pop(
                selected.relative_to(self._root).as_posix(), None
            )
        )

    def _update_policy(self, update: Callable[[dict[str, Any]], Any]) -> None:
        with CrossProcessFileLock(
            self._root / ".storage-policy.lock",
            shared=False,
            timeout_seconds=5,
            poll_interval_seconds=0.05,
        ):
            policy = self._storage_policy()
            for field in ("pins", "publications"):
                if not isinstance(policy.get(field, {}), dict):
                    raise ValueError("Invalid checkpoint retention evidence; refusing mutation.")
            update(policy)
            atomic_json(self._root / ".storage-policy.json", policy)

    def _storage_policy(self) -> dict[str, Any]:
        path = self._root / ".storage-policy.json"
        if path.is_symlink():
            raise ValueError("Checkpoint storage policy cannot be symbolic.")
        value = json.loads(path.read_text()) if path.exists() else {"storage_policy_version": 1}
        if not isinstance(value, dict) or value.get("storage_policy_version") != 1:
            raise ValueError("Invalid checkpoint storage policy; retention fails closed.")
        return value

    def _publication_committed(self, names: list[str], witness: dict[str, Any]) -> None:
        """Record explicit redundant state only after verified durable publication."""
        from lambdaforge.work.managed import fingerprint

        records = {}
        for name in names:
            source = owned_path(self._root, name, must_exist=True)
            if source == self._root:
                raise ValueError("Release named checkpoints, not the collection root.")
            digest, size = fingerprint(source)
            records[source.relative_to(self._root).as_posix()] = {
                "sha256": digest,
                "size_bytes": size,
                "durable": witness,
            }
        self._update_policy(lambda policy: policy.setdefault("publications", {}).update(records))


__all__ = ["CheckpointCollection"]
