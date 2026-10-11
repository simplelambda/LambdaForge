"""Run-owned state used to resume compatible Work attempts."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.checkpoint_policy import CheckpointPolicy
from lambdaforge.work.checkpoint_policy import checkpoint_payloads as checkpoint_payloads
from lambdaforge.work.managed import ManagedFile, ManagedFileStore, owned_path
from lambdaforge.work.models import atomic_json


class CheckpointCollection:
    """Own safe resumable state for a Run across Attempts."""

    def __init__(self, root: Path, *, owner: Mapping[str, Any] | None = None) -> None:
        self._root = root.resolve()
        self._store = ManagedFileStore(self._root, scope="checkpoint")
        self._owner = dict(owner or {})

    def path(self, name: str, *, create_parent: bool = True) -> Path:
        """Return a contained checkpoint path for advanced state formats."""
        selected = owned_path(self._root, name)
        if create_parent:
            selected.parent.mkdir(parents=True, exist_ok=True)
        return selected

    def exists(self, name: str) -> bool:
        """Return whether a safe checkpoint path exists."""
        return self.path(name, create_parent=False).exists()

    def save_json(self, name: str, value: Any, *, step: int | None = None) -> Path:
        """Atomically store trusted JSON-shaped checkpoint state."""
        content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        self._registered_record(name)
        managed = self._store.put_bytes(name, content, metadata=self._annotations(step))
        self._committed(managed)
        return managed.path

    def load_json(self, name: str) -> Any:
        """Load existing JSON state; corrupt state raises rather than becoming empty."""
        path = self.path(name, create_parent=False)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        record = self._registered_record(name)
        if record is not None and self._store.restore(name) is None:
            raise ValueError(f"Managed checkpoint integrity failed: {name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def file(
        self,
        name: str,
        *,
        build: Callable[[Path], Any] | None = None,
        validate: Callable[[ManagedFile], bool] | None = None,
        step: int | None = None,
    ) -> ManagedFile:
        """Return a valid checkpoint file or atomically rebuild it."""
        self._registered_record(name)
        managed = self._store.file(
            name, build=build, validate=validate, metadata=self._annotations(step)
        )
        self._committed(managed)
        return managed

    def configure(
        self,
        *,
        interval_steps: int | None = None,
        interval_seconds: float | None = None,
        retention_seconds: float | None = None,
        storage_durability: str = "unspecified",
    ) -> None:
        """Declare advisory cadence/terminal retention; this never saves or deletes state.

        The Work must call should_save at a safe boundary and serialize its own algorithm.
        Failed/interrupted evidence and explicit pins remain protected regardless of cadence.
        """
        policy = CheckpointPolicy(
            interval_steps,
            interval_seconds,
            retention_seconds,
            storage_durability,
        )
        self._update_policy(lambda value: value.__setitem__("checkpoint_policy", policy.to_dict()))

    def should_save(self, *, step: int | None = None, now: float | None = None) -> bool:
        """Check the declared cadence against committed records, never a poll counter."""
        self._validate_step(step)
        stored = self._storage_policy()
        policy = CheckpointPolicy(**stored.get("checkpoint_policy", {}))
        if policy.interval_steps is None and policy.interval_seconds is None:
            return False
        last = stored.get("last_checkpoint")
        if not isinstance(last, Mapping) or not last.get("written_at_utc"):
            return True
        at = datetime.fromisoformat(last["written_at_utc"]).timestamp()
        clock = time.time() if now is None else now
        return bool(
            (policy.interval_seconds is not None and clock - at >= policy.interval_seconds)
            or (
                policy.interval_steps is not None
                and step is not None
                and isinstance(last.get("step"), int)
                and step - last["step"] >= policy.interval_steps
            )
        )

    def inventory(
        self, *, verify: bool = False, offset: int = 0, limit: int = 256
    ) -> dict[str, Any]:
        """Page registered checkpoint facts; byte verification is explicit and host-local.

        Advanced path()-written/legacy files without a managed record are not claimed verified.
        Integrity never implies that an application can deserialize or continue an epoch.
        """
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 1024
        ):
            raise ValueError("Invalid checkpoint inventory page.")
        records = sorted(self._store.records_root.glob("*.json"))
        durability = (
            self._storage_policy()
            .get("checkpoint_policy", {})
            .get("storage_durability", "unspecified")
        )
        items = []
        for path in records[offset : offset + limit]:
            if path.is_symlink() or path.resolve() != path or path.stat().st_size > 65536:
                raise ValueError("Unsafe managed checkpoint record.")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("scope") != "checkpoint":
                raise ValueError("Invalid managed checkpoint record.")
            key = self._store._key(value["key"])
            if self._store._record_path(key) != path:
                raise ValueError("Checkpoint record key does not match its identity.")
            self._check_owner(value)
            target = owned_path(self._root, key)
            exists = target.is_file()
            items.append(
                {
                    **value,
                    "path": str(target),
                    "integrity": "verified"
                    if verify and self._store.restore(key) is not None
                    else "invalid"
                    if verify or not exists
                    else "not_checked",
                    "compatibility": "application_validation_required",
                    "storage_durability": durability,
                }
            )
        return {
            "items": items,
            "total": len(records),
            "offset": offset,
            "next_offset": offset + limit if offset + limit < len(records) else None,
        }

    def _annotations(self, step: int | None) -> dict[str, Any]:
        self._validate_step(step)
        return {
            "checkpoint_metadata_version": 1,
            "owner": self._owner,
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "step": step,
        }

    def _registered_record(self, name: str) -> Mapping[str, Any] | None:
        """Reject corrupt/foreign registration before restoring or replacing any bytes."""
        key = self._store._key(name)
        path = self._store._record_path(key)
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("Unsafe managed checkpoint metadata.")
        if not path.exists():
            return None  # legacy unregistered state is not a fabricated ownership certificate
        if not path.is_file() or path.stat().st_size > 65536:
            raise ValueError("Invalid managed checkpoint metadata.")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise ValueError("Invalid managed checkpoint metadata.") from error
        if (
            not isinstance(value, Mapping)
            or value.get("managed_file_version") != self._store.RECORD_VERSION
            or value.get("scope") != "checkpoint"
            or value.get("key") != key
        ):
            raise ValueError("Invalid managed checkpoint registration identity.")
        self._check_owner(value)
        return value

    def _check_owner(self, value: Mapping[str, Any]) -> None:
        """Bind managed state to its logical Execution/Run, allowing older Attempts.

        Scientific/code compatibility is enforced by native recovery before constructing this
        service. An acknowledged compatible code revision must not invalidate the same Run's
        older checkpoint merely because its immutable writer fingerprint differs.
        """
        registered = value.get("owner")
        if registered is None:
            return  # historical registration without writer annotations remains readable
        if not isinstance(registered, Mapping):
            raise ValueError("Invalid managed checkpoint owner.")
        for field in ("execution_id", "run_id"):
            if field in registered and field in self._owner:
                if (
                    not isinstance(registered[field], str)
                    or registered[field] != self._owner[field]
                ):
                    raise ValueError(
                        f"Managed checkpoint belongs to another {field}: {value.get('key')}"
                    )

    def _committed(self, managed: ManagedFile) -> None:
        """Cadence advances on durable writer evidence, never opening an existing file."""
        written = managed.metadata.get("written_at_utc")
        if not isinstance(written, str):
            return
        current = self._storage_policy().get("last_checkpoint", {})
        if isinstance(current, Mapping) and written <= str(current.get("written_at_utc", "")):
            return

        def update(value: dict[str, Any]) -> None:
            last = value.get("last_checkpoint", {})
            if not isinstance(last, Mapping):
                raise ValueError("Invalid last checkpoint evidence.")
            if written > str(last.get("written_at_utc", "")):
                value["last_checkpoint"] = {
                    "key": managed.key,
                    "written_at_utc": written,
                    "step": managed.metadata.get("step"),
                    "sha256": managed.sha256,
                    "size_bytes": managed.size_bytes,
                    "owner": managed.metadata.get("owner"),
                }

        self._update_policy(update)

    @staticmethod
    def _validate_step(step: int | None) -> None:
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("Checkpoint step must be a non-negative integer.")

    def restore_reference(
        self,
        name: str,
        *,
        sha256: str,
        size_bytes: int,
    ) -> ManagedFile | None:
        """Restore a logical map dependency only when its checkpoint bytes still match."""
        self._registered_record(name)
        return self._store.restore(name, sha256=sha256, size_bytes=size_bytes)

    def pin(self, name: str, *, reason: str = "researcher-retained") -> Path:
        """Keep one owned checkpoint beyond terminal retention; outputs remain preferable."""
        selected = owned_path(self._root, name)
        if selected == self._root:
            raise ValueError("Pin a named checkpoint, not the collection root.")
        selected = owned_path(self._root, name, must_exist=True)
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
            source = owned_path(self._root, name)
            if source == self._root:
                raise ValueError("Release named checkpoints, not the collection root.")
            source = owned_path(self._root, name, must_exist=True)
            digest, size = fingerprint(source)
            records[source.relative_to(self._root).as_posix()] = {
                "sha256": digest,
                "size_bytes": size,
                "durable": witness,
            }
        self._update_policy(lambda policy: policy.setdefault("publications", {}).update(records))


__all__ = ["CheckpointCollection"]
