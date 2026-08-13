"""Atomic local persistence for explicit job groups."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from lambdaforge.controlplane.JobGroup import JobGroup
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class JobGroupStore:
    """Store small group metadata separately from authoritative remote job state."""

    def __init__(self, root: str | Path | None = None) -> None:
        state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        self.root = Path(root or state / "lambdaforge" / "job-groups").resolve()

    def put(self, group: JobGroup) -> JobGroup:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{group.group_id}.json"
        with CrossProcessFileLock(
            self.root / ".lock", shared=False, timeout_seconds=10.0, poll_interval_seconds=0.05
        ):
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(
                    json.dumps(group.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return group

    def get(self, group_id: str) -> JobGroup:
        path = self.root / f"{group_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise KeyError(f"Unknown job group {group_id!r}.") from error
        return JobGroup.from_mapping(value)

    def list(self) -> tuple[JobGroup, ...]:
        if not self.root.is_dir():
            return ()
        values = []
        for path in sorted(self.root.glob("group-*.json")):
            try:
                values.append(JobGroup.from_mapping(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return tuple(values)
