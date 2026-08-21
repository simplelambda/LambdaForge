"""Atomic local persistence for control-plane job metadata."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class JobStore:
    """Store one immutable JSON snapshot per job without a server database."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            root = state_home / "lambdaforge" / "jobs"
        self.root = Path(root).expanduser().resolve()

    def write(self, record: JobRecord) -> Path:
        """Atomically create or update the exact job record."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{record.job_id}.json"
        with CrossProcessFileLock(
            path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(
                    json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def get(self, job_id: str) -> JobRecord:
        """Read one exact record and reject path-like identifiers."""
        if not job_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in job_id
        ):
            raise ValueError("Invalid LambdaForge job id.")
        path = self.root / f"{job_id}.json"
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError(f"Job record is not an object: {path}")
        return JobRecord.from_mapping(value)

    def records(self) -> tuple[JobRecord, ...]:
        """List valid records in reverse creation order."""
        if not self.root.is_dir():
            return ()
        records = []
        for path in sorted(self.root.glob("*.json")):
            if path.is_symlink():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
                if isinstance(value, dict):
                    records.append(JobRecord.from_mapping(value))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return tuple(sorted(records, key=lambda item: item.created_at_utc, reverse=True))

    def delete(self, job_id: str) -> None:
        """Delete only local metadata for one validated job id."""
        self.get(job_id)
        path = self.root / f"{job_id}.json"
        with CrossProcessFileLock(
            path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            path.unlink()
            (self.root / f"{job_id}.events.jsonl").unlink(missing_ok=True)

    def append_event(
        self,
        job_id: str,
        *,
        state: str,
        message: str,
        phase: str | None = None,
        source: str = "lambdaforge",
    ) -> dict[str, Any]:
        """Append one non-secret lifecycle fact without rewriting prior history."""
        self.get(job_id)
        event = {
            "event_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "state": state,
            "phase": phase,
            "source": source,
            "message": str(message).replace("\r", " ").replace("\n", " "),
        }
        path = self.root / f"{job_id}.events.jsonl"
        with CrossProcessFileLock(
            path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return event

    def events(self, job_id: str) -> tuple[dict[str, Any], ...]:
        """Read the append-only lifecycle stream, ignoring only malformed partial lines."""
        self.get(job_id)
        path = self.root / f"{job_id}.events.jsonl"
        if not path.is_file() or path.is_symlink():
            return ()
        values: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping) and value.get("job_id") == job_id:
                    values.append(dict(value))
        return tuple(values)
