"""Atomic local persistence for control-plane job metadata."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class JobStore:
    """Store one immutable JSON snapshot per job without a server database."""

    def __init__(
        self, root: str | Path | None = None, *, project: ProjectContext | None = None
    ) -> None:
        self.project = (project or ProjectContext.discover()) if root is None else project
        self.legacy_root: Path | None = None
        if root is None:
            state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            base = state_home / "lambdaforge"
            assert self.project is not None
            self.legacy_root = (base / "jobs").expanduser().resolve()
            root = base / "projects" / self.project.project_id / "jobs"
        self.root = Path(root).expanduser().resolve()

    def write(self, record: JobRecord) -> Path:
        """Atomically create or update the exact job record."""
        if self.project is not None:
            owner = record.metadata.get("project_id")
            if owner is not None and owner != self.project.project_id:
                raise ValueError("Cannot write a Job belonging to another project.")
            record = record.with_updates(
                metadata={
                    **record.metadata,
                    "project_id": self.project.project_id,
                    "project_root": str(self.project.root),
                }
            )
        path = self.root / f"{record.job_id}.json"
        if self.project is not None:
            try:
                existing = self._record_path(record.job_id)
            except FileNotFoundError:
                pass
            else:
                self.get(record.job_id)  # Never adopt another project's existing record.
                path = existing
        path.parent.mkdir(parents=True, exist_ok=True)
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
        self._validate_id(job_id)
        path = self._record_path(job_id)
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError(f"Job record is not an object: {path}")
        record = JobRecord.from_mapping(value)
        if not self._visible(record):
            raise FileNotFoundError(f"Job {job_id!r} is not in the current project.")
        return record

    @staticmethod
    def _validate_id(job_id: str) -> None:
        if not job_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in job_id
        ):
            raise ValueError("Invalid LambdaForge job id.")

    def _record_path(self, job_id: str) -> Path:
        """Resolve current storage first, then one read-compatible legacy record."""
        current = self.root / f"{job_id}.json"
        if current.is_file() and not current.is_symlink():
            return current
        if self.legacy_root is not None:
            legacy = self.legacy_root / f"{job_id}.json"
            if legacy.is_file() and not legacy.is_symlink():
                return legacy
        raise FileNotFoundError(f"Unknown LambdaForge Job {job_id!r}.")

    def _roots(self) -> tuple[Path, ...]:
        return (self.root,) if self.legacy_root is None else (self.root, self.legacy_root)

    def _visible(self, record: JobRecord) -> bool:
        if self.project is None:
            return True
        owner = record.metadata.get("project_id")
        if owner is not None:
            return owner == self.project.project_id
        return self.project.owns_source(
            record.metadata.get("source_config_path") or record.config_path
        )

    def records(self) -> tuple[JobRecord, ...]:
        """List valid records in reverse creation order."""
        records: list[JobRecord] = []
        seen: set[str] = set()
        for root in self._roots():
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.json")):
                if path.is_symlink():
                    continue
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        value = json.load(handle)
                    if isinstance(value, dict):
                        record = JobRecord.from_mapping(value)
                        if record.job_id not in seen and self._visible(record):
                            records.append(record)
                            seen.add(record.job_id)
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return tuple(sorted(records, key=lambda item: item.created_at_utc, reverse=True))

    def delete(self, job_id: str) -> None:
        """Delete only local metadata for one validated job id."""
        self.get(job_id)
        path = self._record_path(job_id)
        root = path.parent
        submission = root / "submissions" / job_id
        if submission.exists() and (submission.is_symlink() or not submission.is_dir()):
            raise RuntimeError(f"Unsafe Job submission history path: {submission}")
        if submission.exists():
            if submission.is_symlink() or not submission.is_dir():
                raise RuntimeError(f"Unsafe Job submission history path: {submission}")
            shutil.rmtree(submission)
        with CrossProcessFileLock(
            path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            path.unlink()
            (root / f"{job_id}.events.jsonl").unlink(missing_ok=True)

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
        path = self._record_path(job_id).with_name(f"{job_id}.events.jsonl")
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
        path = self._record_path(job_id).with_name(f"{job_id}.events.jsonl")
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
