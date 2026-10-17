"""Atomic local persistence for control-plane job metadata."""

from __future__ import annotations

import heapq
import json
import os
import shutil
import stat
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.jobs import JobRecord, JobState
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
        self._validate_id(record.job_id)
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
            self._atomic_write(path, record)
        return path

    @staticmethod
    def _atomic_write(path: Path, record: JobRecord) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def record_observation(
        self,
        job_id: str,
        *,
        started: str,
        finished: str,
        source: str,
        state: JobState | None = None,
        details: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> JobRecord:
        """Merge one ordered operational fact under the existing exact Job lock.

        Re-read after acquiring the lock: a late response cannot replace a newer observation,
        erase a concurrent cancellation or overwrite submission/retained scientific metadata.
        """
        from lambdaforge.controlplane.OperationalHealth import OperationalHealth, timestamp
        from lambdaforge.controlplane.OperationalIncidents import (
            OperationalIncidents,
            read_document,
        )

        self.get(job_id)
        path = self._record_path(job_id)
        with CrossProcessFileLock(
            path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            current = self.get(job_id)
            old = current.metadata.get("operational_observation", {})
            old = old if isinstance(old, Mapping) else {}
            previous_start = timestamp(old.get("last_attempt_started_at_utc"))
            incoming_start = timestamp(started)
            if incoming_start is None or timestamp(finished) is None:
                raise ValueError("Operational observation requires valid observer timestamps.")
            if previous_start is not None and previous_start >= incoming_start:
                return current
            metadata = dict(current.metadata)
            settings = read_document(self.root / ".operational" / "monitor" / "config.json")
            if settings.get("observation"):
                metadata["observation_policy"] = settings["observation"]
            observation = OperationalHealth.observation(
                current,
                started=started,
                finished=finished,
                state=state,
                details=details,
                error=error,
                source=source,
            )
            metadata["operational_observation"] = observation
            metadata["last_refresh_at_utc"] = finished
            target = current.state
            if error is not None:
                metadata["unreachable_error"] = observation["last_error"]["message"]
                metadata["last_known_state"] = metadata.get("last_known_state", current.state.value)
                if not current.state.terminal:
                    target = JobState.UNKNOWN
            else:
                metadata.pop("unreachable_error", None)
                metadata.pop("last_known_state", None)
                remote = metadata.get("remote_state", {})
                remote = dict(remote) if isinstance(remote, Mapping) else {}
                metadata["remote_state"] = {**remote, **dict(details or {})}
                if state is not None and not current.state.terminal:
                    target = state
            # A local cancel may precede the completion of an in-flight remote read.
            # Keep the accepted cancellation/lifecycle while exposing the provider facts.
            if current.state.terminal and state is not current.state:
                observation["last_confirmed_state"] = old.get("last_confirmed_state")
                observation["last_confirmed_at_utc"] = old.get("last_confirmed_at_utc")
            result = current.with_updates(state=target, metadata=metadata, updated_at_utc=finished)
            self._atomic_write(path, result)
            OperationalIncidents(self).observe(result)
            return result

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

    def merge_metadata(self, job_id: str, changes: Mapping[str, Any]) -> JobRecord:
        """Merge operational annotations without overwriting a concurrent lifecycle/probe."""
        self.get(job_id)
        path = self._record_path(job_id)
        with CrossProcessFileLock(
            path.with_suffix(".lock"), shared=False, timeout_seconds=10, poll_interval_seconds=0.05
        ):
            current = self.get(job_id)
            updated = current.with_updates(metadata={**current.metadata, **changes})
            self._atomic_write(path, updated)
            return updated

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

    def bounded_metadata(
        self, *, limit: int = 500, record_bytes: int = 2 * 1024 * 1024
    ) -> dict[str, Any]:
        """Read a bounded project-owned metadata window, never Run/artifact/checkpoint bytes.

        A truncated/corrupt window is explicit. Native visibility and record validation remain
        authoritative; this is a diagnostic projection, not another Job catalog or status store.
        """
        if (
            type(limit) is not int
            or type(record_bytes) is not int
            or not 1 <= limit <= 1000
            or not 1 <= record_bytes <= 2 * 1024 * 1024
        ):
            raise ValueError("Bounded Job metadata requires <=1000 records of <=2 MiB each.")
        records: dict[str, JobRecord] = {}
        skipped = truncated = 0
        for root in self._roots():
            if not root.is_dir():
                continue
            candidates: list[Path] = []
            count = 0
            for path in root.glob("job-*.json"):
                count += 1
                if count > 10000:
                    truncated += 1
                    break
                candidates.append(path)
            selected = heapq.nlargest(limit, candidates, key=lambda path: path.name)
            truncated += max(0, len(candidates) - len(selected))
            for path in selected:
                try:
                    fd = os.open(
                        path,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                    )
                    with os.fdopen(fd, "rb") as stream:
                        info = os.fstat(stream.fileno())
                        if (
                            not stat.S_ISREG(info.st_mode)
                            or info.st_nlink != 1
                            or info.st_size > record_bytes
                            or os.name == "posix"
                            and info.st_uid != os.getuid()
                        ):
                            skipped += 1
                            continue
                        raw = stream.read(record_bytes + 1)
                    if len(raw) > record_bytes:
                        skipped += 1
                        continue
                    record = JobRecord.from_mapping(json.loads(raw))
                    if path.stem != record.job_id:
                        raise ValueError("Job metadata filename/identity mismatch.")
                    if record.job_id not in records and self._visible(record):
                        records[record.job_id] = record
                except (OSError, ValueError, KeyError, TypeError):
                    skipped += 1
        values = sorted(records.values(), key=lambda row: row.created_at_utc, reverse=True)
        truncated += max(0, len(values) - limit)
        return {"records": tuple(values[:limit]), "skipped": skipped, "truncated": truncated}

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
        recovery_request_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one non-secret lifecycle fact without rewriting prior history."""
        record = self.get(job_id)
        event = {
            "event_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "state": state,
            "phase": phase,
            "source": source,
            "message": str(message).replace("\r", " ").replace("\n", " "),
            **(
                {"details": json.loads(json.dumps(dict(details), allow_nan=False))}
                if details
                else {}
            ),
            **(
                {"recovery_request_id": recovery_request_id}
                if recovery_request_id is not None
                else {"recovery_request_id": record.metadata["recovery_request"]["request_id"]}
                if isinstance(record.metadata.get("recovery_request"), Mapping)
                else {}
            ),
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

    def event_page(self, job_id: str, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        """Read a bounded newest-first native audit page, without consulting the provider.

        Count then slice under the existing event lock rather than loading the full history.
        Malformed/partial historical lines are ignored as in ``events``; oversized records
        are not split into apparently valid suffixes. Pagination is an explicit read.
        """
        self.get(job_id)
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 200
        ):
            raise ValueError("Audit pagination requires offset >= 0 and limit between 1 and 200.")
        path = self._record_path(job_id).with_name(f"{job_id}.events.jsonl")
        if path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError("Unsafe Job audit path; refusing linked operational evidence.")
        if not path.exists():
            return {"items": [], "total": 0, "offset": offset, "next_offset": None}

        def records() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                while line := handle.readline(65537):
                    if len(line) > 65536:
                        while not line.endswith("\n") and (line := handle.readline(65537)):
                            pass
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict) and value.get("job_id") == job_id:
                        yield value

        with CrossProcessFileLock(
            path.with_suffix(".lock"),
            shared=True,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            total = sum(1 for _ in records())
            start, end = max(0, total - offset - limit), max(0, total - offset)
            items = []
            for position, value in enumerate(records()):
                if position >= end:
                    break
                if position >= start:
                    items.append(value)
        return {
            "items": list(reversed(items)),
            "total": total,
            "offset": offset,
            "next_offset": offset + len(items) if offset + len(items) < total else None,
        }

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
