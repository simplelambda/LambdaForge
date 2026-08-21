"""Provider-neutral presentation of job timing and observed resource usage."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from lambdaforge.controlplane.jobs import JobRecord


class JobObservation:
    """Derive honest display/API facts without changing durable job identity."""

    @classmethod
    def describe(cls, record: JobRecord, *, now: datetime | None = None) -> dict[str, Any]:
        """Return requested/observed usage and separate queue, run and age timings."""
        current = now or datetime.now(timezone.utc)
        metadata = record.metadata
        remote = metadata.get("remote_state", {})
        remote = remote if isinstance(remote, Mapping) else {}
        started = cls._timestamp(remote.get("started_at_utc"))
        finished = cls._timestamp(remote.get("finished_at_utc"))
        remote_updated = cls._timestamp(remote.get("updated_at_utc"))
        created = cls._timestamp(record.created_at_utc)
        updated = cls._timestamp(record.updated_at_utc)
        terminal_end = finished or remote_updated or updated
        effective_end = terminal_end if record.state.terminal else current
        runtime = (
            max(0.0, (effective_end - started).total_seconds())
            if started is not None and effective_end is not None
            else None
        )
        elapsed_end = terminal_end if record.state.terminal else current
        elapsed = (
            max(0.0, (elapsed_end - created).total_seconds())
            if created is not None and elapsed_end is not None
            else None
        )
        queue = max(0.0, (started - created).total_seconds()) if started and created else None
        observed = remote.get("observed_usage", {})
        observed = copy.deepcopy(observed) if isinstance(observed, Mapping) else {}
        return {
            "timing": {
                "created_at_utc": record.created_at_utc,
                "started_at_utc": remote.get("started_at_utc"),
                "finished_at_utc": remote.get("finished_at_utc"),
                "age_seconds": (max(0.0, (current - created).total_seconds()) if created else None),
                "queue_seconds": queue,
                "runtime_seconds": runtime,
                "elapsed_seconds": elapsed,
            },
            "usage": {
                "requested": copy.deepcopy(record.resources),
                "observed": observed,
                "observed_at_utc": observed.get("timestamp_utc"),
                "source": "process-supervisor" if observed else "unavailable",
            },
        }

    @staticmethod
    def _timestamp(value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
