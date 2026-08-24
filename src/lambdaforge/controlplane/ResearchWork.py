"""Derived research-centric view over durable low-level jobs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.controlplane.jobs import JobRecord


@dataclass(frozen=True, slots=True)
class ResearchAttempt:
    """Human-numbered scheduler attempt linked to its machine Job identifier."""

    number: int
    job_id: str
    state: str
    cluster: str
    scheduler: str
    scheduler_id: str | None
    job_type: str
    created_at_utc: str
    updated_at_utc: str
    retry_of: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a display-first record while retaining the Job ID for automation."""
        return {
            "number": self.number,
            "label": f"Attempt {self.number}",
            "job_id": self.job_id,
            "state": self.state,
            "cluster": self.cluster,
            "scheduler": self.scheduler,
            "scheduler_id": self.scheduler_id,
            "job_type": self.job_type,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "retry_of": self.retry_of,
        }


@dataclass(frozen=True, slots=True)
class ResearchWork:
    """Group attempts of one scientific revision on one target without owning new state."""

    work_id: str
    name: str
    kind: str
    cluster: str
    state: str
    scientific_identity: str | None
    scientific_revision: str | None
    completed_units: int | None
    planned_units: int | None
    unit: str
    attempts: int
    primary_job_id: str
    job_ids: tuple[str, ...]
    attempt_history: tuple[ResearchAttempt, ...]
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable additive read-model payload used by TUI and wrappers."""
        return {
            "work_id": self.work_id,
            "name": self.name,
            "kind": self.kind,
            "cluster": self.cluster,
            "state": self.state,
            "scientific_identity": self.scientific_identity,
            "scientific_revision": self.scientific_revision,
            "progress": {
                "completed": self.completed_units,
                "total": self.planned_units,
                "unit": self.unit,
            },
            "attempts": self.attempts,
            "primary_job_id": self.primary_job_id,
            "job_ids": list(self.job_ids),
            "attempt_history": [attempt.to_dict() for attempt in self.attempt_history],
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


def aggregate_research_work(records: Sequence[JobRecord]) -> tuple[ResearchWork, ...]:
    """Derive semantic work groups while leaving JobStore as the sole authority."""
    grouped: dict[tuple[str, str, str], list[JobRecord]] = defaultdict(list)
    for record in records:
        identity_key = str(record.metadata.get("scientific_identity") or record.job_id)
        name = str(record.metadata.get("name") or record.config_path or record.job_id)
        grouped[(identity_key, record.cluster, name)].append(record)
    output = []
    for (identity_key, cluster, name), attempts in grouped.items():
        ordered = sorted(attempts, key=lambda value: value.created_at_utc)
        active = [record for record in ordered if not record.state.terminal]
        primary = active[-1] if active else ordered[-1]
        identity_value = primary.metadata.get("scientific_identity")
        identity = str(identity_value) if identity_value else None
        revision = (
            str(primary.metadata.get("scientific_revision"))
            if primary.metadata.get("scientific_revision")
            else identity.removeprefix("sha256:")[:12]
            if identity
            else None
        )
        remote = primary.metadata.get("remote_state", {})
        remote = remote if isinstance(remote, dict) else {}
        progress = remote.get("progress", {})
        progress = progress if isinstance(progress, dict) else {}
        planned_raw = progress.get("total", primary.metadata.get("planned_units"))
        planned = int(planned_raw) if isinstance(planned_raw, int) else None
        completed_raw = progress.get("completed", primary.metadata.get("completed_units"))
        completed = int(completed_raw) if isinstance(completed_raw, int) else None
        if completed is None and primary.state.value == "succeeded":
            completed = planned
        semantic_kind = "work"
        digest = hashlib.sha256(f"{identity_key}\0{cluster}\0{name}".encode()).hexdigest()[:16]
        output.append(
            ResearchWork(
                f"work-{digest}",
                name,
                semantic_kind,
                cluster,
                primary.state.value,
                identity,
                revision,
                completed,
                planned,
                "items" if progress else str(primary.metadata.get("unit") or "runs"),
                len(ordered),
                primary.job_id,
                tuple(record.job_id for record in ordered),
                tuple(
                    ResearchAttempt(
                        number,
                        record.job_id,
                        record.state.value,
                        record.cluster,
                        record.scheduler,
                        record.scheduler_id,
                        record.job_type,
                        record.created_at_utc,
                        record.updated_at_utc,
                        record.retry_of,
                    )
                    for number, record in enumerate(ordered, 1)
                ),
                ordered[0].created_at_utc,
                max(record.updated_at_utc for record in ordered),
            )
        )
    return tuple(sorted(output, key=lambda value: value.updated_at_utc, reverse=True))
