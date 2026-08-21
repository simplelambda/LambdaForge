"""Derived research-centric view over durable low-level jobs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.controlplane.jobs import JobRecord


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
        planned_raw = primary.metadata.get("planned_units")
        planned = int(planned_raw) if isinstance(planned_raw, int) else None
        completed_raw = primary.metadata.get("completed_units")
        completed = int(completed_raw) if isinstance(completed_raw, int) else None
        if completed is None and primary.state.value == "succeeded":
            completed = planned
        semantic_kind = {
            "dataset-build": "dataset",
            "hpo": "experiment",
            "preprocessing": "task",
        }.get(primary.job_type, primary.job_type)
        digest = hashlib.sha256(
            f"{identity_key}\0{cluster}\0{name}".encode()
        ).hexdigest()[:16]
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
                str(primary.metadata.get("unit") or "jobs"),
                len(ordered),
                primary.job_id,
                tuple(record.job_id for record in ordered),
                ordered[0].created_at_utc,
                max(record.updated_at_utc for record in ordered),
            )
        )
    return tuple(sorted(output, key=lambda value: value.updated_at_utc, reverse=True))
