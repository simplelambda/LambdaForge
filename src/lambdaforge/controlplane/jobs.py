"""Portable in-memory and persisted models for durable control-plane jobs."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


class JobState(str, Enum):
    """Normalize provider-specific scheduler states."""

    CREATED = "created"
    PLANNED = "planned"
    STAGING = "staging"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        """Return whether no further state transition is expected."""
        return self in {self.PLANNED, self.SUCCEEDED, self.FAILED, self.CANCELLED, self.TIMEOUT}


@dataclass(frozen=True, slots=True)
class JobHandle:
    """Identify a locally tracked job and its provider scheduler ID."""

    job_id: str
    cluster: str
    state: JobState
    scheduler_id: str | None = None
    preview: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a concise CLI/API payload."""
        return {
            "job_id": self.job_id,
            "cluster": self.cluster,
            "state": self.state.value,
            "scheduler_id": self.scheduler_id,
            **({"scheduler_preview": self.preview} if self.preview is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Persist enough state to reconnect, audit, cancel and retry a job."""

    job_id: str
    cluster: str
    scheduler: str
    scheduler_id: str | None
    state: JobState
    command: tuple[str, ...]
    work_dir: str
    resources: Mapping[str, Any]
    created_at_utc: str
    updated_at_utc: str
    bundle_id: str | None = None
    config_path: str | None = None
    retry_of: str | None = None
    stdout: str = ""
    stderr: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    job_type: str = "command"
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.job_id.strip() or not self.cluster.strip():
            raise ValueError("Job and cluster identifiers cannot be empty.")
        object.__setattr__(self, "state", JobState(self.state))
        object.__setattr__(self, "resources", FrozenJsonMapping(self.resources))
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> JobRecord:
        """Restore one validated persisted record."""
        command = value.get("command", ())
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes, bytearray)):
            raise TypeError("Persisted job command must be a sequence.")
        return cls(
            job_id=str(value["job_id"]),
            cluster=str(value["cluster"]),
            scheduler=str(value["scheduler"]),
            scheduler_id=str(value["scheduler_id"]) if value.get("scheduler_id") else None,
            state=JobState(str(value["state"])),
            command=tuple(str(item) for item in command),
            work_dir=str(value["work_dir"]),
            resources=value.get("resources", {}),
            created_at_utc=str(value["created_at_utc"]),
            updated_at_utc=str(value["updated_at_utc"]),
            bundle_id=str(value["bundle_id"]) if value.get("bundle_id") else None,
            config_path=str(value["config_path"]) if value.get("config_path") else None,
            retry_of=str(value["retry_of"]) if value.get("retry_of") else None,
            stdout=str(value.get("stdout", "")),
            stderr=str(value.get("stderr", "")),
            metadata=value.get("metadata", {}),
            job_type=str(value.get("job_type", "command")),
            group_id=str(value["group_id"]) if value.get("group_id") else None,
        )

    def with_updates(self, **changes: Any) -> JobRecord:
        """Return a new immutable record after one lifecycle transition."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the durable JSON representation."""
        return {
            "job_record_version": 2,
            "job_id": self.job_id,
            "cluster": self.cluster,
            "scheduler": self.scheduler,
            "scheduler_id": self.scheduler_id,
            "state": self.state.value,
            "command": list(self.command),
            "work_dir": self.work_dir,
            "resources": copy.deepcopy(self.resources),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "bundle_id": self.bundle_id,
            "config_path": self.config_path,
            "retry_of": self.retry_of,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metadata": copy.deepcopy(self.metadata),
            "job_type": self.job_type,
            "group_id": self.group_id,
        }


@dataclass(frozen=True, slots=True)
class JobGroup:
    """Keep independent replica jobs related without claiming distributed execution."""

    group_id: str
    name: str
    job_ids: tuple[str, ...]
    clusters: tuple[str, ...]
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return the durable group representation."""
        return {
            "job_group_version": 1,
            "group_id": self.group_id,
            "name": self.name,
            "job_ids": list(self.job_ids),
            "clusters": list(self.clusters),
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> JobGroup:
        """Restore a persisted group record."""
        return cls(
            str(value["group_id"]),
            str(value["name"]),
            tuple(str(item) for item in value.get("job_ids", ())),
            tuple(str(item) for item in value.get("clusters", ())),
            str(value["created_at_utc"]),
        )
