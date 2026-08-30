"""Immutable public views and persisted results for the Work runtime."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lambdaforge.work.atomic import atomic_write_json


def immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze a JSON-shaped mapping for runtime exposure."""

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(nested) for key, nested in item.items()})
        if isinstance(item, list | tuple):
            return tuple(freeze(nested) for nested in item)
        return item

    return freeze(value)


@dataclass(frozen=True, slots=True)
class WorkInput:
    """Read-only provenance for one explicitly typed external input."""

    name: str
    kind: str
    logical_source: str
    path: Path
    sha256: str | None = None
    content_id: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted logical and physical input evidence."""
        return {
            "name": self.name,
            "kind": self.kind,
            "logical_source": self.logical_source,
            "path": str(self.path),
            "sha256": self.sha256,
            "content_id": self.content_id,
            "size_bytes": self.size_bytes,
        }

    def identity_dict(self) -> dict[str, Any]:
        """Return only machine-independent scientific identity fields."""
        return {
            "name": self.name,
            "kind": self.kind,
            "sha256": self.sha256,
            "content_id": self.content_id,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class WorkResources:
    """Small immutable resource view available to scientific code."""

    cpu: int
    memory: int
    gpu: int
    gpu_memory: int
    time: float | None
    storage: int
    processes: int

    def to_dict(self) -> dict[str, Any]:
        """Return portable requested-resource metadata."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkTrial:
    """Minimal immutable context for one parameter-study member."""

    index: int
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", immutable_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class WorkConfiguration:
    """Immutable researcher-facing configuration selected before execution."""

    name: str
    work_class: str
    parameters: Mapping[str, Any]
    resources: WorkResources

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", immutable_mapping(self.parameters))

    def parameter(self, name: str) -> Any:
        """Return one normalized run parameter or raise ``KeyError``."""
        return self.parameters[name]


@dataclass(frozen=True, slots=True)
class WorkArtifact:
    """Content-verified artifact owned by one Work attempt."""

    name: str
    path: str
    role: str
    sha256: str
    size_bytes: int
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable result representation."""
        return {
            "name": self.name,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class WorkResult:
    """Persisted outcome of exactly one Work Run attempt."""

    name: str
    work_class: str
    execution_id: str
    run_id: str
    attempt_id: str
    attempt_number: int
    scientific_fingerprint: str
    status: str
    run_dir: Path
    created_at_utc: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    seed: int | None
    trial: Mapping[str, Any] | None
    parameters: Mapping[str, Any]
    inputs: tuple[WorkInput, ...]
    requested_resources: WorkResources
    primary_result: Any = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, int | float] = field(default_factory=dict)
    metric_observations: int = 0
    artifacts: tuple[WorkArtifact, ...] = ()
    datasets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    environment_manifest: str = "environment.json"
    logs: str = "work.log"
    failure: Mapping[str, Any] | None = None
    resumed_from_checkpoint: bool = False
    job_id: str | None = None
    pruned: bool = False
    prune_reason: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether all execution and output finalization completed."""
        return self.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        """Return the complete machine-readable result envelope."""
        return {
            "result_version": 1,
            "name": self.name,
            "work_class": self.work_class,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "scientific_fingerprint": self.scientific_fingerprint,
            "status": self.status,
            "run_dir": str(self.run_dir),
            "created_at_utc": self.created_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "trial": copy.deepcopy(dict(self.trial)) if self.trial is not None else None,
            "parameters": _json_value(self.parameters),
            "inputs": [value.to_dict() for value in self.inputs],
            "resources": {"requested": self.requested_resources.to_dict()},
            "result": _json_value(self.primary_result),
            "outputs": _json_value(self.outputs),
            "metrics": dict(self.metrics),
            "metric_observations": self.metric_observations,
            "artifacts": [value.to_dict() for value in self.artifacts],
            "datasets": _json_value(self.datasets),
            "environment_manifest": self.environment_manifest,
            "logs": self.logs,
            "failure": copy.deepcopy(dict(self.failure)) if self.failure is not None else None,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "job_id": self.job_id,
            "pruned": self.pruned,
            "prune_reason": self.prune_reason,
        }

    def write(self, path: str | Path) -> Path:
        """Atomically persist the attempt result."""
        return atomic_json(path, self.to_dict())


def atomic_json(path: str | Path, value: Any) -> Path:
    """Write JSON atomically and never reinterpret corrupt prior state as empty."""
    return atomic_write_json(Path(path), _json_value(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value
