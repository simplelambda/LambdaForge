"""Typed terminal result for one generic task attempt."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.tasks.TaskArtifact import TaskArtifact
from lambdaforge.tasks.TaskStatus import TaskStatus


class TaskResult(JsonResult):
    """Persist structured task outputs while remaining dictionary-compatible."""

    def __init__(
        self,
        *,
        name: str,
        run_dir: str | Path,
        status: TaskStatus | str = TaskStatus.UNKNOWN,
        result_version: int = 1,
        seconds: float | None = None,
        outputs: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: Sequence[TaskArtifact | Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        attempt_id: str | None = None,
        config_fingerprint: str | None = None,
        started_at_utc: str | None = None,
        finished_at_utc: str | None = None,
        skipped_existing: bool = False,
    ) -> None:
        if isinstance(result_version, bool) or int(result_version) < 1:
            raise ValueError("Task result_version must be at least 1.")
        self.result_version = int(result_version)
        self.name = str(name)
        self.run_dir = str(run_dir)
        raw_status = status.value if isinstance(status, TaskStatus) else str(status)
        try:
            self.status = TaskStatus(raw_status)
        except ValueError:
            self.status = TaskStatus.UNKNOWN
        self._serialized_status = raw_status
        self.seconds = float(seconds) if seconds is not None else None
        self.outputs = FrozenJsonMapping(outputs or {})
        self.metrics = FrozenJsonMapping(metrics or {})
        self.artifacts = tuple(
            value if isinstance(value, TaskArtifact) else TaskArtifact.from_mapping(value)
            for value in artifacts
        )
        self.metadata = FrozenJsonMapping(metadata or {})
        self.error = FrozenJsonMapping(error) if error is not None else None
        self.attempt_id = str(attempt_id) if attempt_id is not None else None
        self.config_fingerprint = (
            str(config_fingerprint) if config_fingerprint is not None else None
        )
        self.started_at_utc = str(started_at_utc) if started_at_utc is not None else None
        self.finished_at_utc = str(finished_at_utc) if finished_at_utc is not None else None
        self.skipped_existing = bool(skipped_existing)
        self._freeze_mapping(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskResult:
        """Restore a current or compatible older task result mapping."""
        metrics = value.get("metrics", value.get("final_metrics", {}))
        return cls(
            result_version=int(value.get("result_version", 1)),
            name=str(value.get("name", "")),
            run_dir=str(value.get("run_dir", "")),
            status=str(value.get("status", TaskStatus.UNKNOWN.value)),
            seconds=value.get("seconds"),
            outputs=cls._mapping(value.get("outputs"), "outputs"),
            metrics=cls._mapping(metrics, "metrics"),
            artifacts=cls._sequence(value.get("artifacts"), "artifacts"),
            metadata=cls._mapping(value.get("metadata"), "metadata"),
            error=cls._optional_mapping(value.get("error"), "error"),
            attempt_id=value.get("attempt_id"),
            config_fingerprint=value.get("config_fingerprint"),
            started_at_utc=value.get("started_at_utc"),
            finished_at_utc=value.get("finished_at_utc"),
            skipped_existing=bool(value.get("skipped_existing", False)),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> TaskResult:
        """Read and validate one persisted task result."""
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("Task result JSON must contain an object.")
        return cls.from_mapping(value)

    def with_updates(self, **changes: Any) -> TaskResult:
        """Return a new task result with selected serialized fields changed."""
        value = self.to_dict()
        value.update(changes)
        return self.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible result representation."""
        payload: dict[str, Any] = {
            "result_version": self.result_version,
            "kind": "task",
            "name": self.name,
            "run_dir": self.run_dir,
            "variant": None,
            "seed": None,
            "status": self._serialized_status,
            "outputs": copy.deepcopy(self.outputs),
            "metrics": copy.deepcopy(self.metrics),
            "final_metrics": copy.deepcopy(self.metrics),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": copy.deepcopy(self.metadata),
            "skipped_existing": self.skipped_existing,
        }
        optional = {
            "seconds": self.seconds,
            "error": self.error,
            "attempt_id": self.attempt_id,
            "config_fingerprint": self.config_fingerprint,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = copy.deepcopy(value)
        return payload

    @staticmethod
    def _mapping(value: Any, label: str) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"Task result {label} must be a mapping.")
        return value

    @classmethod
    def _optional_mapping(cls, value: Any, label: str) -> Mapping[str, Any] | None:
        return None if value is None else cls._mapping(value, label)

    @staticmethod
    def _sequence(value: Any, label: str) -> Sequence[Any]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"Task result {label} must be a sequence.")
        return value
