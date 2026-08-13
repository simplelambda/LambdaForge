"""Durable result and artifact provenance for one post-run action."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.tasks.TaskArtifact import TaskArtifact


class PostRunActionReceipt(JsonResult):
    """Record one action independently of the neural training result."""

    def __init__(
        self,
        *,
        name: str,
        target: str,
        action_identity: str,
        required: bool,
        status: str,
        checkpoint_role: str,
        checkpoint_path: str | None = None,
        checkpoint_sha256: str | None = None,
        seconds: float = 0.0,
        outputs: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: Sequence[TaskArtifact | Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        started_at_utc: str | None = None,
        finished_at_utc: str | None = None,
        skipped_existing: bool = False,
    ) -> None:
        if status not in {"ok", "failed"}:
            raise ValueError("Post-run action status must be 'ok' or 'failed'.")
        self.name = str(name)
        self.target = str(target)
        self.action_identity = str(action_identity)
        self.required = bool(required)
        self.status = status
        self.checkpoint_role = str(checkpoint_role)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_sha256 = str(checkpoint_sha256) if checkpoint_sha256 is not None else None
        self.seconds = float(seconds)
        self.outputs = FrozenJsonMapping(outputs or {})
        self.metrics = FrozenJsonMapping(metrics or {})
        self.artifacts = tuple(
            artifact if isinstance(artifact, TaskArtifact) else TaskArtifact.from_mapping(artifact)
            for artifact in artifacts
        )
        self.metadata = FrozenJsonMapping(metadata or {})
        self.error = FrozenJsonMapping(error) if error is not None else None
        self.started_at_utc = started_at_utc
        self.finished_at_utc = finished_at_utc
        self.skipped_existing = bool(skipped_existing)
        self._freeze_mapping(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PostRunActionReceipt:
        """Restore a validated action receipt from JSON."""
        artifacts = value.get("artifacts", ())
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes, bytearray)):
            raise TypeError("Post-run receipt artifacts must be a sequence.")
        return cls(
            name=str(value["name"]),
            target=str(value["target"]),
            action_identity=str(value["action_identity"]),
            required=bool(value["required"]),
            status=str(value["status"]),
            checkpoint_role=str(value["checkpoint_role"]),
            checkpoint_path=value.get("checkpoint_path"),
            checkpoint_sha256=value.get("checkpoint_sha256"),
            seconds=float(value.get("seconds", 0.0)),
            outputs=value.get("outputs") if isinstance(value.get("outputs"), Mapping) else {},
            metrics=value.get("metrics") if isinstance(value.get("metrics"), Mapping) else {},
            artifacts=artifacts,
            metadata=value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {},
            error=value.get("error") if isinstance(value.get("error"), Mapping) else None,
            started_at_utc=value.get("started_at_utc"),
            finished_at_utc=value.get("finished_at_utc"),
            skipped_existing=bool(value.get("skipped_existing", False)),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> PostRunActionReceipt:
        """Read one action receipt from a regular JSON object."""
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("Post-run receipt JSON must contain an object.")
        return cls.from_mapping(value)

    def with_updates(self, **changes: Any) -> PostRunActionReceipt:
        """Return an immutable copy with presentation fields changed."""
        value = self.to_dict()
        value.update(changes)
        return self.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize action status, provenance and shared artifact records."""
        payload: dict[str, Any] = {
            "receipt_version": 1,
            "name": self.name,
            "target": self.target,
            "action_identity": self.action_identity,
            "required": self.required,
            "status": self.status,
            "checkpoint_role": self.checkpoint_role,
            "seconds": self.seconds,
            "outputs": copy.deepcopy(self.outputs),
            "metrics": copy.deepcopy(self.metrics),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": copy.deepcopy(self.metadata),
            "skipped_existing": self.skipped_existing,
        }
        optional = {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "error": copy.deepcopy(self.error),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload
