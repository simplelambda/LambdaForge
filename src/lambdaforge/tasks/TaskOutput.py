"""Structured user output from a generic task."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.tasks.ArtifactDeclaration import ArtifactDeclaration


class TaskOutput:
    """Carry JSON outputs, scalar metrics, artifact declarations and metadata."""

    def __init__(
        self,
        *,
        outputs: Mapping[str, Any] | None = None,
        metrics: Mapping[str, int | float] | None = None,
        artifacts: Sequence[ArtifactDeclaration | str | Path | Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.outputs = FrozenJsonMapping(outputs or {})
        self.metrics = FrozenJsonMapping(metrics or {})
        self.artifacts = tuple(ArtifactDeclaration.from_value(value) for value in artifacts)
        self.metadata = FrozenJsonMapping(metadata or {})
        try:
            json.dumps(
                {
                    "outputs": self.outputs,
                    "metrics": self.metrics,
                    "metadata": self.metadata,
                    "artifacts": [artifact.metadata for artifact in self.artifacts],
                }
            )
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Task outputs, metrics and metadata must be JSON-compatible: {error}"
            ) from error
        for name, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Task metric {name!r} must be a numeric scalar.")

    @classmethod
    def from_value(cls, value: TaskOutput | Mapping[str, Any] | None) -> TaskOutput:
        """Normalize the supported task return forms without reserved mapping magic."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(outputs=value)
        raise TypeError("Task.run() must return TaskOutput, a mapping, or None.")
