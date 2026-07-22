"""Spawn-safe artifact-retention application job."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.retention.ArtifactRetentionManager import (
    ArtifactRetentionManager,
)


class RetentionApplyJob:
    """Synchronize concurrent apply attempts and report their typed status."""

    def __init__(
        self,
        config: Mapping[str, Any],
        ready_queue: Any,
        start_event: Any,
        result_queue: Any,
    ) -> None:
        self.config = dict(config)
        self.ready_queue = ready_queue
        self.start_event = start_event
        self.result_queue = result_queue

    def __call__(self) -> None:
        self.ready_queue.put(True)
        if not self.start_event.wait(10.0):
            raise TimeoutError("Concurrent retention start gate was not released.")
        result = ArtifactRetentionManager().apply(ExperimentConfig(self.config))
        self.result_queue.put((result.status.value, result.plan_id))
