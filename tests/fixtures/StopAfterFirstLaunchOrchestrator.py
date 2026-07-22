"""Spawn-safe orchestrator that requests a stop after its first launch."""

from __future__ import annotations

from collections.abc import Sequence

import torch.multiprocessing as mp

from lambdaforge.training.orchestration.TrainingJob import TrainingJob
from lambdaforge.training.orchestration.TrainingOrchestrator import TrainingOrchestrator


class StopAfterFirstLaunchOrchestrator(TrainingOrchestrator):
    """Exercise the no-new-launches invariant after a cooperative stop."""

    def _launch(
        self,
        job: TrainingJob,
        devices: Sequence[int] | None,
        slot_index: int,
    ) -> mp.Process:
        process = super()._launch(job, devices, slot_index)
        self.request_stop()
        return process
