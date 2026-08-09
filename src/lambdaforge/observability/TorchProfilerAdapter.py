"""Optional bounded PyTorch profiler adapter."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch

from lambdaforge.observability.ProfilerAdapter import ProfilerAdapter


class TorchProfilerAdapter(ProfilerAdapter):
    """Configure a finite PyTorch trace schedule and TensorBoard trace output."""

    def __init__(
        self,
        *,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        record_shapes: bool = False,
    ) -> None:
        if min(wait, warmup, active, repeat) < 0 or active < 1 or repeat < 1:
            raise ValueError(
                "Profiler schedule requires active/repeat >= 1 and non-negative phases."
            )
        self.wait, self.warmup, self.active, self.repeat = wait, warmup, active, repeat
        self.record_shapes = record_shapes

    def profile(self, output_dir: str | Path) -> AbstractContextManager[Any]:
        """Create a finite CPU/CUDA profiler context."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
            record_shapes=self.record_shapes,
        )
