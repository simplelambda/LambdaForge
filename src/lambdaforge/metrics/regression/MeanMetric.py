"""MeanMetric – tracks the running mean of a scalar output key."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from lambdaforge.metrics.Metric import Metric


class MeanMetric(Metric):
    """Tracks the arithmetic mean of a scalar value extracted from outputs.

    Useful for monitoring average loss or any other per-batch scalar.

    Formula:
        mean = total / count

    Parameters
    ----------
    name : str
        Metric name.
    output_key : str
        Key in ``outputs`` holding the scalar (or scalar tensor) to average.
    higher_is_better : bool
        Whether larger values are better. Default: ``False``.
    """

    def __init__(
        self,
        name: str,
        output_key: str,
        higher_is_better: bool = False,
    ) -> None:
        super().__init__(name=name, higher_is_better=higher_is_better)
        self.output_key = output_key
        self.reset()

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        value = outputs[self.output_key]

        if isinstance(value, Tensor):
            value = value.detach().float().mean().cpu().item()

        self.total += float(value)
        self.count += 1

    def compute(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def distributed_state(self) -> dict[str, float | int]:
        """Return running sum and count for DDP merging."""
        return {"total": self.total, "count": self.count}

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Add one worker's running sum and count."""
        self.total += float(state["total"])
        self.count += int(state["count"])
