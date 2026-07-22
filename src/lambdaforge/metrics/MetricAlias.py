"""Configurable public name wrapper for any metric object."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lambdaforge.metrics.Metric import Metric


class MetricAlias(Metric):
    """Delegate a metric while overriding its logged name and/or direction."""

    def __init__(
        self,
        metric: Metric,
        name: str,
        higher_is_better: bool | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("MetricAlias name cannot be empty.")
        super().__init__(
            name=name,
            higher_is_better=(
                metric.higher_is_better if higher_is_better is None else higher_is_better
            ),
        )
        self.metric = metric

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        """Delegate one state update."""
        self.metric.update(outputs, batch, context)

    def compute(self) -> float:
        """Return the delegated scalar value."""
        return self.metric.compute()

    def reset(self) -> None:
        """Reset the delegated state."""
        self.metric.reset()

    def state_dict(self) -> dict[str, Any]:
        """Return the delegated serializable state."""
        return self.metric.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the delegated serializable state."""
        self.metric.load_state_dict(state)

    def distributed_state(self) -> dict[str, Any]:
        """Return the delegated DDP-mergeable state."""
        return self.metric.distributed_state()

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Merge one delegated DDP worker state."""
        self.metric.merge_distributed_state(state)

    def synchronize(self) -> None:
        """Delegate synchronization so specialized collectives remain available."""
        self.metric.synchronize()
