"""Type-safe neighborhood reducers for Principal Neighbourhood Aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import torch

from lambdaforge.nn.models.Scatter import Scatter


class PNAAggregator(str, Enum):
    """Select one sparse reducer used by a PNA layer.

    Values deliberately match their compact YAML representation. Reductions
    operate over incoming messages grouped by destination node and define
    empty segments as zero through :class:`~lambdaforge.nn.models.Scatter`.
    """

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    STD = "std"

    def reduce(
        self,
        messages: torch.Tensor,
        destination: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Reduce edge messages into one row per destination node."""
        if self is PNAAggregator.MEAN:
            return Scatter.mean(messages, destination, num_nodes)
        if self is PNAAggregator.MIN:
            return Scatter.minimum(messages, destination, num_nodes)
        if self is PNAAggregator.MAX:
            return Scatter.maximum(messages, destination, num_nodes)
        return Scatter.standard_deviation(messages, destination, num_nodes)

    @classmethod
    def normalize_many(
        cls,
        values: PNAAggregator | str | Iterable[PNAAggregator | str],
    ) -> tuple[PNAAggregator, ...]:
        """Return a non-empty, duplicate-free tuple of reducer policies."""
        raw_values = (values,) if isinstance(values, (cls, str)) else tuple(values)
        normalized = tuple(cls(value) for value in raw_values)
        if not normalized:
            raise ValueError("aggregators must contain at least one value.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("aggregators cannot contain duplicate values.")
        return normalized
