"""Scalar reduction modes supported by training losses."""

from __future__ import annotations

from enum import Enum

import torch


class Reduction(str, Enum):
    """Eliminate magic reduction strings while remaining YAML-compatible.

    LambdaForge training losses intentionally support only scalar reductions;
    unreduced tensors belong in task-specific code rather than ``LightningTask``.
    """

    MEAN = "mean"
    SUM = "sum"

    @classmethod
    def from_value(cls, value: Reduction | str) -> Reduction:
        """Normalize an enum or YAML string into a scalar reduction."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            raise ValueError("reduction must be 'mean' or 'sum'.") from error

    def reduce(self, value: torch.Tensor) -> torch.Tensor:
        """Apply this reduction to an element-wise loss tensor."""
        return value.mean() if self is Reduction.MEAN else value.sum()
