"""Per-sample stochastic-depth regularization."""

from __future__ import annotations

import torch

from lambdaforge.nn.regularization.Regularization import Regularization


class DropPath(Regularization):
    """Drop complete residual paths independently for each batch element.

    Parameters
    ----------
    probability:
        Probability of dropping one sample's path.
    scale_by_keep:
        Divide retained paths by the keep probability so their expectation is
        unchanged.
    """

    def __init__(self, probability: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("probability must be in [0, 1).")
        self.probability = float(probability)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one broadcast path mask per leading batch element."""
        if not self.training or self.probability == 0.0:
            return x
        if x.ndim < 1:
            raise ValueError("DropPath requires a batch dimension.")
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        if self.scale_by_keep:
            mask = mask / keep
        return x * mask
