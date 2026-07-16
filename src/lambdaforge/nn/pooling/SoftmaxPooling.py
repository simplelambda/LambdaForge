"""Implementation of the SoftmaxPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class SoftmaxPooling(Pooling):
    r"""Softmax-weighted pooling over the set dimension.

    This operator computes one weight per valid element and feature channel:

    .. math::

        a_{i,d} = softmax_i(\beta h_{i,d})

    .. math::

        z_d = \sum_i a_{i,d} h_{i,d}

    Small ``beta`` values behave close to mean pooling. Larger values make the
    operator progressively closer to max pooling while keeping dense gradients.

    Parameters
    ----------
    beta : float
        Positive sharpness parameter used before the softmax.
    name : str | None
        Optional name used to identify the pooling layer.
    """

    def __init__(
        self,
        beta: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.beta = float(beta)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"SoftmaxPooling expects x with shape (B, N, D), got {tuple(x.shape)}."
            )

        scores = self.beta * x  # (B, N, D)
        valid = None if mask is None else mask.bool()  # (B, N)

        if valid is not None:
            scores = scores.masked_fill(~valid.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(scores, dim=1)  # (B, N, D)
        weights = torch.nan_to_num(weights, nan=0.0)

        if valid is not None:
            weights = weights.masked_fill(~valid.unsqueeze(-1), 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return (weights * x).sum(dim=1)  # (B, D)
