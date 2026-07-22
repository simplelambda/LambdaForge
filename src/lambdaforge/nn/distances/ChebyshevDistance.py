"""Implementation of the ChebyshevDistance object."""

from __future__ import annotations

import torch

from lambdaforge.nn.distances.Distance import Distance


class ChebyshevDistance(Distance):
    r"""Pairwise Chebyshev (L-infinity) distance for batched point sets.

    Inputs must have shapes ``(B, T_x, F)`` and ``(B, T_y, F)``. The output
    has shape ``(B, T_x, T_y)``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y must both have shape (B, T, F)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same batch size")
        if x.shape[-1] != y.shape[-1]:
            raise ValueError("x and y must have the same feature dimension")
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError("x and y must have the same device and dtype")
        if not x.is_floating_point() or not y.is_floating_point():
            raise TypeError("x and y must be floating-point tensors")
        return torch.cdist(x, y, p=float("inf"))
