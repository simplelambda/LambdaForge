"""Implementation of the EuclideanDistance object."""

from __future__ import annotations

import torch

from lambdaforge.nn.distances.Distance import Distance


class EuclideanDistance(Distance):
    r"""Euclidean distance between two batched point sets.

    This module computes:

        d(x, y) = ||x - y||_2

    Shape convention
    ----------------
    B:
        Batch size.

    T_x:
        Number of points in ``x``.

    T_y:
        Number of points in ``y``.

    F:
        Feature dimension. For 3D coordinates, ``F = 3``.

    Expected input
    --------------
    x : torch.Tensor
        Tensor with shape ``(B, T_x, F)``.

    y : torch.Tensor
        Tensor with shape ``(B, T_y, F)``.

    Output
    ------
    torch.Tensor
        Euclidean distances with shape ``(B, T_x, T_y)``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y)
