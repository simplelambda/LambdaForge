"""Implementation of the SquaredEuclideanDistance object."""

from __future__ import annotations

import torch

from lambdaforge.nn.distances.Distance import Distance


class SquaredEuclideanDistance(Distance):
    r"""Squared Euclidean distance between two batched point sets.

    This module computes:

        d(x, y) = ||x - y||_2^2

    Squared distances are often more convenient than Euclidean distances inside
    neural models because they avoid the square root and remain monotonic with
    respect to nearest-neighbor ordering.

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
        Squared Euclidean distances with shape ``(B, T_x, T_y)``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y).square()
