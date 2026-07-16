"""Implementation of the Distance object."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Distance(nn.Module, ABC):
    r"""Base class for pairwise distances between batched point sets.

    A distance module receives two batched sets of points and returns the
    pairwise distance matrix between both sets for every batch element. It is
    intentionally generic: points may represent atoms, surface points, graph
    nodes, tokens with geometric coordinates, or any other vectors in a common
    feature space.

    Shape convention
    ----------------
    B:
        Batch size.

    T_x:
        Number of points in the first input set.

    T_y:
        Number of points in the second input set.

    F:
        Feature dimension used to compute the distance. For Cartesian
        coordinates, ``F = 3``.

    Expected input
    --------------
    x : torch.Tensor
        First point set with shape ``(B, T_x, F)``.

    y : torch.Tensor
        Second point set with shape ``(B, T_y, F)``.

    Output
    ------
    torch.Tensor
        Pairwise distance matrix with shape ``(B, T_x, T_y)``.

    Notes
    -----
    The batch dimension is always explicit. If only one point set pair is
    available, use ``x.unsqueeze(0)`` and ``y.unsqueeze(0)`` before calling the
    distance.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Compute pairwise distances.

        Parameters
        ----------
        x : torch.Tensor
            First point set with shape ``(B, T_x, F)``.
        y : torch.Tensor
            Second point set with shape ``(B, T_y, F)``.

        Returns
        -------
        torch.Tensor
            Pairwise distances with shape ``(B, T_x, T_y)``.
        """
        raise NotImplementedError
