"""Implementation of the Pooling object."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Pooling(nn.Module, ABC):
    r"""Base class for set pooling layers.

    A pooling layer reduces a set of :math:`N` feature vectors of dimension
    :math:`D` into a single representation per batch element.

    Optionally, a binary mask can mark which elements of the set are valid.
    Implementations are responsible for ignoring masked-out positions.

    Shape convention
    ----------------
    B:
        Batch size.

    N:
        Number of elements per sample.  For example, vertices, atoms, nodes,
        tokens, or neighbors in a local patch.

    D:
        Feature dimension per element.

    Expected input
    --------------
    x : torch.Tensor
        Feature matrix with shape ``(B, N, D)``.

    mask : torch.Tensor | None
        Optional validity mask with shape ``(B, N)``.  Float or bool —
        ``1`` / ``True`` marks valid positions; ``0`` / ``False`` marks
        positions that should be ignored during pooling.

    Output
    ------
    torch.Tensor
        Pooled representation.  Shape depends on the implementation,
        typically ``(B, D)`` or ``(B, D')``.

    Notes
    -----
    This module assumes that the batch dimension is always present.  It does
    not support unbatched inputs such as ``(N, D)``.  If only one sample is
    available, use ``x.unsqueeze(0)`` before calling the pooling layer.

    Parameters
    ----------
    name : str | None
        Optional name used to identify the pooling layer.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        r"""Pool the set of feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Feature matrix with shape ``(B, N, D)``.
        mask : torch.Tensor | None
            Optional validity mask with shape ``(B, N)``.  Non-zero entries
            indicate valid positions; zero entries are ignored.

        Returns
        -------
        torch.Tensor
            Pooled representation.
        """
        raise NotImplementedError
