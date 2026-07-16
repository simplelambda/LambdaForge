"""Implementation of the Normalization object."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Normalization(nn.Module, ABC):
    r"""Base class for normalization layers.

    A normalization layer transforms an input tensor into a normalized output
    tensor. Concrete subclasses decide which statistics are used, which axes
    are normalized, and whether the layer has trainable parameters.

    This base class intentionally does not define affine parameters, running
    statistics, or epsilon values. Those details belong to specific
    normalization implementations.

    Parameters
    ----------
    name : str | None
        Optional name used to identify the normalization layer.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the normalization layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        raise NotImplementedError
