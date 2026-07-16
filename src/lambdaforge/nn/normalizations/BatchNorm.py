"""Implementation of the BatchNorm object."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class BatchNorm(Normalization):
    r"""Batch Normalization.

    Batch normalization normalizes inputs using statistics computed over the
    batch and keeps running statistics for evaluation.

    This implementation delegates to one of:

    - ``torch.nn.BatchNorm1d``
    - ``torch.nn.BatchNorm2d``
    - ``torch.nn.BatchNorm3d``

    Parameters
    ----------
    num_features : int
        Number of features or channels.
    dim : Literal[1, 2, 3]
        Dimensionality of the batch normalization layer.
    eps : float
        Small numerical value used to avoid divisions by zero.
    momentum : float | None
        Value used for the running mean and running variance update.
    affine : bool
        Whether to use learnable affine parameters.
    track_running_stats : bool
        Whether to keep running mean and running variance.
    name : str | None
        Optional name used to identify the normalization layer.
    device : torch.device | str | None
        Optional device for the internal PyTorch module.
    dtype : torch.dtype | None
        Optional dtype for the internal PyTorch module.
    """

    def __init__(
        self,
        num_features: int,
        dim: Literal[1, 2, 3] = 1,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)

        norm_cls: type[nn.Module]

        if dim == 1:
            norm_cls = nn.BatchNorm1d
        elif dim == 2:
            norm_cls = nn.BatchNorm2d
        elif dim == 3:
            norm_cls = nn.BatchNorm3d
        else:
            raise ValueError(f"Unsupported BatchNorm dim: {dim}")

        self.norm = norm_cls(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
