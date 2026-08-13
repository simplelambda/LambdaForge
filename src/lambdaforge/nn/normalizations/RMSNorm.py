"""Implementation of the RMSNorm object."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class RMSNorm(Normalization):
    r"""Root Mean Square Layer Normalization.

    RMSNorm normalizes the input using the root mean square value, without
    subtracting the mean.

    This implementation delegates to ``torch.nn.RMSNorm``.

    Parameters
    ----------
    normalized_shape : int | Sequence[int]
        Input shape from an expected input of size ``[..., normalized_shape]``.
    eps : float | None
        Small numerical value used to avoid divisions by zero. If ``None``,
        PyTorch uses its default value.
    elementwise_affine : bool
        Whether to use learnable per-element scale parameters.
    name : str | None
        Optional name used to identify the normalization layer.
    device : torch.device | str | None
        Optional device for the internal PyTorch module.
    dtype : torch.dtype | None
        Optional dtype for the internal PyTorch module.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float | None = None,
        elementwise_affine: bool = True,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)

        resolved_shape = (
            normalized_shape if isinstance(normalized_shape, int) else list(normalized_shape)
        )
        rms_norm = vars(nn)["RMSNorm"]
        self.norm = rms_norm(
            normalized_shape=resolved_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
