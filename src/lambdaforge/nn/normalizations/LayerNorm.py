"""Implementation of the LayerNorm object."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class LayerNorm(Normalization):
    r"""Layer Normalization.

    Layer normalization normalizes each sample using statistics computed over
    the last dimensions of the input tensor.

    This implementation delegates to ``torch.nn.LayerNorm``.

    Parameters
    ----------
    normalized_shape : int | Sequence[int]
        Input shape from an expected input of size ``[..., normalized_shape]``.
    eps : float
        Small numerical value used to avoid divisions by zero.
    elementwise_affine : bool
        Whether to use learnable per-element affine parameters.
    bias : bool
        Whether to include the bias parameter when ``elementwise_affine`` is
        enabled.
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
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)

        resolved_shape = (
            normalized_shape if isinstance(normalized_shape, int) else list(normalized_shape)
        )
        self.norm = nn.LayerNorm(
            normalized_shape=resolved_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)
