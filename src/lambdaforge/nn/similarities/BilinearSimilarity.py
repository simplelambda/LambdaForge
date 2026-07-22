"""Learnable bilinear pairwise similarity."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.similarities.Similarity import Similarity


class BilinearSimilarity(Similarity):
    r"""Compute :math:`x^T W y + b` for every pair in a batch.

    Parameters
    ----------
    in_features:
        Feature dimension of both inputs.
    bias:
        Add a learnable scalar bias.
    symmetric:
        Use ``(W + W.T) / 2`` at runtime, guaranteeing symmetric scores when
        the two inputs represent the same vector space.
    device, dtype:
        Placement of learnable parameters.
    """

    def __init__(
        self,
        in_features: int,
        bias: bool = True,
        symmetric: bool = False,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(in_features, bool) or not isinstance(in_features, int):
            raise TypeError("in_features must be an integer.")
        if in_features < 1:
            raise ValueError("in_features must be positive.")
        if not isinstance(bias, bool) or not isinstance(symmetric, bool):
            raise TypeError("bias and symmetric must be booleans.")
        self.weight = nn.Parameter(
            torch.empty(in_features, in_features, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.empty((), device=device, dtype=dtype)) if bias else None
        self.in_features = int(in_features)
        self.symmetric = bool(symmetric)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the bilinear form with Xavier-uniform weights."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.validate_inputs(x, y)
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected feature dimension {self.in_features}, got {x.shape[-1]}.")
        if x.device != self.weight.device or x.dtype != self.weight.dtype:
            raise ValueError(
                "Inputs and BilinearSimilarity parameters must share device and dtype."
            )
        weight = (
            (self.weight + self.weight.transpose(0, 1)) * 0.5 if self.symmetric else self.weight
        )
        scores = torch.matmul(torch.matmul(x, weight), y.transpose(-1, -2))
        return scores if self.bias is None else scores + self.bias
