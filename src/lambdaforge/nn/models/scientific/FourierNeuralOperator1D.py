"""One-dimensional Fourier neural operator for batch-first fields."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class FourierNeuralOperator1D(Model):
    """Mix low-frequency global modes with learned pointwise residual paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: int = 16,
        num_layers: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, width, modes, num_layers) < 1:
            raise ValueError("Channel sizes, modes and num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.modes = modes
        self.num_layers = num_layers
        self.input = nn.Linear(in_channels, width, bias=bias)
        scale = 1.0 / width
        self.spectral_weights = nn.Parameter(
            scale
            * torch.randn(
                num_layers,
                width,
                width,
                modes,
                dtype=torch.complex64,
            )
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1, bias=bias) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x=(batch,length,in_channels)`` to a field on the same grid."""
        if x.ndim != 3 or not torch.is_floating_point(x):
            raise TypeError("x must be a rank-three floating tensor.")
        encoded = self.input(x).transpose(1, 2)
        for index, pointwise in enumerate(self.pointwise):
            frequency = torch.fft.rfft(encoded, dim=-1)
            usable_modes = min(self.modes, frequency.shape[-1])
            transformed = torch.zeros_like(frequency)
            weights = self.spectral_weights[index, :, :, :usable_modes].to(frequency.dtype)
            transformed[:, :, :usable_modes] = torch.einsum(
                "bim,iom->bom", frequency[:, :, :usable_modes], weights
            )
            spectral = torch.fft.irfft(transformed, n=encoded.shape[-1], dim=-1)
            encoded = spectral + pointwise(encoded)
            if index + 1 < self.num_layers:
                encoded = self.dropout(torch.nn.functional.gelu(encoded))
        return self.output(encoded.transpose(1, 2))
