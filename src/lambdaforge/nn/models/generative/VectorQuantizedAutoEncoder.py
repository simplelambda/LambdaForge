"""Composable vector-quantized autoencoder with straight-through gradients."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class VectorQuantizedAutoEncoder(Model):
    """Quantize the encoder's last dimension against a learned codebook."""

    output_schema = {
        "reconstruction": "Tensor[...]",
        "encoded": "Tensor[..., code_features]",
        "quantized": "Tensor[..., code_features]",
        "code_indices": "LongTensor[...]",
        "codebook_loss": "Tensor[]",
        "commitment_loss": "Tensor[]",
        "quantization_loss": "Tensor[]",
        "perplexity": "Tensor[]",
    }

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_codes: int,
        code_features: int,
        commitment_weight: float = 0.25,
        initialization_scale: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
            raise TypeError("encoder and decoder must be torch.nn.Module instances.")
        if min(num_codes, code_features) < 1:
            raise ValueError("num_codes and code_features must be positive.")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative.")
        if initialization_scale is not None and initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive when provided.")
        self.encoder = encoder
        self.decoder = decoder
        self.num_codes = num_codes
        self.code_features = code_features
        self.commitment_weight = float(commitment_weight)
        self.codebook = nn.Embedding(num_codes, code_features)
        scale = initialization_scale or 1.0 / num_codes
        nn.init.uniform_(self.codebook.weight, -scale, scale)

    def quantize(
        self,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return straight-through codes, indices, losses and code perplexity."""
        if encoded.ndim < 2 or encoded.shape[-1] != self.code_features:
            raise ValueError("encoded must end in the configured code_features dimension.")
        flattened = encoded.reshape(-1, self.code_features)
        codebook = self.codebook.weight.to(dtype=flattened.dtype)
        distances = (
            flattened.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flattened @ codebook.transpose(0, 1)
        )
        indices = distances.argmin(dim=1)
        selected = self.codebook(indices).to(dtype=encoded.dtype).reshape_as(encoded)
        codebook_loss = (selected - encoded.detach()).square().mean()
        commitment_loss = (selected.detach() - encoded).square().mean()
        straight_through = encoded + (selected - encoded).detach()
        probabilities = torch.bincount(indices, minlength=self.num_codes).to(encoded.dtype)
        probabilities = probabilities / probabilities.sum().clamp_min(1.0)
        perplexity = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        return (
            straight_through,
            indices.reshape(encoded.shape[:-1]),
            codebook_loss,
            commitment_loss,
            perplexity,
        )

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Encode, quantize and decode while exposing trainable VQ objective terms."""
        encoded = self.encoder(*args, **kwargs)
        if not torch.is_tensor(encoded):
            raise TypeError("encoder must return a tensor.")
        quantized, indices, codebook_loss, commitment_loss, perplexity = self.quantize(encoded)
        reconstruction = self.decoder(quantized)
        if not torch.is_tensor(reconstruction):
            raise TypeError("decoder must return a tensor.")
        quantization_loss = codebook_loss + self.commitment_weight * commitment_loss
        return {
            "reconstruction": reconstruction,
            "encoded": encoded,
            "quantized": quantized,
            "code_indices": indices,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "quantization_loss": quantization_loss,
            "perplexity": perplexity,
        }
