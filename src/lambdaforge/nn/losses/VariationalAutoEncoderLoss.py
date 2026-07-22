"""Reusable reconstruction-plus-KL objective for variational autoencoders."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss


class VariationalAutoEncoderLoss(Loss):
    """Combine a configurable reconstruction objective with beta-weighted KL."""

    def __init__(
        self,
        reconstruction: str = "mse",
        reconstruction_key: str = "reconstruction",
        target_key: str = "target",
        mean_key: str = "mean",
        log_variance_key: str = "log_variance",
        kl_key: str = "kl_divergence",
        beta: float = 1.0,
        free_bits: float = 0.0,
        reduction: str = "mean",
        weight: float = 1.0,
        name: str = "variational_autoencoder",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if reconstruction not in {"mse", "l1", "binary_cross_entropy"}:
            raise ValueError("reconstruction must be mse, l1 or binary_cross_entropy.")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be mean or sum.")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative.")
        if not math.isfinite(free_bits) or free_bits < 0:
            raise ValueError("free_bits must be finite and non-negative.")
        self.reconstruction = reconstruction
        self.reconstruction_key = reconstruction_key
        self.target_key = target_key
        self.mean_key = mean_key
        self.log_variance_key = log_variance_key
        self.kl_key = kl_key
        self.beta = float(beta)
        self.free_bits = float(free_bits)
        self.reduction = reduction

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        """Return the scalar weighted evidence-lower-bound objective."""
        del context
        reconstruction = outputs[self.reconstruction_key]
        target = batch[self.target_key]
        if not torch.is_tensor(reconstruction) or not torch.is_tensor(target):
            raise TypeError("VAE reconstruction and target must be tensors.")
        target = target.to(device=reconstruction.device, dtype=reconstruction.dtype)
        if self.reconstruction == "mse":
            reconstruction_loss = F.mse_loss(reconstruction, target, reduction=self.reduction)
        elif self.reconstruction == "l1":
            reconstruction_loss = F.l1_loss(reconstruction, target, reduction=self.reduction)
        else:
            reconstruction_loss = F.binary_cross_entropy(
                reconstruction, target, reduction=self.reduction
            )
        divergence = outputs.get(self.kl_key)
        if divergence is None:
            mean = outputs[self.mean_key]
            log_variance = outputs[self.log_variance_key]
            if not torch.is_tensor(mean) or not torch.is_tensor(log_variance):
                raise TypeError("VAE distribution outputs must be tensors.")
            divergence = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(
                dim=-1
            )
        if not torch.is_tensor(divergence):
            raise TypeError("VAE KL divergence must be a tensor.")
        divergence = divergence.clamp_min(self.free_bits)
        kl_loss = divergence.mean() if self.reduction == "mean" else divergence.sum()
        return self.weight * (reconstruction_loss + self.beta * kl_loss)
