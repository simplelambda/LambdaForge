"""Composable branch/trunk neural operator inspired by DeepONet."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class DeepONet(Model):
    """Combine function observations and query coordinates through latent products."""

    def __init__(
        self,
        branch: nn.Module,
        trunk: nn.Module,
        latent_features: int,
        out_features: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(branch, nn.Module) or not isinstance(trunk, nn.Module):
            raise TypeError("branch and trunk must be torch.nn.Module instances.")
        if min(latent_features, out_features) < 1:
            raise ValueError("latent_features and out_features must be positive.")
        self.branch = branch
        self.trunk = trunk
        self.latent_features = latent_features
        self.out_features = out_features
        self.output_bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(
        self,
        function_observations: torch.Tensor,
        locations: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate operators at shared ``(P,D)`` or batched ``(B,P,D)`` locations."""
        branch = self.branch(function_observations, *args, **kwargs)
        trunk = self.trunk(locations)
        if not torch.is_tensor(branch) or not torch.is_tensor(trunk):
            raise TypeError("branch and trunk must return tensors.")
        expected = self.latent_features * self.out_features
        if branch.ndim != 2 or branch.shape[-1] != expected:
            raise ValueError("branch output must have shape (batch, latent_features*out_features).")
        if trunk.ndim not in {2, 3} or trunk.shape[-1] != expected:
            raise ValueError("trunk output must end in latent_features*out_features.")
        branch = branch.reshape(branch.shape[0], self.latent_features, self.out_features)
        if trunk.ndim == 2:
            trunk = trunk.reshape(trunk.shape[0], self.latent_features, self.out_features)
            result = torch.einsum("blo,plo->bpo", branch, trunk)
        else:
            if trunk.shape[0] != branch.shape[0]:
                raise ValueError("Batched locations must match the observation batch size.")
            trunk = trunk.reshape(
                trunk.shape[0], trunk.shape[1], self.latent_features, self.out_features
            )
            result = torch.einsum("blo,bplo->bpo", branch, trunk)
        return result + self.output_bias if self.output_bias is not None else result
