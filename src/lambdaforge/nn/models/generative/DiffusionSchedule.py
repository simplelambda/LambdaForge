"""Precomputed linear or cosine Gaussian diffusion coefficients."""

from __future__ import annotations

import math

import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    """Own immutable diffusion coefficients as device-aware module buffers."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_cumulative: torch.Tensor
    alpha_cumulative_previous: torch.Tensor
    sqrt_alpha_cumulative: torch.Tensor
    sqrt_one_minus_alpha_cumulative: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(
        self,
        timesteps: int = 1000,
        kind: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_offset: float = 0.008,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least two.")
        if kind not in {"linear", "cosine"}:
            raise ValueError("kind must be linear or cosine.")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("Linear beta bounds must satisfy 0 < start < end < 1.")
        if not math.isfinite(cosine_offset) or cosine_offset < 0:
            raise ValueError("cosine_offset must be finite and non-negative.")
        self.timesteps = timesteps
        self.kind = kind
        if kind == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        else:
            points = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
            cumulative = torch.cos(
                ((points / timesteps + cosine_offset) / (1 + cosine_offset)) * math.pi / 2
            ).square()
            cumulative = cumulative / cumulative[0]
            betas = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(1e-8, 0.999)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), cumulative[:-1]])
        posterior_variance = betas * (1.0 - previous) / (1.0 - cumulative)
        self.register_buffer("betas", betas.float(), persistent=True)
        self.register_buffer("alphas", alphas.float(), persistent=True)
        self.register_buffer("alpha_cumulative", cumulative.float(), persistent=True)
        self.register_buffer("alpha_cumulative_previous", previous.float(), persistent=True)
        self.register_buffer("sqrt_alpha_cumulative", cumulative.sqrt().float(), persistent=True)
        self.register_buffer(
            "sqrt_one_minus_alpha_cumulative",
            (1.0 - cumulative).sqrt().float(),
            persistent=True,
        )
        self.register_buffer(
            "posterior_variance", posterior_variance.clamp_min(1e-20).float(), persistent=True
        )

    def extract(
        self,
        values: torch.Tensor,
        timesteps: torch.Tensor,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one scalar per sample and reshape it for broadcasting over ``like``."""
        if timesteps.ndim != 1 or timesteps.shape[0] != like.shape[0]:
            raise ValueError("timesteps must have shape (batch,).")
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise TypeError("timesteps must use an integer dtype.")
        if timesteps.numel() and bool(((timesteps < 0) | (timesteps >= self.timesteps)).any()):
            raise ValueError("timesteps contains an out-of-range index.")
        selected = values.to(device=like.device, dtype=like.dtype)[timesteps.long()]
        return selected.reshape(like.shape[0], *([1] * (like.ndim - 1)))

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw uniform training timesteps on the requested device."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return torch.randint(
            self.timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        )
