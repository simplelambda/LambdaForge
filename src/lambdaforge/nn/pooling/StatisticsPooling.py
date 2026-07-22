"""Configurable masked statistical pooling."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class StatisticsPooling(Pooling):
    """Concatenate selected mean, standard deviation, minimum and maximum.

    Statistics are emitted in that fixed order. ``unbiased=True`` uses the
    sample standard deviation and returns zero when fewer than two values are
    available. Fully masked samples return zero for every selected statistic.
    """

    def __init__(
        self,
        include_mean: bool = True,
        include_std: bool = True,
        include_min: bool = False,
        include_max: bool = False,
        unbiased: bool = False,
        eps: float = 1e-8,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not any((include_mean, include_std, include_min, include_max)):
            raise ValueError("At least one statistic must be enabled.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.include_mean = bool(include_mean)
        self.include_std = bool(include_std)
        self.include_min = bool(include_min)
        self.include_max = bool(include_max)
        self.unbiased = bool(unbiased)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("StatisticsPooling expects non-empty x shaped (B, N, D).")
        if mask is not None and tuple(mask.shape) != tuple(x.shape[:2]):
            raise ValueError("mask must have shape (B, N).")
        valid = (
            torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            if mask is None
            else mask.bool()
        )
        count = valid.sum(dim=1, keepdim=True)
        has_values = count > 0
        weighted = x * valid.unsqueeze(-1)
        mean = weighted.sum(dim=1) / count.clamp_min(1)
        outputs: list[torch.Tensor] = []
        if self.include_mean:
            outputs.append(torch.where(has_values, mean, torch.zeros_like(mean)))
        if self.include_std:
            correction = 1 if self.unbiased else 0
            denominator = (count - correction).clamp_min(1)
            squared = ((x - mean.unsqueeze(1)) * valid.unsqueeze(-1)).square().sum(dim=1)
            std = torch.sqrt((squared / denominator).clamp_min(0.0) + self.eps)
            sufficient = count > correction
            outputs.append(torch.where(sufficient, std, torch.zeros_like(std)))
        if self.include_min:
            minimum = x.masked_fill(~valid.unsqueeze(-1), float("inf")).min(dim=1).values
            outputs.append(torch.where(has_values, minimum, torch.zeros_like(minimum)))
        if self.include_max:
            maximum = x.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values
            outputs.append(torch.where(has_values, maximum, torch.zeros_like(maximum)))
        return torch.cat(outputs, dim=-1)
