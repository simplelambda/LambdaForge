"""Concatenated masked mean and max pooling."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class ConcatMeanMaxPooling(Pooling):
    """Concatenate mean and max set summaries into shape ``(B, 2*D)``.

    Fully masked samples return zeros for both summaries.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("ConcatMeanMaxPooling expects non-empty x shaped (B, N, D).")
        if mask is None:
            return torch.cat((x.mean(dim=1), x.max(dim=1).values), dim=-1)
        if tuple(mask.shape) != tuple(x.shape[:2]):
            raise ValueError("mask must have shape (B, N).")
        valid = mask.bool()
        count = valid.sum(dim=1, keepdim=True)
        mean = (x * valid.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
        maximum = x.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values
        has_values = count > 0
        mean = torch.where(has_values, mean, torch.zeros_like(mean))
        maximum = torch.where(has_values, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)
