"""Implementation of the FractionalTopKMeanPooling object."""

from __future__ import annotations

import math

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class FractionalTopKMeanPooling(Pooling):
    r"""Mean of a top fraction of valid set elements.

    ``fraction=0.01`` means "average the top 1% of valid instances", with
    optional lower/upper bounds. This is useful when the number of instances
    per bag varies strongly and a fixed ``k`` is too coarse.
    """

    def __init__(
        self,
        fraction: float = 0.01,
        min_k: int = 1,
        max_k: int | None = None,
        largest: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        if min_k < 1:
            raise ValueError("min_k must be >= 1.")
        if max_k is not None and max_k < min_k:
            raise ValueError("max_k must be >= min_k.")

        self.fraction = float(fraction)
        self.min_k = int(min_k)
        self.max_k = int(max_k) if max_k is not None else None
        self.largest = bool(largest)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"FractionalTopKMeanPooling expects x with shape (B, N, D), got {tuple(x.shape)}."
            )

        valid = (
            torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            if mask is None
            else mask.bool()
        )
        fill_value = float("-inf") if self.largest else float("inf")
        values = x.masked_fill(~valid.unsqueeze(-1), fill_value)

        outputs = []
        for batch_id in range(x.shape[0]):
            count = int(valid[batch_id].sum().item())
            if count == 0:
                outputs.append(x.new_zeros(x.shape[2]))
                continue

            k = max(self.min_k, int(math.ceil(count * self.fraction)))
            if self.max_k is not None:
                k = min(k, self.max_k)
            k = min(k, count)
            outputs.append(values[batch_id].topk(k, dim=0, largest=self.largest).values.mean(dim=0))

        return torch.stack(outputs, dim=0)
