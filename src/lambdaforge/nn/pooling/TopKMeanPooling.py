"""Implementation of the TopKMeanPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class TopKMeanPooling(Pooling):
    r"""Mean of the top-K values along the set dimension.

    This is a parameter-free MIL-friendly pooling operator. For each feature
    channel independently, it keeps the largest ``k`` valid instance values and
    averages them. Compared with max pooling, gradients are spread over several
    high-scoring instances instead of a single winner.

    Parameters
    ----------
    k : int
        Number of valid elements to average. If a sample has fewer valid
        elements than ``k``, all valid elements are used.
    largest : bool
        If ``True`` average the largest values; if ``False`` average the
        smallest values.
    name : str | None
        Optional name used to identify the pooling layer.
    """

    def __init__(
        self,
        k: int = 40,
        largest: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = int(k)
        self.largest = bool(largest)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"TopKMeanPooling expects x with shape (B, N, D), got {tuple(x.shape)}."
            )

        if mask is None:
            k = min(self.k, x.shape[1])
            return x.topk(k, dim=1, largest=self.largest).values.mean(dim=1)

        valid = mask.bool()
        fill_value = float("-inf") if self.largest else float("inf")
        values = x.masked_fill(~valid.unsqueeze(-1), fill_value)

        outputs = []
        for batch_id in range(x.shape[0]):
            count = int(valid[batch_id].sum().item())
            if count == 0:
                outputs.append(x.new_zeros(x.shape[2]))
                continue
            k = min(self.k, count)
            outputs.append(values[batch_id].topk(k, dim=0, largest=self.largest).values.mean(dim=0))

        return torch.stack(outputs, dim=0)
