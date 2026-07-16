"""Implementation of the NoisyOrPooling object."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lambdaforge.nn.pooling.Pooling import Pooling


class NoisyOrPooling(Pooling):
    r"""Noisy-OR pooling for instance logits.

    Treats each element as an independent Bernoulli logit and returns the bag
    logit corresponding to ``1 - prod_i(1 - sigmoid(x_i))``. This is useful for
    MIL baselines, but it can saturate when bags contain many instances.
    """

    def __init__(
        self,
        eps: float = 1e-7,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"NoisyOrPooling expects x with shape (B, N, D), got {tuple(x.shape)}."
            )

        log_p_none = F.logsigmoid(-x)
        if mask is not None:
            log_p_none = log_p_none.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        log_p_none = log_p_none.sum(dim=1)
        p_none = log_p_none.exp().clamp(min=self.eps, max=1.0 - self.eps)
        return torch.log1p(-p_none) - torch.log(p_none)
