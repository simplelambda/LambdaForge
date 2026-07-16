"""Implementation of the ProbabilityGeMPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class ProbabilityGeMPooling(Pooling):
    r"""Generalized-mean pooling for binary MIL logits.

    The input is interpreted as independent instance logits. The operator maps
    logits to probabilities, applies a generalized mean and returns the bag
    probability as a logit again:

    .. math::

        p_{bag,d} =
        \left(\frac{1}{|\mathcal{V}|}
        \sum_{i \in \mathcal{V}} sigmoid(h_{i,d})^p\right)^{1/p}

    .. math::

        z_d = logit(p_{bag,d})

    ``p=1`` is the mean probability. Larger ``p`` values make the pooling more
    selective without using a hard top-k threshold or the saturating product of
    noisy-OR.

    Parameters
    ----------
    p : float
        Positive generalized-mean exponent.
    eps : float
        Numerical clamp applied to probabilities.
    name : str | None
        Optional name used to identify the pooling layer.
    """

    def __init__(
        self,
        p: float = 4.0,
        eps: float = 1e-6,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if p <= 0:
            raise ValueError("p must be positive.")
        if eps <= 0 or eps >= 0.5:
            raise ValueError("eps must be in (0, 0.5).")
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"ProbabilityGeMPooling expects x with shape (B, N, D), got {tuple(x.shape)}."
            )

        probs = torch.sigmoid(x).clamp(self.eps, 1.0 - self.eps)  # (B, N, D)
        powered = probs.pow(self.p)  # (B, N, D)

        if mask is None:
            mean_powered = powered.mean(dim=1)  # (B, D)
        else:
            valid = mask.to(dtype=x.dtype)  # (B, N)
            numerator = (powered * valid.unsqueeze(-1)).sum(dim=1)
            denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_powered = numerator / denominator  # (B, D)

        bag_prob = mean_powered.clamp_min(self.eps).pow(1.0 / self.p)
        bag_prob = bag_prob.clamp(self.eps, 1.0 - self.eps)
        return torch.logit(bag_prob)  # (B, D)
