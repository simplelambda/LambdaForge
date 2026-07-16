"""Implementation of the LogSumExpPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class LogSumExpPooling(Pooling):
    r"""Log-sum-exp (smooth max) pooling over the set dimension.

    Computes a smooth, differentiable approximation to max pooling via a
    numerically stable log-sum-exp trick:

    .. math::

        z_d = \frac{1}{\beta}\,
              \log\!\left(
                  \frac{1}{|\mathcal{V}|}
                  \sum_{i \in \mathcal{V}} \exp(\beta\, h_{i,d})
              \right)

    The temperature parameter :math:`\beta` controls the interpolation
    between pooling modes:

    - :math:`\beta \to +\infty` — converges to max pooling.
    - :math:`\beta \to 0^+`    — converges to mean pooling (normalised log).
    - :math:`\beta = 1`        — the standard log-sum-exp (default).

    A numerically stable implementation subtracts the per-dimension maximum
    before exponentiating.  Masked-out positions are excluded by setting
    their pre-exponent value to :math:`-\infty`.

    Parameters
    ----------
    beta : float
        Temperature controlling hardness.  Must be positive.
    normalize : bool
        If true, divide by the number of valid elements inside the logarithm.
        This makes the operator interpolate between mean and max. If false,
        compute the unnormalised log-sum-exp.
    name : str | None
        Optional name used to identify the pooling layer.

    Shape convention
    ----------------
    B:
        Batch size.

    N:
        Number of elements per sample.  For example, vertices, atoms, nodes,
        tokens, or neighbors in a local patch.

    D:
        Feature dimension per element.

    x:
        Shape ``(B, N, D)``.
    mask:
        Shape ``(B, N)``.  Float or bool — ``1`` / ``True`` = valid.
    output:
        Shape ``(B, D)``.
    """

    def __init__(
        self,
        beta: float = 1.0,
        normalize: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        self.beta = beta
        self.normalize = bool(normalize)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scaled = self.beta * x  # (B, N, D)
        count = None

        if mask is not None:
            invalid = ~mask.bool()
            scaled = scaled.masked_fill(invalid.unsqueeze(-1), float("-inf"))
            count = mask.to(dtype=x.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)

        # Numerically stable: subtract max before exp
        max_val = scaled.max(dim=1, keepdim=True).values  # (B, 1, D)
        exp = torch.exp(scaled - max_val)  # (B, N, D)

        if mask is not None:
            exp = exp.masked_fill(invalid.unsqueeze(-1), 0.0)

        exp_sum = exp.sum(dim=1).clamp_min(1e-8)
        if self.normalize:
            if count is None:
                count = x.new_full((x.shape[0], 1), float(x.shape[1]))
            exp_sum = exp_sum / count

        lse = max_val.squeeze(1) + torch.log(exp_sum)  # (B, D)

        return lse / self.beta  # (B, D)
