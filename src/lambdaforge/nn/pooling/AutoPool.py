"""Implementation of the AutoPool object."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lambdaforge.nn.pooling.Pooling import Pooling


class AutoPool(Pooling):
    r"""Adaptive softmax pooling over set elements.

    This follows the spirit of auto-pool operators for weakly labelled MIL:
    weights are a softmax over ``alpha * x`` and ``alpha`` is learned jointly
    with the model. Small ``alpha`` behaves close to mean pooling; large
    ``alpha`` approaches max-like pooling.

    Parameters
    ----------
    init_alpha : float
        Initial non-negative pooling sharpness.
    learnable : bool
        If true, ``alpha`` is learned.
    per_feature : bool
        If true, learn one ``alpha`` per feature channel. This requires
        ``in_features``.
    in_features : int | None
        Feature dimension used when ``per_feature=True``.
    max_alpha : float | None
        Optional upper clamp for numerical stability.
    """

    def __init__(
        self,
        init_alpha: float = 1.0,
        learnable: bool = True,
        per_feature: bool = False,
        in_features: int | None = None,
        max_alpha: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if init_alpha < 0:
            raise ValueError("init_alpha must be non-negative.")
        if per_feature and (in_features is None or in_features < 1):
            raise ValueError("in_features must be provided when per_feature=True.")
        if max_alpha is not None and max_alpha <= 0:
            raise ValueError("max_alpha must be positive.")

        if per_feature:
            assert in_features is not None  # Narrowed by the validation above.
            shape = (in_features,)
        else:
            shape = (1,)
        raw = torch.full(shape, self._inverse_softplus(float(init_alpha)))
        if learnable:
            self.raw_alpha = torch.nn.Parameter(raw)
        else:
            self.register_buffer("raw_alpha", raw)
        self.per_feature = bool(per_feature)
        self.max_alpha = max_alpha

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"AutoPool expects x with shape (B, N, D), got {tuple(x.shape)}.")

        alpha = F.softplus(self.raw_alpha).view(1, 1, -1)
        if self.max_alpha is not None:
            alpha = alpha.clamp(max=self.max_alpha)

        scores = alpha * x
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))

        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return (weights * x).sum(dim=1)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value == 0:
            return -20.0
        return float(torch.log(torch.expm1(torch.tensor(value))).item())
