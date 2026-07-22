"""Implementation of the Snake object."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lambdaforge.nn.activations.Activation import Activation


class Snake(Activation):
    r"""Periodic activation with an optionally learnable positive frequency.

    Computes ``x + sin(alpha * x)^2 / alpha``. Positivity of ``alpha`` is
    preserved by learning it in logarithmic space.

    Parameters
    ----------
    num_parameters : int
        Number of alpha values. ``1`` shares alpha across all features; larger
        values use one alpha per entry of ``channel_dim``. Default: ``1``.
    alpha : float
        Positive initial frequency. Default: ``1.0``.
    trainable : bool
        Whether alpha is optimized with the containing model. Default: ``True``.
    channel_dim : int
        Dimension associated with per-channel alphas. Default: ``-1``.
    epsilon : float
        Positive numerical lower bound for alpha. Default: ``1e-9``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        num_parameters: int = 1,
        alpha: float = 1.0,
        trainable: bool = True,
        channel_dim: int = -1,
        epsilon: float = 1e-9,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(num_parameters, bool) or not isinstance(num_parameters, int):
            raise TypeError("num_parameters must be an integer")
        if num_parameters <= 0:
            raise ValueError("num_parameters must be greater than zero")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a real number")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("alpha must be finite and greater than zero")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean")
        if isinstance(channel_dim, bool) or not isinstance(channel_dim, int):
            raise TypeError("channel_dim must be an integer")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a real number")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and greater than zero")

        self.num_parameters = num_parameters
        self.initial_alpha = float(alpha)
        self.trainable = trainable
        self.channel_dim = channel_dim
        self.epsilon = float(epsilon)
        initial = torch.full((num_parameters,), math.log(self.initial_alpha))
        if trainable:
            self.log_alpha = nn.Parameter(initial)
        else:
            self.register_buffer("log_alpha", initial)

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.log_alpha.exp().clamp_min(self.epsilon)
        if self.num_parameters == 1:
            effective_alpha = alpha[0]
        else:
            if x.ndim == 0 or not -x.ndim <= self.channel_dim < x.ndim:
                raise ValueError(
                    f"channel_dim={self.channel_dim} is invalid for an input with "
                    f"{x.ndim} dimensions"
                )
            if x.shape[self.channel_dim] != self.num_parameters:
                raise ValueError(
                    "Snake expects channel_dim to have size "
                    f"{self.num_parameters}, got {x.shape[self.channel_dim]}"
                )
            normalized_dim = self.channel_dim % x.ndim
            shape = [1] * x.ndim
            shape[normalized_dim] = self.num_parameters
            effective_alpha = alpha.reshape(shape)
        return x + torch.sin(effective_alpha * x).square() / effective_alpha

    def extra_repr(self) -> str:
        return (
            f"num_parameters={self.num_parameters}, alpha={self.initial_alpha}, "
            f"trainable={self.trainable}, channel_dim={self.channel_dim}, "
            f"epsilon={self.epsilon}, name={self.name!r}"
        )
