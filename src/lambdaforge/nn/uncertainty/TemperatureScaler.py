"""Positive scalar temperature calibration for binary or multiclass logits."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class TemperatureScaler(nn.Module):
    """Calibrate logits on held-out data through explicit bounded optimization."""

    def __init__(
        self,
        initial_temperature: float = 1.0,
        minimum_temperature: float = 1e-3,
        maximum_temperature: float = 100.0,
    ) -> None:
        super().__init__()
        if not 0 < minimum_temperature < maximum_temperature:
            raise ValueError("Temperature bounds must satisfy 0 < minimum < maximum.")
        if not math.isfinite(initial_temperature) or not (
            minimum_temperature <= initial_temperature <= maximum_temperature
        ):
            raise ValueError("initial_temperature must be finite and within bounds.")
        self.minimum_temperature = float(minimum_temperature)
        self.maximum_temperature = float(maximum_temperature)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive bounded temperature tensor."""
        return self.log_temperature.exp().clamp(self.minimum_temperature, self.maximum_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale arbitrary-rank logits without changing their shape."""
        if not torch.is_floating_point(logits):
            raise TypeError("logits must use a floating-point dtype.")
        return logits / self.temperature.to(device=logits.device, dtype=logits.dtype)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iterations: int = 50,
        learning_rate: float = 0.1,
    ) -> TemperatureScaler:
        """Fit temperature on held-out logits and return this module."""
        if logits.ndim not in {1, 2} or logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets must share a batch dimension.")
        if max_iterations < 1 or learning_rate <= 0:
            raise ValueError("max_iterations and learning_rate must be positive.")
        detached_logits = logits.detach()
        detached_targets = targets.detach().to(device=logits.device)
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=learning_rate,
            max_iter=max_iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self(detached_logits)
            if scaled.ndim == 1 or scaled.shape[-1] == 1:
                loss = F.binary_cross_entropy_with_logits(
                    scaled.reshape(-1), detached_targets.to(scaled.dtype).reshape(-1)
                )
            else:
                loss = F.cross_entropy(scaled, detached_targets.long().reshape(-1))
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.log_temperature.clamp_(
                math.log(self.minimum_temperature), math.log(self.maximum_temperature)
            )
        return self
