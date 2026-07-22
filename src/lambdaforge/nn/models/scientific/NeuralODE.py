"""Differentiable fixed-step integration around an injected neural vector field."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.scientific.ODEMethod import ODEMethod


class NeuralODE(Model):
    """Integrate ``dy/dt = dynamics(t, y)`` without an optional solver package."""

    def __init__(
        self,
        dynamics: nn.Module,
        method: ODEMethod | str = ODEMethod.RK4,
        steps_per_interval: int = 1,
        return_trajectory: bool = True,
        time_dimension: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(dynamics, nn.Module):
            raise TypeError("dynamics must be a torch.nn.Module.")
        if steps_per_interval < 1:
            raise ValueError("steps_per_interval must be positive.")
        if isinstance(time_dimension, bool) or not isinstance(time_dimension, int):
            raise TypeError("time_dimension must be an integer.")
        self.dynamics = dynamics
        self.method = ODEMethod(method)
        self.steps_per_interval = steps_per_interval
        self.return_trajectory = return_trajectory
        self.time_dimension = time_dimension

    def forward(self, initial_state: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Integrate over strictly increasing one-dimensional evaluation times."""
        if not torch.is_floating_point(initial_state):
            raise TypeError("initial_state must use a floating-point dtype.")
        if times.ndim != 1 or times.numel() < 2:
            raise ValueError("times must be one-dimensional with at least two values.")
        resolved_times = times.to(device=initial_state.device, dtype=initial_state.dtype)
        if not torch.isfinite(resolved_times).all() or bool(
            (resolved_times[1:] <= resolved_times[:-1]).any()
        ):
            raise ValueError("times must be finite and strictly increasing.")
        state = initial_state
        trajectory = [state]
        for start, end in zip(resolved_times[:-1], resolved_times[1:], strict=True):
            step = (end - start) / self.steps_per_interval
            current_time = start
            for _ in range(self.steps_per_interval):
                state = self._step(current_time, state, step)
                current_time = current_time + step
            trajectory.append(state)
        if not self.return_trajectory:
            return state
        dimension = self.time_dimension
        if dimension < 0:
            dimension += initial_state.ndim + 1
        if not 0 <= dimension <= initial_state.ndim:
            raise ValueError("time_dimension is invalid for the state rank.")
        return torch.stack(trajectory, dim=dimension)

    def _step(self, time: torch.Tensor, state: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        if self.method is ODEMethod.EULER:
            return state + step * self._derivative(time, state)
        if self.method is ODEMethod.MIDPOINT:
            first = self._derivative(time, state)
            return state + step * self._derivative(time + 0.5 * step, state + 0.5 * step * first)
        first = self._derivative(time, state)
        second = self._derivative(time + 0.5 * step, state + 0.5 * step * first)
        third = self._derivative(time + 0.5 * step, state + 0.5 * step * second)
        fourth = self._derivative(time + step, state + step * third)
        return state + step * (first + 2 * second + 2 * third + fourth) / 6

    def _derivative(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        derivative = self.dynamics(time, state)
        if not torch.is_tensor(derivative) or derivative.shape != state.shape:
            raise ValueError("dynamics must return a tensor with the state shape.")
        return derivative
