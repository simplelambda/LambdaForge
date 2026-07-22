"""Piecewise-linear controlled differential equation sequence model."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class NeuralCDE(Model):
    """Integrate ``dz = vector_field(t,z) dX`` over observed control increments."""

    def __init__(
        self,
        vector_field: nn.Module,
        input_channels: int,
        hidden_channels: int,
        initial_encoder: nn.Module | None = None,
        output: nn.Module | None = None,
        return_trajectory: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(vector_field, nn.Module):
            raise TypeError("vector_field must be a torch.nn.Module.")
        if initial_encoder is not None and not isinstance(initial_encoder, nn.Module):
            raise TypeError("initial_encoder must be a torch.nn.Module or None.")
        if output is not None and not isinstance(output, nn.Module):
            raise TypeError("output must be a torch.nn.Module or None.")
        if min(input_channels, hidden_channels) < 1:
            raise ValueError("input_channels and hidden_channels must be positive.")
        self.vector_field = vector_field
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.initial_encoder = initial_encoder
        self.output = output if output is not None else nn.Identity()
        self.return_trajectory = return_trajectory

    def forward(
        self,
        control: torch.Tensor,
        times: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate a batch-first piecewise-linear control path ``(B,L,C)``."""
        if control.ndim != 3 or control.shape[-1] != self.input_channels:
            raise ValueError("control must have shape (batch, length, input_channels).")
        if times.ndim != 1 or times.shape[0] != control.shape[1]:
            raise ValueError("times must have one value per control position.")
        resolved_times = times.to(device=control.device, dtype=control.dtype)
        if not torch.isfinite(resolved_times).all() or bool(
            (resolved_times[1:] <= resolved_times[:-1]).any()
        ):
            raise ValueError("times must be finite and strictly increasing.")
        state = initial_state
        if state is None:
            if self.initial_encoder is None:
                raise ValueError("initial_state is required when initial_encoder is omitted.")
            state = self.initial_encoder(control[:, 0])
        if not torch.is_tensor(state) or state.shape != (
            control.shape[0],
            self.hidden_channels,
        ):
            raise ValueError("Initial CDE state must have shape (batch, hidden_channels).")
        trajectory = [state]
        for index in range(control.shape[1] - 1):
            increment = control[:, index + 1] - control[:, index]
            field = self.vector_field(resolved_times[index], state)
            expected = (control.shape[0], self.hidden_channels, self.input_channels)
            if not torch.is_tensor(field) or field.shape != expected:
                raise ValueError(
                    "vector_field must return (batch, hidden_channels, input_channels)."
                )
            state = state + torch.einsum("bhc,bc->bh", field, increment)
            trajectory.append(state)
        selected = torch.stack(trajectory, dim=1) if self.return_trajectory else state
        result = self.output(selected)
        if not torch.is_tensor(result):
            raise TypeError("output must return a tensor.")
        return result
