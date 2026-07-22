"""Injection boundary for optional S4, Mamba and other state-space modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class StateSpaceAdapter(Model):
    """Normalize layout and output contracts around an injected sequence module.

    No S4/Mamba package is imported by LambdaForge. Install the selected provider
    in the consumer project and inject its module recursively through YAML.
    """

    def __init__(
        self,
        module: nn.Module,
        module_batch_first: bool = True,
        output_key: str | None = None,
        return_state: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("module must be a torch.nn.Module.")
        if output_key is not None and not output_key:
            raise ValueError("output_key cannot be empty.")
        self.module = module
        self.module_batch_first = module_batch_first
        self.output_key = output_key
        self.return_state = return_state

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, Any]:
        """Call the provider and return a batch-first sequence, optionally with state."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, length, features).")
        provider_input = x if self.module_batch_first else x.transpose(0, 1)
        result = (
            self.module(provider_input, **kwargs)
            if state is None
            else self.module(provider_input, state, **kwargs)
        )
        provider_state: Any = None
        if isinstance(result, Mapping):
            key = self.output_key or "sequence"
            if key not in result:
                raise KeyError(f"State-space provider output has no {key!r} tensor.")
            sequence = result[key]
            provider_state = result.get("state")
        elif isinstance(result, tuple):
            if not result:
                raise ValueError("State-space provider returned an empty tuple.")
            sequence = result[0]
            provider_state = result[1] if len(result) > 1 else None
        else:
            sequence = result
        if not torch.is_tensor(sequence) or sequence.ndim != 3:
            raise TypeError("State-space provider must expose a rank-three sequence tensor.")
        sequence = sequence if self.module_batch_first else sequence.transpose(0, 1)
        if sequence.shape[:2] != x.shape[:2]:
            raise ValueError("State-space provider changed batch or sequence length.")
        if self.return_state:
            return {"sequence": sequence, "state": provider_state}
        return sequence
