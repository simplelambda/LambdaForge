"""Mixture-of-experts model with dense evaluation and optional top-k routing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from lambdaforge.nn.models.Model import Model


class MixtureOfExperts(Model):
    """Route each input through configurable expert modules.

    The gate must return logits shaped ``(..., num_experts)``. Every expert
    must return a tensor whose leading dimensions match the gate's leading
    dimensions; remaining dimensions are treated as the expert output. The
    class stacks, routes and combines all experts internally.

    ``top_k=None`` gives dense routing. A positive ``top_k`` retains only the
    highest gate logits per input and uses a straight-through dense gradient,
    including when only one expert is selected. Every expert is still
    evaluated before outputs are mixed: top-k routing changes the mathematical
    mixture, not compute or memory dispatch. :meth:`load_balance_loss` exposes
    a Switch-style auxiliary loss; ``balance_loss_weight`` scales it without
    changing the tensor-only forward contract.
    """

    log_temperature: Tensor

    def __init__(
        self,
        experts: Sequence[nn.Module],
        gate: nn.Module,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        minimum_temperature: float = 1e-4,
        top_k: int | None = None,
        balance_loss_weight: float = 0.0,
        gate_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(experts, Sequence) or isinstance(experts, str | bytes):
            raise TypeError("experts must be a sequence of torch modules.")
        if not experts:
            raise ValueError("experts cannot be empty.")
        if any(not isinstance(expert, nn.Module) for expert in experts):
            raise TypeError("Every expert must be a torch.nn.Module.")
        if not isinstance(gate, nn.Module):
            raise TypeError("gate must be a torch.nn.Module.")
        for name, value in (
            ("temperature", temperature),
            ("minimum_temperature", minimum_temperature),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        for name, value in (
            ("balance_loss_weight", balance_loss_weight),
            ("gate_noise_std", gate_noise_std),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(experts)
        ):
            raise ValueError("top_k must be in [1, num_experts] or None.")

        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = len(experts)
        self.learnable_temperature = learnable_temperature
        self.minimum_temperature = float(minimum_temperature)
        self.top_k = top_k
        self.balance_loss_weight = float(balance_loss_weight)
        self.gate_noise_std = float(gate_noise_std)

        initial = torch.tensor(math.log(float(temperature)))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(initial)
        else:
            self.register_buffer("log_temperature", initial, persistent=True)

    def routing_temperature(self) -> Tensor:
        """Return the positive effective routing temperature."""
        return self.log_temperature.exp().clamp_min(self.minimum_temperature)

    def gate_logits(self, x: Tensor) -> Tensor:
        """Return validated gate logits, including optional training noise."""
        logits = self.gate(x)
        if not isinstance(logits, Tensor):
            raise TypeError("gate must return a Tensor.")
        if logits.ndim < 1 or logits.shape[-1] != self.num_experts:
            raise ValueError(
                f"gate must return (..., {self.num_experts}) logits; got {tuple(logits.shape)}."
            )
        if not torch.is_floating_point(logits):
            raise TypeError("gate logits must use a floating-point dtype.")
        if self.training and self.gate_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.gate_noise_std
        return logits

    def routing_weights_from_logits(self, logits: Tensor) -> Tensor:
        """Convert validated logits into dense or top-k normalized weights."""
        if logits.ndim < 1 or logits.shape[-1] != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} logits in the final dimension.")
        if not torch.is_floating_point(logits):
            raise TypeError("gate logits must use a floating-point dtype.")
        scaled = logits / self.routing_temperature()
        dense = torch.softmax(scaled, dim=-1)
        if self.top_k is not None and self.top_k < self.num_experts:
            _, indices = torch.topk(scaled, self.top_k, dim=-1)
            mask = torch.zeros_like(scaled, dtype=torch.bool).scatter(-1, indices, True)
            sparse = dense * mask.to(dense.dtype)
            sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(sparse.dtype).tiny
            )
            return dense + (sparse - dense).detach()
        return dense

    def routing_weights(self, x: Tensor) -> Tensor:
        """Return normalized routing weights for every input position."""
        return self.routing_weights_from_logits(self.gate_logits(x))

    def forward_experts(self, x: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, ...]:
        """Evaluate every expert without applying gate weights."""
        outputs: list[Tensor] = []
        for index, expert in enumerate(self.experts):
            output = expert(x, *args, **kwargs)
            if not isinstance(output, Tensor):
                raise TypeError(f"Expert {index} must return a Tensor.")
            outputs.append(output)
        return tuple(outputs)

    def load_balance_loss(self, x: Tensor, apply_weight: bool = True) -> Tensor:
        """Return the differentiable expert-usage balancing objective.

        The unscaled loss has a minimum of one under uniform routing. Dense
        routing balances mean probability mass. Top-k routing combines mean
        probability with normalized hard top-k load, following Switch-style
        auxiliary routing objectives.
        """
        logits = self.gate_logits(x)
        dense_probabilities = torch.softmax(logits / self.routing_temperature(), dim=-1)
        routed = self.routing_weights_from_logits(logits)
        importance = dense_probabilities.reshape(-1, self.num_experts).mean(dim=0)
        if self.top_k is None or self.top_k == self.num_experts:
            load = routed.reshape(-1, self.num_experts).mean(dim=0)
        else:
            assignments = (routed > 0).to(routed.dtype).reshape(-1, self.num_experts)
            load = assignments.mean(dim=0) / self.top_k
        loss = self.num_experts * torch.sum(importance * load)
        return loss * self.balance_loss_weight if apply_weight else loss

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Return the gate-weighted sum of all expert predictions."""
        weights = self.routing_weights(x)
        outputs = self.forward_experts(x, *args, **kwargs)
        prefix = tuple(weights.shape[:-1])
        for index, output in enumerate(outputs):
            if tuple(output.shape[: len(prefix)]) != prefix:
                raise ValueError(
                    f"Expert {index} leading shape {tuple(output.shape[: len(prefix)])} "
                    f"does not match gate leading shape {prefix}."
                )
        first_tail = tuple(outputs[0].shape[len(prefix) :])
        if any(tuple(output.shape[len(prefix) :]) != first_tail for output in outputs[1:]):
            raise ValueError("All experts must return identical output shapes.")
        expert_axis = len(prefix)
        stacked = torch.stack(outputs, dim=expert_axis)
        expanded_weights = weights.reshape(
            *prefix,
            self.num_experts,
            *([1] * len(first_tail)),
        ).to(device=stacked.device, dtype=stacked.dtype)
        return (stacked * expanded_weights).sum(dim=expert_axis)
