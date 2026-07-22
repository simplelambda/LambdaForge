"""Composable neural ensemble model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from lambdaforge.nn.models.composition.EnsembleReduction import EnsembleReduction
from lambdaforge.nn.models.Model import Model


class EnsembleModel(Model):
    """Execute a collection of models and reduce their tensor predictions.

    Member modules are recursively injectable from YAML. The reduction is a
    type-safe :class:`EnsembleReduction`; strings are accepted only as a YAML
    convenience and normalized during construction. Weighted ensembles use a
    positive softmax distribution that can optionally be learned.

    Parameters
    ----------
    models:
        Non-empty sequence of modules receiving the same forward arguments.
    reduction:
        Strategy used to combine the predictions.
    weights:
        Optional non-negative initial probabilities, one per member.
    learnable_weights:
        Optimize the softmax logits backing ``weights``.
    weight_temperature:
        Positive temperature used by :meth:`model_weights`.
    stack_dimension, concatenate_dimension:
        Output dimensions used by the corresponding structural reductions.
    """

    weight_logits: Tensor

    def __init__(
        self,
        models: Sequence[nn.Module],
        reduction: EnsembleReduction | str = EnsembleReduction.MEAN,
        weights: Sequence[float] | Tensor | None = None,
        learnable_weights: bool = False,
        weight_temperature: float = 1.0,
        stack_dimension: int = 0,
        concatenate_dimension: int = -1,
    ) -> None:
        super().__init__()
        if not isinstance(models, Sequence) or isinstance(models, str | bytes):
            raise TypeError("models must be a sequence of torch modules.")
        if not models:
            raise ValueError("models cannot be empty.")
        if any(not isinstance(model, nn.Module) for model in models):
            raise TypeError("Every ensemble member must be a torch.nn.Module.")
        if (
            isinstance(weight_temperature, bool)
            or not math.isfinite(float(weight_temperature))
            or float(weight_temperature) <= 0
        ):
            raise ValueError("weight_temperature must be a finite positive number.")
        if isinstance(stack_dimension, bool) or not isinstance(stack_dimension, int):
            raise TypeError("stack_dimension must be an integer.")
        if isinstance(concatenate_dimension, bool) or not isinstance(concatenate_dimension, int):
            raise TypeError("concatenate_dimension must be an integer.")

        self.models = nn.ModuleList(models)
        self.reduction = EnsembleReduction.from_value(reduction)
        self.learnable_weights = learnable_weights
        self.weight_temperature = float(weight_temperature)
        self.stack_dimension = stack_dimension
        self.concatenate_dimension = concatenate_dimension

        if weights is None:
            probabilities = torch.full((len(models),), 1.0 / len(models))
        else:
            probabilities = (
                torch.as_tensor(
                    weights,
                    dtype=torch.get_default_dtype(),
                )
                .detach()
                .clone()
            )
            if probabilities.ndim != 1 or probabilities.numel() != len(models):
                raise ValueError("weights must contain exactly one value per model.")
            if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
                raise ValueError("weights must be finite and non-negative.")
            if float(probabilities.sum()) <= 0:
                raise ValueError("At least one ensemble weight must be positive.")
            probabilities = probabilities / probabilities.sum()
        tiny = torch.finfo(probabilities.dtype).tiny
        logits = probabilities.clamp_min(tiny).log()
        if learnable_weights:
            self.weight_logits = nn.Parameter(logits)
        else:
            self.register_buffer("weight_logits", logits, persistent=True)

        if self.reduction is not EnsembleReduction.WEIGHTED_MEAN and (
            weights is not None or learnable_weights
        ):
            raise ValueError("weights and learnable_weights require reduction='weighted_mean'.")

    def model_weights(self) -> Tensor:
        """Return normalized positive member weights."""
        return torch.softmax(self.weight_logits / self.weight_temperature, dim=0)

    def forward_members(self, *args: Any, **kwargs: Any) -> tuple[Tensor, ...]:
        """Return every member prediction without reducing it."""
        outputs: list[Tensor] = []
        for index, model in enumerate(self.models):
            output = model(*args, **kwargs)
            if not isinstance(output, Tensor):
                raise TypeError(f"Ensemble member {index} must return a Tensor.")
            outputs.append(output)
        return tuple(outputs)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Run all members and apply the configured reduction."""
        outputs = self.forward_members(*args, **kwargs)
        if self.reduction is EnsembleReduction.CONCATENATE:
            return torch.cat(outputs, dim=self.concatenate_dimension)
        if self.reduction is EnsembleReduction.STACK:
            return torch.stack(outputs, dim=self.stack_dimension)

        stacked = torch.stack(outputs, dim=0)
        if self.reduction is EnsembleReduction.MEAN:
            return stacked.mean(dim=0)
        if self.reduction is EnsembleReduction.SUM:
            return stacked.sum(dim=0)
        if self.reduction is EnsembleReduction.MEDIAN:
            return stacked.median(dim=0).values
        if self.reduction is EnsembleReduction.MIN:
            return stacked.amin(dim=0)
        if self.reduction is EnsembleReduction.MAX:
            return stacked.amax(dim=0)
        weights = (
            self.model_weights()
            .reshape(
                len(self.models),
                *([1] * (stacked.ndim - 1)),
            )
            .to(device=stacked.device, dtype=stacked.dtype)
        )
        return (stacked * weights).sum(dim=0)
