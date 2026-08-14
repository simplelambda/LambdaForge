"""PyTorch neural core inspired by the GRANDE tree ensemble."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lambdaforge.nn.activations.sparse import Entmax15
from lambdaforge.nn.models.Model import Model


class GRANDE(Model):
    r"""Hard axis-aligned tree ensemble with instance-wise leaf weighting.

    Every estimator is a balanced non-oblivious tree. Feature selection and
    routing are hard during the forward pass and differentiable through
    straight-through estimators. A learned scalar at every leaf determines a
    sample-dependent softmax weight over estimators, which is the central
    ensemble mechanism proposed by GRANDE.

    This class is the reusable neural core, not the complete published GRANDE
    estimator. It excludes dataframe preprocessing, categorical/numerical
    encoders, objective-specific losses, row bootstrap, schedulers, SWA and
    early stopping. It exposes named optimizer parameter groups through the
    standard :meth:``Model.parameter_groups`` contract. It consumes numeric tensors and
    returns task-agnostic logits suitable for composition in larger networks.

    ``selected_features`` may be an integer count or a fraction in ``(0, 1]``.
    Feature subsets are created with a private CPU generator seeded by
    ``feature_seed``; construction never resets PyTorch's global RNG.

    ``max_route_elements_per_sample`` bounds the dominant routing tensor as
    ``num_estimators * 2**depth * depth``. Batch size and autograd multiply
    this per-sample estimate.

    Shape
    -----
    Input ``(..., in_features)`` maps to ``(..., out_features)``.
    """

    feature_indices: Tensor
    path_nodes: Tensor
    path_directions: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 5,
        num_estimators: int = 128,
        selected_features: int | float | None = 1.0,
        feature_seed: int = 0,
        feature_selector: str | type[nn.Module] | nn.Module = "softmax",
        feature_selector_kwargs: dict[str, Any] | None = None,
        selector_temperature: float = 1.0,
        split_function: str | type[nn.Module] | nn.Module = "softsign",
        split_function_kwargs: dict[str, Any] | None = None,
        split_temperature: float = 1.0,
        hard_feature_selection: bool = True,
        hard_routing: bool = True,
        estimator_weight_temperature: float = 1.0,
        estimator_dropout: float = 0.0,
        nan_policy: str = "error",
        selector_init_std: float = 0.05,
        threshold_init_std: float = 0.05,
        leaf_init_std: float = 0.05,
        weight_init_std: float = 0.05,
        max_leaves: int | None = 65_536,
        max_total_leaves: int | None = 1_000_000,
        max_route_elements_per_sample: int | None = 262_144,
    ) -> None:
        super().__init__()
        self._validate_dimensions(
            in_features,
            out_features,
            depth,
            num_estimators,
            max_leaves,
            max_total_leaves,
            max_route_elements_per_sample,
        )
        for name, value in (
            ("selector_temperature", selector_temperature),
            ("split_temperature", split_temperature),
            ("estimator_weight_temperature", estimator_weight_temperature),
        ):
            self._validate_positive(name, value)
        for name, value in (
            ("selector_init_std", selector_init_std),
            ("threshold_init_std", threshold_init_std),
            ("leaf_init_std", leaf_init_std),
            ("weight_init_std", weight_init_std),
        ):
            self._validate_non_negative(name, value)
        if isinstance(estimator_dropout, bool) or not 0.0 <= float(estimator_dropout) < 1.0:
            raise ValueError("estimator_dropout must be in [0, 1).")
        if isinstance(feature_seed, bool) or not isinstance(feature_seed, int):
            raise ValueError("feature_seed must be an integer.")
        if nan_policy not in {"error", "zero"}:
            raise ValueError("nan_policy must be either 'error' or 'zero'.")

        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth
        self.num_estimators = num_estimators
        self.num_internal_nodes = 2**depth - 1
        self.num_leaves = 2**depth
        self.num_selected_features = self._selected_feature_count(
            selected_features,
            in_features,
        )
        self.selector_temperature = float(selector_temperature)
        self.split_temperature = float(split_temperature)
        self.hard_feature_selection = hard_feature_selection
        self.hard_routing = hard_routing
        self.estimator_weight_temperature = float(estimator_weight_temperature)
        self.estimator_dropout = float(estimator_dropout)
        self.nan_policy = nan_policy

        self.feature_selector = self._build_selector(
            feature_selector,
            feature_selector_kwargs or {},
        )
        self.split_function = self._build_split_function(
            split_function,
            split_function_kwargs or {},
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(feature_seed)
        if self.num_selected_features == in_features:
            feature_indices = torch.arange(in_features, dtype=torch.long).repeat(
                num_estimators,
                1,
            )
        else:
            feature_indices = torch.stack(
                [
                    torch.randperm(in_features, generator=generator)[: self.num_selected_features]
                    for _ in range(num_estimators)
                ]
            )
        self.register_buffer("feature_indices", feature_indices, persistent=True)

        parameter_shape = (
            num_estimators,
            self.num_internal_nodes,
            self.num_selected_features,
        )
        self.feature_logits = nn.Parameter(torch.empty(parameter_shape))
        self.thresholds = nn.Parameter(torch.empty(parameter_shape))
        self.leaf_values = nn.Parameter(torch.empty(num_estimators, self.num_leaves, out_features))
        self.leaf_estimator_logits = nn.Parameter(torch.empty(num_estimators, self.num_leaves))
        nn.init.normal_(self.feature_logits, mean=0.0, std=selector_init_std)
        nn.init.normal_(self.thresholds, mean=0.0, std=threshold_init_std)
        nn.init.normal_(self.leaf_values, mean=0.0, std=leaf_init_std)
        nn.init.normal_(self.leaf_estimator_logits, mean=0.0, std=weight_init_std)

        leaf_index = torch.arange(self.num_leaves, dtype=torch.long).unsqueeze(1)
        level = torch.arange(1, depth + 1, dtype=torch.long).unsqueeze(0)
        nodes = (
            2 ** (level - 1)
            + torch.div(
                leaf_index,
                2 ** (depth - (level - 1)),
                rounding_mode="floor",
            )
            - 1
        )
        directions = torch.remainder(
            torch.div(leaf_index, 2 ** (depth - level), rounding_mode="floor"),
            2,
        )
        self.register_buffer("path_nodes", nodes, persistent=True)
        self.register_buffer("path_directions", directions.to(torch.bool), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        """Return the instance-weighted sum of per-estimator logits."""
        leading_shape = self._leading_shape(x)
        flat_x = self._prepare_input(x)
        routes = self._flat_routes(flat_x)
        per_estimator = torch.einsum("bel,elo->beo", routes, self.leaf_values)
        weights = self._weights_from_routes(routes)
        output = torch.einsum("be,beo->bo", weights, per_estimator)
        return output.reshape(*leading_shape, self.out_features)

    def route(self, x: Tensor) -> Tensor:
        """Return routing weights shaped ``(..., estimators, leaves)``."""
        leading_shape = self._leading_shape(x)
        routes = self._flat_routes(self._prepare_input(x))
        return routes.reshape(*leading_shape, self.num_estimators, self.num_leaves)

    def estimator_weights(self, x: Tensor) -> Tensor:
        """Return sample-dependent normalized estimator weights."""
        leading_shape = self._leading_shape(x)
        routes = self._flat_routes(self._prepare_input(x))
        weights = self._weights_from_routes(routes)
        return weights.reshape(*leading_shape, self.num_estimators)

    def forward_estimators(self, x: Tensor) -> Tensor:
        """Return uncombined tree logits shaped ``(..., estimators, outputs)``."""
        leading_shape = self._leading_shape(x)
        routes = self._flat_routes(self._prepare_input(x))
        outputs = torch.einsum("bel,elo->beo", routes, self.leaf_values)
        return outputs.reshape(*leading_shape, self.num_estimators, self.out_features)

    def feature_importances(self, per_estimator: bool = False, hard: bool = True) -> Tensor:
        """Return normalized structural feature usage in original feature space."""
        with torch.no_grad():
            selectors = self._feature_weights(force_hard=hard).sum(dim=1)
            importance = torch.zeros(
                self.num_estimators,
                self.in_features,
                dtype=selectors.dtype,
                device=selectors.device,
            )
            importance.scatter_add_(1, self.feature_indices, selectors)
            if per_estimator:
                denominator = importance.sum(dim=1, keepdim=True).clamp_min(
                    torch.finfo(importance.dtype).eps
                )
                return importance / denominator
            aggregate = importance.sum(dim=0)
            return aggregate / aggregate.sum().clamp_min(torch.finfo(aggregate.dtype).eps)

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Expose parameter families used by GRANDE-style optimizer recipes."""
        groups = {
            "selectors": (self.feature_logits, *tuple(self.feature_selector.parameters())),
            "thresholds": (self.thresholds,),
            "leaves": (self.leaf_values,),
            "estimator_weights": (self.leaf_estimator_logits,),
        }
        routing = tuple(self.split_function.parameters())
        if routing:
            groups["routing"] = routing
        return groups

    def _flat_routes(self, flat_x: Tensor) -> Tensor:
        estimator_x = flat_x[:, self.feature_indices]
        selectors = self._feature_weights()
        selected_features = torch.einsum("bes,eis->bei", estimator_x, selectors)
        selected_thresholds = torch.einsum("eis,eis->ei", self.thresholds, selectors)
        split_logits = (selected_thresholds - selected_features) / self.split_temperature
        split_probabilities = self.split_function(split_logits)
        if isinstance(self.split_function, nn.Softsign):
            split_probabilities = (split_probabilities + 1.0) / 2.0
        if self.hard_routing:
            hard = torch.round(split_probabilities)
            split_probabilities = split_probabilities + (hard - split_probabilities).detach()

        path_probabilities = split_probabilities[:, :, self.path_nodes]
        directions = self.path_directions.to(dtype=path_probabilities.dtype)
        branch_probabilities = (1.0 - directions) * path_probabilities + directions * (
            1.0 - path_probabilities
        )
        return branch_probabilities.prod(dim=-1)

    def _weights_from_routes(self, routes: Tensor) -> Tensor:
        logits = torch.einsum("bel,el->be", routes, self.leaf_estimator_logits)
        weights = torch.softmax(logits / self.estimator_weight_temperature, dim=-1)
        if self.training and self.estimator_dropout > 0.0:
            dropped = F.dropout(weights, p=self.estimator_dropout, training=True)
            denominator = dropped.sum(dim=-1, keepdim=True)
            normalized = dropped / denominator.clamp_min(torch.finfo(dropped.dtype).eps)
            weights = torch.where(denominator > 0.0, normalized, weights)
        return weights

    def _feature_weights(self, force_hard: bool | None = None) -> Tensor:
        soft = self.feature_selector(self.feature_logits / self.selector_temperature)
        hard_enabled = self.hard_feature_selection if force_hard is None else force_hard
        if not hard_enabled:
            return soft
        hard = F.one_hot(
            soft.argmax(dim=-1),
            num_classes=self.num_selected_features,
        ).to(soft.dtype)
        return soft + (hard - soft).detach()

    def _prepare_input(self, x: Tensor) -> Tensor:
        self._leading_shape(x)
        if not torch.is_floating_point(x):
            raise TypeError("GRANDE requires floating-point inputs.")
        if not torch.isfinite(x).all():
            if self.nan_policy == "error":
                raise ValueError("GRANDE received non-finite input with nan_policy='error'.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.reshape(-1, self.in_features)

    def _leading_shape(self, x: Tensor) -> tuple[int, ...]:
        if x.ndim < 2:
            raise ValueError("GRANDE input must have at least batch and feature dimensions.")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, received {x.shape[-1]}.")
        return tuple(x.shape[:-1])

    @staticmethod
    def _selected_feature_count(spec: int | float | None, in_features: int) -> int:
        if spec is None:
            return in_features
        if isinstance(spec, bool):
            raise ValueError("selected_features cannot be boolean.")
        if isinstance(spec, int):
            if not 1 <= spec <= in_features:
                raise ValueError("Integer selected_features must be in [1, in_features].")
            return spec
        if isinstance(spec, float):
            if not math.isfinite(spec) or not 0.0 < spec <= 1.0:
                raise ValueError("Fractional selected_features must be in (0, 1].")
            return max(1, math.ceil(in_features * spec))
        raise TypeError("selected_features must be an int, float or None.")

    @staticmethod
    def _build_selector(
        spec: str | type[nn.Module] | nn.Module,
        kwargs: dict[str, Any],
    ) -> nn.Module:
        if isinstance(spec, nn.Module):
            if kwargs:
                raise ValueError("feature_selector_kwargs cannot accompany a module instance.")
            return spec
        if isinstance(spec, str):
            key = spec.strip().lower().replace("_", "").replace("-", "")
            if key == "entmax15":
                return Entmax15(dim=-1, **kwargs)
            if key == "softmax":
                return nn.Softmax(dim=-1, **kwargs)
            raise ValueError("feature_selector must be 'entmax15', 'softmax', a class or a module.")
        return spec(**kwargs)

    @staticmethod
    def _build_split_function(
        spec: str | type[nn.Module] | nn.Module,
        kwargs: dict[str, Any],
    ) -> nn.Module:
        if isinstance(spec, nn.Module):
            if kwargs:
                raise ValueError("split_function_kwargs cannot accompany a module instance.")
            return spec
        if isinstance(spec, str):
            key = spec.strip().lower().replace("_", "").replace("-", "")
            if key == "softsign":
                return nn.Softsign(**kwargs)
            if key == "sigmoid":
                return nn.Sigmoid()
            raise ValueError("split_function must be 'softsign', 'sigmoid', a class or a module.")
        return spec(**kwargs)

    @staticmethod
    def _validate_dimensions(
        in_features: int,
        out_features: int,
        depth: int,
        num_estimators: int,
        max_leaves: int | None,
        max_total_leaves: int | None,
        max_route_elements_per_sample: int | None,
    ) -> None:
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("depth", depth),
            ("num_estimators", num_estimators),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        leaves = 2**depth
        if max_leaves is not None:
            if isinstance(max_leaves, bool) or not isinstance(max_leaves, int) or max_leaves < 2:
                raise ValueError("max_leaves must be an integer >= 2 or None.")
            if leaves > max_leaves:
                raise ValueError(
                    f"depth={depth} creates {leaves} leaves, exceeding max_leaves={max_leaves}."
                )
        if max_total_leaves is not None:
            if (
                isinstance(max_total_leaves, bool)
                or not isinstance(max_total_leaves, int)
                or max_total_leaves < 2
            ):
                raise ValueError("max_total_leaves must be an integer >= 2 or None.")
            total = leaves * num_estimators
            if total > max_total_leaves:
                raise ValueError(
                    f"The ensemble creates {total} leaf slots, exceeding "
                    f"max_total_leaves={max_total_leaves}."
                )
        GRANDE._validate_route_limit(
            num_estimators * leaves * depth,
            max_route_elements_per_sample,
        )

    @staticmethod
    def _validate_route_limit(cost: int, limit: int | None) -> None:
        if limit is None:
            return
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_route_elements_per_sample must be a positive integer or None.")
        if cost > limit:
            raise ValueError(
                f"GRANDE routing needs {cost} elements per sample "
                f"(estimators * leaves * depth), exceeding "
                f"max_route_elements_per_sample={limit}. Batch size and autograd "
                "multiply this memory cost."
            )

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number.")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number.")
