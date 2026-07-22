"""Hierarchical Neural Oblivious Decision Ensemble for PyTorch."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.trees.ObliviousDecisionTree import ObliviousDecisionTree


class NODE(Model):
    r"""Dense hierarchy of differentiable oblivious tree ensembles.

    Each layer contains a vectorized :class:`ObliviousDecisionTree` ensemble.
    With the default ``dense_connections=True``, a layer receives the original
    features concatenated with every preceding tree response, matching NODE's
    DenseNet-like hierarchical representation. Users configure each layer with
    either scalar values or lists of exactly ``num_layers`` values.

    ``readout="mean"`` (default) averages all tree responses and requires each
    tree's ``tree_dim`` to equal ``out_features``. ``"sum"`` uses the same
    architecture without normalization. ``"linear"`` concatenates every tree
    response and learns a final projection, allowing arbitrary per-layer
    ``tree_dim`` values.

    This class is an embeddable NODE architecture. Dataset preprocessing,
    objective-specific losses, optimizers and training schedules remain the
    responsibility of LambdaForge's data and training abstractions.

    When ``max_features`` truncates a dense connection, the newest learned
    responses are retained first and any remaining capacity is filled with
    original features. Thus every layer after the first can depend on an
    earlier learned representation even when the cap is no larger than the
    original input width.

    ``max_route_elements_per_sample`` bounds the sum of routing intermediates
    retained across layers as
    ``sum(num_trees[i] * 2**depth[i] * depth[i])``. Batch size and autograd
    multiply this per-sample estimate.

    Shape
    -----
    Input ``(..., in_features)`` maps to ``(..., out_features)``.
    """

    layers: nn.ModuleList

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 2,
        num_trees: int | list[int] = 64,
        depth: int | list[int] = 4,
        tree_dim: int | list[int] | None = None,
        dense_connections: bool = True,
        max_features: int | None = None,
        input_dropout: float | list[float] = 0.0,
        feature_selector: str | type[nn.Module] | nn.Module | list[Any] = "entmax15",
        feature_selector_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        bin_function: str | type[nn.Module] | nn.Module | list[Any] = "entmoid15",
        bin_function_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        selector_temperature: float | list[float] = 1.0,
        routing_temperature: float | list[float] = 1.0,
        learnable_temperature: bool | list[bool] = True,
        hard_feature_selection: bool | list[bool] = False,
        hard_routing: bool | list[bool] = False,
        readout: str = "mean",
        head_bias: bool = True,
        nan_policy: str = "error",
        selector_init_std: float = 0.02,
        threshold_init_mean: float = 0.0,
        threshold_init_std: float = 0.02,
        response_init_std: float = 0.02,
        max_leaves: int | None = 65_536,
        max_total_leaves: int | None = 1_000_000,
        max_route_elements_per_sample: int | None = 262_144,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("num_layers", num_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None.")
        if readout not in {"mean", "sum", "linear"}:
            raise ValueError("readout must be 'mean', 'sum' or 'linear'.")
        if nan_policy not in {"error", "zero"}:
            raise ValueError("nan_policy must be either 'error' or 'zero'.")

        trees = self._expand(num_trees, num_layers, "num_trees")
        depths = self._expand(depth, num_layers, "depth")
        dimensions = self._expand(
            out_features if tree_dim is None else tree_dim,
            num_layers,
            "tree_dim",
        )
        dropouts = self._expand(input_dropout, num_layers, "input_dropout")
        selectors = self._expand(feature_selector, num_layers, "feature_selector")
        selector_kwargs = self._expand_mapping(
            feature_selector_kwargs,
            num_layers,
            "feature_selector_kwargs",
        )
        bins = self._expand(bin_function, num_layers, "bin_function")
        bin_kwargs = self._expand_mapping(
            bin_function_kwargs,
            num_layers,
            "bin_function_kwargs",
        )
        selector_temperatures = self._expand(
            selector_temperature,
            num_layers,
            "selector_temperature",
        )
        routing_temperatures = self._expand(
            routing_temperature,
            num_layers,
            "routing_temperature",
        )
        learnable_temperatures = self._expand(
            learnable_temperature,
            num_layers,
            "learnable_temperature",
        )
        hard_selections = self._expand(
            hard_feature_selection,
            num_layers,
            "hard_feature_selection",
        )
        hard_routes = self._expand(hard_routing, num_layers, "hard_routing")

        for index, value in enumerate(dropouts):
            if isinstance(value, bool) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"input_dropout[{index}] must be in [0, 1).")
        for name, values in (
            ("num_trees", trees),
            ("depth", depths),
            ("tree_dim", dimensions),
        ):
            for index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name}[{index}] must be a positive integer.")
        if readout in {"mean", "sum"} and any(value != out_features for value in dimensions):
            raise ValueError("mean/sum readout requires every tree_dim to equal out_features.")

        total_leaf_slots = sum(
            tree_count * 2**layer_depth
            for tree_count, layer_depth in zip(trees, depths, strict=True)
        )
        if max_total_leaves is not None:
            if (
                isinstance(max_total_leaves, bool)
                or not isinstance(max_total_leaves, int)
                or max_total_leaves < 2
            ):
                raise ValueError("max_total_leaves must be an integer >= 2 or None.")
            if total_leaf_slots > max_total_leaves:
                raise ValueError(
                    f"NODE creates {total_leaf_slots} leaf slots, exceeding "
                    f"max_total_leaves={max_total_leaves}."
                )
        route_elements = sum(
            tree_count * 2**layer_depth * layer_depth
            for tree_count, layer_depth in zip(trees, depths, strict=True)
        )
        self._validate_route_limit(route_elements, max_route_elements_per_sample)

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_trees = [int(value) for value in trees]
        self.depths = [int(value) for value in depths]
        self.tree_dims = [int(value) for value in dimensions]
        self.dense_connections = dense_connections
        self.max_features = max_features
        self.input_dropout = [float(value) for value in dropouts]
        self.readout = readout
        self.nan_policy = nan_policy

        self.layers = nn.ModuleList()
        current_features = in_features
        total_response_features = 0
        for index in range(num_layers):
            layer_features = min(current_features, max_features or current_features)
            layer = ObliviousDecisionTree(
                in_features=layer_features,
                num_trees=self.num_trees[index],
                depth=self.depths[index],
                tree_dim=self.tree_dims[index],
                flatten_output=False,
                feature_selector=copy.deepcopy(selectors[index]),
                feature_selector_kwargs=dict(selector_kwargs[index]),
                bin_function=copy.deepcopy(bins[index]),
                bin_function_kwargs=dict(bin_kwargs[index]),
                selector_temperature=float(selector_temperatures[index]),
                routing_temperature=float(routing_temperatures[index]),
                learnable_temperature=bool(learnable_temperatures[index]),
                hard_feature_selection=bool(hard_selections[index]),
                hard_routing=bool(hard_routes[index]),
                nan_policy=nan_policy,
                selector_init_std=selector_init_std,
                threshold_init_mean=threshold_init_mean,
                threshold_init_std=threshold_init_std,
                response_init_std=response_init_std,
                max_leaves=max_leaves,
                max_total_leaves=None,
                max_route_elements_per_sample=None,
            )
            self.layers.append(layer)
            response_features = self.num_trees[index] * self.tree_dims[index]
            total_response_features += response_features
            current_features = (
                current_features + response_features if dense_connections else response_features
            )

        self.head: nn.Module
        if readout == "linear":
            self.head = nn.Linear(total_response_features, out_features, bias=head_bias)
        else:
            self.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Return task-agnostic NODE logits for ``x``."""
        leading_shape = self._leading_shape(x)
        outputs, _ = self._forward_layers(x, collect_routes=False)
        if self.readout in {"mean", "sum"}:
            tree_outputs = torch.cat(outputs, dim=-2)
            result = (
                tree_outputs.mean(dim=-2) if self.readout == "mean" else tree_outputs.sum(dim=-2)
            )
        else:
            features = torch.cat([value.flatten(start_dim=-2) for value in outputs], dim=-1)
            result = self.head(features)
        return result.reshape(*leading_shape, self.out_features)

    def features(self, x: Tensor) -> Tensor:
        """Return all layer responses concatenated along the feature axis."""
        outputs, _ = self._forward_layers(x, collect_routes=False)
        return torch.cat([value.flatten(start_dim=-2) for value in outputs], dim=-1)

    def route(self, x: Tensor) -> list[Tensor]:
        """Return one ``(..., trees, leaves)`` routing tensor per layer."""
        _, routes = self._forward_layers(x, collect_routes=True)
        return routes

    def feature_importances(self, per_tree: bool = False, hard: bool = False) -> list[Tensor]:
        """Return selector importances for each layer's effective input space."""
        return [
            cast(ObliviousDecisionTree, layer).feature_importances(
                per_tree=per_tree,
                hard=hard,
            )
            for layer in self.layers
        ]

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Collect named parameter families across all oblivious-tree layers."""
        collected: dict[str, list[nn.Parameter]] = {
            "selectors": [],
            "thresholds": [],
            "leaves": [],
            "temperatures": [],
            "routing": [],
            "head": [],
        }
        for layer_module in self.layers:
            layer = cast(ObliviousDecisionTree, layer_module)
            for name, parameters in layer.parameter_groups().items():
                collected[name].extend(parameters)
        collected["head"].extend(self.head.parameters())
        return {name: tuple(parameters) for name, parameters in collected.items() if parameters}

    def initialize_from_data(
        self,
        x: Tensor,
        threshold_quantile: float | Tensor = 0.5,
        temperature_quantile: float = 0.9,
        eps: float = 1e-6,
    ) -> None:
        """Explicitly initialize every layer from hierarchical activations.

        Invoke before DDP construction, or broadcast the resulting state from
        one rank. The method temporarily disables dropout and restores the
        previous training mode.
        """
        self._leading_shape(x)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                dense_input = self._prepare_input(x)
                original = dense_input
                for layer_module in self.layers:
                    layer = cast(ObliviousDecisionTree, layer_module)
                    layer_input = self._limit_features(dense_input, original)
                    layer.initialize_from_data(
                        layer_input,
                        threshold_quantile=threshold_quantile,
                        temperature_quantile=temperature_quantile,
                        eps=eps,
                    )
                    output = layer(layer_input)
                    flat_output = output.flatten(start_dim=-2)
                    dense_input = (
                        torch.cat([dense_input, flat_output], dim=-1)
                        if self.dense_connections
                        else flat_output
                    )
        finally:
            self.train(was_training)

    def _forward_layers(
        self,
        x: Tensor,
        collect_routes: bool,
    ) -> tuple[list[Tensor], list[Tensor]]:
        dense_input = self._prepare_input(x)
        original = dense_input
        outputs: list[Tensor] = []
        routes: list[Tensor] = []
        for index, layer_module in enumerate(self.layers):
            layer = cast(ObliviousDecisionTree, layer_module)
            layer_input = self._limit_features(dense_input, original)
            if self.training and self.input_dropout[index] > 0.0:
                layer_input = F.dropout(
                    layer_input,
                    p=self.input_dropout[index],
                    training=True,
                )
            output = layer(layer_input)
            outputs.append(output)
            if collect_routes:
                routes.append(layer.route(layer_input))
            flat_output = output.flatten(start_dim=-2)
            dense_input = (
                torch.cat([dense_input, flat_output], dim=-1)
                if self.dense_connections
                else flat_output
            )
        return outputs, routes

    def _limit_features(self, dense_input: Tensor, original: Tensor) -> Tensor:
        if self.max_features is None or dense_input.shape[-1] <= self.max_features:
            return dense_input
        if not self.dense_connections:
            return dense_input[..., -self.max_features :]

        learned = dense_input[..., original.shape[-1] :]
        learned_count = min(learned.shape[-1], self.max_features)
        original_count = self.max_features - learned_count
        if original_count == 0:
            return learned[..., -learned_count:]
        return torch.cat(
            [original[..., :original_count], learned[..., -learned_count:]],
            dim=-1,
        )

    def _prepare_input(self, x: Tensor) -> Tensor:
        self._leading_shape(x)
        if not torch.is_floating_point(x):
            raise TypeError("NODE requires floating-point inputs.")
        if not torch.isfinite(x).all():
            if self.nan_policy == "error":
                raise ValueError("NODE received non-finite input with nan_policy='error'.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    @staticmethod
    def _validate_route_limit(cost: int, limit: int | None) -> None:
        if limit is None:
            return
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_route_elements_per_sample must be a positive integer or None.")
        if cost > limit:
            raise ValueError(
                f"NODE routing needs {cost} elements per sample across its layers "
                f"(sum(trees * leaves * depth)), exceeding "
                f"max_route_elements_per_sample={limit}. Batch size and autograd "
                "multiply this memory cost."
            )

    def _leading_shape(self, x: Tensor) -> tuple[int, ...]:
        if x.ndim < 2:
            raise ValueError("NODE input must have at least batch and feature dimensions.")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, received {x.shape[-1]}.")
        return tuple(x.shape[:-1])

    @staticmethod
    def _expand(value: Any, count: int, name: str) -> list[Any]:
        if isinstance(value, list | tuple):
            if len(value) != count:
                raise ValueError(f"{name} must contain exactly {count} values.")
            return list(value)
        return [value for _ in range(count)]

    @staticmethod
    def _expand_mapping(
        value: dict[str, Any] | list[dict[str, Any]] | None,
        count: int,
        name: str,
    ) -> list[dict[str, Any]]:
        if value is None:
            return [{} for _ in range(count)]
        if isinstance(value, dict):
            return [dict(value) for _ in range(count)]
        if len(value) != count:
            raise ValueError(f"{name} must contain exactly {count} mappings.")
        if any(not isinstance(item, dict) for item in value):
            raise TypeError(f"Every {name} value must be a mapping.")
        return [dict(item) for item in value]
