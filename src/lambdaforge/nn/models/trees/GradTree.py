"""PyTorch neural core inspired by the GradTree architecture."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lambdaforge.nn.activations.sparse import Entmax15
from lambdaforge.nn.models.Model import Model


class GradTree(Model):
    r"""Hard, axis-aligned decision tree trained with straight-through gradients.

    The module implements the dense balanced-tree neural core described by
    GradTree: every internal node learns a feature selector and threshold,
    while the forward pass uses a hard one-hot feature and a hard branch. The
    soft selector and split function supply surrogate gradients during
    backpropagation.

    This is an embeddable PyTorch block, not the complete GradTree estimator.
    It deliberately excludes task-specific preprocessing, losses, optimizers,
    class balancing, early stopping and the paper's training recipe. Outputs
    are generic logits or representations and can be placed inside a larger
    network.

    Parameters
    ----------
    in_features:
        Number of input features in the final tensor dimension.
    out_features:
        Number of values stored at each leaf.
    depth:
        Balanced binary-tree depth. The tree owns ``2**depth`` leaves.
    feature_selector:
        ``"entmax15"`` (paper-style default), ``"softmax"`` or a module/class
        that maps the final dimension to simplex weights.
    feature_selector_kwargs:
        Keyword arguments used when constructing a selector class.
    selector_temperature:
        Positive temperature applied to feature-selection logits.
    split_function:
        ``"softsign"`` (matching the current official core), ``"sigmoid"``
        or a module/class mapping real values to ``[0, 1]``.
    split_function_kwargs:
        Keyword arguments used when constructing a split-function class.
    split_temperature:
        Positive scale for threshold comparisons.
    hard_feature_selection:
        Use one-hot selections in the forward pass with a soft backward pass.
    hard_routing:
        Round split probabilities in the forward pass with a soft backward
        pass. Disable it for a fully soft relaxation.
    nan_policy:
        ``"error"`` rejects NaN and infinite inputs; ``"zero"`` replaces every
        non-finite value with zero.
    selector_init_std, threshold_init_std, leaf_init_std:
        Non-negative standard deviations for trainable tensor initialization.
    max_leaves:
        Safety ceiling for ``2**depth``. Set ``None`` only after explicitly
        accepting the exponential memory and compute cost.
    max_route_elements_per_sample:
        Safety ceiling for the dominant routing tensor per sample,
        ``2**depth * depth``. Batch size and autograd multiply this cost; set
        ``None`` only after sizing the complete training workload explicitly.

    Shape
    -----
    Input ``(..., in_features)`` maps to ``(..., out_features)``.
    """

    path_nodes: Tensor
    path_directions: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 4,
        feature_selector: str | type[nn.Module] | nn.Module = "entmax15",
        feature_selector_kwargs: dict[str, Any] | None = None,
        selector_temperature: float = 1.0,
        split_function: str | type[nn.Module] | nn.Module = "softsign",
        split_function_kwargs: dict[str, Any] | None = None,
        split_temperature: float = 1.0,
        hard_feature_selection: bool = True,
        hard_routing: bool = True,
        nan_policy: str = "error",
        selector_init_std: float = 0.02,
        threshold_init_std: float = 0.02,
        leaf_init_std: float = 0.02,
        max_leaves: int | None = 65_536,
        max_route_elements_per_sample: int | None = 262_144,
    ) -> None:
        super().__init__()
        self._validate_dimensions(
            in_features,
            out_features,
            depth,
            max_leaves,
            max_route_elements_per_sample,
        )
        self._validate_positive("selector_temperature", selector_temperature)
        self._validate_positive("split_temperature", split_temperature)
        for name, value in (
            ("selector_init_std", selector_init_std),
            ("threshold_init_std", threshold_init_std),
            ("leaf_init_std", leaf_init_std),
        ):
            self._validate_non_negative(name, value)
        if nan_policy not in {"error", "zero"}:
            raise ValueError("nan_policy must be either 'error' or 'zero'.")

        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth
        self.num_internal_nodes = 2**depth - 1
        self.num_leaves = 2**depth
        self.selector_temperature = float(selector_temperature)
        self.split_temperature = float(split_temperature)
        self.hard_feature_selection = hard_feature_selection
        self.hard_routing = hard_routing
        self.nan_policy = nan_policy

        self.feature_selector = self._build_selector(
            feature_selector,
            feature_selector_kwargs or {},
        )
        self.split_function = self._build_split_function(
            split_function,
            split_function_kwargs or {},
        )

        self.feature_logits = nn.Parameter(torch.empty(self.num_internal_nodes, in_features))
        self.thresholds = nn.Parameter(torch.empty(self.num_internal_nodes, in_features))
        self.leaf_values = nn.Parameter(torch.empty(self.num_leaves, out_features))
        nn.init.normal_(self.feature_logits, mean=0.0, std=selector_init_std)
        nn.init.normal_(self.thresholds, mean=0.0, std=threshold_init_std)
        nn.init.normal_(self.leaf_values, mean=0.0, std=leaf_init_std)

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
        """Route inputs through the hard tree and return leaf logits."""
        leading_shape = self._leading_shape(x)
        flat_routes = self._flat_routes(self._prepare_input(x))
        output = torch.einsum("bl,lo->bo", flat_routes, self.leaf_values)
        return output.reshape(*leading_shape, self.out_features)

    def route(self, x: Tensor) -> Tensor:
        """Return leaf routing weights shaped ``(..., 2**depth)``."""
        leading_shape = self._leading_shape(x)
        routes = self._flat_routes(self._prepare_input(x))
        return routes.reshape(*leading_shape, self.num_leaves)

    def feature_importances(self, hard: bool = True) -> Tensor:
        """Return normalized structural feature usage across internal nodes."""
        with torch.no_grad():
            selectors = self._feature_weights(force_hard=hard)
            importance = selectors.sum(dim=0)
            return importance / importance.sum().clamp_min(torch.finfo(importance.dtype).eps)

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Separate selectors, thresholds, leaves and custom routing parameters."""
        groups = {
            "selectors": (self.feature_logits, *tuple(self.feature_selector.parameters())),
            "thresholds": (self.thresholds,),
            "leaves": (self.leaf_values,),
        }
        routing = tuple(self.split_function.parameters())
        if routing:
            groups["routing"] = routing
        return groups

    def _flat_routes(self, flat_x: Tensor) -> Tensor:
        selectors = self._feature_weights()
        selected_features = torch.einsum("bf,if->bi", flat_x, selectors)
        selected_thresholds = torch.einsum("if,if->i", self.thresholds, selectors)
        split_logits = (selected_thresholds - selected_features) / self.split_temperature
        split_probabilities = self.split_function(split_logits)
        if isinstance(self.split_function, nn.Softsign):
            split_probabilities = (split_probabilities + 1.0) / 2.0
        if self.hard_routing:
            hard = torch.round(split_probabilities)
            split_probabilities = split_probabilities + (hard - split_probabilities).detach()

        path_probabilities = split_probabilities[:, self.path_nodes]
        directions = self.path_directions.to(dtype=path_probabilities.dtype)
        branch_probabilities = (1.0 - directions) * path_probabilities + directions * (
            1.0 - path_probabilities
        )
        return branch_probabilities.prod(dim=-1)

    def _feature_weights(self, force_hard: bool | None = None) -> Tensor:
        soft = self.feature_selector(self.feature_logits / self.selector_temperature)
        hard_enabled = self.hard_feature_selection if force_hard is None else force_hard
        if not hard_enabled:
            return soft
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=self.in_features).to(soft.dtype)
        return soft + (hard - soft).detach()

    def _prepare_input(self, x: Tensor) -> Tensor:
        self._leading_shape(x)
        if not torch.is_floating_point(x):
            raise TypeError("GradTree requires floating-point inputs.")
        if not torch.isfinite(x).all():
            if self.nan_policy == "error":
                raise ValueError("GradTree received non-finite input with nan_policy='error'.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.reshape(-1, self.in_features)

    def _leading_shape(self, x: Tensor) -> tuple[int, ...]:
        if x.ndim < 2:
            raise ValueError("GradTree input must have at least batch and feature dimensions.")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, received {x.shape[-1]}.")
        return tuple(x.shape[:-1])

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
        max_leaves: int | None,
        max_route_elements_per_sample: int | None,
    ) -> None:
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("depth", depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if max_leaves is not None:
            if isinstance(max_leaves, bool) or not isinstance(max_leaves, int) or max_leaves < 2:
                raise ValueError("max_leaves must be an integer >= 2 or None.")
            if 2**depth > max_leaves:
                raise ValueError(
                    f"depth={depth} creates {2**depth} leaves, exceeding max_leaves={max_leaves}."
                )
        GradTree._validate_route_limit(
            2**depth * depth,
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
                f"GradTree routing needs {cost} elements per sample "
                f"(leaves * depth), exceeding max_route_elements_per_sample={limit}. "
                "Batch size and autograd multiply this memory cost."
            )

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number.")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number.")
