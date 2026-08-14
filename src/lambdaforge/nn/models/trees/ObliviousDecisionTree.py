"""Differentiable oblivious decision-tree ensemble for PyTorch."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lambdaforge.nn.activations.sparse import Entmax15, Entmoid15
from lambdaforge.nn.models.Model import Model


class ObliviousDecisionTree(Model):
    r"""One vectorized layer of differentiable oblivious decision trees.

    All nodes at a given depth of one tree share the same learned feature
    selector and threshold. Sparse feature selection and branch probabilities
    default to the entmax/entmoid relaxations used by NODE. Each of
    ``num_trees`` trees stores ``tree_dim`` values in every leaf.

    Parameters are initialized without inspecting data. The optional
    :meth:`initialize_from_data` method performs an explicit deterministic
    quantile initialization; it is never triggered by ``forward`` so DDP ranks
    cannot silently diverge on their first local batch.

    ``max_route_elements_per_sample`` bounds the dominant routing tensor as
    ``num_trees * 2**depth * depth``. Batch size and autograd multiply this
    per-sample estimate.

    Shape
    -----
    With ``flatten_output=True``, input ``(..., in_features)`` maps to
    ``(..., num_trees * tree_dim)``. Otherwise the output is
    ``(..., num_trees, tree_dim)``.
    """

    log_temperatures: Tensor
    leaf_directions: Tensor

    def __init__(
        self,
        in_features: int,
        num_trees: int = 32,
        depth: int = 4,
        tree_dim: int = 1,
        flatten_output: bool = True,
        feature_selector: str | type[nn.Module] | nn.Module = "entmax15",
        feature_selector_kwargs: dict[str, Any] | None = None,
        bin_function: str | type[nn.Module] | nn.Module = "entmoid15",
        bin_function_kwargs: dict[str, Any] | None = None,
        selector_temperature: float = 1.0,
        routing_temperature: float = 1.0,
        learnable_temperature: bool = True,
        hard_feature_selection: bool = False,
        hard_routing: bool = False,
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
        self._validate_dimensions(
            in_features,
            num_trees,
            depth,
            tree_dim,
            max_leaves,
            max_total_leaves,
            max_route_elements_per_sample,
        )
        for name, value in (
            ("selector_temperature", selector_temperature),
            ("routing_temperature", routing_temperature),
        ):
            self._validate_positive(name, value)
        for name, value in (
            ("selector_init_std", selector_init_std),
            ("threshold_init_std", threshold_init_std),
            ("response_init_std", response_init_std),
        ):
            self._validate_non_negative(name, value)
        if not math.isfinite(float(threshold_init_mean)):
            raise ValueError("threshold_init_mean must be finite.")
        if nan_policy not in {"error", "zero"}:
            raise ValueError("nan_policy must be either 'error' or 'zero'.")

        self.in_features = in_features
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.num_leaves = 2**depth
        self.flatten_output = flatten_output
        self.selector_temperature = float(selector_temperature)
        self.learnable_temperature = learnable_temperature
        self.hard_feature_selection = hard_feature_selection
        self.hard_routing = hard_routing
        self.nan_policy = nan_policy

        self.feature_selector = self._build_selector(
            feature_selector,
            feature_selector_kwargs or {},
        )
        self.bin_function = self._build_bin_function(
            bin_function,
            bin_function_kwargs or {},
        )

        self.feature_logits = nn.Parameter(torch.empty(in_features, num_trees, depth))
        self.thresholds = nn.Parameter(torch.empty(num_trees, depth))
        self.responses = nn.Parameter(torch.empty(num_trees, tree_dim, self.num_leaves))
        nn.init.normal_(self.feature_logits, mean=0.0, std=selector_init_std)
        nn.init.normal_(
            self.thresholds,
            mean=float(threshold_init_mean),
            std=threshold_init_std,
        )
        nn.init.normal_(self.responses, mean=0.0, std=response_init_std)

        initial_log_temperature = torch.full(
            (num_trees, depth),
            math.log(float(routing_temperature)),
        )
        if learnable_temperature:
            self.log_temperatures = nn.Parameter(initial_log_temperature)
        else:
            self.register_buffer(
                "log_temperatures",
                initial_log_temperature,
                persistent=True,
            )

        leaf_index = torch.arange(self.num_leaves, dtype=torch.long).unsqueeze(1)
        level = torch.arange(depth, dtype=torch.long).unsqueeze(0)
        directions = torch.remainder(
            torch.div(leaf_index, 2 ** (depth - level - 1), rounding_mode="floor"),
            2,
        )
        self.register_buffer("leaf_directions", directions.to(torch.bool), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        """Return the learned response mixture for every oblivious tree."""
        leading_shape = self._leading_shape(x)
        routes = self._flat_routes(self._prepare_input(x))
        output = torch.einsum("btl,tol->bto", routes, self.responses)
        if self.flatten_output:
            return output.reshape(*leading_shape, self.num_trees * self.tree_dim)
        return output.reshape(*leading_shape, self.num_trees, self.tree_dim)

    def route(self, x: Tensor) -> Tensor:
        """Return leaf probabilities shaped ``(..., num_trees, 2**depth)``."""
        leading_shape = self._leading_shape(x)
        routes = self._flat_routes(self._prepare_input(x))
        return routes.reshape(*leading_shape, self.num_trees, self.num_leaves)

    def feature_importances(self, per_tree: bool = False, hard: bool = False) -> Tensor:
        """Return normalized selector mass in the original input space."""
        with torch.no_grad():
            selectors = self._feature_weights(force_hard=hard).sum(dim=-1).transpose(0, 1)
            if per_tree:
                denominator = selectors.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(selectors.dtype).eps
                )
                return selectors / denominator
            aggregate = selectors.sum(dim=0)
            return aggregate / aggregate.sum().clamp_min(torch.finfo(aggregate.dtype).eps)

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Expose selectors, thresholds, responses and temperatures separately."""
        groups = {
            "selectors": (self.feature_logits, *tuple(self.feature_selector.parameters())),
            "thresholds": (self.thresholds,),
            "leaves": (self.responses,),
        }
        if isinstance(self.log_temperatures, nn.Parameter):
            groups["temperatures"] = (self.log_temperatures,)
        routing = tuple(self.bin_function.parameters())
        if routing:
            groups["routing"] = routing
        return groups

    def initialize_from_data(
        self,
        x: Tensor,
        threshold_quantile: float | Tensor = 0.5,
        temperature_quantile: float = 0.9,
        eps: float = 1e-6,
    ) -> None:
        """Initialize thresholds and routing scales from an explicit data batch.

        The method uses deterministic empirical quantiles and mutates model
        parameters. Call it once before wrapping the model in DDP, or call it
        on one rank and broadcast the resulting ``state_dict``.
        """
        self._validate_probability("temperature_quantile", temperature_quantile)
        self._validate_positive("eps", eps)
        flat_x = self._prepare_input(x)
        if flat_x.shape[0] < 2:
            raise ValueError("initialize_from_data requires at least two samples.")

        with torch.no_grad():
            selectors = self._feature_weights()
            feature_values = torch.einsum("bf,ftd->btd", flat_x, selectors)
            quantiles = torch.as_tensor(
                threshold_quantile,
                dtype=feature_values.dtype,
                device=feature_values.device,
            )
            if quantiles.ndim == 0:
                quantiles = quantiles.expand(self.num_trees, self.depth)
            else:
                try:
                    quantiles = torch.broadcast_to(quantiles, (self.num_trees, self.depth))
                except RuntimeError as error:
                    raise ValueError(
                        "threshold_quantile must broadcast to (num_trees, depth)."
                    ) from error
            if not bool(((quantiles >= 0.0) & (quantiles <= 1.0)).all()):
                raise ValueError("threshold_quantile values must lie in [0, 1].")

            sorted_values = feature_values.sort(dim=0).values
            threshold_index = torch.round(quantiles * (flat_x.shape[0] - 1)).to(torch.long)
            thresholds = sorted_values.gather(0, threshold_index.unsqueeze(0)).squeeze(0)
            deviations = (feature_values - thresholds).abs().sort(dim=0).values
            temperature_index = round(temperature_quantile * (flat_x.shape[0] - 1))
            temperatures = deviations[temperature_index].clamp_min(float(eps))
            self.thresholds.copy_(thresholds)
            self.log_temperatures.copy_(temperatures.log())

    def _flat_routes(self, flat_x: Tensor) -> Tensor:
        selectors = self._feature_weights()
        selected_features = torch.einsum("bf,ftd->btd", flat_x, selectors)
        split_logits = (selected_features - self.thresholds) * torch.exp(-self.log_temperatures)
        branch_right = self.bin_function(split_logits)
        if self.hard_routing:
            hard = torch.round(branch_right)
            branch_right = branch_right + (hard - branch_right).detach()

        directions = self.leaf_directions.to(dtype=branch_right.dtype)
        branch_probabilities = directions * branch_right.unsqueeze(-2) + (1.0 - directions) * (
            1.0 - branch_right.unsqueeze(-2)
        )
        return branch_probabilities.prod(dim=-1)

    def _feature_weights(self, force_hard: bool | None = None) -> Tensor:
        soft = self.feature_selector(self.feature_logits / self.selector_temperature)
        hard_enabled = self.hard_feature_selection if force_hard is None else force_hard
        if not hard_enabled:
            return soft
        indices = soft.argmax(dim=0)
        hard = F.one_hot(indices, num_classes=self.in_features).permute(2, 0, 1).to(soft.dtype)
        return soft + (hard - soft).detach()

    def _prepare_input(self, x: Tensor) -> Tensor:
        self._leading_shape(x)
        if not torch.is_floating_point(x):
            raise TypeError("ObliviousDecisionTree requires floating-point inputs.")
        if not torch.isfinite(x).all():
            if self.nan_policy == "error":
                raise ValueError(
                    "ObliviousDecisionTree received non-finite input with nan_policy='error'."
                )
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.reshape(-1, self.in_features)

    def _leading_shape(self, x: Tensor) -> tuple[int, ...]:
        if x.ndim < 2:
            raise ValueError("ObliviousDecisionTree input must have batch and feature dimensions.")
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
                return Entmax15(dim=0, **kwargs)
            if key == "softmax":
                return nn.Softmax(dim=0, **kwargs)
            raise ValueError("feature_selector must be 'entmax15', 'softmax', a class or a module.")
        return spec(**kwargs)

    @staticmethod
    def _build_bin_function(
        spec: str | type[nn.Module] | nn.Module,
        kwargs: dict[str, Any],
    ) -> nn.Module:
        if isinstance(spec, nn.Module):
            if kwargs:
                raise ValueError("bin_function_kwargs cannot accompany a module instance.")
            return spec
        if isinstance(spec, str):
            key = spec.strip().lower().replace("_", "").replace("-", "")
            if key == "entmoid15":
                return Entmoid15(**kwargs)
            if key == "sigmoid":
                return nn.Sigmoid()
            raise ValueError("bin_function must be 'entmoid15', 'sigmoid', a class or a module.")
        return spec(**kwargs)

    @staticmethod
    def _validate_dimensions(
        in_features: int,
        num_trees: int,
        depth: int,
        tree_dim: int,
        max_leaves: int | None,
        max_total_leaves: int | None,
        max_route_elements_per_sample: int | None,
    ) -> None:
        for name, value in (
            ("in_features", in_features),
            ("num_trees", num_trees),
            ("depth", depth),
            ("tree_dim", tree_dim),
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
            total = leaves * num_trees
            if total > max_total_leaves:
                raise ValueError(
                    f"The layer creates {total} leaf slots, exceeding "
                    f"max_total_leaves={max_total_leaves}."
                )
        ObliviousDecisionTree._validate_route_limit(
            num_trees * leaves * depth,
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
                f"ObliviousDecisionTree routing needs {cost} elements per sample "
                f"(trees * leaves * depth), exceeding "
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

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
