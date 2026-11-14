"""Portable class/kwargs factories; each call creates independent module state."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.catalog.OperationCatalog import OperationCatalog


class ComponentFactory:
    """A constructor specification, not a shared module or a graph execution object.

    `build(features=...)` supplies the receiving layer's normalization dimension. Normalization
    specs may not prescribe dimensions; only standalone direct normalization construction may.
    Pass `component_type()` and `kwargs` into native constructors supporting class/kwargs pairs.
    Existing MLP/ResidualMLP/CNN2D/graph APIs and their state_dict keys remain unchanged.
    """

    DIMENSIONS = {"features", "num_features", "num_channels", "normalized_shape"}

    def __init__(self, operation: str, kind: str, kwargs: Mapping[str, Any] | None = None) -> None:
        self.operation, self.kind = operation, kind
        self._kwargs = copy.deepcopy(dict(kwargs or {}))
        json_value = self.to_dict()
        import json

        json.dumps(json_value, allow_nan=False)
        if kind == "normalization" and self.DIMENSIONS.intersection(self._kwargs):
            raise ContractError(
                "RECEIVER_DIMENSION", "Normalization dimensions belong to receiver."
            )

    @property
    def kwargs(self) -> dict[str, Any]:
        return copy.deepcopy(self._kwargs)

    def component_type(self) -> type[Any]:
        """Load only when selected; activation aliases retain native registration semantics."""
        if self.kind in {"activation", "normalization"} and ":" not in self.operation:
            from lambdaforge.nn.ComponentRegistry import ComponentRegistry

            key = OperationCatalog._alias(self.operation)
            if key in ComponentRegistry.aliases(self.kind):
                return (
                    ComponentRegistry.resolve_activation(self.operation)
                    if self.kind == "activation"
                    else ComponentRegistry.resolve_normalization(self.operation)
                )
        return OperationCatalog().resolve(self.operation, kind=self.kind)

    def build(self, *, features: int | None = None) -> Any:
        selected = self.component_type()
        if self.kind == "normalization":
            if type(features) is not int or features < 1:
                raise ContractError("RECEIVER_DIMENSION", "Receiving features must be positive.")
            return selected(features, **self.kwargs)
        if features is not None:
            raise ContractError("INVALID_PARAMETERS", "features is only valid for normalization.")
        return selected(**self.kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factory_version": 1,
            "operation": self.operation,
            "kind": self.kind,
            "kwargs": copy.deepcopy(self._kwargs),
            "instance_policy": "new-per-build",
        }

    def provenance(self) -> dict[str, Any]:
        selector = self.operation
        if self.kind in {"activation", "normalization"} and ":" not in selector:
            from lambdaforge.nn.ComponentRegistry import ComponentRegistry

            selector = ComponentRegistry.aliases(self.kind).get(
                OperationCatalog._alias(selector), selector
            )
        return {
            **self.to_dict(),
            "component": OperationCatalog().describe(selector, kind=self.kind),
        }
