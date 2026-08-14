"""Registry for YAML-friendly neural-network component names."""

from __future__ import annotations

from typing import Any

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.periodic import Sine, Snake
from lambdaforge.nn.activations.rectifiers import CELU, ELU, SELU, LeakyReLU, PReLU, ReLU, ReLU6
from lambdaforge.nn.activations.smooth import (
    GELU,
    Hardsigmoid,
    Hardswish,
    Identity,
    Mish,
    Sigmoid,
    SiLU,
    Softplus,
    Softsign,
    SquarePlus,
    Tanh,
)
from lambdaforge.nn.activations.sparse import Entmax15, Entmoid15
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.ChannelLayerNorm import ChannelLayerNorm
from lambdaforge.nn.normalizations.GroupNorm import GroupNorm
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.InstanceNorm import InstanceNorm
from lambdaforge.nn.normalizations.L2Norm import L2Norm
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization
from lambdaforge.nn.normalizations.RMSNorm import RMSNorm
from lambdaforge.nn.normalizations.ScaleNorm import ScaleNorm
from lambdaforge.plugins.PluginKind import PluginKind
from lambdaforge.plugins.PluginReference import PluginReference
from lambdaforge.plugins.PluginRegistry import PluginRegistry


class ComponentRegistry:
    """Resolve short, case-insensitive component names to framework classes.

    The registry is deliberately small and explicit. YAML files may use names
    such as ``relu`` or ``layernorm``, while Python callers remain free to pass
    any compatible class directly. Custom aliases can be registered without
    modifying model implementations.
    """

    _activations: dict[str, type[Activation]] = {
        "celu": CELU,
        "elu": ELU,
        "entmax15": Entmax15,
        "entmoid15": Entmoid15,
        "gelu": GELU,
        "hardsigmoid": Hardsigmoid,
        "hardswish": Hardswish,
        "identity": Identity,
        "leakyrelu": LeakyReLU,
        "mish": Mish,
        "prelu": PReLU,
        "relu": ReLU,
        "relu6": ReLU6,
        "selu": SELU,
        "sigmoid": Sigmoid,
        "sine": Sine,
        "silu": SiLU,
        "snake": Snake,
        "softplus": Softplus,
        "softsign": Softsign,
        "squareplus": SquarePlus,
        "tanh": Tanh,
    }
    _normalizations: dict[str, type[Normalization]] = {
        "batchnorm": BatchNorm,
        "channellayernorm": ChannelLayerNorm,
        "groupnorm": GroupNorm,
        "identity": IdentityNorm,
        "instancenorm": InstanceNorm,
        "l2norm": L2Norm,
        "layernorm": LayerNorm,
        "none": IdentityNorm,
        "rmsnorm": RMSNorm,
        "scalenorm": ScaleNorm,
    }

    @classmethod
    def register_activation(
        cls,
        name: str,
        component: type[Activation],
        *,
        replace: bool = False,
    ) -> None:
        """Register an activation alias, rejecting accidental replacement."""
        key = cls._key(name)
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean.")
        cls._validate_component(component, Activation, "activation")
        if key in cls._activations and not replace:
            raise ValueError(
                f"Activation alias {name!r} is already registered; pass replace=True explicitly."
            )
        cls._activations[key] = component

    @classmethod
    def register_normalization(
        cls,
        name: str,
        component: type[Normalization],
        *,
        replace: bool = False,
    ) -> None:
        """Register a normalization alias, rejecting accidental replacement."""
        key = cls._key(name)
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean.")
        cls._validate_component(component, Normalization, "normalization")
        if key in cls._normalizations and not replace:
            raise ValueError(
                f"Normalization alias {name!r} is already registered; pass replace=True explicitly."
            )
        cls._normalizations[key] = component

    @classmethod
    def resolve_activation(cls, spec: type[Activation] | str) -> type[Any]:
        """Return the activation class represented by ``spec``."""
        if not isinstance(spec, str):
            cls._validate_component(spec, Activation, "activation")
            return spec
        key = cls._key(spec)
        if key in cls._activations:
            return cls._activations[key]
        return PluginRegistry.default().resolve(
            PluginReference(kind=PluginKind.ACTIVATION, name=key)
        )

    @classmethod
    def resolve_normalization(
        cls,
        spec: type[Normalization] | str,
    ) -> type[Any]:
        """Return the normalization class represented by ``spec``."""
        if not isinstance(spec, str):
            cls._validate_component(spec, Normalization, "normalization")
            return spec
        key = cls._key(spec)
        if key in cls._normalizations:
            return cls._normalizations[key]
        return PluginRegistry.default().resolve(
            PluginReference(kind=PluginKind.NORMALIZATION, name=key)
        )

    @staticmethod
    def _key(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Component aliases must be strings.")
        key = name.strip().lower().replace("_", "").replace("-", "")
        if not key:
            raise ValueError("Component aliases cannot be empty.")
        return key

    @staticmethod
    def _validate_component(
        component: object,
        expected_base: type[Any],
        label: str,
    ) -> None:
        """Require registered and directly supplied components to obey their contract."""
        if not isinstance(component, type) or not issubclass(component, expected_base):
            raise TypeError(
                f"{label.capitalize()} components must subclass "
                f"{expected_base.__module__}.{expected_base.__name__}."
            )
