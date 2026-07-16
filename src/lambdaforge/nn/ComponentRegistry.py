"""Registry for YAML-friendly neural-network component names."""

from __future__ import annotations

from typing import Any

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ELU import ELU
from lambdaforge.nn.activations.GELU import GELU
from lambdaforge.nn.activations.Identity import Identity
from lambdaforge.nn.activations.LeakyReLU import LeakyReLU
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.activations.Sigmoid import Sigmoid
from lambdaforge.nn.activations.SiLU import SiLU
from lambdaforge.nn.activations.Tanh import Tanh
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization
from lambdaforge.nn.normalizations.RMSNorm import RMSNorm


class ComponentRegistry:
    """Resolve short, case-insensitive component names to framework classes.

    The registry is deliberately small and explicit. YAML files may use names
    such as ``relu`` or ``layernorm``, while Python callers remain free to pass
    any compatible class directly. Custom aliases can be registered without
    modifying model implementations.
    """

    _activations: dict[str, type[Activation]] = {
        "elu": ELU,
        "gelu": GELU,
        "identity": Identity,
        "leakyrelu": LeakyReLU,
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "silu": SiLU,
        "tanh": Tanh,
    }
    _normalizations: dict[str, type[Normalization]] = {
        "batchnorm": BatchNorm,
        "identity": IdentityNorm,
        "layernorm": LayerNorm,
        "none": IdentityNorm,
        "rmsnorm": RMSNorm,
    }

    @classmethod
    def register_activation(cls, name: str, component: type[Activation]) -> None:
        """Register or replace an activation alias."""
        cls._activations[cls._key(name)] = component

    @classmethod
    def register_normalization(cls, name: str, component: type[Normalization]) -> None:
        """Register or replace a normalization alias."""
        cls._normalizations[cls._key(name)] = component

    @classmethod
    def resolve_activation(cls, spec: type[Activation] | str) -> type[Any]:
        """Return the activation class represented by ``spec``."""
        if not isinstance(spec, str):
            return spec
        key = cls._key(spec)
        if key not in cls._activations:
            raise ValueError(f"Unknown activation {spec!r}. Options: {sorted(cls._activations)}.")
        return cls._activations[key]

    @classmethod
    def resolve_normalization(
        cls,
        spec: type[Normalization] | str,
    ) -> type[Any]:
        """Return the normalization class represented by ``spec``."""
        if not isinstance(spec, str):
            return spec
        key = cls._key(spec)
        if key not in cls._normalizations:
            raise ValueError(
                f"Unknown normalization {spec!r}. Options: {sorted(cls._normalizations)}."
            )
        return cls._normalizations[key]

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower().replace("_", "").replace("-", "")
