"""Activation interfaces and built-in implementations."""

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.gated import GEGLU, GLU, ReGLU, SwiGLU
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

__all__ = [
    "Activation",
    "CELU",
    "ELU",
    "Entmax15",
    "Entmoid15",
    "GEGLU",
    "GELU",
    "GLU",
    "Hardsigmoid",
    "Hardswish",
    "Identity",
    "LeakyReLU",
    "Mish",
    "PReLU",
    "ReGLU",
    "ReLU",
    "ReLU6",
    "SELU",
    "Sigmoid",
    "SiLU",
    "Sine",
    "Snake",
    "Softplus",
    "Softsign",
    "SquarePlus",
    "SwiGLU",
    "Tanh",
]
