"""Cohesive base activation contracts and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Activation(nn.Module, ABC):
    """
    Base class for all activation functions in the engine.

    Activations receive a Tensor and return a Tensor with the same general shape.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"name={self.name!r}"
