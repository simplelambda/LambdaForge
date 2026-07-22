"""Implementation of the Model object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class Model(nn.Module, ABC):
    """Base class for trainable LambdaForge models.

    Models may accept tensors, mappings, or multiple structured arguments. The
    training task controls how a batch is forwarded, so this abstraction does
    not impose one domain-specific input or output schema.
    """

    output_schema: dict[str, Any] = {}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute model outputs for the supplied inputs."""
        raise NotImplementedError

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference without gradients and restore the previous mode."""
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                return self.forward(*args, **kwargs)
        finally:
            self.train(was_training)

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return the number of parameters, optionally only trainable ones."""
        parameters = self.parameters()

        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)

        return sum(p.numel() for p in parameters)

    def freeze(self) -> None:
        """Disable gradients for every parameter."""
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        """Enable gradients for every parameter."""
        for parameter in self.parameters():
            parameter.requires_grad = True

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return named model parameters for optimizer-specific overrides.

        Models with meaningful parameter families may override this method.
        The default exposes one ``default`` group, so ordinary models remain
        effortless to configure.
        """
        return {"default": tuple(self.parameters())}
