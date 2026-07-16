"""Example project-defined loss used by YAML extension tests."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as functional

from lambdaforge.nn.losses.Loss import Loss


class UserLoss(Loss):
    """Compute BCE over the output and target keys chosen by a user project."""

    def __init__(self, name: str = "user_bce") -> None:
        super().__init__(name=name)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        """Return a differentiable binary cross-entropy scalar."""
        del context
        return functional.binary_cross_entropy_with_logits(
            outputs["user_logits"],
            batch["target"].to(outputs["user_logits"].dtype),
        )
