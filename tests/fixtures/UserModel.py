"""Example project-defined model used by YAML extension tests."""

import torch
from torch import nn


class UserModel(nn.Module):
    """Small external-style model proving that framework inheritance is optional-friendly."""

    def __init__(self, in_features: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a user-selected output mapping."""
        return {"user_logits": self.projection(x)}
