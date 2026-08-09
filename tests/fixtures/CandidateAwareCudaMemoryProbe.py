"""Candidate-aware representative CUDA step for HPO tests."""

from __future__ import annotations

from typing import Any

import torch


class CandidateAwareCudaMemoryProbe:
    """Build the candidate-sized model and perform forward/backward/update."""

    def __call__(self, configuration: dict[str, Any], resource_context: dict[str, Any]) -> None:
        """Use both the concrete configuration and generic resource features."""
        width = int(configuration["probe_width"])
        batch_size = int(resource_context["resource_features"]["batch_size"])
        model = torch.nn.Linear(width, width, device="cuda")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs = torch.randn(batch_size, width, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
