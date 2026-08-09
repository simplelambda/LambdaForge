"""Intentional isolated CUDA OOM used to validate censored evidence."""

from __future__ import annotations

from typing import Any

import torch


class CudaOOMMemoryProbe:
    """Request more tensor storage than the child allocator ceiling."""

    def __call__(self, configuration: dict[str, Any], resource_context: dict[str, Any]) -> None:
        """Fail inside the isolated child without affecting the parent process."""
        del configuration
        budget = int(resource_context["memory_budget_bytes"])
        elements = max(1, budget // 4 * 2)
        torch.empty(elements, dtype=torch.float32, device="cuda").fill_(1)
