"""Validation and device conversion for sparse graph edge indices."""

from __future__ import annotations

import torch


class GraphEdgeIndex:
    """Centralize the integer and range contract for sparse graph topology.

    Public graph layers accept any standard integer tensor and normalize it to
    `torch.long` on the feature tensor's device. Floating-point, complex and
    Boolean tensors are rejected instead of being silently truncated.
    """

    INTEGER_DTYPES = frozenset(
        {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }
    )

    @classmethod
    def normalize(
        cls,
        edge_index: torch.Tensor,
        *,
        device: torch.device,
        num_nodes: int,
    ) -> torch.Tensor:
        """Validate `edge_index` and return a device-local long tensor."""
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be a torch.Tensor.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")
        if edge_index.dtype not in cls.INTEGER_DTYPES:
            raise TypeError("edge_index must use an integer dtype (Boolean is not accepted).")
        normalized = edge_index.to(device=device, dtype=torch.long)
        if normalized.numel() and (int(normalized.min()) < 0 or int(normalized.max()) >= num_nodes):
            raise IndexError("edge_index contains a node outside [0, N).")
        return normalized
