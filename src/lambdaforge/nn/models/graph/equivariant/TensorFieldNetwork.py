"""Native scalar/vector E(3)-equivariant tensor-field message passing."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class TensorFieldNetwork(Model):
    """Propagate invariant scalars and order-one vectors over directed 3D edges."""

    def __init__(
        self,
        scalar_in_channels: int,
        scalar_out_channels: int,
        vector_in_channels: int,
        vector_out_channels: int,
        edge_channels: int = 0,
        radial_hidden_channels: int = 64,
        distance_epsilon: float = 1e-8,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(scalar_in_channels, scalar_out_channels, vector_out_channels) < 1:
            raise ValueError("Scalar channels and vector_out_channels must be positive.")
        if min(vector_in_channels, edge_channels) < 0 or radial_hidden_channels < 1:
            raise ValueError("Input vector/edge channels must be non-negative.")
        if distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be positive.")
        self.scalar_in_channels = scalar_in_channels
        self.scalar_out_channels = scalar_out_channels
        self.vector_in_channels = vector_in_channels
        self.vector_out_channels = vector_out_channels
        self.edge_channels = edge_channels
        self.distance_epsilon = float(distance_epsilon)
        radial_outputs = (
            scalar_out_channels
            + vector_out_channels
            + vector_in_channels * vector_out_channels
            + vector_in_channels * scalar_out_channels
        )
        radial_inputs = 2 * scalar_in_channels + 1 + edge_channels
        self.radial = nn.Sequential(
            nn.Linear(radial_inputs, radial_hidden_channels, bias=bias),
            nn.SiLU(),
            nn.Linear(radial_hidden_channels, radial_outputs, bias=bias),
        )
        self.scalar_root = nn.Linear(scalar_in_channels, scalar_out_channels, bias=bias)
        self.vector_root = (
            nn.Parameter(torch.empty(vector_in_channels, vector_out_channels))
            if vector_in_channels
            else None
        )
        if self.vector_root is not None:
            nn.init.xavier_uniform_(self.vector_root)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        edge_index: torch.Tensor,
        coordinates: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``scalars=(N,Sout)`` and ``vectors=(N,Vout,3)``."""
        node_count = scalars.shape[0]
        if scalars.ndim != 2 or scalars.shape[1] != self.scalar_in_channels:
            raise ValueError("scalars must have shape (nodes, scalar_in_channels).")
        if vectors.shape != (node_count, self.vector_in_channels, 3):
            raise ValueError("vectors must have shape (nodes, vector_in_channels, 3).")
        if coordinates.shape != (node_count, 3):
            raise ValueError("coordinates must have shape (nodes, 3).")
        if coordinates.device != scalars.device or coordinates.dtype != scalars.dtype:
            raise ValueError("coordinates and scalars must share device and dtype.")
        if vectors.device != scalars.device or vectors.dtype != scalars.dtype:
            raise ValueError("vectors and scalars must share device and dtype.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("edge_index must use an integer dtype.")
        source, target = edge_index.to(device=scalars.device, dtype=torch.long)
        if source.numel() and bool(
            ((source < 0) | (source >= node_count) | (target < 0) | (target >= node_count)).any()
        ):
            raise ValueError("edge_index contains an out-of-range node index.")
        if self.edge_channels:
            if edge_features is None or edge_features.shape != (
                source.numel(),
                self.edge_channels,
            ):
                raise ValueError("edge_features must have shape (edges, edge_channels).")
            edge_values = edge_features.to(device=scalars.device, dtype=scalars.dtype)
        else:
            if edge_features is not None and edge_features.shape != (source.numel(), 0):
                raise ValueError("edge_features were supplied but edge_channels is zero.")
            edge_values = scalars.new_empty((source.numel(), 0))
        displacement = coordinates[source] - coordinates[target]
        distance = displacement.square().sum(dim=-1, keepdim=True).sqrt()
        direction = displacement / distance.clamp_min(self.distance_epsilon)
        coefficients = self.radial(
            torch.cat([scalars[source], scalars[target], distance, edge_values], dim=-1)
        )
        cursor = self.scalar_out_channels
        scalar_messages = coefficients[:, :cursor]
        directional = coefficients[:, cursor : cursor + self.vector_out_channels]
        cursor += self.vector_out_channels
        if self.vector_in_channels:
            mixing_size = self.vector_in_channels * self.vector_out_channels
            mixing = coefficients[:, cursor : cursor + mixing_size].reshape(
                -1, self.vector_in_channels, self.vector_out_channels
            )
            cursor += mixing_size
            vector_source = vectors[source]
            vector_messages = torch.einsum("evc,evo->eoc", vector_source, mixing)
            invariant_weights = coefficients[:, cursor:].reshape(
                -1, self.vector_in_channels, self.scalar_out_channels
            )
            projections = torch.einsum("evc,ec->ev", vector_source, direction)
            scalar_messages = scalar_messages + torch.einsum(
                "ev,evo->eo", projections, invariant_weights
            )
        else:
            vector_messages = scalars.new_zeros((source.numel(), self.vector_out_channels, 3))
        vector_messages = vector_messages + directional.unsqueeze(-1) * direction.unsqueeze(1)
        scalar_output = self.scalar_root(scalars)
        vector_output = scalars.new_zeros((node_count, self.vector_out_channels, 3))
        if self.vector_root is not None:
            vector_output = torch.einsum("nvc,vo->noc", vectors, self.vector_root)
        scalar_output.index_add_(0, target, scalar_messages)
        vector_output.index_add_(0, target, vector_messages)
        return {"scalars": scalar_output, "vectors": vector_output}
