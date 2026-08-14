"""Implementation of the ECMP object."""

from __future__ import annotations

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.MLP import MLP
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.Scatter import Scatter
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class ECMP(Model):
    r"""Edge-Conditioned Message Passing (ECMP) layer stack.

    This is an MPNN: every directed edge ``j -> i`` builds a message with
    ``MLP([h_j, h_i, edge_attr, relation_embedding])``. Messages are reduced by
    destination node and fused back into node states through a residual update
    MLP. It is not a GCN, GraphSAGE or GAT implementation.
    """

    output_schema = {"node_embeddings": "Tensor[N, H]"}

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        edge_dim: int = 0,
        num_relations: int = 0,
        relation_embedding_dim: int = 16,
        dropout: float = 0.0,
        residual: bool = True,
        norm: type[Normalization] | str = LayerNorm,
        mlp_norm: type[Normalization] | str = IdentityNorm,
        activation: type[Activation] | str = ReLU,
        aggregation: Aggregation = Aggregation.MEAN,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_relations = num_relations
        self.aggregation = Aggregation(aggregation)
        self.use_residual = bool(residual)
        norm = ComponentRegistry.resolve_normalization(norm)
        mlp_norm = ComponentRegistry.resolve_normalization(mlp_norm)
        activation = ComponentRegistry.resolve_activation(activation)

        self.relation_embedding = (
            nn.Embedding(num_relations, relation_embedding_dim) if num_relations > 0 else None
        )

        relation_dim = relation_embedding_dim if num_relations > 0 else 0

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        message_in = 2 * hidden_dim + edge_dim + relation_dim

        self.message_mlps = nn.ModuleList(
            MLP(
                in_features=message_in,
                out_features=hidden_dim,
                hidden=[hidden_dim],
                activation=activation,
                normalization=mlp_norm,
            )
            for _ in range(num_layers)
        )

        self.update_mlps = nn.ModuleList(
            MLP(
                in_features=2 * hidden_dim,
                out_features=hidden_dim,
                hidden=[hidden_dim],
                activation=activation,
                normalization=mlp_norm,
            )
            for _ in range(num_layers)
        )

        self.norms = nn.ModuleList(norm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _edge_context(
        self,
        edge_attr: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        parts = []
        if self.edge_dim > 0:
            if edge_attr is None:
                raise ValueError(
                    f"ECMP was built with edge_dim={self.edge_dim}, but edge_attr is None."
                )
            parts.append(edge_attr.to(device=device, dtype=dtype))
        if self.relation_embedding is not None:
            if edge_type is None:
                raise ValueError(
                    "ECMP was built with "
                    f"num_relations={self.num_relations}, but edge_type is None."
                )
            parts.append(self.relation_embedding(edge_type.to(device=device, dtype=torch.long)))
        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = x.shape[0]
        h = self.input_proj(x)  # (N, H)

        edge_index = edge_index.to(device=h.device, dtype=torch.long)
        has_edges = edge_index.numel() > 0

        if has_edges:
            source, destination = edge_index[0], edge_index[1]
            context = self._edge_context(edge_attr, edge_type, h.device, h.dtype)

        for message_mlp, update_mlp, norm in zip(
            self.message_mlps, self.update_mlps, self.norms, strict=True
        ):
            if has_edges:
                message_parts = [h[source], h[destination]]
                if context is not None:
                    message_parts.append(context)
                messages = message_mlp(torch.cat(message_parts, dim=-1))  # (E, H)
                aggregated = Scatter.reduce(
                    messages, destination, num_nodes, self.aggregation
                )  # (N, H)
            else:
                aggregated = h.new_zeros((num_nodes, self.hidden_dim))

            updated = update_mlp(torch.cat([h, aggregated], dim=-1))  # (N, H)
            updated = self.dropout(norm(updated))
            h = h + updated if self.use_residual else updated

        return h
