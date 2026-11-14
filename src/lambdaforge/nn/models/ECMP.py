"""Implementation of the ECMP object."""

from __future__ import annotations

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.MLP import MLP
from lambdaforge.nn.models.Model import Model
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
        """Project inputs then use the same public message/aggregate/update stages."""
        h = self.project_inputs(x)
        for index in range(len(self.message_mlps)):
            messages = self.messages(h, edge_index, edge_attr, edge_type, layer_index=index)
            h = self.update(h, self.aggregate(messages, edge_index, len(h)), layer_index=index)
        return h

    def project_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw (N,in_channels) descriptors to the existing hidden state once."""
        if x.ndim != 2 or x.shape[1] != self.input_proj.in_features or not x.is_floating_point():
            raise ValueError("x must be floating (N,in_channels).")
        return self.input_proj(x)

    def _layer_index(self, index: int) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.message_mlps)
        ):
            raise ValueError("layer_index is outside this ECMP stack.")
        return index

    def messages(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        *,
        layer_index: int = 0,
    ) -> torch.Tensor:
        """Return pre-update (E,H) MLP([sender,receiver,edge,relation]) messages.

        h is already projected, not raw input. Separate operators may supply relation
        messages while one chosen operator owns the common node update.
        """
        index = self._layer_index(layer_index)
        if h.ndim != 2 or h.shape[1] != self.hidden_dim or not h.is_floating_point():
            raise ValueError("h must be floating (N,hidden_dim).")
        edges = GraphEdgeIndex.normalize(edge_index, device=h.device, num_nodes=len(h))
        if edges.shape[1] == 0:
            # Preserve the legacy empty-graph path, including optional context.
            return h.new_zeros((0, self.hidden_dim))
        features = GraphEdgeData.normalize_features(
            edge_attr if self.edge_dim else None,
            edge_channels=self.edge_dim,
            edge_count=edges.shape[1],
            reference=h,
            name="edge_attr",
        )
        if self.relation_embedding is not None:
            if edge_type is None:
                raise ValueError("edge_type is required for relational ECMP.")
            edge_type = GraphEdgeData.normalize_relation_types(
                edge_type,
                edge_count=edges.shape[1],
                num_relations=self.num_relations,
                device=h.device,
            )
        context = self._edge_context(features, edge_type, h.device, h.dtype)
        parts = [h[edges[0]], h[edges[1]]]
        if context is not None:
            parts.append(context)
        return self.message_mlps[index](torch.cat(parts, dim=-1))

    def aggregate(
        self, messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Reduce messages with the configured sum/mean/max, no root/update."""
        if messages.ndim != 2 or messages.shape[1] != self.hidden_dim:
            raise ValueError("messages must have shape (E,hidden_dim).")
        return GraphEdgeData.aggregate_messages(messages, edge_index, num_nodes, self.aggregation)

    def update(
        self, h: torch.Tensor, aggregated: torch.Tensor, *, layer_index: int = 0
    ) -> torch.Tensor:
        """Apply one update MLP, normalization, dropout and optional residual."""
        index = self._layer_index(layer_index)
        if h.ndim != 2 or h.shape[1] != self.hidden_dim or not h.is_floating_point():
            raise ValueError("h must be floating (N,hidden_dim).")
        combined = GraphEdgeData.normalize_features(
            aggregated,
            edge_channels=self.hidden_dim,
            edge_count=len(h),
            reference=h,
            name="aggregated",
        )
        assert combined is not None
        updated = self.update_mlps[index](torch.cat((h, combined), dim=-1))
        updated = self.dropout(self.norms[index](updated))
        return h + updated if self.use_residual else updated
