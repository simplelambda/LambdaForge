"""Pure-PyTorch relational graph-convolution message-passing layer."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex


class RelationalGCNLayer(nn.Module):
    r"""Aggregate relation-specific messages on a directed sparse graph.

    Every edge is interpreted as ``source -> destination``. For relation
    :math:`r`, the layer transforms the source feature with :math:`W_r`.
    ``mean`` aggregation divides messages by the number of edges that share
    the same ``(destination, relation)`` pair before summing the relation
    contributions. ``sum`` leaves every message unnormalized.

    When ``num_bases`` is set, relation matrices are composed as
    :math:`W_r = \sum_b a_{rb} V_b`. A separate root projection represents
    each node's own features; it does not add or reserve a magic self-loop
    relation. The optional bias is added once after all aggregation.

    Edges are grouped once by relation and projected in chunks bounded by
    `message_chunk_size`. This avoids per-edge relation matrices and dense
    node-by-relation accumulators while preserving exact sparse results.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int | None = None,
        aggregation: Aggregation | str = Aggregation.MEAN,
        root_weight: bool = True,
        bias: bool = True,
        message_chunk_size: int | None = 65_536,
    ) -> None:
        """Create one relational convolution with eager trainable parameters."""
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("num_relations", num_relations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if num_bases is not None:
            if isinstance(num_bases, bool) or not isinstance(num_bases, int):
                raise TypeError("num_bases must be an integer or None.")
            if num_bases < 1:
                raise ValueError("num_bases must be positive when provided.")
            if num_bases > num_relations:
                raise ValueError("num_bases cannot exceed num_relations.")
        if message_chunk_size is not None:
            if isinstance(message_chunk_size, bool) or not isinstance(message_chunk_size, int):
                raise TypeError("message_chunk_size must be an integer or None.")
            if message_chunk_size < 1:
                raise ValueError("message_chunk_size must be positive when provided.")
        if not isinstance(root_weight, bool):
            raise TypeError("root_weight must be Boolean.")
        if not isinstance(bias, bool):
            raise TypeError("bias must be Boolean.")

        aggregation_mode = Aggregation(aggregation)
        if aggregation_mode not in {Aggregation.SUM, Aggregation.MEAN}:
            raise ValueError("aggregation must be 'sum' or 'mean'.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.message_chunk_size = message_chunk_size
        self.aggregation = aggregation_mode

        self.relation_weight: nn.Parameter | None
        self.basis_weight: nn.Parameter | None
        self.basis_coefficients: nn.Parameter | None
        if num_bases is None:
            self.relation_weight = nn.Parameter(
                torch.empty(num_relations, in_channels, out_channels)
            )
            self.register_parameter("basis_weight", None)
            self.register_parameter("basis_coefficients", None)
        else:
            self.register_parameter("relation_weight", None)
            self.basis_weight = nn.Parameter(torch.empty(num_bases, in_channels, out_channels))
            self.basis_coefficients = nn.Parameter(torch.empty(num_relations, num_bases))

        self.root_linear = nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    @property
    def uses_basis_decomposition(self) -> bool:
        """Return whether relation matrices are composed from shared bases."""
        return self.num_bases is not None

    def effective_relation_weights(self) -> torch.Tensor:
        """Return all composed matrices with shape ``(R, in, out)``."""
        if self.relation_weight is not None:
            return self.relation_weight
        if self.basis_weight is None or self.basis_coefficients is None:
            raise RuntimeError("Basis parameters were not initialized.")
        return torch.einsum(
            "rb,bio->rio",
            self.basis_coefficients,
            self.basis_weight,
        )

    def reset_parameters(self) -> None:
        """Initialize direct or basis relation matrices and additive terms."""
        bound = math.sqrt(6.0 / (self.in_channels + self.out_channels))
        if self.relation_weight is not None:
            nn.init.uniform_(self.relation_weight, -bound, bound)
        else:
            if self.basis_weight is None or self.basis_coefficients is None:
                raise RuntimeError("Basis parameters were not initialized.")
            nn.init.uniform_(self.basis_weight, -bound, bound)
            nn.init.xavier_uniform_(self.basis_coefficients)
        if self.root_linear is not None:
            self.root_linear.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        """Return relation-aware node embeddings with shape ``(N, out_channels)``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N, {self.in_channels}).")
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype.")

        num_nodes = x.shape[0]
        routed = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        relations = GraphEdgeData.normalize_relation_types(
            edge_types,
            edge_count=routed.shape[1],
            num_relations=self.num_relations,
            device=x.device,
        )
        source, destination = routed
        relation_weights = self.effective_relation_weights()
        order = torch.argsort(relations, stable=True)
        sorted_relations = relations[order]
        sorted_source = source[order]
        sorted_destination = destination[order]
        relation_ids, relation_counts = torch.unique_consecutive(
            sorted_relations,
            return_counts=True,
        )
        output = x.new_zeros((num_nodes, self.out_channels))
        if relations.numel() == 0:
            output = output + (relation_weights.sum() + x.sum()) * 0.0
        offset = 0
        for relation_id, relation_count in zip(
            relation_ids.tolist(),
            relation_counts.tolist(),
            strict=True,
        ):
            stop = offset + relation_count
            self._accumulate_relation(
                output,
                x,
                sorted_source[offset:stop],
                sorted_destination[offset:stop],
                relation_weights[relation_id],
            )
            offset = stop
        if self.root_linear is not None:
            output = output + self.root_linear(x)
        if self.bias is not None:
            output = output + self.bias
        return output

    def _accumulate_relation(
        self,
        output: torch.Tensor,
        x: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        relation_weight: torch.Tensor,
    ) -> None:
        """Project and accumulate one relation without materializing per-edge weights."""
        normalizer: torch.Tensor | None = None
        if self.aggregation is Aggregation.MEAN:
            _, inverse, counts = torch.unique(
                destination,
                return_inverse=True,
                return_counts=True,
            )
            normalizer = counts[inverse].to(device=x.device, dtype=x.dtype)

        edge_count = source.shape[0]
        chunk_size = edge_count if self.message_chunk_size is None else self.message_chunk_size
        for start in range(0, edge_count, chunk_size):
            stop = min(start + chunk_size, edge_count)
            messages = x[source[start:stop]] @ relation_weight
            if normalizer is not None:
                messages = messages / normalizer[start:stop].unsqueeze(-1)
            output.index_add_(0, destination[start:stop], messages)
