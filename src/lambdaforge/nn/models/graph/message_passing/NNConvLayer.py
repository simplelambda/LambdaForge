"""Full and basis-decomposed edge-conditioned linear filters."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.validation import channels, graph
from lambdaforge.nn.models.Scatter import Scatter


class NNConvLayer(nn.Module):
    """Apply x_source @ W(edge), then sum/mean by destination.

    num_bases=None generates full (in, out) matrices per edge in bounded chunks.
    A positive num_bases instead learns W0 + sum_b q_b(edge) B_b; optional
    centering makes zero edge attributes recover W0. Basis aggregation applies
    matrices per node, never materializes an (E, in, out) tensor. These bases
    are an explicit restricted variant, not identical to unrestricted ECC.
    Chunking bounds temporaries, NOT total autograd activation memory.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        hidden_channels: int = 64,
        num_bases: int | None = None,
        aggregation: str = "mean",
        root_weight: bool = True,
        center_conditioner: bool = False,
        message_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        channels(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            message_chunk_size=message_chunk_size,
        )
        if num_bases is not None:
            channels(num_bases=num_bases)
        if aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be sum or mean.")
        if not isinstance(root_weight, bool) or not isinstance(center_conditioner, bool):
            raise TypeError("root_weight and center_conditioner must be Boolean.")
        if center_conditioner and num_bases is None:
            raise ValueError("Centered W0 conditioning requires the explicit basis variant.")
        self.in_channels, self.out_channels, self.edge_channels = (
            in_channels,
            out_channels,
            edge_channels,
        )
        self.num_bases, self.aggregation = num_bases, aggregation
        self.center_conditioner, self.message_chunk_size = center_conditioner, message_chunk_size
        self.conditioner = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(
                hidden_channels, num_bases if num_bases is not None else in_channels * out_channels
            ),
        )
        self.root = nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        if num_bases is not None:
            self.bases = nn.Parameter(torch.empty(num_bases, in_channels, out_channels))
            self.base_weight = nn.Parameter(torch.empty(in_channels, out_channels))
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.base_weight)
        else:
            self.register_parameter("bases", None)
            self.register_parameter("base_weight", None)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        """Return (N, out_channels), preserving duplicates and directed edge alignment."""
        edges, features = graph(x, edge_index, self.in_channels, edge_features, self.edge_channels)
        assert features is not None
        source, target = edges
        output = x.new_zeros((len(x), self.out_channels)) + x.sum() * 0 + features.sum() * 0
        if self.num_bases is not None:
            coefficients = self.conditioner(features)
            if self.center_conditioner:
                coefficients = coefficients - self.conditioner(torch.zeros_like(features))
            output = Scatter.sum(x[source], target, len(x)) @ self.base_weight
            for index in range(self.num_bases):
                aggregate = Scatter.sum(
                    x[source] * coefficients[:, index : index + 1], target, len(x)
                )
                output = output + aggregate @ self.bases[index]
        else:
            for start in range(0, source.numel(), self.message_chunk_size):
                stop = start + self.message_chunk_size
                matrices = self.conditioner(features[start:stop]).reshape(
                    -1, self.in_channels, self.out_channels
                )
                messages = torch.bmm(x[source[start:stop]].unsqueeze(1), matrices).squeeze(1)
                output = output + Scatter.sum(messages, target[start:stop], len(x))
        if self.aggregation == "mean":
            degree = Scatter.sum(x.new_ones((len(source), 1)), target, len(x)).clamp_min(1)
            output = output / degree
        return output if self.root is None else output + self.root(x)
