"""Dense graph-structural attention with explicit support and memory bounds."""

from __future__ import annotations

import math

import torch
from torch import nn


class StructuralGraphTransformerLayer(nn.Module):
    """Graphormer-inspired attention on padded independent graphs (B,N,D).

    Explicit in/out degrees and shortest-path bucket indices condition attention.
    Spatial code 0 means unreachable; hop d uses code d+1 (self=1). A caller may
    supply permutation-equivariant path/edge biases (B,H,N,N). No index-selected
    shortest path, molecular readout or graph token is silently introduced.
    This is a generic structural variant, not a reproduction of Graphormer's
    full benchmark recipe. max_attention_bytes bounds score storage only, not
    total training memory. Global communication changes the local graph's reach.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        max_degree: int = 128,
        num_spatial_codes: int = 32,
        dropout: float = 0.0,
        max_attention_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        super().__init__()
        from lambdaforge.nn.models.graph.validation import channels as validate_channels

        validate_channels(
            channels=channels,
            num_heads=num_heads,
            max_degree=max_degree,
            num_spatial_codes=num_spatial_codes,
            max_attention_bytes=max_attention_bytes,
        )
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads.")
        if num_spatial_codes < 2:
            raise ValueError("num_spatial_codes must include unreachable=0 and self=1.")
        self.channels, self.num_heads, self.max_degree = channels, num_heads, max_degree
        self.num_spatial_codes, self.max_attention_bytes = num_spatial_codes, max_attention_bytes
        self.in_degree = nn.Embedding(max_degree + 1, channels)
        self.out_degree = nn.Embedding(max_degree + 1, channels)
        self.spatial = nn.Embedding(num_spatial_codes, num_heads)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.output = nn.Linear(channels, channels)
        self.first_norm, self.second_norm = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def _indices(
        self, value: torch.Tensor, shape: tuple[int, ...], maximum: int, x: torch.Tensor, name: str
    ) -> torch.Tensor:
        from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex

        if value.shape != shape or value.dtype not in GraphEdgeIndex.INTEGER_DTYPES:
            raise ValueError(f"{name} must be integer with shape {shape}.")
        result = value.to(device=x.device, dtype=torch.long)
        if result.numel() and ((result < 0).any() or (result > maximum).any()):
            raise ValueError(
                f"{name} exceeds its declared codebook; bucket explicitly before calling."
            )
        return result

    def forward(
        self,
        x: torch.Tensor,
        spatial_indices: torch.Tensor,
        in_degree: torch.Tensor,
        out_degree: torch.Tensor,
        valid_nodes: torch.Tensor | None = None,
        edge_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (B,N,D), zero for padding; no attention crosses batch elements."""
        if x.ndim != 3 or x.shape[-1] != self.channels or not x.is_floating_point():
            raise ValueError("x must be floating point (B,N,channels).")
        batch, nodes, width = x.shape
        # Spatial embeddings can promote bf16/fp16 scores to fp32 under autocast.
        # Bound the materialized score dtype, not just the input feature dtype.
        score_dtype = torch.promote_types(x.dtype, self.spatial.weight.dtype)
        required = batch * self.num_heads * nodes * nodes * (torch.finfo(score_dtype).bits // 8)
        if required > self.max_attention_bytes:
            raise MemoryError(
                f"Dense attention scores need {required} bytes; limit={self.max_attention_bytes}."
            )
        spatial = self._indices(
            spatial_indices, (batch, nodes, nodes), self.num_spatial_codes - 1, x, "spatial_indices"
        )
        indeg = self._indices(in_degree, (batch, nodes), self.max_degree, x, "in_degree")
        outdeg = self._indices(out_degree, (batch, nodes), self.max_degree, x, "out_degree")
        valid = torch.ones((batch, nodes), device=x.device, dtype=torch.bool)
        if valid_nodes is not None:
            if valid_nodes.shape != (batch, nodes) or valid_nodes.dtype != torch.bool:
                raise ValueError("valid_nodes must be Boolean (B,N).")
            valid = valid_nodes.to(x.device)
        if nodes == 0:
            return x
        if (~valid.any(-1)).any():
            raise ValueError("Each padded graph must contain at least one valid node.")
        hidden = self.first_norm(x + self.in_degree(indeg) + self.out_degree(outdeg))
        qkv = self.qkv(hidden).view(batch, nodes, 3, self.num_heads, width // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.num_heads)
        scores = scores + self.spatial(spatial).permute(0, 3, 1, 2)
        if edge_bias is not None:
            if (
                edge_bias.shape != scores.shape
                or not edge_bias.is_floating_point()
                or not torch.isfinite(edge_bias).all()
            ):
                raise ValueError("edge_bias must be finite floating (B,H,N,N).")
            scores = scores + edge_bias.to(x)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = self.dropout(scores.softmax(-1)) @ v
        attended = attended.transpose(1, 2).reshape(batch, nodes, width)
        result = x + self.dropout(self.output(attended))
        result = result + self.dropout(self.feedforward(self.second_norm(result)))
        return result.masked_fill(~valid[..., None], 0)
