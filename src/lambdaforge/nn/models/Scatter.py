"""Object-oriented sparse segment operations used by graph models."""

from __future__ import annotations

import torch

from lambdaforge.nn.models.Aggregation import Aggregation


class Scatter:
    """Stateless sparse reductions for rows grouped by integer indices.

    These operations are intentionally separate from dense pooling layers.
    ``src`` has shape ``(E, ...)`` and ``index`` has shape ``(E,)``; row ``e``
    contributes to output row ``index[e]``.
    """

    @staticmethod
    def sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Sum rows of ``src`` by segment; empty segments remain zero."""
        Scatter._validate(src, index, num_segments)
        out = src.new_zeros((num_segments,) + tuple(src.shape[1:]))
        expanded = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        return out.scatter_add_(0, expanded, src)

    @staticmethod
    def mean(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Average rows of ``src`` by segment; empty segments remain zero."""
        summed = Scatter.sum(src, index, num_segments)
        ones = src.new_ones((src.shape[0], 1))
        count = Scatter.sum(ones, index, num_segments).clamp_min(1.0)
        return summed / count

    @staticmethod
    def reduce(
        src: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
        aggregation: Aggregation | str,
    ) -> torch.Tensor:
        """Apply the requested :class:`Aggregation`."""
        mode = Aggregation(aggregation)
        if mode is Aggregation.SUM:
            return Scatter.sum(src, index, num_segments)
        return Scatter.mean(src, index, num_segments)

    @staticmethod
    def segment_softmax(
        scores: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
        eps: float = 1e-16,
    ) -> torch.Tensor:
        """Compute a numerically stable softmax independently per segment."""
        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional.")
        Scatter._validate(scores, index, num_segments)
        max_per_segment = scores.new_full((num_segments,), float("-inf"))
        max_per_segment = max_per_segment.scatter_reduce(
            0,
            index,
            scores,
            reduce="amax",
            include_self=True,
        )
        shifted = scores - max_per_segment[index]
        exponential = shifted.exp()
        denominator = Scatter.sum(
            exponential.unsqueeze(-1),
            index,
            num_segments,
        ).squeeze(-1)
        return exponential / (denominator[index] + eps)

    @staticmethod
    def _validate(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> None:
        if src.ndim < 1:
            raise ValueError("src must have at least one dimension.")
        if index.ndim != 1 or index.shape[0] != src.shape[0]:
            raise ValueError("index must have shape (src.shape[0],).")
        if index.dtype != torch.long:
            raise TypeError("index must use torch.long dtype.")
        if num_segments < 0:
            raise ValueError("num_segments must be non-negative.")
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= num_segments):
            raise IndexError("index contains a segment outside [0, num_segments).")
