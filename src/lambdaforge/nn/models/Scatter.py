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
        count = Scatter.sum(src.new_ones(src.shape[0]), index, num_segments).clamp_min(1.0)
        denominator = count.view(num_segments, *([1] * (src.ndim - 1)))
        return summed / denominator

    @staticmethod
    def maximum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Return segment maxima while defining empty segments as zero."""
        return Scatter._extreme(src, index, num_segments, reduce="amax")

    @staticmethod
    def minimum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Return segment minima while defining empty segments as zero."""
        return Scatter._extreme(src, index, num_segments, reduce="amin")

    @staticmethod
    def standard_deviation(
        src: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
        epsilon: float = 1e-12,
    ) -> torch.Tensor:
        """Return population standard deviation per segment.

        Empty and singleton segments are finite. Empty segments are represented
        by zero, matching the other sparse reductions.
        """
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative.")
        mean = Scatter.mean(src, index, num_segments)
        second_moment = Scatter.mean(src.square(), index, num_segments)
        variance = (second_moment - mean.square()).clamp_min(0.0)
        positive = variance > epsilon
        safe_variance = torch.where(positive, variance, torch.ones_like(variance))
        standard_deviation = safe_variance.sqrt()
        return torch.where(
            positive,
            standard_deviation,
            torch.zeros_like(standard_deviation),
        )

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
        if mode is Aggregation.MEAN:
            return Scatter.mean(src, index, num_segments)
        return Scatter.maximum(src, index, num_segments)

    @staticmethod
    def segment_softmax(
        scores: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
        eps: float = 1e-16,
    ) -> torch.Tensor:
        """Compute a stable softmax per segment for scalar or multi-head scores.

        ``scores`` may have shape ``(E,)`` or ``(E, ...)``. Every trailing
        channel is normalized independently over rows that share the same
        segment index.
        """
        Scatter._validate(scores, index, num_segments)
        output_shape = (num_segments,) + tuple(scores.shape[1:])
        max_per_segment = scores.new_full(output_shape, float("-inf"))
        expanded = index.view(-1, *([1] * (scores.dim() - 1))).expand_as(scores)
        max_per_segment = max_per_segment.scatter_reduce(
            0,
            expanded,
            scores,
            reduce="amax",
            include_self=True,
        )
        shifted = scores - max_per_segment[index]
        exponential = shifted.exp()
        denominator = Scatter.sum(exponential, index, num_segments)
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

    @staticmethod
    def _extreme(
        src: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
        *,
        reduce: str,
    ) -> torch.Tensor:
        Scatter._validate(src, index, num_segments)
        if not src.is_floating_point():
            raise TypeError("Sparse min/max reductions require floating-point values.")
        fill = float("-inf") if reduce == "amax" else float("inf")
        output = src.new_full((num_segments,) + tuple(src.shape[1:]), fill)
        expanded = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        output.scatter_reduce_(0, expanded, src, reduce=reduce, include_self=True)
        counts = Scatter.sum(src.new_ones((src.shape[0], 1)), index, num_segments)
        if output.ndim == 1:
            return output.masked_fill(counts.squeeze(-1) == 0, 0.0)
        empty = (counts.squeeze(-1) == 0).view(
            num_segments,
            *([1] * (output.ndim - 1)),
        )
        return output.masked_fill(empty, 0.0)
