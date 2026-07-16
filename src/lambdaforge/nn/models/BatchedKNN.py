"""Implementation of the BatchedKNN object."""

from __future__ import annotations

import torch

from lambdaforge.nn.distances import Distance, SquaredEuclideanDistance
from lambdaforge.nn.models.Model import Model


class BatchedKNN(Model):
    r"""K-nearest-neighbor search for batched point sets.

    This model builds a fixed-size KNN neighborhood for each query point in a
    batched tensor. It is intentionally domain-agnostic: it only knows about
    tensors, distances and nearest-neighbor indices. It does not know about
    atoms, surfaces, proteins, chemistry, message passing or pooling.

    Shape convention
    ----------------
    B:
        Batch size.

    T_q:
        Number of query points per batch element.

    T_s:
        Number of source points per batch element.

    F:
        Feature dimension used by the distance. For Cartesian coordinates,
        ``F = 3``.

    K:
        Number of neighbors returned per query.

    Expected input
    --------------
    query : torch.Tensor
        Query point set with shape ``(B, T_q, F)``.

    source : torch.Tensor
        Source point set with shape ``(B, T_s, F)``.

    Output
    ------
    indices : torch.Tensor
        Source indices with shape ``(B, T_q, K)``. The indices are local to the
        second dimension of ``source``: ``indices[b, i, j]`` selects
        ``source[b, indices[b, i, j]]``.

    distances : torch.Tensor
        Distances with shape ``(B, T_q, K)``. Their scale and meaning are
        defined by the injected ``distance`` module.

    Notes
    -----
    The batch dimension is always explicit. For a single query/source pair,
    call this module with ``query.unsqueeze(0)`` and ``source.unsqueeze(0)``.

    If ``exclude_self=True``, pairs with the same local query/source index are
    masked out. This is meant for same-set neighborhoods such as atom-to-atom
    or surface-to-surface KNN:

        idx, d = knn(x, x)

    If fewer than ``K`` valid sources are available, the last valid neighbor is
    repeated so the output shape remains fixed.

    Parameters
    ----------
    k : int
        Number of neighbors returned per query.
    distance : Distance | None
        Pairwise distance module. If ``None``, squared Euclidean distance is
        used.
    exclude_self : bool
        Whether to mask pairs with identical local query/source indices.
    chunk_size : int
        Number of query points processed at once. This bounds the temporary
        pairwise distance tensor to approximately ``(B, chunk_size, T_s)``.
    """

    output_schema = {
        "indices": "LongTensor[B, T_q, K] with local source indices",
        "distances": "Tensor[B, T_q, K] with distances returned by the distance module",
    }

    def __init__(
        self,
        k: int = 16,
        distance: Distance | None = None,
        exclude_self: bool = False,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.k = k
        self.distance = distance if distance is not None else SquaredEuclideanDistance()
        self.exclude_self = exclude_self
        self.chunk_size = chunk_size

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Find nearest source points for every query point.

        Parameters
        ----------
        query : torch.Tensor
            Query point set with shape ``(B, T_q, F)``.
        source : torch.Tensor
            Source point set with shape ``(B, T_s, F)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``indices`` and ``distances`` with shape ``(B, T_q, K)``.
        """
        if self.k < 1:
            raise ValueError("k must be >= 1.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        batch_size, n_query, _ = query.shape
        _, n_source, _ = source.shape

        indices_out = torch.empty(
            (batch_size, n_query, self.k),
            dtype=torch.long,
            device=query.device,
        )
        distances_out = torch.empty(
            (batch_size, n_query, self.k),
            dtype=query.dtype,
            device=query.device,
        )

        if n_query == 0:
            return indices_out, distances_out

        max_k = n_source - 1 if self.exclude_self else n_source
        if max_k <= 0:
            raise RuntimeError("Cannot compute KNN: no valid source points.")

        kk = min(self.k, max_k)

        source_ids = torch.arange(n_source, device=query.device)

        for start in range(0, n_query, self.chunk_size):
            end = min(start + self.chunk_size, n_query)
            query_chunk = query[:, start:end, :]
            distances = self.distance(query_chunk, source)

            if self.exclude_self:
                query_ids = torch.arange(start, end, device=query.device)
                self_mask = query_ids[None, :, None] == source_ids[None, None, :]
                distances = distances.masked_fill(self_mask, float("inf"))

            chunk_distances, chunk_indices = torch.topk(
                distances,
                k=kk,
                dim=2,
                largest=False,
            )

            if kk < self.k:
                pad = self.k - kk
                chunk_indices = torch.cat(
                    [chunk_indices, chunk_indices[:, :, -1:].expand(-1, -1, pad)],
                    dim=2,
                )
                chunk_distances = torch.cat(
                    [chunk_distances, chunk_distances[:, :, -1:].expand(-1, -1, pad)],
                    dim=2,
                )

            indices_out[:, start:end, :] = chunk_indices
            distances_out[:, start:end, :] = chunk_distances

        return indices_out, distances_out
