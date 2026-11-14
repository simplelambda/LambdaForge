"""Explicit validation of dense dissimilarities; no triangle inequality is assumed."""

from __future__ import annotations

import numpy as np


def pairwise_memory_guard(size: int, max_bytes: int | None) -> None:
    """Check the float64 matrix budget before conversion or allocation."""
    if max_bytes is not None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_pairwise_bytes must be a positive integer or null.")
        if size * size * 8 > max_bytes:
            raise MemoryError(
                f"Dense dissimilarities require {size * size * 8} bytes (O(N^2)), "
                f"above max_pairwise_bytes={max_bytes}."
            )


def validate_dissimilarity(
    matrix: np.ndarray,
    *,
    max_pairwise_bytes: int | None = 512 * 1024 * 1024,
    symmetry_atol: float = 1e-10,
    symmetry_rtol: float = 1e-8,
    symmetry: str = "check",
) -> np.ndarray:
    """Validate finite, nonnegative, square dense distances with an exact zero diagonal.

    ``check`` accepts symmetry within the stated tolerances without altering data.
    ``average`` explicitly replaces both directions by their arithmetic mean; this
    changes the scientific dissimilarity and must be requested by the caller.
    Negative values and nonzero self-distances are never silently clipped.
    Returns float64 data, potentially sharing input storage. Validation scratch
    space is bounded by a row, not another N-by-N boolean matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Dissimilarities must be a numpy.ndarray.")
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Dissimilarities must be a non-empty square [N, N] matrix.")
    pairwise_memory_guard(len(matrix), max_pairwise_bytes)
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise TypeError("Dissimilarities must be real numeric values.")
    if symmetry not in {"check", "average"}:
        raise ValueError("symmetry must be 'check' or explicit 'average'.")
    if any(not np.isfinite(t) or t < 0 for t in (symmetry_atol, symmetry_rtol)):
        raise ValueError("Symmetry tolerances must be finite and nonnegative.")
    for index, row in enumerate(matrix):
        if not np.isfinite(row).all() or np.any(row < 0) or np.any(row > np.finfo(np.float64).max):
            raise ValueError("Dissimilarities must be finite and nonnegative.")
        if row[index] != 0:
            raise ValueError("Dissimilarities must have an exact zero diagonal.")
        if symmetry == "check" and not np.allclose(
            row, matrix[:, index], atol=symmetry_atol, rtol=symmetry_rtol
        ):
            raise ValueError("Dissimilarities must be symmetric within explicit tolerances.")
    if symmetry == "average":
        # Half before adding avoids overflowing two finite large distances.
        result = np.empty(matrix.shape, dtype=np.float64)
        for index in range(len(matrix)):
            result[index] = (
                np.asarray(matrix[index], dtype=np.float64) * 0.5
                + np.asarray(matrix[:, index], dtype=np.float64) * 0.5
            )
        return result
    return np.asarray(matrix, dtype=np.float64)
