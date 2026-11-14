"""Binary or fuzzy spatial overlap with explicitly declared vertex masses."""

import numpy as np


def _masses(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray | None
) -> tuple[float, float, float]:
    a, b = np.asarray(first), np.asarray(second)
    if a.ndim != 1 or b.shape != a.shape or np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError("Memberships must be matching real vectors in [0, 1].")
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any((a < 0) | (a > 1))
        or np.any((b < 0) | (b > 1))
    ):
        raise ValueError("Memberships must be finite values in [0, 1].")
    w = np.ones(len(a)) if weights is None else np.asarray(weights)
    if w.shape != a.shape or np.iscomplexobj(w) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be matching finite nonnegative real masses.")
    intersection = float(np.minimum(a, b) @ w)
    union = float(np.maximum(a, b) @ w)
    total = float(a @ w) + float(b @ w)
    if not all(np.isfinite(m) for m in (intersection, union, total)):
        raise ValueError("Weighted overlap mass exceeds float64; explicitly rescale weights.")
    return intersection, union, total


def weighted_iou(
    first: np.ndarray,
    second: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    empty_value: float = 1.0,
) -> float:
    """sum(w min(a,b))/sum(w max(a,b)); indicator inputs give set weighted IoU.

    Fractional membership is fuzzy overlap, not statistical event probability.
    Empty union uses the explicit ``empty_value`` convention (default identical=1).
    """
    if not np.isfinite(empty_value) or not 0 <= empty_value <= 1:
        raise ValueError("empty_value must be in [0, 1].")
    intersection, union, _ = _masses(first, second, weights)
    return intersection / union if union else empty_value


def weighted_dice(
    first: np.ndarray,
    second: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    empty_value: float = 1.0,
) -> float:
    """2 sum(w min(a,b))/sum(w(a+b)); identical empty inputs use empty_value."""
    if not np.isfinite(empty_value) or not 0 <= empty_value <= 1:
        raise ValueError("empty_value must be in [0, 1].")
    intersection, _, total = _masses(first, second, weights)
    return 2 * intersection / total if total else empty_value
