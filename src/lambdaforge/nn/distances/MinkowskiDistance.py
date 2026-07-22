"""Implementation of the MinkowskiDistance object."""

from __future__ import annotations

import math

import torch

from lambdaforge.nn.distances.Distance import Distance


class MinkowskiDistance(Distance):
    r"""Pairwise Minkowski distance of configurable order.

    Parameters
    ----------
    p : float
        Norm order. Values greater than or equal to one define a metric.
        Default: ``2.0``.
    compute_mode : str
        PyTorch ``cdist`` strategy. It is relevant to Euclidean distance and
        may be ``"use_mm_for_euclid_dist_if_necessary"``,
        ``"use_mm_for_euclid_dist"`` or ``"donot_use_mm_for_euclid_dist"``.
    """

    def __init__(
        self,
        p: float = 2.0,
        compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
    ) -> None:
        super().__init__()
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError("p must be a real number")
        if not math.isfinite(float(p)) or float(p) < 1.0:
            raise ValueError("p must be finite and greater than or equal to one")
        valid_modes = {
            "use_mm_for_euclid_dist_if_necessary",
            "use_mm_for_euclid_dist",
            "donot_use_mm_for_euclid_dist",
        }
        if compute_mode not in valid_modes:
            raise ValueError(f"compute_mode must be one of {sorted(valid_modes)}")
        self.p = float(p)
        self.compute_mode = compute_mode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y must both have shape (B, T, F)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same batch size")
        if x.shape[-1] != y.shape[-1]:
            raise ValueError("x and y must have the same feature dimension")
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError("x and y must have the same device and dtype")
        if not x.is_floating_point() or not y.is_floating_point():
            raise TypeError("x and y must be floating-point tensors")
        return torch.cdist(x, y, p=self.p, compute_mode=self.compute_mode)

    def extra_repr(self) -> str:
        return f"p={self.p}, compute_mode={self.compute_mode!r}"
