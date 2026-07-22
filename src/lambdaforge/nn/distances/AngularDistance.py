"""Implementation of the AngularDistance object."""

from __future__ import annotations

import math

import torch

from lambdaforge.nn.distances.Distance import Distance


class AngularDistance(Distance):
    r"""Pairwise angle derived from cosine similarity.

    Parameters
    ----------
    norm_eps : float
        Positive lower bound used when normalizing vector norms. Default:
        ``1e-12``.
    clamp_eps : float
        Margin from ``-1`` and ``1`` before applying ``acos``. Exact endpoints
        are restored to zero and pi. This improves gradient stability. Default:
        ``1e-7``.
    normalized : bool
        Divide angles by pi to return values in ``[0, 1]`` instead of radians.
        Default: ``False``.
    """

    def __init__(
        self,
        norm_eps: float = 1e-12,
        clamp_eps: float = 1e-7,
        normalized: bool = False,
    ) -> None:
        super().__init__()
        for label, value in {"norm_eps": norm_eps, "clamp_eps": clamp_eps}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if float(norm_eps) <= 0.0:
            raise ValueError("norm_eps must be greater than zero")
        if not 0.0 <= float(clamp_eps) < 1.0:
            raise ValueError("clamp_eps must be in the interval [0, 1)")
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a boolean")
        self.norm_eps = float(norm_eps)
        self.clamp_eps = float(clamp_eps)
        self.normalized = normalized

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

        normalized_x = torch.nn.functional.normalize(x, p=2.0, dim=-1, eps=self.norm_eps)
        normalized_y = torch.nn.functional.normalize(y, p=2.0, dim=-1, eps=self.norm_eps)
        similarity = torch.bmm(normalized_x, normalized_y.transpose(1, 2)).clamp(-1.0, 1.0)
        safe_similarity = similarity.clamp(-1.0 + self.clamp_eps, 1.0 - self.clamp_eps)
        angles = torch.acos(safe_similarity)
        angles = torch.where(similarity >= 1.0, torch.zeros_like(angles), angles)
        angles = torch.where(similarity <= -1.0, torch.full_like(angles, math.pi), angles)
        if self.normalized:
            angles = angles / math.pi
        return angles

    def extra_repr(self) -> str:
        return f"norm_eps={self.norm_eps}, clamp_eps={self.clamp_eps}, normalized={self.normalized}"
